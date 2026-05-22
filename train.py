#!/usr/bin/env python3
import os
import argparse
import json
import torch
from dotenv import load_dotenv
load_dotenv() # Load variables from .env (including HF_TOKEN)

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from trl import SFTTrainer, SFTConfig


SYSTEM_PROMPT = """
You are a 25 year old native Telugu speaker from Hyderabad.

Rules:
- Respond only in natural romanized Telugu
- Telugu should be the matrix language
- English should be the embedded language
- English words should appear naturally inside Telugu sentences
- Do not make English the dominant language
- Do not use Telugu script
- Sound like casual real-life conversation between Telugu friends
- Use modern Hyderabad/Telangana urban speech patterns
- Keep responses short and conversational
- Keep responses to 1-2 lines maximum
- Avoid formal Telugu
- Avoid bookish Telugu
- Avoid translation-style wording
- Avoid repetitive phrases
- Avoid assistant-like tone
- Do not explain yourself
- Do not switch fully into English
- Responses should feel like WhatsApp or casual spoken conversation
"""


def load_local_dataset(json_path: str) -> Dataset:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset file not found at: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {json_path}")
    
    prompts = []
    completions = []
    for x in data:
        prompts.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["prompt"]}
        ])
        completions.append([
            {"role": "assistant", "content": x["response"]}
        ])
    return Dataset.from_dict({"prompt": prompts, "completion": completions})

def get_device_info() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma on Telglish (Romanized Telugu) dataset using LoRA")
    parser.add_argument("--model_id", type=str, default="google/gemma-4-e4b-it", help="Hugging Face model ID to fine-tune")
    parser.add_argument("--dataset_path", type=str, default="data/train_sft.json", help="Path to the training json dataset")
    parser.add_argument("--output_dir", type=str, default="./gemma_lora_output", help="Directory to save the fine-tuned model and checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0, limit the number of training steps and ignore epochs")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run smoke test with a tiny test model and small batch")
    parser.add_argument("--hub_model_id", type=str, default="vamsibhagi/CodeSwitch-Gemma", help="Hugging Face repo ID to push adapters to")
    
    args = parser.parse_args()

    
    device = get_device_info()
    print(f"--- Device detected: {device.upper()} ---")
    
    # 1. Resolve model ID and dry-run specific parameters
    model_id = args.model_id
    if args.dry_run:
        # Use a tiny model for fast dry run testing
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        print(f"Dry-run mode: overriding model to '{model_id}'")
    else:
        # Patch the tokenizer_config.json in Hugging Face cache to avoid transformers bug
        try:
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(model_id, "tokenizer_config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if isinstance(config.get("extra_special_tokens"), list):
                print("Patching tokenizer_config.json 'extra_special_tokens' list to empty dict to avoid transformers bug...")
                config["extra_special_tokens"] = {}
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Warning: could not patch tokenizer_config.json: {e}")
        
    print(f"Loading tokenizer for: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    
    # Ensure correct padding configuration
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Load dataset
    if args.dry_run:
        print("Dry-run mode: generating a tiny mock dataset (bypassing local file).")
        mock_data = [
            {"prompt": "hello how are you", "response": "nenu chala bagunnanu, nuvvu ela unnav?"},
            {"prompt": "what is your name", "response": "na peru AI assistant andi, cheppandi."}
        ]
        prompts = []
        completions = []
        for x in mock_data:
            prompts.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["prompt"]}
            ])
            completions.append([
                {"role": "assistant", "content": x["response"]}
            ])
        full_dataset = Dataset.from_dict({"prompt": prompts, "completion": completions})
    else:
        print(f"Loading dataset: {args.dataset_path}")
        full_dataset = load_local_dataset(args.dataset_path)

    # Split dataset into train and validation sets
    dataset_dict = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["test"]
    
    if args.dry_run:
        train_dataset = train_dataset.select(range(min(2, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(1, len(val_dataset))))
        print(f"Dry-run: truncated datasets to {len(train_dataset)} train and {len(val_dataset)} val examples.")

    # 3. Load model
    print(f"Loading model: {model_id}")
    if device == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device == "mps":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    # MPS does not support 8-bit/4-bit quantization natively via bitsandbytes well, so we load in half-precision.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    
    # Direct model to the correct device
    model = model.to(device)
    
    # 4. Configure LoRA
    # Target attention projection layers and MLP/Feed-forward layers
    # Dynamic target modules: Gemma 4 wraps Linear layers in Gemma4ClippableLinear, so we append ".linear"
    # to target the inner torch.nn.Linear layer. Other models (like LLaMA in dry-run) use standard nn.Linear.
    is_gemma4 = False
    for name, module in model.named_modules():
        if module.__class__.__name__ == "Gemma4ClippableLinear":
            is_gemma4 = True
            break
            
    base_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if is_gemma4:
        print("Detected Gemma 4 architecture. Appending '.linear' to target modules to target inner Linear layers.")
        target_modules = [f"{t}.linear" for t in base_targets]
    else:
        target_modules = base_targets

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,   # Slightly higher dropout to resist overfitting on small 1k dataset
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    print("Applying LoRA configuration...")
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    
    # 5. SFTConfig (inherits from TrainingArguments)
    # Enable completion-only loss (masks out prompt tokens in the loss calculation)
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size if not args.dry_run else 1,
        per_device_eval_batch_size=args.batch_size if not args.dry_run else 1,
        gradient_accumulation_steps=2 if not args.dry_run else 1,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        num_train_epochs=args.epochs if not args.dry_run else 1,
        logging_steps=1 if args.dry_run else 5,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=(torch_dtype == torch.bfloat16),
        fp16=(torch_dtype == torch.float16),
        logging_dir=f"{args.output_dir}/logs",
        report_to="none",
        max_grad_norm=1.0,
        completion_only_loss=True, # Calculate loss strictly on completion, mask prompt
        max_length=512,
        max_steps=args.max_steps,
        dataset_text_field=None # SFTTrainer auto-detects 'messages' column and applies chat template
    )
    
    # 6. SFTTrainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=None, # Already wrapped manually with get_peft_model
        args=sft_config
    )
    
    if args.dry_run:
        print("\n--- Running Dry-Run Verification Step ---")
        # Run a single evaluation step to verify data collation and model forward pass
        eval_results = trainer.evaluate()
        print(f"Evaluation verify completed. Results: {eval_results}")
        
        # Run a single training step
        print("Running single training step verify...")
        trainer.train()
        print("Training step verify completed successfully!")
        print("\n--- DRY RUN VERIFICATION PASSED ---")
    else:
        print("\n--- Starting Fine-Tuning ---")
        trainer.train()
        print("Saving fine-tuned model adapters...")
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Fine-tuning complete. Model saved to: {args.output_dir}")
        
        # Automatic upload to Hugging Face Model Hub
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                print(f"Uploading fine-tuned adapters to Hugging Face Hub: {args.hub_model_id}...")
                trainer.model.push_to_hub(args.hub_model_id, token=hf_token)
                tokenizer.push_to_hub(args.hub_model_id, token=hf_token)
                print(f"Successfully uploaded adapters to Hugging Face Hub: https://huggingface.co/{args.hub_model_id}")
            except Exception as e:
                print(f"Warning: Failed to push to Hugging Face Hub: {e}")
        else:
            print("HF_TOKEN not found in environment. Skipping automatic Hugging Face upload.")

if __name__ == "__main__":
    main()
