#!/usr/bin/env python3
import os
import json
import torch
import argparse
import re
from dotenv import load_dotenv
load_dotenv()

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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

def main():
    parser = argparse.ArgumentParser(description="Generate completions for base or fine-tuned models")
    parser.add_argument("--model_id", type=str, default="google/gemma-4-e4b-it", help="Hugging Face model ID")
    parser.add_argument("--adapter_id", type=str, default="./gemma_lora_output", help="Path to PEFT adapters")
    parser.add_argument("--prompts_path", type=str, default="data/eval_prompts.json", help="Path to evaluation prompts file")
    parser.add_argument("--output", type=str, default="outputs/baseline_gemma.json", help="Path to save the generated completions")
    parser.add_argument("--baseline", action="store_true", help="Run evaluation on baseline model without PEFT adapters")
    args = parser.parse_args()

    # Load prompts
    if not os.path.exists(args.prompts_path):
        raise FileNotFoundError(f"Prompts file not found at: {args.prompts_path}")
    with open(args.prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    model_id = args.model_id
    adapter_id = args.adapter_id
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    print(f"Loading base model: {model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Apply PEFT adapters if not running baseline
    if args.baseline:
        print("Running in BASELINE mode (skipping PEFT adapters).")
        model = base_model
    else:
        print(f"Loading PEFT adapters from: {adapter_id}")
        model = PeftModel.from_pretrained(base_model, adapter_id)
        
    model = model.to(device)
    model.eval()
    
    results = []
    
    # Set up end of sequence stop IDs dynamically
    stop_tokens = ["<turn|>", "<|END_OF_TURN|>", "<|end_of_turn|>", "<|im_end|>"]
    eos_token_ids = [tokenizer.eos_token_id]
    for stop_tok in stop_tokens:
        tok_id = tokenizer.convert_tokens_to_ids(stop_tok)
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            eos_token_ids.append(tok_id)
            print(f"Registered additional stop token: '{stop_tok}' (ID: {tok_id})")
        
    print(f"Using EOS token IDs: {eos_token_ids}")
    
    print(f"\nStarting generation for {len(prompts)} evaluation prompts...")
    for idx, prompt in enumerate(prompts, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=eos_token_ids,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
            
        # Decode only the generated response
        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
        
        # Strip thought blocks (the tags and the thinking text inside them)
        response = re.sub(r'<\|channel\|?>thought\n.*?<channel\|?>', '', response, flags=re.DOTALL)
        
        # Clean any remaining special tokens
        for token in tokenizer.all_special_tokens:
            response = response.replace(token, "")
            
        # Post-process to prevent leaks of delimiters or thoughts in plain text
        for stop_word in ["<turn|>", "<|turn>", "<|think|>", "thought\nThinking Process:", "Thinking Process:", "thought\n"]:
            if stop_word in response:
                response = response.split(stop_word)[0].strip()
                
        # Clean trailing punctuation and formatting noise like :// or ^
        response = re.sub(r'[\s:/\\^\-_]+$', '', response).strip()
        
        print(f"[{idx}/{len(prompts)}] Prompt: {prompt}")
        print(f"      Response: {response}")
        
        results.append({
            "prompt": prompt,
            "response": response
        })
        
    # Ensure parent output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nGeneration complete! Saved results to {args.output}")

if __name__ == "__main__":
    main()
