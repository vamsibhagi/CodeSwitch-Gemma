---
license: apache-2.0
base_model: google/gemma-4-e4b-it
tags:
- gemma
- lora
- peft
- trl
- codeswitching
- telglish
- telugu
language:
- te
- en
---

# CodeSwitch-Gemma: Romanized Telugu-English (Telglish) Fine-Tuning

This repository contains the dataset, evaluation pipeline, and fine-tuning scripts to train a conversational AI model (specifically **Gemma-4-e4b-it**) to speak natural, conversational **Romanized Telugu (Telglish)**.

The model is optimized to use **Telugu as the Matrix Language** (handling grammar, Subject-Object-Verb word order, and verbal helpers) and **English as the Embedded Language** (handling nouns, active verbs, and technical terms).

---

## 📊 Evaluation & Metrics (LLM-as-a-Judge)

We evaluated the model across two different distributions: **Casual Chat** (WhatsApp-style conversational prompts) and **Informational Queries** (technical and explanatory prompts matching the fine-tuning distribution).

The evaluation is judged by a Gemini-based judge across two non-overlapping axes:
1. **Grammatical Integrity (Telugu Syntax)**: Score 1–4
2. **Code-Switch Naturalness (Matrix Frame)**: Score 1–4

### 1. Casual Chat Evaluation (50 Prompts)
Evaluated on short, social conversational messages (e.g., *"nenu meeting lo unna. tarvata call chestha"*):

| Model | Avg. Grammar Score | Avg. Code-Switch Score | Collapses (Score 1 or 2) |
| :--- | :---: | :---: | :---: |
| **Baseline Gemma-4-it** | **2.84 / 4.00** | **2.48 / 4.00** | **29 / 50** |
| **Fine-Tuned Gemma-4-it** | **2.56 / 4.00** | **2.36 / 4.00** | **31 / 50** |

*Note: In the casual set, the fine-tuned model experienced distribution pressure due to the training dataset being exclusively long-form informational content (average 388 words).*

### 2. Informational & Technical Evaluation (20-Prompt Held-Out Set)
Evaluated on complex technical and informational prompts (e.g., modernizing military equipment, smart grid integration, data structures) using a 20-prompt set sampled from the held-out LIMA test set:

| Model | Avg. Grammar Score | Avg. Code-Switch Score | Collapses (Score 1 or 2) |
| :--- | :---: | :---: | :---: |
| **Baseline Gemma-4-it** | **2.55 / 4.00** | **2.30 / 4.00** | **12 / 20** |
| **Fine-Tuned Gemma-4-it** | **2.65 / 4.00** | **2.30 / 4.00** | **13 / 20** |

#### 🔍 Failure Modes & SFT Limitations on Long-Form Technical Queries
*   **The "Pure English" Drift**: Under the strict rubric, any pure English sentence violates the matrix language constraint. When explaining highly technical concepts (like ADAS or EV charging grids), both models frequently drifted into pure English sentences (e.g., *"So, it prevents a crash before it happens."*), resulting in low average scores and high collapses.
*   **Multilingual Prior Leakage (Hindi)**: Base model priors for Indian languages are extremely strong. SFT on a compact 200-sample dataset was not enough to fully suppress Hindi helper words like `hai`, `bahut`, and `aur` on unseen technical prompts.
*   **Telugu Script Leakage**: The fine-tuned model occasionally outputted words in native Telugu script (e.g., `అనేది` instead of `anedi`) due to vocabulary token association leakage.

#### 💡 Structured Alignment Successes
Despite low scores, the fine-tuned model successfully eliminated Hindi contamination in conversational technical prompts (e.g. prompt 10, achieving 4/4) and generated highly detailed, multi-part structured explanations matching the training style perfectly.


---

## 🛠️ Repository Structure

*   `train.py`: PEFT/LoRA fine-tuning script optimized for Gemma 4 architectures (handles `Gemma4ClippableLinear` wrappers).
*   `generate_completions.py`: Evaluation completion generation script (supports `--baseline` and `--informational` modes).
*   `evaluate_judge.py`: Automated LLM-as-a-Judge script utilizing the Gemini API to score outputs.
*   `eval_rubrics.md`: Scoring rubrics and anchoring examples for grading quality.
*   `data/train_sft_lima_200.json`: High-quality training dataset containing **1,010 clean, conversational Telglish informational pairs** with stiff translation replacements applied.

---

## 🚀 How to Run Inference

You can load this model using Hugging Face `transformers` and `peft`. Here is a complete script to generate responses:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "google/gemma-4-e4b-it"
adapter_id = "vamsibhagi/CodeSwitch-Gemma"

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load the PEFT adapter
model = PeftModel.from_pretrained(model, adapter_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# System Prompt for Informational tasks
sys_prompt = """
You are a helpful AI assistant.
Rules:
- Respond only in natural romanized Telugu
- Telugu should be the matrix language
- English should be the embedded language
- English words should appear naturally inside Telugu sentences
- Do not make English the dominant language
- Do not use Telugu script
- Avoid formal Telugu
- Avoid bookish Telugu
- Avoid translation-style wording
- Do not switch fully into English
"""

messages = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": "Vijayawada lo ICT (Information and Communication Technology) sector abhivruddhi cheyadaniki mukhyamaina avakasalu emiti?"}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

---

## 🧠 Code & Optimization Details

The training script incorporates several advanced adaptations:
1.  **Dynamic Gemma-4 Targeting**: Configures LoRA adapters correctly by scanning the model structure and targeting the inner `.linear` layer within the `Gemma4ClippableLinear` wrappers.
2.  **Completion-Only Loss Masking**: Masks out input prompt tokens from the loss function so that the model updates gradients solely based on the assistant's completions (`completion_only_loss=True`).
3.  **Low-Rank Adaptations**: Optimized with `lora_r=16`, `lora_alpha=32`, and `lora_dropout=0.1` to prevent overfitting on the 1,000-sample dataset while preserving the underlying instruction-following behavior of Gemma 4.
