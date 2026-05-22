# CodeSwitch-Gemma: Romanized Telugu (Telglish) Fine-Tuning

This repository contains the dataset, evaluation pipeline, and fine-tuning scripts to train a conversational AI model (specifically **Gemma-4-e4b-it**) to speak natural, conversational **Romanized Telugu (Telglish)**.

The model is optimized to use **Telugu as the Matrix Language** (handling grammar, Subject-Object-Verb word order, and verbal helpers) and **English as the Embedded Language** (handling nouns, active verbs, and technical terms) in a WhatsApp-style casual tone.

---

## 📊 Project Overview & Baseline Comparison

We evaluated two baseline models (**Gemma-2b-it** and **tiny-aya-fire**) on a test set of 50 conversational prompts using an LLM-as-a-judge setup. The evaluation judged responses across two non-overlapping axes:
1. **Grammatical Integrity (Telugu Syntax)**: Score 1–4
2. **Code-Switch Naturalness (Matrix Frame)**: Score 1–4

### Baseline Metrics

| Metric | Gemma Baseline (`gemma-2b-it`) | Aya Fire Baseline (`tiny-aya-fire`) |
| :--- | :---: | :---: |
| **Total Prompts** | 50 | 50 |
| **Average Grammar Score** | **2.94 / 4.00** | **1.10 / 4.00** |
| **Average Code-Switch Score** | **2.96 / 4.00** | **1.06 / 4.00** |
| **Total Collapses (Score 1 or 2)** | **22 / 50 (44.0%)** | **50 / 50 (100.0%)** |
| **Pre-check Failures (Telugu Script)** | 0 / 50 (0%) | 12 / 50 (24.0%) |

---

## 🛠️ Repository Structure

* `train_gemma_lora.py`: PEFT/LoRA fine-tuning script optimized for Gemma-4 architecture.
* `run_llm_eval.py`: Automated evaluation script utilizing LLM-as-a-Judge with custom rubrics.
* `eval.md`: Linguistic rubrics and anchoring examples for grading quality.
* `tenglish_train_data_cleaned.json`: The high-quality training dataset containing **1,019 clean conversational pairs**.
* `requirements.txt`: Python package requirements.
* `initeval.py`: Script to generate baseline model outputs.

---

## 🚀 RunPod GPU Fine-Tuning Guide

Follow these instructions to run the fine-tuning on a cloud GPU (e.g., RunPod RTX 3090/4090, which takes **15–30 minutes** and costs **<$0.20** total):

### 1. Rent a GPU
1. Go to [RunPod.io](https://runpod.io).
2. Rent a GPU pod with at least **24GB VRAM** (RTX 3090, RTX 4090, or A10G).
3. Choose the standard **PyTorch** template.

### 2. Set Up the Terminal & Repository
Connect to the pod via **Web Terminal** and run:
```bash
# Clone the repository
git clone https://github.com/vamsibhagi/CodeSwitch-Gemma.git
cd CodeSwitch-Gemma

# Install the dependencies
pip install -r requirements.txt
```

### 3. Start Training
Set your Hugging Face Token (required to download the gated Gemma-4 base model) and start the training process:
```bash
# Set Hugging Face Token
export HF_TOKEN="your_huggingface_token"

# Run training (3 epochs, batch size 4)
python train_gemma_lora.py --epochs 3 --batch_size 4
```

The adapters will automatically be saved to `./gemma_lora_output` once training completes.

---

## 🧠 Code & Optimization Details

The training script incorporates several advanced adaptations:
1. **Dynamic Gemma-4 Targeting**: PEFT does not natively recognize `Gemma4ClippableLinear` wrapper layers. The script scans model modules and appends `.linear` (e.g. `q_proj.linear`) to configure LoRA adapters correctly.
2. **Unified System Persona**: The dataset is converted dynamically to the conversational `messages` schema containing the identical `SYSTEM_PROMPT` used during evaluation to align training inputs with inference.
3. **Completion-Only Loss Masking**: The trainer ignores tokens belonging to the system prompt and user query during backpropagation (`completion_only_loss=True`), focusing gradient updates strictly on the assistant's response.
4. **Stable IT Optimization**: Uses a lower learning rate of `1e-4` with `warmup_ratio=0.05` and `lora_dropout=0.1` to prevent overriding the pre-trained instruction-following behaviors of the base model.
