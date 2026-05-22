#!/usr/bin/env python3
import os
import json
import torch
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

PROMPTS = [
    "nenu meeting lo unna call chestha later",
    "bro ivala office lo full chaos ga unde",
    "amma already dinner ready chesindi ra",
    "nuvvu weekend plans emaina fix chesava",
    "ee movie climax actually mind blowing undi",
    "naku morning nundi headache vastundi yaar",
    "manager sudden ga deadline prepone chesadu",
    "recharge ayipoyindi hotspot on cheyyava",
    "ivala traffic literally unbearable ga undi",
    "nenu gym lo join avvali anukuntunna",
    "aah cafe lo coffee surprisingly baagundi",
    "exam easy anukunna kani tough ga vachindi",
    "laptop charge almost aipoyindi charger unda",
    "arey evening cricket aadadaniki vastava",
    "ee app UI konchem confusing ga undi",
    "nuvvu Hyderabad ki eppudu move ayyav",
    "weather chala pleasant ga undi today",
    "maa team lo andariki burnout aipothondi",
    "food order cheddama leka bayataki veldama",
    "interview baane jarigindi but not sure",
    "nenu aa series binge watch chesthunna",
    "dad already tickets book chesesaru",
    "ee feature customers ki useful ga untunda",
    "morning leche motivation assalu ledu",
    "naku biryani ante weak spot honestly",
    "nuvvu camera on cheyyi properly vinapadatledu",
    "ee month expenses konchem ekkuva aipoyayi",
    "aame English Telugu mix chesi maatladtundi",
    "salary vachaka trip plan cheddam",
    "office politics choosi visugu vastundi",
    "nenu message chesa kani reply raledu",
    "ee phone battery backup worst ga undi",
    "vaadu chaala overaction chestunnadu bro",
    "meeting entire time useless discussion eh",
    "naku AI models ante genuine curiosity undi",
    "ivala work complete cheyyadam kastame",
    "nuvvu screenshots pampu once free ayyaka",
    "aah restaurant hype ki taggattu ledu",
    "ee joke naaku late ga artham ayyindi",
    "sleep schedule completely damage aipoyindi",
    "mom video call lo Atreya ni adigindi",
    "andaru reels chusthu time waste chestunnaru",
    "ee bug reproduce cheyyadam easy kaadu",
    "vaalla accent valla konchem confuse ayya",
    "nuvvu mute lo unnava entire time",
    "project launch mundu full tension unde",
    "aah teacher chaala chill ga untaru",
    "delivery guy wrong address ki velladu",
    "nenu Telugu lo think chesi English lo maatladta",
    "ee response natural ga unda leka forced ga unda"
]

def main():
    model_id = "google/gemma-4-e4b-it"
    adapter_id = "./gemma_lora_output"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading base model: {model_id}")
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Apply PEFT adapters
    print(f"Loading PEFT adapters from: {adapter_id}")
    model = PeftModel.from_pretrained(base_model, adapter_id)
    model = model.to(device)
    model.eval()
    
    results = []
    
    print("\nStarting generation for 50 evaluation prompts...")
    for idx, prompt in enumerate(PROMPTS, 1):
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
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
            
        # Decode only the generated response
        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        print(f"[{idx}/50] Prompt: {prompt}")
        print(f"      Response: {response}")
        
        results.append({
            "prompt": prompt,
            "response": response
        })
        
    output_file = "fine_tuned_eval_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nGeneration complete! Saved results to {output_file}")

if __name__ == "__main__":
    main()
