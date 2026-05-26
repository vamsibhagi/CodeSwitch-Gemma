#!/usr/bin/env python3
import os
import re
import json
import time
import random
import requests
from typing import List, Dict, Any

# Load environment variables
def load_env_file(filepath: str = ".env"):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and val:
                        os.environ[key] = val

load_env_file()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Error: GEMINI_API_KEY not found in environment or .env file.")

# Define 20 categories to lookup subtopics
from generate_lima_dataset import CATEGORIES, SYSTEM_PROMPT, GREETING_STYLES, has_telugu_script

def jaccard(s1: str, s2: str) -> float:
    w1 = set(s1.lower().split())
    w2 = set(s2.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

def generate_diverse_pair(category: str, context: str, avoid_list: List[str], greeting_style: str) -> Dict[str, str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    avoid_clause = ""
    if avoid_list:
        avoid_clause = "\n\nCRITICAL: To ensure high topic diversity and avoid repetition, you MUST NOT reuse the scenarios, questions, locations, or specific topics of these prompts:\n"
        # Shuffle and take up to 25 to fit within context but give strong negative constraints
        sample_avoid = list(avoid_list)
        if len(sample_avoid) > 25:
            sample_avoid = random.sample(sample_avoid, 25)
        for idx, p in enumerate(sample_avoid, 1):
            avoid_clause += f"- {p}\n"
        avoid_clause += "\nMake your scenario, entities, and locations completely different from the ones listed above. For example, if travel/trips are mentioned above, use a completely different destination. If family/kids are mentioned, use a different dynamic. If exams/study are mentioned, use a different context.\n"
        
    greeting_clause = ""
    if greeting_style:
        greeting_clause = f"\n6. GREETING STYLE RULE: {greeting_style}"
        
    user_instruction = f"""Generate exactly 1 unique, high-quality Telglish instruction-response pair for the category: "{category}".
Use these context concepts for inspiration: {context}. Alternatively, brainstorm a completely unique, highly creative scenario in the category: "{category}".
{avoid_clause}
Rules:
1. The USER prompt must be in natural, conversational Telglish.
2. The ASSISTANT response must directly answer the request, follow the matrix grammar rules, and sound like a natural Hyderabad friend responding in colloquial Telglish.
3. STRICTLY AVOID textbook, robotic, or corporate coaching style.
4. Do NOT use generic English list headers with Telugu suffixes.
5. Never ask follow-up questions or include disclaimers about real-time knowledge/dates. Make it a complete, helpful, final answer.{greeting_clause}

SCENARIO DIVERSITY CHECKLIST:
- Do NOT always use prototypical cities/destinations like "Bengaluru/Bangalore" or "Goa" for travel/trips. Vary with Vizag, Araku, Tirupati, Chennai, Ooty, Coorg, Munnar, Pondicherry, etc.
- Do NOT always use "Biryani" or "Chicken Biryani" for cooking. Vary with Dosa, Idli, Upma, Pulihora, Rasam, Pappu, Kheer, etc.
- Do NOT always use "semester exams" or "failing exams" for exams/studies. Vary with competitive exams (GATE, GRE, UPSC), coding certs, project presentations, learning a language, lab experiments.
- Do NOT always use "kids fighting over toys/remotes" for parenting. Vary with kids learning to ride a bike, asking for a dog, first school stage performance, telling stories, food tantrums.
- Ensure the specific scenario, names (e.g. Ramesh, Suresh, Rahul, Priya), and numbers (prices, dates) are unique and varied."""

    payload = {
        "contents": [{
            "parts": [{"text": f"{user_instruction}"}]
        }],
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "thinkingConfig": {
                "thinkingBudget": 0
            },
            "temperature": 0.9, # Higher temperature for maximum creativity
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "prompt": {"type": "STRING"},
                    "response": {"type": "STRING"}
                },
                "required": ["prompt", "response"]
            }
        }
    }
    
    # Retry logic up to 5 times
    for attempt in range(1, 6):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=45)
            response.raise_for_status()
            res_json = response.json()
            
            text_content = res_json["candidates"][0]["content"]["parts"][0]["text"]
            item = json.loads(text_content.strip())
            
            p = item.get("prompt", "").strip()
            r = item.get("response", "").strip()
            
            if not p or not r:
                continue
                
            if has_telugu_script(p) or has_telugu_script(r):
                continue
                
            # Basic validation
            telugish_indicators = ["nenu", "nuvvu", "memu", "meeru", "ledu", "avunu", "lo", "ki", "ra", "ga", "enti", "enduku", "ela", "ekkada", "sare", "alage", "vachha", "vacha", "unda", "undhi", "undi", "chesa", "chesya", "chey", "vaddu", "vaddhu", "kuda", "kooda", "chala", "chaala", "kani", "kaani", "leda", "ante", "cheppandi", "cheppu"]
            p_lower = p.lower()
            
            has_p_telugish = any(word in p_lower.split() or f" {word} " in f" {p_lower} " or p_lower.endswith(f" {word}") or p_lower.startswith(f"{word} ") for word in telugish_indicators)
            
            if not has_p_telugish:
                english_only_indicators = ["you", "should", "the", "before", "running", "how", "what", "where", "why", "who", "which", "is", "a", "an"]
                is_p_pure_english = all(word in english_only_indicators for word in p_lower.split() if len(word) > 2) and len(p_lower.split()) > 3
                if is_p_pure_english:
                    continue

            return {"prompt": p, "response": r}
        except Exception as e:
            time.sleep(attempt * 2)
            
    return {}

def main():
    dataset_path = "data/train_sft_lima.json"
    metadata_path = "data/train_sft_lima_metadata.json"
    
    if not os.path.exists(dataset_path) or not os.path.exists(metadata_path):
        print("Error: train_sft_lima.json or train_sft_lima_metadata.json not found.", flush=True)
        return
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    if len(data) != len(metadata):
        print("Error: Length mismatch between dataset and metadata.", flush=True)
        return
        
    print(f"Loaded {len(data)} items.", flush=True)
    
    # Identify waste indices
    seen_prompts = {}
    for i, x in enumerate(data):
        p = x['prompt']
        if p in seen_prompts:
            seen_prompts[p].append(i)
        else:
            seen_prompts[p] = [i]

    exact_dup_indices = set()
    for p, idxs in seen_prompts.items():
        if len(idxs) > 1:
            for dup_idx in idxs[1:]:
                exact_dup_indices.add(dup_idx)

    near_dup_indices = set()
    threshold = 0.45
    for idx in range(len(data)):
        if idx in exact_dup_indices:
            continue
        p = data[idx]['prompt']
        for other_idx in range(idx):
            if other_idx in exact_dup_indices or other_idx in near_dup_indices:
                continue
            if jaccard(p, data[other_idx]['prompt']) > threshold:
                near_dup_indices.add(idx)
                break

    waste_indices = sorted(list(exact_dup_indices | near_dup_indices))
    print(f"Found {len(waste_indices)} waste items to replace (out of 1000).", flush=True)
    
    if not waste_indices:
        print("No duplicates or near-duplicates found. Done!", flush=True)
        return
        
    # Rewrite loop
    for step, idx in enumerate(waste_indices):
        category = metadata[idx].get("category")
        old_prompt = data[idx]["prompt"]
        print(f"\n[{step+1}/{len(waste_indices)}] Rewriting index {idx} in category '{category}'", flush=True)
        print(f"  Old prompt: '{old_prompt[:80]}...'", flush=True)
        
        # Build avoid list: all other useful prompts in the same category
        avoid_list = [
            data[i]["prompt"] for i in range(len(data))
            if i not in waste_indices and metadata[i]["category"] == category
        ]
        
        # Select category details to get subtopics
        cat_info = next((c for c in CATEGORIES if c["category"] == category), {})
        subtopics = cat_info.get("subtopics", [])
        context_desc = random.choice(subtopics) if subtopics else category
        
        # Generate new pair
        success = False
        for gen_attempt in range(1, 10):
            greeting_style = random.choice(GREETING_STYLES)
            new_pair = generate_diverse_pair(category, context_desc, avoid_list, greeting_style)
            
            if new_pair:
                new_p = new_pair["prompt"]
                new_r = new_pair["response"]
                
                # Check Jaccard similarity against all existing useful prompts in the dataset
                is_too_similar = False
                for other_idx in range(len(data)):
                    if other_idx == idx or other_idx in waste_indices[step:]:
                        # Skip comparing with ourselves or elements we haven't rewritten yet
                        continue
                    sim = jaccard(new_p, data[other_idx]["prompt"])
                    if sim > 0.45:
                        is_too_similar = True
                        break
                        
                if is_too_similar:
                    print(f"    Attempt {gen_attempt}: Generated prompt too similar to existing prompt. Retrying...", flush=True)
                    continue
                    
                # Success! Update in-place
                data[idx] = new_pair
                metadata[idx] = {
                    "prompt": new_p,
                    "response": new_r,
                    "category": category
                }
                
                # Incrementally save files
                with open(dataset_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
                print(f"    Successfully rewritten index {idx}.", flush=True)
                print(f"    New prompt: '{new_p[:80]}...'", flush=True)
                success = True
                break
            else:
                print(f"    Attempt {gen_attempt}: API generation failed. Retrying...", flush=True)
                
        if not success:
            print(f"    CRITICAL: Failed to regenerate index {idx} after multiple attempts. Skipping.", flush=True)
            
        time.sleep(1.2) # Rate limit gap
        
    print("\n=== DEDUPLICATION AND DIVERSIFICATION COMPLETED ===", flush=True)

if __name__ == "__main__":
    main()
