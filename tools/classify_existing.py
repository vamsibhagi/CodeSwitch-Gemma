#!/usr/bin/env python3
import os
import json
import time
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

# Category names only
CATEGORIES = [
    "Cooking & Recipes",
    "Parenting & Childhood",
    "Relationships & Family",
    "Transportation & Commuting",
    "Fitness & Routines",
    "Weather & Nature",
    "Movies & TV Shows",
    "Cricket & Sports",
    "Gaming & YouTube",
    "Memes & Social Media",
    "Trip Planning & Travel",
    "Budgeting & Personal Finance",
    "Customer Support & Products",
    "Home Maintenance & Chores",
    "Life Advice & Motivation",
    "Youth & Casual Slang",
    "Exams & Study Plans",
    "Career Prep & Job Hunt",
    "Technical Explanations & AI",
    "Coding & Debugging"
]

def classify_prompts_batch(prompts: List[str]) -> List[str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    prompt_list_str = "\n".join([f"{idx+1}. {p}" for idx, p in enumerate(prompts)])
    
    user_instruction = f"""You are a classification assistant. Classify each of the following Telglish prompts into exactly one of these 20 categories:
{", ".join(CATEGORIES)}

Return a JSON object containing a list 'categories' where each element corresponds to the category of the prompt at that index.

Prompts to classify:
{prompt_list_str}"""

    payload = {
        "contents": [{
            "parts": [{"text": user_instruction}]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "thinkingConfig": {
                "thinkingBudget": 0
            },
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "categories": {
                        "type": "ARRAY",
                        "items": {
                            "type": "STRING",
                            "enum": CATEGORIES
                        }
                    }
                },
                "required": ["categories"]
            }
        }
    }
    
    for attempt in range(1, 4):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            res_json = response.json()
            text_content = res_json["candidates"][0]["content"]["parts"][0]["text"]
            item = json.loads(text_content.strip())
            cats = item.get("categories", [])
            if len(cats) == len(prompts):
                return cats
            print(f"Warning: classification batch length mismatch ({len(cats)} vs {len(prompts)}). Retrying... (Attempt {attempt})", flush=True)
        except Exception as e:
            print(f"Classification attempt {attempt} failed: {e}", flush=True)
            time.sleep(2)
            
    # Fallback to default/first category if classification failed
    return [CATEGORIES[0]] * len(prompts)

def main():
    input_file = "data/train_sft_lima.json"
    output_metadata = "data/train_sft_lima_metadata.json"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist.")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} prompts from {input_file}.", flush=True)
    
    prompts = [item["prompt"] for item in data]
    
    # Classify in batches of 50
    batch_size = 50
    all_categories = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        print(f"Classifying batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1} (size {len(batch)})...", flush=True)
        batch_cats = classify_prompts_batch(batch)
        all_categories.extend(batch_cats)
        time.sleep(1.0)
        
    print(f"Finished classification. Got {len(all_categories)} categories.", flush=True)
    
    # Construct metadata list
    metadata_list = []
    for idx, item in enumerate(data):
        cat = all_categories[idx] if idx < len(all_categories) else CATEGORIES[0]
        metadata_list.append({
            "prompt": item["prompt"],
            "response": item["response"],
            "category": cat
        })
        
    with open(output_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
        
    print(f"Saved metadata file to {output_metadata}", flush=True)

if __name__ == "__main__":
    main()
