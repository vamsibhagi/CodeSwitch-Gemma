#!/usr/bin/env python3
import os
import re
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

OUTPUT_FILE = "tenglish_sample_100.json"

# Define 20 diverse everyday categories (avoiding purely coding/technical/corporate office focus)
CATEGORIES = [
    {"category": "Casual Food & Cravings", "context": "eating street food, ordering pizza/biryani, tea/coffee breaks, sweet cravings, cooking simple food"},
    {"category": "Everyday Gadget Issues", "context": "phone battery drain, screen crack, missing charger, poor Wi-Fi, hotspot sharing"},
    {"category": "Sports & Play", "context": "playing badminton, weekend cricket, watching match highlights, PS5 gaming, carroms"},
    {"category": "Movies & Series", "context": "watching new movies, bingeing Netflix, theater experiences, music playlists, favorite actors"},
    {"category": "Urban Commuting", "context": "booking auto/cab, metro crowd, bike rides, train journey, station arrival"},
    {"category": "Household Chores", "context": "cleaning the room, washing dishes, trash disposal, water scarcity, laundry day"},
    {"category": "Weather & Nature", "context": "heavy rain, hot summer days, pleasant evening breeze, morning walks, winter cold"},
    {"category": "Gym & Fitness", "context": "lifting weights, running in park, muscle soreness, protein intake, yoga"},
    {"category": "Shopping & Clothes", "context": "buying clothes, online discounts, mall shopping, shoe sizes, return policies"},
    {"category": "Family & Relatives", "context": "parent phone calls, cousin marriages, relatives visiting, mother's cooking, home environment"},
    {"category": "College & Exams", "context": "semester exams, group studies, library hours, assignment deadlines, results tension"},
    {"category": "Flatmates & Rent", "context": "paying security deposit, sharing grocery bills, room cleaning rotations, landlord issues"},
    {"category": "Social Media Scrolling", "context": "scrolling reels/shorts, sending memes, WhatsApp status updates, notification distractions"},
    {"category": "Pets & Animals", "context": "feeding pet dog, stray cats, pet shop visits, vaccinations, playful puppy behaviors"},
    {"category": "Minor Sickness & Health", "context": "headache, cough and cold, visiting pharmacy, taking syrup, resting in bed"},
    {"category": "Travel & Trips", "context": "weekend getaway plans, packing luggage, hotel booking, scenic viewpoints, sightseeing"},
    {"category": "Sleep & Morning Routines", "context": "snoozing alarm, waking up late, late-night sleep, feeling sleepy at afternoon"},
    {"category": "Hobbies & Leisure", "context": "gardening plants, learning guitar/music, watercolor painting, reading fiction novels"},
    {"category": "Street Food & Chat", "context": "eating pani puri, samosa cravings, local tea stall chats, kulfi on hot days"},
    {"category": "Job Hunting & Career Setup", "context": "updating resumes, checking job portals, interview calls, salary packages, mock tests"}
]

SYSTEM_PROMPT = """You are a master generator of natural Romanized Telugu-English (Telglish) conversation datasets.
Your goal is to generate 5 conversation pairs (prompt & response) for a specified everyday category.

Rules for Natural Telglish (Host Matrix = Telugu):
1. The host/matrix grammar MUST be Telugu (Subject-Object-Verb structure).
2. DO NOT use English words when simple, everyday Telugu words exist.
   - NO "sometimes" -> use "appudappudu"
   - NO "very" -> use "chala" or "chana"
   - NO "okay/sure" -> use "sare" or "alage"
   - NO "always" -> use "eppudu"
   - NO "after/later" -> use "taruvatha"
   - NO "before" -> use "mundhu"
   - NO "why" -> use "enduku"
   - NO "how" -> use "ela"
   - NO "what" -> use "emiti" or "em"
   - NO "where" -> use "ekkada"
   - NO "now" -> use "ippudu"
   - NO "late/fast" -> use "late ga" / "thwaraga"
   - NO "new/old" -> use "kotha" / "paatha"
   - NO "small/big" -> use "chinna" / "pedda"
   - NO "little/bit" -> use "konchem" / "koddiga"
   - NO "yes/no" -> use "avunu" / "ledu"
   - NO "also/too" -> use "kuda"
   - NO "about" -> use "gurinchi"
   - NO "but" -> use "kani"
   - NO "or" -> use "leda"
   - NO "I/we/you" -> use "nenu" / "memu" / "nuvvu" / "meeru"
3. ONLY use English for nouns, adjectives, and verbs that do NOT have a simple/natural everyday Telugu counterpart in casual conversation (e.g., "reels", "battery", "Wi-Fi", "traffic", "metro", "biryani", "pizza", "movie", "serial", "post", "status", "gym", "shopping", "ticket", "booking", "flat", "rent", "fridge", "car", "PS5", "badminton", "charger", "laptop", "office", "class", "exam", "college", "interview", "resume", "salary", "bonus", "meeting").
4. Attach Telugu case-markers or verb-suffixes to English words to make them flow naturally:
   - "Wi-Fi check chesanu"
   - "gym ki vella"
   - "reels scroll chesthunna"
   - "auto book chey"
5. Avoid raw English clauses or phrases.
   - BAD: "Sometimes. Kani traffic untundi sometimes."
   - GOOD: "Appudappudu. Kani heavy traffic untundi."
   - BAD: "You should try it next time."
   - GOOD: "Nuvvu kuda next time try chesi choodu."
6. Ensure absolutely NO Telugu script characters (Telugu Unicode range) are present. Only use the Roman alphabet.

Return ONLY a JSON array containing exactly 5 objects matching this schema:
[
  {
    "prompt": "user prompt in casual, natural Telglish",
    "response": "assistant response in natural, conversational Telglish conforming to Telugu matrix grammar and rules above"
  }
]"""

def has_telugu_script(text: str) -> bool:
    telugu_pattern = re.compile(r"[\u0c00-\u0c7f]")
    return bool(telugu_pattern.search(text))

def generate_five_examples(category: str, context: str) -> List[Dict[str, str]]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    user_instruction = f"""Generate exactly 5 unique and natural Telglish conversation pairs for the category: "{category}".
Use these context concepts for inspiration: {context}.
Keep responses warm, short, and highly conversational (1-2 sentences maximum).
Ensure strict adherence to the system rules: do not replace basic words like 'very', 'sometimes', 'okay', 'but', 'why', etc., with English. Keep the host matrix as Telugu."""

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
            "responseSchema": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "prompt": {"type": "STRING"},
                        "response": {"type": "STRING"}
                    },
                    "required": ["prompt", "response"]
                }
            }
        }
    }
    
    for attempt in range(1, 4):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            res_json = response.json()
            
            text_content = res_json["candidates"][0]["content"]["parts"][0]["text"]
            batch_data = json.loads(text_content)
            
            valid_pairs = []
            for item in batch_data:
                p = item.get("prompt", "").strip()
                r = item.get("response", "").strip()
                if not p or not r:
                    continue
                if has_telugu_script(p) or has_telugu_script(r):
                    print(f"Skipped an item due to Telugu script: {item}")
                    continue
                valid_pairs.append({"prompt": p, "response": r})
            
            if len(valid_pairs) == 5:
                return valid_pairs
            else:
                print(f"Attempt {attempt}: Only got {len(valid_pairs)} valid pairs instead of 5. Retrying...")
        except Exception as e:
            print(f"Attempt {attempt} failed for category '{category}': {e}")
            time.sleep(2)
    return []

def main():
    print("--- STARTING GENERATION OF 100 SAMPLE EXAMPLES ---")
    all_data = []
    
    for idx, cat_info in enumerate(CATEGORIES):
        cat_name = cat_info["category"]
        context_desc = cat_info["context"]
        print(f"[{idx+1}/20] Generating 5 examples for: {cat_name}")
        
        pairs = generate_five_examples(cat_name, context_desc)
        if pairs:
            # Add category tag to helper metadata if we want, but user schema is prompt/response.
            # We keep it as prompt/response to match exact train requirements.
            all_data.extend(pairs)
            print(f"Successfully generated 5 pairs for {cat_name}.")
        else:
            print(f"CRITICAL WARNING: Failed to generate pairs for {cat_name}.")
            
        time.sleep(1.0)
        
    print(f"\nCompleted! Total examples generated: {len(all_data)}")
    
    # Save output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
