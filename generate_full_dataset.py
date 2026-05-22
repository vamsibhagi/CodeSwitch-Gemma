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

OUTPUT_FILE = "tenglish_train_data.json"

# Define 30 diverse everyday categories to cover a wide spectrum of life without office/coding dominance
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
    {"category": "Job Hunting & Career Setup", "context": "updating resumes, checking job portals, interview calls, salary packages, mock tests"},
    {"category": "Nostalgic Childhood Games", "context": "gilli-danda, hide and seek, school ground games, old school friends"},
    {"category": "Festival & Celebrations", "context": "Diwali crackers, Sankranti kites, Dussehra pooja, making sweets, new clothes for festivals"},
    {"category": "House Hunting & Real Estate", "context": "searching for 2BHK, house rent agreement, shifting packers and movers, house warming"},
    {"category": "Bank & Personal Savings", "context": "opening bank account, fixed deposit, ATM not working, UPI transaction fail, saving cash"},
    {"category": "Books & Reading Habits", "context": "buying books at book fair, reading newspapers, storybooks, libraries"},
    {"category": "Gardening & Home Plants", "context": "watering flowers, growing tomatoes, organic composting, plant nurseries"},
    {"category": "Cooking Failures & Successes", "context": "burning curry, learning to cook round rotis, adding too much salt, trying new recipes"},
    {"category": "Vehicle Maintenance", "context": "scooty puncture, car washing, engine oil change, driving license test, pollution check"},
    {"category": "Local Sightseeing & Markets", "context": "bargaining with vendors, buying fresh vegetables, Sunday flower market, street shopping"},
    {"category": "Tech Hacks & Home Appliances", "context": "using smart TV, washing machine settings, microwave ovens, power cuts"}
]

SYSTEM_PROMPT = """You are a master generator of natural Romanized Telugu-English (Telglish) conversation datasets.
Your goal is to generate 34 unique, highly realistic conversation pairs (prompt & response) for a specified everyday category.

Rules for Natural Telglish (Matrix/Host Language = Telugu):
1. Telugu Subject-Object-Verb (SOV) structure MUST be strictly followed.
2. REDUCE ENGLISH TO THE ABSOLUTE MINIMUM. Do NOT substitute common, simple everyday Telugu words with English.
   - Use Telugu nouns:
     * Use "varsham" instead of "rain"
     * Use "yenda" instead of "summer heat" or "sun"
     * Use "cinema" instead of "movie"
     * Use "paata" or "paatalu" instead of "song" / "songs"
     * Use "tindi" or "bhojanam" instead of "food"
     * Use "neellu" instead of "water"
     * Use "pusthakam" instead of "book"
     * Use "illu" instead of "house / home"
     * Use "pani" instead of "work / task"
     * Use "jeetham" instead of "salary"
     * Use "kukka" or "pilli" instead of "dog" / "cat"
     * Use "roju" instead of "day"
     * Use "panti noppi" or "thala noppi" instead of "toothache" / "headache"
     * Use "bayam" or "digulu" instead of "fear" / "tension" / "panic"
   - Use Telugu verbs where possible:
     * Use "choodu" instead of "check chey"
     * Use "ethuku" instead of "search chey"
     * Use "konalu" instead of "buy chey"
     * Use "matladu" instead of "talk chey"
     * Use "thinadam" instead of "eat chey"
     * Use "vellu" instead of "go chey"
   - Use Telugu grammar and connectors ALWAYS:
     * NO "sometimes" -> use "appudappudu"
     * NO "very" -> use "chala" or "chana"
     * NO "okay/sure" -> use "sare" or "alage"
     * NO "always" -> use "eppudu"
     * NO "after/later" -> use "taruvatha"
     * NO "before" -> use "mundhu"
     * NO "why" -> use "enduku"
     * NO "how" -> use "ela"
     * NO "what" -> use "emiti" or "em"
     * NO "where" -> use "ekkada"
     * NO "now" -> use "ippudu"
     * NO "late/fast" -> use "late ga" / "thwaraga"
     * NO "new/old" -> use "kotha" / "paatha"
     * NO "small/big" -> use "chinna" / "pedda"
     * NO "little/bit" -> use "konchem" / "koddiga"
     * NO "yes/no" -> use "avunu" / "ledu"
     * NO "also/too" -> use "kuda"
     * NO "about" -> use "gurinchi"
     * NO "but" -> use "kani"
     * NO "or" -> use "leda"
     * NO "I/we/you" -> use "nenu" / "memu" / "nuvvu" / "meeru"
3. ONLY use English for nouns, adjectives, or verbs that have NO natural everyday Telugu equivalent in modern urban spoken speech (e.g. "reels", "battery", "Wi-Fi", "traffic", "metro", "biryani", "pizza", "Netflix", "PS5", "badminton", "charger", "laptop", "office", "class", "exam", "college", "interview", "resume", "flatmate", "rent", "fridge", "smart TV", "microwave").
4. Attach Telugu case-markers or verb-suffixes to English words to make them flow naturally:
   - "Wi-Fi signals asala levu"
   - "metro lo baga crowd undhi"
   - "reels scroll chesthu kurchunna"
5. Avoid raw English clauses or phrases. No sentences that are syntactically English.
6. Ensure absolutely NO Telugu script characters (Telugu Unicode range) are present. Only use the Roman alphabet.

Return ONLY a JSON array containing exactly 34 objects matching this schema:
[
  {
    "prompt": "user prompt in casual, natural Telglish with minimal English",
    "response": "assistant response conforming to Telugu matrix grammar and rules above"
  }
]"""

def has_telugu_script(text: str) -> bool:
    telugu_pattern = re.compile(r"[\u0c00-\u0c7f]")
    return bool(telugu_pattern.search(text))

def generate_category_batch(category: str, context: str) -> List[Dict[str, str]]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    user_instruction = f"""Generate exactly 34 unique and natural Telglish conversation pairs for the category: "{category}".
Use these context concepts for inspiration: {context}.
Keep responses warm, short, and highly conversational (1-2 sentences maximum).
Ensure strict adherence to the system rules: reduce English to the absolute minimum, and use Telugu words instead of English for basic vocabulary like 'movie', 'song', 'water', 'rain', 'food', 'work', etc. Use Telugu verbs instead of English verbs where natural."""

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
            response = requests.post(url, headers=headers, json=payload, timeout=45)
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
                    continue
                valid_pairs.append({"prompt": p, "response": r})
            
            if len(valid_pairs) >= 30:
                print(f"Generated {len(valid_pairs)} valid pairs for {category}", flush=True)
                return valid_pairs
            else:
                print(f"Attempt {attempt}: Only got {len(valid_pairs)} valid pairs. Retrying...", flush=True)
        except Exception as e:
            print(f"Attempt {attempt} failed for category '{category}': {e}", flush=True)
            time.sleep(2 ** attempt)
    return []

def main():
    print("--- STARTING GENERATION OF FULL DATASET (1020 EXAMPLES) ---", flush=True)
    all_data = []
    
    # Support resumption of progress
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                all_data = json.load(f)
            print(f"Loaded {len(all_data)} existing examples from {OUTPUT_FILE}", flush=True)
        except Exception as e:
            print(f"Failed to load existing file, starting fresh: {e}", flush=True)

    completed_categories = len(all_data) // 34
    print(f"Already completed {completed_categories} / {len(CATEGORIES)} categories.", flush=True)
    
    # Ensure all_data is clean (truncated to multiple of 34 to avoid partial corrupted batches)
    all_data = all_data[:completed_categories * 34]
    
    for idx in range(completed_categories, len(CATEGORIES)):
        cat_info = CATEGORIES[idx]
        cat_name = cat_info["category"]
        context_desc = cat_info["context"]
        print(f"\n[{idx+1}/30] Generating 34 examples for: {cat_name}", flush=True)
        
        pairs = generate_category_batch(cat_name, context_desc)
        if pairs:
            # Crop to exactly 34 pairs to maintain structure
            pairs = pairs[:34]
            all_data.extend(pairs)
            # Incremental save
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)
            print(f"Successfully generated batch. Cumulative count: {len(all_data)}", flush=True)
        else:
            print(f"CRITICAL WARNING: Failed to generate batch for {cat_name}", flush=True)
            
        time.sleep(1.5) # Sleep to avoid rate limits
        
    print(f"\nCompleted generating full dataset! Total examples generated: {len(all_data)}", flush=True)
    print(f"Saved to {OUTPUT_FILE}", flush=True)

if __name__ == "__main__":
    main()
