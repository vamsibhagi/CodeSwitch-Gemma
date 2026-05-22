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

# Define 20 topics to ensure rich vocabulary and diversity
TOPICS = [
    {"topic": "Tech & Coding", "context": "debugging, code review, compiler blockers, git commands, deployment, API issues, server crash"},
    {"topic": "Casual Food & Dining", "context": "ordering biryani, cafe visits, cooking dinner, trying new restaurants, coffee, tea preference"},
    {"topic": "Office & Workplace", "context": "meetings, strict deadlines, manager updates, coworker chat, performance reviews, appraisals, slack messages"},
    {"topic": "Gym, Health & Fitness", "context": "daily workouts, gym memberships, diet plans, muscle soreness, running, yoga"},
    {"topic": "Shopping & Finance", "context": "monthly expenses, credit card bills, online shopping, discounts, saving money, rent payment"},
    {"topic": "Movies & Entertainment", "context": "watching series, movie reviews, climax twists, binge-watching Netflix, actors, background score"},
    {"topic": "Sports & Games", "context": "cricket match, playing badminton, weekend football, video games, PS5, gaming stream"},
    {"topic": "Weather & Commute", "context": "heavy rain, pleasant evening, morning walk, heavy traffic, metro ride, bike commute"},
    {"topic": "Travel & Holidays", "context": "weekend trip plans, booking tickets, packing bags, sightseeing, itinerary, hotels"},
    {"topic": "Social Media & Apps", "context": "reels scrolling, screen sharing, screenshot request, WhatsApp status, notifications"},
    {"topic": "Family & Home", "context": "parent calls, sibling talks, cleaning room, grocery shopping, childhood memory, relatives visiting"},
    {"topic": "Exams & Education", "context": "college exams, exam preparation, library study, results day, college admission, professor attitude"},
    {"topic": "Everyday Tech Issues", "context": "charger missing, battery drain, hotspot connection, mobile data recharge, Wi-Fi outage"},
    {"topic": "Interviews & Career", "context": "resume update, job application, HR round, technical interview, offer letter, salary package negotiation"},
    {"topic": "Daily Routines", "context": "waking up early, alarm snooze, sleeping schedule, running late, morning coffee, weekend laziness"},
    {"topic": "Hobbies & Leisure", "context": "playing guitar, photography, painting, gardening, reading novels, listening to music"},
    {"topic": "Rentals & Flatmates", "context": "finding a flat, flatmate behavior, cooking rotations, sharing bills, landlord issues"},
    {"topic": "Medical & Well-being", "context": "doctor checkup, fever, buying medicine, headache, taking rest, dental pain"},
    {"topic": "Urban Life & Shopping", "context": "mall shopping, street food, grocery delivery apps, local markets, bargaining"},
    {"topic": "General Friendly Chat", "context": "greetings, asking about life, sharing small jokes, casual plans, gossip, catch-up"}
]

SYSTEM_PROMPT = """You are a master generator of high-quality Romanized Telugu-English (Telglish) conversation datasets.
Your goal is to generate 50 unique conversational instruction-response pairs where:
1. The USER prompt is a casual query or statement in natural Telglish.
2. The ASSISTANT response is a natural, conversational response in flawless, high-quality Telglish that would score a perfect 4/4 on both evaluation axes:
   - Axis 1 (Grammatical Integrity): Flows natively, strictly maintains Telugu Subject-Object-Verb (SOV) structure. No broken or hallucinated words. FULLY written in Romanized alphabet (NO Telugu script characters).
   - Axis 2 (Code-Switch Naturalness): Perfect Matrix Language Frame. Telugu is the host/matrix grammar. English is embedded purely as nouns/verbs/adjectives conforming to Telugu case markers and inflections (e.g. "trip ki", "plan chesthanu", "reset chey"). No raw English syntax or clauses (e.g. no sentences like "You should try it next time").

Return ONLY a JSON array containing exactly 50 objects matching this schema:
[
  {
    "prompt": "user prompt in casual Telglish",
    "response": "assistant response in flawless Telglish (Score 4/4 on grammar and code-switching)"
  }
]"""

def has_telugu_script(text: str) -> bool:
    # Telugu Unicode Range: \u0c00 to \u0c7f
    telugu_pattern = re.compile(r"[\u0c00-\u0c7f]")
    return bool(telugu_pattern.search(text))

def generate_batch(topic: str, context: str, batch_num: int) -> List[Dict[str, str]]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    user_instruction = f"""Generate exactly 50 unique and natural Telglish conversation pairs.
The conversations MUST focus on the topic: "{topic}" (Context words to use/inspire: {context}).
Make sure the prompts and responses vary in sentence length, structure, and vocabulary.
Keep responses short, warm, and highly conversational (1-2 sentences maximum, like chat/WhatsApp messages).
Do not use any Telugu script characters anywhere in the prompts or responses.
"""
    
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
    
    # Retry logic up to 3 times
    for attempt in range(1, 4):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=45)
            response.raise_for_status()
            res_json = response.json()
            
            text_content = res_json["candidates"][0]["content"]["parts"][0]["text"]
            batch_data = json.loads(text_content)
            
            # Validate output and filter script leakage
            valid_pairs = []
            for item in batch_data:
                p = item.get("prompt", "").strip()
                r = item.get("response", "").strip()
                if not p or not r:
                    continue
                if has_telugu_script(p) or has_telugu_script(r):
                    print(f"Skipped an item due to Telugu script leakage: {item}")
                    continue
                valid_pairs.append({"prompt": p, "response": r})
                
            return valid_pairs
        except Exception as e:
            print(f"Attempt {attempt} failed for topic '{topic}': {e}")
            if attempt < 3:
                time.sleep(2 ** attempt)
            else:
                print("Max retries exceeded for this batch.")
                return []

def main():
    print("--- STARTING DATASET GENERATION (1000 EXAMPLES) ---")
    
    # Load existing data if file exists to support resumption
    all_data = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                all_data = json.load(f)
            print(f"Loaded {len(all_data)} existing examples from {OUTPUT_FILE}")
        except Exception as e:
            print(f"Failed to load existing file, starting fresh: {e}")

    # Determine which topics still need to be generated
    # Each topic represents 50 examples.
    completed_batches = len(all_data) // 50
    print(f"Completed batches so far: {completed_batches} / {len(TOPICS)}")
    
    for i in range(completed_batches, len(TOPICS)):
        t_info = TOPICS[i]
        topic_name = t_info["topic"]
        context_desc = t_info["context"]
        
        print(f"\nGenerating Batch {i+1}/{len(TOPICS)} - Topic: {topic_name}")
        batch_pairs = generate_batch(topic_name, context_desc, i+1)
        
        if batch_pairs:
            all_data.extend(batch_pairs)
            # Incremental save
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved. Total examples now: {len(all_data)}")
        else:
            print(f"Warning: Failed to generate batch {i+1}")
            
        time.sleep(1.0) # Small rate-limiting gap

    print(f"\nDataset generation completed. Total entries: {len(all_data)} saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
