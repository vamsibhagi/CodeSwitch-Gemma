#!/usr/bin/env python3
import os
import re
import json
import time
import argparse
import requests
import random
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

# 200 cleaned, unique categories derived from the user's list
CATEGORIES = [
    "Disease", "Nutrition", "Exercise", "Restaurant", "Movies", "Books", "Sports", "Government", "School", "College",
    "Online shopping", "Computers", "Home appliances", "Philosophy", "Psychology", "Telugu", "Travel booking", "Site seeing", "Rivalry", "Friends",
    "Weather", "Economy", "War", "History", "Mathematics", "Music", "Parenting", "Astronomy", "Professions", "Chores",
    "Babies, kids", "Nature", "Politics", "Chess", "Fashion", "Pets", "Farming", "Real estate", "Air travel", "Trekking",
    "Crime", "Aliens", "Army", "Mythology", "Religion", "Europe", "Public speaking", "Taxes", "Savings", "Startups",
    "Street food", "Coffee, tea", "Earthquakes, tsunamis, flood", "Wifi, internet", "Tiktok, instagram, facebook", "ChatGPT, Claude, Gemini", "Mobile phones", "Old people", "Mother", "Father",
    "Husband, wife", "Son and daughter", "Sleep", "Memes", "Dance", "Ethics", "Logic", "Birds", "Animals", "K-pop",
    "Pollution", "Trains", "Culture", "Superstitions", "Cheating", "Corruption", "Police", "Expenses", "Rich, poor",
    "Traffic, metro", "Suburbs, cities, towns", "Hyderabad", "South India", "Swiggy, Zomato", "Retirement, pension", "Subsidy", "Capitalism", "Communism", "Telangana",
    "Andhra Pradesh", "Vijayawada", "Bangalore", "Busses", "Cleanliness", "Rain", "Uber, Ola, Rapido", "Lawyers, police", "Yoga", "Time management",
    "Advertisements, marketing, branding", "Foreigners", "Immigration", "Factories", "China, USA", "Biryani", "Cricket", "Adventure", "Traditions", "Festivals",
    "Billionaires", "Holiday", "Geography", "Constitution", "Hostels", "Siblings", "Shoes, bags", "Tattoos, hairstyle", "Apps",
    "Crypto", "Dark web", "Meetings", "Temple, church, mosque", "Lists", "Toilet", "House, apartment, villa", "Maid, cook, watchman", "Gated community", "Trucks",
    "Construction", "Groceries", "Marketplace", "Women", "Men", "LGBTQ", "Society", "Parking", "Waiting", "Fun",
    "Comedy", "Event", "Contract", "Commission", "Feedback, criticism, praise", "Ambition, goal", "Wishes, wants", "Rent", "Daily routine", "Army, navy, Airforce",
    "King, queen", "Laundry", "Beauty", "Kindness", "Arrogance", "Chitchat", "Hobby", "Surprised", "Lie, betray", "Snow",
    "Home design", "Auto repair", "Carpenter", "Colors", "Storage", "Interview prep", "Influencer", "Ladder, spanner", "Feminist", "Anarchy",
    "Tollywood", "Liquor", "DIY", "Footwear", "Meditation", "Weight loss", "Scam", "Spam", "Eco-friendly", "Revolution",
    "Unions", "Evolution", "Chocolate, candy, cake", "Gold", "Visa", "Gossip", "Death", "Africa", "Dog, cat", "Snacks",
    "Soft drinks", "Law", "Business, profit, loss", "Cash, UPI", "Democracy", "Beach, mountain, dessert, forest", "Fake news, fact checking", "Podcast, series", "Donation", "Fishing, hunting",
    "Break up, patch up", "Ministers", "Toys, video games", "Exams"
]

# Ensure unique list
CATEGORIES = list(sorted(set(CATEGORIES)))

# Tone Profiles
TONE_PROFILES = [
    {
        "profile": "Direct / Task-Oriented (Weight: 35%)",
        "description": "The user prompt is a direct instruction, request, or question with absolutely no greetings, slang, or conversational fillers. The response must start directly with the detailed solution, code, or explanation. Under no circumstances should the assistant use casual slang (Macha, Mama, Arey, Bro). Tone is objective, structured, and concise.",
        "weight": 35
    },
    {
        "profile": "Academic / Expert-Learner (Weight: 25%)",
        "description": "The user is a learner, student, or citizen asking an expert (professor, doctor, scientist, historian, lawyer) a detailed, deep question. The response should be educational, structured, and informative. Do NOT use casual slang (Macha, Mama, Arey, Bro). Tone is polite, clear, and highly detailed.",
        "weight": 25
    },
    {
        "profile": "Professional / Business (Weight: 20%)",
        "description": "A business, workplace, or service context (e.g. employee talking to manager, customer writing to support, client consulting a professional). The tone is highly professional, polite, and helpful. Avoid casual slangs entirely.",
        "weight": 20
    },
    {
        "profile": "Casual / Friendly (Weight: 20%)",
        "description": "A conversation between friends, family members, neighbors, or classmates. Tone is casual, colloquial, and warm. Natural use of local slang and friendly terms is encouraged, but must be demographically diverse (not just young males).",
        "weight": 20
    }
]

# Casual Demographic Profiles
CASUAL_DEMOGRAPHICS = [
    "An elderly grandmother talking to her young grandchild in a warm, affectionate tone. Endearments: 'Kanna', 'Bangaram', 'Nanna'. Dialect: Traditional home-style.",
    "A grandchild asking their grandparent for advice or traditional stories in a warm, respectful but casual tone.",
    "A mother advising, guiding, or asking her child about their day/education. Warm, caring, maternal tone. Endearments: 'Kanna', 'Nanna', 'Chitti'.",
    "A father discussing career planning, college choices, or giving practical advice to his child. Guiding, protective parent tone. Endearments: 'Nanna', 'Abbayi'.",
    "A sister and brother (or siblings) discussing home chores, studies, or playfully teasing each other. Slang: 'Annayya', 'Chelli', 'Akka', 'Thammudu'.",
    "Two middle-aged female neighbors or friends chatting about home design, cooking, gardening, or local events. Friendly, warm, everyday household tone. Names/Slang: 'Akka', 'Vadina'.",
    "Two young female professionals or classmates discussing career prep, shopping, or office routines. Modern, educated, urban Telglish. Names: 'Priya', 'Kiran', 'Dear'.",
    "Two young male friends/classmates talking about cricket, gaming, or outdoor adventure. Slang: 'Bro', 'Mama', 'Macha', 'Arey', 'Yaar'. Dialect: Urban/Hyderabad slang.",
    "A customer interacting casually with a local merchant, auto driver, or delivery guy. Tone: Polite, everyday street conversation. Greetings: 'Anna', 'Bhaiya'."
]

SYSTEM_PROMPT = """You are an expert creator of high-quality Telugu-English code-switched SFT datasets for aligning multilingual LLMs.

Your task is to generate realistic, general-purpose instruction-following conversations in natural Romanized Telugu-English ("Tenglish" / "Telglish").

The goal is to teach the model:
- natural Telugu-led code switching
- realistic bilingual conversational flow
- correct Telugu grammatical structure
- natural English word insertion patterns
- stable colloquial speech patterns
- consistency across domains and tones

--------------------------------------------------
LANGUAGE STYLE RULES
--------------------------------------------------

1. **Matrix Language Frame (MLF)**:
   - Telugu grammar is the backbone.
   - Telugu word order must dominate (Subject-Object-Verb structure).
   
2. **Strict English Limitation (Nouns and Technical Entities Only)**:
   - English words are ONLY permitted for concrete nouns/entities (e.g. "smart meter", "database", "exam", "budget", "brand", "marketing", "stress", "corporate social responsibility", "CSR", "AI", "computer", "app", "sensor", "IoT") or domain-specific actions ("calculate cheyyadam", "install cheyyadam", "test cheyyadam", "A/B test").
   - English words are STRICTLY PROHIBITED for:
     - General verbs: Do NOT use English verbs (e.g., "reduce", "discuss", "explain", "improve", "compare", "support", "create", "promote", "increase", "prevent", "develop"). Use Telugu verbs instead (e.g. *thagginchadam*, *charchinchadam*, *vivarinchadam*, *pempodhinchadam*, *polchadam*, *sahayam cheyadam*, *srushtinchadam*, *uthsahaparachadam*, *penchadam*, *vaarincha-dam/aipovadam*, *abhivruddhi cheyadam*).
     - General adjectives: Do NOT use English adjectives (e.g., "effective", "important", "significant", "detailed", "different", "similar", "complex", "easy", "simple"). Use Telugu adjectives instead (e.g. *samardhavanthamaina*, *mukhyamaina*, *keelakamaina*, *vivaramaina*, *veru veru*, *oke laanti*, *klistamaina*, *sulabhamaina*, *saadhaarana*).
     - Common nouns: Do NOT use common English nouns (e.g., "benefits", "strategies", "challenges", "process", "inspection", "quality", "school", "education", "rules", "reputation", "loyalty", "trust", "morals", "science", "diagrams", "walking", "sitting", "time", "ideas", "thoughts", "judgement", "work", "pressure", "documents", "charges", "turnover", "conflict", "clothing"). Use Telugu nouns instead (e.g. *prayojanalu*, *upayalu*, *savaallu*, *vidhanam*, *pariseelana*, *gunathmakatha*, *badi/patashala*, *chaduvu*, *niyamalu*, *peru/khyathi*, *nammakam*, *viluvalu*, *vignanam*, *chithrapathralu*, *nadavadam*, *kurchovadam*, *samayam*, *alochanalu*, *abhiprayalu*, *theerpu*, *pani*, *otthidi*, *pathralu*, *rudhymalu*, *mandi vellipovadam*, *godavalu*, *vastralu/battalu*).
   - In case of doubt, always lean towards a Telugu word than an English word.
   - Do NOT confuse similar-sounding Telugu words: e.g., do NOT translate "clothing" to *pathralu* (which means documents/leaves); instead use *vastralu* or *battalu*.
   - Do NOT write full sentences, bullet-point headers, or lists in English. All structural elements, headers, and bullet lists must be in Romanized Telugu.
   - Ensure the Romanized Telugu vocabulary is rich, grammatically correct, and natural.

3. **No Telugu Script**:
   - STRICTLY AVOID any native Telugu Unicode script characters. ONLY Roman script is allowed.

--------------------------------------------------
ASSISTANT RESPONSE QUALITY RULES
--------------------------------------------------

1. **Reasonably Lengthy & Detailed**: The response must be thorough, detailed, and complete (usually 100-300 words). Do NOT write short, superficial, or 1-2 sentence answers.
2. **Formatting**: Use Markdown formatting (bullet points, numbered steps, tables, or code blocks) to explain concepts in-depth.
3. **No Placeholders**: Never use generic placeholders like [Your Name] or variables like ₹X. Use concrete details.
4. **No Disclaimers**: Never include AI disclaimers or excuses. Provide information directly.
5. **No Follow-up Questions**: Do not ask follow-up questions at the end of the response. Make the response a complete, helpful, final answer.
"""

def has_indic_script(text: str) -> bool:
    indic_pattern = re.compile(r"[\u0900-\u0d7f]")
    return bool(indic_pattern.search(text))

def jaccard(s1: str, s2: str) -> float:
    w1 = set(s1.lower().split())
    w2 = set(s2.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

def generate_one_pair(category: str, tone_profile: Dict[str, Any], casual_demographic: str, avoid_list: List[str]) -> Dict[str, str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    avoid_clause = ""
    if avoid_list:
        avoid_clause = "\n\nCRITICAL: To ensure high topic diversity and avoid repetition, you MUST NOT reuse the scenarios, questions, locations, or topics of these prompts:\n"
        for p in avoid_list[-15:]:
            avoid_clause += f"- {p}\n"
        avoid_clause += "\nMake your scenario, entities, and questions completely different from the ones listed above.\n"
        
    demographic_clause = ""
    if casual_demographic and "Casual" in tone_profile["profile"]:
        demographic_clause = f"\nDEMOGRAPHIC PROFILE FOR CASUAL CONVERSATION:\n- Relationship/Context: {casual_demographic}\nEnsure the language and tone reflect this demographic context (e.g. if grandmother/grandchild, use grandparent-style endearments; if female neighbor, avoid young male slang like 'macha' or 'mama')."

    user_instruction = f"""Generate exactly 1 unique, high-quality, general-purpose SFT pair in Telglish for the category: "{category}".

TONE PROFILE SPECIFICATION:
- Profile: {tone_profile["profile"]}
- Description & Guidelines: {tone_profile["description"]}
{demographic_clause}
{avoid_clause}
RULES FOR USER PROMPT:
1. The USER prompt must have sufficient depth, detail, or complexity (e.g. asking for explanations, comparisons, step-by-step instructions, or code with comments) to naturally justify a detailed, lengthy response. Do NOT generate simple or short questions.
2. Tone: Must align with the specified Tone Profile.
   - If Direct, Academic, or Professional: **Do NOT use casual slangs like 'Macha', 'Mama', 'Arey', or 'Bro'.**
   - If Casual: Vary the slang based on the specified demographic profile.
3. Language: Minimize English words. Use as much Romanized Telugu as possible. In case of doubt, lean towards a Telugu word than an English word.

RULES FOR ASSISTANT RESPONSE:
1. Provide a detailed, comprehensive response (usually 100-300 words). Use formatting (lists, tables, code blocks) to explain in-depth.
2. Language: Minimize English words. Use as much Romanized Telugu as possible. In case of doubt, lean towards a Telugu word than an English word. Do NOT write full sentences or bullet-point headers in English if Telugu words/phrases are available.
3. The tone must mirror the user's tone profile (e.g., if Direct, start directly with the solution. No chatty greetings).

OUTPUT FORMAT:
Return ONLY valid JSON matching this schema:
{{
  "prompt": "...",
  "response": "..."
}}
No markdown formatting fences. No extra text.
"""

    payload = {
        "contents": [{
            "parts": [{"text": user_instruction}]
        }],
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "thinkingConfig": {
                "thinkingBudget": 0
            },
            "temperature": 0.9, # Higher temperature for maximum scenario diversity
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
                
            if has_indic_script(p) or has_indic_script(r):
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

def select_tone_profile() -> Dict[str, Any]:
    # Weighted choice based on profile weights
    profiles = TONE_PROFILES
    weights = [p["weight"] for p in profiles]
    return random.choices(profiles, weights=weights, k=1)[0]

def main():
    parser = argparse.ArgumentParser(description="LIMA 2.0 Telglish SFT Dataset Generator")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3], help="Rollout phase: 1 (5 samples), 2 (40 samples), 3 (1000 samples)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the generated JSON dataset")
    args = parser.parse_args()

    print(f"=== STARTING LIMA 2.0 GENERATOR (PHASE {args.phase}) ===", flush=True)

    # 1. Determine targets based on phase
    total_expected = 0
    target_per_cat = 0
    
    if args.phase == 1:
        total_expected = 5
        target_per_cat = 1
        # Use a random subset of 5 categories for phase 1
        selected_categories = random.sample(CATEGORIES, 5)
    elif args.phase == 2:
        total_expected = 40
        target_per_cat = 1
        # Use a random subset of 40 categories for phase 2
        selected_categories = random.sample(CATEGORIES, 40)
    else: # Phase 3
        total_expected = 1000
        target_per_cat = 5
        selected_categories = CATEGORIES * 5 # 200 * 5 = 1000 jobs
        random.seed(42)
        random.shuffle(selected_categories)

    # Load existing progress if any
    all_data = []
    metadata_data = []
    metadata_path = args.output.replace(".json", "_metadata.json")
    
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                all_data = json.load(f)
            print(f"Loaded {len(all_data)} existing entries from {args.output}", flush=True)
        except Exception as e:
            print(f"Warning: could not parse existing file: {e}. Starting fresh.", flush=True)

    if all_data and os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata_data = json.load(f)
        except Exception as e:
            pass

    # Count how many we currently have
    current_count = len(all_data)
    if current_count >= total_expected:
        print(f"Dataset already has {current_count} items. Targets met!", flush=True)
        return

    # Build remaining jobs
    remaining_categories = selected_categories[current_count:]
    print(f"Remaining items to generate: {len(remaining_categories)}", flush=True)

    for idx, category in enumerate(remaining_categories):
        current_idx = current_count + idx
        
        # Pick Tone Profile
        tone_profile = select_tone_profile()
        
        # Pick Demographic profile if Casual
        casual_demographic = ""
        if "Casual" in tone_profile["profile"]:
            casual_demographic = random.choice(CASUAL_DEMOGRAPHICS)
            
        print(f"\n[{current_idx+1}/{total_expected}] Generating for Category: '{category}' | Tone: {tone_profile['profile'].split('(')[0].strip()}", flush=True)
        if casual_demographic:
            print(f"  Demographic: {casual_demographic.split('.')[0]}", flush=True)

        # Get existing prompts for this category to avoid duplication
        avoid_list = [
            item["prompt"] for item in metadata_data if item.get("category") == category
        ]
        
        # Also avoid recent prompts across all categories to prevent repeating global patterns
        recent_global = [item["prompt"] for item in metadata_data[-15:]]
        avoid_list.extend(recent_global)
        avoid_list = list(set(avoid_list))

        success = False
        for attempt in range(1, 6):
            pair = generate_one_pair(category, tone_profile, casual_demographic, avoid_list)
            if pair:
                new_p = pair["prompt"]
                new_r = pair["response"]
                
                # Double-check Jaccard similarity against all existing prompts in this category
                is_duplicate = False
                for other_item in metadata_data:
                    if other_item.get("category") == category:
                        sim = jaccard(new_p, other_item["prompt"])
                        if sim > 0.45:
                            is_duplicate = True
                            break
                if is_duplicate:
                    print(f"  Attempt {attempt}: Generated prompt too similar to an existing prompt in this category. Retrying...", flush=True)
                    continue
                
                all_data.append(pair)
                metadata_data.append({
                    "prompt": new_p,
                    "response": new_r,
                    "category": category,
                    "tone_profile": tone_profile["profile"],
                    "casual_demographic": casual_demographic
                })
                
                # Save progress
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(all_data, f, indent=2, ensure_ascii=False)
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata_data, f, indent=2, ensure_ascii=False)
                    
                print(f"  Successfully saved. Prompt snippet: '{new_p[:70]}...'", flush=True)
                success = True
                break
            else:
                print(f"  Attempt {attempt} failed (Telugu script or validation error). Retrying...", flush=True)
                
        if not success:
            print(f"  CRITICAL: Failed to generate for Category '{category}' after 5 attempts. Skipping to keep pipeline alive.", flush=True)
            
        time.sleep(1.2) # Rate limit gap

    print(f"\n=== GENERATION COMPLETED (PHASE {args.phase}) ===", flush=True)
    print(f"Total entries: {len(all_data)} saved to {args.output}", flush=True)

if __name__ == "__main__":
    main()
