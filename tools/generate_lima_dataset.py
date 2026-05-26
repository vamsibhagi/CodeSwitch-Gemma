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

# Define 20 diverse categories for general-purpose instruct tasks
CATEGORIES = [
    {
        "category": "Cooking & Recipes",
        "subtopics": [
            "cooking traditional chicken biryani for a family get-together",
            "easy, quick evening snacks for sudden guests",
            "healthy breakfast options under 10 minutes for busy mornings",
            "how to bake a simple chocolate cake without an oven",
            "what to cook when there are only basic eggs, bread, and milk left",
            "best way to make hot Hyderabadi filter coffee or Irani chai",
            "fixing a curry that became way too salty or too spicy",
            "making soft, round rotis for a beginner",
            "preparing traditional Telugu pulihora (tamarind rice) for a festival",
            "how to make a refreshing summer drink like majjiga (buttermilk) or lemon juice"
        ]
    },
    {
        "category": "Parenting & Childhood",
        "subtopics": [
            "dealing with kids fighting over toys or TV remote",
            "remembering nostalgic games played in childhood (like gilli-danda or hide-and-seek)",
            "how to limit children's screen time on mobile phones and tablets",
            "preparing a child for their first day at school",
            "helping kids with their homework without getting frustrated",
            "storytelling ideas for bedtime to put children to sleep",
            "handling a teenager's sudden mood swings or stubbornness",
            "planning a fun weekend family activity with kids",
            "dealing with sibling rivalry between an older and younger child",
            "funny childhood mischief memories and sharing them with friends"
        ]
    },
    {
        "category": "Relationships & Family",
        "subtopics": [
            "convincing parents about career choices or moving to a different city",
            "planning a surprise anniversary gift for parents",
            "handling dinner preparations when prospective bride/groom family is visiting",
            "dealing with annoying relatives asking about marriage or salary",
            "resolving a petty argument with a sibling or cousin",
            "organizing a cousin's get-together or wedding dance rehearsal",
            "sharing a deep conversation with father about future planning",
            "helping mother learn how to use smartphone apps (like WhatsApp or YouTube)",
            "feeling homesick and calling parents for comfort",
            "planning a family trip to native village or grandparents' home"
        ]
    },
    {
        "category": "Transportation & Commuting",
        "subtopics": [
            "getting stuck in Hyderabad peak-hour traffic and venting about it",
            "negotiating fares with local auto drivers or dealing with cancelled cabs",
            "crowd issues in local metro trains during office hours",
            "preparing a car or bike for a long road trip (servicing, tyres check)",
            "dealing with a sudden tyre puncture on the highway",
            "planning a train journey and booking tatkal tickets",
            "experience of taking a sleeper bus for overnight travel",
            "tips for safe driving on rainy days or foggy winter mornings",
            "learning how to ride a scooty or drive a car for the first time",
            "getting lost in a new area and asking locals for directions"
        ]
    },
    {
        "category": "Fitness & Routines",
        "subtopics": [
            "overcoming morning laziness to go for a run or walk",
            "experiencing severe muscle soreness after the first day at gym",
            "setting up a simple, realistic home workout routine without equipment",
            "maintaining a consistent sleep schedule and avoiding late-night scrolling",
            "eating healthy, tracking protein intake, and avoiding junk food",
            "practicing yoga or breathing exercises at home for mental peace",
            "feeling sleepy and unproductive during afternoon office hours",
            "planning a daily morning routine for a fresh start",
            "choosing the right running shoes or gym accessories",
            "struggling to drink enough water daily during hot summer days"
        ]
    },
    {
        "category": "Weather & Nature",
        "subtopics": [
            "enjoying a sudden heavy rain with hot tea and pakodas",
            "complaining about unbearable summer heat and planning cool drinks",
            "enjoying a cool breeze during an early morning walk in the park",
            "dealing with power cuts and water logging on rainy days",
            "feeling cozy and lazy during winter mornings",
            "experience of watching a beautiful sunset from a hilltop or lake view",
            "gardening tips for protecting balcony plants from extreme summer heat",
            "planning a nature walk or forest drive to escape concrete city life",
            "heavy monsoon traffic issues and waterlogged roads",
            "enjoying pleasant weather in the evening with outdoor seating"
        ]
    },
    {
        "category": "Movies & TV Shows",
        "subtopics": [
            "reviewing a recent blockbuster movie and explaining the climax twist",
            "asking friends for recommendations for a gripping thriller series to binge-watch",
            "the difference between theater experience vs watching on OTT platforms",
            "discussing favorite actors, their acting styles, and background music (BGM)",
            "making a music playlist for a road trip or gym workout",
            "debating about overhyped movies that did not meet expectations",
            "discussing nostalgic 90s Telugu movies and childhood favorites",
            "climax twists that blew your mind and sharing the shock",
            "binge-watching a series overnight and feeling tired next day",
            "discussing the trailer of a highly anticipated upcoming movie"
        ]
    },
    {
        "category": "Cricket & Sports",
        "subtopics": [
            "watching a highly intense India vs Pakistan cricket match",
            "getting tickets for an IPL match at Uppal stadium and planning the day",
            "playing a friendly weekend badminton match with colleagues",
            "discussing a legendary sports comeback or century by favorite player",
            "planning a local playground box cricket tournament with friends",
            "getting muscle cramps while playing football after a long time",
            "explaining cricket rules or terms (like offside, legbefore, powerplay) to a beginner",
            "visiting a sports club to learn table tennis or swimming",
            "debating about the best captain in Indian cricket history",
            "watching highlights of Olympics or athletic competitions"
        ]
    },
    {
        "category": "Gaming & YouTube",
        "subtopics": [
            "buying a new PS5 or gaming console and setting it up",
            "recommending multiplayer games to play with friends on weekend",
            "streaming gameplay on Twitch/YouTube and setting up capture cards",
            "watching viral comedy sketches or tech review videos on YouTube",
            "gaming room setup ideas (lighting, chair, monitor)",
            "feeling frustrated after losing a difficult level in a game repeatedly",
            "discussing open-world games like GTA or Witcher and their storylines",
            "watching cooking channels on YouTube and trying to copy the dish",
            "discussing nostalgia of playing old video games (like Mario or Contra)",
            "following a favorite gaming creator and their stream highlights"
        ]
    },
    {
        "category": "Memes & Social Media",
        "subtopics": [
            "scrolling Instagram reels for hours and feeling guilty",
            "sharing funny memes with friends on WhatsApp and laughing",
            "distractions from constant phone notifications while working",
            "discussing trending viral challenges or internet memes",
            "setting boundaries on social media usage and digital detox",
            "updating status or stories on WhatsApp and checking views",
            "dealing with spam messages or random group adds on social media",
            "finding helpful study or career tips from an Instagram page",
            "funny group chat dynamics between close friends",
            "reacting to cringy reels or viral dance videos"
        ]
    },
    {
        "category": "Trip Planning & Travel",
        "subtopics": [
            "planning a weekend getaway to Araku Hills or Horsley Hills",
            "booking hotel rooms online and checking reviews",
            "packing luggage and deciding what clothes to carry",
            "creating a detailed travel itinerary for a 3-day trip",
            "exploring local street food and shopping markets in a new city",
            "finding offbeat, peaceful spots away from main tourist crowds",
            "budgeting for a trip (transport, food, accommodation)",
            "packing a travel first-aid kit and emergency medicines",
            "hiring a local guide vs exploring on your own",
            "visiting historical temples or monuments and learning their history"
        ]
    },
    {
        "category": "Budgeting & Personal Finance",
        "subtopics": [
            "tracking monthly salary and planning fixed expenses (rent, bills)",
            "dealing with high credit card bills and learning how to avoid debt",
            "splitting grocery and electricity bills with flatmates",
            "setting up savings goals for a new laptop or a future trip",
            "opening a savings bank account or starting a fixed deposit",
            "UPI payment failures at local stores and alternative options",
            "tips for reducing unnecessary expenses on eating out or shopping",
            "managing personal finance as a beginner earning first salary",
            "investing small amounts in mutual funds or gold",
            "understanding emergency funds and why they are important"
        ]
    },
    {
        "category": "Customer Support & Products",
        "subtopics": [
            "raising a complaint about a delayed delivery package on Amazon/Flipkart",
            "comparing two mobile phones before buying (camera, battery, price)",
            "requesting a refund for a damaged item received online",
            "complaining about poor customer service at a local restaurant or showroom",
            "deciding between buying a laptop vs a tablet for study purposes",
            "writing a product review for a gadget that exceeded expectations",
            "dealing with warranty claims for a malfunctioning home appliance",
            "asking for recommendations for budget-friendly noise-cancelling headphones",
            "buying a second-hand item and verifying its condition",
            "dealing with electricity bill discrepancy and visiting the office"
        ]
    },
    {
        "category": "Home Maintenance & Chores",
        "subtopics": [
            "organizing a messy wardrobe or cleaning the bedroom on a weekend",
            "dealing with a leaking kitchen tap or plumbing issues",
            "shifting to a new flat and booking packers and movers",
            "repairing a washing machine or refrigerator that stopped working",
            "setting up flatmate rotation charts for washing dishes and sweeping",
            "cleaning the refrigerator and throwing out expired food items",
            "laundry day routines and folding ironed clothes",
            "fixing loose doors or creaky cupboards at home",
            "setting up room decoration with fairy lights and posters",
            "dealing with water scarcity or tank refill problems at apartment"
        ]
    },
    {
        "category": "Life Advice & Motivation",
        "subtopics": [
            "handling work stress and learning how to maintain peace of mind",
            "boosting self-confidence before a major presentation or speech",
            "overcoming loneliness when moving to a new city alone",
            "finding motivation to learn a new skill when feeling stuck",
            "how to say no to people without feeling guilty",
            "handling failures or setbacks in career with a positive mindset",
            "importance of maintaining a work-life balance",
            "dealing with toxic people or workplace politics calmly",
            "setting realistic long-term goals for personal growth",
            "finding happiness in small daily achievements"
        ]
    },
    {
        "category": "Youth & Casual Slang",
        "subtopics": [
            "planning a casual evening hangout at a local tea stall (tapri)",
            "gossiping about college or office updates with close friends",
            "discussing local street shopping discounts and bargaining tricks",
            "funny slang terms used in Hyderabad (like light le, baigan, sahi hai)",
            "ordering food online late at night with friends",
            "deciding what to wear for a casual party or gathering",
            "talking about the excitement of weekend plans on a boring Friday",
            "sharing embarrassing moments or funny mistakes with friends",
            "reminiscing about school day bunking experiences",
            "planning a surprise birthday celebration for a flatmate"
        ]
    },
    {
        "category": "Exams & Study Plans",
        "subtopics": [
            "preparing for semester exams with a vast syllabus and limited time",
            "organizing a group study session with friends at the library",
            "dealing with assignment deadlines and last-minute submissions",
            "handling exam result tension and anxious waiting",
            "tips for memorizing difficult formulas or historical dates",
            "creating a study timetable that balances different subjects",
            "preparing notes for key concepts before the exam day",
            "how to avoid sleepiness while studying late at night",
            "understanding grading systems and CGPA calculation",
            "discussing professor's strict evaluation or lecture style"
        ]
    },
    {
        "category": "Career Prep & Job Hunt",
        "subtopics": [
            "updating a resume to highlight recent projects and internships",
            "preparing for a mock interview or HR round questions",
            "searching for job openings on portals like LinkedIn or Naukri",
            "negotiating salary packages and discussing notice periods",
            "deciding between joining a startup vs a large corporate company",
            "dealing with job application rejections and keeping hope alive",
            "preparing for coding tests or aptitude rounds",
            "asking senior colleagues for referral in their companies",
            "attending a job fair or walk-in interviews",
            "career path dilemmas (e.g. data science vs web development)"
        ]
    },
    {
        "category": "Technical Explanations & AI",
        "subtopics": [
            "explaining how AI chatbots generate text to a non-tech friend",
            "understanding cloud storage and how Google Drive stores data",
            "explaining what is open-source software and why it is free",
            "understanding cryptocurrency and blockchain in simple terms",
            "explaining how the internet works (routers, IPs, servers)",
            "understanding the role of databases in keeping user records safe",
            "explaining what is caching and why websites load faster",
            "understanding virtual reality and augmented reality concepts",
            "explaining cyber security basics like passwords and phishing",
            "explaining why smart home devices respond to voice commands"
        ]
    },
    {
        "category": "Coding & Debugging",
        "subtopics": [
            "writing a Python script to filter a list of dictionaries by status",
            "debugging an indent error or syntax error in a Python program",
            "writing a basic SQL query to retrieve employee names and salaries",
            "resolving database connection failures or socket errors",
            "understanding git conflicts and how to merge branches",
            "writing a simple calculator program with basic operations",
            "handling API call failures and writing try-catch blocks",
            "understanding loops and writing a nested loop example",
            "debugging a memory leak or slow loading issue in a website",
            "writing unit tests for a simple python function"
        ]
    }
]

SYSTEM_PROMPT = """You are an expert creator of high-quality Telugu-English code-switched SFT datasets for aligning multilingual LLMs.

Your task is to generate realistic instruction-following conversations in natural Romanized Telugu-English ("Tenglish" / "Telglish") for general-purpose assistant use.

The goal is to teach the model:
- natural Telugu-led code switching
- realistic bilingual conversational flow
- correct Telugu grammatical structure
- natural English word insertion patterns
- stable colloquial speech patterns
- consistency across domains and tones

You must generate:
- one USER prompt
- one ASSISTANT response

Both must feel like real conversations between fluent bilingual Telugu speakers.

--------------------------------------------------
LANGUAGE STYLE RULES
--------------------------------------------------

The language MUST follow a Telugu-dominant Matrix Language Frame (MLF):

- Telugu grammar is the backbone
- Telugu word order must dominate (Subject-Object-Verb structure)
- English words should appear naturally where bilingual Telugu speakers genuinely use them

Examples:
- "trip plan chestha"
- "code run avvatledu"
- "database lo samasya undi"
- "meeting reschedule cheddama"

The language should sound like:
- natural spoken Telugu-English
- fluent urban bilingual communication
- casual modern Telugu speech
- effortless conversational flow

The language should NOT sound like:
- textbook Telugu
- formal translation Telugu
- English sentences with Telugu fillers
- corporate jargon-heavy bilingual speech

--------------------------------------------------
TELUGU-FIRST CODE SWITCHING POLICY
--------------------------------------------------

The dataset MUST optimize for maximum natural Telugu usage while preserving realistic bilingual speech.

Use Telugu words whenever bilingual Telugu speakers naturally use Telugu in conversation.

English should appear ONLY when:
- the English word is dominant in real usage
- the Telugu equivalent sounds forced, outdated, overly formal, or unnatural
- the topic is technical or internet-native
- the English term is commonly used even in Telugu speech

Common acceptable English usage:
- code
- API
- laptop
- mobile
- server
- database
- bug
- app
- startup
- resume
- meeting

Avoid unnecessary English insertion.

Preferred Telugu usage examples:

BAD:
- "Actually naaku aa movie antha nachaledu"

GOOD:
- "Nijaniki naaku aa cinema antha nachaledu"

BAD:
- "Basically vaadu chaala attitude chupisthunnadu"

GOOD:
- "Asalu vaadu chaala pogaruga untunnadu"

BAD:
- "Issue enti ante fridge cool avvatledu"

GOOD:
- "Samasya enti ante fridge challaga avvatledu"

BAD:
- "Proper ga nidra povatledu"

GOOD:
- "Sarigga nidra povatledu"

BAD:
- "Shopping complete chesi vachha"

GOOD:
- "Konukkoni vachha"

BAD:
- "Dinner order cheddama"

GOOD:
- "Food bayata nunchi teppinchukundama"

The generated language should feel:
- Telugu-first
- colloquial
- modern
- natural
- emotionally expressive
- easy to read aloud naturally

The model must NOT learn that random English insertion equals fluency.

--------------------------------------------------
STRICTLY AVOID
--------------------------------------------------

1. Pure English sentences

BAD:
- "You should update the file before running the script."

GOOD:
- "Script run cheyyadaniki mundu file update cheyyali"

2. Excessive English stuffing

BAD:
- "Basically nee workflow optimize cheyyadaniki proper structure implement cheyyali"

GOOD:
- "Nee pani vidhanam inka baaga undela chudali"

3. Literal formal Telugu translation style

BAD:
- "Nenu meeku sahayam chesthanu"

GOOD:
- "Nenu help chestha"

4. Artificial over-mixing

BAD:
- Every sentence unnecessarily packed with English words

5. Repetitive software-engineer bilingual speech

BAD:
- Constant references to productivity, startups, debugging, workflows, optimization, implementation, etc.

6. Telugu Unicode characters

ONLY Roman script allowed.

--------------------------------------------------
USER PROMPT RULES
--------------------------------------------------

The USER prompt:
- must be written in natural Romanized Telugu-English
- must feel spontaneous and realistic
- must resemble actual assistant usage
- must NOT sound benchmark-generated or synthetic
- may be short or long
- may contain emotion, confusion, frustration, excitement, curiosity, or casual speech

Prompt styles can include:
- casual questions
- multi-part requests
- troubleshooting
- emotional conversations
- advice seeking
- planning
- comparisons
- storytelling
- brainstorming

Avoid repetitive templates.

--------------------------------------------------
ASSISTANT RESPONSE RULES
--------------------------------------------------

The ASSISTANT response must:
- directly answer the request
- be highly useful and complete
- maintain consistent Telugu-first bilingual style
- preserve Telugu grammatical flow
- sound natural and human
- avoid robotic assistant phrasing
- NEVER use generic placeholders like `[Your Name]`, `[Manager's Name]`, or variables like `₹X`, `₹Y`. Instead, use realistic, concrete details (e.g. `Ramesh`, `Rahul`, `₹15,000`, `June 20th`).
- NEVER include disclaimers, meta-commentary, or excuses (e.g. do NOT write: "naku exact details/dates teliyavu kani..." or "nenu AI assistant kabatti..."). Just provide the best possible information directly.
- NEVER ask follow-up questions or prompt the user at the end of the response (e.g. do NOT write: "idi correct ga run ayyinda?", "neeku emaina doubt unda?", "ee advice follow avvu all the best"). Make the response a complete, helpful, final answer.

When useful:
- use markdown
- use bullet points
- use numbered steps
- use tables
- use code blocks

For coding tasks:
- code itself should remain standard programming syntax
- explanations outside code must remain in Telugu-first Tenglish

For emotional or conversational prompts:
- sound warm and realistic
- avoid exaggerated politeness
- avoid corporate tone

--------------------------------------------------
STYLE TARGET
--------------------------------------------------

The target style is:
- fluent bilingual Telugu speech
- Telugu-first communication
- colloquial modern language
- natural internet/chat speech
- emotionally expressive
- grammatically stable
- concise but natural

The dataset should help align the model toward:
- realistic Telugu-English code switching
- stable bilingual grammar
- natural conversational rhythm
- strong Telugu linguistic grounding
- general-purpose assistant behavior

--------------------------------------------------
FEW-SHOT EXAMPLES (TARGET STYLE)
--------------------------------------------------

Example 1 (Cooking / Household):
{
  "prompt": "macha, aakaliga undi ra, fridge lo em ledu. Instant ga 5 mins lo ayipoye edaina simple recipe cheppava?",
  "response": "Arey, fridge lo em lekapothe simple ga bread omlette chesko ra. Just bread slices, eggs, konchem karam, uppu unte chalu. Pan meeda oil vesi bread kaalchesi, egg beat chesi omlette la vesei. Less than 5 mins lo ready aipotundi, koddiga sauce eskoni tineseyyi."
}

Example 2 (Life Advice / Emotional):
{
  "prompt": "manager andari mundu project presentation thappulu cheppadu mama. Chaala embarrassing ga anipinchindi, motivation pothundi asalu.",
  "response": "Ayyo, lite teesko macha. Office lo ilantivi common. Manager ki personal ga nee meeda em kopam undadu, just work-related antaru anthe. Ee roju mistake nunchi nerchukoni next time presentation inka perfect ga chey. Job lo ups and downs untayi, tension padaku. Evvala evening room ki ra, chill avdam."
}

Example 3 (Tech / Coding):
{
  "prompt": "macha python script run chesthunte indent error ani vasthundi ra, asalu code ekkada check cheyalo teliyatledu.",
  "response": "Python lo indent error ante spacings sarigga levanamata. Editor lo lines alignment check chey, VS Code vaaduthunte spaces tab check checkskondi. If statement or loop rasinappudu correct ga 4 spaces indentation unda ledho chusko, simple ga solve aipotundi."
}

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Return ONLY valid JSON matching this schema:
{
  "prompt": "...",
  "response": "..."
}

No explanations.
No markdown fences.
No extra commentary.
"""

def has_telugu_script(text: str) -> bool:
    # Telugu Unicode Range check
    telugu_pattern = re.compile(r"[\u0c00-\u0c7f]")
    return bool(telugu_pattern.search(text))

def classify_prompts_batch(prompts: List[str]) -> List[str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    categories_enum = [c["category"] for c in CATEGORIES]
    prompt_list_str = "\n".join([f"{idx+1}. {p}" for idx, p in enumerate(prompts)])
    
    user_instruction = f"""You are a classification assistant. Classify each of the following Telglish prompts into exactly one of these 20 categories:
{", ".join(categories_enum)}

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
                            "enum": categories_enum
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
    return [categories_enum[0]] * len(prompts)

def classify_all_prompts(prompts: List[str]) -> List[str]:
    print(f"Classifying {len(prompts)} existing prompts...", flush=True)
    all_categories = []
    batch_size = 50
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        print(f"  Classifying batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1} (size {len(batch)})...", flush=True)
        batch_cats = classify_prompts_batch(batch)
        all_categories.extend(batch_cats)
        time.sleep(1.0)
    return all_categories

GREETING_STYLES = [
    "Start the prompt directly with the query/statement, with absolutely no greeting/slang prefixes (e.g., do NOT start with 'Macha', 'Bro', 'Arey', 'Mama', 'Oi', 'Hey', etc.). Just start with the question/request directly.",
    "Use a casual friend greeting like 'Bro' or 'Hey Bro' to start.",
    "Use a slang greeting like 'Mama' or 'Arey' to start.",
    "Use a greeting like 'Macha' or 'Yaar' to start.",
    "Start with a general greeting like 'Oi' or 'Hey'."
]

def generate_one_pair(category: str, context: str, existing_prompts: List[str] = [], recent_global_prompts: List[str] = [], greeting_style: str = "") -> Dict[str, str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    avoid_clause = ""
    if existing_prompts or recent_global_prompts:
        avoid_clause = "\n\nCRITICAL: To ensure high topic diversity and avoid repetition, you MUST NOT reuse the scenarios, questions, locations, or specific topics of these prompts:\n"
        unique_avoid = []
        # Add category-specific prompts (up to 15)
        for p in existing_prompts[-15:]:
            if p not in unique_avoid:
                unique_avoid.append(p)
        # Add recent global prompts (up to 15)
        for p in recent_global_prompts[-15:]:
            if p not in unique_avoid:
                unique_avoid.append(p)
                
        for idx, p in enumerate(unique_avoid, 1):
            avoid_clause += f"- {p}\n"
        avoid_clause += "\nMake your scenario, entities, and locations completely different from the ones listed above. For example, if travel/trips are mentioned above, use a completely different destination. If family/kids are mentioned, use a different dynamic. If exams/study are mentioned, use a different context.\n"
        
    greeting_clause = ""
    if greeting_style:
        greeting_clause = f"\n6. GREETING STYLE RULE: {greeting_style}"
        
    user_instruction = f"""Generate exactly 1 unique, high-quality Telglish instruction-response pair for the category: "{category}".
Use these context concepts for inspiration: {context}.
{avoid_clause}
Rules:
1. The USER prompt must be in natural, conversational Telglish.
2. The ASSISTANT response must directly answer the request, follow the matrix grammar rules, and sound like a natural Hyderabad friend responding in colloquial Telglish.
3. STRICTLY AVOID textbook, robotic, or corporate coaching style.
4. Do NOT use generic English list headers with Telugu suffixes (e.g. "1. Budget Categories Set cheyyandi" is BAD; "1. Mee budget categories ready chesukondi" is GOOD).
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
                print(f"Empty results generated. Retrying... (Attempt {attempt})", flush=True)
                continue
                
            if has_telugu_script(p) or has_telugu_script(r):
                print(f"Telugu script leakage detected in generation. Retrying... (Attempt {attempt})", flush=True)
                continue
                
            # Basic validation: ensure the prompt is not in pure English
            telugish_indicators = ["nenu", "nuvvu", "memu", "meeru", "ledu", "avunu", "lo", "ki", "ra", "ga", "enti", "enduku", "ela", "ekkada", "sare", "alage", "vachha", "vacha", "unda", "undhi", "undi", "chesa", "chesya", "chey", "vaddu", "vaddhu", "kuda", "kooda", "chala", "chaala", "kani", "kaani", "leda", "ante", "cheppandi", "cheppu"]
            p_lower = p.lower()
            r_lower = r.lower()
            
            has_p_telugish = any(word in p_lower.split() or f" {word} " in f" {p_lower} " or p_lower.endswith(f" {word}") or p_lower.startswith(f"{word} ") for word in telugish_indicators)
            has_r_telugish = any(word in r_lower.split() or f" {word} " in f" {r_lower} " or r_lower.endswith(f" {word}") or r_lower.startswith(f"{word} ") for word in telugish_indicators)
            
            if not has_p_telugish:
                english_only_indicators = ["you", "should", "the", "before", "running", "how", "what", "where", "why", "who", "which", "is", "a", "an"]
                is_p_pure_english = all(word in english_only_indicators for word in p_lower.split() if len(word) > 2) and len(p_lower.split()) > 3
                if is_p_pure_english:
                    print(f"Prompt '{p}' flagged as pure English. Retrying... (Attempt {attempt})", flush=True)
                    continue

            return {"prompt": p, "response": r}
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}", flush=True)
            time.sleep(attempt * 2)
            
    return {}

def main():
    parser = argparse.ArgumentParser(description="LIMA-style Telglish SFT Dataset Generator")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3], help="Rollout phase: 1 (5 samples), 2 (40 samples), 3 (960 samples)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the generated JSON dataset")
    args = parser.parse_args()

    print(f"=== STARTING LIMA TELGLISH GENERATOR (PHASE {args.phase}) ===", flush=True)

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
            print(f"Warning: could not parse existing file {args.output}: {e}. Starting fresh.", flush=True)

    if all_data:
        # Load or create metadata
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata_data = json.load(f)
            except Exception as e:
                print(f"Warning: could not load metadata: {e}. Re-classifying.", flush=True)
                
        if len(metadata_data) != len(all_data):
            print(f"Metadata mismatch or missing. Re-classifying existing prompts...", flush=True)
            prompts_to_classify = [item["prompt"] for item in all_data]
            classified_categories = classify_all_prompts(prompts_to_classify)
            metadata_data = []
            for item, cat in zip(all_data, classified_categories):
                metadata_data.append({
                    "prompt": item["prompt"],
                    "response": item["response"],
                    "category": cat
                })
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata_data, f, indent=2, ensure_ascii=False)
            print(f"Saved re-classified metadata to {metadata_path}", flush=True)

    # 1. Determine targets based on phase
    category_names = [c["category"] for c in CATEGORIES]
    target_counts = {cat: 0 for cat in category_names}
    
    if args.phase == 1:
        selected_cats = ["Cooking & Recipes", "Movies & TV Shows", "Budgeting & Personal Finance", "Life Advice & Motivation", "Coding & Debugging"]
        for cat in selected_cats:
            target_counts[cat] = 1
        total_expected = 5
    elif args.phase == 2:
        for cat in category_names:
            target_counts[cat] = 2
        total_expected = 40
    else: # Phase 3
        for cat in category_names:
            target_counts[cat] = 50
        total_expected = 1000

    # Count current categories in metadata
    current_counts = {cat: 0 for cat in category_names}
    for item in metadata_data:
        cat_name = item.get("category")
        if cat_name in current_counts:
            current_counts[cat_name] += 1
            
    # Print current status
    print("\nCurrent category distribution in dataset:", flush=True)
    for cat in category_names:
        print(f"  - {cat}: {current_counts[cat]} / {target_counts[cat]}", flush=True)
        
    # Build remaining jobs
    remaining_jobs = []
    for cat_info in CATEGORIES:
        cat_name = cat_info["category"]
        needed = target_counts[cat_name] - current_counts[cat_name]
        for _ in range(max(0, needed)):
            remaining_jobs.append(cat_info)
            
    # Interleave remaining jobs by category to avoid generating same category consecutively
    jobs_by_cat = {cat["category"]: [] for cat in CATEGORIES}
    for job in remaining_jobs:
        jobs_by_cat[job["category"]].append(job)
        
    interleaved_jobs = []
    while True:
        added_any = False
        for cat_info in CATEGORIES:
            cat_name = cat_info["category"]
            if jobs_by_cat[cat_name]:
                interleaved_jobs.append(jobs_by_cat[cat_name].pop(0))
                added_any = True
        if not added_any:
            break
            
    print(f"\nRemaining jobs to generate: {len(interleaved_jobs)}", flush=True)
    if not interleaved_jobs:
        print("All targets met. Generation complete!", flush=True)
        return

    # Generation loop
    start_index = len(all_data)
    total_to_generate = len(interleaved_jobs)
    
    for step_idx, job in enumerate(interleaved_jobs):
        current_idx = start_index + step_idx
        cat_name = job["category"]
        subtopics = job.get("subtopics", [])
        context_desc = random.choice(subtopics) if subtopics else job.get("context", "")
        
        # Pick greeting style randomly
        greeting_style = random.choice(GREETING_STYLES)
        
        print(f"\n[{current_idx+1}/{total_expected}] (Remaining Job {step_idx+1}/{total_to_generate}) Generating for Category: '{cat_name}'", flush=True)
        
        # Get existing prompts for this category from the metadata
        existing_prompts = [item["prompt"] for item in metadata_data if item["category"] == cat_name]
        
        # Get recent global prompts (last 15 items across all categories) to avoid global patterns repeating
        recent_global_prompts = [item["prompt"] for item in metadata_data[-15:]]
        
        pair = generate_one_pair(cat_name, context_desc, existing_prompts, recent_global_prompts, greeting_style)
        if pair:
            all_data.append(pair)
            
            metadata_item = {
                "prompt": pair["prompt"],
                "response": pair["response"],
                "category": cat_name
            }
            metadata_data.append(metadata_item)
            
            # Ensure parent directories exist
            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                
            # Write main dataset incrementally
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)
                
            # Write metadata dataset incrementally
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata_data, f, indent=2, ensure_ascii=False)
                
            print(f"Successfully saved. Prompt snippet: '{pair['prompt'][:60]}...'", flush=True)
        else:
            print(f"CRITICAL: Generation failed for Category '{cat_name}'. Skipping to keep pipeline alive.", flush=True)
            
        time.sleep(1.2) # Rate limit gap

    print(f"\n=== GENERATION COMPLETED ===", flush=True)
    print(f"Total entries: {len(all_data)} saved to {args.output}", flush=True)

if __name__ == "__main__":
    main()
