import requests
import json
import time

MODEL = "tiny-aya-fire"
URL = "http://127.0.0.1:1234/v1/chat/completions"

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

# Load prompts from the baseline result file to guarantee they are identical
try:
    with open("tenglish_eval_results.json", "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    PROMPTS = [item["prompt"] for item in baseline_data]
    print(f"Loaded {len(PROMPTS)} prompts from tenglish_eval_results.json")
except Exception as e:
    print(f"Could not load prompts from tenglish_eval_results.json: {e}")
    # Fallback to the hardcoded prompts from initeval.py
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
        "ee joke naaku late ga ardam ayyindi",
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

results = []

for i, prompt in enumerate(PROMPTS, 1):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 500,
        "stream": False
    }

    try:
        response = requests.post(
            URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        output = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error querying {prompt}: {e}")
        output = f"Error: {e}"

    results.append({
        "prompt": prompt,
        "response": output
    })

    print(f"\n[{i}/{len(PROMPTS)}] PROMPT: {prompt}")
    print(f"RESPONSE: {output}")

    time.sleep(0.1)

with open("aya_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nSaved Aya Fire results to aya_eval_results.json")
