import json
import re
import sys
import os
import requests
import time

# Load API key from .env file
def load_api_key():
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if line.startswith('GEMINI_API_KEY=') or line.startswith('GOOGLE_API_KEY='):
                    return line.split('=', 1)[1].strip()
    return os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')

GEMINI_API_KEY = load_api_key()
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found.")
    sys.exit(1)

def call_gemini(prompt_text):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt_text}]
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    for attempt in range(5):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                res_json = response.json()
                text = res_json['candidates'][0]['content']['parts'][0]['text']
                return text
            elif response.status_code == 429:
                print("Rate limited, backing off...")
                time.sleep(2 ** attempt + 1)
            else:
                print(f"Error {response.status_code}: {response.text}")
                time.sleep(2)
        except Exception as e:
            print(f"Request exception: {e}")
            time.sleep(2)
    return None

def main():
    dataset_path = 'data/train_sft_lima_200.json'
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items to scan for stiff translations.")
    
    batch_size = 5
    stiff_terms = []
    
    # We will scan the first 25 items to get a highly representative list of stiff translations across all categories.
    # Processing 25 items in 5 batches of 5 is safe and avoids read timeout issues.
    max_items = 25
    for i in range(0, min(len(data), max_items), batch_size):
        batch = data[i:i+batch_size]
        print(f"Scanning items {i} to {i+len(batch)}...")
        
        # Prepare text for Gemini
        batch_text = []
        for idx, item in enumerate(batch):
            item_idx = i + idx
            batch_text.append(f"--- Item {item_idx} ---\nPrompt: {item['prompt']}\nResponse: {item['response']}\n")
            
        system_prompt = """
You are an expert Telugu-English (Telglish) linguist.
Analyze the following Telglish text samples. Identify any Telugu words or phrases that sound stiff, overly formal, literal, or "translationese" (e.g. translated word-for-word from English where a native speaker would just use the English word or a simpler Telugu word in conversation).

Examples of stiff/bad translations:
- 'online sadanidhi' for online presence (should be 'online presence' or 'online lo undadam')
- 'prabhuthva vethireka rangam' for private sector (should be 'private sector' or 'private rangam')
- 'avadhi theedhulu' for expiration dates (should be 'expiry dates' or 'gaduwu dates')
- 'samkshemam' for gated community (should be 'gated community')
- 'nisedha' or similar literal words.

For each stiff term found, output a JSON object with:
- "item_index": the index of the item
- "stiff_term": the exact stiff Telugu word/phrase used
- "context": the short snippet where it appears
- "suggested_replacement": the natural conversational Telglish replacement
- "reason": why it sounds stiff and why the replacement is better

Output the result strictly as a JSON array of objects:
[
  {
    "item_index": 12,
    "stiff_term": "...",
    "context": "...",
    "suggested_replacement": "...",
    "reason": "..."
  }
]
"""
        full_prompt = system_prompt + "\n\n" + "\n".join(batch_text)
        result = call_gemini(full_prompt)
        
        if result:
            try:
                # Clean markdown backticks if present in JSON mime response
                clean_result = result.strip()
                if clean_result.startswith("```json"):
                    clean_result = clean_result[7:]
                if clean_result.endswith("```"):
                    clean_result = clean_result[:-3]
                clean_result = clean_result.strip()
                
                batch_terms = json.loads(clean_result)
                stiff_terms.extend(batch_terms)
                print(f"Found {len(batch_terms)} stiff terms in this batch.")
            except Exception as e:
                print(f"Error parsing JSON response: {e}")
                print(result[:500])
        else:
            print("Failed to get response for this batch.")
            
        time.sleep(2) # rate limit politeness

    # Save findings to markdown file
    output_path = 'data/stiff_translations_review.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Stiff Translations Review\n\n")
        f.write("Below is the list of stiff, formal, or literal Telugu translations identified in the dataset, along with their suggested natural replacements. You can review this list to decide which replacements should be applied.\n\n")
        f.write("| Item Index | Stiff Term | Context | Suggested Replacement | Reason |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        # Deduplicate stiff terms to make the table clean
        seen = set()
        deduped_terms = []
        for term in stiff_terms:
            key = (term.get('stiff_term', '').lower(), term.get('suggested_replacement', '').lower())
            if key not in seen:
                seen.add(key)
                deduped_terms.append(term)
                
        for term in sorted(deduped_terms, key=lambda x: x.get('item_index', 0)):
            idx = term.get('item_index', '')
            stiff = term.get('stiff_term', '').replace('|', '\\|')
            ctx = term.get('context', '').replace('|', '\\|').replace('\n', ' ')
            repl = term.get('suggested_replacement', '').replace('|', '\\|')
            reason = term.get('reason', '').replace('|', '\\|').replace('\n', ' ')
            f.write(f"| {idx} | **{stiff}** | *\"{ctx}\"* | **{repl}** | {reason} |\n")
            
    print(f"Saved {len(deduped_terms)} unique stiff translations to {output_path}.")

if __name__ == '__main__':
    main()
