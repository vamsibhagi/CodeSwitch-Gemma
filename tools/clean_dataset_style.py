import json
import re
import sys
import os

# Abbreviations to keep in parentheses
keep_abbrevs = {
    'SGBs', 'DRS', 'FIR', 'VPN', 'AI', 'SGBs', 'GSAs', 'RFID', 'ICC', 'SWOT',
    'LGBTQ', 'LGBTQ+', 'UBI', 'CSR', 'PPP', 'EV', 'CNC', 'CAD', 'BIS', 'FDR',
    'DRS', 'ETFs', 'ETF', 'API', 'SEO', 'URL', 'IP', 'ML', 'AI-powered',
    'RFID tags', 'RFID taglu', 'RFID tag', 'GPS', 'SMS', 'OTP', 'PIN', 'CVV',
    'USB', 'HDMI', 'OS', 'RAM', 'ROM', 'CPU', 'GPU', 'GB', 'TB', 'MB', 'KB',
    'Q&A', 'BTS', 'GST', 'PAN', 'Aadhaar', 'EPFO', 'PPF', 'NPS', 'FD', 'RD',
    'EMI', 'KYC', 'UI', 'UX', 'HTML', 'CSS', 'JS', 'URL', 'PDF', 'doc', 'docs'
}

def should_keep_parenthesis(inside):
    inside_clean = inside.strip()
    # 1. Keep if it has numbers
    if re.search(r'\d', inside_clean):
        return True
    # 2. Keep if it contains specific markers
    if any(marker in inside_clean.lower() for marker in ['e.g.', 'ud:', 'ex:', 'i.e.', 'example', 'such as']):
        return True
    # 3. Keep if it is in our list of abbreviations
    if inside_clean in keep_abbrevs or inside_clean.upper() in keep_abbrevs:
        return True
    # 4. Keep if it's a short English abbreviation/word commonly kept
    if len(inside_clean) <= 4 and inside_clean.isupper():
        return True
    return False

def clean_text(text):
    if not isinstance(text, str):
        return text
    # Find all parentheses and their contents
    matches = list(re.finditer(r'\s*\(([^)]+)\)', text))
    # Process from right to left to avoid index shifting
    for match in reversed(matches):
        inside = match.group(1)
        if not should_keep_parenthesis(inside):
            start, end = match.span()
            text = text[:start] + text[end:]
            
    # Remove single quotes around words/phrases
    text = re.sub(r"\'([^']+)\'", r'\1', text)
    # Clean up double spaces
    text = re.sub(r'  +', ' ', text)
    return text

def clean_file(path):
    print(f"Cleaning file: {path}...")
    if not os.path.exists(path):
        print(f"Error: {path} does not exist.")
        return
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    modified_count = 0
    for idx, item in enumerate(data):
        changed = False
        for key in ['prompt', 'response']:
            if key in item and isinstance(item[key], str):
                orig = item[key]
                cleaned = clean_text(orig)
                if orig != cleaned:
                    item[key] = cleaned
                    changed = True
        if changed:
            modified_count += 1
            
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"Done cleaning {path}. Modified {modified_count} out of {len(data)} items.")

def main():
    clean_file('data/train_sft_lima_200.json')
    clean_file('data/train_sft_lima_200_metadata.json')

if __name__ == '__main__':
    main()
