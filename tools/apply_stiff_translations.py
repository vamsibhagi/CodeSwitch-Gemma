import json
import os
import re

def parse_review_file(path):
    print(f"Parsing review file: {path}")
    replacements = []
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return replacements
        
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('|'):
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 6:
                continue
            # Skip header and separator lines
            if parts[1] == 'Item Index' or parts[1].startswith(':---'):
                continue
            
            # Clean formatting asterisks
            stiff_term = parts[2].replace('**', '').strip()
            replacement = parts[4].replace('**', '').strip()
            
            if stiff_term and replacement:
                replacements.append((stiff_term, replacement))
                
    # Deduplicate and sort by length descending to avoid replacing substrings first
    replacements = list(set(replacements))
    replacements.sort(key=lambda x: len(x[0]), reverse=True)
    return replacements

def apply_replacements(path, replacements):
    print(f"Applying replacements to: {path}")
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    total_modifications = 0
    for idx, item in enumerate(data):
        changed = False
        for key in ['prompt', 'response']:
            if key in item and isinstance(item[key], str):
                orig = item[key]
                new_text = orig
                for stiff, repl in replacements:
                    # Case-sensitive replace first
                    new_text = new_text.replace(stiff, repl)
                    # Capitalized replace (if the term is at start of sentence)
                    stiff_cap = stiff[0].upper() + stiff[1:] if len(stiff) > 0 else stiff
                    repl_cap = repl[0].upper() + repl[1:] if len(repl) > 0 else repl
                    new_text = new_text.replace(stiff_cap, repl_cap)
                    
                if new_text != orig:
                    item[key] = new_text
                    changed = True
        if changed:
            total_modifications += 1
            
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"Finished {path}. Modified {total_modifications} out of {len(data)} items.")

def main():
    review_path = 'data/stiff_translations_review.md'
    replacements = parse_review_file(review_path)
    print(f"Extracted {len(replacements)} replacements from review file.")
    
    # Print a few to verify
    for stiff, repl in replacements[:10]:
         print(f"  '{stiff}' -> '{repl}'")
         
    apply_replacements('data/train_sft_lima_200.json', replacements)
    apply_replacements('data/train_sft_lima_200_metadata.json', replacements)

if __name__ == '__main__':
    main()
