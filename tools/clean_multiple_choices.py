import os

mapping = {
    "Bilding Lopa": "**foundation lenappudu**",
    "vyakthi": "**manishi**",
    "pariseelinchandi": "**chudandi**",
    "udhaharinchadam": "**cheppadam**",
    "alochanalu theesukondi": "**alochinchandi**",
    "cheruvaina panule": "**physical activity ga count avvuthaayi**",
    "valasa vidhanala pai prabhavam": "**valasa patterns pai prabhavam**",
    "Ushnograthalu": "**vedi**",
    "pakshula gurthinchadam": "**gudlu pettadam mariyu pillalani penchadam**",
    "aahara labhyatha": "**food dorakadam**",
    "chaitanyam penchadam": "**teliyajeyadam**",
    "Nijam kanna": "**Nijamgaa**",
    "carbonika chettani": "**thadicha chettani**",
    "puttu lingam": "**biological sex**",
    "Samajika Tiraskarana": "**Social rejection**",
    "vivekham chupistharu": "**pakshapaatham chupistharu**",
    "hinsalu": "**daadulu**",
    "pariharinchali": "**tagginchali**"
}

def clean_review_file(path):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return
        
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith('|'):
                lines.append(line)
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 6:
                lines.append(line)
                continue
            if parts[1] == 'Item Index' or parts[1].startswith(':---'):
                lines.append(line)
                continue
                
            stiff_term = parts[2].replace('**', '').strip()
            if stiff_term in mapping:
                # Update suggested replacement (index 4 in parts because parts[0] is empty before the first '|')
                old_val = parts[4]
                new_val = mapping[stiff_term]
                print(f"Replacing '{stiff_term}': '{old_val}' -> '{new_val}'")
                parts[4] = new_val
                # Reconstruct the line
                new_line = " | ".join(parts).strip()
                # Ensure the leading and trailing '|' are intact
                if not new_line.startswith('|'):
                    new_line = '| ' + new_line
                if not new_line.endswith('|'):
                    new_line = new_line + ' |'
                lines.append(new_line + '\n')
            else:
                lines.append(line)
                
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("Done cleaning review file.")

if __name__ == '__main__':
    clean_review_file('data/stiff_translations_review.md')
