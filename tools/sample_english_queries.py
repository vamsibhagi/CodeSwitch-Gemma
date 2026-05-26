#!/usr/bin/env python3
import os
import json
import random
import argparse
from datasets import load_dataset
from tqdm import tqdm

def is_ascii(s: str) -> bool:
    try:
        s.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False

def main():
    parser = argparse.ArgumentParser(description="Sample diverse English queries from UltraChat 200k")
    parser.add_argument("--num-queries", type=int, default=2000, help="Number of queries to sample")
    parser.add_argument("--output", type=str, default="data/sampled_english_prompts.json", help="Path to save the sampled queries JSON")
    args = parser.parse_args()

    print("=== LOADING ULTRACHAT 200K DATASET ===", flush=True)
    # Load the train_sft split of ultrachat_200k
    try:
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    except Exception as e:
        print(f"Error loading dataset: {e}", flush=True)
        return

    print(f"Loaded dataset containing {len(dataset)} entries.", flush=True)
    print("Extracting, filtering, and deduplicating queries...", flush=True)

    unique_queries = set()
    
    # Iterate with a progress bar
    for row in tqdm(dataset, desc="Processing rows"):
        # Extract prompt
        prompt = row.get("prompt")
        
        # Fallback to messages content if prompt field is empty
        if not prompt and "messages" in row and row["messages"]:
            first_msg = row["messages"][0]
            if first_msg.get("role") == "user":
                prompt = first_msg.get("content")

        if not prompt:
            continue

        prompt = prompt.strip()

        # Filtering rules
        # 1. Length constraint (between 15 and 600 characters)
        if len(prompt) < 15 or len(prompt) > 600:
            continue

        # 2. Must be clean ASCII (English)
        if not is_ascii(prompt):
            continue

        # 3. Basic cleanup
        # Skip if it is empty, placeholder, or too short in word count (e.g., < 3 words)
        words = prompt.split()
        if len(words) < 3:
            continue

        unique_queries.add(prompt)

    queries_list = list(unique_queries)
    print(f"Extracted {len(queries_list)} unique, clean English queries.", flush=True)

    if len(queries_list) < args.num_queries:
        print(f"Warning: Only found {len(queries_list)} queries, which is less than requested {args.num_queries}.", flush=True)
        sampled_queries = queries_list
    else:
        # Shuffle to ensure maximum diversity in sampling
        random.seed(42)
        random.shuffle(queries_list)
        sampled_queries = queries_list[:args.num_queries]

    print(f"Sampled {len(sampled_queries)} queries.", flush=True)

    # Ensure parent directories exist
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Save to output file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sampled_queries, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved sampled queries to {args.output}", flush=True)

if __name__ == "__main__":
    main()
