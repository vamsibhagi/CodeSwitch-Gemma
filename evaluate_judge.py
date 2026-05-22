#!/usr/bin/env python3
import os
import re
import json
import time
import requests
from typing import Dict, Any, Optional, Tuple

# Helper function to read simple .env files if present in the workspace
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

# Load local environment variables
load_env_file()

# Identify the API keys available
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_judge_provider() -> Tuple[str, str, str]:
    """
    Returns (provider_name, model_name, api_key) based on available environment variables.
    """
    if GEMINI_API_KEY:
        # Defaulting to gemini-2.5-flash
        return "gemini", "gemini-2.5-flash", GEMINI_API_KEY
    elif ANTHROPIC_API_KEY:
        return "anthropic", "claude-3-5-sonnet-20241022", ANTHROPIC_API_KEY
    elif OPENAI_API_KEY:
        return "openai", "gpt-4o", OPENAI_API_KEY
    else:
        raise ValueError(
            "Error: No API key found. Please export GEMINI_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY."
        )

# Pre-check filter: flag empty responses or native script leakage (Telugu script ranges from \u0c00 to \u0c7f)
def run_pre_check(response: str) -> Optional[Dict[str, Any]]:
    if not response or not response.strip():
        return {
            "grammatical_integrity_analysis": "Pre-check Failure: Response is empty or whitespace only.",
            "grammatical_integrity_score": 1,
            "codeswitch_naturalness_analysis": "Pre-check Failure: Response is empty or whitespace only.",
            "codeswitch_naturalness_score": 1,
            "precheck_flagged": True,
            "flag_reason": "Empty Response"
        }
    
    # Telugu Unicode Range check: \u0c00 to \u0c7f
    telugu_script_pattern = re.compile(r"[\u0c00-\u0c7f]")
    if telugu_script_pattern.search(response):
        return {
            "grammatical_integrity_analysis": "Pre-check Failure: Response contains native Telugu script characters.",
            "grammatical_integrity_score": 1,
            "codeswitch_naturalness_analysis": "Pre-check Failure: Response contains native Telugu script characters.",
            "codeswitch_naturalness_score": 1,
            "precheck_flagged": True,
            "flag_reason": "Telugu Script Leakage"
        }
    
    return None

def load_rubric(filepath: str = "eval_rubrics.md") -> str:
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    else:
        # Fallback to string literal if not found, but it should be present in the workspace
        return "Rubric file eval_rubrics.md not found."

# Calling Gemini API
def call_gemini(model: str, api_key: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    # Instruct Gemini to output JSON matching our schema
    payload = {
        "contents": [{
            "parts": [{"text": f"{system_prompt}\n\nUser Input to evaluate:\n{user_prompt}"}]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "thinkingConfig": {
                "thinkingBudget": 0
            },
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "grammatical_integrity_analysis": {"type": "STRING"},
                    "grammatical_integrity_score": {"type": "INTEGER"},
                    "codeswitch_naturalness_analysis": {"type": "STRING"},
                    "codeswitch_naturalness_score": {"type": "INTEGER"}
                },
                "required": [
                    "grammatical_integrity_analysis",
                    "grammatical_integrity_score",
                    "codeswitch_naturalness_analysis",
                    "codeswitch_naturalness_score"
                ]
            }
        }
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    res_data = response.json()
    
    try:
        text_content = res_data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text_content)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to parse Gemini output: {e}. Raw: {res_data}")

# Calling OpenAI API
def call_openai(model: str, api_key: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "eval_report",
                "schema": {
                    "type": "object",
                    "properties": {
                        "grammatical_integrity_analysis": {"type": "string"},
                        "grammatical_integrity_score": {"type": "integer"},
                        "codeswitch_naturalness_analysis": {"type": "string"},
                        "codeswitch_naturalness_score": {"type": "integer"}
                    },
                    "required": [
                        "grammatical_integrity_analysis",
                        "grammatical_integrity_score",
                        "codeswitch_naturalness_analysis",
                        "codeswitch_naturalness_score"
                    ],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    res_data = response.json()
    
    try:
        text_content = res_data["choices"][0]["message"]["content"]
        return json.loads(text_content)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to parse OpenAI output: {e}. Raw: {res_data}")

# Calling Anthropic API
def call_anthropic(model: str, api_key: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    # We ask Claude to output strictly JSON as part of system/user instructions
    full_user_prompt = f"Please evaluate the following response according to our system prompt guidelines and return the JSON evaluation.\n\nInput:\n{user_prompt}"
    
    payload = {
        "model": model,
        "max_tokens": 1000,
        "system": f"{system_prompt}\n\nYou must return ONLY a valid JSON object matching the following structure:\n{{\n  \"grammatical_integrity_analysis\": \"...\",\n  \"grammatical_integrity_score\": <int 1-4>,\n  \"codeswitch_naturalness_analysis\": \"...\",\n  \"codeswitch_naturalness_score\": <int 1-4>\n}}",
        "messages": [
            {"role": "user", "content": full_user_prompt}
        ]
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    res_data = response.json()
    
    try:
        text_content = res_data["content"][0]["text"].strip()
        # Find JSON boundaries just in case Claude wraps it in markdown blocks
        match = re.search(r"\{.*\}", text_content, re.DOTALL)
        if match:
            text_content = match.group(0)
        return json.loads(text_content)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to parse Anthropic output: {e}. Raw: {res_data}")

def main():
    print("--- STEP 1: INITIALIZING LLM EVALUATION RUNNER ---")
    
    import argparse
    parser = argparse.ArgumentParser(description="LLM Evaluation Judge")
    parser.add_argument("--input", default="outputs/baseline_gemma.json", help="Input file path")
    parser.add_argument("--output", default="reports/baseline_gemma.json", help="Output report file path")
    args = parser.parse_args()
    
    try:
        provider, model_name, api_key = get_judge_provider()
        print(f"Detected Provider: {provider.upper()}")
        print(f"Judge Model: {model_name}")
    except ValueError as e:
        print(e)
        return

    # Load Rubrics and Anchors
    rubric_text = load_rubric("eval_rubrics.md")
    
    # Build System Prompt for the Judge
    system_prompt = f"""You are a senior LLM evaluation judge specializing in South Asian code-switching patterns (specifically Romanized Telugu/Telglish blended with English).
Your task is to evaluate the grammatical integrity and code-switching naturalness of an AI response to a casual Telugu prompt.

Here is the exact evaluation rubric specification, definitions, scoring criteria, and anchoring examples:
---
{rubric_text}
---

CRITICAL JUDGING GUIDELINES:
1. Be extremely objective and strict.
2. Read the anchors. Ensure you do not score responses higher than the criteria define.
3. In your analysis fields, first extract "evidence text strings" from the response. Show the word, phrase, or construction that justifies your score.
4. Output your analysis and score strictly in the requested JSON schema.
"""

    # Load the baseline prompts and responses
    with open(args.input, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    print(f"Loaded {len(baseline_data)} baseline entries from {args.input}")
    
    # Load existing progress if available
    existing_results = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                if isinstance(old_data, dict) and "results" in old_data:
                    for r in old_data["results"]:
                        g_analysis = r.get("grammatical_integrity_analysis", "") or ""
                        cs_analysis = r.get("codeswitch_naturalness_analysis", "") or ""
                        # Only skip if the prompt was successfully evaluated (no failure or API error fallback)
                        if "Evaluation Failed" not in g_analysis and "429" not in g_analysis and not r.get("api_error", False):
                            existing_results[r["prompt"]] = r
            if existing_results:
                print(f"Found {len(existing_results)} already evaluated prompts in {args.output}. Resuming evaluation...")
        except Exception as e:
            print(f"Warning: could not parse existing output file {args.output}: {e}. Starting fresh.")

    results = []
    
    # Run the evaluation
    for i, item in enumerate(baseline_data, 1):
        prompt = item["prompt"]
        response = item["response"]
        
        # Check if already evaluated in previous run
        if prompt in existing_results:
            print(f"[{i}/{len(baseline_data)}] Skipping prompt (already evaluated): '{prompt}'")
            results.append(existing_results[prompt])
            continue
            
        print(f"\n[{i}/{len(baseline_data)}] Evaluating prompt: '{prompt}'")
        print(f"Response: '{response}'")
        
        # Pre-check filter
        precheck_result = run_pre_check(response)
        if precheck_result:
            print(f" -> FLAGGED by pre-check: {precheck_result['flag_reason']}")
            eval_result = precheck_result
        else:
            # Prepare judge payload
            judge_input = json.dumps({
                "user_prompt": prompt,
                "model_response": response
            }, ensure_ascii=False, indent=2)
            
            # API retry logic with exponential backoff
            retries = 5
            eval_result = None
            for attempt in range(retries):
                try:
                    if provider == "gemini":
                        eval_result = call_gemini(model_name, api_key, system_prompt, judge_input)
                    elif provider == "openai":
                        eval_result = call_openai(model_name, api_key, system_prompt, judge_input)
                    elif provider == "anthropic":
                        eval_result = call_anthropic(model_name, api_key, system_prompt, judge_input)
                    
                    # Validate scores are within 1-4 range
                    g_score = int(eval_result.get("grammatical_integrity_score", 0))
                    cs_score = int(eval_result.get("codeswitch_naturalness_score", 0))
                    if not (1 <= g_score <= 4) or not (1 <= cs_score <= 4):
                        raise ValueError(f"Scores out of bounds: G={g_score}, CS={cs_score}")
                    
                    break  # Success
                except Exception as e:
                    # Parse status code to display clearer errors
                    status_msg = str(e)
                    print(f"    Attempt {attempt+1} failed: {status_msg}")
                    if attempt < retries - 1:
                        sleep_time = (attempt + 1) * 2
                        print(f"    Waiting {sleep_time} seconds before retrying...")
                        time.sleep(sleep_time)
                    else:
                        print("    All attempts failed. Assigning score 1 as fallback.")
                        eval_result = {
                            "grammatical_integrity_analysis": f"Evaluation Failed: {status_msg}",
                            "grammatical_integrity_score": 1,
                            "codeswitch_naturalness_analysis": f"Evaluation Failed: {status_msg}",
                            "codeswitch_naturalness_score": 1,
                            "api_error": True
                        }
            
            # Print brief summary of judge output
            print(f" -> Grammar Score: {eval_result['grammatical_integrity_score']}")
            print(f" -> Code-Switch Score: {eval_result['codeswitch_naturalness_score']}")
            
        results.append({
            "id": i,
            "prompt": prompt,
            "response": response,
            "grammatical_integrity_analysis": eval_result.get("grammatical_integrity_analysis"),
            "grammatical_integrity_score": eval_result.get("grammatical_integrity_score"),
            "codeswitch_naturalness_analysis": eval_result.get("codeswitch_naturalness_analysis"),
            "codeswitch_naturalness_score": eval_result.get("codeswitch_naturalness_score"),
            "precheck_flagged": eval_result.get("precheck_flagged", False),
            "flag_reason": eval_result.get("flag_reason", None),
            "api_error": eval_result.get("api_error", False)
        })
        
        # Calculate dynamic summary for incremental write
        temp_g_total = 0
        temp_cs_total = 0
        temp_collapses = 0
        for r in results:
            temp_g_total += r["grammatical_integrity_score"]
            temp_cs_total += r["codeswitch_naturalness_score"]
            if r["grammatical_integrity_score"] <= 2 or r["codeswitch_naturalness_score"] <= 2:
                temp_collapses += 1
        temp_avg_g = temp_g_total / len(results) if len(results) > 0 else 0
        temp_avg_cs = temp_cs_total / len(results) if len(results) > 0 else 0
        
        # Save progress incrementally after each step
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "total_prompts": len(results),
                    "average_grammatical_integrity": temp_avg_g,
                    "average_codeswitch_naturalness": temp_avg_cs,
                    "total_collapses": temp_collapses
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        # Rate limit friendly pause - reduced since user is on paid tier
        time.sleep(0.5)
        
    # Calculate final statistics
    total_g_score = 0
    total_cs_score = 0
    total_valid = len(results)
    
    collapses = []
    
    for r in results:
        g = r["grammatical_integrity_score"]
        cs = r["codeswitch_naturalness_score"]
        total_g_score += g
        total_cs_score += cs
        
        # Identify collapses: Score 1 or 2 in either axis
        if g <= 2 or cs <= 2:
            collapses.append(r)
            
    avg_g = total_g_score / total_valid if total_valid > 0 else 0
    avg_cs = total_cs_score / total_valid if total_valid > 0 else 0
    
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total Evaluated Prompts: {total_valid}")
    print(f"Average Axis 1 (Grammar) Score: {avg_g:.2f}/4.00")
    print(f"Average Axis 2 (Code-Switching) Score: {avg_cs:.2f}/4.00")
    print(f"Total Collapses (Score 1 or 2): {len(collapses)}")
    print(f"\nSaved evaluation report to {args.output}")
    
    # Print formatted markdown table of collapses
    print("\n--- COLLAPSE SAMPLES (SCORE 1 or 2) ---")
    print("| ID | Prompt | Response | Grammar Score | CS Score | Reason |")
    print("|---|---|---|---|---|---|")
    for c in collapses:
        # Truncate response if too long
        resp = c["response"]
        if len(resp) > 40:
            resp = resp[:37] + "..."
        # Extract quick reason summary
        reason = c["flag_reason"] if c["precheck_flagged"] else "LLM collapse"
        if not reason:
            if c["grammatical_integrity_score"] <= 2:
                reason = "Grammar collapse"
            else:
                reason = "CS collapse"
        print(f"| {c['id']} | {c['prompt']} | {resp} | {c['grammatical_integrity_score']} | {c['codeswitch_naturalness_score']} | {reason} |")

if __name__ == "__main__":
    main()
