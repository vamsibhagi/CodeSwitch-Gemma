#!/usr/bin/env python3
import os
import json
import pandas as pd
import phoenix as px
from phoenix.client import Client

def main():
    report_path = "reports/baseline_gemma.json"
    
    if not os.path.exists(report_path):
        print(f"Error: {report_path} not found. Please run the evaluation script first or specify a valid file.")
        return
        
    print(f"Loading evaluation report from {report_path}...")
    with open(report_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)
        
    results = report_data.get("results", [])
    if not results:
        print("Error: No evaluation results found in the report file.")
        return
        
    # Convert report results to a Pandas DataFrame
    data = []
    for item in results:
        data.append({
            "prompt": item.get("prompt"),
            "response": item.get("response"),
            "grammatical_integrity_score": item.get("grammatical_integrity_score"),
            "grammatical_integrity_analysis": item.get("grammatical_integrity_analysis"),
            "codeswitch_naturalness_score": item.get("codeswitch_naturalness_score"),
            "codeswitch_naturalness_analysis": item.get("codeswitch_naturalness_analysis"),
            "has_collapse": 1 if (item.get("grammatical_integrity_score", 4) <= 2 or item.get("codeswitch_naturalness_score", 4) <= 2) else 0
        })
        
    df = pd.DataFrame(data)
    
    print("\n--- STEP 1: LAUNCHING PHOENIX SERVER LOCALLY ---")
    session = px.launch_app()
    print(f"Phoenix UI is running at: {session.url}")
    
    print("\n--- STEP 2: CREATING PHOENIX DATASET ---")
    px_client = Client()
    
    # Define inputs and outputs for the dataset
    dataset_name = "CodeSwitch_Gemma_Baseline_Evals"
    
    # Delete if already exists to start clean
    try:
        import requests
        datasets = px_client.datasets.list()
        for ds in datasets:
            if ds.name == dataset_name:
                print(f"Removing existing dataset: {dataset_name}")
                url = f"{px_client.base_url}/v1/datasets/{ds.id}"
                resp = requests.delete(url)
                resp.raise_for_status()
    except Exception as e:
        print(f"Note: Could not list/delete datasets: {e}")
        
    # Create the dataset in Phoenix
    dataset = px_client.datasets.create_dataset(
        dataframe=df,
        name=dataset_name,
        input_keys=["prompt"],
        output_keys=["response", "grammatical_integrity_score", "grammatical_integrity_analysis", "codeswitch_naturalness_score", "codeswitch_naturalness_analysis", "has_collapse"]
    )
    
    print(f"Successfully uploaded dataset '{dataset_name}' with {len(df)} items to Phoenix!")
    print("\n==================================================")
    print("🚀 HOW TO TRY IT:")
    print(f"1. Open the Phoenix UI in your browser: {session.url}")
    print("2. Click on 'Datasets' in the left-hand navigation pane.")
    print(f"3. Click on '{dataset_name}' to see your evaluation prompts, responses, scores, and judge reasoning in a beautiful tabular view.")
    print("==================================================")
    
    # Keep the script running so the server stays active for the user
    print("\nPress Ctrl+C to stop the Phoenix server...")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping Phoenix server. Goodbye!")

if __name__ == "__main__":
    main()
