#!/usr/bin/env python3
import time
from opentelemetry import trace
from phoenix.otel import register

def main():
    print("--- STEP 1: REGISTERING TRACER WITH PHOENIX ---")
    # Register the OpenTelemetry provider to send spans to http://localhost:6006
    register(project_name="codeswitch-demo")
    
    # Get the tracer
    t = trace.get_tracer(__name__)
    
    print("\n--- STEP 2: GENERATING A NESTED TRACE ---")
    print("Simulating a model evaluation run...")
    
    # 1. Start the parent span (The overall pipeline step)
    with t.start_as_current_span("evaluate_prompt_pipeline") as parent_span:
        parent_span.set_attribute("input.value", "Vijayawada lo ICT sector abhivruddhi cheyadaniki avakasalu emiti?")
        
        # 2. Start a child span representing the model generation step
        with t.start_as_current_span("gemma_lora_generation") as gen_span:
            gen_span.set_attribute("llm.model_name", "google/gemma-4-e4b-it-lora")
            gen_span.set_attribute("llm.temperature", 0.7)
            
            # Simulate local inference delay
            time.sleep(0.5) 
            
            completion = "Vijayawada lo ICT sector growth ki chala opportunities unnayi. Startups scale up cheyadaniki incubation centers build cheyali."
            gen_span.set_attribute("output.value", completion)
            print(" -> Simulated Model Generation complete.")
            
        # 3. Start a sibling child span representing the LLM-as-a-judge step
        with t.start_as_current_span("gemini_judge_evaluation") as judge_span:
            judge_span.set_attribute("llm.model_name", "gemini-2.5-flash")
            judge_span.set_attribute("input.value", f"Prompt: ... \nResponse: {completion}")
            
            # Simulate API latency
            time.sleep(0.8) 
            
            judge_response = {
                "grammatical_integrity_score": 4,
                "codeswitch_naturalness_score": 4,
                "reasoning": "Sentence matches matrix frame correctly and uses natural Hyderabad speech style."
            }
            judge_span.set_attribute("output.value", str(judge_response))
            judge_span.set_attribute("evaluation.score.grammatical_integrity", 4)
            judge_span.set_attribute("evaluation.score.codeswitch_naturalness", 4)
            print(" -> Simulated Judge Evaluation complete.")
            
        # Add final output to the parent span
        parent_span.set_attribute("output.value", str(judge_response))

    print("\n==================================================")
    print("🚀 TRACE SENT!")
    print("1. Open the Phoenix UI: http://localhost:6006/")
    print("2. Click on 'Projects' on the left sidebar.")
    print("3. Select 'codeswitch-demo'.")
    print("4. Click on the trace to see the nested spans, attributes, and delays.")
    print("==================================================")

if __name__ == "__main__":
    main()
