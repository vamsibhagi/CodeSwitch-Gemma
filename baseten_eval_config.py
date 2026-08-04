from truss_train import definitions
from truss.base import truss_config

# 1. Define the Compute Resources for Generation
# Offload GPU generation to an A10G instance on Baseten.
training_compute = definitions.Compute(
    node_count=1,
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.A10G,
        count=1,
    ),
)

# 2. Define the Evaluation Runtime
training_runtime = definitions.Runtime(
    start_commands=[
        # Install C/C++ compiler required for Triton runtime kernel compilation
        "apt-get update && apt-get install -y gcc g++ build-essential",
        
        # Install requirements
        "pip install --upgrade pip",
        "pip install -r baseten_requirements.txt",
        
        # 1. Run baseline model generation
        "python generate_completions.py --model_id google/gemma-4-e4b-it --baseline --output $BT_CHECKPOINT_DIR/baseline_completions.json",
        
        # 2. Run fine-tuned model generation using the newly trained adapters from Hugging Face
        "python generate_completions.py --model_id google/gemma-4-e4b-it --adapter_id vamsibhagi/CodeSwitch-Gemma --output $BT_CHECKPOINT_DIR/finetuned_completions.json"
    ],
    environment_variables={
        # HF_TOKEN to authenticate downloading the base model and your private Hugging Face adapter
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
    },
    # Enable project caching to speed up model downloading on subsequent runs
    cache_config=definitions.CacheConfig(enabled=True),
    
    # Automatically syncs generated outputs written to $BT_CHECKPOINT_DIR back to Baseten
    checkpointing_config=definitions.CheckpointingConfig(enabled=True),
)

# 3. Create the Project definition
training_project = definitions.TrainingProject(
    name="codeswitch-gemma-evals",
    job=definitions.TrainingJob(
        image=definitions.Image(
            base_image="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime"
        ),
        compute=training_compute,
        runtime=training_runtime,
    )
)
