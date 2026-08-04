from truss_train import definitions
from truss.base import truss_config

# 1. Define the Compute Resources
# Configure the GPU hardware required for training.
# For fine-tuning Gemma 4 (a 4B parameter model), a single A100 or A10G is recommended.
training_compute = definitions.Compute(
    node_count=1,
    accelerator=truss_config.AcceleratorSpec(
        # Choose from: A10G, A100, H100, L4 depending on your Baseten account quota
        accelerator=truss_config.Accelerator.A10G,
        count=1,
    ),
)

# 2. Define the Runtime Environment
# Controls environment variables, pip dependencies, start commands, caching, and checkpoints.
training_runtime = definitions.Runtime(
    start_commands=[
        # Install C/C++ compiler required for Triton runtime kernel compilation
        "apt-get update && apt-get install -y gcc g++ build-essential",
        
        # Install training packages from baseten_requirements.txt
        "pip install --upgrade pip",
        "pip install -r baseten_requirements.txt",
        
        # Execute training script pointing checkpoints to the Baseten managed volume
        "python train.py --model_id google/gemma-4-e4b-it --dataset_path data/train_sft_lima_200.json --output_dir $BT_CHECKPOINT_DIR"
    ],
    environment_variables={
        # HF_TOKEN must be registered as a secret in your Baseten dashboard
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
    },
    # Enables package/model caching to accelerate startup on subsequent training runs
    cache_config=definitions.CacheConfig(enabled=True),
    
    # Automatically syncs any checkpoints written to $BT_CHECKPOINT_DIR to Baseten's storage
    checkpointing_config=definitions.CheckpointingConfig(enabled=True),
)

# 3. Create the Training Job definition
# Combines the base Docker container, GPU compute, and execution script runtime.
training_job = definitions.TrainingJob(
    image=definitions.Image(
        # Standard PyTorch image with CUDA 12.1 support
        base_image="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime"
    ),
    compute=training_compute,
    runtime=training_runtime,
)

# 4. Create the Training Project definition
# Required by Baseten to group jobs and organize experiments.
training_project = definitions.TrainingProject(
    name="codeswitch-gemma-tuning",
    job=training_job,
)

