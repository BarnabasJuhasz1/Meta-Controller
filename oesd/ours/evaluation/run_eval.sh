#!/bin/bash

# Update this path to the downloaded zip file
MODEL_PATH="ours/train_results/checkpoints/rl_model_3200_steps.zip"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model file not found at $ModelPath"
    echo "Please ensure the .zip file is downloaded to oesd/ours/train_results/checkpoints/"
fi

# Assuming python is in path, otherwise use absolute path
python ours/evaluation/eval.py \
    --env_name minigrid \
    --skill_count_per_algo 8 \
    --config_path "ours/configs/config1.py" \
    --model_path "$MODEL_PATH" \
    --num_episodes 10 \
    --output_dir "ours/evaluation/results"
