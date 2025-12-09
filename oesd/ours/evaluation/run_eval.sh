#!/bin/bash

# Update this path to the downloaded zip file
MODEL_PATH="oesd/ours/train_results/checkpoints/model.zip"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model file not found at $ModelPath"
    echo "Please ensure the .zip file is downloaded to oesd/ours/train_results/checkpoints/"
fi

# Assuming python is in path, otherwise use absolute path
python oesd/ours/evaluation/eval.py \
    --env_name minigrid \
    --skill_count_per_algo 8 \
    --config_path "oesd/ours/configs/config1.py" \
    --model_path "$MODEL_PATH" \
    --num_episodes 10 \
    --output_dir "oesd/ours/evaluation/results"
