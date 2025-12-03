#!/bin/bash

# Default model path (can be overridden)
MODEL_PATH="ours/train_results/checkpoints/rl_model_61440_steps.zip"

/home/juhasz/miniforge3/envs/rsd/bin/python ours/contoller/run_controller.py \
    --env_name minigrid \
    --skill_count_per_algo 8 \
    --config_path "ours/configs/config1.py" \
    --model_path "$MODEL_PATH" \
    --num_episodes 3
