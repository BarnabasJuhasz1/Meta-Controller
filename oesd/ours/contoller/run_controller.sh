#!/bin/bash

MODEL_PATH="ours/train_results/checkpoints/controller_test3/controller_89600_steps"

/home/juhasz/miniforge3/envs/rsd/bin/python ours/contoller/run_controller.py \
    --env_name minigrid \
    --skill_count_per_algo 8 \
    --config_path "ours/configs/config1.py" \
    --model_path "$MODEL_PATH" \
    --num_episodes 3
