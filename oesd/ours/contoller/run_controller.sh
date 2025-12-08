#!/bin/bash

#MODEL_PATH="./ours/train_results/checkpoints/rl_model_3070_steps"
MODEL_PATH="/home/juhasz/Desktop/UZH/Reinforcement_Learning/Project_31/Open-Ended-Skill-Discovery/oesd/ours/train_results/checkpoints/rl_model_30720_steps.zip"

/home/juhasz/miniforge3/envs/rsd/bin/python ours/contoller/run_controller.py \
    --env_name minigrid \
    --skill_count_per_algo 8 \
    --config_path "ours/configs/config1.py" \
    --model_path "$MODEL_PATH" \
    --num_episodes 3
