#!/bin/bash

# Update this path to the downloaded zip file
# MODEL_PATH="ours/train_results/checkpoints/controller_12800_steps.zip"

#51200 -> 14% on 50 episodes (3 skills used)
#64000 -> 2% on 50 episodes  (4 skills used)
#76800 -> 8% on 50 episodes  (6 skills used)
#89600 -> 12% on 50 episodes (8 skills used)
# MODEL_PATH="ours/train_results/checkpoints/controller_test3/controller_89600_steps.zip"

MODEL_DIR="ours/train_results/checkpoints/test4"

# Assuming python is in path, otherwise use absolute path
python ours/evaluation/eval.py \
    --env_name minigrid \
    --skill_count_per_algo 8 \
    --config_path "ours/configs/config1.py" \
    --model_dir "$MODEL_DIR" \
    --num_episodes 25 \
    --output_dir "ours/evaluation/results/test4"
