#!/bin/bash

# get the directory where this .sh file is located
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python ours/unified_baseline_utils/visualize.py \
    --env_name minigrid \
    --algo_name RSD \
    --skill_idx 0 \
    --config ours/unified_baseline_utils/configs/config1.py \
    --horizon 100 \
    --episodes 1 \
    --deterministic False \
    --render_mode human