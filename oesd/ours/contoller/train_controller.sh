
#!/bin/bash

python ours/contoller/controller.py \
    --env_name minigrid \
    --skill_count_per_algo 8 \
    --num_timesteps 100000 \
    --learning_rate 3e-4 \
    --n_steps 16 \
    --batch_size 16 \
    --gamma 0.99 \
    --verbose 1 \
    --tensorboard_log "ours/train_results/logs/" \
    --save_path "ours/train_results/" \
    --config_path "ours/configs/config1.py" \
    --checkpoint_freq 20 \
    --render-mode "human"
