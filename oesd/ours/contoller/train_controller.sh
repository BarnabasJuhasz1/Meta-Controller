
#!/bin/bash

python ours/contoller/controller.py \
    --env_name minigrid \
    --skill_count_per_algo 8 \
    --num_timesteps 200000 \
    --learning_rate 3e-4 \
    --n_steps 64 \
    --batch_size 64 \
    --gamma 0.99 \
    --verbose 1 \
    --tensorboard_log "ours/train_results/logs/" \
    --save_path "ours/train_results/checkpoints/test6" \
    --config_path "ours/configs/config1.py" \
    --checkpoint_freq 100 \
    # --render-mode "human"
