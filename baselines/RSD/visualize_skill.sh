
    #--checkpoint_dir baselines/RSD/exp/RSD_exp/minigrid300epochsd042_1763588968_minigrid_RSD \ 


python scripts/visualize_skill.py \
    --checkpoint_dir /home/juhasz/Desktop/UZH/Reinforcement_Learning/Project_31/Open-Ended-Skill-Discovery/baselines/RSD/exp/RSD_exp/mini_8d_1k_cont_obs_fixedsd042_1764074471_minigrid_RSD \
    --epoch 75 \
    --max_steps 50 \
    --env minigrid \
    --save_gif \
    --fps 8

