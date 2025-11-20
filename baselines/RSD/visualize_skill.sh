
    #--checkpoint_dir baselines/RSD/exp/RSD_exp/minigrid300epochsd042_1763588968_minigrid_RSD \ 


python scripts/visualize_skill.py \
    --checkpoint_dir /home/juhasz/Desktop/UZH/Reinforcement_Learning/Project_31/Open-Ended-Skill-Discovery/baselines/RSD/exp/RSD_exp/minigrid500epoch_contsd042_1763595052_minigrid_RSD \
    --epoch 400 \
    --skill 1 \
    --max_steps 50 \
    --env minigrid

    