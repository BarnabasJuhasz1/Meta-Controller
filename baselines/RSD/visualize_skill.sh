
    #--checkpoint_dir baselines/RSD/exp/RSD_exp/minigrid300epochsd042_1763588968_minigrid_RSD \ 


python scripts/visualize_skill.py \
    --checkpoint_dir /home/juhasz/Desktop/UZH/Reinforcement_Learning/Project_31/Open-Ended-Skill-Discovery/baselines/RSD/exp/RSD_exp/minigrid_500_discrete_PLEASE_PLEASE2sd042_1763917484_minigrid_RSD \
    --epoch 100 \
    --skill 1 \
    --max_steps 75 \
    --env minigrid

    