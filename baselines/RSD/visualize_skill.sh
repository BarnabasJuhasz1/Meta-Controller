
# epoch 75, skill 2 picks up the key.
#--checkpoint_dir /home/juhasz/Desktop/UZH/Reinforcement_Learning/Project_31/Open-Ended-Skill-Discovery/baselines/RSD/exp/RSD_exp/mini_4d_1k_cont_obs_fixedsd042_1764111212_minigrid_RSD \

# /home/juhasz/Desktop/UZH/Reinforcement_Learning/Project_31/Open-Ended-Skill-Discovery/baselines/RSD/exp/RSD_small_exp/mini_8d_3P_img_dir_carrysd042_1764457992_minigrid_small_RSD
# epoch 225, skill 1 likes to walk up and down in the room pretty consistently irrespectively of the room size

# /home/juhasz/Desktop/UZH/Reinforcement_Learning/Project_31/Open-Ended-Skill-Discovery/baselines/RSD/exp/RSD_small_exp/mini_4d_3P_img_dir_carry_originalsd042_1764463166_minigrid_small_RSD
# epoch 100, some skills move around and like to keep picking up and putting down the key repeatedly
# skill 6 --> mostly follows the side of the room, and when key found just picks up and places down
# skill 0 --> follow the side of the room, when key found, just pick it up and go forward until you get to the other side of the room

# epoch 175, skill 0 almost always seem to find and pick up the key
# epoch 200, skill 0 most of the time looks for the key and picks it up, but then returns to the bottom right corner
# epoch 300, skill 5 seems to always find the key and pick it up

python scripts/visualize_skill.py \
    --checkpoint_dir /home/juhasz/Desktop/UZH/Reinforcement_Learning/Project_31/Open-Ended-Skill-Discovery/baselines/RSD/exp/RSD_small_exp/mini_4d_3P_img_dir_carry_originalsd042_1764463166_minigrid_small_RSD \
    --epoch 50 \
    --max_steps 50 \
    --env minigrid_small \
    --save_gif \
    --fps 8 \
    # --repeat 20 \
    # --only_skill_index 2 \


#--env minigrid_random_key \
