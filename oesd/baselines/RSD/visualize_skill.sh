
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


# 8d_3P_img_dir_carry is alright (but didn't really learn to pick up the key)
# 4d_3P_img_dir_carry_original is not too bad (but latent space looks bad)
# 4d_3P_img_dir_carry_new (continuous) is bad (but latent space looks a bit better)

# 2d_3P_img_dir_carry_new_C seems pretty bad --> 2D is not the way to go
# 4d_3P_img_dir_carry_original_C is pretty good



# --checkpoint_dir /home/juhasz/Desktop/UZH/Reinforcement_Learning/Project_31/Open-Ended-Skill-Discovery/baselines/RSD/exp/RSD_small_exp/mini_4d_3P_img_dir_carry_original_Csd042_1764514102_minigrid_small_RSD \

python baselines/RSD/scripts/visualize_skill.py \
    --checkpoint_dir baseline_checkpoints/RSD/4D_3P_img_dir_carry_orig_C/ \
    --epoch 175 \
    --max_steps 50 \
    --env minigrid_small \
    --save_gif \
    --fps 8 \
    --repeat 20 \
    --only_skill_index 6 
    # --seed 5

#--env minigrid_random_key \



# picks up the key
# 0 175 --> sometimes picks up the key
# 2 175 --> mostly circle but if key is in the way it picks it up
# 6 175 --> cycle until key is picked up

# 2 200 --> mostly circle but if key is in the way it picks it up
# 4 200 
# 6 200 --> cycle until key is picked up

# 2 225
# 4 225

# 4 250 --> find key wherever it is but just stand next to it

# 3 275

# 5 300
# 6 300

# pick up and place
# 4 275 --> sometimes look for key and start place / pickup endless cycle
# 5 250

# circle in the room
# 3 175
# 4 175

# if it sees the key, it picks up key and place it somewhere else
# 3 200
