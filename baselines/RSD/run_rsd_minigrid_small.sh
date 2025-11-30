#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL="osmesa"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
export PYTHONPATH=$PYTHONPATH:.:$(pwd)/garaged/src

# to surpress import error warning
export D4RL_SUPPRESS_IMPORT_ERROR=1

# Run RSD on minigrid_small (DoorKeyEnv-5x5)
# Using same hyperparameters as minigrid_random_key but with new env name
# python tests/main.py --run_group RSD_small_exp --env minigrid_small --max_path_length 50 --seed 42 --traj_batch_size 128 --n_parallel 16 --n_epochs_per_eval 50 --n_epochs_per_save 25 --n_epochs_per_pt_save 25 --dim_option 8 --algo RSD --exp_name mini_small_8d --phi_type Projection --explore_type SZN  --trans_optimization_epochs 100  --is_wandb 1  --SZN_w2 3 --SZN_w3 1 --SZN_window_size 5 --SZN_repeat_time 2 --n_epochs 301 --Repr_max_step 50 --dual_slack 5e-3 --dual_lam 1.0 --discrete 0

# switched to size 8
# python3 tests/main.py --run_group RSD_small_exp --env minigrid_small --max_path_length 50 --seed 42 --traj_batch_size 64 --n_parallel 16 --n_epochs_per_eval 50 --n_epochs_per_save 25 --n_epochs_per_pt_save 25 --dim_option 8 --algo RSD --exp_name mini_small_8d_size8_4 --phi_type Projection --explore_type SZN  --trans_optimization_epochs 100  --is_wandb 1  --SZN_w2 3 --SZN_w3 1 --SZN_window_size 5 --SZN_repeat_time 2 --n_epochs 301 --Repr_max_step 50 --dual_slack 5e-3 --dual_lam 1 --discrete 0 --render_mode human

# trying previously good working setting
python tests/main.py --run_group RSD_small_exp --env minigrid_small --max_path_length 50 --seed 42 --traj_batch_size 64 --n_parallel 16 --n_epochs_per_eval 50 --n_epochs_per_save 25 --n_epochs_per_pt_save 25 --dim_option 4 --algo RSD --exp_name mini_4d_3P_img_dir_carry_new --phi_type Projection --explore_type SZN  --trans_optimization_epochs 100  --is_wandb 1  --SZN_w2 3 --SZN_w3 1 --SZN_window_size 5 --SZN_repeat_time 2 --n_epochs 301 --Repr_max_step 150 --dual_slack 5e-3 --dual_lam 1.0 --discrete 0
