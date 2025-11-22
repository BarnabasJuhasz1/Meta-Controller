export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL="osmesa"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

# to surpress import error warning
export D4RL_SUPPRESS_IMPORT_ERROR=1

# added --discrete 1 to use discrete skills
# in this setting --dim_option 4 sets the number of skills
# (because they are one-hot encoded in a discrete setting and dimensionality is 4)

# old 
# python tests/main.py --run_group RSD_exp --env minigrid --max_path_length 50 --seed 42 --traj_batch_size 4 --n_parallel 4 --n_epochs_per_eval 50 --n_epochs_per_save 50 --n_epochs_per_pt_save 50 --dim_option 4 --algo RSD --exp_name minigrid100 --phi_type Projection --explore_type SZN --trans_optimization_epochs 10 --is_wandb 1 --SZN_w2 10 --SZN_w3 3 --SZN_window_size 10 --SZN_repeat_time 3 --n_epochs 1000 --Repr_max_step 200 --dual_slack 1e-3 --discrete 1

# reduced discrete:
# python tests/main.py --run_group RSD_exp --env minigrid --max_path_length 50 --seed 42 --traj_batch_size 16 --n_parallel 8 --n_epochs_per_eval 100 --n_epochs_per_save 100 --n_epochs_per_pt_save 100 --dim_option 4  --algo RSD --exp_name minigrid_500_discrete_ --phi_type Projection --explore_type SZN  --trans_optimization_epochs 20  --is_wandb 1  --SZN_w2 3 --SZN_w3 1 --SZN_window_size 5 --SZN_repeat_time 2  --n_epochs 501 --Repr_max_step 60 --dual_slack 1e-3 --discrete 1

# reduced continuous
# python tests/main.py --run_group RSD_exp --env minigrid --max_path_length 50 --seed 42 --traj_batch_size 16 --n_parallel 8 --n_epochs_per_eval 100 --n_epochs_per_save 100 --n_epochs_per_pt_save 100 --dim_option 4  --algo RSD --exp_name minigrid500epoch_cont --phi_type Projection --explore_type SZN  --trans_optimization_epochs 20  --is_wandb 1  --SZN_w2 3 --SZN_w3 1 --SZN_window_size 5 --SZN_repeat_time 2  --n_epochs 500 --Repr_max_step 60 --dual_slack 1e-3 #--render_mode human

# trying to increase diversity:
#python tests/main.py --run_group RSD_exp --env minigrid --max_path_length 75 --seed 42 --traj_batch_size 64 --n_parallel 8 --n_epochs_per_eval 5 --n_epochs_per_save 5 --n_epochs_per_pt_save 5 --dim_option 6  --algo RSD --exp_name minigrid_500_discrete_AC_Proj --phi_type Projection --explore_type SZN  --trans_optimization_epochs 5  --is_wandb 1  --SZN_w2 1 --SZN_w3 0.3 --SZN_window_size 3 --SZN_repeat_time 1  --n_epochs 501 --Repr_max_step 75 --dual_slack 5e-2 --discrete 1

# fixing temporal cheating:
python tests/main.py --run_group RSD_exp --env minigrid --max_path_length 75 --seed 42 --traj_batch_size 256 --n_parallel 16 --n_epochs_per_eval 50 --n_epochs_per_save 100 --n_epochs_per_pt_save 100 --dim_option 8  --algo RSD --exp_name minigrid_500_discrete_PLEASE --phi_type Projection --explore_type SZN  --trans_optimization_epochs 150  --is_wandb 1  --SZN_w2 30 --SZN_w3 10 --SZN_window_size 20 --SZN_repeat_time 5 --n_epochs 3001 --Repr_max_step 75 --dual_slack 1e-4 --discrete 1
