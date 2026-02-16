export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL="osmesa"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

#python tests/main.py --run_group Large --env ant_maze_large --max_path_length 300 --seed 0 --traj_batch_size 16 --n_parallel 4 --n_epochs_per_eval 500  --dim_option 4  --algo SZPC --exp_name TheBest --phi_type Projection --explore_type SZN  --trans_optimization_epochs 150  --is_wandb 1  --SZN_w2 10 --SZN_w3 3 --SZN_window_size 20 --SZN_repeat_time 3  --n_epochs 20000 --Repr_max_step 300 --dual_slack 1e-3 
# python tests/main.py --run_group Large --env ant_maze_large --max_path_length 300 --seed 0 --traj_batch_size 16 --n_parallel 4 --n_epochs_per_eval 500  --dim_option 4  --algo metra --exp_name TheBest --phi_type Projection --explore_type SZN  --trans_optimization_epochs 150  --is_wandb 1  --SZN_w2 10 --SZN_w3 3 --SZN_window_size 20 --SZN_repeat_time 3  --n_epochs 20000 --Repr_max_step 300 --dual_slack 1e-3 

# disabled video recording for now, as it is not supported for the normal 'maze' environment

# maze 
# python tests/main.py --run_group RSD_exp --env maze --max_path_length 100 --seed 0 --traj_batch_size 16 --n_parallel 2 --n_epochs_per_eval 10  --dim_option 4  --algo RSD --exp_name RSDExp --phi_type Projection --explore_type SZN  --trans_optimization_epochs 50  --is_wandb 1  --SZN_w2 10 --SZN_w3 3 --SZN_window_size 10 --SZN_repeat_time 3  --n_epochs 10000 --Repr_max_step 300 --dual_slack 1e-3 --eval_record_video 0

# maze2d-large-v1
python tests/main.py --run_group RSD_exp --env lm --max_path_length 100 --seed 0 --traj_batch_size 16 --n_parallel 2 --n_epochs_per_eval 250  --dim_option 4  --algo RSD --exp_name maze2D_10k_epochs --phi_type Projection --explore_type SZN  --trans_optimization_epochs 50  --is_wandb 1  --SZN_w2 10 --SZN_w3 3 --SZN_window_size 10 --SZN_repeat_time 3  --n_epochs 10000 --Repr_max_step 300 --dual_slack 1e-3

