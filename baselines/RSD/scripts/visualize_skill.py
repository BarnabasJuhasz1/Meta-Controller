#!/usr/bin/env python3
"""Script to load RSD checkpoint and visualize a specific skill in minigrid environment."""

import argparse
import numpy as np
import torch
import sys
import os

# Add the baselines/RSD directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.make_env import make_env
from garagei.envs.consistent_normalized_env import consistent_normalize

import multiprocessing as mp
mp.set_start_method("spawn", force=True)


def load_checkpoint(checkpoint_dir, epoch):
    """Load the three checkpoint files."""
    option_policy_path = os.path.join(checkpoint_dir, f'option_policy{epoch}.pt')
    traj_encoder_path = os.path.join(checkpoint_dir, f'traj_encoder{epoch}.pt')
    sample_z_policy_path = os.path.join(checkpoint_dir, f'SampleZPolicy{epoch}.pt')
    
    if not os.path.exists(option_policy_path):
        raise FileNotFoundError(f"Checkpoint not found: {option_policy_path}")
    if not os.path.exists(traj_encoder_path):
        #raise FileNotFoundError(f"Warning: Checkpoint not found: {traj_encoder_path}")
        print(f"Warning: Checkpoint not found: {traj_encoder_path}")
    if not os.path.exists(sample_z_policy_path):
        #raise FileNotFoundError(f"Checkpoint not found: {sample_z_policy_path}")
        print(f"Warning: Checkpoint not found: {sample_z_policy_path}")

    print(f"Loading checkpoints from {checkpoint_dir} at epoch {epoch}...")
    
    # Use weights_only=False to allow loading custom classes (PolicyEx, etc.)
    option_policy_ckpt = torch.load(option_policy_path, map_location='cpu', weights_only=False)
    traj_encoder_ckpt = None
    sample_z_ckpt = None
    
    if os.path.exists(traj_encoder_path):
        traj_encoder_ckpt = torch.load(traj_encoder_path, map_location='cpu', weights_only=False)
    if os.path.exists(sample_z_policy_path):
        sample_z_ckpt = torch.load(sample_z_policy_path, map_location='cpu', weights_only=False)
    
    return option_policy_ckpt, traj_encoder_ckpt, sample_z_ckpt

def get_option_vector(k, dim_option, discrete, unit_length=False):
    """Get the option vector for skill k."""
    if discrete:
        # One-hot vector for discrete skills
        if k >= dim_option:
            raise ValueError(f"Skill index {k} must be < {dim_option} for discrete skills")
        option = np.eye(dim_option)[k]
    else:
        # For continuous skills, use a deterministic mapping
        # Create a grid of skills in the option space
        if dim_option == 2:
            # Map k to angles on a circle
            num_skills = 8
            angle = (k % num_skills) * 2 * np.pi / num_skills
            radius = 1.0 if unit_length else 1.5
            option = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        else:
            # For higher dimensions, use a deterministic pattern
            np.random.seed(k)  # Deterministic based on k
            option = np.random.randn(dim_option)
            if unit_length:
                option = option / np.linalg.norm(option)
    
    return option


def visualize_skill(checkpoint_dir, epoch, skill_k, max_steps=100, env_name='minigrid'):
    """Load checkpoint and visualize skill k."""
    
    # Load checkpoints
    option_policy_ckpt, traj_encoder_ckpt, sample_z_ckpt = load_checkpoint(checkpoint_dir, epoch)
    
    discrete = option_policy_ckpt['discrete']
    dim_option = option_policy_ckpt['dim_option']
    
    print(f"Loaded checkpoint: discrete={discrete}, dim_option={dim_option}")
    
    # Get option vector for skill k
    option = get_option_vector(skill_k, dim_option, discrete, unit_length=True)
    print(f"Using skill {skill_k} with option vector: {option}")
    
    # Create environment
    from argparse import Namespace
    args = Namespace(
        env=env_name,
        max_path_length=max_steps,
        normalizer_type='off',
        frame_stack=None,
    )
    env = make_env(args, max_path_length=max_steps, render_mode='human')
    
    # Load models
    option_policy = option_policy_ckpt['policy']
    option_policy.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    option_policy = option_policy.to(device)
    
    # Reset environment
    obs = env.reset()
    if hasattr(env, "render"):
        env.render('human', title=f"Skill {skill_k}")
    done = False
    step_count = 0
    
    print(f"\nStarting visualization of skill {skill_k}...")
    print("The environment window should open. Watch the agent execute the skill.")
    print("Press Ctrl+C to exit early.\n")
    
    # Run the agent with the selected skill
    while not done and step_count < max_steps:
        # Convert observation to tensor
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        option_tensor = torch.from_numpy(option).float().unsqueeze(0).to(device)
        
        # Concatenate observation and option
        if hasattr(option_policy, 'process_observations'):
            processed_obs = option_policy.process_observations(obs_tensor)
        else:
            processed_obs = obs_tensor
        
        concat_obs = torch.cat([processed_obs, option_tensor], dim=1)
        
        # Get action from policy (deterministic: use mean)
        with torch.no_grad():
            dist, _ = option_policy(concat_obs)
            # Use mean for deterministic behavior
            action = dist.mean
            action = action.cpu().numpy()[0]
        
        # Step environment
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        # Render (environment should auto-render in human mode)
        if hasattr(env, 'render'):
            env.render('human', title=f"Skill {skill_k}")
        
        if step_count % 10 == 0:
            print(f"Step {step_count}, Reward: {reward:.3f}")
    
    print(f"\nEpisode finished after {step_count} steps")
    print("Closing environment window...")


def assign_unique_viewer(env, name):
    """Force Minigrid env to create independent render windows."""
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "render"):
        # force renderer to recreate viewer with a unique window title
        try:
            env.unwrapped.window = None
        except:
            pass
        env.render('human', title=name)


# def visualize_all_skills(checkpoint_dir, epoch, max_steps=100, env_name='minigrid'):
#     """Visualize all skills at once in separate windows. Each env resets when hitting max_steps."""
#     # Load checkpoints
#     option_policy_ckpt, traj_encoder_ckpt, sample_z_ckpt = load_checkpoint(checkpoint_dir, epoch)
#     discrete = option_policy_ckpt['discrete']
#     dim_option = option_policy_ckpt['dim_option']

#     print(f"Loaded checkpoint: discrete={discrete}, dim_option={dim_option}")
#     print("Starting visualization of ALL skills... Ctrl+C to stop.")

#     from argparse import Namespace
#     args = Namespace(
#         env=env_name,
#         max_path_length=max_steps,
#         normalizer_type='off',
#         frame_stack=None,
#     )

#     # Create one env per skill
#     envs = []
#     for k in range(dim_option if discrete else 8):  # for continuous: show 8 skills
#         env = make_env(args, max_path_length=max_steps)
#         assign_unique_viewer(env, name=f"Skill {k}")
#         envs.append(env)

#     # Load policy
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     option_policy = option_policy_ckpt['policy'].to(device).eval()

#     # Precompute option vectors
#     options = [
#         torch.from_numpy(get_option_vector(k, dim_option, discrete, unit_length=True)).float().unsqueeze(0).to(device)
#         for k in range(len(envs))
#     ]

#     # Init all envs
#     obss = [env.reset() for env in envs]
#     step_counts = [0 for _ in envs]

#     while True:
#         for i, env in enumerate(envs):
#             obs = obss[i]
#             if step_counts[i] >= max_steps:
#                 obs = env.reset()
#                 step_counts[i] = 0

#             obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)

#             if hasattr(option_policy, 'process_observations'):
#                 processed_obs = option_policy.process_observations(obs_tensor)
#             else:
#                 processed_obs = obs_tensor

#             concat_obs = torch.cat([processed_obs, options[i]], dim=1)

#             with torch.no_grad():
#                 dist, _ = option_policy(concat_obs)
#                 action = dist.mean.cpu().numpy()[0]

#             obs, reward, done, info = env.step(action)
#             env.render('human', title=f"Skill {i}")

#             step_counts[i] += 1
#             obss[i] = obs

#             if done:
#                 obss[i] = env.reset()
#                 step_counts[i] = 0

import multiprocessing as mp

def _run_skill_in_process(checkpoint_dir, epoch, skill_k, max_steps, env_name):
    """Internal worker used by multiprocessing."""
    visualize_skill(
        checkpoint_dir=checkpoint_dir,
        epoch=epoch,
        skill_k=skill_k,
        max_steps=max_steps,
        env_name=env_name
    )


def visualize_all_skills(checkpoint_dir, epoch, max_steps=100, env_name='minigrid'):
    """Run each skill in its own process → multiple real windows."""
    option_policy_ckpt, traj_encoder_ckpt, sample_z_ckpt = load_checkpoint(checkpoint_dir, epoch)
    discrete = option_policy_ckpt['discrete']
    dim_option = option_policy_ckpt['dim_option']

    if discrete:
        print(f"DISCRETE SKILLS WERE SET DURING TRAINIG")
        num_skills = dim_option
    else:
        print(f"CONTINUOUS SKILLS WERE SET DURING TRAINIG")
        num_skills = 4  # same choice you used for circular continuous skills

    print(f"Launching {num_skills} windows (one per skill). Ctrl+C to stop them all.")

    procs = []
    for k in range(num_skills):
        p = mp.Process(
            target=_run_skill_in_process,
            args=(checkpoint_dir, epoch, k, max_steps, env_name),
        )
        p.start()
        procs.append(p)

    # Wait so processes don't instantly exit
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("Stopping all processes...")
        for p in procs:
            p.terminate()


def main():
    parser = argparse.ArgumentParser(description='Visualize a specific RSD skill in minigrid')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing the .pt checkpoint files')
    parser.add_argument('--epoch', type=int, required=True,
                        help='Epoch number of the checkpoint to load')
    parser.add_argument('--skill', type=int, required=True,
                        help='Skill index k to visualize')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum number of steps to run')
    parser.add_argument('--env', type=str, default='minigrid',
                        help='Environment name (default: minigrid)')
    
    args = parser.parse_args()
    
    #visualize_skill(
    #    checkpoint_dir=args.checkpoint_dir,
    #    epoch=args.epoch,
    #    skill_k=args.skill,
    #    max_steps=args.max_steps,
    #    env_name=args.env
    #)

    visualize_all_skills(
        checkpoint_dir=args.checkpoint_dir,
        epoch=args.epoch,
        max_steps=args.max_steps,
        env_name=args.env
    )



if __name__ == '__main__':
    main()

