#!/usr/bin/env python3
"""Modified script: windows never close. When max_steps is reached, the env resets and continues forever."""

import argparse
import numpy as np
import torch
import sys
import os
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.make_env import make_env
from garagei.envs.consistent_normalized_env import consistent_normalize

mp.set_start_method("spawn", force=True)


def load_checkpoint(checkpoint_dir, epoch):
    option_policy_path = os.path.join(checkpoint_dir, f'option_policy{epoch}.pt')
    traj_encoder_path = os.path.join(checkpoint_dir, f'traj_encoder{epoch}.pt')
    sample_z_policy_path = os.path.join(checkpoint_dir, f'SampleZPolicy{epoch}.pt')

    print(f"Loading checkpoints from {checkpoint_dir} at epoch {epoch}...")
    option_policy_ckpt = torch.load(option_policy_path, map_location='cpu', weights_only=False)
    traj_encoder_ckpt = None
    sample_z_ckpt = None

    if os.path.exists(traj_encoder_path):
        traj_encoder_ckpt = torch.load(traj_encoder_path, map_location='cpu', weights_only=False)
    else:
        print(f"Warning: Missing {traj_encoder_path}")

    if os.path.exists(sample_z_policy_path):
        sample_z_ckpt = torch.load(sample_z_policy_path, map_location='cpu', weights_only=False)
    else:
        print(f"Warning: Missing {sample_z_policy_path}")

    return option_policy_ckpt, traj_encoder_ckpt, sample_z_ckpt


def get_option_vector(k, dim_option, discrete, unit_length=False):
    if discrete:
        return np.eye(dim_option)[k]

    if dim_option == 2:
        num_skills = 8
        angle = (k % num_skills) * 2 * np.pi / num_skills
        r = 1.0 if unit_length else 1.5
        return np.array([r * np.cos(angle), r * np.sin(angle)])

    np.random.seed(k)
    v = np.random.randn(dim_option)
    if unit_length:
        v = v / np.linalg.norm(v)
    return v


def visualize_skill(checkpoint_dir, epoch, skill_k, max_steps=100, env_name='minigrid'):
    option_policy_ckpt, _, _ = load_checkpoint(checkpoint_dir, epoch)

    discrete = option_policy_ckpt['discrete']
    dim_option = option_policy_ckpt['dim_option']
    option = get_option_vector(skill_k, dim_option, discrete, unit_length=True)

    from argparse import Namespace
    args = Namespace(env=env_name, max_path_length=max_steps, normalizer_type='off', frame_stack=None)
    env = make_env(args, max_path_length=max_steps, render_mode='human')

    # Force window title if supported
    try:
        env.render('human', title=f"Skill {skill_k}")
        try:
            import pygame
            pygame.display.set_caption(f"Skill {skill_k}")
        except Exception:
            pass
        if hasattr(env.unwrapped, 'window') and env.unwrapped.window:
            env.unwrapped.window.set_caption(f"Skill {skill_k}")
    except Exception:
        pass

    option_policy = option_policy_ckpt['policy']
    option_policy.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    option_policy = option_policy.to(device)

    obs = env.reset()
    if hasattr(env, "render"):
        env.render('human', title=f"Skill {skill_k}")
        try:
            import pygame
            pygame.display.set_caption(f"Skill {skill_k}")
        except Exception:
            pass
            try:
                if hasattr(env.unwrapped, 'window') and env.unwrapped.window:
                    env.unwrapped.window.set_caption(f"Skill {skill_k}")
            except Exception:
                pass
        env.render('human', title=f"Skill {skill_k}")
        try:
            import pygame
            pygame.display.set_caption(f"Skill {skill_k}")
        except Exception:
            pass

    print(f"Running skill {skill_k} forever. Window will not close.")

    step_count = 0

    while True:  # infinite loop
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        option_tensor = torch.from_numpy(option).float().unsqueeze(0).to(device)

        if hasattr(option_policy, 'process_observations'):
            processed_obs = option_policy.process_observations(obs_tensor)
        else:
            processed_obs = obs_tensor

        concat_obs = torch.cat([processed_obs, option_tensor], dim=1)

        with torch.no_grad():
            dist, _ = option_policy(concat_obs)
            action = dist.mean.cpu().numpy()[0]

        obs, reward, done, info = env.step(action)
        step_count += 1

        if hasattr(env, 'render'):
            env.render('human', title=f"Skill {skill_k}")
        try:
            import pygame
            pygame.display.set_caption(f"Skill {skill_k}")
        except Exception:
            pass

        if done or step_count >= max_steps:
            obs = env.reset()
            step_count = 0



def _run_skill_in_process(checkpoint_dir, epoch, skill_k, max_steps, env_name):
    visualize_skill(checkpoint_dir, epoch, skill_k, max_steps, env_name)


def visualize_all_skills(checkpoint_dir, epoch, max_steps=100, env_name='minigrid'):
    option_policy_ckpt, _, _ = load_checkpoint(checkpoint_dir, epoch)
    discrete = option_policy_ckpt['discrete']
    dim_option = option_policy_ckpt['dim_option']

    num_skills = dim_option if discrete else 4
    print(f"Launching {num_skills} persistent windows.")

    procs = []
    for k in range(num_skills):
        p = mp.Process(target=_run_skill_in_process,
                       args=(checkpoint_dir, epoch, k, max_steps, env_name))
        p.start()
        procs.append(p)

    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--skill', type=int, required=False)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--env', type=str, default='minigrid')
    args = parser.parse_args()

    visualize_all_skills(args.checkpoint_dir, args.epoch, args.max_steps, args.env)


if __name__ == '__main__':
    main()
