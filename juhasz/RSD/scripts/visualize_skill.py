#!/usr/bin/env python3
"""Modified script: windows never close. When max_steps is reached, the env resets and continues forever."""

import argparse
import numpy as np
import torch
import sys
import os
import multiprocessing as mp
from PIL import Image, ImageDraw, ImageFont

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


def save_frames_as_gif(frames, path, fps=30):
    if not frames:
        return
    
    # Convert to PIL images if they are numpy arrays
    pil_images = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    
    pil_images[0].save(
        path,
        save_all=True,
        append_images=pil_images[1:],
        duration=1000/fps,
        loop=0
    )
    print(f"Saved GIF to {path}")


def visualize_skill(checkpoint_dir, epoch, skill_k, max_steps=100, env_name='minigrid', save_gif=False, gif_dir='./gifs', fps=30, repeat=1):
    option_policy_ckpt, _, _ = load_checkpoint(checkpoint_dir, epoch)

    discrete = option_policy_ckpt['discrete']
    dim_option = option_policy_ckpt['dim_option']
    option = get_option_vector(skill_k, dim_option, discrete, unit_length=True)

    from argparse import Namespace
    args = Namespace(env=env_name, max_path_length=max_steps, normalizer_type='off', frame_stack=None)
    
    render_mode = 'rgb_array' if save_gif else 'human'
    env = make_env(args, max_path_length=max_steps, render_mode=render_mode)

    if not save_gif:
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

    for r in range(repeat):
        obs = env.reset()
        
        frames = []
        if save_gif:
            frame = env.render('rgb_array')
            # Add step count to frame
            if isinstance(frame, np.ndarray):
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                # Use default font
                try:
                    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
                except IOError:
                    font = ImageFont.load_default()
                
                text = f"Step: 0"
                # Draw text with black outline for visibility
                x, y = 10, 10
                draw.text((x-1, y), text, font=font, fill="black")
                draw.text((x+1, y), text, font=font, fill="black")
                draw.text((x, y-1), text, font=font, fill="black")
                draw.text((x, y+1), text, font=font, fill="black")
                draw.text((x, y), text, font=font, fill="white")
                
                frames.append(img)
            else:
                frames.append(frame)
        elif hasattr(env, "render"):
            env.render('human', title=f"Skill {skill_k}")
            try:
                import pygame
                pygame.display.set_caption(f"Skill {skill_k}")
            except Exception:
                pass

        if not save_gif:
            print(f"Running skill {skill_k} forever. Window will not close.")
        else:
            print(f"Recording skill {skill_k} for episode {r+1}/{repeat}...")

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

            if save_gif:
                frame = env.render('rgb_array')
                # Add step count to frame
                if isinstance(frame, np.ndarray):
                    img = Image.fromarray(frame)
                    draw = ImageDraw.Draw(img)
                    # Use default font
                    try:
                        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
                    except IOError:
                        font = ImageFont.load_default()
                    
                    text = f"Step: {step_count}"
                    # Draw text with black outline for visibility
                    x, y = 10, 10
                    draw.text((x-1, y), text, font=font, fill="black")
                    draw.text((x+1, y), text, font=font, fill="black")
                    draw.text((x, y-1), text, font=font, fill="black")
                    draw.text((x, y+1), text, font=font, fill="black")
                    draw.text((x, y), text, font=font, fill="white")
                    
                    frames.append(img)
                else:
                    frames.append(frame)
            elif hasattr(env, 'render'):
                env.render('human', title=f"Skill {skill_k}")
                try:
                    import pygame
                    pygame.display.set_caption(f"Skill {skill_k}")
                except Exception:
                    pass

            if done or step_count >= max_steps:
                if save_gif:
                    os.makedirs(gif_dir, exist_ok=True)
                    suffix = f"_rep{r}" if repeat > 1 else ""
                    save_path = os.path.join(gif_dir, f"skill_{skill_k}_epoch_{epoch}{suffix}.gif")
                    save_frames_as_gif(frames, save_path, fps=fps)
                    break
                
                obs = env.reset()
                step_count = 0



def _run_skill_in_process(checkpoint_dir, epoch, skill_k, max_steps, env_name, save_gif, gif_dir, fps, repeat):
    visualize_skill(checkpoint_dir, epoch, skill_k, max_steps, env_name, save_gif, gif_dir, fps, repeat)


def visualize_all_skills(checkpoint_dir, epoch, max_steps=100, env_name='minigrid', save_gif=False, gif_dir='./gifs', fps=30, repeat=1):
    option_policy_ckpt, _, _ = load_checkpoint(checkpoint_dir, epoch)
    discrete = option_policy_ckpt['discrete']
    dim_option = option_policy_ckpt['dim_option']

    num_skills = dim_option if discrete else 8
    
    if save_gif:
        print(f"Saving GIFs for {num_skills} skills...")
    else:
        print(f"Launching {num_skills} persistent windows.")

    procs = []
    for k in range(num_skills):
        p = mp.Process(target=_run_skill_in_process,
                       args=(checkpoint_dir, epoch, k, max_steps, env_name, save_gif, gif_dir, fps, repeat))
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
    parser.add_argument('--epoch', type=int, nargs='+', required=True)
    parser.add_argument('--only_skill_index', type=int, required=False, help='Visualize only this specific skill index')
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--env', type=str, default='minigrid')
    parser.add_argument('--save_gif', action='store_true', help='Save skills as GIFs instead of rendering')
    parser.add_argument('--gif_dir', type=str, default='./gifs', help='Directory to save GIFs')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the GIF')
    parser.add_argument('--repeat', type=int, default=1, help='Number of times to repeat the visualization (re-initializing env)')
    args = parser.parse_args()

    for epoch in args.epoch:
        if args.only_skill_index is not None:
            visualize_skill(args.checkpoint_dir, epoch, args.only_skill_index, args.max_steps, args.env, args.save_gif, args.gif_dir, args.fps, args.repeat)
        else:
            visualize_all_skills(args.checkpoint_dir, epoch, args.max_steps, args.env, args.save_gif, args.gif_dir, args.fps, args.repeat)


if __name__ == '__main__':
    main()
