"""
Fixed visualization script for DIAYN on MiniGrid.
This version fixes: frozen frames, agent not moving, rgb_array_list behavior, and new Gym reset API.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
import imageio
from src.agents.diayn_agent import DIAYNAgent
from src.envs.minigrid_wrapper import MiniGridWrapper

# ------------------------------------
#  Utilities
# ------------------------------------
def extract_frame(env, last_obs):
    """Extract a proper RGB frame from MiniGrid (Gymnasium 0.29+).
    MiniGrid now returns render() as either RGB array or [RGB array].
    """
    frame = env.render()
    if isinstance(frame, list) and len(frame) > 0:
        frame = frame[0]
    return frame

# ------------------------------------
#  Arg parsing
# ------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env_id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--num_skills", type=int, default=8)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="visualizations")
    parser.add_argument("--render_mode", type=str, default="rgb_array")
    return parser.parse_args()

# ------------------------------------
#  Load agent safely
# ------------------------------------
def load_agent(checkpoint_path, env):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Fallback config
    config = checkpoint.get("config", {})
    if "agent" not in config:
        config = {
            "agent": {
                "obs_shape": env.observation_space["observation"].shape,
                "action_dim": env.action_space.n,
                "skill_dim": 8,
            }
        }

    agent = DIAYNAgent(config["agent"])

    # State dict loading
    if "model_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["state_dict"])
    else:
        raise ValueError("Checkpoint missing model state dict.")

    agent.eval()
    return agent

# ------------------------------------
#  Visualization (main logic)
# ------------------------------------
def visualize_skills(agent, env, args):
    os.makedirs(args.save_dir, exist_ok=True)
    skills = torch.eye(agent.skill_dim, device="cpu")

    for skill_idx in range(min(args.num_skills, agent.skill_dim)):
        skill = skills[skill_idx]

        for ep in range(args.num_episodes):
            obs, _ = env.reset(skill=skill.cpu().numpy())

            frames = []
            frame = extract_frame(env, obs)
            frames.append(frame)

            for step in range(args.max_steps):
                obs_tensor = torch.FloatTensor(obs["observation"]).unsqueeze(0)

                with torch.no_grad():
                    logits = agent.forward(obs_tensor, skill.unsqueeze(0))
                    action = logits.argmax(dim=-1).item()

                obs, reward, terminated, truncated, info = env.step(action)

                frame = extract_frame(env, obs)
                frames.append(frame)

                if terminated or truncated:
                    break

            # Save GIF
            out_path = os.path.join(args.save_dir, f"skill_{skill_idx}_ep_{ep}.gif")
            imageio.mimsave(out_path, frames, fps=6)
# ------------------------------------
#  Main
# ------------------------------------
def main():
    args = parse_args()

    # Create proper MiniGrid env
    env = gym.make(args.env_id, render_mode="rgb_array")
    env = MiniGridWrapper(env, skill_dim=args.num_skills, obs_type="rgb")

    agent = load_agent(args.checkpoint, env)

    with torch.no_grad():
        visualize_skills(agent, env, args)

    env.close()

if __name__ == "__main__":
    main()