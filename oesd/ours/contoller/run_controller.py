from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry
from oesd.ours.contoller.meta_env_wrapper import MetaControllerEnv
from oesd.ours.unified_baseline_utils.SingleLoader import load_model_from_config, load_config

import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default="minigrid")
parser.add_argument("--skill_count_per_algo", type=int, default=10)
parser.add_argument("--skill_duration", type=int, default=10)
parser.add_argument("--config_path", type=str, default="ours/configs/config1.py")
parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model .zip file")
parser.add_argument("--num_episodes", type=int, default=5)

def main(_A: argparse.Namespace):

    # initialize skill registry
    skill_registry = SkillRegistry(_A.skill_count_per_algo)

    # load model configs from config file
    config = load_config(_A.config_path)
    
    # load models via adapters (while feeding skill_registry to adapters)
    adapters = [load_model_from_config(m, skill_registry=skill_registry) for m in config.model_cfgs]
    model_interfaces = {adapter.algo_name: adapter for adapter in adapters}

    # initialize environment with human rendering
    # Note: MetaControllerEnv handles the render_mode internally if passed to the base env, 
    # but here we might need to rely on the render() call in the loop.
    # Let's check how MetaControllerEnv handles rendering.
    # It calls self._env.render() if render=True in step().
    
    meta_env = MetaControllerEnv(skill_registry, model_interfaces, env_name=_A.env_name, skill_duration=_A.skill_duration, render_mode="human")

    # Load the trained model
    print(f"Loading model from {_A.model_path}...")
    model = PPO.load(_A.model_path)

    print("Starting execution...")
    for episode in range(_A.num_episodes):
        obs, info = meta_env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        print(f"Episode {episode + 1}/{_A.num_episodes}")
        
        while not (terminated or truncated):
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            # We pass render=True to see the environment
            obs, reward, terminated, truncated, info = meta_env.step(action, render=True)
            
            total_reward += reward
            step_count += 1
            
            # Optional: Add a small sleep to make it watchable if it's too fast
            # time.sleep(0.1) 
            
            if "render" in info:
                # If the environment returns a frame in info, we might want to display it
                # But usually render=True in step() handles the window display for human mode
                pass

        print(f"Episode finished. Total Reward: {total_reward}, Steps: {step_count}")

    print("Execution finished.")

if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
