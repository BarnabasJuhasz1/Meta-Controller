import gymnasium as gym
import minigrid
import torch
import numpy as np
import time
import yaml
from wrappers import HybridObservationWrapper
from agent import DIAYNAgent

def visualize():
    # Load Config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    env_name = cfg['env']['id']
    num_skills = cfg['agent']['skill_dim']
    
    # Setup Env with Human Render
    env = gym.make(env_name, render_mode="human", max_episode_steps=500)
    env = HybridObservationWrapper(env)
    
    # Load Agent
    agent = DIAYNAgent(env, cfg)
    try:
        agent.load("diayn_doorkey.pth")
        print("✅ Model loaded successfully.")
    except:
        print("❌ Model not found. Run train.py first.")
        return

    print(f"\n--- Visualizing {num_skills} Skills ---")
    print("Each skill should show a distinct behavior (e.g., Go Top-Left, Pick Key, etc.)\n")

    for skill_idx in range(num_skills):
        print(f"▶️ Running SKILL {skill_idx}")
        skill_vec = np.zeros(num_skills, dtype=np.float32)
        skill_vec[skill_idx] = 1.0
        
        obs, _ = env.reset()
        
        for _ in range(200):
            env.render()
            action, _ = agent.get_action(obs, skill_vec, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            time.sleep(0.05)
            
            if term or trunc:
                break
        
        time.sleep(1.0)
    
    env.close()

if __name__ == "__main__":
    visualize()
