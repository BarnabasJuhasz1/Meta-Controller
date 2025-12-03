
import gymnasium as gym
import minigrid
import torch
import yaml
import numpy as np
import time
from models import Agent
from wrappers import HybridWrapper

def visualize():
    with open("config.yaml", "r") as f: cfg = yaml.safe_load(f)
    env = gym.make(cfg['env']['id'], render_mode="human")
    env = HybridWrapper(env)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Need dummy setup to load weights
    agent = Agent(env.action_space.n, cfg['agent']['skill_dim']).to(device)
    agent.load_state_dict(torch.load("continual_agent.pth", map_location=device))
    
    print(f"Loaded Agent. Testing {cfg['agent']['skill_dim']} Skills.")
    
    while True:
        try:
            val = input("Enter Skill ID (0-7): ")
            idx = int(val)
        except: continue
            
        print(f"Playing Skill {idx}...")
        skill = torch.zeros(1, cfg['agent']['skill_dim']).to(device)
        skill[0][idx] = 1.0
        
        obs, _ = env.reset()
        for _ in range(300):
            env.render()
            img = torch.Tensor(obs['image']).unsqueeze(0).to(device)
            state = torch.Tensor(obs['state']).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # deterministic=False allows natural movement (sampling)
                action, _, _, _ = agent.get_action_and_value(img, state, skill)
                
            obs, _, term, trunc, _ = env.step(action.item())
            time.sleep(0.03)
            
            if term or trunc: break
    
    env.close()

if __name__ == "__main__":
    visualize()
