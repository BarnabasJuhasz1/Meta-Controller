import gymnasium as gym
import minigrid
import yaml
import numpy as np
import torch
from tqdm import tqdm
from wrappers import HybridObservationWrapper
from agent import DIAYNAgent

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def train():
    cfg = load_config()
    env_name = cfg['env']['id']
    print(f"🚀 Starting DIAYN Training on {env_name}")
    
    # Init Env
    env = gym.make(env_name, max_episode_steps=cfg['env']['max_steps'], render_mode=None)
    env = HybridObservationWrapper(env)
    
    # Init Agent
    agent = DIAYNAgent(env, cfg)
    
    # Training Loop
    pbar = tqdm(range(cfg['training']['max_episodes']))
    metrics = {"rew": 0.0, "loss": 0.0}
    
    for ep in pbar:
        # 1. Sample a Skill z ~ p(z)
        skill_idx = np.random.randint(0, cfg['agent']['skill_dim'])
        skill_vec = np.zeros(cfg['agent']['skill_dim'], dtype=np.float32)
        skill_vec[skill_idx] = 1.0
        
        # 2. Run Episode
        obs, _ = env.reset()
        memory = []
        done = False
        
        while not done:
            action, log_prob = agent.get_action(obs, skill_vec)
            next_obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
            
            # Store transition
            memory.append({
                "img": obs['image'],
                "state": obs['state'],
                "skill": skill_vec,
                "action": action,
                "log_prob": log_prob
            })
            obs = next_obs
        
        # 3. Update (PPO Style - Update at end of episode)
        # Only update if we have enough data (MiniGrid eps can be short)
        if len(memory) > 10:
            r, l = agent.update(memory)
            metrics = {"rew": r, "loss": l}
        
        # 4. Logging
        if ep % cfg['training']['log_interval'] == 0:
            pbar.set_description(f"Ep {ep} | IntRew: {metrics['rew']:.3f} | DiscLoss: {metrics['loss']:.3f}")
            
        if ep % cfg['training']['save_interval'] == 0:
            agent.save("diayn_doorkey.pth")
            
    agent.save("diayn_doorkey.pth")
    print("✅ Training Complete.")

if __name__ == "__main__":
    train()
