import gymnasium as gym
# from SimpleEnv import example_minigrid
from example_minigrid import SimpleEnv
import numpy as np
import torch.nn as nn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random



def extract_state(obs, env):
    

    x, y = env.env.agent_pos
    return np.array([x, y], dtype=np.float32)

class MetraWrapper:
    def __init__(self, env):
        self.env = env
        self.s = None
        self.reset()

    def reset(self):
        obs, info = self.env.reset()
        self.s = extract_state(obs, self)
        return self.s

    def step(self, a):
        obs_next, reward, terminated, truncated, info = self.env.step(a)
        s_next = extract_state(obs_next, self)
        old_s = self.s
        self.s = s_next
        done = terminated or truncated
        return old_s, s_next, done, info


class metraTester:
    def __init__(self,env, phi, policy):
        self.env = MetraWrapper(env) 
        self.policy=policy
    def test(self,render=False):
        z = torch.tensor(self.sample_skill(), dtype=torch.float32).unsqueeze(0)

        

        if render:
            obs, _ = self.env.env.reset()
            done = False
            while not done:
                self.env.env.render()
                s_tensor = torch.tensor(self.env.s, dtype=torch.float32).unsqueeze(0)
                a = self.policy.sample_action(s_tensor, z)
                _, _, done, _ = self.env.step(a)

        else:
            s = self.env.reset()
            done = False
            trajectory = [s.copy()]

            while not done:
                s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                a = self.policy.sample_action(s_tensor, z)
                s, s_next, done, _ = self.env.step(a)
                trajectory.append(s_next.copy())
                s = s_next

                print("Trajectory:", trajectory)

        return
    
    def sample_skill(self,dim=2):
        z = np.random.randn(dim).astype(np.float32)
        return z / np.linalg.norm(z)