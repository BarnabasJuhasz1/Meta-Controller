import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import ImgObsWrapper

class HybridObservationWrapper(gym.ObservationWrapper):
    """
    Splits observation into:
    1. 'image': 7x7x3 Grid View (For CNN)
    2. 'state': [x, y, carrying] (For Coordinate Logic)
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # Access underlying Minigrid properties
        self.width = env.unwrapped.width
        self.height = env.unwrapped.height
        
        # Define complex space
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, (3, 7, 7), dtype=np.float32),
            "state": gym.spaces.Box(0, 1, (3,), dtype=np.float32)
        })

    def observation(self, obs):
        # 1. Process Image (Channels First for PyTorch: C, H, W)
        img = obs['image'].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) # (7,7,3) -> (3,7,7)
        
        # 2. Process State
        pos = self.unwrapped.agent_pos
        x = pos[0] / self.width
        y = pos[1] / self.height
        carrying = 1.0 if self.unwrapped.carrying is not None else 0.0
        
        state_vec = np.array([x, y, carrying], dtype=np.float32)
        
        return {
            "image": img,
            "state": state_vec
        }
