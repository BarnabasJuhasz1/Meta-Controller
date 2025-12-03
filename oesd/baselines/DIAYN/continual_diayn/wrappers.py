
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict

class HybridWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.width = env.unwrapped.width
        self.height = env.unwrapped.height
        
        self.observation_space = Dict({
            "image": Box(0, 255, (3, 7, 7), dtype=np.float32),
            "state": Box(0, 1, (3,), dtype=np.float32) # x, y, key
        })

    def observation(self, obs):
        # Image: Channel First
        img = obs['image'].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        
        # State: [X, Y, HasKey]
        pos = self.unwrapped.agent_pos
        carrying = 1.0 if self.unwrapped.carrying is not None else 0.0
        state = np.array([pos[0]/self.width, pos[1]/self.height, carrying], dtype=np.float32)
        
        return {"image": img, "state": state}
