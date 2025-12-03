import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MiniGridWrapper(gym.Wrapper):
    """Wrapper for the MiniGrid environment"""
    def __init__(self,env,skill_dim=8,obs_type="rgb"):
        super().__init__(env)
        self.skill_dim = skill_dim
        self.obs_type = obs_type
        
        if obs_type == "rgb":
            self.obs_shape = (7,7,3)
        else: #obs_type = grid
            self.obs_shape = (7,7)
        
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=0,high=255,
                shape=self.obs_shape,
                dtype=np.uint8
            ),
            "skill": spaces.Box(
                low=-1.0,high=1.0,
                shape=(skill_dim,),
                dtype=np.float32
            )
        })
    
    def reset(self, skill=None, **kwargs):
        obs, info = super().reset(**kwargs)

        if skill is None:
            skill = np.zeros(self.skill_dim, dtype=np.float32)
        self.current_skill = skill

        return self._process_obs(obs, skill), info

    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_obs(obs, self.current_skill), reward, terminated, truncated, info

    
    def _process_obs(self, obs, skill=None):
        if self.obs_type == "rgb":
            obs_array = obs["image"][...,:3]
        else:
            obs_array = obs["image"][...,0]

        # Use provided skill, otherwise keep previous one
        if skill is None:
            skill = self.current_skill

        return {
            "observation": obs_array,
            "skill": skill
        }

        

class ObservationExtractor(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space['observation']
        
    def observation(self, obs):
        return obs['observation']
        