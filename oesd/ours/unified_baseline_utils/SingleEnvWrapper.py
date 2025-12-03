# SingleEnvWrapper.py
from __future__ import annotations
import gymnasium as gym
from typing import Any, Tuple
import numpy as np
import torch
from collections import deque
import random
from enviroments.example_minigrid import SimpleEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import akro


class SingleEnvWrapper:
   
    def __init__(
        self,
        env,  
        baseline = "lsd",
        skill_dim = 8,
        obs_type= "position",  #"position", "rgb", "grid", "full"
        normalize_obs= False,
        max_steps= 200,
        action_scale= 1.0,
        render_mode= None,
    ):
        self.env = env
        self.baseline = baseline.lower()
        self.skill_dim = skill_dim
        self.obs_type = obs_type
        self.normalize_obs = normalize_obs
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.render_mode = render_mode
        
    
        self.current_state = None
        self.current_skill = None
        self.step_count = 0
        
        # Setup based on baseline
        self._setup_baseline()
        
    def _setup_baseline(self):
       
        # Common MiniGrid discrete actions
        self._discrete_actions = [
            self.env.actions.forward,    # 0
            self.env.actions.left,       # 1
            self.env.actions.right,      # 2
            self.env.actions.pickup,     # 3
            self.env.actions.drop,       # 4
            self.env.actions.toggle,     # 5
        ]
        
        # Define action space based on baseline
        if self.baseline in ["lsd", "metra"]:
            # Discrete action space
            self.action_space = gym.spaces.Discrete(len(self._discrete_actions))
        elif self.baseline in ["rsd", "dyan", "dads"]:
            # Continuous action space (common in these baselines)
            self.action_space = akro.Box(
                low=-self.action_scale,
                high=self.action_scale,
                shape=(len(self._discrete_actions),),
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unknown baseline: {self.baseline}")
        
        # Define observation space based on baseline and obs_type
        obs_shape = self._get_obs_shape()
        self.observation_space = akro.Box(
            low=0.0,
            high=1.0 if self.normalize_obs else 255.0,
            shape=obs_shape,
            dtype=np.float32,
        )
        
        # Skill space for skill-conditioned baselines
        if self.baseline in ["dyan", "dads", "rsd"]:
            self.skill_space = akro.Box(
                low=-1.0,
                high=1.0,
                shape=(self.skill_dim,),
                dtype=np.float32,
            )
        else:
            self.skill_space = None
    
    def _get_obs_shape(self):
        
        if self.obs_type == "position":
            return (2,)  # (x, y) coordinates
        elif self.obs_type == "rgb":
            # MiniGrid image is 7x7x3 (H, W, C) for RGB
            return (7, 7, 3)
        elif self.obs_type == "grid":
            # Grid encoding only (7x7)
            return (7, 7)
        elif self.obs_type == "full":
            # Full state: image + direction + carrying + position
            return (7 * 7 * 3 + 1 + 1 + 2,)
        else:
            raise ValueError(f"Unknown obs_type: {self.obs_type}")
    
    def _extract_state(self, obs):
        
        if self.obs_type == "position":
            # Return agent position
            x, y = self.env.agent_pos
            state = np.array([x, y], dtype=np.float32)
            
        elif self.obs_type == "rgb":
            # Return RGB image
            state = obs["image"][..., :3].astype(np.float32)
            
        elif self.obs_type == "grid":
            # Return grid encoding
            state = obs["image"][..., 0].astype(np.float32)
            
        elif self.obs_type == "full":
            # Full state information
            image = obs["image"].astype(np.float32).flatten() / 255.0
            direction = np.array([obs["direction"] / 3.0], dtype=np.float32)
            carrying = np.array([1.0 if self.env.carrying is not None else 0.0], dtype=np.float32)
            x, y = self.env.agent_pos
            position = np.array([x / self.env.width, y / self.env.height], dtype=np.float32)
            state = np.concatenate([image, direction, carrying, position])
        
        # Normalize if requested
        if self.normalize_obs:
            if self.obs_type == "position":
                # Normalize position to [0, 1]
                state = state / np.array([self.env.width, self.env.height])
            elif self.obs_type in ["rgb", "grid"]:
                state = state / 255.0
        
        return state
    
    def _map_action(self, action):
        if isinstance(action, np.ndarray):
            if self.baseline in ["rsd", "dyan", "dads"]:
               
                return int(np.argmax(action))
            else:
                # Already discrete but in array form
                return int(action)
        return int(action)
    
    def reset(
        self, 
        skill= None,
        **kwargs
    ):
 
        # Reset base environment
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        
        # Extract state
        self.current_state = self._extract_state(obs)
        self.step_count = 0
        
        # Set skill if provided
        if skill is not None:
            self.current_skill = skill
        elif self.current_skill is None and self.skill_space is not None:
            # Initialize random skill
            self.current_skill = np.random.uniform(-1, 1, self.skill_dim).astype(np.float32)
        
        # Return format based on baseline
        if self.baseline in ["dyan", "rsd"]:
            # Return dict with observation and skill
            return {"observation": self.current_state, "skill": self.current_skill}, info
        elif self.baseline == "dads":
            # DADS expects separate state and skill
            return self.current_state, info
        else:
            # LSD and METRA just return state
            return self.current_state, info
    
    def step(
        self, 
        action,
        skill,
        render= False
    ) :
        
        # Update skill if provided
        if skill is not None:
            self.current_skill = skill
        
        # Map action to discrete
        discrete_action = self._discrete_actions[self._map_action(action)]
        
        # Record position before step
        coord_before = np.array(self.env.agent_pos, dtype=np.float32)
        
        # Take step
        result = self.env.step(discrete_action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        
        # Extract next state
        next_state = self._extract_state(obs)
        
        # Record position after step
        coord_after = np.array(self.env.agent_pos, dtype=np.float32)
        
        # Update state
        self.current_state = next_state
        self.step_count += 1
        
        # Create info dict with additional data
        info = dict(info)
        info.update({
            "coordinates": coord_before,
            "next_coordinates": coord_after,
            "skill": self.current_skill.copy() if self.current_skill is not None else None,
            "step": self.step_count,
        })
        
        # Add render frame if requested
        if render and hasattr(self.env, 'render'):
            frame = self.env.render()
            if frame is not None:
                info['render'] = frame
        
        # Check max steps
        if self.step_count >= self.max_steps:
            done = True
            info["TimeLimit.truncated"] = True
        
        # Format return based on baseline
        if self.baseline in ["dyan", "rsd"]:
            next_obs = {"observation": next_state, "skill": self.current_skill}
        elif self.baseline == "dads":
            # For DADS training loop compatibility
            info['s_old'] = self.current_state  # Previous state
            info['s_next'] = next_state  # Next state
            next_obs = next_state
        else:
            # LSD and METRA
            next_obs = next_state
        
        return next_obs, reward, done, info
    
    def get_agent_pos(self):
        
        return np.array(self.env.agent_pos, dtype=np.float32)
    
    def get_state(self):
        
        return self.current_state.copy() if self.current_state is not None else None
    
    def get_skill(self):
       
        return self.current_skill.copy() if self.current_skill is not None else None
    
    def set_skill(self, skill):
        
        if skill.shape != (self.skill_dim,):
            raise ValueError(f"Skill must have shape ({self.skill_dim},), got {skill.shape}")
        self.current_skill = skill.copy()
    
    def sample_skill(self):
        
        if self.skill_space is None:
            raise ValueError("No skill space defined for this baseline")
        return np.random.uniform(-1, 1, self.skill_dim).astype(np.float32)
    
    def render(self, mode= "rgb_array", **kwargs):
        
        if hasattr(self.env, 'render'):
            return self.env.render(mode=mode, **kwargs)
        return None
    
    def close(self):
        
        if hasattr(self.env, 'close'):
            self.env.close()


    def step_metra(self, action):
        
        s_old = self.current_state.copy()
        next_obs, reward, done, info = self.step(action)
        return s_old, self.current_state, done, info
    
    def step_dads(self, action, skill):
        
        self.set_skill(skill)
        return self.step_metra(action)


class SingleEnvWrapperOld(gym.Wrapper):
    """
    A unified environment wrapper that connects:
        - the environment
        - the algorithm-specific adapter
        - the algorithm-specific model

    For each step:
        raw_obs  --> adapter.preprocess_observation(...)
                 --> adapter.get_action(model, obs_vec, skill)
                 --> primitive env action

    This keeps *all algorithm logic* out of the environment loop.
    """

    def __init__(self, env: gym.Env, model: Any, adapter):
        super().__init__(env)
        self.model = model
        self.adapter = adapter
        self._last_obs = None      # raw observation dict
        self._last_info = None

    # ===================================================================
    # STANDARD GYM API
    # ===================================================================

    def reset(self, **kwargs) -> Tuple[Any, dict]:
        """
        Reset the environment normally.

        Gymnasium returns: (obs, info)
        Gym returns: obs  (we convert it to (obs, {}))
        """
        out = self.env.reset(**kwargs)

        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:  # old gym compat
            obs, info = out, {}

        self._last_obs = obs
        self._last_info = info
        return obs, info

    # -------------------------------------------------------------------

    def step(self, action) -> Tuple[Any, float, bool, bool, dict]:
        """
        Standard step that preserves Gymnasium’s output format:

            obs, reward, terminated, truncated, info
        """
        out = self.env.step(action)

        if len(out) == 5:  # Gymnasium
            obs, reward, terminated, truncated, info = out
        else:  # old Gym
            obs, reward, done, info = out
            terminated = done
            truncated = False

        self._last_obs = obs
        self._last_info = info

        return obs, reward, terminated, truncated, info

    # ===================================================================
    # MODEL-DRIVEN STEP
    # ===================================================================

    def step_with_model(
        self,
        skill_z,
        deterministic: bool = False
    ):
        """
        Compute action via the adapter, then step environment:

            obs_vec = adapter.preprocess_observation(_last_obs)
            action  = adapter.get_action(model, obs_vec, skill_z)
            obs, reward, ter, trunc, info = env.step(action)

        Returns:
            obs, reward, terminated, truncated, info, action
        """
        if self._last_obs is None:
            raise RuntimeError(
                "step_with_model() called before reset(). Call env.reset() first."
            )

        # 1. Convert raw MiniGrid obs → model input vector
        # obs_vec = self.adapter.preprocess_observation(self._last_obs)

        # 2. Query the algorithm for primitive action
        action = self.adapter.get_action(
            self.model,
            # obs_vec,
            self._last_obs,
            skill_z,
            deterministic=deterministic,
        )

        # 3. Execute in environment
        obs, reward, terminated, truncated, info = self.step(action)

        setattr(self.env, "_last_action", int(action))

        return obs, reward, terminated, truncated, info, action
