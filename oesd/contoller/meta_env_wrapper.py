import gymnasium as gym
from gymnasium import spaces
import numpy as np
import gym
import akro
import matplotlib.pyplot as plt
import random

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.wrappers import FullyObsWrapper
from minigrid.envs import DoorKeyEnv

class MetaControllerEnv(gym.Env):
    def __init__(
        self,
        skill_registry,        
        env_name="minigrid_small",
        skill_duration=10,
        max_steps=None,
        action_scale=1.0,
        render_mode="rgb_array",
    ):
        super().__init__()
        if env_name == "minigrid_small":
            # DoorKeyEnv-5x5 (there is also size 6, 8, and 16)
            self._env = DoorKeyEnv(size=8, max_steps=max_steps, render_mode=render_mode)
        else:
            raise ValueError(f"Unknown environment name: {env_name}")
        self.registry = skill_registry
        self.skill_duration = skill_duration # "How long selected skill should run" 
        self._max_steps = max_steps or self._env.max_steps
        self._action_scale = action_scale

        # Action Space: Select one of the frozen skills
        # self.action_space = spaces.Discrete(self.registry.num_actions)
        self._discrete_actions = [
            self._env.actions.forward,
            self._env.actions.left,
            self._env.actions.right,
            self._env.actions.pickup,
            self._env.actions.drop,
            self._env.actions.toggle,
        ]
        self._num_actions = len(self._discrete_actions)
        self.action_space = akro.Box(
            low=-action_scale,
            high=action_scale,
            shape=(self._num_actions,),
            dtype=np.float32,
        )

        self.env_name = env_name

        reset_result = self._env.reset()
        if isinstance(reset_result, tuple):
            initial_obs, _ = reset_result
        else:
            initial_obs = reset_result
        sample_obs = self._process_obs(initial_obs)
        self.observation_space = akro.Box(
            low=0.0,
            high=1.0,
            shape=sample_obs.shape,
            dtype=np.float32,
        )
        self._last_obs = sample_obs

    # def reset(self, seed=None, options=None):
    #     return self._env.reset(seed=seed, options=options)

    def reset(self, **kwargs):
        result = self._env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, _ = result
        else:
            obs = result
        self._last_obs = self._process_obs(obs)
        return self._last_obs

    def _map_action(self, action):
        if action.ndim == 0:
            return int(action)
        return int(np.argmax(action))

    def _process_obs(self, obs):
        if isinstance(obs, tuple):
            obs = obs[0]
        
        # 1. Flatten and normalize image
        image = obs["image"].astype(np.float32) / 255.0
        
        # 2. Normalize direction (0-3 becomes 0.0-1.0)
        direction = np.array([obs["direction"] / 3.0], dtype=np.float32)

        # 3. Add Carrying (Binary)
        # Accessing .unwrapped is safer in case of other wrappers
        # env_base = self.env.unwrapped 
        carrying_val = 1.0 if self._env.carrying is not None else 0.0
        carrying = np.array([carrying_val], dtype=np.float32)

        # 4. Add Agent Position (Normalized)
        # We divide by width/height to keep inputs within [0, 1] range
        agent_x = self._env.agent_pos[0] / self._env.width
        agent_y = self._env.agent_pos[1] / self._env.height
        position = np.array([agent_x, agent_y], dtype=np.float32)

        # debug print
        # if carrying_val > 0:
        #     print(f"AGENT CARRYING at {self._env.agent_pos}")
        
        # Concatenate everything: [Image flat, Direction, Carrying, PosX, PosY]
        return np.concatenate([
            image.flatten(), 
            direction, 
            carrying, 
            # position
        ], axis=0)


    def step(self, global_skill_idx, render=False):
        """
        The Meta-Step: Execute one skill for k steps.
        """
        total_reward = 0
        terminated = False
        truncated = False

        # 1. DECODE: Retrieve the specific model and z vector
        # This handles the "Selection of skill & model" automatically
        policy_net, z_vector = self.registry.get_skill(global_skill_idx)

        # --- The Scheduler Loop  ---
        for _ in range(self.skill_duration):
            # 1. Get current observation from environment
            # Note: We need the LAST observation to query the policy
            # (In a real implementation, cache 'obs' from the loop start)
            current_obs = self._process_obs(self.last_obs)

            # 2. Ask the specific sub-skill for a primitive action
            # We use the current observation (self.last_obs) 
            # primitive_action = self.registry.get_action(action, self.last_obs)

            # Get primitive action from the selected model
            # Note: Ensure your policy_net can handle the observation format
            with torch.no_grad():
                primitive_action = policy_net.get_action(current_obs, z_vector)
            
            # 3. Step the physical environment
            obs, reward, terminated, truncated, info = self._env.step(primitive_action)
            
            if render:
                frame = self._env.render()
                if frame is not None:
                    info['render'] = frame.transpose(2, 0, 1)

            self.last_obs = obs # Update for next micro-step
            
            total_reward += reward
            
            # If the task is solved or failed during the skill, stop early
            if terminated or truncated:
                break
                
        # Return the aggregated experience to the Meta-Controller
        return obs, total_reward, terminated, truncated, info

    # Helper to capture the obs for the registry
    # def reset(self, **kwargs):
    #     obs, info = self._env.reset(**kwargs)
    #     self.last_obs = obs
    #     return obs, info

    def render(self, mode="rgb_array", **kwargs):
        # --- FIX START: Robust Render ---
        # Try standard render
        frame = self._env.render()
        
        # If frame is None (likely due to render_mode mismatch or gym version issues), 
        # force generation of the frame using the minigrid method directly.
        if frame is None:
            if hasattr(self._env, 'get_frame'):
                # Standard tile_size for minigrid is 32
                frame = self._env.get_frame(highlight=False, tile_size=32)
            elif hasattr(self._env.unwrapped, 'get_frame'):
                frame = self._env.unwrapped.get_frame(highlight=False, tile_size=32)
        
        return frame

    def render_trajectories(self, trajectories, colors, plot_axis, ax):
        for trajectory, color in zip(trajectories, colors):
            coords = trajectory["env_infos"]["coordinates"]
            if coords.ndim == 1:
                coords = coords[None, :]
            ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=0.8)
            ax.scatter(coords[-1, 0], coords[-1, 1], color=color, marker="*", edgecolors="black")
        if plot_axis is not None:
            ax.axis(plot_axis)
        else:
            ax.set_aspect("equal")

    def draw(self, ax=None):
        if ax is None:
            ax = plt.gca()
        frame = self._env.render()
        if frame is None:
            return ax
        ax.imshow(frame)
        ax.set_axis_off()
        return ax