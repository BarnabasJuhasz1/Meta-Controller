import gymnasium as gym
from gymnasium import spaces
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import akro
import matplotlib.pyplot as plt
import random
import torch
import torch

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
        model_interfaces,       
        env_name="minigrid",
        skill_duration=10,
        max_steps=None,
        action_scale=1.0,
        render_mode="rgb_array",
        **kwargs
    ):
        super().__init__()
        if env_name == "minigrid":
            # DoorKeyEnv-5x5 (there is also size 6, 8, and 16)
            self._env = DoorKeyEnv(size=8, max_steps=max_steps, render_mode=render_mode)
        else:
            raise ValueError(f"Unknown environment name: {env_name}")
        self.registry = skill_registry
        self.skill_duration = skill_duration # "How long selected skill should run" 
        self._max_steps = max_steps or self._env.max_steps
        self._action_scale = action_scale
        self.model_interfaces = model_interfaces
        
        # Reward shaping parameters
        self.key_pickup_reward = kwargs.get("key_pickup_reward", 0.0)
        self.door_open_reward = kwargs.get("door_open_reward", 0.0)
        self.key_drop_reward = kwargs.get("key_drop_reward", 0.0)
        self.not_move_reward = kwargs.get("not_move_reward", 0.0)

        # Visualization tracking
        self.current_algo = "None"
        self.current_global_skill = -1
        self.current_local_skill = -1
        self.current_step_in_skill = 0
        self.render_mode = render_mode

        self._discrete_actions = [
            self._env.actions.forward,
            self._env.actions.left,
            self._env.actions.right,
            self._env.actions.pickup,
            self._env.actions.drop,
            self._env.actions.toggle,
        ]
        # self._num_actions = len(self._discrete_actions)
        # self.action_space = spaces.Box(
        #     low=-action_scale,
        #     high=action_scale,
        #     shape=(self._num_actions,),
        #     dtype=np.float32,
        # )

        base_env = self._env.unwrapped  # accessing the core environment behind all wrappers

        # Try printing these common attributes to see which one exists:
        # try:
        #     print(f"Grid Size: {base_env.width} x {base_env.height}") # Common in MiniGrid
        # except AttributeError:
        #     pass

        # try:
        #     print(f"Grid Shape: {base_env.grid_size}") # Common in some custom envs
        # except AttributeError:
        #     pass

        # try:
        #     print(f"Arena Size: {base_env.arena.width} x {base_env.arena.height}") # Common in continuous envs
        # except AttributeError:
        #     pass

        self.action_space = spaces.Discrete(len(self.registry.bag_of_skills))

        self.env_name = env_name

        reset_result = self._env.reset()
        if isinstance(reset_result, tuple):
            initial_obs, _ = reset_result
        else:
            initial_obs = reset_result
        sample_obs = self._process_obs(initial_obs)
        self.observation_space = spaces.Box(
            # low=0.0,
            # high=1.0,
            low = -np.inf,
            high= np.inf,
            shape=sample_obs.shape, # this should be the largest obs shape of all algorithms
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
        self._last_raw_obs = obs
        
        # Reset reward flags
        self._rewarded_key = False
        self._rewarded_door = False
        
        # Reset tracking
        self.current_algo = "None"
        self.current_global_skill = -1
        self.current_local_skill = -1
        self.current_step_in_skill = 0
        
        
        info = {}
        return self._process_obs(obs), info

    def _map_action(self, action):
        if isinstance(action, int):
            return action
        if action.ndim == 0:
            return int(action)
        return int(np.argmax(action))


    def _process_obs(self, obs):
        """
            just a process observation wrapper,
            each algorithm implements its own expected way of processing observations
            
        """
        
        if self.current_algo == "None":
            # return self._process_obs_for_meta_controller(obs)
            return self.model_interfaces['RSD'].process_obs(obs, self._env)

        processed_obs = self.model_interfaces[self.current_algo].process_obs(obs, self._env)

        # returned SHAPE should always be (149,)
        current_size = processed_obs.shape[0]
        padding = np.zeros(149-current_size, dtype=np.float32)
        return np.concatenate((processed_obs, padding))

    def _process_obs_for_meta_controller(self, obs):
        pass

    def step(self, global_skill_idx, render=False):
        """
        The Meta-Step: Execute one skill for k steps.
        """
        total_reward = 0
        terminated = False
        truncated = False

        # 1. DECODE: Retrieve the specific model and z vector
        # This handles the "Selection of skill & model" automatically
        algo_name, z_vector = self.registry.get_algo_and_skill_from_skill_idx(global_skill_idx)
        
        # Update tracking info
        self.current_algo = algo_name
        self.current_global_skill = global_skill_idx
        # Calculate local skill index: global_idx % skills_per_algo
        # Assuming skills are registered in blocks of equal size per algo as per registry implementation
        self.current_local_skill = global_skill_idx % self.registry.skill_count_per_algo
        self.current_step_in_skill = 0

        # --- The Scheduler Loop  ---
        frames = []
        for step_i in range(self.skill_duration):
            self.current_step_in_skill = step_i + 1
            
            # 1. Get current observation from environment
            # Note: We need the LAST observation to query the policy
            # (In a real implementation, cache 'obs' from the loop start)
            current_obs = self._process_obs(self._last_raw_obs)

            # 2. Ask the specific sub-skill for a primitive action
            # We use the current observation (self.last_obs) 
            # primitive_action = self.registry.get_action(action, self.last_obs)
            with torch.no_grad():
                if self.current_algo == "DIAYN":
                    primitive_action = self.model_interfaces[self.current_algo].get_action(self._last_raw_obs, z_vector)
                else:
                    primitive_action = self.model_interfaces[self.current_algo].get_action(current_obs, z_vector)
                    # if self.current_algo == "RSD":
                        # print(f"RSD IS USED WITH PARAMS: {self.current_local_skill}, {z_vector}, action: {primitive_action}")

            # Update title before step if human rendering, because minigrid might render in step
            if self.render_mode == "human":
                self._update_window_title()

            # 3. Step the physical environment
            
            # Check if we are carrying a key BEFORE the step
            was_carrying_key = isinstance(self._env.carrying, Key)
            # Capture agent position before the step
            prev_pos = self._env.agent_pos

            if self.current_algo == "RSD":
                mapped_action = self._discrete_actions[self._map_action(np.asarray(primitive_action))]
            else:
                mapped_action = self._map_action(primitive_action)
            
            obs, reward, terminated, truncated, info = self._env.step(mapped_action)

            # Check if we are carrying a key AFTER the step
            is_carrying_key = isinstance(self._env.carrying, Key)
            # Apply penalty if agent did not move (position unchanged)
            if prev_pos == self._env.agent_pos:
                reward += self.not_move_reward

            # If we were carrying a key, and now we are not, AND the action was DROP
            # then we apply the penalty (reward is usually negative)
            if was_carrying_key and not is_carrying_key:
                if mapped_action == self._env.actions.drop:
                    reward += self.key_drop_reward
                    # print(f"KEY DROPPED! Penalty applied: {self.key_drop_reward}")

            # --- Reward Shaping ---
            # Check for Key Pickup
            if not self._rewarded_key and self._env.carrying is not None:
                if isinstance(self._env.carrying, Key):
                    reward += self.key_pickup_reward
                    self._rewarded_key = True
            
            # Check for Door Opening
            if not self._rewarded_door:
                # Find the door in the grid
                # Note: This iterates over the grid, which is small (8x8), so it's cheap.
                # If performance is critical, we could cache the door object or position.
                for obj in self._env.grid.grid:
                    if isinstance(obj, Door) and obj.is_open:
                        reward += self.door_open_reward
                        self._rewarded_door = True
                        break

            
            if render:
                # Use our custom render to update title
                frame = self.render()
                if frame is not None:
                    # Capture every frame
                    # Store as CHW for consistency with previous single-frame behavior
                    frames.append(frame.transpose(2, 0, 1))

            self._last_raw_obs = obs # Update for next micro-step
            
            total_reward += reward
            
            # If the task is solved or failed during the skill, stop early
            if terminated or truncated:
                break
            
        # Attach collected frames to the final info dict
        if render and frames:
            info['render'] = frames
            
        # Return the aggregated experience to the Meta-Controller
        return self._process_obs(self._last_raw_obs), total_reward, terminated, truncated, info

    # Helper to capture the obs for the registry
    # def reset(self, **kwargs):
    #     obs, info = self._env.reset(**kwargs)
    #     self.last_obs = obs
    #     return obs, info

    def _update_window_title(self):
        """Helper to update the window title with current skill info."""
        if self.render_mode == "human":
            title = f"Algo: {self.current_algo} | Global Skill: {self.current_global_skill} | Local Skill: {self.current_local_skill} | Step: {self.current_step_in_skill}/{self.skill_duration}"
            
            # Try PyGame (Minigrid uses PyGame)
            try:
                import pygame
                pygame.display.set_caption(title)
            except ImportError:
                pass
            except Exception:
                pass
                
            # Try Matplotlib (fallback)
            try:
                if hasattr(self._env, 'window') and self._env.window is not None:
                     # Minigrid's Window class might expose the underlying backend
                     pass
            except Exception:
                pass

    def _update_window_title(self):
        """Helper to update the window title with current skill info."""
        if self.render_mode == "human":
            title = f"Algo: {self.current_algo} | Global Skill: {self.current_global_skill} | Local Skill: {self.current_local_skill} | Step: {self.current_step_in_skill}/{self.skill_duration}"
            
            # Try PyGame (Minigrid uses PyGame)
            try:
                import pygame
                pygame.display.set_caption(title)
            except ImportError:
                pass
            except Exception:
                pass
                
            # Try Matplotlib (fallback)
            try:
                if hasattr(self._env, 'window') and self._env.window is not None:
                     # Minigrid's Window class might expose the underlying backend
                     pass
            except Exception:
                pass

    def render(self, mode="rgb_array", **kwargs):
        # --- FIX START: Robust Render ---
        # Try standard render
        frame = self._env.render()
        
        # If frame is None (likely due to render_mode mismatch or gym version issues), 
        # force generation of the frame using the minigrid method directly.
        if frame is None:
            if hasattr(self._env, 'get_frame'):
                # Standard tile_size for minigrid is 32
                frame = self._env.get_frame(highlight=False, tile_size=128)
            elif hasattr(self._env.unwrapped, 'get_frame'):
                frame = self._env.unwrapped.get_frame(highlight=False, tile_size=128)
        
        # Update Window Title if in human mode
        self._update_window_title()

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