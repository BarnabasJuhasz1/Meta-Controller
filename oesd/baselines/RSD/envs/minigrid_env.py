import numpy as np
import gym
import akro
import matplotlib.pyplot as plt
import random
try:
    from minigrid.minigrid_env import MiniGridEnv
    from minigrid.core.mission import MissionSpace
    from minigrid.core.grid import Grid
    from minigrid.core.world_object import Door, Goal, Key, Wall
    from minigrid.wrappers import FullyObsWrapper
    from minigrid.envs import DoorKeyEnv
except ImportError as exc:
    raise ImportError(
        "The minigrid package is required to use BaselineMiniGridEnv. "
        "Install it with `pip install minigrid gymnasium`."
    ) from exc


class _SimpleMiniGrid(MiniGridEnv):
    """A simple locked-room MiniGrid layout used for baseline testing."""

    def __init__(
        self,
        size=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        key_pos=None,
        max_steps=None,
        **kwargs,
    ):
        self._size = size
        self._agent_start_pos = agent_start_pos
        # agent_start_pos=None,
        self._agent_start_dir = agent_start_dir
        # self._agent_start_dir = random.randint(0, 3)
        self._key_pos = key_pos
        self.rng = random.Random() # Use dedicated RNG for key placement
        
        mission_space = MissionSpace(mission_func=lambda: "testing")

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        grid = Grid(width, height)
        grid.wall_rect(0, 0, width, height)

        for i in range(0, height):
            grid.set(width // 2, i, Wall())

        grid.set(width // 2, height // 2, Door("red", is_locked=True))
        if self._key_pos == 'random':
            # Randomly place key in the left room
            # Left room x range: 1 to width // 2 - 1
            # y range: 1 to height - 2
            # Avoid agent start position if it's fixed
            while True:
                key_x = self.rng.randint(1, width // 2 - 1)
                key_y = self.rng.randint(1, height - 2)
                if self._agent_start_pos is not None:
                    if (key_x, key_y) == self._agent_start_pos:
                        continue
                grid.set(key_x, key_y, Key("red"))
                # print(f"Key placed at ({key_x}, {key_y})")
                break
        elif self._key_pos is not None:
            grid.set(self._key_pos[0], self._key_pos[1], Key("red"))
        else:
            grid.set(width // 2 - 2, height // 2, Key("red"))

        # FIXED: Removed Goal object to prevent corner bias
        # self.put_obj(Goal(), width - 2, height - 2)

        self.grid = grid
        if self._agent_start_pos is not None:
            self.agent_pos = self._agent_start_pos
            self.agent_dir = self._agent_start_dir
        else:
            self.place_agent()

        self.mission = "testing"


class BaselineMiniGridEnv(gym.Env):
    """Wraps _SimpleMiniGrid to expose a continuous control-style interface."""

    def __init__(
        self,
        size=10,
        action_scale=1.0,
        max_steps=None,
        render_mode="rgb_array",
        # render_mode="human",
        # render_mode=None,
        env_name="minigrid",
        seed=0,
    ):
        super().__init__()
        if env_name == "minigrid":
            self._env = _SimpleMiniGrid(size=size, max_steps=max_steps, render_mode=render_mode, seed=seed)
        elif env_name == "minigrid_random_key":
            self._env = _SimpleMiniGrid(size=size, max_steps=max_steps, render_mode=render_mode, key_pos='random', seed=seed)
        elif env_name == "minigrid_small":
            # DoorKeyEnv-5x5 (there is also size 6, 8, and 16)
            self._env = DoorKeyEnv(size=8, max_steps=max_steps, render_mode=render_mode)
        else:
            raise ValueError(f"Unknown environment name: {env_name}")
        # FIXED: Enable Full Observability to solve spatial aliasing (corner bias)
        # self._env = FullyObsWrapper(self._env)
        
        self._max_steps = max_steps or self._env.max_steps
        self._action_scale = action_scale
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

    # def _process_obs(self, obs):

    #     if isinstance(obs, tuple):
    #         obs = obs[0]
    #     image = obs["image"].astype(np.float32) / 255.0
    #     direction = np.array([obs["direction"] / 3.0], dtype=np.float32)
        
    #     # FIXED: Add carrying state to observation
    #     carrying = 1.0 if self._env.carrying is not None else 0.0
    #     carrying = np.array([carrying], dtype=np.float32)
    #     if carrying:
    #         print("AGENT CARRYING")

    #     return np.concatenate([image.flatten(), direction, carrying], axis=0)

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

        #return image.flatten()
        
        # Concatenate everything: [Image flat, Direction, Carrying, PosX, PosY]
        # return np.concatenate([
        #     image.flatten(), 
        #     direction, 
        #     carrying, 
        #     position
        # ], axis=0)
        return np.concatenate([
            image.flatten(), 
            direction, 
            carrying, 
            # position
        ], axis=0)

    def _map_action(self, action):
        if action.ndim == 0:
            return int(action)
        return int(np.argmax(action))

    def reset(self, **kwargs):
        result = self._env.reset(**kwargs, seed=5)
        if isinstance(result, tuple):
            obs, _ = result
        else:
            obs = result
        self._last_obs = self._process_obs(obs)
        return self._last_obs

    def step(self, action, render=False):
        coord_before = np.array(self._env.agent_pos, dtype=np.float32)
        discrete_action = self._discrete_actions[self._map_action(np.asarray(action))]
        result = self._env.step(discrete_action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        next_obs = self._process_obs(obs)

        coord_after = np.array(self._env.agent_pos, dtype=np.float32)
        info = dict(info)
        info.update(
            coordinates=coord_before,
            next_coordinates=coord_after,
            ori_obs=self._last_obs,
            next_ori_obs=next_obs,
        )
        if render:
            frame = self._env.render()
            if frame is not None:
                info['render'] = frame.transpose(2, 0, 1)
        self._last_obs = next_obs
        return next_obs, reward, done, info

    # def render(self, mode="rgb_array", **kwargs):
    #     return self._env.render()

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

    def calc_eval_metrics(self, trajectories, is_option_trajectories=True):
        coords = []
        for traj in trajectories:
            coords.append(traj["env_infos"]["coordinates"])
            coords.append(traj["env_infos"]["next_coordinates"][-1:])
        coords = np.concatenate(coords, axis=0)
        uniq = np.unique(np.floor(coords), axis=0)
        return {"MiniGridUniqueCoords": len(uniq)}


