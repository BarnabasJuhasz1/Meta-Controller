from __future__ import absolute_import, division, print_function

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import Wrapper
except ImportError:
    import gym
    from gym import Wrapper


class SkillWrapper(Wrapper):
    """
    Wraps an env and concatenates a latent skill vector to the observation.

    - Supports:
        * Plain Box observation spaces
        * Dict observations of the form:
            - {"observation": vector}              (old style)
            - {"image": HxWxC, "direction": int}   (MiniGrid)
    - Observation returned is always a flat 1D float32 vector, optionally
    concatenated with a skill vector z.
    """

    def __init__(
        self,
        env,
        num_latent_skills=None,
        skill_type="discrete_uniform",
        preset_skill=None,
        min_steps_before_resample=10,
        resample_prob=0.0,
    ):
        super(SkillWrapper, self).__init__(env)

        self._skill_type = skill_type
        self._num_skills = 0 if num_latent_skills is None else num_latent_skills
        self._preset_skill = preset_skill
        self._min_steps_before_resample = min_steps_before_resample
        self._resample_prob = resample_prob

        base_space = self.env.observation_space

        # ---------------- Determine base observation dimension ----------------
        if isinstance(base_space, gym.spaces.Dict):
            spaces = base_space.spaces

            if "observation" in spaces:
                # Old-style dict: {"observation": vector}
                obs_dim = int(np.prod(spaces["observation"].shape))
                self._obs_mode = "vector_obs"
            elif "image" in spaces and "direction" in spaces:
                # MiniGrid-style dict: {"image": HxWxC, "direction": Discrete, "mission": str}
                img_shape = spaces["image"].shape
                img_dim = int(np.prod(img_shape))
                obs_dim = img_dim + 1  # +1 for direction scalar
                self._obs_mode = "minigrid"
            else:
                raise ValueError(
                    "SkillWrapper: Unsupported Dict observation structure. "
                    "Expected either key 'observation' or ('image' + 'direction'). "
                    f"Got keys: {list(spaces.keys())}"
                )
        else:
            # Simple Box observation
            obs_dim = int(np.prod(base_space.shape))
            self._obs_mode = "box"

        total_dim = obs_dim + self._num_skills

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32,
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _extract_base_obs(self, obs):
        """Return a flat 1D float32 numpy array from raw env obs."""
        base_space = self.env.observation_space

        if isinstance(base_space, gym.spaces.Dict):
            if self._obs_mode == "vector_obs":
                vec = obs["observation"]
                return np.asarray(vec, dtype=np.float32).reshape(-1)

            elif self._obs_mode == "minigrid":
                img = obs["image"]        # H x W x C
                direction = obs["direction"]  # int 0..3

                img_flat = np.asarray(img, dtype=np.float32).reshape(-1)
                direction_arr = np.asarray([direction], dtype=np.float32)

                return np.concatenate([img_flat, direction_arr], axis=0)

            else:
                raise RuntimeError("Unexpected _obs_mode for Dict obs: " + str(self._obs_mode))

        # Non-dict Box obs
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _make_obs_with_skill(self, obs):
        """Take raw env obs, flatten it, and append skill if needed."""
        base_vec = self._extract_base_obs(obs)

        if self._num_skills == 0:
            return base_vec.astype(np.float32)

        return np.concatenate([base_vec, self.skill], axis=0).astype(np.float32)

    def _set_skill(self):
        """Sample or set the latent skill vector self.skill."""
        if self._num_skills == 0:
            self.skill = np.zeros(0, dtype=np.float32)
            return

        if self._preset_skill is not None:
            self.skill = np.asarray(self._preset_skill, dtype=np.float32)
            print("Skill (preset):", self.skill)
        elif self._skill_type == "discrete_uniform":
            self.skill = np.random.multinomial(
                1, [1.0 / self._num_skills] * self._num_skills
            ).astype(np.float32)
        elif self._skill_type == "gaussian":
            self.skill = np.random.multivariate_normal(
                np.zeros(self._num_skills), np.eye(self._num_skills)
            ).astype(np.float32)
        elif self._skill_type == "cont_uniform":
            self.skill = np.random.uniform(
                low=-1.0, high=1.0, size=self._num_skills
            ).astype(np.float32)
        else:
            raise ValueError("Unknown skill_type: {}".format(self._skill_type))

    # -------------------------------------------------------------------------
    # Gym / Gymnasium API
    # -------------------------------------------------------------------------
    def reset(self, **kwargs):
        """Reset env and sample a new skill. Returns (obs, info)."""
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            obs, info = out
        else:
            obs, info = out, {}

        self._set_skill()
        self._step_count = 0
        return self._make_obs_with_skill(obs), info

    def step(self, action):
        """
        Step env and return (obs_with_skill, reward, done, info).

        Works for both Gym (4-tuple) and Gymnasium (5-tuple).
        """
        out = self.env.step(action)

        # Gymnasium: obs, reward, terminated, truncated, info
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:
            # Classic Gym: obs, reward, done, info
            obs, reward, done, info = out

        self._step_count += 1

        if (
            self._preset_skill is None
            and self._step_count >= self._min_steps_before_resample
            and np.random.random() < self._resample_prob
        ):
            self._set_skill()
            self._step_count = 0

        return self._make_obs_with_skill(obs), reward, done, info

    def close(self):
        return self.env.close()
