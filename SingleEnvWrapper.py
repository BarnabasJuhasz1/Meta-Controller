# SingleEnvWrapper.py
from __future__ import annotations
import gymnasium as gym
from typing import Any, Tuple


class SingleEnvWrapper(gym.Wrapper):
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
        skill_vector,
        deterministic: bool = False
    ):
        """
        Compute action via the adapter, then step environment:

            obs_vec = adapter.preprocess_observation(_last_obs)
            action  = adapter.get_action(model, obs_vec, skill_vector)
            obs, reward, ter, trunc, info = env.step(action)

        Returns:
            obs, reward, terminated, truncated, info, action
        """
        if self._last_obs is None:
            raise RuntimeError(
                "step_with_model() called before reset(). Call env.reset() first."
            )

        # 1. Convert raw MiniGrid obs → model input vector
        obs_vec = self.adapter.preprocess_observation(self._last_obs)

        # 2. Query the algorithm for primitive action
        action = self.adapter.get_action(
            self.model,
            obs_vec,
            skill_vector,
            deterministic=deterministic,
        )

        # 3. Execute in environment
        obs, reward, terminated, truncated, info = self.step(action)

        return obs, reward, terminated, truncated, info, action
