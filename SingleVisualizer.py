# SingleVisualizer.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Sequence

import gymnasium as gym

from SingleEnvWrapper import SingleEnvWrapper
from SingleLoader import ModelConfig, load_model_from_config


# =====================================================================
# CONFIG STRUCTURES
# =====================================================================

@dataclass
class EnvConfig:
    """
    How to create the environment.

    Either:
        env_id="MiniGrid-Empty-8x8-v0"
    OR
        factory=my_custom_env_constructor
    """
    env_id: str | None = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    factory: Callable[[], gym.Env] | None = None


@dataclass
class VisualizerConfig:
    env: EnvConfig
    models: Sequence[ModelConfig]

    horizon: int = 200
    num_episodes: int = 1
    deterministic: bool = False
    seed: int = 0

    # Optional: a function that returns a 2D position for plotting
    # signature: fn(env, obs, info) -> np.array([x, y])
    position_fn: Callable[[gym.Env, Any, dict], np.ndarray] | None = None


# =====================================================================
# VISUALIZER
# =====================================================================

class SingleVisualizer:
    """
    The unified, adapter-based visualization tool.

    This class:
    - loads all models using SingleLoader
    - creates wrapped envs
    - samples trajectories for each model
    - plots them side-by-side

    IMPORTANT:
    This file knows *nothing* about any algorithm internals.
    It only calls adapter functions.
    """

    def __init__(self, cfg: VisualizerConfig):
        self.cfg = cfg
        self.seed = cfg.seed
        self.rng = np.random.default_rng(self.seed)

        # Load all models & adapters from the registry system
        self.loaded_models = []   # list of (algo_name, adapter, model)
        for model_cfg in cfg.models:
            adapter, model = load_model_from_config(model_cfg)
            self.loaded_models.append((model_cfg.algo, adapter, model))

        # default position extractor if none provided
        if cfg.position_fn is None:
            self.position_fn = self._default_position_fn
        else:
            self.position_fn = cfg.position_fn

    # --------------------------------------------------------------
    # ENV CREATION
    # --------------------------------------------------------------

    def _make_env(self) -> gym.Env:
        ecfg = self.cfg.env
        if ecfg.factory is not None:
            return ecfg.factory()
        elif ecfg.env_id is not None:
            return gym.make(ecfg.env_id, **ecfg.env_kwargs)
        else:
            raise ValueError("EnvConfig must have either env_id or factory defined.")

    # --------------------------------------------------------------
    # POSITION EXTRACTION
    # --------------------------------------------------------------

    @staticmethod
    def _default_position_fn(env: gym.Env, obs: Any, info: dict) -> np.ndarray:
        """
        Default: flatten observation and take the first 2 dims.
        For MiniGrid this gives (tile0, tile1) — roughly okay.
        For custom envs you should override this with a better function.
        """
        try:
            arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        except Exception:
            return np.zeros(2, dtype=np.float32)
        if arr.size < 2:
            return np.zeros(2, dtype=np.float32)
        return arr[:2]

    # --------------------------------------------------------------
    # SAMPLE TRAJECTORIES FOR A SINGLE MODEL
    # --------------------------------------------------------------

    def _sample_single_model(self, algo_name: str, adapter, model) -> np.ndarray:
        """
        Returns trajectories of shape:
            (num_episodes, horizon, 2)
        """

        env = self._make_env()
        wrapped = SingleEnvWrapper(env, model, adapter)

        trajectories = []

        for ep in range(self.cfg.num_episodes):
            obs, info = wrapped.reset(seed=self.seed + ep)

            ep_positions = []

            # one skill per episode
            skill = adapter.sample_skill(self.rng)

            for t in range(self.cfg.horizon):
                # record position
                pos = self.position_fn(wrapped.env, obs, info)
                ep_positions.append(pos)

                obs, _, terminated, truncated, info, _ = wrapped.step_with_model(
                    skill,
                    deterministic=self.cfg.deterministic,
                )

                if terminated or truncated:
                    obs, info = wrapped.reset()

            ep_positions = np.stack(ep_positions, axis=0)
            trajectories.append(ep_positions)

        wrapped.close()
        return np.stack(trajectories, axis=0)

    # --------------------------------------------------------------
    # SAMPLE ALL TRAJECTORIES
    # --------------------------------------------------------------

    def sample_trajectories(self) -> Dict[str, np.ndarray]:
        out = {}
        for algo_name, adapter, model in self.loaded_models:
            trajs = self._sample_single_model(algo_name, adapter, model)
            out[algo_name] = trajs
        return out

    # --------------------------------------------------------------
    # PLOT TRAJECTORIES
    # --------------------------------------------------------------

    def plot_trajectories(self, trajectories: Dict[str, np.ndarray]):
        num_models = len(trajectories)
        if num_models == 0:
            raise ValueError("No trajectories to plot.")

        fig, axes = plt.subplots(
            1, num_models,
            figsize=(5 * num_models, 5),
            squeeze=False
        )
        axes = axes[0]

        for ax, (algo_name, trajs) in zip(axes, trajectories.items()):
            for ep_traj in trajs:
                ax.plot(ep_traj[:, 0], ep_traj[:, 1], alpha=0.6)

            ax.set_title(f"{algo_name.upper()}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True)
            ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------
    # LIVE ROLL OUT
    # --------------------------------------------------------------

    def render_rollout(self, model_idx: int = 0, max_steps: int | None = None):
        if not self.loaded_models:
            raise RuntimeError("No models loaded.")

        algo_name, adapter, model = self.loaded_models[model_idx]
        env = self._make_env()
        env = SingleEnvWrapper(env, model, adapter)

        obs, info = env.reset()
        skill = adapter.sample_skill(self.rng)

        if max_steps is None:
            max_steps = self.cfg.horizon

        for t in range(max_steps):
            env.render()

            obs, _, terminated, truncated, info, _ = env.step_with_model(
                skill,
                deterministic=self.cfg.deterministic,
            )
            if terminated or truncated:
                break

        env.close()
