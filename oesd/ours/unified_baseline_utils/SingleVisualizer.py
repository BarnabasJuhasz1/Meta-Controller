# SingleVisualizer.py

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Callable, Dict, List, Any

from SingleLoader import load_model_from_config

from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry


# ============================================================================
# Dataclasses for configs
# ============================================================================

@dataclass
class EnvConfig:
    factory: Callable[[], Any]  # callable that returns env instances


@dataclass
class VisualizerConfig:
    env: EnvConfig
    models: list               # list[ModelConfig]
    horizon: int
    num_episodes: int
    deterministic: bool
    seed: int
    position_fn: Callable      # unified_position_fn
    skill_count: int


# ============================================================================
# SingleVisualizer
# ============================================================================

class SingleVisualizer:
    """
    Unified visualization helper that:
    - Loads multiple models (via SingleLoader)
    - Samples trajectories
    - Plots trajectories on top of environment background
    """

    def __init__(self, cfg: VisualizerConfig):
        self.cfg = cfg
        self.env_config = cfg.env
        self.models = cfg.models
        self.horizon = cfg.horizon
        self.num_episodes = cfg.num_episodes
        self.deterministic = cfg.deterministic
        self.seed = cfg.seed
        self.position_fn = cfg.position_fn
        self.skill_count = cfg.skill_count

        # INITIALIZE SKILL REGISTRY
        self.skill_registry = SkillRegistry(self.skill_count)
    
        # LOAD MODELS VIA ADAPTERS (while feeding skill_registry to adapters)
        self.model_interfaces = [load_model_from_config(m, skill_registry=self.skill_registry) for m in self.models]

    # ----------------------------------------------------------------------
    # Helper to run one episode for one model
    # ----------------------------------------------------------------------
    def _run_episode(self, model_interface):
        env = self.env_config.factory()
        obs, _ = env.reset(seed=self.seed)

        traj = []
        pos = self.position_fn(env, obs, {})
        traj.append(pos)

        for t in range(self.horizon):
            # compute action
            action = model_interface.get_action(obs, deterministic=self.deterministic)

            # step env
            obs, reward, terminated, truncated, info = env.step(action)

            # store last_action so relative tracker can use it
            setattr(env, "_last_action", int(action))

            # update relative motion if needed
            if hasattr(self, "update_relative_motion"):
                self.update_relative_motion(env)

            # extract position
            pos = self.position_fn(env, obs, info)
            traj.append(pos)

            if terminated or truncated:
                break

        return np.array(traj, dtype=np.float32)

    # ----------------------------------------------------------------------
    # Sample trajectories for all models
    # ----------------------------------------------------------------------
    def sample_trajectories(self):
        results = {}

        for mconfig, model in zip(self.models, self.model_interfaces):
            name = mconfig.algo_name
            results[name] = []
            print(f"[Visualizer] Sampling {self.num_episodes} episodes for {name}...")

            for ep in range(self.num_episodes):
                traj = self._run_episode(model)
                results[name].append(traj)

        return results

    # ----------------------------------------------------------------------
    # METRA-style visualization
    # ----------------------------------------------------------------------
    def plot_trajectories(self, trajectories: Dict[str, List[np.ndarray]]):
        """
        Draw environment (rgb_array) background, then overlay all trajectories.
        Each model gets one permanent color.
        """

        # create one environment just to draw background
        env_bg = self.env_config.factory()
        try:
            bg = env_bg.render()  # must be rgb_array
        except:
            bg = None

        fig, ax = plt.subplots(figsize=(7, 7))

        # ------------------------------------------------------------------
        # 1. Draw background
        # ------------------------------------------------------------------
        if bg is not None:
            H, W, _ = bg.shape
            ax.imshow(bg, origin="lower")
        else:
            ax.set_facecolor("black")

        # ------------------------------------------------------------------
        # 2. Assign model colors (shades not needed since each model has 1 color)
        # ------------------------------------------------------------------
        base_colors = [
            "#1f77b4",  # blue
            "#d62728",  # red
            "#2ca02c",  # green
            "#9467bd",  # purple
            "#ff7f0e",  # orange
            "#17becf"   # cyan
        ]

        colors = {}
        for i, mconfig in enumerate(self.models):
            colors[mconfig.algo_name] = base_colors[i % len(base_colors)]

        # ------------------------------------------------------------------
        # 3. Overlay all trajectories
        # ------------------------------------------------------------------
        for model_name, traj_list in trajectories.items():
            color = colors[model_name]

            for traj in traj_list:
                traj = np.array(traj)
                if len(traj) < 2:
                    continue
                ax.plot(traj[:, 0], traj[:, 1],
                        color=color, alpha=0.7, linewidth=1.2)

        # ------------------------------------------------------------------
        # 4. Clean formatting
        # ------------------------------------------------------------------
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        if len(self.models) == 1:
            ax.set_title(self.models[0].algo_name.upper(), fontsize=16)
        else:
            ax.set_title("Trajectory Overlay", fontsize=16)

        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()
