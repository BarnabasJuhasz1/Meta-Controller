# SingleVisualizer.py

from __future__ import annotations
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
    skill_sets: list | None = None  # optional list of per-model skill index lists


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
        # optional per-model skill selections: list of lists (one per model)
        # e.g. cfg.skill_sets = [[0,1], [2]] or None
        self.skill_sets = cfg.skill_sets

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
            # model_interface.get_action supports optional skill override via keyword skill_z
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
            name = getattr(mconfig, "algo", None) or getattr(mconfig, "algo_name", None) or "model"
            print(f"[Visualizer] Sampling {self.num_episodes} episodes for {name}...")

            # determine which skill indices to visualize for this model
            skill_idxs = None
            if self.skill_sets is not None:
                # if skill_sets provided and matches models length, use per-model list
                if len(self.skill_sets) == len(self.models):
                    skill_idxs = list(self.skill_sets[self.models.index(mconfig)])
                elif len(self.skill_sets) == 1:
                    skill_idxs = list(self.skill_sets[0])
            # default: use current skill only (None special marker)
            if skill_idxs is None:
                skill_idxs = [None]

            # collect per-skill
            for sidx in skill_idxs:
                key = f"{name}" if sidx is None else f"{name}_s{sidx}"
                results[key] = []
                for ep in range(self.num_episodes):
                    # if adapter supports set_skill and we have an index, set it
                    try:
                        if sidx is not None and hasattr(model, "set_skill"):
                            model.set_skill(int(sidx))
                            traj = self._run_episode(model)
                        elif sidx is not None and hasattr(model, "sample_skill"):
                            # try to obtain skill vector by index via skill registry if possible
                            # otherwise pass None and adapter will use its default
                            try:
                                # skill_registry may expose registered skills
                                skill_vec = model.skill_registry.get_skill_by_index(name, int(sidx))
                            except Exception:
                                skill_vec = None
                            if skill_vec is not None:
                                traj = self._run_episode_with_skill(model, skill_vec)
                            else:
                                traj = self._run_episode(model)
                        else:
                            traj = self._run_episode(model)
                    except Exception:
                        traj = self._run_episode(model)
                    results[key].append(traj)

        return results

    def _run_episode_with_skill(self, model_interface, skill_vec):
        """Run episode while forcing a skill vector (numpy or tensor)."""
        env = self.env_config.factory()
        obs, _ = env.reset(seed=self.seed)

        traj = []
        pos = self.position_fn(env, obs, {})
        traj.append(pos)

        for t in range(self.horizon):
            action = model_interface.get_action(obs, deterministic=self.deterministic, skill_z=skill_vec)
            obs, reward, terminated, truncated, info = env.step(action)
            setattr(env, "_last_action", int(action))
            if hasattr(self, "update_relative_motion"):
                self.update_relative_motion(env)
            pos = self.position_fn(env, obs, info)
            traj.append(pos)
            if terminated or truncated:
                break

        return np.array(traj, dtype=np.float32)

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
            key = getattr(mconfig, "algo", None) or getattr(mconfig, "algo_name", None) or f"model{i}"
            colors[key] = base_colors[i % len(base_colors)]

        # ------------------------------------------------------------------
        # 3. Overlay all trajectories
        # ------------------------------------------------------------------
        def _lighten_color(hexcolor, amount=0.5):
            # mix with white to lighten
            import matplotlib.colors as mc
            c = np.array(mc.to_rgb(hexcolor))
            white = np.ones_like(c)
            return tuple(c + (white - c) * float(amount))

        for model_name, traj_list in trajectories.items():
            # allow keys like 'LSD' or 'LSD_s0'
            if "_s" in model_name:
                base, sstr = model_name.split("_s", 1)
                try:
                    sidx = int(sstr)
                except Exception:
                    sidx = 0
            else:
                base = model_name
                sidx = None

            base_color = colors.get(base, "#888888")

            for traj in traj_list:
                traj = np.array(traj)
                if len(traj) < 2:
                    continue
                if sidx is None:
                    plot_color = base_color
                else:
                    # For different skills, slightly lighten the base color
                    amt = min(0.8, 0.15 * (sidx + 1))
                    plot_color = _lighten_color(base_color, amount=amt)

                ax.plot(traj[:, 0], traj[:, 1], color=plot_color, alpha=0.8, linewidth=1.2)

        # ------------------------------------------------------------------
        # 4. Clean formatting
        # ------------------------------------------------------------------
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        if len(self.models) == 1:
            ax.set_title(self.models[0].algo.upper(), fontsize=16)
        else:
            ax.set_title("Trajectory Overlay", fontsize=16)

        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()
