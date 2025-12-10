#!/usr/bin/env python3
"""
SkillTracer: Visualize skill execution trajectories with interaction annotations.

Traces a skill's performance in the environment, recording:
  - Full trajectory (agent position over time)
  - Interactions (key pickup, door open, etc.)
  - Saves annotated images per skill/algorithm

Usage:
    tracer = SkillTracer(env_factory, adapter, output_dir="./skill_traces")
    tracer.trace_skill(skill_idx=0, num_episodes=3, save_images=True)
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from typing import Callable, List, Tuple, Any


@dataclass
class TraceStep:
    """Single step in a trace: position, action, reward, interactions."""
    position: np.ndarray  # (x, y)
    action: int
    reward: float
    carrying: bool | None = None
    door_opened: bool | None = None
    key_picked: bool | None = None
    info: dict | None = None


class SkillTracer:
    """
    Traces skill execution and generates annotated trajectory images.
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        adapter: Any,
        algo_name: str,
        output_dir: str = "./skill_traces",
    ):
        """
        Args:
            env_factory: callable that returns an env instance
            adapter: loaded adapter with get_action(obs, skill_z) and set_skill(idx)
            algo_name: name of algorithm (e.g., "LSD", "RSD")
            output_dir: where to save images
        """
        self.env_factory = env_factory
        self.adapter = adapter
        self.algo_name = algo_name
        self.output_dir = output_dir
        self.skill_dim = adapter.skill_dim
        
        # Get tile size from environment (MiniGrid specific)
        temp_env = env_factory()
        self.tile_size = getattr(temp_env, "tile_size", 32)  # Default to 32 for MiniGrid
        if hasattr(temp_env, "grid"):
            self.grid_width = temp_env.grid.width
            self.grid_height = temp_env.grid.height
        else:
            self.grid_width = 8
            self.grid_height = 8
        temp_env.close()
        
        os.makedirs(self.output_dir, exist_ok=True)

    def _grid_to_pixel(self, grid_pos: np.ndarray) -> np.ndarray:
        """Convert grid coordinates to pixel coordinates for plotting.
        
        MiniGrid agent_pos is in grid coordinates (e.g., 0-7 for 8x8 grid).
        Rendered image has tiles of size tile_size pixels.
        Convert to pixel coordinates: pixel = grid * tile_size + tile_size/2
        """
        pixel_pos = grid_pos * self.tile_size + self.tile_size / 2
        return pixel_pos

    def _extract_state(self, env, obs, info) -> Tuple[np.ndarray, dict]:
        """Extract position and interaction state from environment."""
        # Position: agent_pos if available
        if hasattr(env, "agent_pos"):
            pos = np.array(env.agent_pos, dtype=np.float32)
        else:
            pos = np.zeros(2, dtype=np.float32)

        state = {
            "carrying": None,
            "door_opened": False,
        }

        # MiniGrid specifics
        if hasattr(env, "carrying") and env.carrying is not None:
            state["carrying"] = True
        
        # Door state (check if any door has been opened)
        if hasattr(env, "grid") and hasattr(env.grid, "grid"):
            grid_list = env.grid.grid
            # Handle both list and numpy array cases
            if isinstance(grid_list, list):
                items = grid_list
            else:
                items = grid_list.flat
            
            for cell in items:
                if cell is not None and hasattr(cell, "is_open") and cell.is_open:
                    state["door_opened"] = True
                    break

        return pos, state

    def trace_episode(self, skill_idx: int | None = None, seed: int = 0, horizon: int = 200) -> Tuple[List[TraceStep], np.ndarray]:
        """
        Run one episode with a chosen skill and return full trace + final env background.

        Args:
            skill_idx: which skill to use (None = adapter default)
            seed: random seed for env reset
            horizon: max steps per episode

        Returns:
            tuple of (list of TraceStep objects, background image array)
        """
        env = self.env_factory()
        obs, info = env.reset(seed=seed)

        # Set skill if specified
        if skill_idx is not None and hasattr(self.adapter, "set_skill"):
            self.adapter.set_skill(int(skill_idx))

        trace = []
        pos, state = self._extract_state(env, obs, info)
        trace.append(TraceStep(position=pos.copy(), action=-1, reward=0.0, **state))

        for step in range(horizon):
            # Get action from adapter
            action = self.adapter.get_action(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Extract position and interactions
            pos, state = self._extract_state(env, obs, info)
            trace.append(
                TraceStep(
                    position=pos.copy(),
                    action=action,
                    reward=reward,
                    **state,
                    info=info,
                )
            )

            if terminated or truncated:
                break

        # Render final environment state as background
        try:
            bg = env.render()
        except Exception:
            bg = None

        return trace, bg

    def plot_trace(self, trace: List[TraceStep], bg: np.ndarray | None = None, skill_idx: int | None = None) -> plt.Figure:
        """
        Plot a trace with trajectory and interaction markers.

        Args:
            trace: list of TraceStep objects
            bg: background image (from env.render() at episode end)
            skill_idx: skill index (for title)

        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw background
        if bg is not None:
            ax.imshow(bg, origin="upper")
        else:
            # Fallback: try to render a fresh env
            try:
                env = self.env_factory()
                env.reset()
                fallback_bg = env.render()
                ax.imshow(fallback_bg, origin="upper")
            except Exception:
                ax.set_facecolor("black")

        # Extract grid positions and convert to pixel coordinates
        grid_positions = np.array([step.position for step in trace])
        pixel_positions = np.array([self._grid_to_pixel(pos) for pos in grid_positions])

        # Draw trajectory
        if len(pixel_positions) > 1:
            ax.plot(
                pixel_positions[:, 0],
                pixel_positions[:, 1],
                color="#1f77b4",
                linewidth=2.0,
                alpha=0.7,
                marker="o",
                markersize=3,
            )

        # Mark key pickup events
        for i, step in enumerate(trace):
            if step.carrying and (i == 0 or not trace[i - 1].carrying):
                # Key picked up at this step
                pixel_pos = self._grid_to_pixel(step.position)
                ax.plot(
                    pixel_pos[0],
                    pixel_pos[1],
                    marker="*",
                    markersize=20,
                    color="gold",
                    label="Key Picked" if i == 0 else "",
                    zorder=10,
                )

        # Mark door open events
        for i, step in enumerate(trace):
            if step.door_opened and (i == 0 or not trace[i - 1].door_opened):
                # Door opened at this step
                pixel_pos = self._grid_to_pixel(step.position)
                ax.plot(
                    pixel_pos[0],
                    pixel_pos[1],
                    marker="s",
                    markersize=12,
                    color="red",
                    label="Door Opened" if i == 0 else "",
                    zorder=10,
                )

        # Mark start and end
        ax.plot(
            pixel_positions[0, 0],
            pixel_positions[0, 1],
            marker="o",
            markersize=12,
            color="green",
            label="Start",
            zorder=11,
        )
        ax.plot(
            pixel_positions[-1, 0],
            pixel_positions[-1, 1],
            marker="X",
            markersize=12,
            color="purple",
            label="End",
            zorder=11,
        )

        # Title
        title = f"{self.algo_name}"
        if skill_idx is not None:
            title += f" - Skill {skill_idx}"
        title += f" (Steps: {len(trace) - 1})"
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Formatting
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_aspect("equal")
        plt.tight_layout()

        return fig

    def trace_skill(
        self,
        skill_idx: int | None = None,
        num_episodes: int = 3,
        save_images: bool = True,
        horizon: int = 200,
    ) -> dict:
        """
        Trace a skill across multiple episodes and optionally save images.

        Args:
            skill_idx: which skill (None = adapter default)
            num_episodes: how many episodes to trace
            save_images: whether to save PNG files
            horizon: max steps per episode

        Returns:
            dict with metadata and traces
        """
        traces = []
        for ep in range(num_episodes):
            trace, bg = self.trace_episode(skill_idx=skill_idx, seed=ep, horizon=horizon)
            traces.append((trace, bg))
            print(
                f"[{self.algo_name} Skill {skill_idx}] Episode {ep + 1}/{num_episodes}: "
                f"{len(trace) - 1} steps"
            )

        if save_images:
            for ep, (trace, bg) in enumerate(traces):
                fig = self.plot_trace(trace, bg=bg, skill_idx=skill_idx)
                filename = f"{self.algo_name}_skill{skill_idx:02d}_ep{ep:03d}.png"
                filepath = os.path.join(self.output_dir, filename)
                fig.savefig(filepath, dpi=100, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved: {filepath}")

        return {
            "algo": self.algo_name,
            "skill_idx": skill_idx,
            "num_episodes": num_episodes,
            "traces": [t[0] for t in traces],  # only return trace, not bg
            "output_dir": self.output_dir,
        }

    def trace_all_skills(self, num_episodes: int = 2, save_images: bool = True) -> dict:
        """
        Trace all skills for this algorithm.

        Args:
            num_episodes: episodes per skill
            save_images: whether to save images

        Returns:
            aggregated results dict
        """
        print(f"\n{'='*60}")
        print(f"Tracing all skills for {self.algo_name} ({self.skill_dim} skills)")
        print(f"{'='*60}")

        results = {
            "algo": self.algo_name,
            "skills": {},
        }

        for skill_idx in range(self.skill_dim):
            result = self.trace_skill(
                skill_idx=skill_idx,
                num_episodes=num_episodes,
                save_images=save_images,
            )
            results["skills"][skill_idx] = result

        print(f"\nAll traces saved to: {self.output_dir}\n")
        return results
