#!/usr/bin/env python3
"""
Produce METRA-style anchored radial plots for skills.

This version guarantees:
 - PCA-projected 2D trajectories
 - Each trajectory anchored at its start
 - Symmetric zoom-out so the origin is centered
 - The origin is visually centered in the image
 - Clean, deterministic layout with centered axes box
"""

from __future__ import annotations
import os
import sys
import argparse
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from SingleLoader import load_config, load_model_from_config
from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry


# -------------------------------------------------------------------------
#  Collect trajectories
# -------------------------------------------------------------------------

def collect_trajs(adapter, env_factory, skill_idx: int, episodes: int=8, horizon: int=200) -> List[np.ndarray]:
    trainer = getattr(adapter, 'trainer', None)

    def extract_phi(obs):
        if hasattr(adapter, 'get_phi'):
            return np.array(adapter.get_phi(obs)).reshape(-1)
        if trainer is not None and hasattr(trainer, 'phi') and hasattr(trainer, '_obs_to_vec'):
            import torch
            with torch.no_grad():
                ovec = trainer._obs_to_vec(obs)
                o = trainer._obs_to_tensor(ovec)
                return trainer.phi(o).detach().cpu().numpy().reshape(-1)
        raise RuntimeError('No phi available')

    trajs = []
    for ep in range(episodes):
        env = env_factory()
        obs, info = env.reset(seed=ep)

        if hasattr(adapter, 'set_skill'):
            adapter.set_skill(skill_idx)

        seq = [extract_phi(obs)]
        for t in range(horizon):
            act = adapter.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(act))
            seq.append(extract_phi(obs))
            if terminated or truncated:
                break

        env.close()
        trajs.append(np.vstack(seq))
    return trajs


# -------------------------------------------------------------------------
#   Axes centering for perfectly centered origin
# -------------------------------------------------------------------------

def center_axes(ax, all_pts):
    """
    Ensures:
      - symmetric x/y limits around 0
      - equal aspect ratio
      - axis box visually centered in figure
    """
    max_extent = np.max(np.abs(all_pts))
    lim = max_extent * 1.05 if max_extent > 0 else 1.0

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')

    # Centered square axes area (80% of figure)
    side = 0.80
    offset = (1 - side) / 2
    ax.set_position([offset, offset, side, side])


# -------------------------------------------------------------------------
#  Plotting
# -------------------------------------------------------------------------

def metra_plot(all_trajs: dict, out_path: str, title: str, z_basis: np.ndarray | None = None):
    # Collect all points for PCA
    all_points = [tr for trajs in all_trajs.values() for tr in trajs if tr.size > 0]
    if len(all_points) == 0:
        raise RuntimeError('No trajectories to plot')

    concat = np.vstack(all_points)

    # PCA transform
    if concat.shape[1] == 2:
        pca_transform = lambda x: x
    else:
        pca = PCA(n_components=2)
        pca.fit(concat)
        pca_transform = lambda x: pca.transform(x)

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    cmap = plt.get_cmap('tab10')

    anchored_points = []

    # Plot skill trajectories
    for k, trajs in sorted(all_trajs.items()):
        color = cmap(k % 10)
        for tr in trajs:
            if tr.size == 0:
                continue
            xy = pca_transform(tr)
            anchored = xy - xy[0:1]  # anchor at start
            ax.plot(anchored[:, 0], anchored[:, 1], color=color, alpha=0.85, linewidth=1.5)
            anchored_points.append(anchored)

    # Overlay Z basis arrows if provided
    if z_basis is not None:
        z_proj = pca_transform(z_basis)
        for i, v in enumerate(z_proj):
            color = cmap(i % 10)
            ax.arrow(0, 0, v[0], v[1], color=color, width=0.01,
                     head_width=0.15, length_includes_head=True)
            ax.text(v[0] * 1.05, v[1] * 1.05, str(i), color=color)

    # Center and scale axes to include all anchored points
    if len(anchored_points) > 0:
        all_pts = np.vstack(anchored_points)
        center_axes(ax, all_pts)

    ax.set_title(title)
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')

    fig = plt.gcf()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------------------
#  Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--algo', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--horizon', type=int, default=200)
    parser.add_argument('--out', type=str, default='metra_mimic.png')
    parser.add_argument('--skills', type=str, default=None)
    parser.add_argument('--use_z', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = None
    for m in config.model_cfgs:
        if m.algo_name.upper() == args.algo.upper():
            model_cfg = m
            break
    if model_cfg is None:
        raise RuntimeError('Algorithm not found in config')

    skill_registry = SkillRegistry(model_cfg.skill_dim)
    adapter = load_model_from_config(model_cfg, skill_registry=skill_registry)

    trainer = getattr(adapter, 'trainer', None)
    if trainer is not None and hasattr(trainer, 'env') and trainer.env is not None:
        def env_factory():
            sz = getattr(trainer.cfg, 'size', None)
            if sz is None:
                from minigrid.envs import DoorKeyEnv
                return DoorKeyEnv(size=8, max_steps=args.horizon, render_mode='rgb_array')
            else:
                from oesd.ours.enviroments.example_minigrid import SimpleEnv
                return SimpleEnv(size=sz, max_steps=args.horizon, render_mode='rgb_array')
    else:
        from minigrid.envs import DoorKeyEnv
        def env_factory():
            return DoorKeyEnv(size=8, max_steps=args.horizon, render_mode='rgb_array')

    if args.skills:
        skill_indices = [int(s.strip()) for s in args.skills.split(',')]
    else:
        skill_indices = list(range(adapter.skill_dim))

    all_trajs = {}
    for k in skill_indices:
        print(f'Collecting skill {k}...')
        all_trajs[k] = collect_trajs(adapter, env_factory, skill_idx=k,
                                     episodes=args.episodes, horizon=args.horizon)

    z_basis = None
    if args.use_z:
        if trainer is not None and hasattr(trainer, 'discrete_Z'):
            Z = trainer.discrete_Z
            if hasattr(Z, 'detach'):
                Z = Z.detach().cpu().numpy()
            z_basis = Z

    metra_plot(all_trajs, args.out, title=f'{model_cfg.algo_name} METRA-style (anchored)')
    print('Saved', args.out)


if __name__ == '__main__':
    main()
