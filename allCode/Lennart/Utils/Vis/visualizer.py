# Visualization/visualizer.py

import os
import numpy as np
import matplotlib.pyplot as plt

from Interface.loader import load_policy
from Env.registry import make_env
from Utils.Vis.traj_plotter import plot_state_trajectory
from Utils.Vis.phi_plotter import plot_phi_trajectory


def visualize_checkpoint(model_path, env_name, num_rollouts=5, out_dir="vis_out"):
    """
    Unified visualization script.
    Loads policy → generates rollouts → calls plotters.
    Works for LSD, DIAYN, RSD, anything using the unified API.
    """

    os.makedirs(out_dir, exist_ok=True)

    # 1. Load model → unified SkillPolicy
    policy = load_policy(model_path)
    baseline_name = model_path.split("/")[-1].split("_")[0]

    # 2. Create environment
    env = make_env(env_name)

    # 3. Rollout
    all_states = []
    all_phi = []
    all_skills = []

    for i in range(num_rollouts):
        obs = env.reset()
        skill = np.random.randn(policy.skill_dim())  # random skill for viz

        traj_states = []
        traj_skills = []
        traj_phi = []

        done = False
        while not done:
            action = policy.act(obs, skill)
            next_obs, reward, done, info = env.step(action)

            traj_states.append(obs)
            traj_skills.append(skill)
            if hasattr(policy, "phi"):      # LSD exposes φ
                traj_phi.append(policy.phi(obs))

            obs = next_obs

        all_states.append(np.array(traj_states))
        all_skills.append(np.array(traj_skills))
        if traj_phi:
            all_phi.append(np.array(traj_phi))

    # 4. Visualize state-space
    plot_state_trajectory(
        env,
        all_states,
        title=f"{baseline_name} – State Space",
        out_path=f"{out_dir}/{baseline_name}_state_traj.png"
    )

    # 5. Visualize φ-space (if available)
    if all_phi:
        plot_phi_trajectory(
            all_phi,
            title=f"{baseline_name} – φ-space Trajectory",
            out_path=f"{out_dir}/{baseline_name}_phi_traj.png"
        )

    print(f"Visualization saved to {out_dir}")
