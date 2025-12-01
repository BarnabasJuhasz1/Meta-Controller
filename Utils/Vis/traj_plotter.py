# Visualization/traj_plotter.py

import matplotlib.pyplot as plt
import numpy as np

def plot_state_trajectory(env, all_trajs, title, out_path):
    plt.figure(figsize=(6, 6))

    # Draw env background if available (e.g., MiniGrid)
    if hasattr(env, "render_background"):
        env.render_background(plt)

    for traj in all_trajs:
        traj = np.array(traj)
        if traj.ndim == 2 and traj.shape[1] >= 2:  # x, y
            plt.plot(traj[:, 0], traj[:, 1], alpha=0.75)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(out_path, dpi=200)
    plt.close()
