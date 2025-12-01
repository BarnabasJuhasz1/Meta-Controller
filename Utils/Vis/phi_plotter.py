# Visualization/phi_plotter.py

import matplotlib.pyplot as plt
import numpy as np

def plot_phi_trajectory(all_phi, title, out_path):
    plt.figure(figsize=(6,6))

    for phi_traj in all_phi:
        if phi_traj.ndim == 2 and phi_traj.shape[1] == 2:
            plt.plot(phi_traj[:,0], phi_traj[:,1], alpha=0.75)
        else:
            # reduce to 2D via PCA
            from sklearn.decomposition import PCA
            phi2 = PCA(2).fit_transform(phi_traj)
            plt.plot(phi2[:,0], phi2[:,1], alpha=0.75)

    plt.title(title)
    plt.xlabel("φ dim 1")
    plt.ylabel("φ dim 2")
    plt.savefig(out_path, dpi=200)
    plt.close()
