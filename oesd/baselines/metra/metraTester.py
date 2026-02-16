import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os
import tempfile
from matplotlib.animation import FuncAnimation

def extract_state(obs, env):

    '''current_env = env.env
    
    while hasattr(current_env, 'env'):
        if hasattr(current_env, 'agent_pos'):
            x, y = current_env.agent_pos
            return np.array([x, y], dtype=np.float32)
        current_env = current_env.env

    if hasattr(current_env, 'agent_pos'):
        x, y = current_env.agent_pos
        return np.array([x, y], dtype=np.float32)
    
    if isinstance(obs, dict):
        if 'agent_pos' in obs:
            return np.array(obs['agent_pos'], dtype=np.float32)
        elif 'observation' in obs and hasattr(obs['observation'], 'agent_pos'):
            return np.array(obs['observation'].agent_pos, dtype=np.float32)
    
    # Last resort: return zeros or raise error
    print("Warning: Could not extract agent position, returning zeros")
    return np.array([0, 0], dtype=np.float32)'''
    if isinstance(obs, tuple):
        obs = obs[0]

    if not isinstance(obs, dict):
        raise RuntimeError("MiniGrid obs must be a dict")

    img = obs["image"]      

    return img.astype(np.float32).flatten()   

class MetraWrapper:
    def __init__(self, env):
        self.env = env
        self.s = None
        self.reset()

    def reset(self):
        try:
            obs, info = self.env.reset()
        except:
            obs = self.env.reset()
            info = {}
        self.s = extract_state(obs, self)
        return self.s

    def step(self, a):
        try:
            obs_next, reward, terminated, truncated, info = self.env.step(a)
        except:
            obs_next, reward, done, info = self.env.step(a)
            terminated = done
            truncated = False
            
        s_next = extract_state(obs_next, self)
        old_s = self.s
        self.s = s_next
        done = terminated or truncated
        return old_s, s_next, done, info

        
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()

class DiscreteSkillTester:
    def __init__(self, env, phi, policy, skill_embeddings, n_skills=5, device='cpu',video_dir="oesd\\baselines\\metra\\recs"):
        self.env = env
        self.phi = phi
        self.policy = policy
        self.skill_embeddings = skill_embeddings
        self.n_skills = n_skills
        self.device = device
        
        self.phi.eval()
        self.policy.eval()
        self.skill_embeddings.eval()
        
        if video_dir is None:
            self.temp_dir = tempfile.mkdtemp()
        else:
            self.temp_dir = video_dir
            os.makedirs(self.temp_dir, exist_ok=True)
        print(f"Video directory: {self.temp_dir}")
        
    def can_record_video(self):
    
        try:
            
            if hasattr(self.env, 'render_mode'):
                return self.env.render_mode in ['rgb_array', 'rgb_array_list']
            return False
        except:
            return False
        
    def test_skill(self, skill_idx, render=False, max_steps=200, record_video=False):
        print(f"Testing Skill {skill_idx}")
        
        env_to_use = self.env
        close_env = False
        
        if record_video and self.can_record_video():
            try:
                video_dir = os.path.join(self.temp_dir, f"skill_{skill_idx}")
                os.makedirs(video_dir, exist_ok=True)
                env_wrapped = RecordVideo(self.env, video_dir, name_prefix=f"skill_{skill_idx}")
                env_to_use = env_wrapped
                close_env = True
                print(f"Recording video for skill {skill_idx}")
            except Exception as e:
                print(f"Video recording failed: {e}. Continuing without video.")
                record_video = False
        else:
            if record_video:
                print(f"Cannot record video for skill {skill_idx}. Environment render_mode: {getattr(self.env, 'render_mode', 'None')}")
            record_video = False
        
        env = MetraWrapper(env_to_use)
        
        states = []
        latent_vals = []
        actions = []
        
        s = env.reset()
        done = False
        step = 0
        
        while not done and step < max_steps:
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                phi_s = self.phi(s_tensor).cpu().numpy().flatten()
            
            skill_tensor = torch.tensor(skill_idx, device=self.device)
            
            a = self.policy.sample_action(s_tensor, skill_tensor)
            
            s_prev, s_next, done, info = env.step(a)
            
            states.append(s_next.copy())
            latent_vals.append(phi_s)
            actions.append(a)
            
            if render:
                env.env.render()
            
            step += 1
            s = s_next
        
        if close_env:
            env_wrapped.close()
        
        env.close()
        
        print(f"Skill {skill_idx}: Total steps: {len(states)}")
        print(f"Final pos: {states[-1] if len(states) > 0 else None}")
        
        return states, actions, latent_vals
    
    def test_all_skills(self, render=False, max_steps=200, record_video=False, 
                       plot_trajectories=True, plot_latent_space=True, plot_unified_grid=True):
        results = {}
        all_latent_vals = []
        
        for skill_idx in range(self.n_skills):
            states, actions, latent_vals = self.test_skill(
                skill_idx, render, max_steps, record_video
            )
            results[skill_idx] = {
                'states': states,
                'actions': actions,
                'latent_vals': latent_vals,
                'final_position': states[-1] if states else None
            }
            all_latent_vals.extend(latent_vals)
        
        # Generate plots based on options
        if plot_trajectories:
            self.plot_individual_trajectories(results)
        
        if plot_latent_space and all_latent_vals:
            self.plot_latent_space(results)
        
        if plot_unified_grid:
            self.plot_unified_grid(results)
        
        return results
    
    def plot_individual_trajectories(self, results):

        for skill_idx, result in results.items():
            if result['states']:
                self.plot_trajectory(result['states'], skill_idx)
    
    def plot_trajectory(self, states, skill_idx):
        states = np.array(states)
        plt.figure(figsize=(8, 6))
        plt.plot(states[:, 0], states[:, 1], marker='o', linewidth=2, markersize=4)
        plt.title(f"Skill {skill_idx} Trajectory")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_latent_space(self, results):

        plt.figure(figsize=(12, 8))
        
        all_latent = []
        skill_labels = []
        
        for skill_idx, result in results.items():
            if result['latent_vals']:
                latent_array = np.array(result['latent_vals'])
                
                mean_latent = np.mean(latent_array, axis=0)
                all_latent.append(mean_latent)
                skill_labels.append(skill_idx)
        
        if len(all_latent) < 2:
            print("Not enough latent dimensions to plot")
            return
        
        all_latent = np.array(all_latent)
        
        if all_latent.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(all_latent)
            print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        else:
            latent_2d = all_latent
        
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                            c=skill_labels, cmap='tab10', s=100, alpha=0.7)
        
        for i, (x, y) in enumerate(latent_2d):
            plt.annotate(f'S{skill_labels[i]}', (x, y), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        plt.colorbar(scatter, label='Skill Index')
        plt.title("Latent Skill Space Visualization")
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_unified_grid(self, results):

        plt.figure(figsize=(12, 10))
        
        colors = plt.cm.Set1(np.linspace(0, 1, self.n_skills))
        
        all_x, all_y = [], []
        for skill_idx, result in results.items():
            if result['states']:
                states = np.array(result['states'])
                all_x.extend(states[:, 0])
                all_y.extend(states[:, 1])
        
        if not all_x:
            return
            
        for skill_idx, result in results.items():
            if result['states']:
                states = np.array(result['states'])
                plt.plot(states[:, 0], states[:, 1], marker='o', 
                        linewidth=3, markersize=6, color=colors[skill_idx],
                        label=f'Skill {skill_idx}', alpha=0.8)
                
                
                plt.scatter(states[0, 0], states[0, 1], 
                          color=colors[skill_idx], s=200, marker='s', 
                          edgecolors='black', linewidth=2, label=f'Start {skill_idx}')
                plt.scatter(states[-1, 0], states[-1, 1], 
                          color=colors[skill_idx], s=200, marker='*', 
                          edgecolors='black', linewidth=2, label=f'End {skill_idx}')
        
        plt.title("Unified Grid: All Skill Trajectories", fontsize=16)
        plt.xlabel("X Position", fontsize=12)
        plt.ylabel("Y Position", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()
    
    def create_animated_trajectory(self, results):

        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, self.n_skills))
        
        lines = []
        points = []
        for skill_idx in range(self.n_skills):
            line, = ax.plot([], [], 'o-', color=colors[skill_idx], 
                          linewidth=2, markersize=6, label=f'Skill {skill_idx}')
            point, = ax.plot([], [], 'o', color=colors[skill_idx], 
                           markersize=10, markeredgecolor='black')
            lines.append(line)
            points.append(point)
        
        max_length = max(len(result['states']) for result in results.values() if result['states'])
        all_states = [result['states'] for result in results.values() if result['states']]
        
        if not all_states:
            return
            
        all_coords = np.vstack([np.array(states) for states in all_states])
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
        
        margin_x = (x_max - x_min) * 0.1
        margin_y = (y_max - y_min) * 0.1
        
        def animate(frame):
            for skill_idx, result in results.items():
                if result['states']:
                    states = np.array(result['states'])
                    current_frame = min(frame, len(states) - 1)
                    
                    lines[skill_idx].set_data(states[:current_frame+1, 0], 
                                            states[:current_frame+1, 1])

                    points[skill_idx].set_data([states[current_frame, 0]], 
                                             [states[current_frame, 1]])
            
            return lines + points
        
        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(y_min - margin_y, y_max + margin_y)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Animated Skill Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        anim = FuncAnimation(fig, animate, frames=max_length, 
                           interval=200, blit=True, repeat=True)
        
        plt.tight_layout()
        plt.show()
        
        return anim