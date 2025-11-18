import numpy as np
import torch
import matplotlib.pyplot as plt


def extract_state(obs, env):
    

    x, y = env.env.agent_pos
    return np.array([x, y], dtype=np.float32)

class MetraWrapper:
    def __init__(self, env):
        self.env = env
        self.s = None
        self.reset()

    def reset(self):
        obs, info = self.env.reset()
        self.s = extract_state(obs, self)
        return self.s

    def step(self, a):
        obs_next, reward, terminated, truncated, info = self.env.step(a)
        s_next = extract_state(obs_next, self)
        old_s = self.s
        self.s = s_next
        done = terminated or truncated
        return old_s, s_next, done, info


class metraTester:
    def __init__(self, env, phi, policy):
        self.env = MetraWrapper(env)
        self.phi = phi
        self.policy = policy

    def sample_skill(self, dim=2):
        z = np.random.randn(dim).astype(np.float32)
        return z / np.linalg.norm(z)

    def test(self, render=False, max_steps=200):
        z = torch.tensor(self.sample_skill(), dtype=torch.float32).unsqueeze(0)
        print("\n=== Running METRA Test ===")
        print(f"Sampled skill z = {z.numpy().round(3)}")

        states = []
        latent_vals = []
        actions = []

        s = self.env.reset()
        done = False
        step = 0

        if render:
            obs, _ = self.env.env.reset()

        while not done and step < max_steps:
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)

            
            with torch.no_grad():
                phi_s = self.phi(s_tensor).cpu().numpy().flatten()

            
            a = self.policy.sample_action(s_tensor, z)

            
            s_prev, s_next, done, info = self.env.step(a)

            # Logging
            states.append(s_next.copy())
            latent_vals.append(phi_s)
            actions.append(a)

            if render:
                self.env.env.render()

            step += 1
            s = s_next

        print(f"Total steps: {len(states)}")
        print(f"Final pos: {states[-1] if len(states) > 0 else None}")

        if len(states) > 1:
            self.plot_trajectory(states, z)

        if len(latent_vals) > 1:
            self.plot_latent(latent_vals)

       
        if len(actions) > 1:
            self.plot_actions(actions)

       

    def plot_trajectory(self, states, z):
        states = np.array(states)
        plt.figure()
        plt.plot(states[:,0], states[:,1], marker='o')
        plt.title(f"Agent Trajectory (skill={z.numpy().round(2)})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.show()

    def plot_latent(self, latent_vals):
        latent_vals = np.array(latent_vals)
        plt.figure()
        plt.plot(latent_vals)
        plt.title("φ(s) over time")
        plt.xlabel("Step")
        plt.ylabel("φ value")
        plt.grid()
        plt.show()

    def plot_actions(self, actions):
        actions = np.array(actions)
        plt.figure()
        plt.plot(actions)
        plt.ylabel("Action index")
        plt.xlabel("Timestep")
        plt.title("Actions chosen over time")
        plt.grid(True)
        plt.show()

        #plt.title("Action magnitude over time")
        #plt.xlabel("Step")
        #plt.ylabel("||a||")
        #plt.grid()
        #plt.show()
