import gymnasium as gym
# from SimpleEnv import example_minigrid
from example_minigrid import SimpleEnv
import numpy as np
import torch.nn as nn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from testerMetra2 import metraTester


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



class PhiNet(nn.Module):
    def __init__(self, state_dim=2, latent_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, s):
        return self.net(s)

class PolicyNet(nn.Module):
    def __init__(self, state_dim=2, latent_dim=2, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, s, z):
        x = torch.cat([s, z], dim=-1)
        return self.net(x)
    
    def sample_action(self, s, z):
        logits = self.forward(s, z)
        probs = torch.softmax(logits, dim=-1)
        return torch.distributions.Categorical(probs).sample().item()


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, z, a, s_next):
        self.buffer.append((s, z, a, s_next))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        s, z, a, s_next = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(z, dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(s_next, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

def sample_skill(dim=2):
    z = np.random.randn(dim).astype(np.float32)
    return z / np.linalg.norm(z)


def metra_update(phi, policy, buffer, phi_opt, pol_opt, lambda_, eps=0.1, batch_size=64):
    if len(buffer) < batch_size:
        return lambda_
    
    s, z, a, s_next = buffer.sample(batch_size)

    phi_s = phi(s)
    phi_s_next = phi(s_next)

    phi_diff = phi_s_next - phi_s

    reward = torch.sum(phi_diff * z, dim=1)

    logits = policy(s, z)
    log_probs = torch.log_softmax(logits, dim=1)
    chosen_log_prob = log_probs[range(batch_size), a]

    policy_loss = -torch.mean(chosen_log_prob * reward.detach())


    norms = torch.norm(phi_diff, dim=1)
    penalty_val = torch.mean(torch.clamp(1 - norms**2, min=0.0))  

    phi_loss = -torch.mean(reward) + lambda_ * penalty_val

    lambda_ += 0.01 * penalty_val.item()
    
    phi_opt.zero_grad()
    phi_loss.backward()
    phi_opt.step()

    pol_opt.zero_grad()
    policy_loss.backward()
    pol_opt.step()


    return lambda_


def train_metra(env, num_epochs=6000, steps_per_epoch=50):
    env = MetraWrapper(env)

    phi = PhiNet()
    policy = PolicyNet(n_actions=env.env.action_space.n)

    phi_opt = optim.Adam(phi.parameters(), lr=3e-4)
    pol_opt = optim.Adam(policy.parameters(), lr=3e-4)

    buffer = ReplayBuffer()
    lambda_ = 0.0

    for epoch in range(num_epochs):
        z = torch.tensor(sample_skill(), dtype=torch.float32).unsqueeze(0)
        s = env.reset()

        for t in range(steps_per_epoch):
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)

            a = policy.sample_action(s_tensor, z)

            s, s_next, done, _ = env.step(a)

            buffer.add(s, z.squeeze().numpy(), a, s_next)

            s = s_next

            if done:
                s = env.reset()

        lambda_ = metra_update(
            phi, policy, buffer,
            phi_opt, pol_opt,
            lambda_
        )

        if epoch % 100 == 0:
            print(f"[Epoch {epoch}] lambda={lambda_:.3f}, buffer={len(buffer)}")

    return phi, policy


def main ():
    env = SimpleEnv()

    #env = MetraWrapper(env)


    #phi, policy = train_metra(env)

    #torch.save(phi.state_dict(), "phi.pth")
    #torch.save(policy.state_dict(), "policy.pth")

    #tester = metraTester(env=env,phi=phi,policy=policy)

    #tester.test(render=False)

    env = SimpleEnv(render_mode="human")

    # 2. Load your trained models
    phi = PhiNet()
    phi.load_state_dict(torch.load("phi.pth"))

    policy = PolicyNet(n_actions=env.action_space.n)
    policy.load_state_dict(torch.load("policy.pth"))

    phi.eval()
    policy.eval()

    # 3. Run test
    tester = metraTester(env, phi, policy)
    tester.test(render=True)     # Live graphical rendering
    # tester.test(render=False)  # Just logs + plots

    return "done"

print(main())