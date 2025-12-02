# dads_minigrid.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from example_minigrid import SimpleEnv

import os
os.environ["SDL_AUDIODRIVER"] = "dummy"   # <- add this line

# ======================
#  CONFIG (easy to tweak)
# ======================

CONFIG = {
    "num_epochs": 2000,          # keep small at first so training is quick
    "steps_per_epoch": 80,      # env steps per epoch
    "batch_size": 128,
    "num_neg_samples": 20,      # K skills for DADS marginal
    "lr_dynamics": 3e-4,
    "lr_policy": 3e-4,
    "buffer_capacity": 50000,

    # Visualization during training
    "render_during_training": False,   # show MiniGrid while training
    "render_every": 50,               # render on these epochs (0, 20, 40, ...)
    "log_every": 50,                  # print stats every N epochs

    # Testing / skill demo
    "test_max_steps": 80,
    "num_demo_skills": 6,
}


# ======================
#  State extraction + wrapper
# ======================

def extract_state(obs, env_wrapper):
    """
    Simple 2D state: agent (x, y) position from MiniGrid.
    """
    x, y = env_wrapper.env.agent_pos
    return np.array([x, y], dtype=np.float32)


class DadsWrapper:
    """
    Thin wrapper around SimpleEnv to expose:
    - reset() -> s (np.array of shape (2,))
    - step(a) -> (s_old, s_next, done, info)
    """
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


# ======================
#  Networks
# ======================

class PolicyNet(nn.Module):
    """
    π(a | s, z)
    """
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
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()


class DynamicsNet(nn.Module):
    """
    Skill-conditioned dynamics p(s_next | s, z) as a diagonal Gaussian.

    We model:
    s_next ~ N( s + delta(s, z), diag(std^2) )
    """
    def __init__(self, state_dim=2, latent_dim=2, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim)  # predict delta_mu
        )
        self.log_std = nn.Parameter(torch.zeros(state_dim))

    def forward(self, s, z):
        """
        s, z: (B, ...)
        returns: mean_next (B, state_dim), log_std (state_dim,)
        """
        x = torch.cat([s, z], dim=-1)
        delta_mu = self.net(x)
        mean_next = s + delta_mu
        return mean_next, self.log_std

    def log_prob(self, s, z, s_next):
        """
        log p(s_next | s, z) under diagonal Gaussian.
        s, z, s_next: (B, dim)
        returns: (B,) log-prob
        """
        mean_next, log_std = self.forward(s, z)
        std = torch.exp(log_std)          # (state_dim,)
        var = std ** 2

        diff = s_next - mean_next         # (B, state_dim)
        quad = (diff ** 2) / var          # (B, state_dim)
        quad = quad.sum(dim=-1)           # (B,)

        d = s.shape[-1]
        log_det = 2 * log_std.sum()
        log_const = d * np.log(2 * np.pi)

        logp = -0.5 * (quad + log_det + log_const)
        return logp


# ======================
#  Replay buffer & skills
# ======================

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, z, a, s_next):
        self.buffer.append((s, z, a, s_next))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        s, z, a, s_next = zip(*batch)

        s = torch.from_numpy(np.array(s, dtype=np.float32))
        z = torch.from_numpy(np.array(z, dtype=np.float32))
        a = torch.tensor(a, dtype=torch.int64)
        s_next = torch.from_numpy(np.array(s_next, dtype=np.float32))

        return s, z, a, s_next

    def __len__(self):
        return len(self.buffer)


def sample_skill(dim=2):
    """
    z ~ N(0, I), normalized.
    """
    z = np.random.randn(dim).astype(np.float32)
    return z / np.linalg.norm(z)


# ======================
#  DADS update
# ======================

def dads_update(
    dynamics,
    policy,
    buffer,
    dyn_opt,
    pol_opt,
    batch_size=64,
    num_neg_samples=10
):
    """
    One DADS update step.

    Returns a small dict of stats for logging.
    """
    if len(buffer) < batch_size:
        return None

    s, z, a, s_next = buffer.sample(batch_size)

    # -------- 1) Dynamics update: maximize log p(s_next | s, z) --------
    logp_pos = dynamics.log_prob(s, z, s_next)   # (B,)
    dyn_loss = -logp_pos.mean()

    dyn_opt.zero_grad()
    dyn_loss.backward()
    dyn_opt.step()

    # -------- 2) Policy update with DADS reward --------
    with torch.no_grad():
        # log p(s_next | s, z)
        logp_pos = dynamics.log_prob(s, z, s_next)  # (B,)

        # negative/marginal term: log p(s_next | s)
        # approx via random skills z_neg
        B = s.shape[0]
        K = num_neg_samples
        latent_dim = z.shape[-1]

        z_neg_list = [
            torch.tensor(sample_skill(latent_dim), dtype=torch.float32)
            for _ in range(K)
        ]
        z_neg = torch.stack(z_neg_list, dim=0)  # (K, latent_dim)

        # expand to (B*K, dim)
        s_expanded = s.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
        s_next_expanded = s_next.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
        z_neg_expanded = z_neg.unsqueeze(0).expand(B, K, -1).reshape(B * K, -1)

        logp_neg_flat = dynamics.log_prob(s_expanded, z_neg_expanded, s_next_expanded)  # (B*K,)
        logp_neg = logp_neg_flat.view(B, K)  # (B, K)

        # log (1/K sum_j p_ij) = logsumexp(logp_ij) - log(K)
        logp_marginal = torch.logsumexp(logp_neg, dim=1) - np.log(K)  # (B,)

        reward = logp_pos - logp_marginal                     # (B,)
        reward_mean = reward.mean().item()
        reward_std = reward.std().item()
        reward_norm = (reward - reward_mean) / (reward_std + 1e-8)

    # REINFORCE update for policy
    logits = policy(s, z)
    log_probs = torch.log_softmax(logits, dim=1)
    chosen_log_prob = log_probs[range(batch_size), a]

    pol_loss = -torch.mean(chosen_log_prob * reward_norm)

    pol_opt.zero_grad()
    pol_loss.backward()
    pol_opt.step()

    return {
        "dyn_loss": dyn_loss.item(),
        "pol_loss": pol_loss.item(),
        "reward_mean": reward_mean,
        "reward_std": reward_std,
    }


# ======================
#  Training loop
# ======================

def train_dads(config: dict):
    # Render during training if requested
    train_env_raw = SimpleEnv(
        render_mode="human" if config["render_during_training"] else None
    )
    env = DadsWrapper(train_env_raw)

    state_dim = 2
    latent_dim = 2
    n_actions = env.env.action_space.n

    dynamics = DynamicsNet(state_dim=state_dim, latent_dim=latent_dim)
    policy = PolicyNet(state_dim=state_dim, latent_dim=latent_dim, n_actions=n_actions)

    dyn_opt = optim.Adam(dynamics.parameters(), lr=config["lr_dynamics"])
    pol_opt = optim.Adam(policy.parameters(), lr=config["lr_policy"])

    buffer = ReplayBuffer(capacity=config["buffer_capacity"])

    for epoch in range(config["num_epochs"]):
        # Sample one skill per epoch
        z_np = sample_skill(dim=latent_dim)
        z_t = torch.tensor(z_np, dtype=torch.float32).unsqueeze(0)  # (1, latent_dim)

        s = env.reset()
        for t in range(config["steps_per_epoch"]):
            s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)  # (1, state_dim)
            a = policy.sample_action(s_t, z_t)

            # Render occasionally during training
            if config["render_during_training"] and epoch % config["render_every"] == 0:
                env.env.render()

            s, s_next, done, _ = env.step(a)
            buffer.add(s, z_np, a, s_next)
            s = s_next

            if done:
                s = env.reset()

        stats = dads_update(
            dynamics, policy, buffer,
            dyn_opt, pol_opt,
            batch_size=config["batch_size"],
            num_neg_samples=config["num_neg_samples"]
        )

        if epoch % config["log_every"] == 0:
            if stats is None:
                print(f"[Epoch {epoch}] buffer={len(buffer)} (warming up)")
            else:
                print(
                    f"[Epoch {epoch}] buffer={len(buffer)}, "
                    f"dyn_loss={stats['dyn_loss']:.3f}, "
                    f"pol_loss={stats['pol_loss']:.3f}, "
                    f"r_mean={stats['reward_mean']:.3f}"
                )

    return dynamics, policy


# ======================
#  Tester: show skills one by one
# ======================

class DadsTester:
    """
    Use a trained policy to:
    - roll out with a fixed skill z
    - render each skill in the MiniGrid window
    """
    def __init__(self, env, policy, latent_dim=2, max_steps=60):
        self.env = DadsWrapper(env)   # wrap SimpleEnv
        self.policy = policy
        self.latent_dim = latent_dim
        self.max_steps = max_steps

    def sample_skill(self):
        return sample_skill(self.latent_dim)

    def rollout(self, z=None, render=False):
        if z is None:
            z_np = self.sample_skill()
        else:
            z_np = np.array(z, dtype=np.float32)
            norm = np.linalg.norm(z_np)
            if norm > 0:
                z_np = z_np / norm

        z_t = torch.tensor(z_np, dtype=torch.float32).unsqueeze(0)

        s = self.env.reset()
        done = False
        traj = [s.copy()]

        steps = 0
        while not done and steps < self.max_steps:
            if render:
                self.env.env.render()

            s_t = torch.tensor(self.env.s, dtype=torch.float32).unsqueeze(0)
            a = self.policy.sample_action(s_t, z_t)

            _, _, done, _ = self.env.step(a)
            traj.append(self.env.s.copy())
            steps += 1

        return z_np, np.array(traj, dtype=np.float32)

    def demo_one_skill(self):
        """
        Sample one skill, print it, and render the rollout.
        """
        z, traj = self.rollout(render=True)
        print("=== Skill demo ===")
        print("z =", np.round(z, 3))
        print("trajectory length:", len(traj))
        print("start:", traj[0], "end:", traj[-1])


# ======================
#  Main: train + show skills
# ======================

def main():
    # 1) Train DADS (short run by default)
    print("Training DADS on SimpleEnv...")
    dynamics, policy = train_dads(CONFIG)

    # Optional: save models
    torch.save(dynamics.state_dict(), "dads_dynamics_minigrid.pth")
    torch.save(policy.state_dict(), "dads_policy_minigrid.pth")

    # 2) New env for testing / demos with rendering
    test_env = SimpleEnv(render_mode="human")

    # Reload from disk just to show the full loop
    dynamics2 = DynamicsNet()
    dynamics2.load_state_dict(torch.load("dads_dynamics_minigrid.pth"))

    policy2 = PolicyNet(n_actions=test_env.action_space.n)
    policy2.load_state_dict(torch.load("dads_policy_minigrid.pth"))

    dynamics2.eval()
    policy2.eval()

    tester = DadsTester(
        env=test_env,
        policy=policy2,
        latent_dim=2,
        max_steps=CONFIG["test_max_steps"]
    )

    print("\nShowing learned skills one by one...")
    for i in range(CONFIG["num_demo_skills"]):
        print(f"\n--- Demo skill {i} ---")
        tester.demo_one_skill()

    print("Done.")
    return "done"


if __name__ == "__main__":
    print(main())
