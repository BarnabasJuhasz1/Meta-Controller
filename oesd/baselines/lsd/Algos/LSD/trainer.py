# Algos/LSD/trainer.py

from __future__ import annotations
import os
import random
from datetime import datetime
from typing import List, Tuple
from dataclasses import dataclass
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import scripts.testing.example_minigrid as example_minigrid

from Interface.policy_base import SkillPolicy
from Algos.LSD.policy import PhiNet, ActorCritic, LSDPolicy


# ============================================================
# CONFIG
# ============================================================

@dataclass
class LSDConfig:
    size: int = 10
    max_steps: int | None = None
    seed: int = 0

    # Training
    num_episodes: int = 2000
    max_steps_per_episode: int = 30
    gamma: float = 0.99
    lr_policy: float = 3e-4
    lr_phi: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Skills
    discrete: bool = True
    num_skills: int = 8
    skill_dim: int = 8

    # Networks
    phi_hidden_dim: int = 128
    policy_hidden_dim: int = 256

    # Loss coefficients
    value_coef: float = 0.5
    entropy_coef: float = 0.10
    phi_coef: float = 5.0

    # Replay
    replay_capacity: int = 150_000
    batch_size: int = 128
    phi_updates_per_episode: int = 32

    reward_clip: float = 5.0
    log_interval: int = 50


# ============================================================
# RUNNING NORMALIZATION
# ============================================================

class RunningNorm:
    def __init__(self, eps=1e-5):
        self.mean = None
        self.var = None
        self.count = eps

    def update(self, x):
        x = x.astype(np.float32)
        if self.mean is None:
            self.mean = x.copy()
            self.var = np.ones_like(x)
            self.count = 1.0
            return

        batch_mean = x
        batch_var = np.zeros_like(x)
        batch_count = 1.0

        delta = batch_mean - self.mean
        total = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total
        new_var = m2 / total

        self.mean = new_mean
        self.var = new_var
        self.count = total

    def normalize(self, x):
        if self.mean is None:
            return x.astype(np.float32)
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# ============================================================
# REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, skill_dim, device):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.skill = np.zeros((capacity, skill_dim), np.float32)

        self.idx = 0
        self.full = False

    def add(self, s, s2, z):
        self.obs[self.idx] = s
        self.next_obs[self.idx] = s2
        self.skill[self.idx] = z

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def size(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size):
        n = self.size()
        if n == 0:
            return None
        idx = np.random.randint(0, n, size=min(batch_size, n))

        return (
            torch.tensor(self.obs[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.next_obs[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.skill[idx], dtype=torch.float32, device=self.device),
        )


# ============================================================
# LSD TRAINER (all logic preserved)
# ============================================================

class LSDTrainer:
    def __init__(self, cfg: LSDConfig, run_dir: str):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.run_dir = run_dir

        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        self.skills_dir = os.path.join(run_dir, "skills")
        self.phi_dir = os.path.join(run_dir, "phi")
        self.logs_dir = os.path.join(run_dir, "logs")



        # ============================
        #  ENV SETUP
        # ============================
        self.env = example_minigrid.SimpleEnv(
            size=cfg.size, max_steps=cfg.max_steps, render_mode=None
        )

        self._seed(cfg.seed)

        # obs_dim
        obs = self._reset_env()
        obs_vec = self._obs_to_vec(obs)
        self.obs_norm = RunningNorm()
        self.obs_norm.update(obs_vec)
        obs_n = self.obs_norm.normalize(obs_vec)
        self.obs_dim = obs_n.shape[0]

        self.action_dim = self.env.action_space.n

        # ============================
        # SKILLS
        # ============================
        self.skill_dim = cfg.skill_dim
        self.discrete_Z = self._build_discrete_skills(cfg.num_skills)

        # ============================
        # NETWORKS
        # ============================
        self.phi = PhiNet(
            obs_dim=self.obs_dim,
            skill_dim=self.skill_dim,
            hidden_dim=cfg.phi_hidden_dim,
        ).to(self.device)

        self.actor_critic = ActorCritic(
            obs_dim=self.obs_dim,
            skill_dim=self.skill_dim,
            action_dim=self.action_dim,
            hidden_dim=cfg.policy_hidden_dim,
        ).to(self.device)

        self.policy = LSDPolicy(
            actor=self.actor_critic,
            phi_net=self.phi,
            num_skills=cfg.num_skills,
            device=self.device,
        )

        self.opt_phi = torch.optim.Adam(self.phi.parameters(), lr=cfg.lr_phi)
        self.opt_policy = torch.optim.Adam(self.actor_critic.parameters(), lr=cfg.lr_policy)

        # ============================
        # REPLAY
        # ============================
        self.replay = ReplayBuffer(
            cfg.replay_capacity, self.obs_dim, self.skill_dim, self.device
        )

        # prefill
        self._prefill_replay(per_skill_episodes=2, max_steps=50)

        # logging storage
        self.phi_points = deque(maxlen=5000)
        self.phi_colors = deque(maxlen=5000)

        print(f"[init] obs_dim={self.obs_dim}, skill_dim={self.skill_dim}")

    # ============================================================
    # UTILS
    # ============================================================

    def _seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            self.env.reset(seed=seed)
        except:
            pass

    def _reset_env(self):
        out = self.env.reset()
        return out[0] if isinstance(out, tuple) else out

    def _step_env(self, act):
        out = self.env.step(act)
        if len(out) == 5:  # gymnasium
            obs, rew, terminated, truncated, info = out
            done = terminated or truncated
        else:
            obs, rew, done, info = out
        return obs, rew, done, info

    def _obs_to_vec(self, obs):
        img = obs["image"]
        return img.astype(np.float32).flatten()

    def _obs_to_tensor(self, vec):
        self.obs_norm.update(vec)
        n = self.obs_norm.normalize(vec)
        return torch.tensor(n, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _build_discrete_skills(self, K):
        Z = np.full((K, K), -1/(K-1), np.float32)
        for i in range(K):
            Z[i, i] = 1.0
        return torch.tensor(Z, device=self.device)

    def _sample_skill(self):
        K = self.cfg.num_skills
        i = np.random.randint(K)
        return self.discrete_Z[i], i

    def _prefill_replay(self, per_skill_episodes=1, max_steps=50):
        K = self.cfg.num_skills
        for k in range(K):
            z = self.discrete_Z[k]
            for _ in range(per_skill_episodes):
                obs = self._reset_env()
                o = self._obs_to_vec(obs)
                for _ in range(max_steps):
                    act = int(self.env.action_space.sample())
                    obs2, _, done, _ = self._step_env(act)
                    o2 = self._obs_to_vec(obs2)
                    self.replay.add(o, o2, z.cpu().numpy())
                    o = o2
                    if done:
                        break

    # ============================================================
    # TRAINING LOOP
    # ============================================================

    def train(self):
        for ep in range(1, self.cfg.num_episodes+1):
            ret, steps, pl, vl, ent = self._run_episode()

            for _ in range(self.cfg.phi_updates_per_episode):
                self._update_phi()

            if ep % self.cfg.log_interval == 0:
                print(f"[Ep {ep:4d}] ret={ret:+.3f}  P={pl:+.4f}  V={vl:+.4f}  H={ent:.3f}  φbuf={self.replay.size()}")

    # ============================================================
    # EPISODE
    # ============================================================

    def _run_episode(self):
        cfg = self.cfg
        obs = self._reset_env()
        ovec = self._obs_to_vec(obs)
        o = self._obs_to_tensor(ovec)

        z, z_idx = self._sample_skill()
        Zb = z.unsqueeze(0)

        logps = []
        values = []
        ents = []
        rewards = []

        total_ret = 0.0

        for t in range(cfg.max_steps_per_episode):
            logits, val = self.actor_critic(o, Zb)
            dist = Categorical(logits=logits)
            act = dist.sample()

            logp = dist.log_prob(act)
            ent = dist.entropy()

            obs2, _, done, _ = self._step_env(act.item())
            ovec2 = self._obs_to_vec(obs2)
            o2 = self._obs_to_tensor(ovec2)

            with torch.no_grad():
                phi_s = self.phi(o)
                phi_s2 = self.phi(o2)
                diff = phi_s2 - phi_s
                r_phi = (diff * Zb).sum().item()

            r = max(-cfg.reward_clip, min(cfg.reward_clip, r_phi))
            total_ret += r

            self.replay.add(ovec, ovec2, z.cpu().numpy())

            logps.append(logp)
            values.append(val)
            ents.append(ent)
            rewards.append(r)

            if t % 5 == 0:
                self.phi_points.append(phi_s.cpu().numpy().flatten())
                self.phi_colors.append(z_idx)

            o = o2
            ovec = ovec2

            if done:
                break

        returns = self._discount(rewards, cfg.gamma)
        V = torch.cat(values).squeeze(-1)
        adv = returns - V.detach()

        policy_loss = -(torch.cat(logps) * adv).mean()
        value_loss = F.mse_loss(V, returns)
        entropy_loss = -torch.cat(ents).mean()

        loss = (policy_loss
                + cfg.value_coef * value_loss
                + cfg.entropy_coef * entropy_loss)

        self.opt_policy.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1.0)
        self.opt_policy.step()

        return total_ret, t+1, policy_loss.item(), value_loss.item(), torch.cat(ents).mean().item()

    # ============================================================
    # RETURNS
    # ============================================================

    def _discount(self, rewards, gamma):
        R = 0.0
        out = []
        for r in reversed(rewards):
            R = r + gamma * R
            out.append(R)
        out.reverse()
        out = torch.tensor(out, dtype=torch.float32, device=self.device)
        return (out - out.mean()) / (out.std() + 1e-8)

    # ============================================================
    # PHI UPDATE (contrastive LSD)
    # ============================================================

    def _update_phi(self):
        batch = self.replay.sample(self.cfg.batch_size)
        if batch is None:
            return

        s, s2, Zi = batch

        # normalize
        mean = torch.tensor(self.obs_norm.mean, device=self.device)
        var = torch.tensor(self.obs_norm.var, device=self.device)
        s = (s - mean) / (var.sqrt() + 1e-8)
        s2 = (s2 - mean) / (var.sqrt() + 1e-8)

        phi_s = self.phi(s)
        phi_s2 = self.phi(s2)
        diff = phi_s2 - phi_s   # [B, D]

        diff_exp = diff.unsqueeze(1)   # [B,1,D]
        Zall = self.discrete_Z.unsqueeze(0)  # [1,K,D]

        logits = torch.sum(diff_exp * Zall, dim=-1)  # [B,K]

        idx_true = torch.argmax(Zi, dim=-1).long()   # [B]

        loss = F.cross_entropy(logits, idx_true)
        loss = self.cfg.phi_coef * loss

        self.opt_phi.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.phi.parameters(), 10.0)
        self.opt_phi.step()

    # ============================================================
    # ATTEMPT MODE
    # ============================================================

    def attempt_episode(self, skill_idx=None, max_steps=None):
        if max_steps is None:
            max_steps = self.cfg.max_steps_per_episode

        old = self.env
        self.env = example_minigrid.SimpleEnv(
            size=self.cfg.size,
            max_steps=self.cfg.max_steps,
            render_mode="human"
        )

        obs = self._reset_env()
        ovec = self._obs_to_vec(obs)
        o = self._obs_to_tensor(ovec)

        if skill_idx is None:
            z, idx = self._sample_skill()
        else:
            idx = int(skill_idx)
            z = self.discrete_Z[idx]
        Zb = z.unsqueeze(0)

        for _ in range(max_steps):
            self.env.render()
            logits, _ = self.actor_critic(o, Zb)
            act = Categorical(logits=logits).sample().item()

            obs2, _, done, _ = self._step_env(act)
            ovec2 = self._obs_to_vec(obs2)
            o = self._obs_to_tensor(ovec2)

            if done:
                break

        self.env = old

    # ============================================================
    # SKILL VISUALIZATION
    # ============================================================

    def visualize_skills(self, save="skills.png", num_skills=None, max_steps=None):
        K = self.cfg.num_skills if num_skills is None else num_skills
        max_steps = self.cfg.max_steps_per_episode if max_steps is None else max_steps

        env = example_minigrid.SimpleEnv(size=self.cfg.size,
                                         max_steps=self.cfg.max_steps)

        trajs = []

        for k in range(K):
            z = self.discrete_Z[k]
            Zb = z.unsqueeze(0)
            obs = self._reset_env()
            ovec = self._obs_to_vec(obs)
            o = self._obs_to_tensor(ovec)

            path = []
            for _ in range(max_steps):
                if hasattr(env, "agent_pos"):
                    x,y = env.agent_pos
                else:
                    x,y = 0,0
                path.append((x,y))

                logits, _ = self.actor_critic(o, Zb)
                act = Categorical(logits=logits).sample().item()

                obs2, _, done, _ = self._step_env(act)
                ovec2 = self._obs_to_vec(obs2)
                o = self._obs_to_tensor(ovec2)

                if done:
                    break

            trajs.append((f"skill {k}", np.array(path)))

        # timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(save)
        if ext == "":
            ext = ".png"
        save_name = os.path.join(self.skills_dir, f"skills_{ts}.png")



        plt.figure(figsize=(6,6))
        for label, P in trajs:
            plt.plot(P[:,0], P[:,1], label=label)
        plt.legend()
        plt.grid(True)
        plt.title("Skill trajectories")
        plt.savefig(save_name, dpi=200)
        plt.close()

        print(f"[skills] saved {save_name}")

    # ============================================================
    # φ PLOTTING
    # ============================================================

    def plot_phi(self, save="phi_plots.pdf"):
        if not self.phi_points:
            print("[phi] no data")
            return

        X = np.array(self.phi_points)
        C = np.array(self.phi_colors)

        if X.shape[1] != 2:
            from sklearn.decomposition import PCA
            X = PCA(2).fit_transform(X)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(save)
        if ext == "":
            ext = ".pdf"
        save_name = os.path.join(self.phi_dir, f"phi_{ts}.pdf")



        with PdfPages(save_name) as pdf:
            fig = plt.figure(figsize=(6,6))
            plt.scatter(X[:,0], X[:,1], c=C, cmap="tab20", s=8)
            plt.grid(True)
            plt.title("φ-space")
            pdf.savefig(fig)
            plt.close(fig)

        print(f"[phi] saved {save_name}")

    # ============================================================
    # SAVE / LOAD
    # ============================================================

    def save(self, name="latest.pth"):
        path = os.path.join(self.ckpt_dir, name)

        state = {
            "phi": self.phi.state_dict(),
            "policy": self.actor_critic.state_dict(),
            "obs_mean": self.obs_norm.mean,
            "obs_var": self.obs_norm.var,
            "obs_count": self.obs_norm.count,
            "skills": self.discrete_Z.cpu().numpy(),
            "cfg": self.cfg.__dict__
        }

        torch.save(state, path)
        torch.save(state, os.path.join(self.ckpt_dir, "latest.pth"))
        print(f"[save] {path}")


    def resume_from(self, path="lsd01/latest.pth"):
        self.load(path)
        print(f"[resume] loaded {path}")
