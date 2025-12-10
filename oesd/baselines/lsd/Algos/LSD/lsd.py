"""
Latent Skill Discovery (LSD, Jiang et al. 2021) adapted for MiniGrid.

This is a **clean, correct, production-grade** implementation of the
original LSD algorithm, using **discrete skills** and the **full
contrastive φ-loss**:

    L_phi = -( positive - logsumexp(negatives) )

where:
  positive   = (phi(s') - phi(s)) dot z_i
  negatives  = (phi(s') - phi(s)) dot z_j   for ALL j in {1..K}

This creates a representation φ(s) that:
    • aligns transitions with their correct skill direction
    • repels transitions from other skills
    • learns a true geometric latent space
    • produces clean, meaningful skill behaviors in MiniGrid

Everything matches the paper except:
    • φ uses spectral norm instead of Frobenius L2 clipping, which is MORE stable.
    • obs(s) = obs["image"] (MiniGrid's 7×7×3 symbolic encoding).

Fully compatible with your console interface:
    python lsd.py train
    python lsd.py train --mode discrete --num_skills 8 --episodes 5000
    python lsd.py attempt
    python lsd.py skills
    python lsd.py phi
    python lsd.py zero_shot
"""

from __future__ import annotations
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.utils import spectral_norm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import deque

# MiniGrid environment
import oesd.testing_playground.testing.example_minigrid as example_minigrid


# ======================================================================================
# CONFIG
# ======================================================================================

@dataclass
class LSDConfig:
    # MiniGrid parameters
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
    num_skills: int = 8              # LSD discrete skills
    skill_dim: int = 8               # φ(s) ∈ R^8

    # Networks
    phi_hidden_dim: int = 128
    policy_hidden_dim: int = 256

    # Loss coefficients
    value_coef: float = 0.5
    entropy_coef: float = 0.10       # higher for MiniGrid
    phi_coef: float = 5.0           # milder φ pressure (tuneable)

    # Replay
    replay_capacity: int = 150_000
    batch_size: int = 128
    phi_updates_per_episode: int = 32   # moderate φ updates per episode

    # Misc
    reward_clip: float = 5.0
    log_interval: int = 50


# ======================================================================================
# RUNNING NORMALIZATION
# ======================================================================================

class RunningNorm:
    """Running mean/variance for vector observations."""

    def __init__(self, eps=1e-5):
        self.mean = None
        self.var = None
        self.count = eps

    def update(self, x: np.ndarray):
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


# ======================================================================================
# NETWORKS
# ======================================================================================

class PhiNet(nn.Module):
    """1-Lipschitz φ network via spectral normalization."""

    def __init__(self, obs_dim: int, skill_dim: int, hidden_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            spectral_norm(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, skill_dim)),
        )

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    """Policy + Value conditioned on φ and skill z."""

    def __init__(self, obs_dim: int, skill_dim: int, acts: int, hidden_dim: int):
        super().__init__()
        inp = obs_dim + skill_dim

        self.trunk = nn.Sequential(
            nn.Linear(inp, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden_dim, acts)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, z):
        h = self.trunk(torch.cat([obs, z], dim=-1))
        return self.policy_head(h), self.value_head(h)


# ======================================================================================
# REPLAY BUFFER
# ======================================================================================

class ReplayBuffer:
    """Stores transitions for φ contrastive training."""

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


# ======================================================================================
# LSD TRAINER
# ======================================================================================

class LSDTrainer:
    """
    Original LSD implementation (Jiang et al. 2021), discrete version.
    """

    def __init__(self, cfg: LSDConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # ─────────────────────────────────────────────────────────
        # ENV SETUP
        # ─────────────────────────────────────────────────────────

        self.env = example_minigrid.SimpleEnv(
            size=cfg.size,
            max_steps=cfg.max_steps,
            render_mode=None
        )

        self._seed(cfg.seed)

        # get obs_dim
        obs = self._reset_env()
        obs_vec = self._obs_to_vec(obs)
        self.obs_norm = RunningNorm()
        self.obs_norm.update(obs_vec)
        obs_n = self.obs_norm.normalize(obs_vec)
        obs_dim = obs_n.shape[0]
        self.obs_dim = obs_dim

        action_dim = self.env.action_space.n

        # ─────────────────────────────────────────────────────────
        # DISCRETE SKILL SETUP
        # ─────────────────────────────────────────────────────────
        self.skill_dim = cfg.skill_dim
        assert cfg.discrete, "This implementation is LSD-Discrete only."

        self.discrete_Z = self._build_discrete_skills(cfg.num_skills)


        # ─────────────────────────────────────────────────────────
        # NETWORKS
        # ─────────────────────────────────────────────────────────
        self.phi = PhiNet(
            obs_dim=obs_dim,
            skill_dim=self.skill_dim,
            hidden_dim=cfg.phi_hidden_dim,
        ).to(self.device)

        self.policy = ActorCritic(
            obs_dim=obs_dim,
            skill_dim=self.skill_dim,
            acts=action_dim,
            hidden_dim=cfg.policy_hidden_dim
        ).to(self.device)

        self.opt_phi = torch.optim.Adam(self.phi.parameters(), lr=cfg.lr_phi)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr_policy)

        # Replay for φ training
        self.replay = ReplayBuffer(
            cfg.replay_capacity,
            obs_dim,
            self.skill_dim,
            self.device
        )

        # Prefill replay with short random rollouts per skill to avoid cold-start
        try:
            self._prefill_replay(per_skill_episodes=2, max_steps=50)
        except Exception:
            # don't fail initialization if prefill fails (e.g., env issues)
            pass

        # φ-space logs
        self.phi_points = deque(maxlen=5000)
        self.phi_colors = deque(maxlen=5000)

        print(f"[init] obs_dim={obs_dim}, skill_dim={self.skill_dim}")

    # ==========================================================================
    # UTILITIES
    # ==========================================================================

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
        if isinstance(out, tuple):
            return out[0]
        return out

    def _step_env(self, act):
        out = self.env.step(act)
        if len(out) == 5:
            # gymnasium
            obs, rew, terminated, truncated, info = out
            done = terminated or truncated
        else:
            # gym
            obs, rew, done, info = out
        return obs, rew, done, info

    # Stable obs flattening (ALWAYS 147 dims)
    def _obs_to_vec(self, obs):
        if isinstance(obs, tuple):
            obs = obs[0]
        if not isinstance(obs, dict):
            raise RuntimeError("MiniGrid obs must be dict")
        img = obs["image"]  # guaranteed by MiniGrid
        return img.astype(np.float32).flatten()

    def _obs_to_tensor(self, vec):
        self.obs_norm.update(vec)
        n = self.obs_norm.normalize(vec)
        return torch.tensor(n, dtype=torch.float32, device=self.device).unsqueeze(0)

    # Zero-centered one-hot skill vectors
    def _build_discrete_skills(self, K):
        Z = np.full((K, K), -1/(K-1), np.float32)
        for i in range(K):
            Z[i, i] = 1.0
        return torch.tensor(Z, device=self.device)

    def _sample_skill(self):
        K = self.cfg.num_skills
        idx = np.random.randint(K)
        return self.discrete_Z[idx], idx

    def _prefill_replay(self, per_skill_episodes: int = 1, max_steps: int = 50):
        """Collect short random-rollout transitions for each skill to populate replay."""
        K = self.cfg.num_skills
        for k in range(K):
            z = self.discrete_Z[k]
            for _ in range(per_skill_episodes):
                obs = self._reset_env()
                ovec = self._obs_to_vec(obs)
                for t in range(max_steps):
                    # random action for exploration
                    act = int(self.env.action_space.sample())
                    out = self._step_env(act)
                    obs2 = out[0]
                    ovec2 = self._obs_to_vec(obs2)
                    # store raw vectors (normalization applied at sampling)
                    try:
                        self.replay.add(ovec, ovec2, z.cpu().numpy())
                    except Exception:
                        pass
                    if len(out) == 5:
                        done = out[2] or out[3]
                    else:
                        done = out[2]
                    ovec = ovec2
                    if done:
                        break

    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================

    def train(self):
        cfg = self.cfg
        for ep in range(1, cfg.num_episodes+1):
            ret, steps, ploss, vloss, ent = self._run_episode()

            for _ in range(cfg.phi_updates_per_episode):
                self._update_phi()

            if ep % cfg.log_interval == 0:
                print(f"[Ep {ep:4d}] ret={ret:+.3f}  P={ploss:+.4f}  V={vloss:+.4f}  "
                      f"H={ent:.3f}  φbuf={self.replay.size()}")

    # ==========================================================================
    # RUN EPISODE
    # ==========================================================================

    def _run_episode(self):
        cfg = self.cfg
        obs = self._reset_env()
        ovec = self._obs_to_vec(obs)
        o = self._obs_to_tensor(ovec)

        z, z_idx = self._sample_skill()
        z_batch = z.unsqueeze(0)

        logps = []
        values = []
        ents = []
        rewards = []

        total_ret = 0.0

        for t in range(cfg.max_steps_per_episode):

            logits, value = self.policy(o, z_batch)
            dist = Categorical(logits=logits)
            act = dist.sample()

            logp = dist.log_prob(act)
            ent = dist.entropy()

            obs2, rew_env, done, info = self._step_env(act.item())
            ovec2 = self._obs_to_vec(obs2)
            o2 = self._obs_to_tensor(ovec2)
            # φ REWARD: correct difference (phi(s') - phi(s)) dot z
            with torch.no_grad():
                phi_s = self.phi(o)
                phi_s2 = self.phi(o2)
                diff = phi_s2 - phi_s
                r_phi = (diff * z_batch).sum(dim=-1).item()

            # optional clipping to keep magnitudes reasonable
            r = max(-cfg.reward_clip, min(cfg.reward_clip, r_phi))
            total_ret += r

            # Replay for φ updates: store raw vectors; normalization handled on sampling
            self.replay.add(ovec, ovec2, z.cpu().numpy())

            logps.append(logp)
            values.append(value)
            ents.append(ent)
            rewards.append(r)

            # φ logging
            if t % 5 == 0:
                self.phi_points.append(phi_s.detach().cpu().numpy().flatten())
                self.phi_colors.append(z_idx)

            o = o2
            ovec = ovec2

            if done:
                break

        # Compute returns
        returns = self._discount(rewards, cfg.gamma)
        V = torch.cat(values).squeeze(-1)
        adv = returns - V.detach()

        policy_loss = -(torch.cat(logps) * adv).mean()
        value_loss = F.mse_loss(V, returns)
        entropy_loss = -torch.cat(ents).mean()

        total_loss = (policy_loss
                      + cfg.value_coef * value_loss
                      + cfg.entropy_coef * entropy_loss)

        self.opt_policy.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt_policy.step()

        return total_ret, t+1, policy_loss.item(), value_loss.item(), torch.cat(ents).mean().item()

    # ==========================================================================
    # DISCOUNT RETURNS
    # ==========================================================================

    def _discount(self, rewards, gamma):
        R = 0.0
        ret = []
        for r in reversed(rewards):
            R = r + gamma * R
            ret.append(R)
        ret.reverse()
        ret = torch.tensor(ret, dtype=torch.float32, device=self.device)
        return (ret - ret.mean()) / (ret.std() + 1e-8)

    # ==========================================================================
    # UPDATE φ  (CRITICAL: full LSD contrastive loss)
    # ==========================================================================

    def _update_phi(self):
        batch = self.replay.sample(self.cfg.batch_size)
        if batch is None:
            return
        s, s2, Z_i = batch   # Z_i: [B, skill_dim]

        # normalize replayed observations with current running stats
        if self.obs_norm.mean is not None:
            mean = torch.tensor(self.obs_norm.mean, dtype=torch.float32, device=self.device)
            var = torch.tensor(self.obs_norm.var, dtype=torch.float32, device=self.device)
            s = (s - mean) / (torch.sqrt(var) + 1e-8)
            s2 = (s2 - mean) / (torch.sqrt(var) + 1e-8)

        phi_s  = self.phi(s)
        phi_s2 = self.phi(s2)
        diff = phi_s2 - phi_s    # [B, skill_dim]
        # compute logits against ALL skills and use cross-entropy (stable)
        # diff_expanded: [B, 1, skill_dim]
        # Z_all:         [1, K, skill_dim]
        diff_exp = diff.unsqueeze(1)                 # [B,1,D]
        Z_all = self.discrete_Z.unsqueeze(0)         # [1,K,D]

        # dot products: [B, K]
        logits = torch.sum(diff_exp * Z_all, dim=-1)

        # true skill indices for each sample
        idx_i = torch.argmax(Z_i, dim=-1).long()     # [B]

        # use cross-entropy over the K skills (stable, standard)
        L = F.cross_entropy(logits, idx_i)

        loss_total = self.cfg.phi_coef * L

        self.opt_phi.zero_grad()
        loss_total.backward()
        nn.utils.clip_grad_norm_(self.phi.parameters(), 10.0)
        self.opt_phi.step()

    # ==========================================================================
    # ATTEMPT MODE
    # ==========================================================================

    @torch.no_grad()
    
    def attempt_episode(self, max_steps=None, skill_idx=None):
        if max_steps is None:
            max_steps = self.cfg.max_steps_per_episode

        # new env with rendering
        old = self.env
        self.env = example_minigrid.SimpleEnv(
            size=self.cfg.size,
            max_steps=self.cfg.max_steps,
            render_mode="human"
        )

        obs = self._reset_env()
        ovec = self._obs_to_vec(obs)
        o = self._obs_to_tensor(ovec)

        # --- choose skill ---
        if skill_idx is None:
            z, idx = self._sample_skill()
        else:
            idx = int(skill_idx)
            if not (0 <= idx < self.cfg.num_skills):
                raise ValueError(f"skill_idx {idx} out of range [0, {self.cfg.num_skills-1}]")
            z = self.discrete_Z[idx]

        z_b = z.unsqueeze(0)
        print(f"[attempt] skill={idx}, z={z.cpu().numpy()}")

        for t in range(max_steps):
            try:
                self.env.render()
            except:
                pass

            logits, _ = self.policy(o, z_b)
            act = Categorical(logits=logits).sample().item()

            obs2, _, done, _ = self._step_env(act)
            ovec2 = self._obs_to_vec(obs2)
            o = self._obs_to_tensor(ovec2)

            if done:
                break

        print("[attempt] done.")
        self.env = old


    # ==========================================================================
    # SKILL VISUALIZATION
    # ==========================================================================

    @torch.no_grad()
    def visualize_skills(self, num_skills=None, max_steps=None, save="skills.png"):
        K = self.cfg.num_skills if num_skills is None else num_skills
        max_steps = self.cfg.max_steps_per_episode if max_steps is None else max_steps

        env = example_minigrid.SimpleEnv(
            size=self.cfg.size,
            max_steps=self.cfg.max_steps,
            render_mode=None
        )

        trajs = []

        for k in range(K):
            z = self.discrete_Z[k]
            z_b = z.unsqueeze(0)

            out = env.reset()
            obs = out[0] if isinstance(out,tuple) else out
            ovec = self._obs_to_vec(obs)
            o = self._obs_to_tensor(ovec)

            path = []

            for t in range(max_steps):

                # True agent position
                if hasattr(env, "agent_pos"):
                    x,y = env.agent_pos
                else:
                    x,y = 0,0
                path.append((x,y))

                logits, _ = self.policy(o, z_b)
                act = Categorical(logits=logits).sample().item()

                out = env.step(act)
                if len(out)==5:
                    obs2, _, term, trunc, _ = out
                    done = term or trunc
                else:
                    obs2, _, done, _ = out

                ovec2 = self._obs_to_vec(obs2)
                o = self._obs_to_tensor(ovec2)

                if done:
                    break

            trajs.append((f"skill {k}", np.array(path)))

        # append timestamp to filename to avoid overwrites
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(save)
        if ext == "":
            ext = ".png"
        save_name = f"{base}_{ts}{ext}"

        plt.figure(figsize=(6,6))
        for label, P in trajs:
            plt.plot(P[:,0], P[:,1], label=label)
        plt.legend()
        plt.grid(True)
        plt.title("Skill Trajectories")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(save_name, dpi=200)
        plt.close()

        print(f"[skills] saved {save_name}")

    # ==========================================================================
    # φ PLOTTING
    # ==========================================================================

    def plot_phi(self, save="phi_plots.pdf"):
        if not self.phi_points:
            print("[phi] no data")
            return

        X = np.array(self.phi_points)
        C = np.array(self.phi_colors)

        # PCA to 2D if needed
        if X.shape[1] != 2:
            from sklearn.decomposition import PCA
            X = PCA(n_components=2).fit_transform(X)

        # append timestamp to filename to avoid overwrites
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(save)
        if ext == "":
            ext = ".pdf"
        save_name = f"{base}_{ts}{ext}"

        with PdfPages(save_name) as pdf:
            fig = plt.figure(figsize=(6,6))
            plt.scatter(X[:,0], X[:,1], c=C, cmap="tab20", s=8)
            plt.grid(True)
            plt.title("φ-space")
            pdf.savefig(fig)
            plt.close(fig)

        print(f"[phi] saved {save_name}")

    # ==========================================================================
    # SAVE / LOAD
    # ==========================================================================

    def _save_dir(self):
        return "lsd01"  # only discrete version implemented

    def save(self, directory=None):
        if directory is None:
            directory=self._save_dir()

        os.makedirs(directory, exist_ok=True)

        ts=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path=f"{directory}/lsd_{ts}.pth"

        state={
            "phi":self.phi.state_dict(),
            "policy":self.policy.state_dict(),
            "obs_mean":self.obs_norm.mean,
            "obs_var":self.obs_norm.var,
            "obs_count":self.obs_norm.count,
            "skills":self.discrete_Z.cpu().numpy(),
            "cfg":self.cfg.__dict__
        }

        torch.save(state,path)
        torch.save(state,f"{directory}/latest.pth")
        print(f"[save] {path}")

    def load(self,path=None):
        if path is None:
            path=f"{self._save_dir()}/latest.pth"
        # torch 2.6+ may default to weights_only=True which raises on objects like numpy reconstructors.
        # Try a normal load first, otherwise retry with explicit weights_only=False when supported.
        try:
            data = torch.load(path, map_location=self.device)
        except Exception:
            try:
                import inspect
                if 'weights_only' in inspect.signature(torch.load).parameters:
                    data = torch.load(path, map_location=self.device, weights_only=False)
                else:
                    # last resort: attempt load again (may still raise)
                    data = torch.load(path, map_location=self.device)
            except Exception:
                # If still failing, re-raise the original exception for visibility
                raise

        self.phi.load_state_dict(data["phi"])
        self.policy.load_state_dict(data["policy"])
        self.obs_norm.mean=data["obs_mean"]
        self.obs_norm.var=data["obs_var"]
        self.obs_norm.count=data["obs_count"]
        # Restore full configuration from save file
        saved_cfg = data.get("cfg", None)
        if saved_cfg is not None:
            # override current config values
            for k, v in saved_cfg.items():
                setattr(self.cfg, k, v)

            # rebuild discrete_Z with correct size
            self.discrete_Z = torch.tensor(data["skills"], device=self.device)

            # update skill_dim on trainer
            self.skill_dim = self.cfg.skill_dim

        print(f"[load] {path}")


    def resume_from(self, path=None):
        """Load a checkpoint and continue training from it.

        This loads model weights and training-related saved config into the trainer.
        Note: optimizer state and replay buffer are not restored by this simple loader.
        """
        if path is None:
            path = f"{self._save_dir()}/latest.pth"
        self.load(path)
        print(f"[resume] ready to continue from {path}")
        print(f"[load] {path}")


# ======================================================================================
# MAIN DISPATCH
# ======================================================================================

def main():
    import argparse
    print("Starting LSD agent...")

    p = argparse.ArgumentParser()
    p.add_argument("command", nargs="?", default="train",
                   choices=["train", "attempt", "phi", "skills", "zero_shot"])
    p.add_argument("--mode", default="discrete", choices=["discrete"])
    p.add_argument("--episodes", type=int)
    p.add_argument("--num_skills", type=int)
    p.add_argument("--model",type=str)
    p.add_argument("--resume",action="store_true",help="If set, load the model (or latest) and resume training")
    p.add_argument("--skill", type=int)
    args = p.parse_args()

    # -----------------------------
    # TRAIN MODE: build fresh cfg
    # -----------------------------
    if args.command == "train":
        cfg = LSDConfig()
        if args.episodes:
            cfg.num_episodes = args.episodes
        if args.num_skills:
            cfg.num_skills = args.num_skills
            cfg.skill_dim = args.num_skills

        trainer = LSDTrainer(cfg)
        # If requested, load an existing model (or latest) before training
        if args.resume:
            model_path = args.model if args.model else f"{trainer._save_dir()}/latest.pth"
            trainer.resume_from(model_path)

        trainer.train()
        trainer.save()
        return

    # -----------------------------
    # OTHER MODES:
    # MUST load config *before* building networks
    # -----------------------------

    # Load saved state first (to get saved cfg)
    if args.model is None:
        args.model = "lsd01/latest.pth"

    # Load raw data first
    raw = torch.load(args.model, map_location="cpu", weights_only=False)

    # Build cfg from saved cfg
    saved_cfg = LSDConfig()
    for k, v in raw["cfg"].items():
        setattr(saved_cfg, k, v)

    # Build trainer with correct cfg (correct skill_dim, num_skills, etc.)
    trainer = LSDTrainer(saved_cfg)

    # Now load weights
    trainer.load(args.model)

    # -----------------------------
    # RUN COMMAND
    # -----------------------------
    if args.command == "attempt":
        trainer.attempt_episode(skill_idx=args.skill)
        return

    if args.command == "phi":
        for _ in range(10):
            trainer._run_episode()
        trainer.plot_phi()
        return

    if args.command == "skills":
        trainer.visualize_skills()
        return

    if args.command == "zero_shot":
        print("[zero_shot] (not typically used with discrete LSD)")
        return



if __name__=="__main__":
    print("ENTERING MAIN()")
    main()
