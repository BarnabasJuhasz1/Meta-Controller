# adapters/lsd_adapter.py
from __future__ import annotations
import numpy as np
import torch
from torch.distributions import Categorical

from adapters.base_adapter import BaseAdapter
from algorithms.lsd import LSDTrainer, LSDConfig     # <-- using your existing implementation


class LSDAdapter(BaseAdapter):
    algo_name = "lsd"

    def __init__(self, action_dim: int, skill_dim: int = 8, **kwargs):
        super().__init__(action_dim, skill_dim)
        self.trainer = None     # LSDTrainer instance

    # ------------------------------------------------------------------
    # 1) LOAD MODEL  (calls your LSDTrainer.load() directly)
    # ------------------------------------------------------------------
    def load_model(self, checkpoint_path: str):
        raw = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        cfg = LSDConfig()
        for k, v in raw["cfg"].items():
            setattr(cfg, k, v)

        trainer = LSDTrainer(cfg)
        trainer.load(checkpoint_path)

        self.trainer = trainer
        return trainer

    # ------------------------------------------------------------------
    # 2) PREPROCESS OBS (calls LSDTrainer._obs_to_vec)
    # ------------------------------------------------------------------
    def preprocess_observation(self, raw_obs):
        vec = self.trainer._obs_to_vec(raw_obs)
        vec = (vec - self.trainer.obs_norm.mean) / (np.sqrt(self.trainer.obs_norm.var) + 1e-8)
        
        return vec.astype(np.float32)

    def get_obs_minmax_stats(self, num_samples: int = 500):
        """
        Sample observations from the environment (with steps) and compute min/max statistics.
        Useful for min-max normalization to [0, 1].

        Returns:
            (obs_min, obs_max) — numpy arrays of shape [obs_dim]
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not loaded. Call load_model() first.")

        obs_samples = []
        
        # Collect observations across multiple episodes + steps
        for episode in range(max(1, num_samples // 50)):
            obs = self.trainer._reset_env()
            obs_vec = self.trainer._obs_to_vec(obs)
            obs_std = (obs_vec - self.trainer.obs_norm.mean) / (np.sqrt(self.trainer.obs_norm.var) + 1e-8)
            obs_samples.append(obs_std)
            
            # Take random steps to collect diverse observations
            for _ in range(min(50, num_samples - len(obs_samples))):
                action = self.trainer.env.action_space.sample()
                obs, _, done, _ = self.trainer._step_env(action)
                obs_vec = self.trainer._obs_to_vec(obs)
                obs_std = (obs_vec - self.trainer.obs_norm.mean) / (np.sqrt(self.trainer.obs_norm.var) + 1e-8)
                obs_samples.append(obs_std)
                
                if done or len(obs_samples) >= num_samples:
                    break

        obs_samples = np.array(obs_samples[:num_samples], dtype=np.float32)  # [num_samples, obs_dim]
        obs_min = obs_samples.min(axis=0)
        obs_max = obs_samples.max(axis=0)

        return obs_min, obs_max

    def normalize_obs_minmax(self, obs_vec, obs_min, obs_max, epsilon: float = 1e-8):   
        """
        Apply min-max normalization: (x - min) / (max - min + eps) → [0, 1]

        Args:
            obs_vec: standardized observation vector [obs_dim]
            obs_min: min values [obs_dim]
            obs_max: max values [obs_dim]
            epsilon: small value to avoid division by zero

        Returns:
            normalized observation in [0, 1]
        """
        return ((obs_vec - obs_min) / (obs_max - obs_min + epsilon)).astype(np.float32)

    # ------------------------------------------------------------------
    # 3) SKILL SAMPLING (use trainer.discrete_Z)
    # ------------------------------------------------------------------
    def sample_skill(self, rng):
        K = self.skill_dim
        idx = int(rng.integers(K))  
        z = self.trainer.discrete_Z[idx].cpu().numpy()
        return z.astype(np.float32)

    # ------------------------------------------------------------------
    # 4) ACTION (just call policy forward pass)
    # ------------------------------------------------------------------
    def get_action(self, model, obs_vec, skill_vec, deterministic=False):
        trainer = self.trainer

        obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        z_t   = torch.tensor(skill_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = trainer.policy(obs_t, z_t)
            dist = Categorical(logits=logits)

            if deterministic:
                return int(torch.argmax(logits, dim=-1).item())
            return int(dist.sample().item())
