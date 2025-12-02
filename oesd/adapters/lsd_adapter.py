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
