# SingleLoader.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch


# ============================================================================
# ModelConfig dataclass
# ============================================================================

@dataclass
class ModelConfig:
    algo: str
    checkpoint_path: str
    action_dim: int
    skill_dim: int
    adapter_kwargs: dict


# ============================================================================
# Base class for all adapters
# ============================================================================

class BaseAdapter:
    """
    Every adapter must implement:
        - get_action(obs, deterministic)
    """

    def get_action(self, obs, deterministic=False):
        raise NotImplementedError


# ============================================================================
# LSD Adapter
# ============================================================================

class LSDAdapter(BaseAdapter):
    """
    Adapter for your LSD implementation.
    Provides a uniform get_action() interface.
    """

    def __init__(self, ckpt_path: str, action_dim: int, save_dir: str):
        # Import here to avoid circular deps
        from algorithms.lsd import LSDTrainer, LSDConfig

        # Load trainer + model weights
        cfg = LSDConfig()
        self.trainer = LSDTrainer(cfg)
        self.trainer.load(ckpt_path)

        self.action_dim = action_dim
        self.skill_dim = self.trainer.cfg.skill_dim
        self.save_dir = save_dir

        # Always use a default skill (e.g., 0) unless user overrides
        self.current_skill_idx = 0
        self.current_skill_vec = self.trainer.discrete_Z[self.current_skill_idx]

    def set_skill(self, idx: int):
        self.current_skill_idx = int(idx)
        self.current_skill_vec = self.trainer.discrete_Z[idx]

    def get_action(self, obs, deterministic=False):
        """
        Unified action interface:
        obs -> tensor -> pass through LSD policy -> primitive action
        """
        import torch
        import numpy as np

        ovec = self.trainer._obs_to_vec(obs)
        o = self.trainer._obs_to_tensor(ovec)

        z = self.current_skill_vec.unsqueeze(0)
        logits, _ = self.trainer.policy(o, z)

        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            act = torch.argmax(dist.logits, dim=-1)
        else:
            act = dist.sample()

        return int(act.item())


# ============================================================================
# Loader function
# ============================================================================

def load_model_from_config(cfg: ModelConfig) -> BaseAdapter:
    """
    ALWAYS return a SINGLE adapter object.
    NEVER return tuples.
    """

    algo = cfg.algo.lower()

    if algo == "lsd":
        adapter = LSDAdapter(
            ckpt_path=cfg.checkpoint_path,
            action_dim=cfg.action_dim,
            save_dir=cfg.adapter_kwargs.get("save_dir", "./")
        )
        return adapter

    # ------------------------------------------------------------
    # ADD FUTURE ALGORITHMS HERE (DIAYN, DADS, CIC, etc.)
    # ------------------------------------------------------------

    raise ValueError(f"Unknown algorithm: {cfg.algo}")
