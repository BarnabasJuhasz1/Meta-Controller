
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch
import numpy as np
from oesd.algorithms.lsd import LSDTrainer, LSDConfig
from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry
from oesd.ours.unified_baseline_utils.adapters.BaseAdapter import BaseAdapter
# ============================================================================
# LSD Adapter
# ============================================================================

class LSDAdapter(BaseAdapter):
    """
    Adapter for your LSD implementation.
    Provides a uniform get_action() interface.
    """

    def __init__(self, algo_name: str, ckpt_path: str, action_dim: int, save_dir: str, skill_registry: SkillRegistry):
        super().__init__(algo_name, ckpt_path, action_dim, save_dir, skill_registry=skill_registry)

        # Load trainer + model weights
        cfg = LSDConfig()
        self.trainer = LSDTrainer(cfg)
        self.trainer.load(ckpt_path)

        self.skill_dim = self.trainer.cfg.skill_dim

        # Always use a default skill (e.g., 0) unless user overrides
        self.current_skill_idx = 0
        self.current_skill_vec = self.trainer.discrete_Z[self.current_skill_idx]


    def set_skill(self, idx: int):
        self.current_skill_idx = int(idx)
        self.current_skill_vec = self.trainer.discrete_Z[idx]

    def get_action(self, obs, deterministic=False, skill_z=None):
        """
        Unified action interface:
        obs -> tensor -> pass through LSD policy -> primitive action
        """

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
