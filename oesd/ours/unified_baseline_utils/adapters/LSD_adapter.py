
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch
import numpy as np
from oesd.baselines.lsd.Algos.LSD.lsd import LSDTrainer, LSDConfig
from oesd.ours.unified_baseline_utils.adapters.BaseAdapter import BaseAdapter
from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry

# ============================================================================
# LSD Adapter
# ============================================================================

class LSDAdapter(BaseAdapter):
    """
    Adapter for your LSD implementation.
    Provides a uniform get_action() interface.
    """

    def __init__(self, algo_name: str, ckpt_path: str, action_dim: int, save_dir: str, skill_registry: SkillRegistry):
        super().__init__(algo_name, ckpt_path, action_dim, save_dir, skill_registry)

        # Load trainer + model weights
        cfg = LSDConfig()
        self.trainer = LSDTrainer(cfg)
        self.trainer.load(ckpt_path)

        self.action_dim = action_dim
        self.skill_dim = self.trainer.cfg.skill_dim
        self.save_dir = save_dir

        # Convert discrete skill vectors to numpy for skill registry
        # discrete_Z is shape (num_skills, skill_dim)
        skill_list = [self.trainer.discrete_Z[i].cpu().numpy() if torch.is_tensor(self.trainer.discrete_Z[i]) else self.trainer.discrete_Z[i] 
                      for i in range(len(self.trainer.discrete_Z))]
        
        # Register skills with the skill registry
        self.skill_registry.register_baseline(self.algo_name, skill_list)

        # Always use a default skill (e.g., 0) unless user overrides
        self.current_skill_idx = 0
        self.current_skill_vec = self.trainer.discrete_Z[self.current_skill_idx]

    def set_skill(self, idx: int):
        self.current_skill_idx = int(idx)
        self.current_skill_vec = self.trainer.discrete_Z[idx]

    def sample_skill(self):
        """
        Sample a random skill uniformly from registered skills.
        Same behavior as RSD adapter.
        """
        skill_vec = self.skill_registry.sample(self.algo_name)
        return skill_vec

    def get_action(self, obs, deterministic=False, skill_z=None):
        """
        Unified action interface:
        obs -> tensor -> pass through LSD policy -> primitive action
        
        If skill_z is provided, use it; otherwise use current_skill_vec.
        """
        if skill_z is not None:
            # Convert to tensor if needed
            if isinstance(skill_z, np.ndarray):
                z_tensor = torch.from_numpy(skill_z).float().to(self.trainer.device)
            else:
                z_tensor = skill_z.float().to(self.trainer.device)
        else:
            z_tensor = self.current_skill_vec

        ovec = self.trainer._obs_to_vec(obs)
        o = self.trainer._obs_to_tensor(ovec)

        z = z_tensor.unsqueeze(0) if z_tensor.dim() == 1 else z_tensor
        logits, _ = self.trainer.policy(o, z)

        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            act = torch.argmax(dist.logits, dim=-1)
        else:
            act = dist.sample()

        return int(act.item())
