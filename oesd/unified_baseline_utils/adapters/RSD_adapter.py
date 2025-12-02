from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch
import numpy as np

from oesd.unified_baseline_utils.SingleLoader import BaseAdapter

# ============================================================================
# RSD Adapter
# ============================================================================

class RSDAdapter(BaseAdapter):
    """
    Adapter for your RSD implementation.
    Provides a uniform get_action() interface.
    """
    def __init__(self, algo: str, ckpt_path: str, action_dim: int, save_dir: str):
        super().__init__(algo, ckpt_path, action_dim, save_dir)
        
        # option_policy_path = os.path.join(ckpt_path, f'option_policy{epoch}.pt')
        self.option_policy_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        self.discrete = self.option_policy_ckpt['discrete']
        self.dim_option = self.option_policy_ckpt['dim_option']
        
        skill_list = [self.get_skill_vector(i) for i in range()]

    """
    Helper function to initialize skill vectors
    """
    def get_skill_vector(self, k, unit_length=False):
        if self.discrete:
            return np.eye(self.dim_option)[k]

        if self.dim_option == 2:
            num_skills = 8
            angle = (k % num_skills) * 2 * np.pi / num_skills
            r = 1.0 if unit_length else 1.5
            return np.array([r * np.cos(angle), r * np.sin(angle)])

        # Use basis vectors for diversity in higher dimensions
        if k < self.dim_option:
            v = np.zeros(self.dim_option)
            v[k] = 1.0
        elif k < 2 * self.dim_option:
            v = np.zeros(self.dim_option)
            v[k - self.dim_option] = 1.0
        else:
            np.random.seed(k)
            v = np.random.randn(self.dim_option)

        if unit_length:
            v = v / (np.linalg.norm(v) + 1e-8)
        return v


    def get_action(self, obs, skill_z, deterministic=False):
        """
        Unified action interface:
        """

