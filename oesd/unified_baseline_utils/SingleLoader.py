# SingleLoader.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch

from adapters.LSD_adapter import LSDAdapter
from adapters.RSD_adapter import RSDAdapter

# ============================================================================
# ModelConfig dataclass
# ============================================================================

@dataclass
class ModelConfig:
    algo_name: str
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
    def __init__(self, algo_name: str, ckpt_path: str, action_dim: int, save_dir: str):
        self.ckpt_path = ckpt_path
        self.action_dim = action_dim
        self.save_dir = save_dir
        print(f"Loading {algo_name} checkpoints from {ckpt_path}. Action dim: {action_dim}")

    def get_action(self, obs, skill_z, deterministic=False):
        raise NotImplementedError

# ============================================================================
# Loader function
# ============================================================================

def load_model_from_config(cfg: ModelConfig) -> BaseAdapter:
    """
    ALWAYS return a SINGLE adapter object.
    NEVER return tuples.
    """

    algo_name = cfg.algo_name.lower()

    if algo_name == "lsd":
        adapter = LSDAdapter(
            algo_name=algo_name,
            ckpt_path=cfg.checkpoint_path,
            action_dim=cfg.action_dim,
            save_dir=cfg.adapter_kwargs.get("save_dir", "./")
        )
        return adapter
    elif algo_name == "RSD":
        adapter = RSDAdapter(
            algo_name=algo_name,
            ckpt_path=cfg.checkpoint_path,
            action_dim=cfg.action_dim,
            save_dir=cfg.adapter_kwargs.get("save_dir", "./")
        )
        return adapter


    # ------------------------------------------------------------
    # ADD FUTURE ALGORITHMS HERE
    # ------------------------------------------------------------

    raise ValueError(f"Unknown algorithm: {cfg.algo}")
