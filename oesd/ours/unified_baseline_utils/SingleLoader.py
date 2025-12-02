# SingleLoader.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch

from adapters.BaseAdapter import BaseAdapter
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
# Loader function
# ============================================================================

def load_model_from_config(cfg: ModelConfig, skill_registry: SkillRegistry = None) -> BaseAdapter:
    """
    ALWAYS return a SINGLE adapter object.
    NEVER return tuples.
    """

    algo_name = cfg.algo_name.lower()

    if "lsd" in algo_name:
        adapter = LSDAdapter(
            algo_name=algo_name,
            ckpt_path=cfg.checkpoint_path,
            action_dim=cfg.action_dim,
            save_dir=cfg.adapter_kwargs.get("save_dir", "./"),
            skill_registry=skill_registry
        )
        return adapter
    elif "rsd" in algo_name:
        adapter = RSDAdapter(
            algo_name=algo_name,
            ckpt_path=cfg.checkpoint_path,
            action_dim=cfg.action_dim,
            save_dir=cfg.adapter_kwargs.get("save_dir", "./"),
            skill_registry=skill_registry
        )
        return adapter


    # ------------------------------------------------------------
    # ADD FUTURE ALGORITHMS HERE
    # ------------------------------------------------------------

    raise ValueError(f"Unknown algorithm: {cfg.algo_name}")
