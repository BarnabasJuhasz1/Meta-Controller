# SingleLoader.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch
import os
import importlib

from oesd.ours.unified_baseline_utils.adapters.diayn_adapter import DIAYNAdapter
from oesd.ours.unified_baseline_utils.adapters.BaseAdapter import BaseAdapter
from oesd.ours.unified_baseline_utils.adapters.LSD_adapter import LSDAdapter
from oesd.ours.unified_baseline_utils.adapters.RSD_adapter import RSDAdapter

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

    elif "diayn" in algo_name:
        adapter = DIAYNAdapter(
            algo_name=algo_name,
            ckpt_path=cfg.checkpoint_path,
            action_dim=cfg.action_dim,
            save_dir=cfg.adapter_kwargs.get("save_dir", "./"),
            skill_registry=skill_registry
        )
        return adapter
    
    elif algo_name == "dads":
        pass
    elif algo_name == "metra":
        pass
    

    # ------------------------------------------------------------
    # ADD FUTURE ALGORITHMS HERE
    # ------------------------------------------------------------

    raise ValueError(f"Unknown algorithm: {cfg.algo_name}")


def load_config(config_path: str):
    module_name = os.path.basename(config_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, config_path)

    # 3. Create the module object
    config_module = importlib.util.module_from_spec(spec)

    # 4. Execute the module to populate it
    spec.loader.exec_module(config_module)

    return config_module
