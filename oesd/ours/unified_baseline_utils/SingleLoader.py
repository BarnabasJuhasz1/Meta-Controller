# SingleLoader.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch
import os
import importlib

from oesd.ours.unified_baseline_utils.adapters.BaseAdapter import BaseAdapter
from oesd.ours.unified_baseline_utils.adapters.LSD_adapter import LSDAdapter
from oesd.ours.unified_baseline_utils.adapters.RSD_adapter import RSDAdapter
from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry

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
    
    elif algo_name == "dads":
        pass
    elif algo_name == "metra":
        pass
    elif algo_name == "dyan":
        pass



    # ------------------------------------------------------------
    # ADD FUTURE ALGORITHMS HERE
    # ------------------------------------------------------------

    raise ValueError(f"Unknown algorithm: {cfg.algo_name}")


def load_config(config_path: str):
    # Normalize separators
    config_path = os.path.normpath(config_path)

    # If a relative `configs/...` path is passed, prefer `ours/configs/...`
    candidates = []

    if os.path.isabs(config_path):
        candidates.append(config_path)
    else:
        # as-provided, relative to cwd
        candidates.append(os.path.join(os.getcwd(), config_path))
        # try with `ours/` prefix (user expects this)
        candidates.append(os.path.join(os.getcwd(), "ours", config_path))
        # try relative to this module's directory
        module_dir = os.path.dirname(__file__)
        candidates.append(os.path.join(module_dir, config_path))
        # try a configs/ next to this module
        basename = os.path.basename(config_path)
        candidates.append(os.path.join(module_dir, "configs", basename))

    # pick the first that exists
    found = None
    for p in candidates:
        p_norm = os.path.normpath(p)
        if os.path.exists(p_norm):
            found = p_norm
            break

    if found is None:
        raise FileNotFoundError(f"Config file not found. Tried: {candidates}")

    module_name = os.path.basename(found).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, found)

    # Create the module object and execute it
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module