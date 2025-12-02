# SingleLoader.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple

from adapters.registry import get_adapter_class


@dataclass
class ModelConfig:
    algo: str
    checkpoint_path: str
    action_dim: int
    skill_dim: int = 8
    adapter_kwargs: Dict[str, Any] = field(default_factory=dict)


def load_model_from_config(cfg: ModelConfig):
    AdapterCls = get_adapter_class(cfg.algo)
    adapter = AdapterCls(cfg.action_dim, cfg.skill_dim, **cfg.adapter_kwargs)
    model = adapter.load_model(cfg.checkpoint_path)
    return adapter, model
