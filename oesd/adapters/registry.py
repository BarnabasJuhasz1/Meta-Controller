# adapters/registry.py
from __future__ import annotations

from adapters.lsd_adapter import LSDAdapter

ADAPTER_REGISTRY = {
    "lsd": LSDAdapter,
   # include other adapters here as they are implemented
}

def get_adapter_class(algo_name: str):
    if algo_name.lower() not in ADAPTER_REGISTRY:
        raise KeyError(f"Unknown algorithm '{algo_name}'. "
                       f"Available: {list(ADAPTER_REGISTRY.keys())}")
    return ADAPTER_REGISTRY[algo_name.lower()]
