# Interface/policy_base.py

from abc import ABC, abstractmethod
import numpy as np

class SkillPolicy(ABC):
    """
    Unified interface for all baseline skill policies (LSD, DIAYN, RSD, etc.)
    """

    @abstractmethod
    def act(self, obs: np.ndarray, skill: np.ndarray) -> np.ndarray:
        """Return action (numpy) for (obs, skill)."""
        pass

    @abstractmethod
    def skill_dim(self) -> int:
        """Return dimensionality of skill vector."""
        pass

    @abstractmethod
    def skill_type(self) -> str:
        """Return 'discrete' or 'continuous'."""
        pass

    def phi(self, obs: np.ndarray):
        """
        Optional: return φ(s) representation.
        """
        raise NotImplementedError
