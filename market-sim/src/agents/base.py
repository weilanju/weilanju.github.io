"""Base agent: observe → belief; quote(prev_price); optional update_after_reveal (train only)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import config as cfg


def clip_nudge(x: float, clip: float = cfg.NUDGE_CLIP) -> float:
    return float(np.clip(x, -clip, clip))


@dataclass
class AgentBelief:
    """Per-security belief / state after observe()."""
    signal: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)


class BaseAgent(ABC):
    name: str = "base"

    def __init__(self) -> None:
        self.frozen: bool = False
        self.beliefs: dict[str, AgentBelief] = {}
        # nudge magnitude params learned in train half
        self.beta: float = 0.02  # mild prior so early quotes move a bit
        self.n_updates: int = 0
        # reaction log helpers
        self.last_quotes: dict[str, float] = {}

    def freeze(self) -> None:
        self.frozen = True

    @abstractmethod
    def observe(self, security_id: str, info: dict[str, Any]) -> AgentBelief:
        ...

    @abstractmethod
    def quote(self, security_id: str, prev_price: float) -> float:
        ...

    def update_after_reveal(
        self,
        security_id: str,
        prev_price: float,
        quoted: float,
        true_price: float,
    ) -> None:
        """Default: single-feature online update of beta from realized return.
        Only called in training half by the simulator.
        """
        if self.frozen:
            return
        belief = self.beliefs.get(security_id)
        if belief is None or prev_price is None or not np.isfinite(prev_price) or prev_price <= 0:
            return
        if true_price is None or not np.isfinite(true_price):
            return
        sig = belief.signal
        if abs(sig) < 1e-12:
            return
        realized = true_price / prev_price - 1.0
        # target beta ≈ realized / sig ; SGD toward that
        target = realized / sig
        if not np.isfinite(target):
            return
        target = float(np.clip(target, -1.0, 1.0))
        self.beta = (1.0 - cfg.LEARN_LR) * self.beta + cfg.LEARN_LR * target
        self.n_updates += 1

    def params_snapshot(self) -> dict[str, float]:
        return {"beta": float(self.beta), "n_updates": float(self.n_updates), "frozen": float(self.frozen)}
