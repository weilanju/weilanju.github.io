"""LearningAgent — multi-feature; train-half sees true price to tune; test-half frozen."""
from __future__ import annotations

from typing import Any

import numpy as np

from .. import config as cfg
from .base import AgentBelief, BaseAgent, clip_nudge

FEATURE_KEYS = (
    "public_signal",
    "channel_signal",
    "z_bottleneck",
    "msg_sign",
    "ret_lag1",
)


class LearningAgent(BaseAgent):
    name = "learning"

    def __init__(self) -> None:
        super().__init__()
        self.weights: dict[str, float] = {k: 0.01 for k in FEATURE_KEYS}
        self._xx = np.zeros((len(FEATURE_KEYS), len(FEATURE_KEYS)), dtype=float)
        self._xy = np.zeros(len(FEATURE_KEYS), dtype=float)

    def observe(self, security_id: str, info: dict[str, Any]) -> AgentBelief:
        extras = {k: float(info.get(k, 0.0) or 0.0) for k in FEATURE_KEYS}
        # composite for logging
        sig = sum(self.weights[k] * extras[k] for k in FEATURE_KEYS)
        b = AgentBelief(signal=float(sig), extras=extras)
        self.beliefs[security_id] = b
        return b

    def quote(self, security_id: str, prev_price: float) -> float:
        if prev_price is None or prev_price <= 0:
            return float("nan")
        b = self.beliefs.get(security_id, AgentBelief())
        raw = sum(self.weights[k] * b.extras.get(k, 0.0) for k in FEATURE_KEYS)
        nudge = clip_nudge(raw)
        q = float(prev_price) * (1.0 + nudge)
        self.last_quotes[security_id] = q
        return q

    def update_after_reveal(
        self,
        security_id: str,
        prev_price: float,
        quoted: float,
        true_price: float,
    ) -> None:
        """Train half only: open true price to retune weights; never called in test."""
        if self.frozen:
            return
        b = self.beliefs.get(security_id)
        if b is None or prev_price is None or prev_price <= 0:
            return
        if true_price is None or not np.isfinite(true_price):
            return
        x = np.array([b.extras.get(k, 0.0) for k in FEATURE_KEYS], dtype=float)
        if not np.isfinite(x).all():
            return
        y = true_price / prev_price - 1.0
        if not np.isfinite(y):
            return
        self._xx += np.outer(x, x)
        self._xy += x * y
        self.n_updates += 1
        # SGD always
        pred = float(x @ np.array([self.weights[k] for k in FEATURE_KEYS]))
        err = y - pred
        for i, k in enumerate(FEATURE_KEYS):
            self.weights[k] = float(
                np.clip(self.weights[k] + cfg.LEARN_LR * err * x[i], -1.0, 1.0)
            )
        if self.n_updates >= 50 and self.n_updates % 25 == 0:
            try:
                coef = np.linalg.solve(
                    self._xx + 1e-5 * np.eye(len(FEATURE_KEYS)), self._xy
                )
                for i, k in enumerate(FEATURE_KEYS):
                    self.weights[k] = float(
                        np.clip(0.7 * self.weights[k] + 0.3 * coef[i], -1.0, 1.0)
                    )
            except np.linalg.LinAlgError:
                pass

    def params_snapshot(self) -> dict[str, float]:
        out = {f"w_{k}": float(v) for k, v in self.weights.items()}
        out["n_updates"] = float(self.n_updates)
        out["frozen"] = float(self.frozen)
        return out
