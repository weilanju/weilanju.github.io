"""StructureAgent — public + position/bottleneck score.
Quote B: prev*(1+clip(α*z(bottleneck)+β*msg_sign)).
α, β learned only in train half.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .. import config as cfg
from .base import AgentBelief, BaseAgent, clip_nudge


class StructureAgent(BaseAgent):
    name = "structure"

    def __init__(self) -> None:
        super().__init__()
        self.alpha: float = 0.01  # weight on z_bottleneck
        self.beta: float = 0.02   # weight on msg_sign
        # accumulators for simple online two-feature update
        self._xx = np.zeros((2, 2), dtype=float)
        self._xy = np.zeros(2, dtype=float)

    def observe(self, security_id: str, info: dict[str, Any]) -> AgentBelief:
        z = float(info.get("z_bottleneck", 0.0) or 0.0)
        msg = float(info.get("msg_sign", 0.0) or 0.0)
        pub = float(info.get("public_signal", 0.0) or 0.0)
        # belief stores features used in quote
        b = AgentBelief(
            signal=msg,  # primary msg for logging
            extras={"z_bottleneck": z, "msg_sign": msg, "public_signal": pub},
        )
        self.beliefs[security_id] = b
        return b

    def quote(self, security_id: str, prev_price: float) -> float:
        if prev_price is None or prev_price <= 0:
            return float("nan")
        b = self.beliefs.get(security_id, AgentBelief())
        z = b.extras.get("z_bottleneck", 0.0)
        msg = b.extras.get("msg_sign", 0.0)
        raw = self.alpha * z + self.beta * msg
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
        if self.frozen:
            return
        b = self.beliefs.get(security_id)
        if b is None or prev_price is None or prev_price <= 0:
            return
        if true_price is None or not np.isfinite(true_price):
            return
        z = b.extras.get("z_bottleneck", 0.0)
        msg = b.extras.get("msg_sign", 0.0)
        x = np.array([z, msg], dtype=float)
        if not np.isfinite(x).all():
            return
        y = true_price / prev_price - 1.0
        if not np.isfinite(y):
            return
        # recursive least-squares style accumulation + soft solve
        self._xx += np.outer(x, x)
        self._xy += x * y
        self.n_updates += 1
        if self.n_updates < 30:
            # early: gentle SGD
            pred = self.alpha * z + self.beta * msg
            err = y - pred
            self.alpha += cfg.LEARN_LR * err * z
            self.beta += cfg.LEARN_LR * err * msg
            self.alpha = float(np.clip(self.alpha, -1.0, 1.0))
            self.beta = float(np.clip(self.beta, -1.0, 1.0))
            return
        try:
            coef = np.linalg.solve(self._xx + 1e-6 * np.eye(2), self._xy)
            # blend to avoid jumps
            self.alpha = 0.7 * self.alpha + 0.3 * float(coef[0])
            self.beta = 0.7 * self.beta + 0.3 * float(coef[1])
            self.alpha = float(np.clip(self.alpha, -1.0, 1.0))
            self.beta = float(np.clip(self.beta, -1.0, 1.0))
        except np.linalg.LinAlgError:
            pass

    def params_snapshot(self) -> dict[str, float]:
        return {
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "n_updates": float(self.n_updates),
            "frozen": float(self.frozen),
        }
