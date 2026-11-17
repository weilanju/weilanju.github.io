"""PublicAgent — only public tape / public events."""
from __future__ import annotations

from typing import Any

from .base import AgentBelief, BaseAgent, clip_nudge


class PublicAgent(BaseAgent):
    name = "public"

    def observe(self, security_id: str, info: dict[str, Any]) -> AgentBelief:
        sig = float(info.get("public_signal", 0.0) or 0.0)
        b = AgentBelief(signal=sig, extras={"public_signal": sig})
        self.beliefs[security_id] = b
        return b

    def quote(self, security_id: str, prev_price: float) -> float:
        if prev_price is None or prev_price <= 0:
            return float("nan")
        b = self.beliefs.get(security_id, AgentBelief())
        nudge = clip_nudge(self.beta * b.signal)
        q = float(prev_price) * (1.0 + nudge)
        self.last_quotes[security_id] = q
        return q
