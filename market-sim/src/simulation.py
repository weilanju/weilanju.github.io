"""Day-loop simulation: info → quote → equal-weight fitted_price → reveal true_price.

fitted_price is a forecast / fitted quote composite. It does NOT replace the
true price surface. true_price is exogenous and revealed after quoting.
Train half: agents may update_after_reveal. Test half: params frozen.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .agents import ChannelAgent, LearningAgent, PublicAgent, StructureAgent
from .agents.base import BaseAgent
from .info_packets import InfoPacketStore


def _finite(x: float) -> bool:
    return x is not None and np.isfinite(x)


class DaySimulator:
    def __init__(
        self,
        panel: pd.DataFrame,
        packet_store: InfoPacketStore,
        value_by_company: dict[str, float],
        train_dates: list[pd.Timestamp],
        test_dates: list[pd.Timestamp],
    ) -> None:
        self.panel = panel
        self.packet_store = packet_store
        self.value_by_company = value_by_company
        self.train_dates = [pd.Timestamp(d) for d in train_dates]
        self.test_dates = [pd.Timestamp(d) for d in test_dates]
        self.train_set = set(self.train_dates)
        self.test_set = set(self.test_dates)

        self.public = PublicAgent()
        self.channel = ChannelAgent()
        self.structure = StructureAgent()
        self.learning = LearningAgent()
        self.agents: list[BaseAgent] = [
            self.public,
            self.channel,
            self.structure,
            self.learning,
        ]

        self._by_date: dict[pd.Timestamp, pd.DataFrame] = {
            pd.Timestamp(dt): g.copy()
            for dt, g in panel.groupby("trade_date", sort=True)
        }

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        reaction_rows: list[dict] = []
        formed_rows: list[dict] = []
        n_days_run = 0

        all_dates = self.train_dates + self.test_dates
        for dt in all_dates:
            day = self._by_date.get(pd.Timestamp(dt))
            if day is None or day.empty:
                continue
            is_train = pd.Timestamp(dt) in self.train_set
            n_days_run += 1

            # 1) publish today's visible info packets (PIT) by walking graph neighbors
            company_ids = day["company_id"].dropna().unique().tolist()
            day_packets = self.packet_store.publish_day(pd.Timestamp(dt), company_ids)

            # 2) each Agent observes + quotes from prev_price
            for row in day.itertuples(index=False):
                sid = row.security_id
                cid = row.company_id
                prev = row.prev_price
                true_px = row.true_price
                if not _finite(prev) or prev <= 0:
                    continue
                if not _finite(true_px):
                    continue

                info = dict(day_packets.get(cid, {}))
                info["z_bottleneck"] = float(self.value_by_company.get(cid, 0.0))
                info.setdefault("public_signal", 0.0)
                info.setdefault("channel_signal", 0.0)
                info.setdefault("msg_sign", 0.0)
                info.setdefault("ret_lag1", 0.0)

                quotes: dict[str, float] = {}
                for agent in self.agents:
                    agent.observe(sid, info)
                    quotes[agent.name] = agent.quote(sid, float(prev))

                # 3) equal-weight → fitted_price (NOT a new trade price)
                qvals = [quotes[a.name] for a in self.agents if _finite(quotes[a.name])]
                if not qvals:
                    continue
                fitted = float(np.mean(qvals))

                # 4) reveal exogenous true_price
                # 5) train half only: update strategies with true price
                if is_train:
                    for agent in self.agents:
                        q = quotes[agent.name]
                        if _finite(q):
                            agent.update_after_reveal(
                                sid, float(prev), float(q), float(true_px)
                            )

                rec = {
                    "security_id": sid,
                    "company_id": cid,
                    "ticker": getattr(row, "ticker", None),
                    "trade_date": pd.Timestamp(dt),
                    "phase": "train" if is_train else "test",
                    "prev_price": float(prev),
                    "q_public": float(quotes["public"]),
                    "q_channel": float(quotes["channel"]),
                    "q_structure": float(quotes["structure"]),
                    "q_learning": float(quotes["learning"]),
                    "fitted_price": fitted,
                    "true_price": float(true_px),
                    "public_signal": info["public_signal"],
                    "channel_signal": info["channel_signal"],
                    "msg_sign": info["msg_sign"],
                    "z_bottleneck": info["z_bottleneck"],
                    "abs_nudge_public": abs(quotes["public"] / prev - 1.0)
                    if _finite(quotes["public"])
                    else np.nan,
                    "abs_nudge_channel": abs(quotes["channel"] / prev - 1.0)
                    if _finite(quotes["channel"])
                    else np.nan,
                    "abs_nudge_structure": abs(quotes["structure"] / prev - 1.0)
                    if _finite(quotes["structure"])
                    else np.nan,
                    "abs_nudge_learning": abs(quotes["learning"] / prev - 1.0)
                    if _finite(quotes["learning"])
                    else np.nan,
                }
                if is_train:
                    reaction_rows.append(rec)
                else:
                    formed_rows.append(rec)

            if is_train and pd.Timestamp(dt) == self.train_dates[-1]:
                for agent in self.agents:
                    agent.freeze()

        reactions = pd.DataFrame(reaction_rows)
        formed = pd.DataFrame(formed_rows)
        meta = {
            "n_days_run": n_days_run,
            "n_train_days": len(self.train_dates),
            "n_test_days": len(self.test_dates),
            "n_reaction_rows": len(reactions),
            "n_formed_rows": len(formed),
            "agent_params_after_train": {
                a.name: a.params_snapshot() for a in self.agents
            },
            "note": (
                "fitted_price = mean(q_public,q_channel,q_structure,q_learning); "
                "true_price exogenous; price_surface never overwritten"
            ),
        }
        return reactions, formed, meta


def export_reaction_sample(
    reactions: pd.DataFrame, out_dir: Path, head_days: int = 3, mid_days: int = 3
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    if reactions.empty:
        sample = reactions
    else:
        dates = sorted(reactions["trade_date"].unique())
        mid = len(dates) // 2
        pick = set(dates[:head_days] + dates[mid : mid + mid_days])
        sample = reactions[reactions["trade_date"].isin(pick)].copy()
    sample.to_csv(out_dir / "daily_reactions_train_sample.csv", index=False)
    return sample


def export_formed_prices(formed: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "security_id",
        "date",
        "q_public",
        "q_channel",
        "q_structure",
        "q_learning",
        "fitted_price",
        "true_price",
    ]
    out = formed.rename(columns={"trade_date": "date"})[cols].copy()
    out.to_csv(out_dir / "agent_formed_prices_test.csv", index=False)
    return out


def train_reaction_summary(reactions: pd.DataFrame) -> pd.DataFrame:
    if reactions.empty:
        return pd.DataFrame()
    rows = []
    for name, col in [
        ("public", "abs_nudge_public"),
        ("channel", "abs_nudge_channel"),
        ("structure", "abs_nudge_structure"),
        ("learning", "abs_nudge_learning"),
    ]:
        s = reactions[col]
        rows.append(
            {
                "agent": name,
                "mean_abs_nudge": float(s.mean()),
                "median_abs_nudge": float(s.median()),
                "p90_abs_nudge": float(s.quantile(0.9)),
                "frac_nonzero_nudge": float((s > 1e-8).mean()),
                "n_quotes": int(s.notna().sum()),
            }
        )
    return pd.DataFrame(rows)
