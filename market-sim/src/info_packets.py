"""Strict PIT info packets: public_packet / channel_packet per company×date."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# The pattern is Chinese on purpose: source fact_summary strings read "EPS 实际 $X; 预期 $Y"
# (= "EPS actual $X; expected $Y").
_EPS_PAT = re.compile(
    r"EPS\s*实际\s*\$?\s*([-+]?\d*\.?\d+)\s*[;；]\s*预期\s*\$?\s*([-+]?\d*\.?\d+)",
    re.IGNORECASE,
)


def parse_event_signal(fact_summary: object) -> float:
    """Parse fail → 0. Success: (actual-expected)/|expected| clipped to [-1,1]."""
    if fact_summary is None or (isinstance(fact_summary, float) and np.isnan(fact_summary)):
        return 0.0
    s = str(fact_summary)
    m = _EPS_PAT.search(s)
    if not m:
        return 0.0
    try:
        actual = float(m.group(1))
        expected = float(m.group(2))
    except ValueError:
        return 0.0
    if expected == 0.0:
        return 0.0
    surprise = (actual - expected) / abs(expected)
    if not np.isfinite(surprise):
        return 0.0
    return float(np.clip(surprise, -1.0, 1.0))


def build_public_event_rows(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in events.iterrows():
        date = r.get("usable_from_trade_date")
        if pd.isna(date):
            continue
        sig = parse_event_signal(r.get("fact_summary"))
        targets = str(r.get("target_companies") or "")
        etype = r.get("event_type")
        if "ALL_MACRO" in targets.upper() or etype == "macro":
            rows.append(
                {
                    "company_id": "__MACRO__",
                    "usable_from_trade_date": pd.Timestamp(date),
                    "public_signal": 0.0,
                    "event_type": etype,
                    "n_events": 1,
                }
            )
            continue
        for tok in targets.replace(",", ";").split(";"):
            cid = tok.strip()
            if not cid or not cid.startswith("C_"):
                continue
            rows.append(
                {
                    "company_id": cid,
                    "usable_from_trade_date": pd.Timestamp(date),
                    "public_signal": sig,
                    "event_type": etype,
                    "n_events": 1,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "company_id",
                "usable_from_trade_date",
                "public_signal",
                "event_type",
                "n_events",
            ]
        )
    df = pd.DataFrame(rows)
    return (
        df.groupby(["company_id", "usable_from_trade_date"], as_index=False)
        .agg(public_signal=("public_signal", "mean"), n_events=("n_events", "sum"))
    )


class InfoPacketStore:
    """
    Day-wise PIT publisher.
    Events indexed by usable_from_trade_date; only that day's fresh events
    enter public_packet (strict: usable_from <= today, we publish on usable_from day).
    Channel packets assembled each day by walking supply-graph neighbors.
    """

    def __init__(
        self,
        public_events: pd.DataFrame,
        neighbors: dict[str, list[str]],
        panel: pd.DataFrame,
    ) -> None:
        self.neighbors = neighbors
        # date -> company_id -> public_signal
        self._by_date: dict[pd.Timestamp, dict[str, float]] = defaultdict(dict)
        if not public_events.empty:
            for r in public_events.itertuples(index=False):
                if r.company_id == "__MACRO__":
                    continue
                dt = pd.Timestamp(r.usable_from_trade_date)
                self._by_date[dt][r.company_id] = float(r.public_signal)

        # company,date -> ret_lag1 (yesterday's realized return)
        tmp = panel[["company_id", "trade_date", "ret"]].copy()
        tmp = tmp.sort_values(["company_id", "trade_date"])
        tmp["ret_lag1"] = tmp.groupby("company_id")["ret"].shift(1).fillna(0.0)
        self._ret_lag: dict[tuple[str, pd.Timestamp], float] = {
            (r.company_id, pd.Timestamp(r.trade_date)): float(r.ret_lag1)
            for r in tmp.itertuples(index=False)
        }

    def publish_day(self, dt: pd.Timestamp, company_ids: list[str]) -> dict[str, dict]:
        """Assemble visible packets for all companies on this trade date (PIT)."""
        dt = pd.Timestamp(dt)
        day_pub = self._by_date.get(dt, {})
        out: dict[str, dict] = {}
        for cid in company_ids:
            pub = float(day_pub.get(cid, 0.0))
            neigh = self.neighbors.get(cid, [])
            if neigh:
                n_pub = [float(day_pub.get(n, 0.0)) for n in neigh]
                n_ret = [float(self._ret_lag.get((n, dt), 0.0)) for n in neigh]
                # deliberate per-neighbor walk (not a single matrix multiply)
                channel = 0.5 * (sum(n_pub) / len(n_pub)) + 0.5 * (sum(n_ret) / len(n_ret))
            else:
                channel = 0.0
            if abs(pub) > 1e-12:
                msg_sign = float(np.sign(pub))
            elif abs(channel) > 1e-12:
                msg_sign = float(np.sign(channel))
            else:
                msg_sign = 0.0
            out[cid] = {
                "public_signal": pub,
                "channel_signal": float(channel),
                "msg_sign": msg_sign,
                "ret_lag1": float(self._ret_lag.get((cid, dt), 0.0)),
            }
        return out


def export_info_packets_sample(
    store: InfoPacketStore,
    panel: pd.DataFrame,
    data_dir: Path,
    n_days: int = 5,
) -> pd.DataFrame:
    data_dir.mkdir(parents=True, exist_ok=True)
    formal = panel[panel["window"] == "formal_2023_2025"]
    dates = sorted(formal["trade_date"].unique())
    pick = list(dates[:n_days]) + list(dates[len(dates) // 2 : len(dates) // 2 + n_days])
    secs = formal[["security_id", "company_id", "ticker"]].drop_duplicates()
    companies = secs["company_id"].tolist()
    rows = []
    for dt in pick:
        packets = store.publish_day(pd.Timestamp(dt), companies)
        for _, s in secs.iterrows():
            pkt = packets.get(s["company_id"], {})
            rows.append(
                {
                    "security_id": s["security_id"],
                    "company_id": s["company_id"],
                    "ticker": s["ticker"],
                    "trade_date": pd.Timestamp(dt),
                    "public_packet": pkt.get("public_signal", 0.0),
                    "channel_packet": pkt.get("channel_signal", 0.0),
                    "msg_sign": pkt.get("msg_sign", 0.0),
                    "ret_lag1": pkt.get("ret_lag1", 0.0),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(data_dir / "info_packets_sample.csv", index=False)
    (data_dir / "info_packets_README.txt").write_text(
        "info_packets_sample.csv\n"
        "- public_packet: same-day public event surprise if usable_from_trade_date==date "
        "(EPS surprise; parse fail→0). Strict PIT: only usable_from_trade_date ≤ decision date.\n"
        "- channel_packet: equal-weight mix of supply-graph neighbors' public_packet "
        "and neighbors' ret_lag1 (assembled each day by walking edges).\n"
        "- msg_sign: sign(public) else sign(channel).\n"
        "- Macro ALL_MACRO events contribute 0 when unparseable.\n",
        encoding="utf-8",
    )
    return out
