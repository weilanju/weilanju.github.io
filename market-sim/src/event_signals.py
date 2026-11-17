# -*- coding: utf-8 -*-
"""
event_signals.py — extends the signal layer from "EPS surprise only" to a direction mapping
covering "earnings + company-level events + FOMC".

Design principles (honest, reproducible):
  1. Every event comes from real data (events.csv / events_manual.csv); events are never
     edited or invented here.
  2. This module does exactly one thing: assign each real event a sign in {-1, 0, +1}.
     The sign is a MODELLING ASSUMPTION, not data; it must be inspectable, editable, auditable.
  3. Sign +/- only when the direction is clear; otherwise 0 (better missing than fake).
  4. ALFRED macro releases (534 rows) are release-calendar markers only, no direction -> keep 0;
     a real direction would need a separately computed surprise (see TODO at the end).

Magnitude convention: non-earnings events take signal +/-1.0 (pure direction); earnings keep
the EPS-surprise magnitude (already clipped to [-1,1]).
  — Magnitude is a hyperparameter; downstream models rescale it with a learned weight beta,
    so only the sign matters here.

NOTE: the keys of SIGN_OVERRIDES are kept in the original Chinese because they are matched
against the Chinese headlines of the source event table (events.csv). Translating them would
break the matching.
"""

import re
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 1) Company-level events: direction hand-annotated by headline.
#    value = (default_sign, {company_id: sign_override})
#    overrides handle "same event, different direction for different companies"
#    (competitors, acquisition targets, etc.).
# ---------------------------------------------------------------------------
SIGN_OVERRIDES = {
    # ---- capacity ----
    "ASML 首台 High-NA EUV": (+1.0, {}),
    "Meta 计划年底部署": (+1.0, {}),
    "美光宣布 HBM3E 量产": (+1.0, {}),
    "SK 海力士 HBM3E 量产": (+1.0, {"C_MU": -1.0}),  # SK hynix mass production = competitive negative for Micron
    # ---- corporate_action ----
    "NVIDIA 与软银终止": (-1.0, {}),
    "CoreWeave 在 Nasdaq 上市": (+1.0, {}),
    "Pure Storage 更名": (0.0, {}),
    "Supermicro 10 拆 1": (0.0, {}),
    "Broadcom 10 拆 1": (0.0, {}),
    "NVIDIA 10 拆 1": (0.0, {}),
    "GE Vernova 分拆上市": (0.0, {}),
    "Sandisk 自 WDC 分拆上市": (0.0, {}),
    "Everpure 代码由 PSTG": (0.0, {}),
    "Arm 在 Nasdaq 上市": (+1.0, {}),
    # ---- other ----
    "Supermicro 审计机构 EY 辞任": (-1.0, {}),
    # ---- partnership ----
    "NVIDIA 与 OpenAI 10GW": (+1.0, {}),
    "HPE 完成收购 Juniper": (+1.0, {"N_JUNIPER": 0.0}),  # acquired side: no direction
    "CoreWeave 与 Meta 签约": (+1.0, {}),
    "AMD 与 OpenAI 6GW": (+1.0, {}),
    "Broadcom 与 OpenAI 10GW": (+1.0, {}),
    "Oracle 部署 5 万颗 AMD": (+1.0, {}),
    "AMD 完成收购 ZT Systems": (+1.0, {"N_ZT": 0.0}),
    "Anthropic 扩大使用 Google TPU": (+1.0, {}),
    "CoreWeave 与 OpenAI 119": (+1.0, {}),
    "NVIDIA 承诺购买 CoreWeave": (+1.0, {}),
    "Cisco 扩大与 NVIDIA": (+1.0, {}),
    "Stargate 项目宣布": (+1.0, {}),
    "微软、NVIDIA 与 Anthropic": (+1.0, {}),
    "NVIDIA 投资 Synopsys": (+1.0, {}),
    "Marvell 与 AWS 五年": (+1.0, {}),
    "Amazon 追加投资 Anthropic": (+1.0, {}),
    "Vertiv 与 NVIDIA GB200": (+1.0, {}),
    "Amkor 与台积电亚利桑那": (+1.0, {}),
    "微软扩大对 OpenAI": (+1.0, {}),
    "HPE 与 NVIDIA 推出": (+1.0, {}),
    "NVIDIA 投资 Intel 50 亿": (+1.0, {}),
    # ---- product_launch ----
    "NVIDIA 发布 Spectrum-X": (+1.0, {}),
    "NVIDIA GTC 2024 发布 Blackwell": (+1.0, {}),
    "OpenAI 发布 ChatGPT": (+1.0, {}),
    # ---- regulation ----
    "BIS 更新先进计算出口管制": (-1.0, {}),
    "商务部宣布撤销 AI 扩散规则": (+1.0, {}),
    "NVIDIA 披露 H20 对华出口需许可": (-1.0, {}),
    "对等关税暂停 90 天": (+1.0, {}),
    "美国商务部 BIS 先进计算与半导体出口管制": (-1.0, {}),
    "BIS 发布人工智能扩散框架": (-1.0, {}),
    "NVIDIA 称美国将批准恢复 H20": (+1.0, {}),
    "美国宣布“对等关税”": (-1.0, {}),
    # ---- technology ----
    "JEDEC 发布 HBM4 标准": (+1.0, {}),
    "DeepSeek-R1 发布": (-1.0, {}),  # efficiency shock; AI-chain repricing
}

# Fallback direction by event_type (used when the headline hits no hand-annotated key)
TYPE_FALLBACK = {
    "partnership": +1.0,
    "product_launch": +1.0,
    "capacity": +1.0,
    "technology": +1.0,
    "regulation": 0.0,
    "corporate_action": 0.0,
    "other": 0.0,
}

_FOMC_RE = re.compile(r"(\+?\d+)\s*bp", re.IGNORECASE)


def event_sign(headline, event_type, target_cid):
    """Return (sign in {-1,0,+1}, whether the hand-annotated table was hit)."""
    h = str(headline)
    for key, (default_s, overrides) in SIGN_OVERRIDES.items():
        if key in h:
            return overrides.get(target_cid, default_s), True
    return TYPE_FALLBACK.get(event_type, 0.0), False


def fomc_sign(fact_summary):
    """FOMC statement: target-range bp change -> hike(-1) / cut(+1) / unchanged(0)."""
    s = str(fact_summary)
    m = _FOMC_RE.search(s)
    if not m:
        return 0.0
    bp = float(m.group(1))
    if bp > 0:
        return -1.0
    elif bp < 0:
        return +1.0
    return 0.0


# ---------------------------------------------------------------------------
# 2) Sign the whole events.csv table; output per-event detail + counts
# ---------------------------------------------------------------------------
def apply(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    signs, hits = [], []
    for _, r in out.iterrows():
        et = r["event_type"]
        if et == "earnings":
            # earnings keep the EPS-surprise magnitude (merged outside from
            # the original parse_event_signal logic)
            s, hit = 0.0, False
        elif et == "macro":
            if "FOMC" in str(r["headline"]) or "FOMC" in str(r["source_id"]):
                s, hit = fomc_sign(r["fact_summary"]), True
            else:
                s, hit = 0.0, False  # ALFRED release calendar -> no direction
        else:
            s, hit = event_sign(r["headline"], et, "__default__")
        signs.append(s)
        hits.append(hit)
    out["direction_sign"] = signs
    out["hand_annotated"] = hits
    return out


def main():
    BASE = r"C:\Users\lanju\Desktop\Market Wave\ai_infra_v3_extracted\ai_infra_2022_2026_v3"
    ev = pd.read_csv(BASE + r"\tables\events.csv")

    # Original EPS surprise (replicates info_packets.parse_event_signal).
    # The pattern is Chinese on purpose: the source fact_summary strings are Chinese
    # ("EPS 实际 $X; 预期 $Y" = "EPS actual $X; expected $Y").
    pat = re.compile(r"EPS\s*实际\s*\$?\s*([-+]?\d*\.?\d+)\s*[;；]\s*预期\s*\$?\s*([-+]?\d*\.?\d+)", re.I)
    def eps_sig(s):
        if pd.isna(s):
            return 0.0
        m = pat.search(str(s))
        if not m:
            return 0.0
        try:
            a = float(m.group(1)); e = float(m.group(2))
        except ValueError:
            return 0.0
        if e == 0:
            return 0.0
        v = np.clip((a - e) / abs(e), -1.0, 1.0)
        return float(v) if np.isfinite(v) else 0.0

    ev["eps_surprise"] = ev["fact_summary"].apply(eps_sig)
    ev = apply(ev)

    # Merge: earnings use EPS surprise, everything else uses direction_sign
    ev["signal"] = np.where(ev["event_type"] == "earnings", ev["eps_surprise"], ev["direction_sign"])
    nz = ev["signal"].abs() > 1e-12

    print("=== before fill-in ===")
    print("events with a non-zero signal:", int((ev['eps_surprise'].abs() > 1e-12).sum()), "(earnings EPS only)")
    print("=== after fill-in ===")
    print("events with a non-zero signal:", int(nz.sum()), "/", len(ev))
    print("\nnon-zero signals by event_type:")
    print(ev[nz].groupby("event_type").size().to_string())
    print("\ncompany-level (non-earnings) events hitting the hand-annotated table:",
          int(((ev['event_type'] != 'earnings') & ev['hand_annotated']).sum()))

    print("\n=== sample: 12 filled company-level signals ===")
    comp = ev[(ev['event_type'] != 'earnings') & (ev['event_type'] != 'macro')]
    for _, r in comp[comp['direction_sign'] != 0].head(12).iterrows():
        print("  %+d  [%s] %s  (targets=%s)" % (int(r['direction_sign']), r['event_type'], r['headline'], r['target_companies']))

    print("\n=== FOMC direction sample ===")
    fomc = ev[(ev['event_type'] == 'macro') & ev['hand_annotated']]
    for _, r in fomc.head(8).iterrows():
        print("  %+d  %s | %s" % (int(r['direction_sign']), r['headline'], r['fact_summary']))

    # Save the sign-augmented intermediate table for the simulation to consume
    out_path = BASE + r"\tables\events_with_signal.csv"
    ev.to_csv(out_path, index=False)
    print("\nwritten:", out_path)


if __name__ == "__main__":
    main()
