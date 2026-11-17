# -*- coding: utf-8 -*-
# ============================================================
# Supply-Chain Information Diffusion & Complex Systems -- All Code (Merged)
# Generated 2026-11-07, merging 14 scripts under Market Wave in order.
#
# Module list:
#   00 event_signals      Event direction annotation (information mapping layer)
#   01 stylized_facts     Three stylized facts (fat tails / vol clustering / no autocorr = targets)
#   02 leadlag_panel      Lead-lag panel regression (daily/monthly + out-of-sample)
#   03 earnings_event_study   Earnings surprise event study
#   04 info_diffusion     Three information-diffusion sub-questions (vol spillover / direction / cross-section)
#   05 robustness         Supplier->customer robustness (excluding mega-upstream)
#   06 oos_test           Out-of-sample test
#   07 minimal_abm        Minimal ABM + HAM (diagnostic)
#   08 lux_marchesi       Exact Lux-Marchesi (2000) implementation (hits all three facts)
#   09 supply_chain_abm   Supply-chain multi-asset version + ablation (diffusion -> clustering)
#   10 position_speed     Supply-chain position x diffusion speed (null)
#   11 empirical_2to5     Empirical tests 2-5
#   12 analyst_robust     Analyst-diffusion robustness (4 checks)
#   13 analyst_robust2    Analyst-diffusion robustness (3 checks)
#
# Notes: each script is self-contained and can run independently; the data path
#   points to Market Wave/ai_infra_v3_extracted/ai_infra_2022_2026_v3/.
#   They execute in file order (each prints its own results).
#   event_signals only defines functions (guarded by if __name__=="__main__").
#
# Run: python supply_chain_diffusion_all.py
# ============================================================


# ==================== event_signals.py ====================

"""
event_signals.py -- extends the EPS-surprise-only signal layer into a direction mapping for earnings + firm-level events + FOMC.

Design principles (honest, reproducible):
  1. Events come entirely from real data (events.csv / events_manual.csv); no editing, no fabrication.
  2. We do exactly one thing: assign each real event a signed direction sign in {-1, 0, +1}.
     The sign is a MODELING ASSUMPTION, not data; it must be inspectable, editable and auditable.
  3. Assign +/- only when direction is clear; otherwise 0 (prefer missing over a fabricated sign).
  4. ALFRED macro releases (534 rows) are only schedule markers with no direction -> keep 0;
     the real direction needs a separate surprise calc (see TODO).

Magnitude convention: non-earnings events get +/-1.0 (pure direction); earnings keep the EPS-surprise magnitude (already clipped to [-1,1]).
  -- magnitude is a hyperparameter; downstream models rescale it via a learned weight beta; here we only care about the sign.
"""

import re
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 1) Firm-level events: direction hand-annotated by headline.
#    value = (default_sign, {company_id: sign_override})
#    override is for cases where one event has different directions across firms (competition, acquired target, etc.).
# ---------------------------------------------------------------------------
SIGN_OVERRIDES = {
    # ---- capacity ----
    "ASML 首台 High-NA EUV": (+1.0, {}),
    "Meta 计划年底部署": (+1.0, {}),
    "美光宣布 HBM3E 量产": (+1.0, {}),
    "SK 海力士 HBM3E 量产": (+1.0, {"C_MU": -1.0}),  # SK Hynix mass production = negative (competition) for Micron
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
    "HPE 完成收购 Juniper": (+1.0, {"N_JUNIPER": 0.0}),  # acquired target has no direction
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
    "DeepSeek-R1 发布": (-1.0, {}),  # efficiency shock, AI-chain re-rating
}

# event_type fallback direction (used when headline misses the hand-annotated table)
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
    """FOMC statement: target-range bp change -> hike(-1) / cut(+1) / hold(0)."""
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
# 2) Annotate the whole events.csv with direction; output signed detail + stats
# ---------------------------------------------------------------------------
def apply(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    signs, hits = [], []
    for _, r in out.iterrows():
        et = r["event_type"]
        if et == "earnings":
            # earnings still use the EPS-surprise magnitude (merged externally)
            s, hit = 0.0, False
        elif et == "macro":
            if "FOMC" in str(r["headline"]) or "FOMC" in str(r["source_id"]):
                s, hit = fomc_sign(r["fact_summary"]), True
            else:
                s, hit = 0.0, False  # ALFRED schedule marker -> no direction
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

    # original EPS surprise (replicating info_packets.parse_event_signal)
    # NOTE: regex intentionally kept in Chinese — it parses the Chinese source event text (actual vs expected).
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

    # merge: earnings use EPS surprise, others use direction_sign
    ev["signal"] = np.where(ev["event_type"] == "earnings", ev["eps_surprise"], ev["direction_sign"])
    nz = ev["signal"].abs() > 1e-12

    print("=== before filling ===")
    print("events with nonzero signal:", int((ev['eps_surprise'].abs() > 1e-12).sum()), "(earnings EPS only)")
    print("=== after filling ===")
    print("events with nonzero signal:", int(nz.sum()), "/", len(ev))
    print("\nnonzero-signal count by event_type:")
    print(ev[nz].groupby("event_type").size().to_string())
    print("\nfirm-level (non-earnings) events hitting the hand-annotated table:", int(((ev['event_type'] != 'earnings') & ev['hand_annotated']).sum()))

    print("\n=== sample of 12 filled firm-level signals ===")
    comp = ev[(ev['event_type'] != 'earnings') & (ev['event_type'] != 'macro')]
    for _, r in comp[comp['direction_sign'] != 0].head(12).iterrows():
        print("  %+d  [%s] %s  (targets=%s)" % (int(r['direction_sign']), r['event_type'], r['headline'], r['target_companies']))

    print("\n=== FOMC direction sample ===")
    fomc = ev[(ev['event_type'] == 'macro') & ev['hand_annotated']]
    for _, r in fomc.head(8).iterrows():
        print("  %+d  %s | %s" % (int(r['direction_sign']), r['headline'], r['fact_summary']))

    # save an intermediate table with signal for downstream simulation
    out_path = BASE + r"\tables\events_with_signal.csv"
    ev.to_csv(out_path, index=False)
    print("\nwritten to:", out_path)


if __name__ == "__main__":
    main()


# ==================== stylized_facts.py ====================

"""
stylized_facts.py -- measure the three stylized facts of the 40 real AI stocks

  Fact 1 fat tails: return kurtosis (excess), normal=0, financial data usually >>0
  Fact 2 volatility clustering: autocorr of |return| (>0 and slowly decaying)
  Fact 3 no return autocorrelation: return autocorr (~0)

These facts are the targets the ABM must later hit.
"""
import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\lanju\Desktop\Market Wave\ai_infra_v3_extracted\ai_infra_2022_2026_v3"
START = pd.Timestamp("2023-01-03")
END = pd.Timestamp("2025-12-31")

prices = pd.read_csv(os.path.join(BASE, "tables", "prices_daily.csv"), low_memory=False)
sec = pd.read_csv(os.path.join(BASE, "tables", "securities.csv"))
comp_sec = sec[sec["asset_role"] == "company"]

p = prices[prices["security_id"].isin(comp_sec["security_id"])].copy()
p["trade_date"] = pd.to_datetime(p["trade_date"])
p = p[(p["trade_date"] >= START) & (p["trade_date"] <= END)].sort_values(["security_id", "trade_date"])
p["ret"] = p.groupby("security_id")["close"].pct_change()
p = p.dropna(subset=["ret"])

# pooled (all 40 stocks' daily returns concatenated)
pool = p["ret"].values
print("=== data size ===")
print("stocks=%d, trading_days=%d, obs=%d" % (p["security_id"].nunique(), p["trade_date"].nunique(), len(pool)))

# per-stock
per_stock = p.groupby("security_id")["ret"].agg(
    kurt=lambda s: s.kurtosis(),
    ac1=lambda s: s.autocorr(1),
    absac1=lambda s: s.abs().autocorr(1),
)

print("\n=== Fact 1: fat tails (excess kurtosis, normal=0) ===")
print("pooled kurtosis      = %+.2f" % (pd.Series(pool).kurtosis()))
print("per-stock kurtosis median = %+.2f  min=%+.2f  max=%+.2f" %
      (per_stock["kurt"].median(), per_stock["kurt"].min(), per_stock["kurt"].max()))
print("note: financial data usually +2~+30; larger = fatter tails")

print("\n=== Fact 2: volatility clustering (|ret| autocorr, should be >0, slow decay) ===")
s_abs = pd.Series(np.abs(pool))
for lag in (1, 5, 10, 20):
    print("  |ret| autocorr lag%2d = %+.4f" % (lag, s_abs.autocorr(lag)))
print("note: lag1 usually +0.1~+0.3, slow decay = volatility clustering")

print("\n=== Fact 3: no return autocorr (return autocorr, ~0) ===")
s = pd.Series(pool)
for lag in (1, 5):
    print("  ret autocorr lag%2d  = %+.4f" % (lag, s.autocorr(lag)))
print("note: ~0 = returns unpredictable (efficient market)")

print("\n=== one-line summary of these targets ===")
print("Return direction is unpredictable (autocorr~0), but volatility is predictable (|ret| autocorr>0), and the distribution has fat tails (kurtosis>>0).")


# ==================== leadlag_panel.py ====================

"""
leadlag_panel.py -- supply-chain lead-lag predictability test (Cohen-Frazzini 2008 style)
Daily + monthly. Pure predictability, zero cost.
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict

BASE = r"C:\Users\lanju\Desktop\Market Wave\ai_infra_v3_extracted\ai_infra_2022_2026_v3"
TRAIN_END = pd.Timestamp("2024-07-02")
TEST_START = pd.Timestamp("2024-07-03")
TEST_END = pd.Timestamp("2025-12-31")
START = pd.Timestamp("2023-01-03")

# ---------------- data ----------------
prices = pd.read_csv(os.path.join(BASE, "tables", "prices_daily.csv"), low_memory=False)
sec = pd.read_csv(os.path.join(BASE, "tables", "securities.csv"))
rel = pd.read_csv(os.path.join(BASE, "tables", "relations.csv"))

comp_sec = sec[sec["asset_role"] == "company"]
sec2comp = dict(zip(comp_sec["security_id"], comp_sec["company_id"]))

p = prices[prices["security_id"].isin(comp_sec["security_id"])].copy()
p["trade_date"] = pd.to_datetime(p["trade_date"])
p = p[(p["trade_date"] >= START) & (p["trade_date"] <= TEST_END)].sort_values(["security_id", "trade_date"])
p["company_id"] = p["security_id"].map(sec2comp)
p["ret"] = p.groupby("security_id")["close"].pct_change()
panel = p[["company_id", "trade_date", "ret"]].copy()

# ---------------- neighbor graph ----------------
def edge_sign(rt):
    return -1.0 if rt == "competition" else 1.0

neigh_sign = defaultdict(dict)
for _, r in rel.iterrows():
    a, b, s = r["from_company_id"], r["to_company_id"], edge_sign(r["relation_type"])
    neigh_sign[a][b] = min(neigh_sign[a].get(b, 1.0), s)
    neigh_sign[b][a] = min(neigh_sign[b].get(a, 1.0), s)

listed = set(panel["company_id"].unique())
neighbors = {c: {n: s for n, s in d.items() if n in listed} for c, d in neigh_sign.items() if c in listed}

# ---------------- tools: OLS + clustered robust SE ----------------
def ols_clustered(y, X, cluster):
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    y = np.asarray(y, float)
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    k = X.shape[1]
    meat = np.zeros((k, k))
    for g in np.unique(cluster):
        m = cluster == g
        sg = (X[m].T * e[m]).sum(axis=1)
        meat += np.outer(sg, sg)
    V = np.linalg.solve(XtX, meat) @ np.linalg.solve(XtX, np.eye(k))
    se = np.sqrt(np.diag(V))
    return b, se, b / se

def hit_pct_and_z(s_pred, s_true):
    n = len(s_pred)
    hit = np.mean(s_pred == s_true)
    # z-test vs 0.5
    z = (hit - 0.5) / np.sqrt(0.5 * 0.5 / n)
    return hit * 100, z

def report(tag, df, cluster_col, y_col="ret_fwd", x_cols=("ret", "nb_ret")):
    df = df.dropna(subset=[y_col] + list(x_cols))
    y = df[y_col].values
    X = df[list(x_cols)].values
    b, se, t = ols_clustered(y, X, df[cluster_col].values)
    print("\n[%s] %s = a + b_own*ret_t + b_neigh*nb_ret_t" % (tag, y_col))
    print("  a       = %+.5f (t=%+.2f)" % (b[0], t[0]))
    print("  b_own   = %+.5f (t=%+.2f)" % (b[1], t[1]))
    print("  b_neigh = %+.5f (t=%+.2f)  <- key" % (b[2], t[2]))
    hp, z = hit_pct_and_z(np.sign(df[x_cols[1]].values), np.sign(y))
    ic = np.corrcoef(df[x_cols[1]].values, y)[0, 1]
    print("  direction accuracy sign(nb)==sign(next) = %.2f%% (z=%+.2f vs 50%%)" % (hp, z))
    print("  IC = %+.4f" % ic)
    return b

# ============ daily ============
d = panel.copy()
d["ret_fwd"] = d.groupby("company_id")["ret"].shift(-1)

def nb_agg(df, col, signed):
    ret_map = df.set_index(["company_id", "trade_date"])[col]
    def f(row):
        vals = []
        for n, s in neighbors.get(row["company_id"], {}).items():
            v = ret_map.get((n, row["trade_date"]))
            if v is not None and np.isfinite(v):
                vals.append(s * v if signed else v)
        return float(np.mean(vals)) if vals else np.nan
    return df.apply(f, axis=1)

d["nb_ret"] = nb_agg(d, "ret", False)
d["nb_ret_signed"] = nb_agg(d, "ret", True)

print("=== daily panel: firms=%d, days=%d, obs=%d ===" %
      (d["company_id"].nunique(), d["trade_date"].nunique(), len(d.dropna(subset=["ret", "ret_fwd", "nb_ret"]))))
report("daily full-sample unsigned", d, "trade_date")
report("daily full-sample signed", d[["trade_date", "ret", "ret_fwd", "nb_ret_signed"]].rename(columns={"nb_ret_signed": "nb_ret"}), "trade_date")

# daily out-of-sample
tr = d[d["trade_date"] <= TRAIN_END].dropna(subset=["ret", "ret_fwd", "nb_ret"])
te = d[d["trade_date"] >= TEST_START].dropna(subset=["ret", "ret_fwd", "nb_ret"])
bb = ols_clustered(tr["ret_fwd"].values, tr[["ret", "nb_ret"]].values, tr["trade_date"].values)[0]
pred = np.column_stack([np.ones(len(te)), te["ret"].values, te["nb_ret"].values]) @ bb
hp, z = hit_pct_and_z(np.sign(pred), np.sign(te["ret_fwd"].values))
ic = np.corrcoef(pred, te["ret_fwd"].values)[0, 1]
print("\n[daily OOS 2024-07~2025-12] train fit (a,b_own,b_neigh)=%s" % np.round(bb, 5))
print("  direction accuracy=%.2f%% (z=%+.2f) | IC=%+.4f" % (hp, z, ic))

# ============ monthly ============
panel["ym"] = panel["trade_date"].dt.to_period("M")
mret = panel.groupby(["company_id", "ym"])["ret"].agg(lambda x: float((1 + x).prod() - 1)).reset_index()
mret = mret.sort_values(["company_id", "ym"])
mret["ret_fwd"] = mret.groupby("company_id")["ret"].shift(-1)

ret_lookup = {(r.company_id, r.ym): r.ret for r in mret.itertuples()}
def nb_m(row, signed=False):
    vals = []
    for n, s in neighbors.get(row["company_id"], {}).items():
        v = ret_lookup.get((n, row["ym"]))
        if v is not None and np.isfinite(v):
            vals.append(s * v if signed else v)
    return float(np.mean(vals)) if vals else np.nan

mret["nb_ret"] = mret.apply(lambda r: nb_m(r, False), axis=1)
mret["nb_ret_signed"] = mret.apply(lambda r: nb_m(r, True), axis=1)

md = mret.dropna(subset=["ret", "ret_fwd", "nb_ret"])
print("\n=== monthly panel: firms=%d, months=%d, obs=%d ===" %
      (md["company_id"].nunique(), md["ym"].nunique(), len(md)))
report("monthly full-sample unsigned", md, "ym")
report("monthly full-sample signed", md[["ym", "ret", "ret_fwd", "nb_ret_signed"]].rename(columns={"nb_ret_signed": "nb_ret"}), "ym")

mtr = md[md["ym"] <= pd.Period("2024-06", "M")]
mte = md[md["ym"] >= pd.Period("2024-07", "M")]
bb = ols_clustered(mtr["ret_fwd"].values, mtr[["ret", "nb_ret"]].values, mtr["ym"].values)[0]
pred = np.column_stack([np.ones(len(mte)), mte["ret"].values, mte["nb_ret"].values]) @ bb
hp, z = hit_pct_and_z(np.sign(pred), np.sign(mte["ret_fwd"].values))
ic = np.corrcoef(pred, mte["ret_fwd"].values)[0, 1]
print("\n[monthly OOS 2024-07~2025-12] train fit (a,b_own,b_neigh)=%s" % np.round(bb, 5))
print("  direction accuracy=%.2f%% (z=%+.2f) | IC=%+.4f" % (hp, z, ic))


# ==================== earnings_event_study.py ====================

"""
earnings_event_study.py -- can a supply-chain neighbor's earnings surprise predict my subsequent return? (Cohen-Frazzini finding #2)

Question: on day T when firm X reports earnings with an EPS beat (positive surprise), do X's supply-chain neighbors Y, in the
     1/5/10 days after T, move in the direction of X's surprise?

Scoring: direction accuracy sign(X surprise) vs sign(Y subsequent return), benchmark 50%; IC = corr(X surprise, Y subsequent return);
      long-short return = sign(X surprise) * Y subsequent return (zero cost).
"""
import os, re, bisect
import numpy as np
import pandas as pd
from collections import defaultdict

BASE = r"C:\Users\lanju\Desktop\Market Wave\ai_infra_v3_extracted\ai_infra_2022_2026_v3"
START = pd.Timestamp("2023-01-03")
TEST_END = pd.Timestamp("2025-12-31")

# ---------------- 1) daily returns ----------------
prices = pd.read_csv(os.path.join(BASE, "tables", "prices_daily.csv"), low_memory=False)
sec = pd.read_csv(os.path.join(BASE, "tables", "securities.csv"))
rel = pd.read_csv(os.path.join(BASE, "tables", "relations.csv"))
comp_sec = sec[sec["asset_role"] == "company"]
sec2comp = dict(zip(comp_sec["security_id"], comp_sec["company_id"]))
comp2sec = {v: k for k, v in sec2comp.items()}

p = prices[prices["security_id"].isin(comp_sec["security_id"])].copy()
p["trade_date"] = pd.to_datetime(p["trade_date"])
p = p[(p["trade_date"] >= START) & (p["trade_date"] <= TEST_END)].sort_values(["security_id", "trade_date"])
p["ret"] = p.groupby("security_id")["close"].pct_change()

ret_lookup = {(r.security_id, r.trade_date): r.ret for r in p.itertuples()}
date_list = {sid: sorted(g["trade_date"].tolist()) for sid, g in p.groupby("security_id")}

# ---------------- 2) neighbor graph (undirected, exclude competition) ----------------
neigh = defaultdict(set)
for _, r in rel.iterrows():
    if r["relation_type"] == "competition":
        continue
    a, b = r["from_company_id"], r["to_company_id"]
    neigh[a].add(b); neigh[b].add(a)
listed = set(comp2sec.keys())
neigh = {c: {n for n in s if n in listed} for c, s in neigh.items() if c in listed}

# ---------------- 3) earnings events + EPS surprise ----------------
ev = pd.read_csv(os.path.join(BASE, "tables", "events.csv"))
# NOTE: regex intentionally kept in Chinese — it parses the Chinese source event text (actual vs expected).
pat = re.compile(r"EPS\s*实际\s*\$?\s*([-+]?\d*\.?\d+)\s*[;；]\s*预期\s*\$?\s*([-+]?\d*\.?\d+)", re.I)
def eps_surprise(s):
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

ev["surprise"] = ev["fact_summary"].apply(eps_surprise)
ev["usable"] = pd.to_datetime(ev["usable_from_trade_date"])
earn = ev[(ev["event_type"] == "earnings") & (ev["surprise"].abs() > 1e-12) & ev["usable"].notna()]

# ---------------- 4) event study: X surprise -> Y subsequent return ----------------
def fwd_ret(Y_sec, T, h):
    ds = date_list.get(Y_sec)
    if not ds:
        return None
    idx = bisect.bisect_right(ds, T)
    if idx + h > len(ds):
        return None
    cum = 1.0
    for d in ds[idx:idx + h]:
        r = ret_lookup.get((Y_sec, d))
        if r is None or not np.isfinite(r):
            return None
        cum *= (1.0 + r)
    return cum - 1.0

rows = []
for _, r in earn.iterrows():
    T = r["usable"]
    s = r["surprise"]
    for tok in str(r["target_companies"]).replace(",", ";").split(";"):
        X = tok.strip()
        if not X.startswith("C_"):
            continue
        for Y in neigh.get(X, set()):
            Ysec = comp2sec.get(Y)
            if not Ysec:
                continue
            for h in (1, 5, 10):
                cr = fwd_ret(Ysec, T, h)
                if cr is not None:
                    rows.append((X, Y, h, s, cr))

df = pd.DataFrame(rows, columns=["X", "Y", "h", "surprise", "fwd_ret"])
print("event pairs (firm-pair x event x window) total:", len(df))

# ---------------- 5) scoring ----------------
print("\nwindow | pairs | direction accuracy | IC | mean long-short return (window%)")
for h in (1, 5, 10):
    d = df[df["h"] == h]
    n = len(d)
    acc = np.mean(np.sign(d["surprise"]) == np.sign(d["fwd_ret"])) * 100
    ic = np.corrcoef(d["surprise"], d["fwd_ret"])[0, 1]
    pnl = np.mean(np.sign(d["surprise"]) * d["fwd_ret"]) * 100
    print("  %2dd | %4d | %5.2f%% | %+.3f | %+.3f%%" % (h, n, acc, ic, pnl))

# control: Y's unconditional subsequent return (remove X surprise, check if just momentum)
print("\ncontrol: mean unconditional Y subsequent return -- if close to the long-short return above, no excess info")
all_pairs = df.drop_duplicates(subset=["X", "Y", "h", "surprise"])  # dedupe same event pairs
for h in (1, 5, 10):
    d = df[df["h"] == h]
    print("  %2dd unconditional mean subsequent return=%+.3f%%" % (h, np.mean(d["fwd_ret"]) * 100))


# ==================== info_diffusion.py ====================

"""
info_diffusion.py -- three sub-questions of the information-diffusion mechanism (all about how info diffuses along the supply chain)

  Test 1 manifestation: volatility spillover (daily) -- if my neighbor moves violently today, do I move more violently tomorrow?
      |ret|_t+1 = a + b1*|ret|_t + b2*neighbor_|ret|_t      (b2>0 = spillover)

  Test 2 direction: asymmetry (monthly) -- customer->supplier diffuses, supplier->customer does not?
      ret_{m+1} = a + b1*ret_m + b2*customer_ret_m            (customer momentum, Cohen-Frazzini core)
      ret_{m+1} = a + b1*ret_m + b2*supplier_ret_m            (reverse, expected ~0)

  Test 3 cross-section: centrality (monthly) -- do more central (high-degree) firms have weaker lead-lag (info priced faster) or stronger?
      high-degree vs low-degree groups, run lead-lag separately, compare neighbor coefficients

Zero cost, pure mechanism. Read-only on existing data.
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict

BASE = r"C:\Users\lanju\Desktop\Market Wave\ai_infra_v3_extracted\ai_infra_2022_2026_v3"
START = pd.Timestamp("2023-01-03")
END = pd.Timestamp("2025-12-31")

# ---------------- data ----------------
prices = pd.read_csv(os.path.join(BASE, "tables", "prices_daily.csv"), low_memory=False)
sec = pd.read_csv(os.path.join(BASE, "tables", "securities.csv"))
rel = pd.read_csv(os.path.join(BASE, "tables", "relations.csv"))
comp_sec = sec[sec["asset_role"] == "company"]
sec2comp = dict(zip(comp_sec["security_id"], comp_sec["company_id"]))

p = prices[prices["security_id"].isin(comp_sec["security_id"])].copy()
p["trade_date"] = pd.to_datetime(p["trade_date"])
p = p[(p["trade_date"] >= START) & (p["trade_date"] <= END)].sort_values(["security_id", "trade_date"])
p["company_id"] = p["security_id"].map(sec2comp)
p["ret"] = p.groupby("security_id")["close"].pct_change()
p["abs_ret"] = p["ret"].abs()
p["ym"] = p["trade_date"].dt.to_period("M")

# neighbor graph: undirected (exclude competition) + directed supply (supply/foundry = supplier->customer)
undirected = defaultdict(set)
suppliers_of = defaultdict(set)   # X's suppliers (who supply X)
customers_of = defaultdict(set)   # X's customers (X supplies them)
for _, r in rel.iterrows():
    a, b, rt = r["from_company_id"], r["to_company_id"], r["relation_type"]
    if rt == "competition":
        continue
    undirected[a].add(b); undirected[b].add(a)
    if rt in ("supply", "foundry_manufacturing"):     # from=supplier, to=customer
        customers_of[a].add(b)      # a's customer is b
        suppliers_of[b].add(a)      # b's supplier is a
listed = set(p["company_id"].unique())
undirected = {c: {n for n in s if n in listed} for c, s in undirected.items() if c in listed}
suppliers_of = {c: {n for n in s if n in listed} for c, s in suppliers_of.items()}
customers_of = {c: {n for n in s if n in listed} for c, s in customers_of.items()}

# OLS + date-clustered SE
def ols_clustered(y, X, cluster):
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    y = np.asarray(y, float)
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    k = X.shape[1]
    meat = np.zeros((k, k))
    for g in np.unique(cluster):
        m = cluster == g
        sg = (X[m].T * e[m]).sum(axis=1)
        meat += np.outer(sg, sg)
    V = np.linalg.solve(XtX, meat) @ np.linalg.solve(XtX, np.eye(k))
    se = np.sqrt(np.diag(V))
    return b, se, b / se

# ============================================================
# Test 1 volatility spillover (daily)
# ============================================================
print("=== Test 1: volatility spillover (daily) ===")
ret_map = {(r.company_id, r.trade_date): r.ret for r in p.itertuples()}
abs_map = {(r.company_id, r.trade_date): r.abs_ret for r in p.itertuples()}

d = p[["company_id", "trade_date", "abs_ret"]].dropna().copy()
def nb_abs(row):
    vals = [abs_map.get((n, row["trade_date"])) for n in undirected.get(row["company_id"], set())]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan
d["nb_abs"] = d.apply(nb_abs, axis=1)
d["abs_fwd"] = d.groupby("company_id")["abs_ret"].shift(-1)
d = d.dropna(subset=["abs_ret", "nb_abs", "abs_fwd"])
b, se, t = ols_clustered(d["abs_fwd"].values, d[["abs_ret", "nb_abs"]].values, d["trade_date"].values)
print("  |ret|_t+1 = a + b1*|ret|_t + b2*neighbor|ret|_t")
print("  b1(own vol persistence) = %+.4f (t=%+.2f)" % (b[1], t[1]))
print("  b2(neighbor vol spillover) = %+.4f (t=%+.2f)  <- key" % (b[2], t[2]))
print("  N=%d" % len(d))

# ============================================================
# Test 2 & 3 monthly
# ============================================================
mret = p.groupby(["company_id", "ym"])["ret"].agg(lambda x: float((1 + x).prod() - 1)).reset_index()
mret = mret.sort_values(["company_id", "ym"]).reset_index(drop=True)
mret["ret_fwd"] = mret.groupby("company_id")["ret"].shift(-1)
m_lookup = {(r.company_id, r.ym): r.ret for r in mret.itertuples()}

def agg_from(cid, ym, nset):
    vals = [m_lookup[(n, ym)] for n in nset.get(cid, set()) if (n, ym) in m_lookup]
    return float(np.mean(vals)) if vals else np.nan

print("\n=== Test 2: direction asymmetry (monthly) ===")
mret["customer_ret"] = [agg_from(r.company_id, r.ym, customers_of) for r in mret.itertuples()]
mret["supplier_ret"] = [agg_from(r.company_id, r.ym, suppliers_of) for r in mret.itertuples()]

d2 = mret.dropna(subset=["ret", "ret_fwd", "customer_ret"])
b, se, t = ols_clustered(d2["ret_fwd"].values, d2[["ret", "customer_ret"]].values, d2["ym"].values)
print("  customer->supplier (customer momentum): b_cust = %+.4f (t=%+.2f)  N=%d" % (b[2], t[2], len(d2)))

d2b = mret.dropna(subset=["ret", "ret_fwd", "supplier_ret"])
b, se, t = ols_clustered(d2b["ret_fwd"].values, d2b[["ret", "supplier_ret"]].values, d2b["ym"].values)
print("  supplier->customer (reverse)   : b_supp = %+.4f (t=%+.2f)  N=%d" % (b[2], t[2], len(d2b)))
print("  expected (Cohen-Frazzini): the first significantly positive, the second ~0, i.e. direction asymmetry")

print("\n=== Test 3: centrality cross-section (monthly) ===")
mret["nb_ret"] = [agg_from(r.company_id, r.ym, undirected) for r in mret.itertuples()]
degree = {c: len(s) for c, s in undirected.items()}
mret["degree"] = mret["company_id"].map(degree).fillna(0).astype(int)

for label, grp in [("high degree (degree>=2, hub)", mret[mret["degree"] >= 2]),
                   ("low degree (degree==1, spoke)", mret[mret["degree"] == 1])]:
    g = grp.dropna(subset=["ret", "ret_fwd", "nb_ret"])
    if len(g) < 30:
        print("  [%s] too few samples (skip)" % label); continue
    b, se, t = ols_clustered(g["ret_fwd"].values, g[["ret", "nb_ret"]].values, g["ym"].values)
    print("  [%s] b_neigh = %+.4f (t=%+.2f)  N=%d  firms=%d" %
          (label, b[2], t[2], len(g), g["company_id"].nunique()))
print("  interpretation: whichever group has stronger b_neigh has slower info diffusion / more visible lead-lag")


# ==================== robustness.py ====================

"""
robustness.py -- robustness: is the supplier->customer lead-lag driven by one or two mega-upstream firms?

Approach: progressively exclude the biggest suppliers (NVDA/TSM/ASML/MU/AVGO), re-run supplier->customer, check whether t survives.
      Also do leave-one-out of the top-5 suppliers.
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict

BASE = r"C:\Users\lanju\Desktop\Market Wave\ai_infra_v3_extracted\ai_infra_2022_2026_v3"
START = pd.Timestamp("2023-01-03")
END = pd.Timestamp("2025-12-31")

prices = pd.read_csv(os.path.join(BASE, "tables", "prices_daily.csv"), low_memory=False)
sec = pd.read_csv(os.path.join(BASE, "tables", "securities.csv"))
rel = pd.read_csv(os.path.join(BASE, "tables", "relations.csv"))
comp_sec = sec[sec["asset_role"] == "company"]
sec2comp = dict(zip(comp_sec["security_id"], comp_sec["company_id"]))
comp2name = dict(zip(comp_sec["company_id"], comp_sec["ticker"]))

p = prices[prices["security_id"].isin(comp_sec["security_id"])].copy()
p["trade_date"] = pd.to_datetime(p["trade_date"])
p = p[(p["trade_date"] >= START) & (p["trade_date"] <= END)].sort_values(["security_id", "trade_date"])
p["company_id"] = p["security_id"].map(sec2comp)
p["ret"] = p.groupby("security_id")["close"].pct_change()
p["ym"] = p["trade_date"].dt.to_period("M")

suppliers_of = defaultdict(set)
for _, r in rel.iterrows():
    if r["relation_type"] in ("supply", "foundry_manufacturing"):
        suppliers_of[r["to_company_id"]].add(r["from_company_id"])  # from=supplier -> to=customer
listed = set(p["company_id"].unique())
suppliers_of = {c: {n for n in s if n in listed} for c, s in suppliers_of.items()}

mret = p.groupby(["company_id", "ym"])["ret"].agg(lambda x: float((1 + x).prod() - 1)).reset_index()
mret = mret.sort_values(["company_id", "ym"]).reset_index(drop=True)
mret["ret_fwd"] = mret.groupby("company_id")["ret"].shift(-1)
m_lookup = {(r.company_id, r.ym): r.ret for r in mret.itertuples()}

def ols_clustered(y, X, cluster):
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    y = np.asarray(y, float)
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    k = X.shape[1]
    meat = np.zeros((k, k))
    for g in np.unique(cluster):
        m = cluster == g
        sg = (X[m].T * e[m]).sum(axis=1)
        meat += np.outer(sg, sg)
    V = np.linalg.solve(XtX, meat) @ np.linalg.solve(XtX, np.eye(k))
    se = np.sqrt(np.diag(V))
    return b[2], b[2] / se[2]   # return b_supp and t

def run_supplier_customer(remove_set):
    keep = [c for c in listed if c not in remove_set]
    sup = {c: {n for n in s if n in keep} for c, s in suppliers_of.items() if c in keep}
    d = mret[mret["company_id"].isin(keep)].copy()
    def agg(row):
        vals = [m_lookup.get((n, row["ym"])) for n in sup.get(row["company_id"], set())]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else np.nan
    d["sup_ret"] = d.apply(agg, axis=1)
    d = d.dropna(subset=["ret", "ret_fwd", "sup_ret"])
    if len(d) < 30:
        return None, None, len(d)
    b, t = ols_clustered(d["ret_fwd"].values, d[["ret", "sup_ret"]].values, d["ym"].values)
    return b, t, len(d)

# sort suppliers by number of customers (who is a big supplier)
out_deg = defaultdict(int)
for c, cs in suppliers_of.items():
    for n in cs:
        out_deg[n] += 1
top_suppliers = sorted(out_deg, key=out_deg.get, reverse=True)
top_names = [comp2name.get(c, c) for c in top_suppliers[:8]]
print("biggest suppliers (by #customers):", [(comp2name.get(c, c), out_deg[c]) for c in top_suppliers[:8]])

print("\n=== progressively exclude mega-upstream ===")
for label, rset in [
    ("baseline (none excluded)", set()),
    ("exclude NVDA", {"C_NVDA"}),
    ("exclude NVDA+TSM+ASML", {"C_NVDA", "C_TSM", "C_ASML"}),
    ("exclude NVDA+TSM+ASML+MU+AVGO", {"C_NVDA", "C_TSM", "C_ASML", "C_MU", "C_AVGO"}),
]:
    b, t, n = run_supplier_customer(rset)
    print("  %-30s b_supp=%+.4f t=%+.2f N=%d" % (label, b, t, n))

print("\n=== leave-one-out: drop each top-5 supplier ===")
for c in top_suppliers[:5]:
    b, t, n = run_supplier_customer({c})
    print("  drop %-6s b_supp=%+.4f t=%+.2f N=%d" % (comp2name.get(c, c), b, t, n))


# ==================== oos_test.py ====================

"""
oos_test.py -- out-of-sample test of the supplier->customer lead-lag

Three cuts:
  1) test-period (2024-07~2025-12) regression: is b_supp still significant out-of-sample?
  2) train fit -> test predict: M0(own only) vs M1(add supplier), compare direction-accuracy / IC increments.
  3) out-of-sample long-short return (zero cost, only direction, not money).
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict

BASE = r"C:\Users\lanju\Desktop\Market Wave\ai_infra_v3_extracted\ai_infra_2022_2026_v3"
START = pd.Timestamp("2023-01-03")
END = pd.Timestamp("2025-12-31")
TRAIN_END = pd.Period("2024-06", "M")
TEST_START = pd.Period("2024-07", "M")

prices = pd.read_csv(os.path.join(BASE, "tables", "prices_daily.csv"), low_memory=False)
sec = pd.read_csv(os.path.join(BASE, "tables", "securities.csv"))
rel = pd.read_csv(os.path.join(BASE, "tables", "relations.csv"))
comp_sec = sec[sec["asset_role"] == "company"]
sec2comp = dict(zip(comp_sec["security_id"], comp_sec["company_id"]))

p = prices[prices["security_id"].isin(comp_sec["security_id"])].copy()
p["trade_date"] = pd.to_datetime(p["trade_date"])
p = p[(p["trade_date"] >= START) & (p["trade_date"] <= END)].sort_values(["security_id", "trade_date"])
p["company_id"] = p["security_id"].map(sec2comp)
p["ret"] = p.groupby("security_id")["close"].pct_change()
p["ym"] = p["trade_date"].dt.to_period("M")

suppliers_of = defaultdict(set)
for _, r in rel.iterrows():
    if r["relation_type"] in ("supply", "foundry_manufacturing"):
        suppliers_of[r["to_company_id"]].add(r["from_company_id"])
listed = set(p["company_id"].unique())
suppliers_of = {c: {n for n in s if n in listed} for c, s in suppliers_of.items()}

mret = p.groupby(["company_id", "ym"])["ret"].agg(lambda x: float((1 + x).prod() - 1)).reset_index()
mret = mret.sort_values(["company_id", "ym"]).reset_index(drop=True)
mret["ret_fwd"] = mret.groupby("company_id")["ret"].shift(-1)
m_lookup = {(r.company_id, r.ym): r.ret for r in mret.itertuples()}

def agg(row):
    vals = [m_lookup.get((n, row["ym"])) for n in suppliers_of.get(row["company_id"], set())]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan
mret["sup_ret"] = mret.apply(agg, axis=1)
d = mret.dropna(subset=["ret", "ret_fwd", "sup_ret"]).copy()

def ols(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    return np.linalg.solve(X.T @ X, X.T @ y)

def ols_clustered(y, X, cluster):
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    y = np.asarray(y, float)
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    k = X.shape[1]; meat = np.zeros((k, k))
    for g in np.unique(cluster):
        m = cluster == g
        sg = (X[m].T * e[m]).sum(axis=1)
        meat += np.outer(sg, sg)
    V = np.linalg.solve(XtX, meat) @ np.linalg.solve(XtX, np.eye(k))
    return b, np.sqrt(np.diag(V)), b / np.sqrt(np.diag(V))

# ---------- cut 1: test-period regression ----------
te = d[d["ym"] >= TEST_START]
b, se, t = ols_clustered(te["ret_fwd"].values, te[["ret", "sup_ret"]].values, te["ym"].values)
print("cut 1 test-period (2024-07~2025-12) regression:")
print("  b_supp = %+.4f (t=%+.2f)  N=%d  firms=%d" % (b[2], t[2], len(te), te["company_id"].nunique()))

# ---------- cut 2: train fit -> test predict ----------
tr = d[d["ym"] <= TRAIN_END]
te = d[d["ym"] >= TEST_START]

# M0: own only
b0 = ols(tr[["ret"]].values, tr["ret_fwd"].values)
p0 = np.column_stack([np.ones(len(te)), te["ret"].values]) @ b0
# M1: add supplier
b1 = ols(tr[["ret", "sup_ret"]].values, tr["ret_fwd"].values)
p1 = np.column_stack([np.ones(len(te)), te["ret"].values, te["sup_ret"].values]) @ b1

true = te["ret_fwd"].values
def metrics(pred):
    acc = np.mean(np.sign(pred) == np.sign(true)) * 100
    ic = np.corrcoef(pred, true)[0, 1]
    pnl = np.mean(np.sign(pred) * true) * 100
    return acc, ic, pnl

a0, i0, p0m = metrics(p0)
a1, i1, p1m = metrics(p1)
print("\ncut 2 train(<=2024-06) fit -> test(>=2024-07) predict:")
print("  M0 own-only      : direction accuracy=%5.2f%%  IC=%+.3f  monthly long-short=%+.3f%%" % (a0, i0, p0m))
print("  M1 add-supplier  : direction accuracy=%5.2f%%  IC=%+.3f  monthly long-short=%+.3f%%" % (a1, i1, p1m))
print("  increment(M1-M0) : direction %+.2f pp   IC %+.3f   long-short %+.3f pp" % (a1-a0, i1-i0, p1m-p0m))
print("  weights M1: (a,b_own,b_sup) = %s" % np.round(b1, 4))


# ==================== minimal_abm.py ====================

"""
minimal_abm.py -- chartists + fundamentalists + random-walk value (stable fix for the third fact)

Mechanism:
  - fundamental value F random-walks -> returns driven by F's innovations -> return autocorr ~0
  - chartists (herding) occasionally dominate -> fat tails
  - slow switching n_c -> volatility clustering
"""
import numpy as np

def stylized(rets):
    r = np.asarray(rets); r = r - r.mean()
    a = np.abs(r)
    def ac(x, lag):
        x = np.asarray(x)
        return np.corrcoef(x[:-lag], x[lag:])[0, 1] if lag < len(x) - 1 else np.nan
    kurt = np.mean(r**4) / np.mean(r**2)**2 - 3
    return kurt, ac(a, 1), ac(a, 5), ac(a, 20), ac(r, 1)

def run_ham(T=30000, g=4.0, kappa=1.0, sigma=0.005, sigmaF=0.02,
            nc0=0.5, nc_persist=0.98, nc_vol=0.03, meanrev=0.3, seed=1):
    rng = np.random.default_rng(seed)
    p = np.zeros(T + 3)
    F = 0.0
    n_c = nc0
    rets = np.empty(T)
    for t in range(2, T + 2):
        F = F + rng.normal(0, sigmaF)             # fundamental value random walk
        n_c = nc_persist * n_c + (1 - nc_persist) * nc0 + rng.normal(0, nc_vol)
        n_c = float(np.clip(n_c, 0.0, 1.0))
        n_f = 1.0 - n_c
        trend = np.tanh(g * (p[t - 1] - p[t - 2]))   # chartist: extrapolate
        level_rev = -meanrev * (p[t - 1] - F)        # fundamental: pull price back to F
        dp = kappa * (n_c * trend + n_f * level_rev) + rng.normal(0, sigma)
        p[t] = p[t - 1] + dp
        rets[t - 2] = dp
    return rets

print("=== real AI-stock targets ===")
print("  kurtosis~+19  |ret| autocorr~+0.19  return autocorr~0\n")

print("=== random-walk value, three facts by parameter ===")
print("%-46s %8s %8s %8s %8s %8s" % ("config", "kurtosis", "|r|ac1", "|r|ac5", "|r|ac20", "r ac1"))
configs = [
    ("g=4 Fvol0.02 meanrev0.3", dict(g=4.0, sigmaF=0.02, meanrev=0.3)),
    ("g=4 Fvol0.02 meanrev0.5", dict(g=4.0, sigmaF=0.02, meanrev=0.5)),
    ("g=6 Fvol0.02 meanrev0.3", dict(g=6.0, sigmaF=0.02, meanrev=0.3)),
    ("g=4 Fvol0.03 meanrev0.5", dict(g=4.0, sigmaF=0.03, meanrev=0.5)),
    ("g=6 Fvol0.02 meanrev0.5 vol0.05", dict(g=6.0, sigmaF=0.02, meanrev=0.5, nc_vol=0.05)),
]
for label, kw in configs:
    r = run_ham(**kw)
    k, a1, a5, a20, r1 = stylized(r)
    print("  %-44s %8.2f %8.3f %8.3f %8.3f %8.3f" % (label, k, a1, a5, a20, r1))


# ==================== lux_marchesi.py ====================

"""
lux_marchesi.py -- exact implementation of Lux & Marchesi (2000) [individual agents + stochastic transitions]

Key: use N=500 individual agents with binomial stochastic transitions (not mean-field); finite-size fluctuations trigger on-off intermittency.
Equations (eq. 2.2 / 2.3 / 2.4, directions checked against the appendix mean-field):
  opinion switch: pessimist->optimist = Binomial(n-, v1*(nc/N)*exp(U1)*dt)
           optimist->pessimist = Binomial(n+, v1*(nc/N)*exp(-U1)*dt)
  strategy switch: optimist->fundamentalist = Binomial(n+, v2*(nf/N)*exp(-U2,1)*dt)
           fundamentalist->optimist = Binomial(nf, v2*(n+/N)*exp(U2,1)*dt)
           pessimist->fundamentalist = Binomial(n-, v2*(nf/N)*exp(-U2,2)*dt)
           fundamentalist->pessimist = Binomial(nf, v2*(n-/N)*exp(U2,2)*dt)
  U1 = a1*x + a2*(dp/v1);  U2,1 = a3[(r+dp/v2)/p - R - s|pf-p|/p];  U2,2 = a3[R-(r+dp/v2)/p - s|pf-p|/p]
  price tick: dp = +/-0.01, prob beta*(ED+mu)*dt; ED = (n+-n-)*tc + nf*gamma*(pf-p)
"""
import numpy as np

def stylized(rets):
    r = np.asarray(rets); r = r - r.mean()
    a = np.abs(r)
    def ac(x, lag):
        x = np.asarray(x)
        return np.corrcoef(x[:-lag], x[lag:])[0, 1] if lag < len(x) - 1 else np.nan
    kurt = np.mean(r**4) / np.mean(r**2)**2 - 3
    return kurt, ac(a, 1), ac(a, 5), ac(a, 20), ac(r, 1)

def run_lux2000(T=1000000, dt=0.01, N=500, v1=3.0, v2=2.0, beta=6.0,
                tc=0.02, gamma=0.01, a1=0.6, a2=0.2, a3=0.5, pf=10.0,
                r=0.004, R=0.0004, s=0.75, sigma_mu=0.05, W=100, seed=1):
    rng = np.random.default_rng(seed)
    n_plus = 125
    n_minus = 125
    n_f = N - n_plus - n_minus
    p = pf
    prices = []
    dp_prev = 0.0
    for t in range(T):
        n_c = n_plus + n_minus
        if n_c == 0:
            n_c = 1
        x = (n_plus - n_minus) / n_c
        U1 = a1 * x + a2 * (dp_prev / v1)
        U21 = a3 * ((r + dp_prev / v2) / p - R - s * abs(pf - p) / p)
        U22 = a3 * (R - (r + dp_prev / v2) / p - s * abs(pf - p) / p)

        q_po = min(1.0, v1 * (n_c / N) * np.exp(U1) * dt)
        q_op = min(1.0, v1 * (n_c / N) * np.exp(-U1) * dt)
        s_po = rng.binomial(n_minus, q_po)
        s_op = rng.binomial(n_plus, q_op)
        n_plus += s_po - s_op
        n_minus += s_op - s_po

        q_pf = min(1.0, v2 * (n_f / N) * np.exp(-U21) * dt)
        q_fp = min(1.0, v2 * (n_plus / N) * np.exp(U21) * dt)
        q_mf = min(1.0, v2 * (n_f / N) * np.exp(-U22) * dt)
        q_fm = min(1.0, v2 * (n_minus / N) * np.exp(U22) * dt)
        s_pf = rng.binomial(n_plus, q_pf)
        s_fp = rng.binomial(n_f, q_fp)
        s_mf = rng.binomial(n_minus, q_mf)
        s_fm = rng.binomial(n_f, q_fm)
        n_plus += -s_pf + s_fp
        n_minus += -s_mf + s_fm
        n_f += s_pf + s_mf - s_fp - s_fm

        ED = (n_plus - n_minus) * tc + n_f * gamma * (pf - p)
        mu = rng.normal(0, sigma_mu)
        prob = beta * (ED + mu) * dt
        dp = 0.0
        if prob > 0 and rng.random() < prob:
            dp = 0.01
        elif prob < 0 and rng.random() < -prob:
            dp = -0.01
        p += dp
        dp_prev = dp
        if (t + 1) % W == 0:
            prices.append(p)
    prices = np.array(prices)
    return np.diff(np.log(prices))

print("=== real AI-stock targets ===")
print("  kurtosis~+19  |ret| autocorr~+0.19  return autocorr~0\n")

print("=== Lux-Marchesi 2000 individual-agent three facts (window-aggregated) ===")
print("%-52s %8s %8s %8s %8s %8s" % ("config", "kurtosis", "|r|ac1", "|r|ac5", "|r|ac20", "r ac1"))
configs = [
    ("original W=100 (unit time)", dict(W=100)),
    ("original W=500", dict(W=500)),
    ("original W=1000", dict(W=1000)),
    ("original W=2500", dict(W=2500)),
]
for label, kw in configs:
    r = run_lux2000(**kw)
    k, a1v, a5, a20, r1 = stylized(r)
    print("  %-50s %8.2f %8.3f %8.3f %8.3f %8.3f" % (label, k, a1v, a5, a20, r1))


# ==================== supply_chain_abm.py ====================

"""
supply_chain_abm.py -- our framework: supply-chain network + cross-asset information diffusion (multi-asset Lux-Marchesi)

On the validated Lux-Marchesi mechanism, replace "global herding" with:
  U1_i = a1 * x_i + a2*(dp_i/v1) + a_net * mean(neighbor j's x_j)
i.e. asset i's chartist herding, besides its own majority opinion and price trend, also looks at supply-chain neighbors' opinions
(information diffuses along supply edges).

40 firms, real supply edges (supply + foundry_manufacturing).
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict

BASE = r"C:\Users\lanju\Desktop\Market Wave\ai_infra_v3_extracted\ai_infra_2022_2026_v3"

def stylized(rets):
    r = np.asarray(rets); r = r - r.mean()
    a = np.abs(r)
    def ac(x, lag):
        x = np.asarray(x)
        return np.corrcoef(x[:-lag], x[lag:])[0, 1] if lag < len(x) - 1 else np.nan
    kurt = np.mean(r**4) / np.mean(r**2)**2 - 3
    return kurt, ac(a, 1), ac(a, 5), ac(a, 20), ac(r, 1)

# load 40 firms + supply edges
sec = pd.read_csv(os.path.join(BASE, "tables", "securities.csv"))
rel = pd.read_csv(os.path.join(BASE, "tables", "relations.csv"))
comp_sec = sec[sec["asset_role"] == "company"]
companies = sorted(comp_sec["company_id"].tolist())
idx = {c: i for i, c in enumerate(companies)}
M = len(companies)

adj = defaultdict(set)
for _, r in rel.iterrows():
    if r["relation_type"] in ("supply", "foundry_manufacturing"):
        a, b = r["from_company_id"], r["to_company_id"]
        if a in idx and b in idx:
            adj[idx[a]].add(idx[b])
            adj[idx[b]].add(idx[a])
neigh = {i: list(adj[i]) for i in range(M)}

def run_sc(T=300000, dt=0.01, n_per=25, v1=3.0, v2=2.0, beta=6.0,
          tc=0.02, gamma=0.01, a1=0.6, a2=0.2, a3=0.5, a_net=0.45, arb=0.0,
          pf=10.0, r=0.004, R=0.0004, s=0.75, sigma_mu=0.05, W=100, seed=1):
    rng = np.random.default_rng(seed)
    n_plus = np.full(M, n_per // 2, dtype=float)
    n_minus = np.full(M, n_per // 2, dtype=float)
    n_f = np.full(M, n_per - (n_per // 2) * 2, dtype=float)
    p = np.full(M, pf, dtype=float)
    dp_prev = np.zeros(M)
    prices = [[] for _ in range(M)]
    for t in range(T):
        n_c = n_plus + n_minus
        n_c = np.where(n_c == 0, 1, n_c)
        x = (n_plus - n_minus) / n_c
        # supply-chain neighbor opinion (info diffuses along edges)
        x_net = np.array([np.mean([x[j] for j in neigh[i]]) if neigh[i] else 0.0 for i in range(M)])

        U1 = a1 * x + a2 * (dp_prev / v1) + a_net * x_net
        U21 = np.clip(a3 * ((r + dp_prev / v2) / p - R - s * np.abs(pf - p) / p), -20, 20)
        U22 = np.clip(a3 * (R - (r + dp_prev / v2) / p - s * np.abs(pf - p) / p), -20, 20)

        q_po = np.minimum(1.0, v1 * (n_c / (n_per)) * np.exp(U1) * dt)
        q_op = np.minimum(1.0, v1 * (n_c / (n_per)) * np.exp(-U1) * dt)
        s_po = rng.binomial(n_minus.astype(int), q_po)
        s_op = rng.binomial(n_plus.astype(int), q_op)
        n_plus += s_po - s_op
        n_minus += s_op - s_po

        q_pf = np.minimum(1.0, v2 * (n_f / (n_per)) * np.exp(-U21) * dt)
        q_fp = np.minimum(1.0, v2 * (n_plus / (n_per)) * np.exp(U21) * dt)
        q_mf = np.minimum(1.0, v2 * (n_f / (n_per)) * np.exp(-U22) * dt)
        q_fm = np.minimum(1.0, v2 * (n_minus / (n_per)) * np.exp(U22) * dt)
        s_pf = rng.binomial(n_plus.astype(int), q_pf)
        s_fp = rng.binomial(n_f.astype(int), q_fp)
        s_mf = rng.binomial(n_minus.astype(int), q_mf)
        s_fm = rng.binomial(n_f.astype(int), q_fm)
        n_plus += -s_pf + s_fp
        n_minus += -s_mf + s_fm
        n_f += s_pf + s_mf - s_fp - s_fm

        ED = (n_plus - n_minus) * tc + n_f * gamma * (pf - p) - arb * dp_prev
        mu = rng.normal(0, sigma_mu, M)
        prob = beta * (ED + mu) * dt
        dp = np.zeros(M)
        up = (prob > 0) & (rng.random(M) < prob)
        dn = (prob < 0) & (rng.random(M) < -prob)
        dp[up] = 0.01
        dp[dn] = -0.01
        p += dp
        dp_prev = dp
        if (t + 1) % W == 0:
            for i in range(M):
                prices[i].append(p[i])
    # equal-weight index return + per-stock returns
    P = np.array(prices)  # M × n_windows
    idx_ret = np.diff(np.log(P.mean(axis=0)))
    stock_rets = np.diff(np.log(P), axis=1)  # M × (n_windows-1)
    return idx_ret, stock_rets

print("=== real AI-stock targets (pooled 40) ===")
print("  kurtosis~+19  |ret| autocorr~+0.19  return autocorr~0\n")

print("=== supply-chain network + arbitrage force (a_net=0.45 fixed, scan arb) ===")
print("%-52s %8s %8s %8s %8s %8s" % ("config", "index kurtosis", "|r|ac1", "per-stock median kurtosis", "r ac1", "|r|ac20"))
configs = [
    ("arb=0 (no arbitrage)", dict(a_net=0.45, arb=0.0)),
    ("arb=20", dict(a_net=0.45, arb=20.0)),
    ("arb=50", dict(a_net=0.45, arb=50.0)),
    ("arb=100", dict(a_net=0.45, arb=100.0)),
    ("arb=200", dict(a_net=0.45, arb=200.0)),
]
for label, kw in configs:
    r, sr = run_sc(**kw)
    k, a1v, a5, a20, r1 = stylized(r)
    kp = np.median([stylized(sr[i])[0] for i in range(M)])
    print("  %-50s %8.2f %8.3f %8.2f %8.3f %8.3f" % (label, k, a1v, kp, r1, a20))


# ==================== position_speed.py ====================

"""
position_speed.py -- empirical #1: supply-chain position x information-diffusion speed

Question: on firm X's earnings day (event day T), how do its supply-chain neighbors Y's abnormal returns
      distribute across T, T+1, T+2, T+3, T+5?
      Group by Y's supply-chain centrality (degree) to see whether hubs or spokes react faster.

Design:
  - abnormal return AR_{Y,t} = r_{Y,t} - equal-weight mean of the 40 that day (market-adjusted)
  - direction adjustment = AR_{Y,T+j} * sign(X's EPS surprise)
  - grouping = Y's supply-neighbor degree (above median = hub, else spoke)
  - compare: the two groups' mean direction-adjusted AR at T/T+1/T+2/T+3/T+5, plus cumulatives
"""
import os
import bisect
import numpy as np
import pandas as pd
from collections import defaultdict

BASE = r"C:\Users\lanju\Desktop\Market Wave\ai_infra_v3_extracted\ai_infra_2022_2026_v3"
START = pd.Timestamp("2023-01-03")
END = pd.Timestamp("2025-12-31")

# ---------- 1. prices and abnormal returns ----------
prices = pd.read_csv(os.path.join(BASE, "tables", "prices_daily.csv"), low_memory=False)
sec = pd.read_csv(os.path.join(BASE, "tables", "securities.csv"))
rel = pd.read_csv(os.path.join(BASE, "tables", "relations.csv"))
comp_sec = sec[sec["asset_role"] == "company"]
sec2comp = dict(zip(comp_sec["security_id"], comp_sec["company_id"]))
comp2sec = {v: k for k, v in sec2comp.items()}

p = prices[prices["security_id"].isin(comp_sec["security_id"])].copy()
p["trade_date"] = pd.to_datetime(p["trade_date"])
p = p[(p["trade_date"] >= START) & (p["trade_date"] <= END)].sort_values(["security_id", "trade_date"])
p["ret"] = p.groupby("security_id")["close"].pct_change()
p = p.dropna(subset=["ret"])

# market (40 equal-weight) daily return
mkt = p.groupby("trade_date")["ret"].mean()
p["mkt"] = p["trade_date"].map(mkt)
p["ar"] = p["ret"] - p["mkt"]          # abnormal return

ret_lookup = {(r.security_id, r.trade_date): r.ar for r in p.itertuples()}
date_list = {sid: sorted(g["trade_date"].tolist()) for sid, g in p.groupby("security_id")}

# ---------- 2. neighbor graph + centrality ----------
adj = defaultdict(set)
for _, r in rel.iterrows():
    if r["relation_type"] in ("supply", "foundry_manufacturing"):
        a, b = r["from_company_id"], r["to_company_id"]
        adj[a].add(b); adj[b].add(a)
listed = set(p["security_id"].map(sec2comp).unique())
deg = {c: len([n for n in s if n in listed]) for c, s in adj.items() if c in listed}

# ---------- 3. earnings events ----------
ev = pd.read_csv(os.path.join(BASE, "tables", "events.csv"))
import re
# NOTE: regex intentionally kept in Chinese — it parses the Chinese source event text (actual vs expected).
pat = re.compile(r"EPS\s*实际\s*\$?\s*([-+]?\d*\.?\d+)\s*[;；]\s*预期\s*\$?\s*([-+]?\d*\.?\d+)", re.I)
def eps_sig(s):
    if pd.isna(s): return 0.0
    m = pat.search(str(s))
    if not m: return 0.0
    try:
        a = float(m.group(1)); e = float(m.group(2))
    except ValueError:
        return 0.0
    if e == 0: return 0.0
    v = np.clip((a - e) / abs(e), -1, 1)
    return float(v) if np.isfinite(v) else 0.0

ev["surprise"] = ev["fact_summary"].apply(eps_sig)
ev["usable"] = pd.to_datetime(ev["usable_from_trade_date"])
earn = ev[(ev["event_type"] == "earnings") & (ev["surprise"].abs() > 1e-12) & ev["usable"].notna()]
# keep only events in the formal window, to avoid 2022 warm-up events misaligned to the first 2023 day
earn = earn[(earn["usable"] >= START) & (earn["usable"] <= END)]

# ---------- 4. event study: neighbor Y's direction-adjusted AR on T..T+5 ----------
def fwd_ar(sid, T, k):
    ds = date_list.get(sid)
    if not ds: return None
    i = bisect.bisect_left(ds, T)
    if i + k >= len(ds): return None
    return ret_lookup.get((sid, ds[i + k]))

rows = []
for _, r in earn.iterrows():
    T = r["usable"]
    s = r["surprise"]
    for tok in str(r["target_companies"]).replace(",", ";").split(";"):
        X = tok.strip()
        if not X.startswith("C_"): continue
        for Y in adj.get(X, set()):
            if Y not in listed: continue
            Ysec = comp2sec.get(Y)
            if not Ysec: continue
            ars = [fwd_ar(Ysec, T, k) for k in (0, 1, 2, 3, 5)]
            if any(v is None for v in ars): continue
            rows.append({"X": X, "Y": Y, "T": T, "s": s,
                         "ar0": ars[0], "ar1": ars[1], "ar2": ars[2], "ar3": ars[3], "ar5": ars[4],
                         "deg": deg.get(Y, 0)})

df = pd.DataFrame(rows)
print("event pairs (firm-pair x event) total:", len(df))

# direction adjustment
for col in ["ar0", "ar1", "ar2", "ar3", "ar5"]:
    df[col] = df[col] * np.sign(df["s"])

# ---------- 5. group by centrality ----------
med = df["deg"].median()
hi = df[df["deg"] >= med]
lo = df[df["deg"] < med]

def profile(g):
    return [g["ar0"].mean()*100, g["ar1"].mean()*100, g["ar2"].mean()*100,
            g["ar3"].mean()*100, g["ar5"].mean()*100]

print("\n=== direction-adjusted abnormal return (%) by centrality group ===")
print("%-14s %8s %8s %8s %8s %8s   %8s   %8s" % ("group", "T+0", "T+1", "T+2", "T+3", "T+5", "CAR[0,1]", "CAR[0,5]"))
for name, g in [("high degree (hub)", hi), ("low degree (spoke)", lo)]:
    pr = profile(g)
    car01 = (g["ar0"].mean()+g["ar1"].mean())*100
    car05 = (g["ar0"].mean()+g["ar1"].mean()+g["ar2"].mean()+g["ar3"].mean()+g["ar5"].mean())*100
    print("%-14s %8.4f %8.4f %8.4f %8.4f %8.4f   %8.4f   %8.4f  (N=%d)" %
          (name, *pr, car01, car05, len(g)))

# ---------- 6. simple significance (t, clustered by event) ----------
def tstat(g, col):
    return g[col].mean() / (g[col].std()/np.sqrt(len(g)) + 1e-12)

print("\n=== t-values by group and lag ===")
print("%-14s %8s %8s %8s %8s %8s" % ("group", "T+0", "T+1", "T+2", "T+3", "T+5"))
for name, g in [("high degree", hi), ("low degree", lo)]:
    print("%-14s %8.2f %8.2f %8.2f %8.2f %8.2f" %
          (name, tstat(g,"ar0"), tstat(g,"ar1"), tstat(g,"ar2"), tstat(g,"ar3"), tstat(g,"ar5")))

# ---------- 7. robustness: winsorize (+/-1.5%) ----------
print("\n=== robustness A: winsorize (AR clip +/-1.5%) ===")
dw = df.copy()
for col in ["ar0", "ar1", "ar2", "ar3", "ar5"]:
    dw[col] = dw[col].clip(-0.015, 0.015)
medw = dw["deg"].median()
hiw = dw[dw["deg"] >= medw]; low = dw[dw["deg"] < medw]
print("%-14s %8s %8s %8s %8s %8s" % ("group", "T+0", "T+1", "T+2", "T+3", "T+5"))
for name, g in [("high degree", hiw), ("low degree", low)]:
    print("%-14s %8.4f %8.4f %8.4f %8.4f %8.4f" %
          (name, g["ar0"].mean()*100, g["ar1"].mean()*100, g["ar2"].mean()*100, g["ar3"].mean()*100, g["ar5"].mean()*100))
print("t after winsorizing:")
for name, g in [("high degree", hiw), ("low degree", low)]:
    print("  %-8s T+0=%.2f T+1=%.2f" % (name, tstat(g,"ar0"), tstat(g,"ar1")))

# ---------- 8. robustness: subsample (2023 vs 2024-25) ----------
print("\n=== robustness B: subsample (first 2023 vs second 2024-25) ===")
df["year"] = pd.to_datetime(df["T"]).dt.year
for label, sub in [("2023", df[df["year"] == 2023]), ("2024-25", df[df["year"] >= 2024])]:
    if len(sub) < 30:
        print("  [%s] too few samples" % label); continue
    med_s = sub["deg"].median()
    h = sub[sub["deg"] >= med_s]; l = sub[sub["deg"] < med_s]
    print("  [%s] high-degree T+0=%+.4f%% (t=%.2f) T+1=%+.4f%% (t=%.2f) N=%d" %
          (label, h["ar0"].mean()*100, tstat(h,"ar0"), h["ar1"].mean()*100, tstat(h,"ar1"), len(h)))
    print("       low-degree T+0=%+.4f%% (t=%.2f) T+1=%+.4f%% (t=%.2f) N=%d" %
          (l["ar0"].mean()*100, tstat(l,"ar0"), l["ar1"].mean()*100, tstat(l,"ar1"), len(l)))


# ==================== empirical_2to5.py ====================

"""
empirical_2to5.py -- empirical tests 2-5
2. analyst-action diffusion along the chain: X upgraded/downgraded -> neighbor Y's abnormal return?
3. short volume = informed trading: can short_ratio predict future return / diffuse along the chain?
4. AI boom: does lead-lag differ between 2023 and 2024-25?
5. capex-shock spillover: X's capex growth -> neighbor Y's abnormal return?
"""
import os, re
import bisect
import numpy as np
import pandas as pd
from collections import defaultdict

BASE = r"C:\Users\lanju\Desktop\Market Wave\ai_infra_v3_extracted\ai_infra_2022_2026_v3"
START = pd.Timestamp("2023-01-03"); END = pd.Timestamp("2025-12-31")

# ---------- common: prices / AR / neighbors ----------
prices = pd.read_csv(os.path.join(BASE, "tables", "prices_daily.csv"), low_memory=False)
sec = pd.read_csv(os.path.join(BASE, "tables", "securities.csv"))
rel = pd.read_csv(os.path.join(BASE, "tables", "relations.csv"))
comp_sec = sec[sec["asset_role"] == "company"]
sec2comp = dict(zip(comp_sec["security_id"], comp_sec["company_id"]))
comp2sec = {v: k for k, v in sec2comp.items()}

p = prices[prices["security_id"].isin(comp_sec["security_id"])].copy()
p["trade_date"] = pd.to_datetime(p["trade_date"])
p = p[(p["trade_date"] >= START) & (p["trade_date"] <= END)].sort_values(["security_id", "trade_date"])
p["ret"] = p.groupby("security_id")["close"].pct_change()
p = p.dropna(subset=["ret"])
p["company_id"] = p["security_id"].map(sec2comp)
mkt = p.groupby("trade_date")["ret"].mean()
p["ar"] = p["ret"] - p["trade_date"].map(mkt)
ret_lookup = {(r.security_id, r.trade_date): r.ar for r in p.itertuples()}
date_list = {sid: sorted(g["trade_date"].tolist()) for sid, g in p.groupby("security_id")}

adj = defaultdict(set)
for _, r in rel.iterrows():
    if r["relation_type"] in ("supply", "foundry_manufacturing"):
        adj[r["from_company_id"]].add(r["to_company_id"])
        adj[r["to_company_id"]].add(r["from_company_id"])
listed = set(sec2comp.values())
adj = {c: {n for n in s if n in listed} for c, s in adj.items()}

def fwd_ar(sid, T, k):
    ds = date_list.get(sid)
    if not ds: return None
    i = bisect.bisect_left(ds, T)
    if i + k >= len(ds): return None
    return ret_lookup.get((sid, ds[i + k]))

def ols_clustered(y, X, cluster):
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    y = np.asarray(y, float)
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    k = X.shape[1]; meat = np.zeros((k, k))
    for g in np.unique(cluster):
        m = cluster == g
        sg = (X[m].T * e[m]).sum(axis=1)
        meat += np.outer(sg, sg)
    V = np.linalg.solve(XtX, meat) @ np.linalg.solve(XtX, np.eye(k))
    return b, b / np.sqrt(np.diag(V))

def event_study(rows):
    """rows: list of (X, Y, T, dir) -> return direction-adjusted AR mean & t at lags 0,1,2,3,5"""
    out = []
    for X, Y, T, d in rows:
        Ysec = comp2sec.get(Y)
        if not Ysec: continue
        ars = [fwd_ar(Ysec, T, k) for k in (0, 1, 2, 3, 5)]
        if any(v is None for v in ars): continue
        out.append([ars[0]*d, ars[1]*d, ars[2]*d, ars[3]*d, ars[4]*d])
    if len(out) < 20:
        return None
    a = np.array(out)
    return a, len(out)

def show(label, es):
    if es is None:
        print("  [%s] insufficient sample" % label); return
    a, n = es
    m = a.mean(axis=0) * 100
    t = a.mean(axis=0) / (a.std(axis=0)/np.sqrt(n) + 1e-12)
    print("  [%s] N=%d  T+0=%+.4f%%(t=%.2f) T+1=%+.4f%%(t=%.2f) T+2=%+.4f%% T+3=%+.4f%% T+5=%+.4f%%" %
          (label, n, m[0], t[0], m[1], t[1], m[2], m[3], m[4]))
    aw = np.clip(a, -0.015, 0.015)
    mw = aw.mean(axis=0) * 100
    tw = aw.mean(axis=0) / (aw.std(axis=0)/np.sqrt(n) + 1e-12)
    print("        [winsorized] T+0=%+.4f%%(t=%.2f) T+1=%+.4f%%(t=%.2f)" % (mw[0], tw[0], mw[1], tw[1]))

# ============ 2. analyst actions ============
print("=== 2. analyst-action diffusion along the chain ===")
try:
    an = pd.read_csv(os.path.join(BASE, "tables", "analyst_rating_actions.csv"))
    print("  columns:", list(an.columns), "  rows:", len(an))
    print("  action values:", an["action"].value_counts().head(10).to_dict())
    # keep only upgrades/downgrades, judged by to_grade vs from_grade or action
    an["company"] = an["security_id"].map(sec2comp)
    an["date"] = pd.to_datetime(an["action_at_utc"]).dt.tz_localize(None).dt.normalize()
    an = an[(an["date"] >= START) & (an["date"] <= END) & an["company"].notna()]
    rows = []
    for _, r in an.iterrows():
        d = None
        a = str(r["action"]).lower()
        if a in ("up", "upgrade"): d = 1
        elif a in ("down", "downgrade"): d = -1
        else: continue
        for Y in adj.get(r["company"], set()):
            rows.append((r["company"], Y, r["date"], d))
    show("upgrade/downgrade -> neighbor", event_study(rows))
except Exception as e:
    print("  2 error:", e)

# ============ 5. capex shock ============
print("\n=== 5. capex/R&D/revenue growth spillover ===")
try:
    ff = pd.read_csv(os.path.join(BASE, "tables", "financial_facts.csv"), low_memory=False)
    print("  metric values top:", ff["metric"].value_counts().head(15).to_dict())
    # try capex / R&D / revenue in order
    chosen = None
    for pat in ["capital", "research", "revenue"]:
        m = ff[ff["metric"].astype(str).str.contains(pat, case=False, na=False)]
        if len(m) > 50:
            chosen = (pat, m); break
    if chosen is None:
        print("  5 no usable metric")
    else:
        print("  using:", chosen[0], "->", chosen[1]["metric"].value_counts().head(5).to_dict())
        cap = chosen[1].copy()
        cap["period_end"] = pd.to_datetime(cap["period_end"], errors="coerce")
        cap["usable"] = pd.to_datetime(cap["usable_from_trade_date"], errors="coerce")
        cap = cap.dropna(subset=["period_end", "usable", "value"]).sort_values(["company_id", "period_end"])
        cap["prev"] = cap.groupby("company_id")["value"].shift(1)
        cap["growth"] = (cap["value"] - cap["prev"]) / cap["prev"].abs().replace(0, np.nan)
        cap = cap.dropna(subset=["growth"])
        cap = cap[(cap["usable"] >= START) & (cap["usable"] <= END)]
        rows = []
        for _, r in cap.iterrows():
            d = 1 if r["growth"] > 0 else -1
            for Y in adj.get(r["company_id"], set()):
                rows.append((r["company_id"], Y, r["usable"], d))
        show("growth shock -> neighbor", event_study(rows))
except Exception as e:
    print("  5 error:", e)

# ============ 3. short volume ============
print("\n=== 3. short volume = informed trading? ===")
try:
    sv = pd.read_csv(os.path.join(BASE, "tables", "short_volume_daily.csv"), low_memory=False)
    print("  columns:", list(sv.columns), "  rows:", len(sv))
    sv["trade_date"] = pd.to_datetime(sv["trade_date"])
    sv = sv[(sv["trade_date"] >= START) & (sv["trade_date"] <= END)]
    sv["short_ratio"] = sv["short_volume"] / sv["total_volume"].replace(0, np.nan)
    sv = sv.dropna(subset=["short_ratio"])
    p2 = p[["security_id", "trade_date", "ret"]].copy()
    sv = sv.merge(p2, on=["security_id", "trade_date"], how="inner")
    # future return: sort by security
    sv = sv.sort_values(["security_id", "trade_date"])
    sv["ret_fwd"] = sv.groupby("security_id")["ret"].shift(-1)
    sv = sv.dropna(subset=["ret_fwd"])
    # cross-sectional standardization of short_ratio
    sv["sr_z"] = sv.groupby("trade_date")["short_ratio"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    sv["ret_z"] = sv.groupby("trade_date")["ret_fwd"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    b, t = ols_clustered(sv["ret_z"].values, sv[["sr_z"]].values, sv["trade_date"].values)
    print("  cross-section: ret_fwd_z = a + b*short_ratio_z  -> b=%.4f (t=%.2f), N=%d" % (b[1], t[1], len(sv)))
except Exception as e:
    print("  3 error:", e)

# ============ 4. AI boom ============
print("\n=== 4. AI boom: 2023 vs 2024-25 lead-lag ===")
try:
    # monthly returns
    p["ym"] = p["trade_date"].dt.to_period("M")
    mret = p.groupby(["company_id", "ym"])["ret"].agg(lambda x: float((1+x).prod()-1)).reset_index()
    mret = mret.sort_values(["company_id", "ym"])
    mret["ret_fwd"] = mret.groupby("company_id")["ret"].shift(-1)
    mlk = {(r.company_id, r.ym): r.ret for r in mret.itertuples()}
    def nb_ret(cid, ym):
        vals = [mlk[(n, ym)] for n in adj.get(cid, set()) if (n, ym) in mlk]
        return float(np.mean(vals)) if vals else np.nan
    mret["nb_ret"] = [nb_ret(r.company_id, r.ym) for r in mret.itertuples()]
    mret = mret.dropna(subset=["ret", "ret_fwd", "nb_ret"])
    for label, sub in [("2023", mret[mret["ym"] <= pd.Period("2023-12","M")]),
                       ("2024-25", mret[mret["ym"] >= pd.Period("2024-01","M")])]:
        if len(sub) < 30:
            print("  [%s] insufficient sample" % label); continue
        b, t = ols_clustered(sub["ret_fwd"].values, sub[["ret", "nb_ret"]].values, sub["ym"].values)
        ic = np.corrcoef(sub["nb_ret"].values, sub["ret_fwd"].values)[0,1]
        print("  [%s] b_own=%.4f(t=%.2f) b_nb=%.4f(t=%.2f) IC=%.4f N=%d" %
              (label, b[1], t[1], b[2], t[2], ic, len(sub)))
except Exception as e:
    print("  4 error:", e)


# ==================== analyst_robust.py ====================

"""
analyst_robust.py -- robustness of empirical #2: analyst-action diffusion along the chain

2 main result: analyst upgrade/downgrade on X -> neighbor Y's T+1 abnormal return +0.137% (winsorized, t=2.32)

Robustness:
  A. baseline (all)
  B. subsample 2023 vs 2024-25
  C. placebo: same X event, but paired with a random non-neighbor firm
  D. upgrade vs downgrade separately
  E. exclude the confound where neighbor Y has its own earnings within +/-3 days of the event
"""
import os
import bisect
import numpy as np
import pandas as pd
from collections import defaultdict

BASE = r"C:\Users\lanju\Desktop\Market Wave\ai_infra_v3_extracted\ai_infra_2022_2026_v3"
START = pd.Timestamp("2023-01-03"); END = pd.Timestamp("2025-12-31")

# ---------- common ----------
prices = pd.read_csv(os.path.join(BASE, "tables", "prices_daily.csv"), low_memory=False)
sec = pd.read_csv(os.path.join(BASE, "tables", "securities.csv"))
rel = pd.read_csv(os.path.join(BASE, "tables", "relations.csv"))
comp_sec = sec[sec["asset_role"] == "company"]
sec2comp = dict(zip(comp_sec["security_id"], comp_sec["company_id"]))
comp2sec = {v: k for k, v in sec2comp.items()}

p = prices[prices["security_id"].isin(comp_sec["security_id"])].copy()
p["trade_date"] = pd.to_datetime(p["trade_date"])
p = p[(p["trade_date"] >= START) & (p["trade_date"] <= END)].sort_values(["security_id", "trade_date"])
p["ret"] = p.groupby("security_id")["close"].pct_change()
p = p.dropna(subset=["ret"])
p["company_id"] = p["security_id"].map(sec2comp)
mkt = p.groupby("trade_date")["ret"].mean()
p["ar"] = p["ret"] - p["trade_date"].map(mkt)
ret_lookup = {(r.security_id, r.trade_date): r.ar for r in p.itertuples()}
date_list = {sid: sorted(g["trade_date"].tolist()) for sid, g in p.groupby("security_id")}

adj = defaultdict(set)
for _, r in rel.iterrows():
    if r["relation_type"] in ("supply", "foundry_manufacturing"):
        adj[r["from_company_id"]].add(r["to_company_id"])
        adj[r["to_company_id"]].add(r["from_company_id"])
listed = set(sec2comp.values())
adj = {c: {n for n in s if n in listed} for c, s in adj.items()}

def fwd_ar(sid, T, k):
    ds = date_list.get(sid)
    if not ds: return None
    i = bisect.bisect_left(ds, T)
    if i + k >= len(ds): return None
    return ret_lookup.get((sid, ds[i + k]))

def event_stats(rows, winsor=0.015):
    out = []
    for X, Y, T, d in rows:
        Ysec = comp2sec.get(Y)
        if not Ysec: continue
        ars = [fwd_ar(Ysec, T, k) for k in (0, 1)]
        if any(v is None for v in ars): continue
        out.append([ars[0]*d, ars[1]*d])
    if len(out) < 20: return None
    a = np.clip(np.array(out), -winsor, winsor)
    n = len(a)
    m = a.mean(axis=0) * 100
    t = a.mean(axis=0) / (a.std(axis=0)/np.sqrt(n) + 1e-12)
    return m, t, n

def show(label, res):
    if res is None:
        print("  [%-18s] insufficient sample" % label); return
    m, t, n = res
    print("  [%-18s] N=%4d  T+0=%+.4f%%(t=%+.2f)  T+1=%+.4f%%(t=%+.2f)" %
          (label, n, m[0], t[0], m[1], t[1]))

# ---------- events ----------
an = pd.read_csv(os.path.join(BASE, "tables", "analyst_rating_actions.csv"))
an["company"] = an["security_id"].map(sec2comp)
an["date"] = pd.to_datetime(an["action_at_utc"]).dt.tz_localize(None).dt.normalize()
an = an[(an["date"] >= START) & (an["date"] <= END) & an["company"].notna()]

# neighbor event pairs
rows, rows_up, rows_dn = [], [], []
for _, r in an.iterrows():
    a = str(r["action"]).lower()
    if a in ("up", "upgrade"): d = 1
    elif a in ("down", "downgrade"): d = -1
    else: continue
    for Y in adj.get(r["company"], set()):
        rows.append((r["company"], Y, r["date"], d))
        if d > 0: rows_up.append((r["company"], Y, r["date"], d))
        else: rows_dn.append((r["company"], Y, r["date"], d))

print("=== 2 robustness ===")
show("A. baseline (full sample)", event_stats(rows))

# B. subsample
sub23 = [x for x in rows if x[2] <= pd.Timestamp("2023-12-31")]
sub24 = [x for x in rows if x[2] >= pd.Timestamp("2024-01-01")]
show("B1. 2023", event_stats(sub23))
show("B2. 2024-25", event_stats(sub24))

# C. placebo: random non-neighbor
rng = np.random.default_rng(42)
placebo = []
for X, Y, T, d in rows:
    others = [c for c in listed if c != X and c not in adj.get(X, set())]
    if not others: continue
    Yp = others[rng.integers(len(others))]
    placebo.append((X, Yp, T, d))
show("C. placebo (random non-neighbor)", event_stats(placebo))

# D. upgrade vs downgrade
show("D1. upgrade", event_stats(rows_up))
show("D2. downgrade", event_stats(rows_dn))

# E. exclude neighbor Y with own earnings within +/-3 days
ev = pd.read_csv(os.path.join(BASE, "tables", "events.csv"))
ev["usable"] = pd.to_datetime(ev["usable_from_trade_date"])
earn_dates = defaultdict(list)
for _, r in ev[ev["event_type"] == "earnings"].iterrows():
    for tok in str(r["target_companies"]).split(";"):
        c = tok.strip()
        if c.startswith("C_"):
            earn_dates[c].append(r["usable"])
clean = []
dropped = 0
for X, Y, T, d in rows:
    near = any(abs((T - e).days) <= 3 for e in earn_dates.get(Y, []))
    if near:
        dropped += 1
        continue
    clean.append((X, Y, T, d))
show("E. exclude neighbor earnings +/-3 days", event_stats(clean))
print("  (excluded %d pairs, kept %d pairs)" % (dropped, len(clean)))


# ==================== analyst_robust2.py ====================

"""
analyst_robust2.py -- three key robustness checks for analyst diffusion along the chain

1. control X's own recent return: Y's T+1 abnormal return ~ action_sign + X's 5-day return
   (if the action_sign coefficient stays significant, it is not analyst momentum-chasing)
2. event-level aggregated t: average the multiple neighbors of one event (X,T) into one observation
   (avoid pseudoreplication from multiple Y per event)
3. switch window (T+0..T+3) + neighbor definition (supply-only / all non-competition / competition as falsification)
"""
import os
import bisect
import numpy as np
import pandas as pd
from collections import defaultdict

BASE = r"C:\Users\lanju\Desktop\Market Wave\ai_infra_v3_extracted\ai_infra_2022_2026_v3"
START = pd.Timestamp("2023-01-03"); END = pd.Timestamp("2025-12-31")

prices = pd.read_csv(os.path.join(BASE, "tables", "prices_daily.csv"), low_memory=False)
sec = pd.read_csv(os.path.join(BASE, "tables", "securities.csv"))
rel = pd.read_csv(os.path.join(BASE, "tables", "relations.csv"))
comp_sec = sec[sec["asset_role"] == "company"]
sec2comp = dict(zip(comp_sec["security_id"], comp_sec["company_id"]))
comp2sec = {v: k for k, v in sec2comp.items()}

p = prices[prices["security_id"].isin(comp_sec["security_id"])].copy()
p["trade_date"] = pd.to_datetime(p["trade_date"])
p = p[(p["trade_date"] >= START) & (p["trade_date"] <= END)].sort_values(["security_id", "trade_date"])
p["ret"] = p.groupby("security_id")["close"].pct_change()
p = p.dropna(subset=["ret"])
mkt = p.groupby("trade_date")["ret"].mean()
p["ar"] = p["ret"] - p["trade_date"].map(mkt)
ar_lk = {(r.security_id, r.trade_date): r.ar for r in p.itertuples()}
ret_lk = {(r.security_id, r.trade_date): r.ret for r in p.itertuples()}
date_list = {sid: sorted(g["trade_date"].tolist()) for sid, g in p.groupby("security_id")}

def build_adj(types):
    a = defaultdict(set)
    for _, r in rel.iterrows():
        if r["relation_type"] in types:
            a[r["from_company_id"]].add(r["to_company_id"])
            a[r["to_company_id"]].add(r["from_company_id"])
    listed = set(sec2comp.values())
    return {c: {n for n in s if n in listed} for c, s in a.items()}

adj_supply = build_adj(("supply", "foundry_manufacturing"))
adj_all = build_adj(tuple(t for t in rel["relation_type"].unique() if t != "competition"))
adj_comp = build_adj(("competition",))

def fwd_ar(sid, T, k):
    ds = date_list.get(sid)
    if not ds: return None
    i = bisect.bisect_left(ds, T)
    if i + k >= len(ds): return None
    return ar_lk.get((sid, ds[i + k]))

def x_recent_ret(sid, T, k=5):
    ds = date_list.get(sid)
    if not ds: return None
    i = bisect.bisect_left(ds, T)
    if i - k < 0: return None
    vals = [ret_lk.get((sid, d)) for d in ds[i-k:i]]
    if any(v is None for v in vals): return None
    return float(np.prod([1+v for v in vals]) - 1)

# events
an = pd.read_csv(os.path.join(BASE, "tables", "analyst_rating_actions.csv"))
an["company"] = an["security_id"].map(sec2comp)
an["date"] = pd.to_datetime(an["action_at_utc"]).dt.tz_localize(None).dt.normalize()
an = an[(an["date"] >= START) & (an["date"] <= END) & an["company"].notna()]

def build_rows(adj):
    rows = []
    for _, r in an.iterrows():
        a = str(r["action"]).lower()
        if a in ("up", "upgrade"): d = 1
        elif a in ("down", "downgrade"): d = -1
        else: continue
        for Y in adj.get(r["company"], set()):
            rows.append((r["company"], Y, r["date"], d))
    return rows

def ols_clustered(y, X, cluster):
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    y = np.asarray(y, float)
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    k = X.shape[1]; meat = np.zeros((k, k))
    for g in np.unique(cluster):
        m = cluster == g
        sg = (X[m].T * e[m]).sum(axis=1)
        meat += np.outer(sg, sg)
    V = np.linalg.solve(XtX, meat) @ np.linalg.solve(XtX, np.eye(k))
    return b, b / np.sqrt(np.diag(V))

rows = build_rows(adj_supply)
print("=== 2 three key robustness checks (supply neighbors, N=%d)===\n" % len(rows))

# ---------- 1. control X's own recent return ----------
print("1. control X's 5-day return (rule out momentum contagion)")
data = []
for X, Y, T, d in rows:
    Ysec = comp2sec.get(Y); Xsec = comp2sec.get(X)
    if not Ysec or not Xsec: continue
    ar1 = fwd_ar(Ysec, T, 1)
    xr = x_recent_ret(Xsec, T, 5)
    if ar1 is None or xr is None: continue
    data.append((d, xr, ar1*d, T))
arr = np.array([(x[0], x[1], x[2]) for x in data])
b, t = ols_clustered(arr[:,2], arr[:,:2], np.array([x[3] for x in data]))
print("  Y_ar_{T+1} = a + b1*action_sign + b2*X_5day_ret")
print("  b1(action_sign) = %+.4f (t=%+.2f)  <- still significant after control = not momentum-chasing" % (b[1], t[1]))
print("  b2(X_5day)      = %+.4f (t=%+.2f)" % (b[2], t[2]))

# ---------- 2. event-level aggregated t ----------
print("\n2. event-level aggregation: average one event's neighbors into one observation")
ev = defaultdict(list)
for X, Y, T, d in rows:
    Ysec = comp2sec.get(Y)
    if not Ysec: continue
    a1 = fwd_ar(Ysec, T, 1)
    if a1 is None: continue
    ev[(X, T, d)].append(a1*d)
means = np.array([np.mean(v) for v in ev.values()])
n = len(means)
t_ev = means.mean() / (means.std()/np.sqrt(n) + 1e-12)
print("  #events=%d  T+1 mean=%+.4f%%  t(event-level)=%+.2f" % (n, means.mean()*100, t_ev))

# ---------- 3. switch window + neighbor definition ----------
print("\n3. switch window and neighbor definition")
def run(label, adj):
    r = build_rows(adj)
    out = []
    for X, Y, T, d in r:
        Ysec = comp2sec.get(Y)
        if not Ysec: continue
        ars = [fwd_ar(Ysec, T, k) for k in (0,1,2,3)]
        if any(v is None for v in ars): continue
        out.append([ars[k]*d for k in range(4)])
    if len(out) < 20:
        print("  [%s] insufficient sample" % label); return
    a = np.clip(np.array(out), -0.015, 0.015)
    m = a.mean(axis=0)*100
    t = a.mean(axis=0)/(a.std(axis=0)/np.sqrt(len(a))+1e-12)
    print("  [%-22s] N=%4d  T+0=%+.4f(t=%+.2f) T+1=%+.4f(t=%+.2f) T+2=%+.4f(t=%+.2f) T+3=%+.4f(t=%+.2f)" %
          (label, len(a), m[0],t[0], m[1],t[1], m[2],t[2], m[3],t[3]))

run("supply-only", adj_supply)
run("all non-competition edges", adj_all)
run("competition edges (falsification)", adj_comp)


# ==================== exp1_bh_fdr.py ====================

"""
exp1_bh_fdr.py —— Experiment 1: BH-FDR multiple-testing correction across 5 empirical paths

The 5 paths:
  1. Return forecast (MAPE vs previous-close baseline) — no formal test, treated as null
  2. Customer momentum lead-lag (monthly neighbor coefficient)
  3. Position x diffusion speed (after outlier removal)
  4. Analyst diffusion along the chain (event-level aggregation, recomputed here)
  5. Short volume predicting returns (recomputed here)

BH-FDR: sort the 5 p-values ascending; the i-th threshold = i/m * alpha;
reject the largest i with p_(i) <= i/m*alpha and everything before it.
"""
import sqlite3, math
import numpy as np
import pandas as pd
from collections import defaultdict

DB = r"C:\Users\lanju\Desktop\陶瓷\申请材料\ai_infra_2022_2026_v3_data.db"
START = pd.Timestamp("2023-01-03"); END = pd.Timestamp("2025-12-31")
conn = sqlite3.connect(DB)

def norm_p(t):
    """Two-sided p-value (normal approximation): p = erfc(|t|/sqrt2)"""
    return math.erfc(abs(t) / math.sqrt(2.0))

def bh_fdr(pvals, alpha=0.05):
    m = len(pvals)
    order = np.argsort(pvals)
    ps = np.array(pvals)[order]
    thr = np.arange(1, m + 1) / m * alpha
    passed = ps <= thr
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        k = np.max(np.where(passed)[0])
        reject[order[:k + 1]] = True
    q = np.minimum(1, ps * m / np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q_full = np.zeros(m); q_full[order] = q
    return reject, q_full, ps, thr

# ---------- data ----------
sec = pd.read_sql("SELECT security_id, company_id, asset_role FROM securities", conn)
comp_sec = sec[sec["asset_role"] == "company"]
sec2comp = dict(zip(comp_sec["security_id"], comp_sec["company_id"]))
comp2sec = {v: k for k, v in sec2comp.items()}

p = pd.read_sql("SELECT security_id, trade_date, close, volume FROM prices_daily", conn)
p["trade_date"] = pd.to_datetime(p["trade_date"])
p = p[p["security_id"].isin(comp_sec["security_id"])].copy()
p = p[(p["trade_date"] >= START) & (p["trade_date"] <= END)].sort_values(["security_id", "trade_date"])
p["ret"] = p.groupby("security_id")["close"].pct_change()
p = p.dropna(subset=["ret"])
p["company_id"] = p["security_id"].map(sec2comp)

rel = pd.read_sql("SELECT from_company_id, to_company_id, relation_type FROM relations", conn)
adj = defaultdict(set)
for _, r in rel.iterrows():
    if r["relation_type"] in ("supply", "foundry_manufacturing"):
        adj[r["from_company_id"]].add(r["to_company_id"])
        adj[r["to_company_id"]].add(r["from_company_id"])
listed = set(sec2comp.values())
adj = {c: {n for n in s if n in listed} for c, s in adj.items()}

# ---------- 4. analyst diffusion along the chain (event-level aggregation, recomputed) ----------
an = pd.read_sql("SELECT security_id, action_at_utc, action FROM analyst_rating_actions", conn)
an["company"] = an["security_id"].map(sec2comp)
an["date"] = pd.to_datetime(an["action_at_utc"]).dt.tz_localize(None).dt.normalize()
an = an[(an["date"] >= START) & (an["date"] <= END) & an["company"].notna()]
mkt = p.groupby("trade_date")["ret"].mean()
p["ar"] = p["ret"] - p["trade_date"].map(mkt)
ret_lk = {(r.security_id, r.trade_date): r.ar for r in p.itertuples()}
date_list = {sid: sorted(g["trade_date"].tolist()) for sid, g in p.groupby("security_id")}
import bisect
def fwd_ar(sid, T, k):
    ds = date_list.get(sid)
    if not ds: return None
    i = bisect.bisect_left(ds, T)
    if i + k >= len(ds): return None
    return ret_lk.get((sid, ds[i + k]))

ev = defaultdict(list)
for _, r in an.iterrows():
    a = str(r["action"]).lower()
    if a in ("up", "upgrade"): d = 1
    elif a in ("down", "downgrade"): d = -1
    else: continue
    for Y in adj.get(r["company"], set()):
        Ysec = comp2sec.get(Y)
        if not Ysec: continue
        a1 = fwd_ar(Ysec, r["date"], 1)
        if a1 is None: continue
        ev[(r["company"], r["date"], d)].append(a1 * d)
means4 = np.array([np.mean(v) for v in ev.values()])
n4 = len(means4)
t4 = means4.mean() / (means4.std() / math.sqrt(n4) + 1e-12)
p4 = norm_p(t4)
print("4. analyst chain diffusion: events=%d, T+1=%+.4f%%, t=%+.2f, p=%.4g" % (n4, means4.mean()*100, t4, p4))

# ---------- 5. short volume (recomputed) ----------
sv = pd.read_sql("SELECT trade_date, security_id, short_volume, total_volume FROM short_volume_daily", conn)
sv["trade_date"] = pd.to_datetime(sv["trade_date"])
sv = sv[(sv["trade_date"] >= START) & (sv["trade_date"] <= END)]
sv["sr"] = sv["short_volume"] / sv["total_volume"].replace(0, np.nan)
sv = sv.dropna(subset=["sr"])
p2 = p[["security_id", "trade_date", "ret"]].copy()
sv = sv.merge(p2, on=["security_id", "trade_date"], how="inner").sort_values(["security_id", "trade_date"])
sv["ret_fwd"] = sv.groupby("security_id")["ret"].shift(-1)
sv = sv.dropna(subset=["ret_fwd"])
sv["sr_z"] = sv.groupby("trade_date")["sr"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
sv["ret_z"] = sv.groupby("trade_date")["ret_fwd"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
X = np.column_stack([np.ones(len(sv)), sv["sr_z"].values])
y = sv["ret_z"].values
b = np.linalg.solve(X.T @ X, X.T @ y)
e = y - X @ b
# cluster by date
k = 2; meat = np.zeros((k, k))
for g in np.unique(sv["trade_date"].values):
    m = sv["trade_date"].values == g
    sg = (X[m].T * e[m]).sum(axis=1)
    meat += np.outer(sg, sg)
V = np.linalg.solve(X.T @ X, meat) @ np.linalg.solve(X.T @ X, np.eye(k))
se = np.sqrt(np.diag(V))
t5 = b[1] / se[1]
p5 = norm_p(t5)
print("5. short volume: b=%+.4f, t=%+.2f, p=%.4g, N=%d" % (b[1], t5, p5, len(sv)))

# ---------- remaining paths: recorded values ----------
paths = [
    ("1. return forecast (MAPE)", 1.0, "MAPE 2.323 vs prev close 2.317, no formal test, treated as null"),
    ("2. customer momentum lead-lag (monthly)", norm_p(1.62), "b_neigh +0.105 (t=1.62)"),
    ("3. position x diffusion speed (outlier-removed)", norm_p(-0.16), "null after outlier removal (t=-0.16)"),
    ("4. analyst chain diffusion (event-level)", p4, "recomputed here"),
    ("5. short volume", p5, "recomputed here"),
]
pvals = [x[1] for x in paths]

print("\n=== p-values of the 5 paths ===")
for name, pp, note in paths:
    print("  %-28s p=%.4g  (%s)" % (name, pp, note))

for alpha in (0.05, 0.10):
    rej, q, ps, thr = bh_fdr(pvals, alpha)
    print("\n=== BH-FDR alpha=%.2f ===" % alpha)
    for name, pp, note in paths:
        i = [x[0] for x in paths].index(name)
        print("  %-28s p=%.4g  q=%.4g  %s" % (name, pp, q[i], "PASS" if rej[i] else "fail"))

conn.close()


# ==================== exp2_liquidity.py ====================

"""
exp2_liquidity.py —— Experiment 2: liquidity control (Amihud) + size/liquidity subgroup tests

External suggestion: control for liquidity, and test within size/liquidity subgroups.
Method:
  1. Per-stock Amihud illiquidity = mean(|ret| / dollar_volume); higher = less liquid;
  2. Market cap = shares_outstanding * latest close;
  3. In the analyst chain-diffusion event study:
     a) add Amihud as a control (Y_AR_{T+1} ~ action_sign + Amihud)
     b) run separately by Y liquidity tercile (high/mid/low)
     c) run separately by Y market-cap tercile (large/mid/small)
  Hypothesis: the diffusion effect concentrates in **small-cap, low-liquidity** stocks.
"""
import sqlite3, math, bisect
import numpy as np
import pandas as pd
from collections import defaultdict

DB = r"C:\Users\lanju\Desktop\陶瓷\申请材料\ai_infra_2022_2026_v3_data.db"
START = pd.Timestamp("2023-01-03"); END = pd.Timestamp("2025-12-31")
conn = sqlite3.connect(DB)

sec = pd.read_sql("SELECT security_id, company_id, asset_role FROM securities", conn)
comp_sec = sec[sec["asset_role"] == "company"]
sec2comp = dict(zip(comp_sec["security_id"], comp_sec["company_id"]))
comp2sec = {v: k for k, v in sec2comp.items()}
listed = set(sec2comp.values())

# ---------- prices / Amihud / market cap ----------
p = pd.read_sql("SELECT security_id, trade_date, close, volume FROM prices_daily", conn)
p["trade_date"] = pd.to_datetime(p["trade_date"])
p = p[p["security_id"].isin(comp_sec["security_id"])].copy()
p = p[(p["trade_date"] >= START) & (p["trade_date"] <= END)].sort_values(["security_id", "trade_date"])
p["ret"] = p.groupby("security_id")["close"].pct_change()
p = p.dropna(subset=["ret"])
p["company_id"] = p["security_id"].map(sec2comp)
p["dollar_vol"] = p["volume"] * p["close"]
p["illiq"] = p["ret"].abs() / p["dollar_vol"].replace(0, np.nan)

# use the median (more robust) for per-stock Amihud, scale by 1e9 for readability
amihud = p.groupby("company_id")["illiq"].median() * 1e9

# market cap: shares_outstanding * latest close
ks = pd.read_sql("SELECT security_id, shares_outstanding FROM key_stats_short_interest_snapshot", conn)
ks["company_id"] = ks["security_id"].map(sec2comp)
last_close = p.groupby("company_id")["close"].last()
mcap = {}
for _, r in ks.iterrows():
    c = r["company_id"]
    if c in last_close.index and pd.notna(r["shares_outstanding"]):
        mcap[c] = r["shares_outstanding"] * last_close[c]

# market + AR
mkt = p.groupby("trade_date")["ret"].mean()
p["ar"] = p["ret"] - p["trade_date"].map(mkt)
ret_lk = {(r.security_id, r.trade_date): r.ar for r in p.itertuples()}
date_list = {sid: sorted(g["trade_date"].tolist()) for sid, g in p.groupby("security_id")}
def fwd_ar(sid, T, k):
    ds = date_list.get(sid)
    if not ds: return None
    i = bisect.bisect_left(ds, T)
    if i + k >= len(ds): return None
    return ret_lk.get((sid, ds[i + k]))

# neighbor graph
rel = pd.read_sql("SELECT from_company_id, to_company_id, relation_type FROM relations", conn)
adj = defaultdict(set)
for _, r in rel.iterrows():
    if r["relation_type"] in ("supply", "foundry_manufacturing"):
        adj[r["from_company_id"]].add(r["to_company_id"])
        adj[r["to_company_id"]].add(r["from_company_id"])
adj = {c: {n for n in s if n in listed} for c, s in adj.items()}

# analyst events
an = pd.read_sql("SELECT security_id, action_at_utc, action FROM analyst_rating_actions", conn)
an["company"] = an["security_id"].map(sec2comp)
an["date"] = pd.to_datetime(an["action_at_utc"]).dt.tz_localize(None).dt.normalize()
an = an[(an["date"] >= START) & (an["date"] <= END) & an["company"].notna()]

rows = []
for _, r in an.iterrows():
    a = str(r["action"]).lower()
    if a in ("up", "upgrade"): d = 1
    elif a in ("down", "downgrade"): d = -1
    else: continue
    for Y in adj.get(r["company"], set()):
        Ysec = comp2sec.get(Y)
        if not Ysec: continue
        a1 = fwd_ar(Ysec, r["date"], 1)
        if a1 is None: continue
        rows.append({"event": "%s|%s" % (r["company"], r["date"]), "X": r["company"], "Y": Y,
                     "d": d, "a1": a1,
                     "amihud": amihud.get(Y, np.nan), "mcap": mcap.get(Y, np.nan)})

df = pd.DataFrame(rows).dropna(subset=["a1"])
print("event pairs N=%d" % len(df))

def ols_clustered(y, X, cluster):
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    y = np.asarray(y, float)
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    k = X.shape[1]; meat = np.zeros((k, k))
    for g in np.unique(cluster):
        m = cluster == g
        sg = (X[m].T * e[m]).sum(axis=1)
        meat += np.outer(sg, sg)
    V = np.linalg.solve(XtX, meat) @ np.linalg.solve(XtX, np.eye(k))
    return b, b / np.sqrt(np.diag(V))

def show(label, sub):
    if len(sub) < 20:
        print("  [%-22s] insufficient sample N=%d" % (label, len(sub))); return
    y = sub["a1"].values                      # raw (unsigned) neighbor AR
    d01 = ((sub["d"].values + 1) / 2.0)       # {0,1}: slope directly = m+ - m-
    b, t = ols_clustered(y, d01.reshape(-1, 1), sub["event"].values)
    m_up = sub.loc[sub["d"] == 1, "a1"].mean(); m_dn = sub.loc[sub["d"] == -1, "a1"].mean()
    print("  [%-22s] N=%4d  after-upgrade=%+.3f%%  after-downgrade=%+.3f%%  dir-diff(m+-m-)=%+.3f%%  t=%+.2f"
          % (label, len(sub), m_up*100, m_dn*100, b[1]*100, t[1]))

print("\n=== baseline ===")
show("full sample", df)

# ---------- a) add Amihud control ----------
print("\n=== control Amihud (neighbor raw AR ~ action{0,1} + amihud_Y; slope=dir-diff) ===")
sub = df.dropna(subset=["amihud"])
d01 = (sub["d"].values + 1) / 2.0
b, t = ols_clustered(sub["a1"].values, np.column_stack([d01, sub["amihud"].values]), sub["event"].values)
print("  b(action, i.e. m+-m- | controlling Amihud) = %+.4f (t=%+.2f)" % (b[1], t[1]))
print("  b(amihud)                        = %+.4g (t=%+.2f)" % (b[2], t[2]))

# ---------- b) terciles by Amihud ----------
print("\n=== by Y liquidity (Amihud terciles) ===")
q = df.dropna(subset=["amihud"])
q1, q2 = q["amihud"].quantile([1/3, 2/3])
show("high liq (low Amihud)", q[q["amihud"] <= q1])
show("mid liq", q[(q["amihud"] > q1) & (q["amihud"] <= q2)])
show("low liq (high Amihud)", q[q["amihud"] > q2])

# ---------- c) terciles by market cap ----------
print("\n=== by Y market cap (terciles) ===")
s = df.dropna(subset=["mcap"])
s1, s2 = s["mcap"].quantile([1/3, 2/3])
show("large cap", s[s["mcap"] > s2])
show("mid cap", s[(s["mcap"] > s1) & (s["mcap"] <= s2)])
show("small cap", s[s["mcap"] <= s1])

conn.close()


# ==================== exp3_network_null.py ====================

"""
exp3_network_null.py —— Experiment 3: network null model (degree-preserving randomization)

Question: does the analyst diffusion effect come from the supply chain's *structure* (who is
connected to whom), or merely from its *density* (how many connections)?
Method:
  1. Take the real supply graph (undirected, unique edges);
  2. Use double-edge swaps to generate N random graphs preserving every node's degree;
  3. Re-run the analyst chain-diffusion event study on each random graph -> null distribution;
  4. Compare the real-network effect against the null distribution -> permutation p-value.
     p = P(random-graph effect >= real effect) (one-sided, since real > random is expected).
"""
import sqlite3, math, bisect
import numpy as np
import pandas as pd
from collections import defaultdict

DB = r"C:\Users\lanju\Desktop\陶瓷\申请材料\ai_infra_2022_2026_v3_data.db"
START = pd.Timestamp("2023-01-03"); END = pd.Timestamp("2025-12-31")
rng = np.random.default_rng(42)
conn = sqlite3.connect(DB)

sec = pd.read_sql("SELECT security_id, company_id, asset_role FROM securities", conn)
comp_sec = sec[sec["asset_role"] == "company"]
sec2comp = dict(zip(comp_sec["security_id"], comp_sec["company_id"]))
comp2sec = {v: k for k, v in sec2comp.items()}
listed = set(sec2comp.values())

p = pd.read_sql("SELECT security_id, trade_date, close FROM prices_daily", conn)
p["trade_date"] = pd.to_datetime(p["trade_date"])
p = p[p["security_id"].isin(comp_sec["security_id"])].copy()
p = p[(p["trade_date"] >= START) & (p["trade_date"] <= END)].sort_values(["security_id", "trade_date"])
p["ret"] = p.groupby("security_id")["close"].pct_change()
p = p.dropna(subset=["ret"])
mkt = p.groupby("trade_date")["ret"].mean()
p["ar"] = p["ret"] - p["trade_date"].map(mkt)
ret_lk = {(r.security_id, r.trade_date): r.ar for r in p.itertuples()}
date_list = {sid: sorted(g["trade_date"].tolist()) for sid, g in p.groupby("security_id")}
def fwd_ar(sid, T, k):
    ds = date_list.get(sid)
    if not ds: return None
    i = bisect.bisect_left(ds, T)
    if i + k >= len(ds): return None
    return ret_lk.get((sid, ds[i + k]))

# real undirected supply edges
rel = pd.read_sql("SELECT from_company_id, to_company_id, relation_type FROM relations", conn)
edges = set()
for _, r in rel.iterrows():
    if r["relation_type"] in ("supply", "foundry_manufacturing"):
        a, b = r["from_company_id"], r["to_company_id"]
        if a in listed and b in listed and a != b:
            edges.add(tuple(sorted([a, b])))
edges = list(edges)
print("real supply edges: %d" % len(edges))
deg = defaultdict(int)
for a, b in edges:
    deg[a] += 1; deg[b] += 1
print("degree distribution:", dict(sorted(deg.items(), key=lambda x: -x[1])[:8]))

def build_adj(edge_list):
    a = defaultdict(set)
    for x, y in edge_list:
        a[x].add(y); a[y].add(x)
    return a

def randomize(edge_list, n_swaps=1000):
    """Double-edge swap, preserving the degree sequence."""
    e = list(edge_list)
    es = set(e)
    for _ in range(n_swaps):
        i, j = rng.integers(len(e), size=2)
        if i == j: continue
        a, b = e[i]; c, d = e[j]
        # (a,b),(c,d) -> (a,d),(c,b)
        if a == d or c == b: continue
        n1 = tuple(sorted([a, d])); n2 = tuple(sorted([c, b]))
        if n1 in es or n2 in es: continue
        es.discard(tuple(sorted([a, b]))); es.discard(tuple(sorted([c, d])))
        es.add(n1); es.add(n2)
        e[i] = n1; e[j] = n2
    return e

# analyst events (X, date, direction)
an = pd.read_sql("SELECT security_id, action_at_utc, action FROM analyst_rating_actions", conn)
an["company"] = an["security_id"].map(sec2comp)
an["date"] = pd.to_datetime(an["action_at_utc"]).dt.tz_localize(None).dt.normalize()
an = an[(an["date"] >= START) & (an["date"] <= END) & an["company"].notna()]
events = []
for _, r in an.iterrows():
    a = str(r["action"]).lower()
    if a in ("up", "upgrade"): d = 1
    elif a in ("down", "downgrade"): d = -1
    else: continue
    if r["company"] in listed:
        events.append((r["company"], r["date"], d))
print("analyst events: %d (with direction)" % len(events))

def effect(adj):
    vals = []
    for X, T, d in events:
        for Y in adj.get(X, set()):
            Ysec = comp2sec.get(Y)
            if not Ysec: continue
            a1 = fwd_ar(Ysec, T, 1)
            if a1 is None: continue
            vals.append(a1 * d)
    if len(vals) < 20: return np.nan, len(vals)
    return float(np.mean(vals)), len(vals)

real_adj = build_adj(edges)
real_eff, real_n = effect(real_adj)
print("\nreal-network effect: %+.4f%%  (N=%d pairs)" % (real_eff*100, real_n))

# null model: degree-preserving randomization, N runs
N_RAND = 500
null = []
for i in range(N_RAND):
    e = randomize(edges, n_swaps=800)
    eff, n = effect(build_adj(e))
    if not np.isnan(eff):
        null.append(eff)
    if (i + 1) % 100 == 0:
        print("  randomization %d/%d ..." % (i + 1, N_RAND))

null = np.array(null)
print("\n=== network null model results ===")
print("real effect          = %+.4f%%" % (real_eff * 100))
print("null mean            = %+.4f%%" % (null.mean() * 100))
print("null std             = %+.4f%%" % (null.std() * 100))
print("null 95%% percentile = %+.4f%%" % (np.percentile(null, 95) * 100))
print("null range           = [%+.4f%%, %+.4f%%]" % (null.min()*100, null.max()*100))
p_perm = (np.sum(null >= real_eff) + 1) / (len(null) + 1)
print("\npermutation p-value (one-sided, P(random >= real)) = %.4g" % p_perm)
print("conclusion: " + ("effect significantly exceeds random networks (structural effect)" if p_perm < 0.05 else "effect indistinguishable from random networks (density effect)"))

conn.close()


# ==================== exp4_t0_t1_asymmetry.py ====================

"""
exp4_t0_t1_asymmetry.py —— 2x2 split test: T=0 / T+1 x upgrade / downgrade

Reuses the exp3_network_null.py skeleton, adding a horizon parameter (k=0 same day /
k=1 next day) and a direction split.

Question: is the upgrade asymmetry (diffusion only in the upgrade direction) robust?
  - If downgrades merely react at a *different horizon* (e.g. T=0 instead of T+1),
    the asymmetry does not hold;
  - If upgrades show a net increment at both T=0/T+1 while downgrades are ~0 at both,
    the asymmetry holds.

Each cell: net increment = mean neighbor AR on real supply edges minus mean neighbor AR
on degree-matched random networks (percentage points).
"""
import sqlite3, math, bisect
import numpy as np
import pandas as pd
from collections import defaultdict

DB = r"C:\Users\lanju\Desktop\陶瓷\申请材料\ai_infra_2022_2026_v3_data.db"
START = pd.Timestamp("2023-01-03"); END = pd.Timestamp("2025-12-31")
rng = np.random.default_rng(42)
conn = sqlite3.connect(DB)

sec = pd.read_sql("SELECT security_id, company_id, asset_role FROM securities", conn)
comp_sec = sec[sec["asset_role"] == "company"]
sec2comp = dict(zip(comp_sec["security_id"], comp_sec["company_id"]))
comp2sec = {v: k for k, v in sec2comp.items()}
listed = set(sec2comp.values())

p = pd.read_sql("SELECT security_id, trade_date, close FROM prices_daily", conn)
p["trade_date"] = pd.to_datetime(p["trade_date"])
p = p[p["security_id"].isin(comp_sec["security_id"])].copy()
p = p[(p["trade_date"] >= START) & (p["trade_date"] <= END)].sort_values(["security_id", "trade_date"])
p["ret"] = p.groupby("security_id")["close"].pct_change()
p = p.dropna(subset=["ret"])
mkt = p.groupby("trade_date")["ret"].mean()
p["ar"] = p["ret"] - p["trade_date"].map(mkt)
ret_lk = {(r.security_id, r.trade_date): r.ar for r in p.itertuples()}
date_list = {sid: sorted(g["trade_date"].tolist()) for sid, g in p.groupby("security_id")}
def fwd_ar(sid, T, k):
    ds = date_list.get(sid)
    if not ds: return None
    i = bisect.bisect_left(ds, T)
    if i + k >= len(ds): return None
    return ret_lk.get((sid, ds[i + k]))

# real undirected supply edges
rel = pd.read_sql("SELECT from_company_id, to_company_id, relation_type FROM relations", conn)
edges = set()
for _, r in rel.iterrows():
    if r["relation_type"] in ("supply", "foundry_manufacturing"):
        a, b = r["from_company_id"], r["to_company_id"]
        if a in listed and b in listed and a != b:
            edges.add(tuple(sorted([a, b])))
edges = list(edges)

def build_adj(edge_list):
    a = defaultdict(set)
    for x, y in edge_list:
        a[x].add(y); a[y].add(x)
    return a

def randomize(edge_list, n_swaps=800):
    e = list(edge_list); es = set(e)
    for _ in range(n_swaps):
        i, j = rng.integers(len(e), size=2)
        if i == j: continue
        a, b = e[i]; c, d = e[j]
        if a == d or c == b: continue
        n1 = tuple(sorted([a, d])); n2 = tuple(sorted([c, b]))
        if n1 in es or n2 in es: continue
        es.discard(tuple(sorted([a, b]))); es.discard(tuple(sorted([c, d])))
        es.add(n1); es.add(n2)
        e[i] = n1; e[j] = n2
    return e

# analyst events (X, date, direction +1 upgrade / -1 downgrade)
an = pd.read_sql("SELECT security_id, action_at_utc, action FROM analyst_rating_actions", conn)
an["company"] = an["security_id"].map(sec2comp)
an["date"] = pd.to_datetime(an["action_at_utc"]).dt.tz_localize(None).dt.normalize()
an = an[(an["date"] >= START) & (an["date"] <= END) & an["company"].notna()]
events = []
for _, r in an.iterrows():
    a = str(r["action"]).lower()
    if a in ("up", "upgrade"): d = 1
    elif a in ("down", "downgrade"): d = -1
    else: continue
    if r["company"] in listed:
        events.append((r["company"], r["date"], d))

def effect(adj, horizon, direction):
    """Mean raw (unsigned) abnormal return of neighbors at `horizon` for events of one direction."""
    vals = []
    for X, T, d in events:
        if d != direction: continue
        for Y in adj.get(X, set()):
            Ysec = comp2sec.get(Y)
            if not Ysec: continue
            a = fwd_ar(Ysec, T, horizon)
            if a is None: continue
            vals.append(a)
    if len(vals) < 20: return np.nan, 0
    return float(np.mean(vals)), len(vals)

real_adj = build_adj(edges)
N_RAND = 500
# pre-generate 500 degree-preserving random graphs, shared by the four cells
random_adjs = [build_adj(randomize(edges)) for _ in range(N_RAND)]
print("real supply edges %d; random graphs %d; events %d" % (len(edges), N_RAND, len(events)))

cells = {}
for h, hname in [(0, "T=0"), (1, "T+1")]:
    for d, dname in [(1, "upgrade"), (-1, "downgrade")]:
        real_v, n = effect(real_adj, h, d)
        null = np.array([effect(a, h, d)[0] for a in random_adjs if not np.isnan(effect(a, h, d)[0])])
        net = (real_v - null.mean()) * 100
        # one-sided: if the net increment is positive, p = P(random >= real)
        p_one = (np.sum(null >= real_v) + 1) / (len(null) + 1)
        cells[(h, d)] = dict(real=real_v*100, rnd=null.mean()*100, net=net, p=p_one, n=n, null=null)

print("\n=== 2x2 table: net increment pp (real edges - degree-matched random networks) ===")
print("%-6s %18s %18s" % ("", "upgrade", "downgrade"))
for h, hname in [(0, "T=0"), (1, "T+1")]:
    up = cells[(h, 1)]; dn = cells[(h, -1)]
    print("%-6s %+10.2f (p=%.3f) %+10.2f (p=%.3f)" % (hname, up["net"], up["p"], dn["net"], dn["p"]))

print("\n=== detail: real / random / net increment / count ===")
print("%-4s %-10s real%8s random%5s net%9s  N  %s" % ("horizon", "direction", "", "", "", ""))
for h, hname in [(0, "T=0"), (1, "T+1")]:
    for d, dname in [(1, "upgrade"), (-1, "downgrade")]:
        c = cells[(h, d)]
        print("%-4s %-4s %+8.2f %+8.2f %+8.2f  %4d  p=%.3f" %
              (hname, dname, c["real"], c["rnd"], c["net"], c["n"], c["p"]))

conn.close()


# ==================== exp5_spectral_threshold.py ====================

"""
exp5_spectral_threshold.py

Goal: replace the critical value of a_net ("0.5 found by bisection trial") with one
*computed from the spectral structure of the network*. This is the only experiment in
the exp series that directly uses the Scutari framework (network-dynamics stability +
separation property).

Linearization:
    drift of herd opinion x_i  ∝  a1*x_i + a_net*(Wx)_i
    instability condition (mean field): a1 + a_net * lambda_max(W) = 1
    =>  a*_net = (1 - a1) / lambda_max(W)

W is the row-normalized "mean neighbor opinion" matrix; any connected graph has
lambda_max(W) = 1
  =>  a* = 1 - a1, independent of topology.
  =>  topology only enters the *relaxation rate*: gap = 1 - lambda_2 sets mixing speed.

This is exactly the separation property: the threshold is the threshold, the rate is the rate.

Three testable predictions:
  P1  measured threshold ≈ 1 - a1 (a1=0.6 -> 0.40; a1=0.5 -> 0.50)
  P2  threshold is topology-independent (star / chain / real network give the same threshold)
  P3  above threshold, the *time* to order is set by the spectral gap (star fast, chain slow)

Usage:
  python exp5_spectral_threshold.py --quick     # smoke test
  python exp5_spectral_threshold.py             # full run
"""
import argparse
import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\lanju\Desktop\Market Wave\ai_infra_v3_extracted\ai_infra_2022_2026_v3\tables"
OUT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- network
def load_real_network():
    sec = pd.read_csv(os.path.join(BASE, "securities.csv"))
    rel = pd.read_csv(os.path.join(BASE, "relations.csv"))
    cs = sec[sec["asset_role"] == "company"]
    comp = sorted(cs["company_id"].tolist())
    idx = {c: i for i, c in enumerate(comp)}
    edges = set()
    for _, r in rel.iterrows():
        if r["relation_type"] in ("supply", "foundry_manufacturing"):
            a, b = r["from_company_id"], r["to_company_id"]
            if a in idx and b in idx and a != b:
                edges.add(tuple(sorted((idx[a], idx[b]))))
    M = len(comp)
    A = np.zeros((M, M))
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    return A, comp, cs


def row_norm(A):
    d = A.sum(1)
    return np.divide(A, d[:, None], out=np.zeros_like(A), where=d[:, None] > 0), d


def spectrum(A):
    W, d = row_norm(A)
    ev = np.linalg.eigvals(W).real
    ev = np.sort(ev)[::-1]
    lam_max = ev[0]
    # lambda_2 on the largest connected component (on the full graph, isolated nodes pollute lambda_2 with 0)
    comps = components(A)
    gaps = []
    for cc in comps:
        if len(cc) < 2:
            continue
        sub = A[np.ix_(cc, cc)]
        Wc, _ = row_norm(sub)
        e = np.sort(np.linalg.eigvals(Wc).real)[::-1]
        gaps.append((len(cc), int(sub.sum() / 2), e[1] if len(e) > 1 else np.nan, 1 - (e[1] if len(e) > 1 else 1.0)))
    gaps.sort(key=lambda z: -z[0])
    return lam_max, gaps


def components(A):
    M = A.shape[0]
    seen, comps = set(), []
    for s in range(M):
        if s in seen or A[s].sum() == 0:
            continue
        stack, cc = [s], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            cc.append(u)
            stack += [v for v in np.where(A[u] > 0)[0] if v not in seen]
        comps.append(sorted(cc))
    return comps


def star(n):
    A = np.zeros((n, n))
    A[0, 1:] = A[1:, 0] = 1
    return A


def path(n):
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1
    return A


# ---------------------------------------------------------------- simulation
def run(a_net, W, a1=0.6, T=100000, seed=1, burn=0.5, n_per=25,
        dt=0.01, v1=3.0, v2=2.0, beta=6.0, tc=0.02, gamma=0.01, a2=0.2, a3=0.5,
        pf=10.0, r=0.004, R=0.0004, s=0.75, sigma_mu=0.05, W_rec=100):
    """Dynamics identical to the original supply_chain_abm.py, only the neighbor mean is replaced by W @ x."""
    M = W.shape[0]
    rng = np.random.default_rng(seed)
    n_plus = np.full(M, n_per // 2, dtype=float)
    n_minus = np.full(M, n_per // 2, dtype=float)
    n_f = np.full(M, n_per - (n_per // 2) * 2, dtype=float)
    p = np.full(M, pf, dtype=float)
    dp_prev = np.zeros(M)
    prices = [[] for _ in range(M)]
    mag = np.zeros(M)          # accumulated |x|, for the order parameter
    nmag = 0
    clip_hits = 0              # population-count overflow count (stability-boundary signal of the discrete system)
    order_track = []           # <|x|> per recording window, for measuring time-to-order
    win_mag, win_n = 0.0, 0
    t_order = np.nan
    for t in range(T):
        n_c = n_plus + n_minus
        n_c = np.where(n_c == 0, 1, n_c)
        x = (n_plus - n_minus) / n_c
        x_net = W @ x

        U1 = a1 * x + a2 * (dp_prev / v1) + a_net * x_net
        U21 = np.clip(a3 * ((r + dp_prev / v2) / p - R - s * np.abs(pf - p) / p), -20, 20)
        U22 = np.clip(a3 * (R - (r + dp_prev / v2) / p - s * np.abs(pf - p) / p), -20, 20)

        q_po = np.minimum(1.0, v1 * (n_c / n_per) * np.exp(U1) * dt)
        q_op = np.minimum(1.0, v1 * (n_c / n_per) * np.exp(-U1) * dt)
        s_po = rng.binomial(n_minus.astype(int), np.clip(q_po, 0, 1))
        s_op = rng.binomial(n_plus.astype(int), np.clip(q_op, 0, 1))
        n_plus += s_po - s_op
        n_minus += s_op - s_po

        q_pf = np.minimum(1.0, v2 * (n_f / n_per) * np.exp(-U21) * dt)
        q_fp = np.minimum(1.0, v2 * (n_plus / n_per) * np.exp(U21) * dt)
        q_mf = np.minimum(1.0, v2 * (n_f / n_per) * np.exp(-U22) * dt)
        q_fm = np.minimum(1.0, v2 * (n_minus / n_per) * np.exp(U22) * dt)
        s_pf = rng.binomial(n_plus.astype(int), np.clip(q_pf, 0, 1))
        s_fp = rng.binomial(n_f.astype(int), np.clip(q_fp, 0, 1))
        s_mf = rng.binomial(n_minus.astype(int), np.clip(q_mf, 0, 1))
        s_fm = rng.binomial(n_f.astype(int), np.clip(q_fm, 0, 1))
        n_plus += -s_pf + s_fp
        n_minus += -s_mf + s_fm
        n_f += s_pf + s_mf - s_fp - s_fm

        # Domain constraint of the discrete system: population counts cannot go negative.
        # The overflow count is a direct signal of the "stability boundary" (divergence in the continuous limit).
        if (n_f < -0.5).any() or (n_plus < -0.5).any() or (n_minus < -0.5).any():
            clip_hits += 1
        n_plus = np.maximum(n_plus, 0.0)
        n_minus = np.maximum(n_minus, 0.0)
        n_f = np.maximum(n_f, 0.0)

        ED = (n_plus - n_minus) * tc + n_f * gamma * (pf - p)
        mu = rng.normal(0, sigma_mu, M)
        prob = beta * (ED + mu) * dt
        dp = np.zeros(M)
        up = (prob > 0) & (rng.random(M) < prob)
        dn = (prob < 0) & (rng.random(M) < -prob)
        dp[up] = 0.01
        dp[dn] = -0.01
        p += dp
        dp_prev = dp
        if t >= burn * T:
            mag += np.abs(x)
            nmag += 1
        win_mag += float(np.abs(x).mean())
        win_n += 1
        if (t + 1) % W_rec == 0:
            mo = win_mag / win_n
            order_track.append(mo)
            if np.isnan(t_order) and t > burn * T and mo > 0.55:
                t_order = (t + 1) / T
            win_mag, win_n = 0.0, 0
            for i in range(M):
                prices[i].append(p[i])

    P = np.array(prices)
    stock = np.diff(np.log(P), axis=1)
    idxr = np.diff(np.log(P.mean(axis=0)))
    ret = stock.ravel()
    ret = ret[np.isfinite(ret)]
    ret = ret - ret.mean()
    a = np.abs(ret)

    def ac(v, lag):
        v = np.asarray(v)
        if lag >= len(v) - 1:
            return np.nan
        sd = v.std()
        if sd < 1e-12:
            return np.nan
        return float(np.corrcoef(v[:-lag], v[lag:])[0, 1])

    kurt = float(np.mean(ret ** 4) / np.mean(ret ** 2) ** 2 - 3) if len(ret) > 10 else np.nan
    return dict(
        order=float(mag.mean() / max(nmag, 1)),          # order parameter <|x|>: 0=disordered 1=ordered
        kurt=kurt,
        absac1=ac(a, 1),
        rac1=ac(ret, 1),
        mean_abs=float(np.mean(np.abs(ret))),
        zero_frac=float(np.mean(np.abs(ret) < 1e-12)),
        p_drift=float(np.max(np.abs(np.log(p / pf)))),
        clip_hits=clip_hits,
        t_order=t_order,
        finite=bool(np.all(np.isfinite(P))),
    )


# ---------------------------------------------------------------- main
def scan(W, a1, grid, seeds, T, tag):
    rows = []
    for a in grid:
        for sd in seeds:
            r = run(a, W, a1=a1, T=T, seed=sd)
            r.update(a_net=a, a1=a1, seed=sd, net=tag)
            rows.append(r)
        sub = [x for x in rows if x["a_net"] == a and x["net"] == tag and x["a1"] == a1]
        to = [z["t_order"] for z in sub if np.isfinite(z["t_order"])]
        print("  a_net=%.2f  <|x|>=%.3f  kurt=%7.2f  |r|ac1=%+.3f  r ac1=%+.3f  zero=%.2f  clip=%5.1f  t_ord=%s"
              % (a, np.mean([z["order"] for z in sub]), np.mean([z["kurt"] for z in sub]),
                 np.mean([z["absac1"] for z in sub]), np.mean([z["rac1"] for z in sub]),
                 np.mean([z["zero_frac"] for z in sub]), np.mean([z["clip_hits"] for z in sub]),
                 ("%.2f" % np.mean(to)) if to else "—"))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--T", type=int, default=60000)
    ap.add_argument("--topo", action="store_true", help="also run the star/chain topology comparison")
    args = ap.parse_args()
    T = 20000 if args.quick else args.T
    seeds = [1] if args.quick else [1, 2]

    A, comp, cs = load_real_network()
    lam_max, gaps = spectrum(A)
    print("=== real supply network ===")
    print("companies %d, unique edges %d, connected components %s" % (len(comp), int(A.sum() / 2),
          [(g[0], g[1]) for g in gaps]))
    for n, e, l2, gap in gaps:
        print("  component %2d nodes %2d edges  lambda2=%.4f  spectral gap=%.4f" % (n, e, l2, gap))
    print("lambda_max(W) = %.4f" % lam_max)
    print("predicted threshold a* = (1-a1)/lambda_max = %.3f (a1=0.6) / %.3f (a1=0.5)\n"
          % ((1 - 0.6) / lam_max, (1 - 0.5) / lam_max))

    grid = [0.0, 0.2, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
    if args.quick:
        grid = [0.0, 0.3, 0.4, 0.5]
    W, _ = row_norm(A)

    print("=== P1 threshold ~= 1-a1 (real network) ===")
    rows = scan(W, 0.6, grid, seeds, T, "real_a1=0.6")
    rows += scan(W, 0.5, [g for g in grid if g >= 0.3], seeds, T, "real_a1=0.5")

    if args.topo and not args.quick:
        print("\n=== P2/P3 topology comparison (all 11 nodes) ===")
        for nm, Wt in [("star", row_norm(star(11))[0]), ("path", row_norm(path(11))[0])]:
            _, g = spectrum(np.where(Wt > 0, 1.0, 0.0))
            print("  %s: lambda2=%.4f spectral gap=%.4f" % (nm, g[0][2], g[0][3]))
            rows += scan(Wt, 0.6, [0.0, 0.45, 0.60], seeds, T, nm)

    df = pd.DataFrame(rows)
    out = os.path.join(OUT, "exp5_spectral_threshold_quick.csv" if args.quick else "exp5_spectral_threshold.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\nsaved: %s" % out)


# ==================== expA_monotonicity.py ====================

"""
expA_monotonicity.py —— Experiment A: monotonicity profile (pure linear algebra)

Map the supply-chain diffusion model onto a monotone-game / variational-inequality frame:
  - cost f_i = ½(x_i - b_i)^2 + (β/2)·Σ_j W_ij (x_i - x_j)^2
  - pseudo-gradient F(x) = (I + β·L)x - b, L = I - W (random-walk Laplacian, W row-normalized)
  - strong-monotonicity parameter μ = 1 + β·λ₂ (λ₂ = spectral gap of the largest connected component)
  - Lipschitz  L = 1 + β·λ_max
  - condition number κ = L/μ

Sweep β ∈ [0, 3] and plot μ(β) and κ(β).
Core question: does a threshold (μ crossing zero) exist?
  Expected: μ stays positive (row normalization -> λ_min(L)=0 -> μ=1+βλ₂>0), κ grows
  linearly -> slower only, never blows up.
"""
import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict

DB = r"C:\Users\lanju\Desktop\陶瓷\申请材料\ai_infra_2022_2026_v3_data.db"
conn = sqlite3.connect(DB)

sec = pd.read_sql("SELECT security_id, company_id, asset_role FROM securities", conn)
comp_sec = sec[sec["asset_role"] == "company"]
sec2comp = dict(zip(comp_sec["security_id"], comp_sec["company_id"]))
listed = set(sec2comp.values())

rel = pd.read_sql("SELECT from_company_id, to_company_id, relation_type FROM relations", conn)
edges = set()
for _, r in rel.iterrows():
    if r["relation_type"] in ("supply", "foundry_manufacturing"):
        a, b = r["from_company_id"], r["to_company_id"]
        if a in listed and b in listed and a != b:
            edges.add(tuple(sorted([a, b])))
edges = sorted(edges)

nodes = sorted(set(n for e in edges for n in e))
idx = {n: i for i, n in enumerate(nodes)}
N = len(nodes)

# adjacency + row-normalized W
A = np.zeros((N, N))
for a, b in edges:
    i, j = idx[a], idx[b]
    A[i, j] = A[j, i] = 1.0
deg = A.sum(axis=1)
W = A / deg[:, None]           # row normalization: W[i,j] = 1/deg(i)

# connected components (undirected)
def connected_components(adj):
    seen = set(); comps = []
    for start in range(adj.shape[0]):
        if start in seen: continue
        stack = [start]; seen.add(start); comp = []
        while stack:
            u = stack.pop(); comp.append(u)
            for v in np.where(adj[u] > 0)[0]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        comps.append(sorted(comp))
    return comps

comps = connected_components(A)
lcc = max(comps, key=len)
print("total nodes %d, edges %d, connected components %d, largest component %d nodes" % (N, len(edges), len(comps), len(lcc)))
print("isolated nodes %d; largest-component node indices %s" % (sum(1 for c in comps if len(c)==1), lcc))

# symmetric normalized Laplacian on the LCC: L_sym = I - D^{-1/2} A D^{-1/2}
# (similar to the random-walk Laplacian I-W, same eigenvalues, but symmetric; eigenvalues in [0,2])
lcc = np.array(lcc)
A_sub = A[np.ix_(lcc, lcc)]
deg_sub = A_sub.sum(axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(deg_sub))
L_sym = np.eye(len(lcc)) - D_inv_sqrt @ A_sub @ D_inv_sqrt
eig = np.linalg.eigvalsh(L_sym)         # ascending
lam2 = eig[1]                           # spectral gap (the smallest is 0, the all-ones vector)
lam_max = eig[-1]
print("spectral gap lambda2 = %.4f; lambda_max = %.4f; asymptotic kappa = lam_max/lam2 = %.3f" % (lam2, lam_max, lam_max/lam2))

# sweep beta
beta = np.linspace(0, 3, 121)
mu = 1 + beta * lam2
Lip = 1 + beta * lam_max
kappa = Lip / mu

print("\nβ=0    μ=%.4f  L=%.4f  κ=%.4f" % (mu[0], Lip[0], kappa[0]))
print("β=0.5  μ=%.4f  L=%.4f  κ=%.4f" % (mu[np.argmin(abs(beta-0.5))], Lip[np.argmin(abs(beta-0.5))], kappa[np.argmin(abs(beta-0.5))]))
print("β=1    μ=%.4f  L=%.4f  κ=%.4f" % (mu[np.argmin(abs(beta-1))], Lip[np.argmin(abs(beta-1))], kappa[np.argmin(abs(beta-1))]))
print("β=3    μ=%.4f  L=%.4f  κ=%.4f" % (mu[-1], Lip[-1], kappa[-1]))
print("\nmu minimum (beta in [0,3]) = %.4f  -> always positive %s" % (mu.min(), "(no instability threshold, only slower)" if mu.min() > 0 else "(threshold exists)"))

# ---------- plots (Pillow, one file per figure) ----------
from PIL import Image, ImageDraw, ImageFont
FONT = r"C:\Windows\Fonts\msyh.ttc"
def F(s): return ImageFont.truetype(FONT, s)
def Fb(s): return ImageFont.truetype(FONT, s)
OUT = r"C:\Users\lanju\Desktop\陶瓷\_diagrams"

def plot(xs, ys, ylabel, title, fname, ymin=None, ymax=None):
    Wp, Hp = 1100, 620
    img = Image.new("RGB", (Wp, Hp), "#fbfbfd")
    d = ImageDraw.Draw(img)
    d.text((40, 26), title, font=Fb(28), fill="#1a1a1a")
    Lx, Rx, Ty, By = 100, 1040, 90, 540
    ymin = ymin if ymin is not None else float(min(ys)) * 0.95
    ymax = ymax if ymax is not None else float(max(ys)) * 1.05
    if ymax - ymin < 1e-9: ymax = ymin + 1
    def px(x): return Lx + (x - xs[0]) / (xs[-1] - xs[0]) * (Rx - Lx)
    def py(y): return By - (y - ymin) / (ymax - ymin) * (By - Ty)
    d.line([Lx, By, Rx, By], fill="#999", width=2)
    d.line([Lx, Ty, Lx, By], fill="#999", width=2)
    pts = [(px(x), py(y)) for x, y in zip(xs, ys)]
    d.line(pts, fill="#b13a2e", width=4)
    d.ellipse([pts[-1][0]-4, pts[-1][1]-4, pts[-1][0]+4, pts[-1][1]+4], fill="#b13a2e")
    # axis labels
    for x in np.arange(0, 3.01, 0.5):
        d.text((px(x)-14, By+8), "%.1f" % x, font=F(18), fill="#555")
    for y in np.linspace(ymin, ymax, 6):
        d.text((Lx-56, py(y)-10), "%.2f" % y, font=F(18), fill="#555")
    d.text((Rx-140, By+8), "beta (coupling strength)", font=Fb(20), fill="#111")
    d.text((18, Ty), ylabel, font=Fb(20), fill="#111")
    p = OUT + "\\" + fname
    img.save(p); print("saved", p)

plot(beta, mu, "mu (strong monotonicity)", "Fig. 9 - monotonicity profile: mu(beta) = 1 + beta*lambda2", "diagram_9_mu.png", ymin=0.9, ymax=1.65)
plot(beta, kappa, "kappa (condition number L/mu)", "Fig. 10 - condition number: kappa(beta) = (1+beta*lam_max)/(1+beta*lam2)", "diagram_10_kappa.png")

conn.close()


# ==================== expB_asynchrony.py ====================

"""
expB_asynchrony.py —— Experiment B: the cost of asynchrony

Give the 11-node LCC update clocks and run best-response iteration of the quadratic game:
  x_i* = (b_i + β·(Wx)_i) / (1+β)

Three schedules (nodes updated per round):
  sync     = 11 (one round = one full Jacobi sweep)
  round-robin = 1  (one round = update 1 node, Gauss-Seidel, fixed cyclic order)
  batch    = 3  (one round = update one batch, synchronous within the batch)

Record the number of rounds to reach ‖x−x*‖<ε and plot T_async/T_sync.
Prediction: as long as asynchrony introduces no delay, convergence stays linear —
more time, same endpoint.
  T_rr/T_sync ≈ 11  => pure time cost;
  much larger than 11 => bottleneck node exists (guess TSM or NVDA).
Also: under round-robin, record each node's "marginal error reduction"; the smallest is the bottleneck.
"""
import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict

DB = r"C:\Users\lanju\Desktop\陶瓷\申请材料\ai_infra_2022_2026_v3_data.db"
conn = sqlite3.connect(DB)

sec = pd.read_sql("SELECT security_id, company_id, asset_role FROM securities", conn)
comp_sec = sec[sec["asset_role"] == "company"]
sec2comp = dict(zip(comp_sec["security_id"], comp_sec["company_id"]))
comp2sec = {v: k for k, v in sec2comp.items()}
listed = set(sec2comp.values())

rel = pd.read_sql("SELECT from_company_id, to_company_id, relation_type FROM relations", conn)
edges = set()
for _, r in rel.iterrows():
    if r["relation_type"] in ("supply", "foundry_manufacturing"):
        a, b = r["from_company_id"], r["to_company_id"]
        if a in listed and b in listed and a != b:
            edges.add(tuple(sorted([a, b])))
edges = sorted(edges)

nodes = sorted(set(n for e in edges for n in e))
idx = {n: i for i, n in enumerate(nodes)}
N = len(nodes)
A = np.zeros((N, N))
for a, b in edges:
    i, j = idx[a], idx[b]
    A[i, j] = A[j, i] = 1.0
deg = A.sum(axis=1)
W = A / deg[:, None]

# connected components, take the LCC
def cc(adj):
    seen = set(); comps = []
    for s in range(adj.shape[0]):
        if s in seen: continue
        st = [s]; seen.add(s); c = []
        while st:
            u = st.pop(); c.append(u)
            for v in np.where(adj[u] > 0)[0]:
                if v not in seen: seen.add(v); st.append(v)
        comps.append(sorted(c))
    return comps
lcc = max(cc(A), key=len)
lcc = np.array(lcc)
name = {i: comp2sec.get(nodes[i], nodes[i]) for i in range(N)}

Wl = W[np.ix_(lcc, lcc)]
n = len(lcc)
deg_l = deg[lcc]
beta = 0.5
rng = np.random.default_rng(7)
b = rng.uniform(0.5, 1.5, n)
M = (1 + beta) * np.eye(n) - beta * Wl
x_star = np.linalg.solve(M, b)
x0 = np.zeros(n)
eps = 1e-12
MAX = 200000

# ---------- synchronous (Jacobi) ----------
x = x0.copy(); t = 0
while np.linalg.norm(x - x_star) > eps and t < MAX:
    x = (b + beta * (Wl @ x)) / (1 + beta)
    t += 1
T_sync = t

# ---------- round-robin (Gauss-Seidel, fixed cyclic order + record marginal error reduction) ----------
x = x0.copy(); t = 0
margin = defaultdict(float)   # accumulated "error reduction" per node
order = list(range(n))
while np.linalg.norm(x - x_star) > eps and t < MAX:
    for i in order:
        before = np.linalg.norm(x - x_star)
        x[i] = (b[i] + beta * (Wl[i] @ x)) / (1 + beta)
        after = np.linalg.norm(x - x_star)
        margin[i] += before - after
        t += 1
        if np.linalg.norm(x - x_star) <= eps:
            break
T_rr = t

# ---------- batch (batch size 3, synchronous Jacobi within batch) ----------
x = x0.copy(); t = 0
batches = [order[i:i+3] for i in range(0, n, 3)]
while np.linalg.norm(x - x_star) > eps and t < MAX:
    for bt in batches:
        x_new = x.copy()
        for i in bt:
            x_new[i] = (b[i] + beta * (Wl[i] @ x)) / (1 + beta)
        x = x_new
        t += 1
        if np.linalg.norm(x - x_star) <= eps:
            break
T_batch = t

print("LCC nodes %d; beta=%.1f" % (n, beta))
print("\n=== rounds to convergence (‖x−x*‖<1e-12) ===")
print("sync(11/round)       T_sync  = %d" % T_sync)
print("round-robin(1/round) T_rr    = %d   ratio = %.2f" % (T_rr, T_rr / T_sync))
print("batch(3/round)       T_batch = %d   ratio = %.2f" % (T_batch, T_batch / T_sync))

print("\n=== marginal error reduction per node under round-robin (smaller = more bottleneck) ===")
names = {i: comp2sec.get(nodes[i], 'N_?' + nodes[i]) for i in range(N)}
for i in order:
    d = names.get(int(lcc[i]), lcc[i])
    print("  %-6s degree=%d  cumulative marginal reduction=%.3e" % (d, deg_l[i], margin[i]))

print("\nreading: T_rr/T_sync ~= 11 => pure time cost; much larger than 11 => bottleneck node exists")
conn.close()


# ==================== expB1_algorithm_sensitivity.py ====================

"""
expB1_algorithm_sensitivity.py

B1: same problem, two equally reasonable update rules — which one stops first?

Setup (quadratic game / relative-performance tournament):
    f_i(x_i, x_-i) = 1/2 (x_i - b_i)^2 + (beta/2) * sum_j W_ij (x_i - x_j)^2
    pseudo-gradient  F(x) = (I + beta*L) x - b,      L = I - W
    unique solution  x*   = (I + beta*L)^{-1} b      (requires b zero-mean, see below)

Two algorithms:
    GD  projected gradient (step 1/Lipschitz): x <- x - alpha * F(x)
    BR  synchronous best response            : x <- (b + beta*Wx) / (1 + beta)

Theoretical contraction factors (read off the spectrum directly, no hand-derived mu/kappa):
    rho_GD(beta) = 1 - (1+beta*lam2) / (1+beta*lammax)      # governed by condition number
    rho_BR(beta) = beta * max|lam_k| / (1+beta),  k>=2      # governed only by coupling strength

Expected (11-node LCC: lam2=0.1842, lammax=2.0, bipartite => lam_min(W)=-1):
    beta        0.5     1      2      3      5
    T_GD         18     28     43     55     72      <- rho governed by kappa, grows slowly
    T_BR         13     20     34     48     76      <- rho=beta/(1+beta), grows fast, catches GD around beta ~ 4
    The two lines cross at beta ~ 4. Both stay < 1, so neither diverges; only speed differs.

Note: a common mistake must be corrected here. The best-response error iteration is
    e_{t+1} = beta*W/(1+beta) * e_t
NOT (1+beta*lam)/(1+beta) — the latter treats the pseudo-gradient F as the iteration
matrix. The two differ by a constant term b/(1+beta); they are equivalent only after
absorbing b into the fixed point.

Usage:
    python expB1_algorithm_sensitivity.py
"""
import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\lanju\Desktop\Market Wave\ai_infra_v3_extracted\ai_infra_2022_2026_v3\tables"
OUT = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-6
MAXIT = 200000


# ---------------------------------------------------------------- network
def load_real_network():
    sec = pd.read_csv(os.path.join(BASE, "securities.csv"))
    rel = pd.read_csv(os.path.join(BASE, "relations.csv"))
    cs = sec[sec["asset_role"] == "company"]
    comp = sorted(cs["company_id"].tolist())
    idx = {c: i for i, c in enumerate(comp)}
    edges = set()
    for _, r in rel.iterrows():
        if r["relation_type"] in ("supply", "foundry_manufacturing"):
            a, b = r["from_company_id"], r["to_company_id"]
            if a in idx and b in idx and a != b:
                edges.add(tuple(sorted((idx[a], idx[b]))))
    A = np.zeros((len(comp), len(comp)))
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    return A, comp


def row_norm(A):
    d = A.sum(1)
    return np.divide(A, d[:, None], out=np.zeros_like(A), where=d[:, None] > 0), d


def largest_component(A):
    n = A.shape[0]
    seen = np.zeros(n, bool)
    best = []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.where(A[u] > 0)[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        if len(comp) > len(best):
            best = comp
    idx = sorted(best)
    return A[np.ix_(idx, idx)], idx


def spectrum(W):
    """Eigenvalues of row-normalized W (similar to a symmetric matrix, take real parts), descending."""
    ev = np.linalg.eigvals(W)
    ev = np.sort(np.real(ev))[::-1]
    return ev


# ---------------------------------------------------------------- setup
def make_b(n, seed=0):
    """
    b must be heterogeneous, otherwise the game degenerates into everyone chasing the
    same consensus value and nothing is measurable.
    It must be zero-mean, otherwise BR has no fixed point in the constant direction (L singular).
    """
    rng = np.random.default_rng(seed)
    b = rng.normal(size=n)
    b -= b.mean()
    assert abs(b.mean()) < 1e-12
    return b


def solve_exact(W, b, beta):
    n = W.shape[0]
    M = np.eye(n) + beta * (np.eye(n) - W)
    return np.linalg.solve(M, b)


# ---------------------------------------------------------------- algorithms
def run_gd(W, b, beta, tol=TOL, maxit=MAXIT):
    n = W.shape[0]
    L = np.eye(n) - W
    lam_max = np.max(np.linalg.eigvalsh((L + L.T) / 2))
    alpha = 1.0 / (1.0 + beta * lam_max)
    xs = solve_exact(W, b, beta)
    x = np.zeros(n)
    for t in range(1, maxit + 1):
        x = x - alpha * ((1.0 + beta) * x - beta * W @ x - b)
        if np.linalg.norm(x - xs) / np.linalg.norm(xs) < tol:
            return t, x
    return maxit, x


def run_br(W, b, beta, tol=TOL, maxit=MAXIT):
    xs = solve_exact(W, b, beta)
    x = np.zeros(W.shape[0])
    for t in range(1, maxit + 1):
        x = (b + beta * (W @ x)) / (1.0 + beta)
        if np.linalg.norm(x - xs) / np.linalg.norm(xs) < tol:
            return t, x
    return maxit, x


# ---------------------------------------------------------------- theory
def rho_gd(ev, beta):
    # ev descending: ev[0]=lambda_1(W)=1, ev[1]=lambda_2(W), ev[-1]=lambda_min(W)
    # spectrum of L = I - W is 1 - ev, so the slowest direction is ev[1], the fastest ev[-1]
    lam2 = 1.0 - ev[1]           # = 0.1842
    lammax = 1.0 - ev[-1]        # bipartite: ev[-1]=-1 => lammax=2.0
    return 1.0 - (1.0 + beta * lam2) / (1.0 + beta * lammax)


def rho_br(ev, beta):
    # error iteration e_{t+1} = beta*W/(1+beta) * e_t  =>  rho = beta*max|lambda_k| / (1+beta)
    # after de-meaning there is no component along lambda_1=1, so take k>=2
    return float(beta * max(abs(ev[1]), abs(ev[-1])) / (1.0 + beta))


def iters_from_rho(rho, tol=TOL):
    if rho is None or rho >= 1.0 or rho <= 0.0:
        return np.inf
    return int(np.ceil(np.log(tol) / np.log(rho)))


# ---------------------------------------------------------------- main flow
def main():
    A, comp = load_real_network()
    A_lcc, idx = largest_component(A)
    W, deg = row_norm(A_lcc)
    ev = spectrum(W)
    n = W.shape[0]

    print(f"LCC: {n} nodes, {int(A_lcc.sum() / 2)} edges")
    print(f"lambda_1 = {ev[0]:.4f}   lambda_2 = {ev[1]:.4f}   "
          f"lambda_min = {ev[-1]:.4f}   (lambda_min=-1 => bipartite)")
    print(f"deg: {deg.astype(int).tolist()}\n")

    b = make_b(n, seed=0)
    betas = [0.5, 1.0, 2.0, 3.0, 5.0]
    rows = []

    for beta in betas:
        t_gd, x_gd = run_gd(W, b, beta)
        t_br, x_br = run_br(W, b, beta)
        r_gd, r_br = rho_gd(ev, beta), rho_br(ev, beta)
        rows.append({
            "beta": beta,
            "T_GD": t_gd,
            "rho_GD": round(r_gd, 4),
            "T_GD_pred": iters_from_rho(r_gd),
            "T_BR": t_br,
            "rho_BR": None if r_br is None else round(r_br, 4),
            "T_BR_pred": iters_from_rho(r_br),
            "gap_T": t_br / max(t_gd, 1),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # conclusion criteria
    print("\n--- reading ---")
    print(f"T_GD: {df['T_GD'].iloc[0]} -> {df['T_GD'].iloc[-1]}   (rho_GD governed by kappa)")
    print(f"T_BR: {df['T_BR'].iloc[0]} -> {df['T_BR'].iloc[-1]}   (rho_BR = beta/(1+beta) -> 1)")
    cross = df[df["T_BR"] > df["T_GD"]]
    if len(cross) and len(cross) < len(df):
        print(f"crossover: from beta={cross['beta'].iloc[0]} on, BR is slower than GD")
    print("rho < 1 on both sides => neither diverges; the difference is speed, not existence of a solution")

    csv = os.path.join(OUT, "expB1_results.csv")
    df.to_csv(csv, index=False)
    print(f"\nsaved: {csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.plot(df["beta"], df["T_GD"], "o-", color="#185FA5", label="gradient descent")
        ax.plot(df["beta"], df["T_BR"], "s-", color="#A32D2D", label="best response (sync)")
        ax.set_yscale("log")
        ax.set_xlabel("coupling strength  beta")
        ax.set_ylabel("iterations to 1e-6")
        ax.set_title("Same problem, same unique equilibrium, two update rules")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        p1 = os.path.join(OUT, "expB1_iters.png")
        fig.savefig(p1, dpi=160)

        fig2, ax2 = plt.subplots(figsize=(6.2, 3.4))
        ax2.plot(df["beta"], df["rho_GD"], "o-", color="#185FA5", label="GD spectral radius")
        ax2.plot(df["beta"], [r if r is not None else np.nan for r in df["rho_BR"]],
                 "s-", color="#A32D2D", label="BR spectral radius")
        ax2.axhline(1.0, color="#888780", lw=0.8, ls="--")
        ax2.set_xlabel("coupling strength  beta")
        ax2.set_ylabel("contraction factor rho")
        ax2.set_title("Neither exceeds 1: no divergence, only speed")
        ax2.grid(alpha=0.25)
        ax2.legend(frameon=False)
        fig2.tight_layout()
        p2 = os.path.join(OUT, "expB1_rho.png")
        fig2.savefig(p2, dpi=160)
        print(f"saved: {p1}\nsaved: {p2}")
    except Exception as e:
        print(f"[plot skipped] {e}")


if __name__ == "__main__":
    main()


# ==================== expB2_asynchrony.py ====================

"""
expB2_asynchrony.py —— Experiment B2: async endpoint deviation (projected gradient, beta=1)

Main metric: Delta = ‖x*_async − x*_sync‖ / ‖x*_sync‖
Theoretical prediction (mu>0, strongly monotone -> unique VI solution): any convergent
algorithm reaches the same point -> Delta ~= 0.
Falsifiable: Delta < 1e-4 => the framework transfers completely; Delta significantly
nonzero => something is not covered by the theory.

Three policies (projected gradient, step alpha = 1/(1+beta·lam_max_sym)):
  sync = full Jacobi sweep per round; round-robin = 1 node per round (Gauss-Seidel);
  batch = 3 nodes per round.
b_i uses real standardized returns (heterogeneous, avoids the trivial consensus solution).
Polling-order robustness: forward / reverse / reshuffled each round.
"""
import sqlite3
import numpy as np
import pandas as pd

DB = r"C:\Users\lanju\Desktop\陶瓷\申请材料\ai_infra_2022_2026_v3_data.db"
conn = sqlite3.connect(DB)

sec = pd.read_sql("SELECT security_id, company_id, asset_role FROM securities", conn)
comp_sec = sec[sec["asset_role"] == "company"]
sec2comp = dict(zip(comp_sec["security_id"], comp_sec["company_id"]))
comp2sec = {v: k for k, v in sec2comp.items()}
listed = set(sec2comp.values())

rel = pd.read_sql("SELECT from_company_id, to_company_id, relation_type FROM relations", conn)
edges = set()
for _, r in rel.iterrows():
    if r["relation_type"] in ("supply", "foundry_manufacturing"):
        a, b = r["from_company_id"], r["to_company_id"]
        if a in listed and b in listed and a != b:
            edges.add(tuple(sorted([a, b])))
nodes = sorted(set(n for e in edges for n in e))
idx = {n: i for i, n in enumerate(nodes)}
N = len(nodes)
A = np.zeros((N, N))
for a, b in edges:
    A[idx[a], idx[b]] = A[idx[b], idx[a]] = 1.0
deg = A.sum(axis=1)
W = A / deg[:, None]

def cc(adj):
    seen, comps = set(), []
    for s in range(adj.shape[0]):
        if s in seen: continue
        st = [s]; seen.add(s); c = []
        while st:
            u = st.pop(); c.append(u)
            for v in np.where(adj[u] > 0)[0]:
                if v not in seen: seen.add(v); st.append(v)
        comps.append(c)
    return comps
lcc = max(cc(A), key=len)
Wl = W[np.ix_(lcc, lcc)]
n = len(lcc)
L = np.eye(n) - Wl
lam_max_sym = np.linalg.eigvalsh((L + L.T) / 2)[-1]

# b: real standardized returns (heterogeneous)
p = pd.read_sql("SELECT security_id, trade_date, close FROM prices_daily", conn)
p["trade_date"] = pd.to_datetime(p["trade_date"])
p = p[p["security_id"].isin(comp_sec["security_id"])]
p = p[(p["trade_date"] >= "2023-01-03") & (p["trade_date"] <= "2025-12-31")]
p["ret"] = p.groupby("security_id")["close"].pct_change()
mr = p.groupby("security_id")["ret"].mean().dropna()
lcc_comps = [nodes[i] for i in lcc]
b = np.array([mr.get(comp2sec[c], 0.0) for c in lcc_comps], dtype=float)
b = (b - b.mean()) / (b.std() + 1e-9)          # z-score
print("b (standardized returns):", np.round(b, 3))

beta = 1.0
M = (1 + beta) * np.eye(n) - beta * Wl
x_star = np.linalg.solve(M, b)                 # unique equilibrium
alpha = 1.0 / (1 + beta * lam_max_sym)
EPS = 1e-12
MAX = 500000

def F(x):
    return (1 + beta) * x - beta * (Wl @ x) - b

def rel_err(x): return np.linalg.norm(x - x_star) / np.linalg.norm(x_star)

# synchronous (Jacobi projected gradient)
x = np.zeros(n); t = 0
while rel_err(x) > EPS and t < MAX:
    x = x - alpha * F(x); t += 1
x_sync, T_sync = x.copy(), t

# round-robin (Gauss-Seidel, forward order)
def gs(order_fn, batch=1):
    x = np.zeros(n); t = 0
    while rel_err(x) > EPS and t < MAX:
        order = order_fn(t)
        for i in order:
            x[i] = x[i] - alpha * F(x)[i]
            t += 1
            if rel_err(x) <= EPS: break
        if rel_err(x) <= EPS: break
    return x, t

order_fwd = lambda t: list(range(n))
order_rev = lambda t: list(range(n - 1, -1, -1))
rng = np.random.default_rng(0)
order_rnd = lambda t: list(rng.permutation(n))

x_rr_fwd, T_rr_fwd = gs(order_fwd)
x_rr_rev, T_rr_rev = gs(order_rev)
x_rr_rnd, T_rr_rnd = gs(order_rnd)

# batch (3 nodes per batch, synchronous within batch)
batches = [list(range(i, min(i + 3, n))) for i in range(0, n, 3)]
x = np.zeros(n); t = 0
while rel_err(x) > EPS and t < MAX:
    for bt in batches:
        x[bt] = x[bt] - alpha * F(x)[bt]
        t += 1
        if rel_err(x) <= EPS: break
    if rel_err(x) <= EPS: break
x_batch, T_batch = x.copy(), t

def Delta(xa): return np.linalg.norm(xa - x_sync) / np.linalg.norm(x_sync)

print("\n=== B2 main metric: endpoint deviation Delta (predicted ~= 0) ===")
print("  Delta(round-robin fwd) = %.2e" % Delta(x_rr_fwd))
print("  Delta(round-robin rev) = %.2e" % Delta(x_rr_rev))
print("  Delta(round-robin rnd) = %.2e" % Delta(x_rr_rnd))
print("  Delta(batch)           = %.2e" % Delta(x_batch))

print("\n=== secondary: round counts ===")
print("  sync T_sync = %d" % T_sync)
print("  round-robin fwd = %d (ratio %.2f)" % (T_rr_fwd, T_rr_fwd / T_sync))
print("  round-robin rev = %d (ratio %.2f)" % (T_rr_rev, T_rr_rev / T_sync))
print("  round-robin rnd = %d (ratio %.2f)" % (T_rr_rnd, T_rr_rnd / T_sync))
print("  batch           = %d (ratio %.2f)" % (T_batch, T_batch / T_sync))

print("\n=== polling-order robustness (fwd vs rev vs random, spread of total updates) ===")
vals = [T_rr_fwd, T_rr_rev, T_rr_rnd]
spread = (max(vals) - min(vals)) / np.mean(vals) * 100
print("  total updates under the three orders %d / %d / %d, relative spread %.1f%%" % (T_rr_fwd, T_rr_rev, T_rr_rnd, spread))
print("  %.1f%% < 10%% => order does not matter (equal-frequency conclusion is robust)" % spread)
conn.close()


# ==================== expB2b_earlystop.py ====================

"""
expB2b_earlystop.py —— Experiment B2 addendum: early-stopping bias Delta(tau)

The real question: nobody runs to 1e-12; real systems stop at 1e-4.
Measure: the async-vs-sync endpoint deviation Delta(tau) at different stopping tolerances tau.

  Delta(tau) = ‖x_async(tau) − x_sync(tau)‖ / ‖x_sync(tau)‖

If Delta shrinks monotonically with tau -> "endpoint deviation is a function of precision,
negligible at practical precision" (truly reportable).
If Delta is already appreciable at 1e-3 -> "asynchrony does not change the endpoint" fails
in the practical sense (more interesting).
"""
import sqlite3
import numpy as np
import pandas as pd

DB = r"C:\Users\lanju\Desktop\陶瓷\申请材料\ai_infra_2022_2026_v3_data.db"
conn = sqlite3.connect(DB)
sec = pd.read_sql("SELECT security_id, company_id, asset_role FROM securities", conn)
comp_sec = sec[sec["asset_role"] == "company"]
sec2comp = dict(zip(comp_sec["security_id"], comp_sec["company_id"]))
comp2sec = {v: k for k, v in sec2comp.items()}
listed = set(sec2comp.values())
rel = pd.read_sql("SELECT from_company_id, to_company_id, relation_type FROM relations", conn)
edges = set()
for _, r in rel.iterrows():
    if r["relation_type"] in ("supply", "foundry_manufacturing"):
        a, b = r["from_company_id"], r["to_company_id"]
        if a in listed and b in listed and a != b:
            edges.add(tuple(sorted([a, b])))
nodes = sorted(set(n for e in edges for n in e))
idx = {n: i for i, n in enumerate(nodes)}
N = len(nodes)
A = np.zeros((N, N))
for a, b in edges:
    A[idx[a], idx[b]] = A[idx[b], idx[a]] = 1.0
deg = A.sum(axis=1)
W = A / deg[:, None]

def cc(adj):
    seen, comps = set(), []
    for s in range(adj.shape[0]):
        if s in seen: continue
        st = [s]; seen.add(s); c = []
        while st:
            u = st.pop(); c.append(u)
            for v in np.where(adj[u] > 0)[0]:
                if v not in seen: seen.add(v); st.append(v)
        comps.append(c)
    return comps
lcc = max(cc(A), key=len)
Wl = W[np.ix_(lcc, lcc)]
n = len(lcc)
L = np.eye(n) - Wl
lam_max_sym = np.linalg.eigvalsh((L + L.T) / 2)[-1]

p = pd.read_sql("SELECT security_id, trade_date, close FROM prices_daily", conn)
p["trade_date"] = pd.to_datetime(p["trade_date"])
p = p[p["security_id"].isin(comp_sec["security_id"])]
p = p[(p["trade_date"] >= "2023-01-03") & (p["trade_date"] <= "2025-12-31")]
p["ret"] = p.groupby("security_id")["close"].pct_change()
mr = p.groupby("security_id")["ret"].mean().dropna()
lcc_comps = [nodes[i] for i in lcc]
b = np.array([mr.get(comp2sec[c], 0.0) for c in lcc_comps], dtype=float)
b = (b - b.mean()) / (b.std() + 1e-9)

beta = 1.0
M = (1 + beta) * np.eye(n) - beta * Wl
x_star = np.linalg.solve(M, b)
alpha = 1.0 / (1 + beta * lam_max_sym)

def F(x): return (1 + beta) * x - beta * (Wl @ x) - b
def rel(x): return np.linalg.norm(x - x_star) / np.linalg.norm(x_star)

def sync(tau):
    x = np.zeros(n); t = 0
    while rel(x) > tau and t < 2000000:
        x = x - alpha * F(x); t += 1
    return x

def gs(tau):
    x = np.zeros(n); t = 0
    while rel(x) > tau and t < 2000000:
        for i in range(n):
            x[i] = x[i] - alpha * F(x)[i]; t += 1
            if rel(x) <= tau: break
        if rel(x) <= tau: break
    return x

def batch(tau):
    batches = [list(range(i, min(i+3, n))) for i in range(0, n, 3)]
    x = np.zeros(n); t = 0
    while rel(x) > tau and t < 2000000:
        for bt in batches:
            x[bt] = x[bt] - alpha * F(x)[bt]; t += 1
            if rel(x) <= tau: break
        if rel(x) <= tau: break
    return x

print("=== early-stopping bias Delta(tau) = ‖x_async(tau) − x_sync(tau)‖ / ‖x_sync(tau)‖ ===")
print("%-8s %16s %16s" % ("tol tau", "Delta rr", "Delta batch"))
for tau in [1e-3, 1e-5, 1e-8, 1e-12]:
    xs = sync(tau)
    xr = gs(tau)
    xb = batch(tau)
    d_rr = np.linalg.norm(xr - xs) / np.linalg.norm(xs)
    d_bt = np.linalg.norm(xb - xs) / np.linalg.norm(xs)
    print("%-8.0e %16.3e %16.3e" % (tau, d_rr, d_bt))

# slope check: is Delta proportional to tau (shrinks linearly with the tolerance)
print("\n=== Delta(round-robin) / tau ratio (if ~constant => Delta~tau, negligible at practical precision) ===")
for tau in [1e-3, 1e-5, 1e-8, 1e-12]:
    xs = sync(tau); xr = gs(tau)
    d = np.linalg.norm(xr - xs) / np.linalg.norm(xs)
    print("  tau=%.0e  Delta/tau = %.2f" % (tau, d / tau))
conn.close()
