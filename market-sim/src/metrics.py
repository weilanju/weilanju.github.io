"""MAPE metrics + comparison plot."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baselines import mape_pct


def eval_models(pred_frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in pred_frames:
        if df is None or df.empty:
            continue
        model = str(df["model"].iloc[0])
        yt = df["true_price"].to_numpy(dtype=float)
        yp = df["pred"].to_numpy(dtype=float)
        m = mape_pct(yt, yp)
        n = int((np.isfinite(yt) & np.isfinite(yp)).sum())
        rows.append({"model": model, "mape_pct": m, "n": n})
    order = ["ma7", "public_only", "lgbm_lag", "four_agent_fit"]
    out = pd.DataFrame(rows)
    if not out.empty:
        out["ord"] = out["model"].map({m: i for i, m in enumerate(order)})
        out = out.sort_values("ord").drop(columns=["ord"]).reset_index(drop=True)
    return out


def plot_baseline_comparison(metrics: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    order = ["ma7", "public_only", "lgbm_lag", "four_agent_fit"]
    sub = metrics[metrics["model"].isin(order)].copy()
    sub["ord"] = sub["model"].map({m: i for i, m in enumerate(order)})
    sub = sub.sort_values("ord")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        sub["model"],
        sub["mape_pct"],
        color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"],
    )
    ax.set_ylabel("Test MAPE %")
    ax.set_title("Blind-protocol baselines vs four_agent_fit (fitted vs true)")
    for i, r in enumerate(sub.itertuples()):
        ax.text(i, r.mape_pct, f"{r.mape_pct:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
