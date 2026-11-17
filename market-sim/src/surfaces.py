"""Price surface (true trades) and value surface (bottleneck proxy)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def export_price_surface(panel: pd.DataFrame, data_dir: Path, out_dir: Path) -> pd.DataFrame:
    """
    Price surface = real traded closes on (security/company × date).
    Fitted quotes must NEVER overwrite this table.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "security_id",
        "company_id",
        "ticker",
        "value_chain_segment",
        "trade_date",
        "window",
        "true_price",
        "volume",
    ]
    surf = panel[cols].copy()
    surf.to_csv(data_dir / "price_surface.csv", index=False)

    # snapshot: one formal day heatmap (companies × that date is 1-D; show segment bars)
    formal = surf[surf["window"] == "formal_2023_2025"]
    if formal.empty:
        snap_date = surf["trade_date"].max()
    else:
        # mid formal day
        dates = sorted(formal["trade_date"].unique())
        snap_date = dates[len(dates) // 2]
    day = surf[surf["trade_date"] == snap_date].copy()
    day = day.dropna(subset=["true_price"]).sort_values("true_price", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    if not day.empty:
        ax.bar(range(len(day)), day["true_price"].values, color="steelblue")
        ax.set_xticks(range(len(day)))
        ax.set_xticklabels(day["ticker"].fillna(day["security_id"]).tolist(), rotation=90, fontsize=7)
        ax.set_ylabel("true close")
        ax.set_title(f"Price surface snapshot (TRUE closes) — {pd.Timestamp(snap_date).date()}")
    else:
        ax.text(0.5, 0.5, "no prices", ha="center")
    fig.tight_layout()
    fig.savefig(out_dir / "price_surface_snapshot.png", dpi=140)
    plt.close(fig)
    return surf


def export_value_surface(value_df: pd.DataFrame, data_dir: Path, out_dir: Path) -> pd.DataFrame:
    """
    Value surface = bottleneck score (research supply-chain proxy, NOT official productivity).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = value_df.copy()
    out.to_csv(data_dir / "value_surface.csv", index=False)

    plot_df = out.sort_values("value_score", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = plot_df["company_id"].str.replace("C_", "", regex=False)
    ax.bar(range(len(plot_df)), plot_df["value_score"].values, color="darkorange")
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(labels.tolist(), rotation=90, fontsize=7)
    ax.set_ylabel("value_score (equal-weight bottleneck)")
    ax.set_title("Value surface — bottleneck proxy (NOT official productivity)")
    fig.tight_layout()
    fig.savefig(out_dir / "value_surface.png", dpi=140)
    plt.close(fig)
    return out
