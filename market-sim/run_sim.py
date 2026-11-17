#!/usr/bin/env python3
"""Entry: six-step day-loop four-agent fitted-price simulation."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_VENV_PY = _ROOT / ".venv" / "bin" / "python"
_VENV_DIR = _ROOT / ".venv"
if _VENV_PY.exists() and Path(sys.prefix).resolve() != _VENV_DIR.resolve():
    os.execv(str(_VENV_PY), [str(_VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]])

import pandas as pd

sys.path.insert(0, str(_ROOT))

from src import config as cfg
from src.baselines import (
    baseline_four_agent,
    baseline_lgbm_lag,
    baseline_ma7,
    baseline_public_only,
)
from src.data_load import (
    build_price_panel,
    build_universe,
    formal_dates,
    load_companies,
    load_events,
    load_prices,
    load_relations,
    load_securities,
    split_train_test,
)
from src.graph import (
    build_supply_graph,
    compute_bottleneck_scores,
    export_graph_tables,
    neighbors_undirected,
    plot_graph_overview,
    segment_counts,
)
from src.info_packets import (
    InfoPacketStore,
    build_public_event_rows,
    export_info_packets_sample,
)
from src.metrics import eval_models, plot_baseline_comparison
from src.simulation import (
    DaySimulator,
    export_formed_prices,
    export_reaction_sample,
    train_reaction_summary,
)
from src.surfaces import export_price_surface, export_value_surface


def main() -> int:
    t0 = time.perf_counter()
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: supply graph ===")
    companies = load_companies()
    relations = load_relations()
    G = build_supply_graph(companies, relations)
    nodes, edges = export_graph_tables(G, cfg.DATA_DIR)
    plot_graph_overview(G, cfg.OUT_DIR / "graph_overview.png")
    seg_ct = segment_counts(nodes)
    print(f"nodes={len(nodes)} edges={len(edges)}")
    print("segment counts:\n", seg_ct.to_string())
    neigh = neighbors_undirected(G)
    value_df = compute_bottleneck_scores(G)
    # z for structure agent
    value_by_company = {
        r.company_id: float(r.z_bottleneck) for r in value_df.itertuples(index=False)
    }

    print("=== Step 2: four Agent classes loaded (Public/Channel/Structure/Learning) ===")
    print("  classes are independent; instantiated inside DaySimulator")

    print("=== Step 3: price surface (TRUE) + value surface (bottleneck proxy) ===")
    securities = load_securities()
    prices = load_prices()
    universe = build_universe(companies, securities)
    panel = build_price_panel(prices, universe)
    export_price_surface(panel, cfg.DATA_DIR, cfg.OUT_DIR)
    export_value_surface(value_df, cfg.DATA_DIR, cfg.OUT_DIR)
    print("  price_surface = true closes; value_surface = bottleneck (NOT productivity)")

    print("=== Step 4: PIT info packets ===")
    events = load_events()
    public_events = build_public_event_rows(events)
    store = InfoPacketStore(public_events, neigh, panel)
    export_info_packets_sample(store, panel, cfg.DATA_DIR)
    print(f"  public event rows={len(public_events)}  (packets assembled per day in loop)")

    print("=== Step 5: day-loop simulation (train then frozen test) ===")
    dates = formal_dates(panel)
    train_dates, test_dates = split_train_test(dates)
    print(
        f"  formal days={len(dates)} train={len(train_dates)} "
        f"({train_dates[0].date()}..{train_dates[-1].date()}) "
        f"test={len(test_dates)} ({test_dates[0].date()}..{test_dates[-1].date()})"
    )
    # warmup rows stay in panel for prev_price / lags; holdout excluded from loop
    sim = DaySimulator(panel, store, value_by_company, train_dates, test_dates)
    reactions, formed, sim_meta = sim.run()
    export_reaction_sample(reactions, cfg.OUT_DIR)
    export_formed_prices(formed, cfg.OUT_DIR)
    react_sum = train_reaction_summary(reactions)
    react_sum.to_csv(cfg.OUT_DIR / "train_reaction_summary.csv", index=False)
    print("  train reaction summary:")
    print(react_sum.to_string(index=False))
    print(f"  formed test rows={len(formed)}")

    print("=== Step 6: baselines + MAPE ===")
    sec_ids = set(formed["security_id"].unique()) if not formed.empty else set(
        panel["security_id"].unique()
    )
    preds = [
        baseline_ma7(panel, test_dates, sec_ids),
        baseline_public_only(formed),
        baseline_lgbm_lag(panel, train_dates, test_dates, sec_ids),
        baseline_four_agent(formed),
    ]
    metrics = eval_models(preds)
    metrics.to_csv(cfg.OUT_DIR / "metrics_summary.csv", index=False)
    plot_baseline_comparison(metrics, cfg.OUT_DIR / "baseline_comparison.png")
    print(metrics.to_string(index=False))

    elapsed = time.perf_counter() - t0
    meta = {
        "elapsed_sec": elapsed,
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "segment_counts": {str(k): int(v) for k, v in seg_ct.items()},
        "n_formal_days": len(dates),
        "n_train_days": len(train_dates),
        "n_test_days": len(test_dates),
        "train_date_range": [str(train_dates[0].date()), str(train_dates[-1].date())],
        "test_date_range": [str(test_dates[0].date()), str(test_dates[-1].date())],
        "holdout_excluded": True,
        "fitted_definition": (
            "fitted_price = equal-weight mean of "
            "q_public, q_channel, q_structure, q_learning; "
            "true_price is exogenous and never overwritten by fitted"
        ),
        "simulation": sim_meta,
        "outputs": {
            "graph_nodes": str(cfg.DATA_DIR / "graph_nodes.csv"),
            "graph_edges": str(cfg.DATA_DIR / "graph_edges.csv"),
            "graph_overview": str(cfg.OUT_DIR / "graph_overview.png"),
            "price_surface": str(cfg.DATA_DIR / "price_surface.csv"),
            "price_surface_snapshot": str(cfg.OUT_DIR / "price_surface_snapshot.png"),
            "value_surface": str(cfg.DATA_DIR / "value_surface.csv"),
            "value_surface_png": str(cfg.OUT_DIR / "value_surface.png"),
            "info_packets_sample": str(cfg.DATA_DIR / "info_packets_sample.csv"),
            "daily_reactions_train_sample": str(
                cfg.OUT_DIR / "daily_reactions_train_sample.csv"
            ),
            "agent_formed_prices_test": str(
                cfg.OUT_DIR / "agent_formed_prices_test.csv"
            ),
            "train_reaction_summary": str(cfg.OUT_DIR / "train_reaction_summary.csv"),
            "metrics_summary": str(cfg.OUT_DIR / "metrics_summary.csv"),
            "baseline_comparison": str(cfg.OUT_DIR / "baseline_comparison.png"),
        },
        "metrics": metrics.to_dict(orient="records"),
    }
    with open(cfg.OUT_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    print(f"=== DONE elapsed_sec={elapsed:.3f} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
