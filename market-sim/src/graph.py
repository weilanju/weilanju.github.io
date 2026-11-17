"""Directed upstream→downstream supply graph + bottleneck value scores."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from . import config as cfg


def _orient_edge(from_id: str, to_id: str, flow_direction: str) -> list[tuple[str, str]]:
    fd = str(flow_direction or "from_to").lower()
    if fd == "bidirectional":
        return [(from_id, to_id), (to_id, from_id)]
    if fd == "to_from":
        return [(to_id, from_id)]
    if fd == "none":
        return []
    return [(from_id, to_id)]


def build_supply_graph(companies: pd.DataFrame, relations: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in companies.iterrows():
        seg = row.get("value_chain_segment") or "unknown"
        rank = cfg.SEGMENT_ORDER.index(seg) if seg in cfg.SEGMENT_ORDER else len(cfg.SEGMENT_ORDER)
        G.add_node(
            row["company_id"],
            ticker=row.get("ticker"),
            company_name=row.get("company_name"),
            value_chain_segment=seg,
            segment_rank=rank,
        )

    for _, row in relations.iterrows():
        rtype = str(row.get("relation_type") or "")
        if rtype not in cfg.FLOW_TYPES:
            continue
        src, dst = row["from_company_id"], row["to_company_id"]
        for n in (src, dst):
            if n not in G:
                G.add_node(
                    n,
                    value_chain_segment="unknown",
                    segment_rank=len(cfg.SEGMENT_ORDER),
                )
        attrs = {
            "relation_id": row.get("relation_id"),
            "relation_type": rtype,
            "flow_direction": row.get("flow_direction"),
        }
        for u, v in _orient_edge(src, dst, row.get("flow_direction")):
            G.add_edge(u, v, **attrs)
    return G


def export_graph_tables(G: nx.DiGraph, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame(
        [
            {
                "company_id": n,
                "ticker": G.nodes[n].get("ticker"),
                "company_name": G.nodes[n].get("company_name"),
                "value_chain_segment": G.nodes[n].get("value_chain_segment"),
                "segment_rank": G.nodes[n].get("segment_rank"),
                "in_degree": G.in_degree(n),
                "out_degree": G.out_degree(n),
            }
            for n in G.nodes()
        ]
    )
    edges = pd.DataFrame(
        [
            {
                "from_company_id": u,
                "to_company_id": v,
                "relation_id": d.get("relation_id"),
                "relation_type": d.get("relation_type"),
                "flow_direction": d.get("flow_direction"),
            }
            for u, v, d in G.edges(data=True)
        ]
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(data_dir / "graph_nodes.csv", index=False)
    edges.to_csv(data_dir / "graph_edges.csv", index=False)
    return nodes, edges


def segment_counts(nodes: pd.DataFrame) -> pd.Series:
    return nodes["value_chain_segment"].value_counts()


def compute_bottleneck_scores(G: nx.DiGraph) -> pd.DataFrame:
    """
    Value = equal-weight bottleneck:
      downstream coverage × low substitutability × edge scarcity
    (each min-max to [0,1], then mean). Research proxy, not official productivity.
    """
    nodes = list(G.nodes())
    n = max(len(nodes), 1)

    downstream_cov = {}
    for u in nodes:
        try:
            reach = nx.descendants(G, u)
        except Exception:  # noqa: BLE001
            reach = set()
        downstream_cov[u] = len(reach) / n

    seg_members: dict[str, list[str]] = {}
    for u in nodes:
        seg = G.nodes[u].get("value_chain_segment") or "unknown"
        seg_members.setdefault(seg, []).append(u)

    low_subst = {}
    for u in nodes:
        seg = G.nodes[u].get("value_chain_segment") or "unknown"
        peers = len(seg_members.get(seg, [])) - 1
        out_d = G.out_degree(u)
        low_subst[u] = 1.0 / (1.0 + peers + out_d)

    edge_scarce = {}
    for u in nodes:
        deg = G.in_degree(u) + G.out_degree(u)
        edge_scarce[u] = 1.0 / (1.0 + deg)

    def _minmax(d: dict[str, float]) -> dict[str, float]:
        vals = np.array([d[u] for u in nodes], dtype=float)
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            return {u: 0.0 for u in nodes}
        return {u: float((d[u] - lo) / (hi - lo)) for u in nodes}

    dc, ls, es = _minmax(downstream_cov), _minmax(low_subst), _minmax(edge_scarce)
    rows = []
    for u in nodes:
        vs = (dc[u] + ls[u] + es[u]) / 3.0
        rows.append(
            {
                "company_id": u,
                "downstream_coverage": dc[u],
                "low_substitutability": ls[u],
                "edge_scarcity": es[u],
                "value_score": vs,
                "z_bottleneck": vs,  # alias for structure agent
                "value_chain_segment": G.nodes[u].get("value_chain_segment"),
                "in_degree": G.in_degree(u),
                "out_degree": G.out_degree(u),
            }
        )
    out = pd.DataFrame(rows)
    # z-score of value_score for structure quote
    mu, sd = out["value_score"].mean(), out["value_score"].std()
    if sd and np.isfinite(sd) and sd > 1e-12:
        out["z_bottleneck"] = (out["value_score"] - mu) / sd
    else:
        out["z_bottleneck"] = 0.0
    return out


def neighbors_undirected(G: nx.DiGraph) -> dict[str, list[str]]:
    out: dict[str, set[str]] = {n: set() for n in G.nodes()}
    for u, v in G.edges():
        out[u].add(v)
        out[v].add(u)
    return {k: sorted(v) for k, v in out.items()}


def plot_graph_overview(G: nx.DiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 10))
    # layout by segment rank (y) + jitter x within segment
    rng = np.random.default_rng(42)
    pos = {}
    by_seg: dict[int, list[str]] = {}
    for n in G.nodes():
        r = int(G.nodes[n].get("segment_rank", 99))
        by_seg.setdefault(r, []).append(n)
    for r, members in by_seg.items():
        for i, n in enumerate(sorted(members)):
            x = (i - (len(members) - 1) / 2) + float(rng.normal(0, 0.05))
            y = -r  # top segments higher on plot
            pos[n] = (x, y)

    seg_colors = plt.cm.tab10(np.linspace(0, 1, max(len(cfg.SEGMENT_ORDER), 1)))
    color_map = {
        seg: seg_colors[i % len(seg_colors)] for i, seg in enumerate(cfg.SEGMENT_ORDER)
    }
    node_colors = [
        color_map.get(G.nodes[n].get("value_chain_segment"), (0.7, 0.7, 0.7, 1.0))
        for n in G.nodes()
    ]
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=8, alpha=0.35, width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=220, alpha=0.9)
    labels = {n: (G.nodes[n].get("ticker") or n.replace("C_", "")) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6)
    ax.set_title("AI infra supply graph (top=power_cooling → bottom=cloud_buyers)")
    ax.axis("off")
    # legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[s],
            markersize=8,
            label=s,
        )
        for s in cfg.SEGMENT_ORDER
        if any(G.nodes[n].get("value_chain_segment") == s for n in G.nodes())
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
