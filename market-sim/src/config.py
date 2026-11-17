"""Paths, segment order, quote rule B, windows (prereg-locked)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"

PRICES = Path("/workspace/ai_infra_v3_pack/prices_daily.csv")
RELATIONS = Path("/workspace/ai_infra_v3_pack/relations.csv")
EVENTS = Path("/workspace/ai_infra_v3_pack/events.csv")
COMPANIES = Path("/workspace/ai-infra-audit/extract/v3__tables__companies.csv")
SECURITIES = Path("/workspace/ai-infra-audit/extract/v3__tables__securities.csv")

# top (energy/power) → bottom (cloud_buyers / AI)
SEGMENT_ORDER = [
    "power_cooling",
    "semi_equipment_eda",
    "foundry_packaging",
    "memory_storage",
    "compute_chips",
    "networking",
    "servers",
    "data_center",
    "cloud_buyers",
]

FLOW_TYPES = {
    "supply",
    "foundry_manufacturing",
    "customer",
    "purchase_commitment",
    "product_integration",
    "joint_design",
    "technology_partnership",
    "ecosystem_partnership",
    "cloud_partnership",
}

WINDOW_WARMUP = "warmup_2022"
WINDOW_FORMAL = "formal_2023_2025"
WINDOW_HOLDOUT = "holdout_2026"

NUDGE_CLIP = 0.05
MA_WINDOW = 7
MAIN_TRAIN_FRAC = 0.5

# online learning step for train-half updates
LEARN_LR = 0.08
LEARN_EPS = 1e-8
