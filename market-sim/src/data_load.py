"""Load pack tables and build company×date price panel (true prices only)."""
from __future__ import annotations

import pandas as pd

from . import config as cfg


def load_companies() -> pd.DataFrame:
    return pd.read_csv(cfg.COMPANIES)


def load_securities() -> pd.DataFrame:
    return pd.read_csv(cfg.SECURITIES)


def load_relations() -> pd.DataFrame:
    return pd.read_csv(cfg.RELATIONS)


def load_events() -> pd.DataFrame:
    df = pd.read_csv(cfg.EVENTS)
    df["usable_from_trade_date"] = pd.to_datetime(df["usable_from_trade_date"])
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    return df


def load_prices() -> pd.DataFrame:
    usecols = [
        "security_id",
        "trade_date",
        "window",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    df = pd.read_csv(cfg.PRICES, usecols=usecols)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def build_universe(companies: pd.DataFrame, securities: pd.DataFrame) -> pd.DataFrame:
    """Company-role securities only (exclude benchmarks)."""
    secs = securities[securities["asset_role"] == "company"].copy()
    uni = secs.merge(
        companies[
            [
                "company_id",
                "ticker",
                "company_name",
                "value_chain_segment",
                "sub_role",
            ]
        ],
        on="company_id",
        how="left",
        suffixes=("", "_co"),
    )
    if "ticker_co" in uni.columns:
        uni["ticker"] = uni["ticker"].fillna(uni["ticker_co"])
        uni = uni.drop(columns=["ticker_co"])
    return uni


def build_price_panel(prices: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """True-price panel: security × date. Never overwrite with fitted quotes."""
    keep = set(universe["security_id"])
    px = prices[prices["security_id"].isin(keep)].copy()
    px = px.merge(
        universe[
            [
                "security_id",
                "company_id",
                "ticker",
                "value_chain_segment",
            ]
        ],
        on="security_id",
        how="left",
    )
    px = px.sort_values(["security_id", "trade_date"]).reset_index(drop=True)
    # true close as decision/eval price
    px["true_price"] = px["close"].astype(float)
    g = px.groupby("security_id", sort=False)
    px["prev_price"] = g["true_price"].shift(1)
    px["ret"] = px["true_price"] / px["prev_price"] - 1.0
    return px


def formal_dates(panel: pd.DataFrame) -> list[pd.Timestamp]:
    d = panel.loc[panel["window"] == cfg.WINDOW_FORMAL, "trade_date"]
    return sorted(pd.to_datetime(d.unique()))


def split_train_test(dates: list[pd.Timestamp]) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    n = len(dates)
    mid = int(n * cfg.MAIN_TRAIN_FRAC)
    return dates[:mid], dates[mid:]
