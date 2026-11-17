"""Blind-protocol baselines on the same test names/dates as four_agent_fit."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def mape_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-12)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100.0)


def baseline_ma7(
    panel: pd.DataFrame, test_dates: list, security_ids: set[str]
) -> pd.DataFrame:
    """Near-7-day mean of TRUE closes through t-1 (no same-day peek)."""
    df = panel[panel["security_id"].isin(security_ids)].copy()
    parts = []
    for _, sub in df.groupby("security_id", sort=False):
        sub = sub.sort_values("trade_date").copy()
        roll = sub["true_price"].rolling(cfg.MA_WINDOW, min_periods=1).mean()
        sub["ma7"] = roll.shift(1)
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True)
    test_set = {pd.Timestamp(d) for d in test_dates}
    out = df[df["trade_date"].isin(test_set)][
        ["security_id", "trade_date", "true_price", "ma7"]
    ].rename(columns={"ma7": "pred", "trade_date": "date"})
    out["model"] = "ma7"
    return out


def baseline_public_only(formed_test: pd.DataFrame) -> pd.DataFrame:
    out = formed_test[["security_id", "trade_date", "true_price", "q_public"]].copy()
    out = out.rename(columns={"q_public": "pred", "trade_date": "date"})
    out["model"] = "public_only"
    return out


def baseline_lgbm_lag(
    panel: pd.DataFrame,
    train_dates: list,
    test_dates: list,
    security_ids: set[str],
) -> pd.DataFrame:
    """LightGBM on lagged price/volume only — no graph / no agent info."""
    from lightgbm import LGBMRegressor

    df = panel[panel["security_id"].isin(security_ids)].copy()
    parts = []
    for _, sub in df.groupby("security_id", sort=False):
        sub = sub.sort_values("trade_date").copy()
        sub["ret_lag1"] = sub["ret"].shift(1)
        sub["ret_lag5"] = sub["ret"].shift(1).rolling(5, min_periods=1).mean()
        sub["vol_lag1"] = sub["volume"].shift(1)
        mu = sub["volume"].shift(1).rolling(20, min_periods=5).mean()
        sd = sub["volume"].shift(1).rolling(20, min_periods=5).std()
        sub["vol_z"] = (
            ((sub["vol_lag1"] - mu) / sd.replace(0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        sub["px_lag1"] = sub["true_price"].shift(1)
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True)
    feats = ["ret_lag1", "ret_lag5", "vol_z", "px_lag1"]
    for c in feats:
        df[c] = df[c].fillna(0.0)

    train_set = {pd.Timestamp(d) for d in train_dates}
    test_set = {pd.Timestamp(d) for d in test_dates}
    tr = df[df["trade_date"].isin(train_set)].dropna(subset=["true_price", "px_lag1"])
    te = df[df["trade_date"].isin(test_set)].dropna(subset=["true_price", "px_lag1"])

    y = (tr["true_price"] / tr["px_lag1"].replace(0, np.nan) - 1.0).to_numpy()
    X = tr[feats].to_numpy(dtype=float)
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[m], y[m]
    # clip extreme returns for stability
    y = np.clip(y, -0.5, 0.5)
    if X.shape[0] < 50:
        raise RuntimeError("lgbm_lag: not enough finite train rows")
    model = LGBMRegressor(
        n_estimators=80,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X, y)
    pred_ret = model.predict(te[feats].to_numpy())
    pred_px = te["px_lag1"].to_numpy() * (1.0 + pred_ret)
    out = te[["security_id", "trade_date", "true_price"]].copy()
    out["pred"] = pred_px
    out = out.rename(columns={"trade_date": "date"})
    out["model"] = "lgbm_lag"
    return out


def baseline_four_agent(formed_test: pd.DataFrame) -> pd.DataFrame:
    out = formed_test[
        ["security_id", "trade_date", "true_price", "fitted_price"]
    ].copy()
    out = out.rename(columns={"fitted_price": "pred", "trade_date": "date"})
    out["model"] = "four_agent_fit"
    return out
