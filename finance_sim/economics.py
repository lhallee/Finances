"""Shared economic draws with explicit assumptions and optional historical blocks."""

from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from .configuration import RunConfig
from .reference import MACROS


def monthly_dates(config: RunConfig) -> pd.DatetimeIndex:
    return pd.date_range("2026-10-01", f"{config.end_year}-12-01", freq="MS")


def economic_paths(config: RunConfig, macro: str, paths: int | None = None) -> np.ndarray:
    """Return (p, t, 8): equity, inflation, home, cash, wage, job U, repair U, business U."""
    p = paths or config.paths
    t = len(monthly_dates(config))
    rng = np.random.default_rng(config.seed)
    normals = rng.standard_t(7, size=(p, t, 3)) / np.sqrt(7 / 5)  # (p, t, 3)
    correlation = np.array([[1, -.15, .35], [-.15, 1, .20], [.35, .20, 1]])  # (3, 3)
    normals = normals @ np.linalg.cholesky(correlation).T  # (p, t, 3)
    draws = np.zeros((p, t, 8), dtype=np.float64)  # (p, t, 8)
    ec = config.economics
    if ec.return_mode == "historical":
        if not ec.history_csv:
            raise ValueError("Historical mode needs history_csv")
        history = pd.read_csv(ec.history_csv, parse_dates=["date"]).sort_values("date")
        columns = ["equity_return", "inflation", "home_return", "cash_rate"]
        observed = history[columns].to_numpy(dtype=float)  # (h, 4)
        if not np.isfinite(observed).all() or len(observed) < ec.block_months * 2:
            raise ValueError("Historical series must be finite and span at least two blocks")
        periods = history.date.dt.to_period("M")
        if periods.duplicated().any():
            raise ValueError("Historical observations must be unique months")
        if history.date.max() >= pd.Timestamp("2026-10-01"):
            raise ValueError("Calibration data must predate the simulation to avoid lookahead")
        for start in range(0, t, ec.block_months):
            length = min(ec.block_months, t - start)
            ordinal = periods.astype("int64").to_numpy()  # (h,)
            valid_starts = np.array([i for i in range(len(observed)-length+1) if ordinal[i+length-1]-ordinal[i] == length-1])  # (v,)
            if not len(valid_starts):
                raise ValueError("No contiguous historical blocks of the requested length")
            indexes = rng.choice(valid_starts, size=p)  # (p,)
            draws[:, start:start + length, :4] = observed[indexes[:, None] + np.arange(length)[None, :]]  # (p, length, 4)
    elif ec.return_mode == "parametric":
        draws[:, :, 0] = np.expm1((np.log1p(ec.annual_return) - ec.annual_volatility ** 2 / 2) / 12 + ec.annual_volatility / np.sqrt(12) * normals[:, :, 0])  # (p, t)
        draws[:, :, 1] = ec.annual_inflation / 12 + .012 / np.sqrt(12) * normals[:, :, 1]  # (p, t)
        draws[:, :, 2] = ec.annual_home_growth / 12 + .08 / np.sqrt(12) * normals[:, :, 2]  # (p, t)
        draws[:, :, 3] = np.maximum(0, config.household.cash_apy + np.cumsum(draws[:, :, 1] - ec.annual_inflation / 12, axis=1) * .2)  # (p, t)
    else:
        raise ValueError("return_mode must be parametric or historical")
    return_offset, inflation_offset, unemployment_multiplier = MACROS[macro]
    for month in range(t):
        shock = 1 if macro != "recession" or 15 <= month < 39 else 0
        draws[:, month, 0] = np.maximum(-.95, draws[:, month, 0] + shock * return_offset / 12)  # (p,)
        draws[:, month, 1] += shock * inflation_offset / 12  # (p,)
        draws[:, month, 2] += shock * return_offset / 30  # (p,)
    draws[:, :, 4] = config.household.annual_raise / 12 + draws[:, :, 1] - ec.annual_inflation / 12  # (p, t)
    if macro in ("ai_substantial", "ai_extreme"):
        draws[:, :, 4] -= .02 / 12 if macro == "ai_substantial" else .05 / 12  # (p, t)
    draws[:, :, 5:8] = rng.random((p, t, 3))  # (p, t, 3)
    return draws  # (p, t, 8)


def history_fingerprint(config: RunConfig) -> str | None:
    if not config.economics.history_csv:
        return None
    from pathlib import Path
    return hashlib.sha256(Path(config.economics.history_csv).read_bytes()).hexdigest()
