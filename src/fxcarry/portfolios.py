"""Daily carry portfolio construction.

Carry of currency i at t: c_{i,t} = i_{i,t} - i_{USD,t} (USD carry is 0).
Long the k highest-carry members of the universe, short the k lowest.
Every portfolio has long weights summing to +1 and short weights summing to -1.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .data import TENORS, UNIVERSE

METHODS = ["EQ", "RANK", "PROP"]
METHOD_LABEL = {"EQ": "equal", "RANK": "rank-based", "PROP": "proportional-to-carry"}
KS = [2, 3]


def strategy_name(tenor: str, k: int, method: str) -> str:
    return f"{tenor}-K{k}-{method}"


def all_specs() -> list[tuple[str, int, str]]:
    return list(product(TENORS, KS, METHODS))


def carry_signal(rates: pd.DataFrame, universe: list[str] = UNIVERSE) -> pd.DataFrame:
    return rates[universe].sub(rates["USD"], axis=0)


def _side_magnitudes(vals: np.ndarray, method: str, ref: float, side: str) -> np.ndarray:
    """Nonnegative magnitudes, ordered most-extreme first, normalised to sum 1."""
    k = len(vals)
    if method == "EQ":
        m = np.ones(k)
    elif method == "RANK":
        m = np.arange(k, 0, -1, dtype=float)       # most extreme gets k, least gets 1
    elif method == "PROP":
        m = (vals - ref) if side == "long" else (ref - vals)
        m = np.clip(m, 0.0, None)
        if m.sum() <= 1e-15:                       # all tied with the median: fall back to equal
            m = np.ones(k)
    else:
        raise ValueError(method)
    return m / m.sum()


def weights_on_date(carry_row: pd.Series, k: int, method: str) -> pd.Series:
    names = list(carry_row.index)
    vals = carry_row.to_numpy(dtype=float)
    # deterministic tie-break: universe order
    order = sorted(range(len(vals)), key=lambda j: (-vals[j], j))
    longs = order[:k]
    shorts = order[::-1][:k]                       # lowest first
    ref = float(np.median(vals))
    w = np.zeros(len(vals))
    w[longs] = _side_magnitudes(vals[longs], method, ref, "long")
    w[shorts] = -_side_magnitudes(vals[shorts], method, ref, "short")
    return pd.Series(w, index=names)


def build_weights(carry: pd.DataFrame, k: int, method: str) -> pd.DataFrame:
    """Weights decided with information at t (row t), applied to the return over (t, t+1]."""
    rows = [weights_on_date(carry.iloc[i], k, method).to_numpy() for i in range(len(carry))]
    return pd.DataFrame(rows, index=carry.index, columns=carry.columns)


def portfolio_returns(weights: pd.DataFrame, excess: pd.DataFrame) -> pd.Series:
    """R_{t+1} = sum_i w_{i,t} e_{i,t+1}; indexed at t+1, first date dropped."""
    w_lag = weights.shift(1)
    r = (w_lag * excess[weights.columns]).sum(axis=1, min_count=1)
    return r.iloc[1:]


def validate_weights(weights: pd.DataFrame, carry: pd.DataFrame, k: int, tol: float = 1e-12) -> dict:
    pos = weights.where(weights > 0, 0.0)
    neg = weights.where(weights < 0, 0.0)
    n_long = (weights > tol).sum(axis=1)
    n_short = (weights < -tol).sum(axis=1)
    long_sum = pos.sum(axis=1)
    short_sum = neg.sum(axis=1)
    c = carry[weights.columns]
    min_long_carry = c.where(weights > tol).min(axis=1)
    max_short_carry = c.where(weights < -tol).max(axis=1)
    max_unselected = c.where(weights.abs() <= tol).max(axis=1)
    min_unselected = c.where(weights.abs() <= tol).min(axis=1)
    checks = {
        "days": len(weights),
        "long_count_ok": bool((n_long == k).all()),
        "short_count_ok": bool((n_short == k).all()),
        "long_sum_ok": bool(np.allclose(long_sum, 1.0, atol=1e-12)),
        "short_sum_ok": bool(np.allclose(short_sum, -1.0, atol=1e-12)),
        "longs_carry_ge_shorts": bool((min_long_carry >= max_short_carry).all()),
        "longs_are_top_k": bool((min_long_carry >= max_unselected - tol).all()),
        "shorts_are_bottom_k": bool((max_short_carry <= min_unselected + tol).all()),
    }
    checks["all_ok"] = all(v for kk, v in checks.items() if kk != "days")
    return checks
