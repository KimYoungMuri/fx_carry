"""Volatility conditioning: switch carry exposure off when volatility is high.

Rule (fixed for every variable, not optimised):
  q_t   = 75th percentile of the variable's observations strictly before the
          latest observation available at t (expanding window, >= WARMUP obs)
  s_t   = 1 if v_t <= q_t else 0          (v_t: latest observed value at close of t)
  R^c_{t+1} = s_t * R_{t+1}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILE = 0.75
WARMUP = 252


def expanding_threshold(raw: pd.Series, index: pd.DatetimeIndex, q: float = QUANTILE,
                        warmup: int = WARMUP) -> pd.Series:
    """Threshold computed on observed values only, excluding the current observation."""
    obs = raw.dropna()
    thr = obs.expanding(min_periods=warmup).quantile(q).shift(1)
    first = obs.index[0]
    out = thr.reindex(index)
    out.loc[first:] = out.loc[first:].ffill()
    return out


def latest_value(raw: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    out = raw.reindex(index)
    first = raw.first_valid_index()
    out.loc[first:] = out.loc[first:].ffill()
    return out


def signal(raw: pd.Series, index: pd.DatetimeIndex, q: float = QUANTILE, warmup: int = WARMUP,
           extra_lag: int = 0) -> pd.DataFrame:
    v = latest_value(raw, index)
    thr = expanding_threshold(raw, index, q, warmup)
    s = pd.Series(np.where(v <= thr, 1.0, 0.0), index=index)
    s[thr.isna() | v.isna()] = np.nan
    if extra_lag:
        s = s.shift(extra_lag)
    return pd.DataFrame({"value": v, "threshold": thr, "invested": s})


def conditioned_returns(base: pd.Series, invested_at_t: pd.Series) -> pd.Series:
    """Exposure decided at t multiplies the return over (t, t+1] (indexed at t+1)."""
    s = invested_at_t.shift(1).reindex(base.index)
    return (s * base).where(s.notna())


def exposure_stats(invested_at_t: pd.Series, dates: pd.DatetimeIndex) -> dict:
    s = invested_at_t.shift(1).reindex(dates).dropna()
    switches = int((s.diff().abs() > 0).sum())
    off = (s == 0).astype(int)
    spells = (off.diff() == 1).sum() + int(off.iloc[0] == 1)
    return {
        "frac_invested": s.mean(),
        "n_switches": switches,
        "n_off_spells": int(spells),
        "avg_off_spell_days": float(off.sum() / spells) if spells else 0.0,
    }


def realized_vol(price: pd.Series, window: int = 21, ann: int = 260) -> pd.Series:
    """Annualised realised vol (vol points) of daily log returns over the last `window` obs ending at t."""
    lr = np.log(price / price.shift(1))
    return lr.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(ann) * 100.0
