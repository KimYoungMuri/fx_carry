"""Spot returns and funded excess returns.

Index convention: a return indexed at date t+1 covers the interval (t, t+1] and
uses only rates observed at t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CCYS, DAYCOUNT, INVERTED, UNIVERSE


def spot_returns(usd_per_ccy: pd.DataFrame) -> pd.DataFrame:
    """r_{i,t+1} = X_{i,t+1}/X_{i,t} - 1 on USD-per-currency levels; USD itself is 0."""
    r = usd_per_ccy / usd_per_ccy.shift(1) - 1.0
    r["USD"] = 0.0
    r.iloc[0] = np.nan
    return r[UNIVERSE]


def year_fraction(dates: pd.DatetimeIndex, daycount: float = DAYCOUNT) -> pd.Series:
    """Delta_t = calendar days from t to t+1 / daycount, indexed at t+1."""
    d = pd.Series(dates, index=dates)
    return (d - d.shift(1)).dt.days / daycount


def excess_returns(spot_ret: pd.DataFrame, rates: pd.DataFrame, delta: pd.Series) -> pd.DataFrame:
    """e_{i,t+1} = (1+r_{i,t+1})(1+i_{i,t} D_t) - (1+i_{USD,t} D_t).

    Return in USD on 1 USD lent in currency i, net of 1 USD borrowed in USD.
    A short (negative weight) is the mirror trade: borrow currency i, lend USD.
    """
    i_lag = rates.shift(1)
    gross_f = (1.0 + spot_ret).mul(1.0 + i_lag.mul(delta, axis=0))
    gross_usd = 1.0 + i_lag["USD"] * delta
    e = gross_f.sub(gross_usd, axis=0)
    e["USD"] = 0.0
    e.iloc[0] = np.nan
    return e[UNIVERSE]


def compare_provided(spot_ret: pd.DataFrame, provided: pd.DataFrame) -> pd.DataFrame:
    """Max abs difference between computed returns and workbook returns, plus naive-negation error."""
    rows = []
    for c in CCYS + ["USD"]:
        diff = (spot_ret[c] - provided[c]).abs().max()
        rows.append({"currency": c, "inverted_quote": c in INVERTED, "max_abs_diff_vs_workbook": diff})
    return pd.DataFrame(rows).set_index("currency")


def naive_negation_error(quotes: pd.DataFrame, spot_ret: pd.DataFrame) -> pd.Series:
    out = {}
    for c in INVERTED:
        q_ret = quotes[c] / quotes[c].shift(1) - 1.0
        out[c] = (-q_ret - spot_ret[c]).abs().max()
    return pd.Series(out, name="max_abs_error_of_negating_quote_return")


def audjpy_cross(usd_per_ccy: pd.DataFrame) -> pd.Series:
    """JPY per AUD = X_AUD / X_JPY."""
    return (usd_per_ccy["AUD"] / usd_per_ccy["JPY"]).rename("AUDJPY")
