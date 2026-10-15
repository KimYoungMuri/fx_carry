"""Return-distribution statistics. All inputs are daily arithmetic excess returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 260  # observations per year (weekday data; sample averages 260.8)


def wealth(r: pd.Series) -> pd.Series:
    return (1.0 + r).cumprod()


def drawdown_info(r: pd.Series) -> dict:
    w = wealth(r)
    w0 = pd.concat([pd.Series([1.0], index=[pd.NaT]), w])  # include starting wealth of 1
    peak = w0.cummax()
    dd = w0 / peak - 1.0
    trough = dd.iloc[1:].idxmin()
    mdd = float(dd.loc[trough])
    pre = w0.loc[:trough]
    peak_val = pre.max()
    peak_date = pre.idxmax()
    after = w.loc[trough:]
    rec = after[after >= peak_val]
    rec_date = rec.index[0] if len(rec) else None
    return {
        "max_drawdown": mdd,
        "dd_peak": "start" if pd.isna(peak_date) else peak_date.date().isoformat(),
        "dd_trough": trough.date().isoformat(),
        "dd_recovery": rec_date.date().isoformat() if rec_date is not None else "not recovered",
        "dd_peak_to_trough_obs": int(len(w0.loc[peak_date:trough]) - 1) if not pd.isna(peak_date) else int(len(w.loc[:trough])),
    }


def drawdown_series(r: pd.Series) -> pd.Series:
    w = wealth(r)
    return w / np.maximum(w.cummax(), 1.0) - 1.0


def summarize(r: pd.Series) -> dict:
    r = r.dropna()
    n = len(r)
    mu = r.mean()
    sd = r.std(ddof=1)
    w_end = float(wealth(r).iloc[-1])
    g_daily = w_end ** (1.0 / n) - 1.0
    q05 = r.quantile(0.05)
    out = {
        "start": r.index[0].date().isoformat(),
        "end": r.index[-1].date().isoformat(),
        "n_obs": n,
        "mean_daily": mu,
        "geo_mean_daily": g_daily,
        "ann_arith_mean": mu * ANN,
        "ann_geo_return": (1.0 + g_daily) ** ANN - 1.0,
        "ann_vol": sd * np.sqrt(ANN),
        "sharpe": (mu * ANN) / (sd * np.sqrt(ANN)),
        "sortino": (mu * ANN) / (np.sqrt((np.minimum(r, 0.0) ** 2).mean()) * np.sqrt(ANN)),
        "skew": r.skew(),
        "excess_kurt": r.kurt(),
        "final_wealth": w_end,
        "var95_daily": -q05,
        "es95_daily": -r[r <= q05].mean(),
        "worst_day": r.min(),
        "worst_day_date": r.idxmin().date().isoformat(),
        "best_day": r.max(),
        "frac_positive": (r > 0).mean(),
        "frac_zero": (r == 0).mean(),
    }
    out.update(drawdown_info(r))
    out["calmar"] = out["ann_geo_return"] / abs(out["max_drawdown"])
    return out


def summary_table(returns: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: summarize(returns[c]) for c in returns}).T


def select_best(table: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Transparent rule: average rank of Sharpe (high), Calmar (high) and daily 95% ES (low); ties by Sharpe."""
    ranks = pd.DataFrame({
        "rank_sharpe": table["sharpe"].astype(float).rank(ascending=False),
        "rank_calmar": table["calmar"].astype(float).rank(ascending=False),
        "rank_es95": table["es95_daily"].astype(float).rank(ascending=True),
    })
    ranks["avg_rank"] = ranks.mean(axis=1)
    ranks["sharpe"] = table["sharpe"].astype(float)
    ranks = ranks.sort_values(["avg_rank", "sharpe"], ascending=[True, False])
    return ranks.index[0], ranks
