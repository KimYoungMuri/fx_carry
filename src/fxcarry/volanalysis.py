"""Part (c): contemporaneous and predictive relations between carry returns and volatility."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .metrics import ANN


def _ols_hac(y: pd.Series, x: pd.Series, lags: int) -> dict:
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    res = sm.OLS(df["y"], sm.add_constant(df["x"])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"beta": res.params["x"], "t": res.tvalues["x"], "r2": res.rsquared, "n": int(res.nobs)}


def vol_changes(raw: pd.Series, filled: pd.Series) -> pd.Series:
    """Change in the vol level over (t, t+1]; NaN where the level was not observed at t+1."""
    return filled.diff().where(raw.notna())


def contemporaneous(base: pd.Series, raw: pd.DataFrame, filled: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for v in raw:
        dv = vol_changes(raw[v], filled[v]).reindex(base.index)
        dlog = np.log(filled[v]).diff().where(raw[v].notna()).reindex(base.index)
        z = (dv - dv.mean()) / dv.std()
        reg = _ols_hac(base, z, 5)
        ok = pd.concat([base, dv], axis=1).dropna()
        rows[v] = {
            "corr_R_dVol": ok.iloc[:, 0].corr(ok.iloc[:, 1]),
            "corr_R_dlogVol": base.corr(dlog),
            "beta_per_1sd_dVol": reg["beta"], "t_HAC": reg["t"], "R2": reg["r2"], "n": reg["n"],
            "frac_zero_dVol": float((ok.iloc[:, 1] == 0).mean()),
        }
    return pd.DataFrame(rows).T


def predictive(base: pd.Series, filled: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Regress future carry returns on vol information known at t."""
    rows = {}
    fwd1 = base                                         # R_{t+1}, indexed t+1
    fwdH = base[::-1].rolling(horizon, min_periods=horizon).sum()[::-1]  # sum R_{t+1..t+H}, indexed t+1
    for v in filled:
        lvl = filled[v].shift(1).reindex(base.index)    # v_t aligned with R_{t+1}
        dlv = filled[v].diff().shift(1).reindex(base.index)
        zl = (lvl - lvl.mean()) / lvl.std()
        zd = (dlv - dlv.mean()) / dlv.std()
        r1 = _ols_hac(fwd1, zl, 5)
        rH = _ols_hac(fwdH, zl, horizon)
        rd = _ols_hac(fwd1, zd, 5)
        rows[v] = {
            "corr_level_t_R_t+1": fwd1.corr(lvl),
            "beta1d_level_bp": r1["beta"] * 1e4, "t1d_level": r1["t"],
            f"corr_level_t_R_{horizon}d": fwdH.corr(lvl),
            f"beta{horizon}d_level_pct": rH["beta"] * 100, f"t{horizon}d_level": rH["t"], f"R2_{horizon}d": rH["r2"],
            "corr_dVol_t_R_t+1": fwd1.corr(dlv),
            "beta1d_dVol_bp": rd["beta"] * 1e4, "t1d_dVol": rd["t"],
        }
    return pd.DataFrame(rows).T


def regimes(base: pd.Series, filled: pd.DataFrame) -> pd.DataFrame:
    """Full-sample terciles of v_t (descriptive only), statistics of R_{t+1} within each."""
    out = []
    for v in filled:
        lvl = filled[v].shift(1).reindex(base.index)
        ok = pd.concat([base.rename("R"), lvl.rename("v")], axis=1).dropna()
        ok["regime"] = pd.qcut(ok["v"], 3, labels=["low", "medium", "high"])
        for g, sub in ok.groupby("regime", observed=True):
            r = sub["R"]
            q = r.quantile(0.05)
            out.append({
                "vol": v, "regime": g, "n": len(r), "v_min": sub["v"].min(), "v_max": sub["v"].max(),
                "ann_mean": r.mean() * ANN, "ann_vol": r.std() * np.sqrt(ANN),
                "sharpe": r.mean() / r.std() * np.sqrt(ANN), "es95_daily": -r[r <= q].mean(),
                "worst_day": r.min(),
            })
    return pd.DataFrame(out)
