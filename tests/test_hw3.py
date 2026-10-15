"""Verification tests for the HW3 carry pipeline.  Run: .venv/bin/python -m pytest -q tests"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fxcarry import conditioning as cond  # noqa: E402
from fxcarry.data import CCYS, INVERTED, UNIVERSE, filled_vols, load_dataset  # noqa: E402
from fxcarry.metrics import drawdown_info, summarize, wealth  # noqa: E402
from fxcarry.portfolios import (all_specs, build_weights, carry_signal, portfolio_returns,  # noqa: E402
                                validate_weights, weights_on_date)
from fxcarry.returns import audjpy_cross, excess_returns, spot_returns, year_fraction  # noqa: E402

DATA = ROOT / "hw3.data.carry.FX.dly.xlsx"


@pytest.fixture(scope="module")
def ds():
    return load_dataset(DATA)


@pytest.fixture(scope="module")
def sr(ds):
    return spot_returns(ds.usd_per_ccy)


def test_dates_chronological_unique(ds):
    assert ds.dates.is_monotonic_increasing and not ds.dates.has_duplicates
    assert ds.audit["layout_verified"]
    assert set(ds.dates.dayofweek) <= {0, 1, 2, 3, 4}


def test_quote_inversion(ds):
    assert INVERTED == ["JPY", "CAD", "CHF", "NOK", "SEK"]
    d = pd.Timestamp("2008-10-24")
    assert ds.usd_per_ccy.loc[d, "JPY"] == pytest.approx(1 / 94.32)
    assert ds.usd_per_ccy.loc[d, "AUD"] == pytest.approx(0.6225)


def test_returns_match_workbook_and_negation_is_wrong(ds, sr):
    for c in UNIVERSE:
        assert np.nanmax(np.abs(sr[c] - ds.provided_ror[c])) < 1e-12
    q = ds.quotes["JPY"]
    exact = 1 / (1 + (q / q.shift(1) - 1)) - 1
    assert np.allclose(exact.dropna(), sr["JPY"].dropna(), atol=1e-14)
    assert np.nanmax(np.abs(-(q / q.shift(1) - 1) - sr["JPY"])) > 1e-4


def test_cross_rate(ds):
    x = audjpy_cross(ds.usd_per_ccy)
    d = pd.Timestamp("2008-10-24")
    assert x.loc[d] == pytest.approx(0.6225 * 94.32)


def test_rate_units_and_accrual(ds):
    r = ds.rates["1M"]
    assert r.abs().max().max() < 0.2           # decimals, not percent
    delta = year_fraction(ds.dates)
    fri, mon = pd.Timestamp("2008-10-24"), pd.Timestamp("2008-10-27")
    assert delta.loc[mon] == pytest.approx(3 / 360)
    assert delta.loc[pd.Timestamp("2008-10-28")] == pytest.approx(1 / 360)
    assert fri in delta.index


def test_excess_return_formula(ds, sr):
    delta = year_fraction(ds.dates)
    e = excess_returns(sr, ds.rates["3M"], delta)
    t, t1 = pd.Timestamp("2008-10-24"), pd.Timestamp("2008-10-27")
    i_a, i_u = ds.rates["3M"].loc[t, "AUD"], ds.rates["3M"].loc[t, "USD"]
    r = 0.6013 / 0.6225 - 1
    assert e.loc[t1, "AUD"] == pytest.approx((1 + r) * (1 + i_a * 3 / 360) - (1 + i_u * 3 / 360), abs=1e-15)
    assert (e["USD"].dropna() == 0).all()


def test_weights_valid_all_strategies(ds):
    for t, k, m in all_specs():
        c = carry_signal(ds.rates[t])
        w = build_weights(c, k, m)
        assert validate_weights(w, c, k)["all_ok"], (t, k, m)


def test_prop_weights_nonnegative_when_all_same_sign():
    row = pd.Series([5.0, 4.0, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3], index=UNIVERSE)  # all positive carry
    w = weights_on_date(row, 3, "PROP")
    assert (w[w != 0].iloc[:3] > 0).all() and (w.iloc[-3:] < 0).all()
    assert w[w > 0].sum() == pytest.approx(1) and w[w < 0].sum() == pytest.approx(-1)
    flat = pd.Series(np.zeros(10), index=UNIVERSE)
    wf = weights_on_date(flat, 2, "PROP")
    assert wf[wf > 0].sum() == pytest.approx(1)


def test_rank_weights():
    row = pd.Series(np.arange(10, 0, -1, dtype=float), index=UNIVERSE)
    w = weights_on_date(row, 3, "RANK")
    assert list(w.iloc[:3].round(6)) == [0.5, round(1 / 3, 6), round(1 / 6, 6)]
    assert list(w.iloc[-3:].round(6)) == [round(-1 / 6, 6), round(-1 / 3, 6), -0.5]


def test_weight_lag_and_no_lookahead(ds, sr):
    rates = ds.rates["3M"].copy()
    delta = year_fraction(ds.dates)
    c = carry_signal(rates)
    w = build_weights(c, 3, "EQ")
    R = portfolio_returns(w, excess_returns(sr, rates, delta))
    t, t1 = ds.dates[1500], ds.dates[1501]
    e = excess_returns(sr, rates, delta)
    assert R.loc[t1] == pytest.approx((w.loc[t] * e.loc[t1]).sum(), abs=1e-15)
    # perturbing data after t must not change weights or returns up to t
    cut = ds.dates[1500]
    rates2 = rates.copy()
    rates2.loc[rates2.index > cut] = rates2.loc[rates2.index > cut][::-1].values
    sr2 = sr.copy()
    sr2.loc[sr2.index > cut] = 0.05
    w2 = build_weights(carry_signal(rates2), 3, "EQ")
    R2 = portfolio_returns(w2, excess_returns(sr2, rates2, delta))
    assert w2.loc[:cut].equals(w.loc[:cut])
    assert np.allclose(R2.loc[:cut], R.loc[:cut])


def test_threshold_no_lookahead(ds):
    raw = ds.vols_raw["AY3M"]
    s = cond.signal(raw, ds.dates)
    cut = ds.dates[1800]
    raw2 = raw.copy()
    raw2.loc[raw2.index > cut] = 500.0
    s2 = cond.signal(raw2, ds.dates)
    pd.testing.assert_frame_equal(s.loc[:cut], s2.loc[:cut])
    # threshold excludes the current observation
    obs = raw.dropna()
    d = obs.index[400]
    assert s.loc[d, "threshold"] == pytest.approx(obs.iloc[:400].quantile(0.75))
    assert np.isnan(s["threshold"].iloc[250])


def test_vix_not_filled_before_start(ds):
    f = filled_vols(ds.vols_raw)
    assert np.isnan(f.loc[pd.Timestamp("1999-01-01"), "VIX"])
    assert f["VIX"].loc[pd.Timestamp("1999-01-04"):].notna().all()
    s = cond.signal(ds.vols_raw["VIX"], ds.dates)
    assert np.isnan(s["value"].iloc[0])


def test_conditioned_return_timing():
    idx = pd.bdate_range("2020-01-01", periods=5)
    base = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=idx)
    inv = pd.Series([1.0, 0.0, 1.0, 1.0, 0.0], index=idx)
    out = cond.conditioned_returns(base, inv)
    assert np.isnan(out.iloc[0])
    assert list(out.iloc[1:]) == [0.02, 0.0, 0.04, 0.05]


def test_compounding_and_drawdown():
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series([0.10, -0.50, 0.20, 1.00], index=idx)
    assert wealth(r).iloc[-1] == pytest.approx(1.1 * 0.5 * 1.2 * 2.0)
    dd = drawdown_info(r)
    assert dd["max_drawdown"] == pytest.approx(-0.5)
    assert dd["dd_trough"] == idx[1].date().isoformat()
    assert dd["dd_recovery"] == idx[3].date().isoformat()
    s = summarize(r)
    assert s["geo_mean_daily"] == pytest.approx((1.1 * 0.5 * 1.2 * 2.0) ** 0.25 - 1)
    assert s["ann_arith_mean"] == pytest.approx(r.mean() * 260)


def test_no_artificial_zero_returns(ds, sr):
    delta = year_fraction(ds.dates)
    for t, k, m in all_specs():
        w = build_weights(carry_signal(ds.rates[t]), k, m)
        R = portfolio_returns(w, excess_returns(sr, ds.rates[t], delta))
        assert (R == 0).sum() == 0 and R.notna().all()


def test_realized_vol_uses_past_only(ds):
    x = audjpy_cross(ds.usd_per_ccy)
    rv = cond.realized_vol(x, 21)
    d = 100
    lr = np.log(x / x.shift(1)).iloc[d - 20:d + 1]
    assert rv.iloc[d] == pytest.approx(lr.std(ddof=1) * np.sqrt(260) * 100)
    assert rv.iloc[:21].isna().all() and not np.isnan(rv.iloc[21])
