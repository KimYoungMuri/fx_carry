"""Run the complete HW3 analysis and write tables, histories, figures and results.json.

Usage (from hw3/):  .venv/bin/python scripts/run_all.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fxcarry import conditioning as cond  # noqa: E402
from fxcarry.data import (CCYS, DAYCOUNT, INVERTED, QUOTE, TENORS, UNIVERSE, VOL_DESCRIPTION, VOLS,  # noqa: E402
                          filled_vols, load_dataset, staleness_by_year)
from fxcarry.metrics import ANN, drawdown_series, select_best, summarize, summary_table, wealth  # noqa: E402
from fxcarry.portfolios import (KS, METHOD_LABEL, METHODS, all_specs, build_weights, carry_signal,  # noqa: E402
                                portfolio_returns, strategy_name, validate_weights)
from fxcarry.returns import (audjpy_cross, compare_provided, excess_returns, naive_negation_error,  # noqa: E402
                             spot_returns, year_fraction)
from fxcarry.volanalysis import contemporaneous, predictive, regimes  # noqa: E402

DATA = ROOT / "hw3.data.carry.FX.dly.xlsx"
OUT = ROOT / "outputs"
TAB, FIG, HIST = OUT / "tables", OUT / "figures", OUT / "histories"
for p in (TAB, FIG, HIST):
    p.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.grid": True,
                     "font.family": "serif", "mathtext.fontset": "cm",
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
METHOD_COLOR = {"EQ": "#1f77b4", "RANK": "#d62728", "PROP": "#2ca02c"}
VOL_COLOR = {"AY1wk": "#9467bd", "AY3M": "#d62728", "AY6M": "#ff7f0e", "JPY3M": "#17becf",
             "AUD3M": "#2ca02c", "VIX": "#1f77b4", "RV21_AUDJPY": "#8c564b"}
EPISODES = {
    "Aug-2007 carry unwind": ("2007-07-20", "2007-08-31"),
    "Sep-Dec 2008 crisis": ("2008-09-01", "2008-12-31"),
    "Apr-Jun 2010 euro stress": ("2010-04-15", "2010-06-04"),
}
log: list[str] = []


def note(msg: str) -> None:
    print(msg)
    log.append(msg)


def savefig(fig, name, layout=True):
    if layout:
        fig.tight_layout(rect=(0, 0, 1, 0.95) if fig._suptitle is not None else (0, 0, 1, 1))
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return x.date().isoformat()
    return x


# ----------------------------------------------------------------------------- data
ds = load_dataset(DATA)
note(f"Loaded {ds.audit['file']} sheet={ds.audit['sheet']} rows={ds.audit['rows']} "
     f"{ds.audit['start'].date()}..{ds.audit['end'].date()} layout_verified={ds.audit['layout_verified']}")
dates = ds.dates
vols_f = filled_vols(ds.vols_raw)

dictionary = []
for c in CCYS:
    dictionary.append({"column": c, "content": f"{c} spot, quoted {QUOTE[c]}",
                       "used_as": "inverted to USD per " + c if c in INVERTED else "USD per " + c + " (as quoted)",
                       "units": "price", "missing": int(ds.quotes[c].isna().sum())})
for c in UNIVERSE:
    dictionary.append({"column": f"{c} ROR", "content": f"workbook 1-day return, long {c} vs USD" if c != "USD" else "USD return (identically 0)",
                       "used_as": "cross-check only", "units": "decimal", "missing": int(ds.provided_ror[c].isna().sum())})
for t in TENORS:
    for c in UNIVERSE:
        dictionary.append({"column": f"{c} {t}", "content": f"{c} {t} interest rate", "used_as": f"carry ranking + daily accrual ({t} versions)",
                           "units": "% per annum (converted to decimal)", "missing": int(ds.rates[t][c].isna().sum())})
for v in VOLS:
    dictionary.append({"column": v, "content": VOL_DESCRIPTION[v], "used_as": "conditioning variable",
                       "units": "vol points (% p.a.)", "missing": int(ds.vols_raw[v].isna().sum())})
pd.DataFrame(dictionary).to_csv(TAB / "data_dictionary.csv", index=False)

stale = staleness_by_year(ds.vols_raw)
stale.to_csv(TAB / "vol_staleness_by_year.csv")
rate_stale = pd.DataFrame({t: ds.rates[t].diff().eq(0).mean() for t in TENORS})
rate_stale.to_csv(TAB / "rate_unchanged_share.csv")
vol_desc = ds.vols_raw.describe().T
vol_desc["first_obs"] = [ds.vols_raw[v].first_valid_index().date().isoformat() for v in VOLS]
vol_desc.to_csv(TAB / "vol_descriptive.csv")

# ----------------------------------------------------------------------------- returns
sr = spot_returns(ds.usd_per_ccy)
cmp_ = compare_provided(sr, ds.provided_ror)
neg_err = naive_negation_error(ds.quotes, sr)
cmp_["naive_negation_max_error"] = neg_err.reindex(cmp_.index)
cmp_.to_csv(TAB / "return_check_vs_workbook.csv")
note(f"Computed vs workbook returns: max abs diff = {cmp_['max_abs_diff_vs_workbook'].max():.3e}; "
     f"negating raw-quote returns would err by up to {neg_err.max():.5f}")
delta = year_fraction(dates)
excess = {t: excess_returns(sr, ds.rates[t], delta) for t in TENORS}
carry = {t: carry_signal(ds.rates[t]) for t in TENORS}

# ----------------------------------------------------------------------------- (a) baselines
weights, rets, validation = {}, {}, {}
for t, k, m in all_specs():
    name = strategy_name(t, k, m)
    w = build_weights(carry[t], k, m)
    weights[name] = w
    validation[name] = validate_weights(w, carry[t], k)
    rets[name] = portfolio_returns(w, excess[t])
names = list(rets)
R = pd.DataFrame(rets)
pd.DataFrame(validation).T.to_csv(TAB / "weight_validation.csv")
assert all(v["all_ok"] for v in validation.values()), validation
note("Weight validation passed for all 12 strategies on every day (counts, signs, +1/-1 sums, top/bottom-k).")

wide = pd.concat({n: w for n, w in weights.items()}, axis=1)
wide.columns = [f"{a}|{b}" for a, b in wide.columns]
wide.index.name = "signal_date"
wide.to_csv(HIST / "baseline_weights_daily.csv", float_format="%.8f")
R.index.name = "return_date"
R.to_csv(HIST / "baseline_returns_daily.csv", float_format="%.10f")
for t in TENORS:
    carry[t].mul(100).to_csv(HIST / f"carry_vs_usd_{t}_pct.csv", float_format="%.5f")

EXAMPLE_DATE = pd.Timestamp("2007-06-29")
next_date = dates[dates.get_loc(EXAMPLE_DATE) + 1]
ex_rows = []
for n in names:
    w = weights[n].loc[EXAMPLE_DATE]
    longs = w[w > 0].sort_values(ascending=False)
    shorts = w[w < 0].sort_values()
    ex_rows.append({"strategy": n,
                    "long": ", ".join(f"{c} {v:+.5f}" for c, v in longs.items()),
                    "short": ", ".join(f"{c} {v:+.5f}" for c, v in shorts.items()),
                    "long_sum": longs.sum(), "short_sum": shorts.sum()})
ex = pd.DataFrame(ex_rows).set_index("strategy")
ex.to_csv(TAB / "a_example_weights_2007-06-29.csv")
ex_carry = pd.DataFrame({t: carry[t].loc[EXAMPLE_DATE] * 100 for t in TENORS})
ex_rates = pd.DataFrame({t: ds.rates[t].loc[EXAMPLE_DATE] * 100 for t in TENORS})
ex_snapshot = pd.concat({"rate_pct": ex_rates, "carry_vs_usd_pct": ex_carry}, axis=1)
ex_snapshot.to_csv(TAB / "a_example_rates_2007-06-29.csv")

LAST_DATE = dates[-2]  # last date whose weights are applied to an observed return
ex_last = []
for n in names:
    w = weights[n].loc[LAST_DATE]
    ex_last.append({"strategy": n,
                    "long": ", ".join(f"{c} {v:+.5f}" for c, v in w[w > 0].sort_values(ascending=False).items()),
                    "short": ", ".join(f"{c} {v:+.5f}" for c, v in w[w < 0].sort_values().items())})
pd.DataFrame(ex_last).set_index("strategy").to_csv(TAB / f"a_example_weights_{LAST_DATE.date()}.csv")

sel_rows = []
for t in TENORS:
    for k in KS:
        w = weights[strategy_name(t, k, "EQ")]
        for c in UNIVERSE:
            sel_rows.append({"tenor": t, "K": k, "currency": c,
                             "pct_days_long": (w[c] > 0).mean() * 100, "pct_days_short": (w[c] < 0).mean() * 100})
sel = pd.DataFrame(sel_rows)
sel.to_csv(TAB / "a_selection_frequencies.csv", index=False)

turn_rows = []
for t in TENORS:
    for k in KS:
        w = weights[strategy_name(t, k, "EQ")]
        comp = (w > 0).astype(int) - (w < 0).astype(int)
        changes = (comp.diff().abs().sum(axis=1) > 0).iloc[1:]
        turn_rows.append({"tenor": t, "K": k, "days_with_composition_change": int(changes.sum()),
                          "pct_days_with_change": changes.mean() * 100,
                          "avg_days_between_changes": len(changes) / max(changes.sum(), 1),
                          "distinct_long_sets": comp.apply(lambda r: tuple(r[r > 0].index), axis=1).nunique(),
                          "distinct_short_sets": comp.apply(lambda r: tuple(r[r < 0].index), axis=1).nunique()})
turn = pd.DataFrame(turn_rows)
turnover = pd.Series({n: weights[n].diff().abs().sum(axis=1).iloc[1:].mean() for n in names}, name="avg_daily_sum_abs_weight_change")
turnover.to_csv(TAB / "a_weight_turnover.csv")
turn.to_csv(TAB / "a_composition_changes.csv", index=False)

# selection frequency figure
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
for ax, (t, k) in zip(axes, [(t, k) for t in TENORS for k in KS]):
    s = sel[(sel.tenor == t) & (sel.K == k)].set_index("currency")
    ax.barh(s.index, s["pct_days_long"], color="#2ca02c", label="long")
    ax.barh(s.index, -s["pct_days_short"], color="#d62728", label="short")
    ax.set_title(f"{t} rates, K={k}")
    ax.set_xlabel("% of days (short | long)")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(-105, 105)
axes[0].invert_yaxis()
axes[0].legend(loc="lower left", fontsize=7)
fig.suptitle("Selection frequency of each currency, 1999-01-01 to 2010-06-03 signal dates")
savefig(fig, "a_selection_frequency.png")

# signed weights through time as heat maps (rows = currencies, columns = signal dates)
fig, axes = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True)
x_num = mdates.date2num(dates.to_pydatetime())
for ax, m in zip(axes, METHODS):
    w = weights[strategy_name("3M", 3, m)]
    im = ax.imshow(w.T.values, aspect="auto", cmap="RdBu", vmin=-0.75, vmax=0.75, interpolation="nearest",
                   extent=[x_num[0], x_num[-1], len(UNIVERSE) - 0.5, -0.5])
    ax.set_yticks(range(len(UNIVERSE)), UNIVERSE, fontsize=7)
    ax.set_title(f"3M-K3-{m} ({METHOD_LABEL[m]}): daily signed weights (blue = long, red = short)")
    ax.grid(False)
    ax.xaxis_date()
fig.colorbar(im, ax=axes, shrink=0.6, label="weight")
savefig(fig, "a_weights_through_time_3M_K3.png", layout=False)

# ----------------------------------------------------------------------------- (b) distributions
T = summary_table(R)
T.to_csv(TAB / "b_baseline_metrics.csv")
best, ranks = select_best(T)
ranks.to_csv(TAB / "b_selection_ranks.csv")
note(f"Best baseline: {best} (avg rank {ranks.loc[best, 'avg_rank']:.3f})")

spec = pd.DataFrame([{"name": strategy_name(t, k, m), "tenor": t, "K": k, "method": m} for t, k, m in all_specs()]).set_index("name")
corr = R.corr()
corr.to_csv(TAB / "b_return_correlations.csv")
factor_rows, factor_summary = [], []
for factor, others in [("tenor", ["K", "method"]), ("K", ["tenor", "method"]), ("method", ["tenor", "K"])]:
    ranges = {"sharpe": [], "ann_arith_mean": [], "ann_geo_return": [], "ann_vol": [], "max_drawdown": [], "es95_daily": [], "min_pair_corr": []}
    for key, grp in spec.groupby(others):
        members = list(grp.index)
        sub = T.loc[members]
        row = {"factor": factor, "held_fixed": ", ".join(f"{o}={v}" for o, v in zip(others, key))}
        for lv in grp[factor]:
            nm = grp.index[grp[factor] == lv][0]
            row[f"sharpe[{lv}]"] = T.loc[nm, "sharpe"]
            row[f"geo[{lv}]"] = T.loc[nm, "ann_geo_return"]
            row[f"mdd[{lv}]"] = T.loc[nm, "max_drawdown"]
            row[f"vol[{lv}]"] = T.loc[nm, "ann_vol"]
        for mtr in ["sharpe", "ann_arith_mean", "ann_geo_return", "ann_vol", "max_drawdown", "es95_daily"]:
            rng = sub[mtr].astype(float).max() - sub[mtr].astype(float).min()
            row[f"range_{mtr}"] = rng
            ranges[mtr].append(rng)
        cc = corr.loc[members, members].values
        row["min_pair_corr"] = cc[np.triu_indices(len(members), 1)].min()
        ranges["min_pair_corr"].append(row["min_pair_corr"])
        factor_rows.append(row)
    factor_summary.append({"factor": factor, "n_groups": len(ranges["sharpe"]),
                           **{f"avg_range_{k}": float(np.mean(v)) for k, v in ranges.items() if k != "min_pair_corr"},
                           "avg_min_pair_corr": float(np.mean(ranges["min_pair_corr"]))})
matched = pd.DataFrame(factor_rows)
matched.to_csv(TAB / "b_matched_comparisons.csv", index=False)
fsum = pd.DataFrame(factor_summary).set_index("factor")
fsum.to_csv(TAB / "b_factor_importance.csv")

# K3 vs K2 wins
k_wins = sum(T.loc[strategy_name(t, 3, m), "sharpe"] > T.loc[strategy_name(t, 2, m), "sharpe"] for t in TENORS for m in METHODS)

fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey=True)
for ax, (t, k) in zip(axes.flat, [(t, k) for t in TENORS for k in KS]):
    for m in METHODS:
        n = strategy_name(t, k, m)
        ax.plot(wealth(R[n]), color=METHOD_COLOR[m], lw=0.9, label=f"{m}")
    ax.set_yscale("log")
    ax.set_yticks([0.8, 1, 1.25, 1.5, 2, 2.5, 3])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(f"{t} rates, K={k}")
    ax.set_ylabel("wealth (log scale), start = 1")
axes[0, 0].legend(title="allocation")
fig.suptitle("Cumulative wealth of the 12 baseline carry portfolios (excess returns, compounded daily)")
savefig(fig, "b_wealth_12.png")

fig, axes = plt.subplots(2, 2, figsize=(11, 6.0), sharex=True, sharey=True)
for ax, (t, k) in zip(axes.flat, [(t, k) for t in TENORS for k in KS]):
    for m in METHODS:
        n = strategy_name(t, k, m)
        ax.plot(drawdown_series(R[n]) * 100, color=METHOD_COLOR[m], lw=0.8, label=m)
    ax.set_title(f"{t} rates, K={k}")
    ax.set_ylabel("drawdown from peak (%)")
axes[0, 0].legend(title="allocation")
fig.suptitle("Drawdowns of the 12 baseline carry portfolios")
savefig(fig, "b_drawdowns_12.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
ax = axes[0]
data = [R[n].values * 100 for n in names]
ax.boxplot(data, whis=(1, 99), showfliers=True, flierprops={"markersize": 1.5, "alpha": 0.4})
ax.set_xticks(range(1, 13), names, rotation=70, fontsize=7)
ax.set_ylabel("daily return (%)")
ax.set_title("Daily returns: box = IQR, whiskers = 1st-99th pct")
ax = axes[1]
probs = np.linspace(0.0005, 0.05, 100)
for n in names:
    t_, k_, m_ = n.split("-")
    ax.plot(probs * 100, R[n].quantile(probs).values * 100, color=METHOD_COLOR[m_],
            ls="-" if k_ == "K3" else "--", lw=1.4 if t_ == "3M" else 0.8, alpha=0.85, label=n)
ax.set_xlabel("left-tail probability (%)")
ax.set_ylabel("daily return quantile (%)")
ax.set_title("Left-tail quantiles (solid = K3, dashed = K2; thick = 3M)")
ax.legend(fontsize=6, ncol=2)
savefig(fig, "b_distribution.png")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, (factor, levels) in zip(axes, [("tenor", TENORS), ("K", KS), ("method", METHODS)]):
    for lv_i, lv in enumerate(levels):
        vals = spec.index[spec[factor] == lv]
        ax.scatter(np.full(len(vals), lv_i) + np.linspace(-0.12, 0.12, len(vals)), T.loc[vals, "sharpe"].astype(float), s=25)
    ax.set_xticks(range(len(levels)), [str(lv) for lv in levels])
    ax.set_title(f"Sharpe by {factor} (avg matched range {fsum.loc[factor, 'avg_range_sharpe']:.3f})")
    ax.set_ylabel("Sharpe ratio")
savefig(fig, "b_factor_sharpe.png")

rolling = R.rolling(ANN).apply(lambda x: x.mean() / x.std() * np.sqrt(ANN), raw=True)
fig, ax = plt.subplots(figsize=(10, 3.4))
for n, c in [(best, "k"), ("1M-K2-RANK", "#d62728"), ("3M-K2-EQ", "#ff7f0e")]:
    ax.plot(rolling[n], lw=0.9, label=n, color=c)
ax.axhline(0, color="grey", lw=0.6)
ax.set_ylabel("rolling 260-day Sharpe")
ax.legend()
ax.set_title("Rolling one-year Sharpe ratio")
savefig(fig, "b_rolling_sharpe.png")

# ----------------------------------------------------------------------------- (c) volatility relations
B = R[best]
c_idx = B.index[B.index > vols_f.dropna().index[0]]
Bc = B.loc[c_idx]
contemp = contemporaneous(Bc, ds.vols_raw, vols_f)
pred = predictive(Bc, vols_f, 21)
reg = regimes(Bc, vols_f)
contemp.to_csv(TAB / "c_contemporaneous.csv")
pred.to_csv(TAB / "c_predictive.csv")
reg.to_csv(TAB / "c_regimes.csv", index=False)
reg_p = reg.pivot(index="vol", columns="regime", values="sharpe")[["low", "medium", "high"]]
reg_p["low_minus_high"] = reg_p["low"] - reg_p["high"]
reg_es = reg.pivot(index="vol", columns="regime", values="es95_daily")[["low", "medium", "high"]]
vol_corr = vols_f.loc[c_idx].corr()
vol_corr.to_csv(TAB / "c_vol_level_correlations.csv")
dvol_corr = pd.DataFrame({v: vols_f[v].diff().where(ds.vols_raw[v].notna()) for v in VOLS}).loc[c_idx].corr()
dvol_corr.to_csv(TAB / "c_vol_change_correlations.csv")
# Candidate rule: predictive evidence only, because conditioning acts on v_t to scale R_{t+1}.
# Primary: low-minus-high tercile Sharpe of R_{t+1}; tie-break: lower high-regime ES.
cand = pd.DataFrame({
    "low_minus_high_sharpe": reg_p["low_minus_high"],
    "sharpe_low": reg_p["low"], "sharpe_high": reg_p["high"],
    "high_regime_es95": reg_es["high"],
    "t21d_level": pred["t21d_level"].astype(float),
    "contemp_corr": contemp["corr_R_dVol"].astype(float),
})
cand = cand.sort_values(["low_minus_high_sharpe", "high_regime_es95"], ascending=[False, True])
cand.to_csv(TAB / "c_candidate_ranking.csv")
candidate = cand.index[0]
note(f"Part (c) candidate conditioning variable: {candidate}")

fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
for ax, v in zip(axes.flat, VOLS):
    dv = vols_f[v].diff().where(ds.vols_raw[v].notna()).reindex(Bc.index)
    ax.scatter(dv, Bc * 100, s=2, alpha=0.35, color=VOL_COLOR[v])
    ok = pd.concat([dv, Bc], axis=1).dropna()
    b1, b0 = np.polyfit(ok.iloc[:, 0], ok.iloc[:, 1] * 100, 1)
    xs = np.linspace(ok.iloc[:, 0].quantile(0.001), ok.iloc[:, 0].quantile(0.999), 10)
    ax.plot(xs, b0 + b1 * xs, color="k", lw=1)
    ax.set_title(f"{v}: corr = {contemp.loc[v, 'corr_R_dVol']:.3f}")
    ax.set_xlabel(f"same-day change in {v} (vol pts)")
    ax.set_ylabel(f"{best} daily return (%)")
fig.suptitle(f"Contemporaneous: {best} return on day t+1 vs change in volatility over the same day")
savefig(fig, "c_scatter_contemporaneous.png")

fwd21 = Bc[::-1].rolling(21, min_periods=21).sum()[::-1]
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
for ax, v in zip(axes.flat, VOLS):
    lvl = vols_f[v].shift(1).reindex(Bc.index)
    ax.scatter(lvl, fwd21 * 100, s=2, alpha=0.3, color=VOL_COLOR[v])
    ax.set_title(f"{v}: corr = {pred.loc[v, 'corr_level_t_R_21d']:.3f}, HAC t = {pred.loc[v, 't21d_level']:.2f}")
    ax.set_xlabel(f"{v} level at t (vol pts)")
    ax.set_ylabel("sum of next 21 daily returns (%)")
fig.suptitle(f"Predictive: volatility level at t vs {best} return over t+1..t+21")
savefig(fig, "c_scatter_predictive.png")

fig, ax = plt.subplots(figsize=(9, 3.6))
x = np.arange(len(VOLS))
for i, g in enumerate(["low", "medium", "high"]):
    ax.bar(x + (i - 1) * 0.27, reg_p.loc[VOLS, g], width=0.27, label=f"{g} tercile of v_t")
ax.set_xticks(x, VOLS)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel(f"annualised Sharpe of {best} on t+1")
ax.set_title("Carry performance by full-sample volatility tercile (descriptive, in-sample)")
ax.legend()
savefig(fig, "c_regime_sharpe.png")

fig, axes = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
axes[0].plot(wealth(B), color="k", lw=0.9)
axes[0].set_ylabel(f"{best} wealth")
axes[0].set_title(f"{best} wealth and volatility series")
for v in VOLS:
    axes[1].plot(vols_f[v], lw=0.7, color=VOL_COLOR[v], label=v)
axes[1].set_ylabel("vol points")
axes[1].legend(ncol=6, fontsize=7)
savefig(fig, "c_wealth_and_vols.png")

# ----------------------------------------------------------------------------- (d) implied-vol conditioning
signals = {v: cond.signal(ds.vols_raw[v], dates) for v in VOLS}
cond_rets = {v: cond.conditioned_returns(B, signals[v]["invested"]) for v in VOLS}
start_d = max(cr.first_valid_index() for cr in cond_rets.values())
D_idx = B.index[B.index >= start_d]
D = pd.DataFrame({"baseline": B.loc[D_idx], **{f"{v}-cond": cond_rets[v].loc[D_idx] for v in VOLS}})
assert not D.isna().any().any()
Dm = summary_table(D)
exp_stats = pd.DataFrame({f"{v}-cond": cond.exposure_stats(signals[v]["invested"], D_idx) for v in VOLS}).T
Dm = Dm.join(exp_stats)
Dm.loc["baseline", ["frac_invested", "n_switches", "n_off_spells", "avg_off_spell_days"]] = [1.0, 0, 0, 0.0]
Dm["frac_positive_when_invested"] = [((D[c] > 0).sum() / max((D[c] != 0).sum(), 1)) for c in D]
Dm.to_csv(TAB / "d_conditioned_metrics.csv")
best_d, ranks_d = select_best(Dm.drop(index="baseline"))
ranks_d.to_csv(TAB / "d_selection_ranks.csv")
note(f"Part (d) sample {D_idx[0].date()}..{D_idx[-1].date()}; best conditioned: {best_d}")
sig_hist = pd.concat({v: signals[v] for v in VOLS}, axis=1)
sig_hist.columns = [f"{a}|{b}" for a, b in sig_hist.columns]
sig_hist.index.name = "signal_date"
sig_hist.to_csv(HIST / "d_signals_daily.csv", float_format="%.6f")
D.index.name = "return_date"
D.to_csv(HIST / "d_conditioned_returns_daily.csv", float_format="%.10f")

ep_rows = []
for ep, (a, b) in EPISODES.items():
    for c in D:
        sub = D.loc[a:b, c]
        if c == "baseline":
            first_off = ""
            inv = 1.0
        else:
            v = c.replace("-cond", "")
            s = signals[v]["invested"].loc[pd.Timestamp(a) - pd.Timedelta(days=45):b]
            off = s[s == 0]
            first_off = off.index[0].date().isoformat() if len(off) else "never"
            inv = signals[v]["invested"].shift(1).loc[a:b].mean()
        ep_rows.append({"episode": ep, "strategy": c, "cum_return": float((1 + sub).prod() - 1),
                        "frac_invested": inv, "first_off_signal_from_45d_before": first_off})
episodes = pd.DataFrame(ep_rows)
episodes.to_csv(TAB / "d_episodes.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(wealth(D["baseline"]), color="k", lw=1.4, label="baseline (unconditioned)")
for v in VOLS:
    ax.plot(wealth(D[f"{v}-cond"]), color=VOL_COLOR[v], lw=0.9, label=f"{v}-cond")
ax.set_ylabel("wealth, start = 1")
ax.set_title(f"{best}: baseline vs six volatility-conditioned versions ({D_idx[0].date()} to {D_idx[-1].date()})")
ax.legend(ncol=4, fontsize=7)
savefig(fig, "d_wealth.png")

fig, axes = plt.subplots(len(VOLS), 1, figsize=(10, 7.5), sharex=True)
for ax, v in zip(axes, VOLS):
    s = signals[v].loc[D_idx[0]:]
    ax.plot(s["value"], color=VOL_COLOR[v], lw=0.7)
    ax.plot(s["threshold"], color="k", lw=0.8, ls="--")
    off = s["invested"] == 0
    ax.fill_between(s.index, 0, 1, where=off, transform=ax.get_xaxis_transform(), color="grey", alpha=0.25, lw=0)
    ax.set_ylabel(v, rotation=0, ha="right")
axes[0].set_title("Volatility level (colour), expanding 75th-percentile threshold (dashed), shaded = carry switched off")
savefig(fig, "d_signals.png")

fig, ax = plt.subplots(figsize=(10, 3.4))
for c in D:
    ax.plot(drawdown_series(D[c]) * 100, lw=1.3 if c == "baseline" else 0.8,
            color="k" if c == "baseline" else VOL_COLOR[c.replace("-cond", "")], label=c)
ax.set_ylabel("drawdown (%)")
ax.legend(ncol=4, fontsize=7)
ax.set_title("Drawdowns: baseline vs conditioned")
savefig(fig, "d_drawdowns.png")

# ----------------------------------------------------------------------------- (e) realised-vol conditioning
audjpy = audjpy_cross(ds.usd_per_ccy)
rv = cond.realized_vol(audjpy, 21)
rv_sig = cond.signal(rv, dates)
rv_ret = cond.conditioned_returns(B, rv_sig["invested"])
start_e = max(start_d, rv_ret.first_valid_index())
E_idx = B.index[B.index >= start_e]
E = pd.DataFrame({"baseline": B.loc[E_idx], best_d: D[best_d].reindex(E_idx),
                  "RV21_AUDJPY-cond": rv_ret.loc[E_idx]})
assert not E.isna().any().any()
Em = summary_table(E)
Em = Em.join(pd.DataFrame({best_d: cond.exposure_stats(signals[best_d.replace("-cond", "")]["invested"], E_idx),
                           "RV21_AUDJPY-cond": cond.exposure_stats(rv_sig["invested"], E_idx)}).T)
Em.loc["baseline", ["frac_invested", "n_switches", "n_off_spells", "avg_off_spell_days"]] = [1.0, 0, 0, 0.0]
Em.to_csv(TAB / "e_metrics.csv")
note(f"Part (e) sample {E_idx[0].date()}..{E_idx[-1].date()}")
rv_sig.to_csv(HIST / "e_rv_signal_daily.csv", float_format="%.6f")
E.index.name = "return_date"
E.to_csv(HIST / "e_returns_daily.csv", float_format="%.10f")
iv_name = best_d.replace("-cond", "")
both = pd.concat([rv.rename("RV21"), vols_f[["AY1wk", "AY3M", iv_name]].loc[:, lambda d: ~d.columns.duplicated()]], axis=1).dropna()
rv_iv = {
    "corr_level_RV_AY1wk": both["RV21"].corr(both["AY1wk"]),
    "corr_level_RV_AY3M": both["RV21"].corr(both["AY3M"]),
    "mean_AY3M_minus_RV21": float((both["AY3M"] - both["RV21"]).mean()),
    "mean_AY1wk_minus_RV21": float((both["AY1wk"] - both["RV21"]).mean()),
    "corr_signal_RV_vs_best_IV": float(pd.concat([rv_sig["invested"], signals[iv_name]["invested"]], axis=1).dropna().corr().iloc[0, 1]),
    "agreement_RV_vs_best_IV": float((rv_sig["invested"] == signals[iv_name]["invested"]).loc[E_idx[0]:].mean()),
}
# lead-lag: does RV respond after IV? corr of daily changes at leads/lags
d_rv = rv.diff()
d_iv = vols_f[iv_name].diff()
leadlag = {lag: float(d_rv.corr(d_iv.shift(lag))) for lag in range(-5, 6)}
rv_iv["leadlag_corr_dRV_t_vs_dIV_t-lag"] = leadlag
ep_e = []
for ep, (a, b) in EPISODES.items():
    for c in E:
        sub = E.loc[a:b, c]
        if c == "baseline":
            first_off, inv = "", 1.0
        else:
            s_ = rv_sig["invested"] if c.startswith("RV") else signals[iv_name]["invested"]
            ss = s_.loc[pd.Timestamp(a) - pd.Timedelta(days=45):b]
            off = ss[ss == 0]
            first_off = off.index[0].date().isoformat() if len(off) else "never"
            inv = s_.shift(1).loc[a:b].mean()
        ep_e.append({"episode": ep, "strategy": c, "cum_return": float((1 + sub).prod() - 1), "frac_invested": inv,
                     "first_off_signal_from_45d_before": first_off})
episodes_e = pd.DataFrame(ep_e)
episodes_e.to_csv(TAB / "e_episodes.csv", index=False)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(rv, color=VOL_COLOR["RV21_AUDJPY"], lw=0.8, label="21-day realised vol, AUD/JPY")
axes[0].plot(vols_f[iv_name], color=VOL_COLOR.get(iv_name, "b"), lw=0.8, label=f"{iv_name} implied vol")
if iv_name != "AY3M":
    axes[0].plot(vols_f["AY3M"], color=VOL_COLOR["AY3M"], lw=0.6, alpha=0.6, label="AY3M implied vol")
axes[0].plot(rv_sig["threshold"], color="k", ls="--", lw=0.8, label="RV expanding 75th pct threshold")
axes[0].set_ylabel("vol points")
axes[0].legend(fontsize=7, ncol=2)
axes[0].set_title("Realised vs implied volatility")
for c, col in [("baseline", "k"), (best_d, VOL_COLOR.get(iv_name, "b")), ("RV21_AUDJPY-cond", VOL_COLOR["RV21_AUDJPY"])]:
    axes[1].plot(wealth(E[c]), color=col, lw=1.0, label=c)
axes[1].set_ylabel("wealth, start = 1")
axes[1].legend(fontsize=7)
axes[1].set_title(f"Common sample {E_idx[0].date()} to {E_idx[-1].date()}")
savefig(fig, "e_rv_vs_iv.png")

fig, ax = plt.subplots(figsize=(10, 3.2))
for c, col in [("baseline", "k"), (best_d, VOL_COLOR.get(iv_name, "b")), ("RV21_AUDJPY-cond", VOL_COLOR["RV21_AUDJPY"])]:
    ax.plot(drawdown_series(E[c]) * 100, color=col, lw=0.9, label=c)
ax.set_ylabel("drawdown (%)")
ax.legend(fontsize=7)
ax.set_title("Drawdowns on the part (e) common sample")
savefig(fig, "e_drawdowns.png")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
for ax, (ep, (a, b)) in zip(axes, EPISODES.items()):
    for c, col in [("baseline", "k"), (best_d, VOL_COLOR.get(iv_name, "b")), ("RV21_AUDJPY-cond", VOL_COLOR["RV21_AUDJPY"])]:
        sub = E.loc[a:b, c]
        ax.plot((1 + sub).cumprod() - 1, color=col, lw=1, label=c)
    ax.set_title(ep, fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax.set_ylabel("cumulative return in episode")
axes[0].legend(fontsize=6)
savefig(fig, "e_episodes.png")

# ----------------------------------------------------------------------------- sensitivity
sens = {}
# (1) USD excluded from the ranking universe (pure funding currency)
ex_usd = {}
for t, k, m in all_specs():
    c9 = carry_signal(ds.rates[t], CCYS)
    w9 = build_weights(c9, k, m)
    assert validate_weights(w9, c9, k)["all_ok"]
    ex_usd[strategy_name(t, k, m)] = portfolio_returns(w9, excess[t][CCYS])
T9 = summary_table(pd.DataFrame(ex_usd))
best9, _ = select_best(T9)
s1 = pd.DataFrame({"sharpe_USD_in_universe": T["sharpe"], "sharpe_USD_excluded": T9["sharpe"],
                   "mdd_USD_in_universe": T["max_drawdown"], "mdd_USD_excluded": T9["max_drawdown"]})
s1.to_csv(TAB / "sens_usd_excluded.csv")
sens["best_if_usd_excluded"] = best9
usd_freq = sel[sel.currency == "USD"]
# (2) ACT/365 accrual for every currency
alt_delta = year_fraction(dates, 365.0)
t_best, k_best, m_best = best.split("-")[0], int(best.split("-")[1][1:]), best.split("-")[2]
R365 = portfolio_returns(weights[best], excess_returns(sr, ds.rates[t_best], alt_delta))
sens["best_act365"] = {k: summarize(R365)[k] for k in ["ann_arith_mean", "ann_geo_return", "sharpe", "max_drawdown"]}
sens["best_act360"] = {k: T.loc[best, k] for k in ["ann_arith_mean", "ann_geo_return", "sharpe", "max_drawdown"]}
# (3) conditioning robustness: extra one-day lag; 90th percentile (not used for selection)
rob = {}
for label, kw in [("q75_lag0 (primary)", {}), ("q75_extra_lag1", {"extra_lag": 1}), ("q90_lag0", {"q": 0.90})]:
    col = {}
    for v in VOLS:
        s_ = cond.signal(ds.vols_raw[v], dates, **kw)
        col[v] = summarize(cond.conditioned_returns(B, s_["invested"]).loc[D_idx[1:]])["sharpe"]
    s_ = cond.signal(rv, dates, **kw)
    col["RV21_AUDJPY"] = summarize(cond.conditioned_returns(B, s_["invested"]).loc[E_idx[1:]])["sharpe"]
    rob[label] = col
rob = pd.DataFrame(rob)
rob.loc["baseline"] = [summarize(B.loc[D_idx[1:]])["sharpe"]] * rob.shape[1]
rob.to_csv(TAB / "sens_conditioning_rules.csv")
# (4) best conditioned variable applied to the other baselines (does the vol effect generalise?)
gen = {}
for n in names:
    s_ = signals[iv_name]["invested"]
    cr = cond.conditioned_returns(R[n], s_).loc[D_idx]
    gen[n] = {"sharpe_base": summarize(R[n].loc[D_idx])["sharpe"], "sharpe_cond": summarize(cr)["sharpe"],
              "mdd_base": summarize(R[n].loc[D_idx])["max_drawdown"], "mdd_cond": summarize(cr)["max_drawdown"]}
gen = pd.DataFrame(gen).T
gen.to_csv(TAB / "sens_best_signal_on_all_baselines.csv")

# ----------------------------------------------------------------------------- manual reconciliation
RECON_T = pd.Timestamp("2008-10-24")
RECON_T1 = dates[dates.get_loc(RECON_T) + 1]
w_t = weights[best].loc[RECON_T]
held = w_t[w_t != 0].index.tolist()
rows = []
dlt = (RECON_T1 - RECON_T).days / DAYCOUNT
for c in held:
    q0 = ds.quotes[c].loc[RECON_T] if c != "USD" else 1.0
    q1 = ds.quotes[c].loc[RECON_T1] if c != "USD" else 1.0
    x0 = ds.usd_per_ccy[c].loc[RECON_T] if c != "USD" else 1.0
    x1 = ds.usd_per_ccy[c].loc[RECON_T1] if c != "USD" else 1.0
    r = x1 / x0 - 1
    i_c = ds.rates[t_best][c].loc[RECON_T]
    i_u = ds.rates[t_best]["USD"].loc[RECON_T]
    e = (1 + r) * (1 + i_c * dlt) - (1 + i_u * dlt)
    rows.append({"currency": c, "quote_convention": QUOTE.get(c, "-"), "quote_t": q0, "quote_t+1": q1,
                 "X_t (USD/ccy)": x0, "X_t+1 (USD/ccy)": x1, "spot_return": r,
                 "rate_t_pct": i_c * 100, "usd_rate_t_pct": i_u * 100, "delta": dlt, "excess_return": e,
                 "weight": w_t[c], "contribution": w_t[c] * e})
recon = pd.DataFrame(rows).set_index("currency")
recon_total = recon["contribution"].sum()
pipeline_val = R[best].loc[RECON_T1]
recon.to_csv(TAB / "reconciliation.csv")
assert abs(recon_total - pipeline_val) < 1e-14
note(f"Manual reconciliation {best} {RECON_T.date()}->{RECON_T1.date()}: manual {recon_total:.10f} pipeline {pipeline_val:.10f}")

# ----------------------------------------------------------------------------- export workbook + results
with pd.ExcelWriter(OUT / "hw3_tables.xlsx") as xw:
    for f in sorted(TAB.glob("*.csv")):
        pd.read_csv(f).to_excel(xw, sheet_name=f.stem[:31], index=False)

results = {
    "audit": {k: v for k, v in ds.audit.items() if k != "vix_missing_dates"},
    "n_vix_missing": len(ds.audit["vix_missing_dates"]),
    "best_baseline": best, "k3_beats_k2_count": int(k_wins),
    "candidate_c": candidate, "best_d": best_d, "d_start": D_idx[0], "d_end": D_idx[-1],
    "e_start": E_idx[0], "e_end": E_idx[-1], "example_date": EXAMPLE_DATE, "example_next": next_date,
    "last_signal_date": LAST_DATE,
    "recon": {"t": RECON_T, "t1": RECON_T1, "manual": recon_total, "pipeline": pipeline_val},
    "rv_iv": rv_iv, "sens": sens,
    "validation": validation,
    "max_abs_diff_returns": float(cmp_["max_abs_diff_vs_workbook"].max()),
    "naive_negation_max_error": float(neg_err.max()),
    "baseline_zero_return_days": int((R == 0).sum().sum()),
    "log": log,
}
(OUT / "results.json").write_text(json.dumps(jsonable(results), indent=2, default=str))
(OUT / "run_log.txt").write_text("\n".join(log) + "\n")
print("done")
