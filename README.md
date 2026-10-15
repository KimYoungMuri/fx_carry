# B9339 HW3 – FX carry

Young Kim (yk3057), B9339 Systematic Investment Strategies, Fall 2026.

Complete, reproducible analysis for Homework 3 (parts a–e): 12 daily-rebalanced G-10 carry portfolios,
their return distributions, their relation to FX and equity volatility, and implied- and realised-volatility
conditioning.

**Report:** `report/hw3_report.pdf` (page 1 = summary of answers).

## Inputs

- `hw3.data.carry.FX.dly.xlsx` – supplied workbook (the assignment calls it `hw3.data.FX.dly.xlsx`). Must sit in this folder.
- The assignment handout itself is not included in this repository.

## Reproduce

Python 3.12, from this `hw3/` directory:

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
MPLCONFIGDIR=.mplcache .venv/bin/python scripts/run_all.py       # analysis -> outputs/
.venv/bin/python -m pytest -q tests                              # 16 verification tests
MPLCONFIGDIR=.mplcache .venv/bin/python scripts/build_report.py  # -> report/hw3_report.pdf
```

`run_all.py` takes about 15 seconds; it fails loudly if the workbook is missing or its column layout differs from
the assignment description.

## Layout

| Path | Content |
|---|---|
| `src/fxcarry/data.py` | Workbook loading, header/layout verification, quote standardisation to USD per currency |
| `src/fxcarry/returns.py` | Spot returns, ACT/360 accrual, funded excess returns, AUD/JPY cross |
| `src/fxcarry/portfolios.py` | Carry ranking, EQ / RANK / PROP weights, daily validation |
| `src/fxcarry/metrics.py` | Distribution statistics, drawdowns, best-portfolio decision rule |
| `src/fxcarry/volanalysis.py` | Part (c) contemporaneous / predictive / regime analysis |
| `src/fxcarry/conditioning.py` | Expanding-percentile thresholds, signals, realised volatility |
| `scripts/run_all.py` | Runs everything, writes tables, histories, figures, `outputs/results.json` |
| `scripts/build_report.py` | Builds the PDF from the outputs |
| `tests/test_hw3.py` | Verification tests (inversion, accrual, weights, look-ahead, compounding, …) |
| `outputs/tables/` | Every report table as CSV; all of them also in `outputs/hw3_tables.xlsx` |
| `outputs/histories/` | Daily weights of all 12 baselines, daily returns, carry signals, volatility signals |
| `outputs/figures/` | All figures (PNG) |

## Key conventions (details in report §1)

- USD is one of the 10 ranked currencies (excluding it is a sensitivity check).
- Excess return of a long position: `e = (1+r)(1+i·Δ) − (1+i_USD·Δ)`, with `Δ` = calendar days / 360.
- Weights set from rates at the close of t and applied to the return over (t, t+1]; long side +1, short side −1.
- RANK = linear ranks within each side; PROP = distance of carry from the cross-sectional median.
  Both are stated assumptions, because no lecture definitions were supplied.
- Conditioning: flat when the latest volatility is above the expanding 75th percentile of earlier observations
  (252-observation warm-up); the same rule is used for every variable.
