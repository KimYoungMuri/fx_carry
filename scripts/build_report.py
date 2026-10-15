"""Build report/hw3_report.pdf from the outputs of scripts/run_all.py.

Usage (from hw3/):  .venv/bin/python scripts/build_report.py
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
OUT, TAB, FIG = ROOT / "outputs", ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
PDF = ROOT / "report" / "hw3_report.pdf"
res = json.loads((OUT / "results.json").read_text())


def tab(name, **kw):
    return pd.read_csv(TAB / f"{name}.csv", **kw)


# ----------------------------------------------------------------------------- styles
AUTHOR = "Young Kim (yk3057)"
plt.rcParams.update({"mathtext.fontset": "cm"})
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="Times-Roman", fontSize=10.5, leading=13.8,
                      spaceAfter=5, firstLineIndent=0)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=9.2, leading=11.8, spaceAfter=3)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=9, leading=11, spaceAfter=3)
H1 = ParagraphStyle("h1", parent=BODY, fontName="Times-Bold", fontSize=13.5, leading=17, spaceBefore=10, spaceAfter=5)
H2 = ParagraphStyle("h2", parent=BODY, fontName="Times-Bold", fontSize=11.2, leading=14, spaceBefore=7, spaceAfter=3)
TITLE = ParagraphStyle("title", parent=BODY, fontName="Times-Roman", fontSize=17, leading=21, alignment=TA_CENTER,
                       spaceAfter=2)
CENTER = ParagraphStyle("center", parent=BODY, alignment=TA_CENTER, spaceAfter=1)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.6, leading=9, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Times-Bold")
BUL = ParagraphStyle("bul", parent=SMALL, leftIndent=14, bulletIndent=3, spaceAfter=2)

story: list = []
MARGIN = 0.85 * inch
W_AVAIL = letter[0] - 2 * MARGIN


_FIRST_PERSON = [(r"\bWe\b", "I"), (r"\bwe\b", "I"), (r"\bOur\b", "My"), (r"\bour\b", "my"), (r"\bus\b", "me")]


def _plain(text):
    text = text.replace("<b>", "").replace("</b>", "")
    for pat, rep in _FIRST_PERSON:
        text = re.sub(pat, rep, text)
    return text


def P(text, style=BODY):
    story.append(Paragraph(_plain(text), style))


def bullets(items, style=BUL, prose=False):
    """Prose paragraphs for discussion; a plain list only where the content is genuinely a list."""
    for it in items:
        if prose:
            story.append(Paragraph(_plain(it), BODY))
        else:
            story.append(Paragraph(_plain(it), style, bulletText="\u2013"))


def h1(t):
    story.append(Paragraph(t, H1))


def h2(t):
    story.append(Paragraph(t, H2))


def eq(tex, size=12.5):
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=size)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.03, transparent=True)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image as PILImage
    w, h = PILImage.open(buf).size
    buf.seek(0)
    wpt, hpt = w / 300 * 72, h / 300 * 72
    max_w = W_AVAIL - 24
    if wpt > max_w:
        wpt, hpt = max_w, hpt * max_w / wpt
    img = Image(buf, width=wpt, height=hpt)
    img.hAlign = "CENTER"
    story.append(Spacer(1, 2))
    story.append(img)
    story.append(Spacer(1, 4))


def _caption(label: str, text: str) -> Paragraph:
    """'Table 3. text' -> bold-free 'Table 3: text' in LaTeX style."""
    head, _, rest = text.partition(". ")
    if head.startswith(label):
        text = f"{head}: {rest}"
    return Paragraph(_plain(text), CAP)


def figure(name, caption, width=W_AVAIL * 0.92):
    from PIL import Image as PILImage
    p = FIG / name
    w, h = PILImage.open(p).size
    img = Image(str(p), width=width, height=width * h / w)
    story.append(KeepTogether([img, Spacer(1, 2), _caption("Figure", caption), Spacer(1, 8)]))


def fmt(x, d=5):
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return ""
    if isinstance(x, (int,)) or (isinstance(x, float) and float(x).is_integer() and abs(x) >= 1):
        return f"{int(x)}"
    return f"{x:.{d}f}"


def table(df: pd.DataFrame, col_widths=None, header=None, font=7.6, index=True, digits=5, bold_rows=(), caption=None):
    if col_widths and sum(col_widths) > W_AVAIL:
        col_widths = [c * W_AVAIL / sum(col_widths) for c in col_widths]
    cell = ParagraphStyle(f"cell{font}", parent=CELL, fontSize=font, leading=font + 1.4)
    cellb = ParagraphStyle(f"cellb{font}", parent=cell, fontName="Times-Bold")
    data = []
    cols = list(df.columns)
    hdr = ([df.index.name or ""] if index else []) + (header or cols)
    data.append([Paragraph(str(h), cell) for h in hdr])
    bold_idx = {list(df.index).index(r) + 1 for r in bold_rows}
    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        st_ = cellb if i in bold_idx else cell
        cells = [Paragraph(str(idx), st_)] if index else []
        for c in cols:
            cells.append(Paragraph(fmt(row[c], digits), st_))
        data.append(cells)
    t = Table(data, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
    st = [("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
          ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
          ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
          ("VALIGN", (0, 0), (-1, 0), "BOTTOM"), ("VALIGN", (0, 1), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
          ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]
    t.setStyle(TableStyle(st))
    block = [_caption("Table", caption), Spacer(1, 2)] if caption else []
    block += [t, Spacer(1, 9)]
    story.append(KeepTogether(block))


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


# ----------------------------------------------------------------------------- load numbers
T = tab("b_baseline_metrics", index_col=0)
ranks = tab("b_selection_ranks", index_col=0)
fimp = tab("b_factor_importance", index_col=0)
best = res["best_baseline"]
Bm = T.loc[best]
contemp = tab("c_contemporaneous", index_col=0)
pred = tab("c_predictive", index_col=0)
regs = tab("c_regimes")
cand = tab("c_candidate_ranking", index_col=0)
candidate = res["candidate_c"]
Dm = tab("d_conditioned_metrics", index_col=0)
ranks_d = tab("d_selection_ranks", index_col=0)
best_d = res["best_d"]
iv = best_d.replace("-cond", "")
Em = tab("e_metrics", index_col=0)
epi = tab("d_episodes")
epi_e = tab("e_episodes")
rob = tab("sens_conditioning_rules", index_col=0)
gen = tab("sens_best_signal_on_all_baselines", index_col=0)
s_usd = tab("sens_usd_excluded", index_col=0)
sel = tab("a_selection_frequencies")
comp = tab("a_composition_changes")
recon = tab("reconciliation", index_col=0)
rv_iv = res["rv_iv"]
sens = res["sens"]
stale = tab("vol_staleness_by_year", index_col=0)
k_pairs = [(f"{t}-K2-{m}", f"{t}-K3-{m}") for t in ["1M", "3M"] for m in ["EQ", "RANK", "PROP"]]
k_wins = sum(T.loc[b, "sharpe"] > T.loc[a, "sharpe"] for a, b in k_pairs)
k_mdd_wins = sum(T.loc[b, "max_drawdown"] > T.loc[a, "max_drawdown"] for a, b in k_pairs)
eq_k3_best = all(T.loc[f"{t}-K3-EQ", "sharpe"] >= T.loc[f"{t}-K3-{m}", "sharpe"] for t in ["1M", "3M"] for m in ["RANK", "PROP"])
dstart, dend, estart, eend = res["d_start"], res["d_end"], res["e_start"], res["e_end"]
regp = regs.pivot(index="vol", columns="regime", values="sharpe")
rv_name = "RV21_AUDJPY-cond"


def ep(df, episode, strat, col="cum_return"):
    return df[(df.episode == episode) & (df.strategy == strat)][col].iloc[0]


# ============================================================================= title + summary
P("B9339 Systematic Investment Strategies", CENTER)
P("Homework 3: FX Carry", TITLE)
P(AUTHOR, CENTER)
P("Prof. Achilles Venetoulias, Fall 2026 &nbsp;&nbsp; October 15, 2026", CENTER)
story.append(Spacer(1, 8))
P(f"Data: hw3.data.carry.FX.dly.xlsx (called hw3.data.FX.dly.xlsx in the assignment), sheet "
  f"&lsquo;{res['audit']['sheet']}&rsquo;, {res['audit']['rows']} weekday observations from "
  f"{res['audit']['start']} to {res['audit']['end']}. Throughout, returns are daily arithmetic USD excess returns on "
  "zero-cost portfolios (long side +1, short side &minus;1), rebalanced daily, with no transaction costs.", SMALL)
h1("Summary of answers")
d_b = Dm.loc["baseline"]
d_best = Dm.loc[best_d]
d_cand = Dm.loc[f"{candidate}-cond"]
e_rv = Em.loc[rv_name]
e_iv = Em.loc[best_d]
e_b = Em.loc["baseline"]
summary_items = [
    f"(a)&nbsp;&nbsp; The universe is the 10 G-10 currencies including USD (USD carry = 0, return = 0). "
    "Each day t the currencies are ranked by carry c<sub>i,t</sub> = i<sub>i,t</sub> &minus; i<sub>USD,t</sub> "
    "using either the 1M or the 3M rate; the K = 2 or 3 highest-carry currencies are held long and the K lowest short, "
    "with equal, rank-based (K, K&minus;1, &hellip;, 1) or proportional-to-carry (distance from the cross-sectional "
    "median carry) magnitudes, normalised to +1 long and &minus;1 short. Holdings change through time; typical "
    "positions are long NZD and AUD (plus GBP or NOK when K = 3) and short JPY and CHF (plus USD, SEK or EUR when K = 3). "
    f"Signed weights for all 12 portfolios on {res['example_date']} are in Table 3; full daily histories are exported.",
    f"(b)&nbsp;&nbsp; Using a rule fixed in advance (average rank of Sharpe, Calmar and 95% expected shortfall), "
    f"the best baseline is <b>{best}</b>: annualised arithmetic mean {Bm.ann_arith_mean:.5f}, CAGR "
    f"{Bm.ann_geo_return:.5f}, volatility {Bm.ann_vol:.5f}, Sharpe {Bm.sharpe:.5f}, skewness {Bm['skew']:.5f}, excess "
    f"kurtosis {Bm.excess_kurt:.5f}, maximum drawdown {Bm.max_drawdown:.5f} ({Bm.dd_peak} to {Bm.dd_trough}, not "
    f"recovered by {res['audit']['end']}). Its 1M twin is a close second (Sharpe {T.loc['1M-K3-EQ','sharpe']:.5f}). "
    f"The <b>number of currencies</b> matters most: K = 3 beats K = 2 on Sharpe in {k_wins}/6 matched pairs and on "
    f"drawdown in {k_mdd_wins}/6; the average matched Sharpe range is {fimp.loc['K','avg_range_sharpe']:.5f} for K vs "
    f"{fimp.loc['method','avg_range_sharpe']:.5f} for allocation and {fimp.loc['tenor','avg_range_sharpe']:.5f} for tenor. "
    "Spreading the book over more currencies diversifies currency-specific crash risk while giving up little carry, which is intuitive.",
    f"(c)&nbsp;&nbsp; Carry losses coincide with rising volatility: the same-day correlation of {best} returns "
    f"with volatility changes is negative for all six series, strongest for VIX ({contemp.loc['VIX','corr_R_dVol']:.5f}) "
    f"and JPY3M ({contemp.loc['JPY3M','corr_R_dVol']:.5f}). Linear regressions of future returns on volatility "
    "<i>levels</i> are insignificant, but regime splits are not: after low-volatility days the next-day Sharpe is far "
    "higher than after high-volatility days. The largest low-minus-high spread is for "
    f"<b>{candidate}</b> ({regp.loc[candidate,'low']:.5f} vs {regp.loc[candidate,'high']:.5f}), followed by AY3M "
    f"({regp.loc['AY3M','low']:.5f} vs {regp.loc['AY3M','high']:.5f}); this makes {candidate} the candidate. VIX has the "
    f"strongest contemporaneous link but the weakest regime spread (high-VIX Sharpe {regp.loc['VIX','high']:.5f} exceeds "
    f"medium {regp.loc['VIX','medium']:.5f}). The data support a carry / FX-vol / equity-vol link contemporaneously; for "
    "conditioning, FX (AUD-related) implied vol is more useful than equity vol. The Jurek reading was not available to us, "
    "so we do not attribute specific claims to it.",
    f"(d)&nbsp;&nbsp; The rule, identical for all six series, is to be invested on day t+1 only if the latest "
    "observed volatility at t is at or below its expanding 75th percentile (computed from prior observations, "
    f"252-observation warm-up); otherwise zero exposure. On the common sample {dstart} to {dend}, <b>{best_d}</b> is best: "
    f"Sharpe {d_best.sharpe:.5f} vs {d_b.sharpe:.5f} unconditioned, CAGR {d_best.ann_geo_return:.5f} vs "
    f"{d_b.ann_geo_return:.5f}, maximum drawdown {d_best.max_drawdown:.5f} vs {d_b.max_drawdown:.5f}, invested "
    f"{pct(d_best.frac_invested)} of days. The (c) candidate {candidate} ranks "
    f"{int(list(ranks_d.index).index(candidate + '-cond')) + 1} of 6 (Sharpe {d_cand.sharpe:.5f}). "
    f"So (c) points to the right family of variables (AUD-linked FX implied vol) but not to the exact winner. "
    "This is why all six have to be tested: a descriptive full-sample association does not determine the real-time "
    "behaviour of a threshold rule, including switch timing, missed rebounds and stale quotes.",
    f"(e)&nbsp;&nbsp; I use the 21-day realised volatility of AUD/JPY (X<sub>AUD</sub>/X<sub>JPY</sub>) "
    f"with the same rule. On {estart} to {eend} it improves on the baseline (Sharpe {e_rv.sharpe:.5f} vs {e_b.sharpe:.5f}, "
    f"maximum drawdown {e_rv.max_drawdown:.5f} vs {e_b.max_drawdown:.5f}) but trails {best_d} (Sharpe {e_iv.sharpe:.5f}, "
    f"drawdown {e_iv.max_drawdown:.5f}). In autumn 2008 the realised-vol signal switched off on "
    f"{ep(epi_e,'Sep-Dec 2008 crisis',rv_name,'first_off_signal_from_45d_before')}, versus "
    f"{ep(epi_e,'Sep-Dec 2008 crisis',best_d,'first_off_signal_from_45d_before')} for {iv}, and lost "
    f"{ep(epi_e,'Sep-Dec 2008 crisis',rv_name):.5f} over Sep&ndash;Dec 2008 versus "
    f"{ep(epi_e,'Sep-Dec 2008 crisis',best_d):.5f}. In August 2007 it reacted earlier. Overall the forward-looking implied "
    "measure works better, which matches intuition, though not uniformly.",
]
SUMM = ParagraphStyle("summ", parent=BODY, fontSize=9.6, leading=12.2, leftIndent=17, firstLineIndent=-17, spaceAfter=4)
for item in summary_items:
    P(item, SUMM)
P("All results are in-sample over a single period (1999&ndash;2010), and picking the best of 12 (or 6) variants is "
  "itself an in-sample choice, so none of the rankings should be read as a forecast.", SMALL)
story.append(PageBreak())

# ============================================================================= 1. data & conventions
h1("1. Data, conventions and methodology")
h2("1.1 Workbook inspection and data dictionary")
au = res["audit"]
P(f"The workbook has one worksheet (&lsquo;{au['sheet']}&rsquo;). The header row is Excel row {au['header_row_excel']}, "
  "and every column header was checked programmatically against the column letters given in the assignment "
  "(B&ndash;J spot, L&ndash;T returns plus U = USD ROR, W&ndash;AP rates, AR&ndash;AW volatilities); all matched. "
  f"There are {au['rows']} rows. Dates are strictly increasing with no duplicates, and the input was already chronological "
  f"(it is sorted defensively anyway). Only weekdays appear; consecutive dates are 1 or 3 calendar days apart. "
  "The 1-day return cells in L&ndash;T are live Excel formulas (e.g. EUR <font face='Courier'>=B3/B2-1</font>, JPY "
  "<font face='Courier'>=C2/C3-1</font>), so they are already expressed as the return of holding the foreign currency.")
dd_rows = [
    ["A", "Date", "Trading date (weekdays, includes non-US holidays)", "date", "0"],
    ["B-J", "EUR ... SEK", "Spot FX vs USD in market convention (see Table 2)", "price", "0"],
    ["L-T, U", "EUR ROR ... SEK ROR, USD ROR", "Workbook 1-day spot return of holding the currency; USD ROR = 0", "decimal", "1 (first row)"],
    ["W-AF", "EUR 1M ... USD 1M", "1-month interest rates, 10 currencies", "% p.a.", "0"],
    ["AG-AP", "EUR 3M ... USD 3M", "3-month interest rates, 10 currencies", "% p.a.", "0"],
    ["AR, AS, AT", "AY1wk, AY3M, AY6M", "AUD/JPY option implied vol, 1-week / 3-month / 6-month", "vol pts", "0"],
    ["AU, AV", "JPY3M, AUD3M", "USD/JPY and AUD/USD 3-month implied vol", "vol pts", "0"],
    ["AW", "VIX", "CBOE VIX (30-day S&P 500 implied vol)", "index pts", f"{res['n_vix_missing']} (US holidays + 1999-01-01)"],
]
table(pd.DataFrame(dd_rows, columns=["Excel cols", "Headers", "Content", "Units", "Missing"]).set_index("Excel cols"),
      col_widths=[48, 112, 225, 48, 92], caption="Table 1. Data dictionary (column-level detail in outputs/tables/data_dictionary.csv).")

chk = tab("return_check_vs_workbook", index_col=0)
quote = {"EUR": "USD per EUR", "JPY": "JPY per USD", "GBP": "USD per GBP", "CAD": "CAD per USD", "AUD": "USD per AUD",
         "NZD": "USD per NZD", "CHF": "CHF per USD", "NOK": "NOK per USD", "SEK": "SEK per USD", "USD": "-"}
orient = pd.DataFrame({
    "workbook quote": [quote[c] for c in chk.index],
    "X (USD per unit)": ["S" if quote[c].startswith("USD") else ("1" if c == "USD" else "1 / S") for c in chk.index],
    "max |computed - workbook return|": [f"{v:.2e}" for v in chk["max_abs_diff_vs_workbook"]],
    "max error if raw-quote return were negated": [f"{v:.5f}" if pd.notna(v) else "n/a" for v in chk["naive_negation_max_error"]],
}, index=chk.index)
orient.index.name = "currency"
table(orient, col_widths=[50, 80, 80, 130, 160],
      caption="Table 2. Exchange-rate orientation. Computed returns on USD-per-currency levels reproduce the workbook to "
              "machine precision. Merely negating the return of an inverted quote would be wrong by up to "
              f"{res['naive_negation_max_error']:.5f} on a single day.")

h2("1.2 Missing, stale and holiday observations")
bullets([
    "Spot rates and interest rates have no missing values. Rows exist for every weekday, including holidays such as "
    "1999-01-01 and US holidays; spot quotes on those days are kept as reported. A zero spot change on such a day is a genuine zero "
    "observation and is not filled in.",
    f"VIX is blank on {res['n_vix_missing']} dates (US exchange holidays, plus 1999-01-01 before the series starts). For "
    "signals we use the <i>latest observed</i> VIX (last value carried forward), never before its first observation "
    "(1999-01-04). Expanding thresholds are computed only from actually observed values, so carried-forward values are "
    "not double-counted. Volatility <i>changes</i> in part (c) are set to missing on days with no new observation.",
    "Stale quotes: AY1wk is unchanged on 99% of days in 1999&ndash;2002 and on "
    f"{pct(stale.loc[2003,'AY1wk'],0)} / {pct(stale.loc[2004,'AY1wk'],0)} of days in 2003 / 2004, so it is effectively "
    "a step function early in the sample. AY3M and AY6M are unchanged on roughly 10&ndash;40% of days before 2008. "
    "We keep the data as supplied but flag this in parts (c) and (d). Interest rates also have runs of unchanged values, "
    "which is normal for fixings.",
    "JPY 3M rates are negative on 376 days and CHF 3M on 1 day; the formulas handle negative rates without modification.",
], SMALL)

h2("1.3 Returns, carry and accrual")
P("Spot levels are standardised to X<sub>i,t</sub> = USD per unit of currency i. Inverted quotes are inverted <i>before</i> "
  "computing returns:")
eq(r"X_{i,t}=S_{i,t}\ \ (\mathrm{EUR,GBP,AUD,NZD}),\qquad X_{i,t}=1/S_{i,t}\ \ (\mathrm{JPY,CAD,CHF,NOK,SEK}),\qquad X_{USD,t}=1")
eq(r"r_{i,t+1}=\frac{X_{i,t+1}}{X_{i,t}}-1,\qquad \Delta_t=\frac{\mathrm{calendar\ days}(t,\,t+1)}{360}")
P("Interest rates are converted from % per annum to decimals. The rate of the chosen tenor observed at t accrues "
  "simple interest over the calendar interval to the next observation (ACT/360; Friday to Monday accrues 3 days). The "
  "annualised 3M rate is <i>not</i> divided by 3: both tenors are annualised quotes, and the tenor only selects which "
  "rate curve point is used for ranking and for the daily accrual. Holding a 1M or 3M deposit and rebalancing it daily "
  "at that rate is the standard textbook simplification, which we adopt. The funded USD excess return of a 1-USD "
  "long position in currency i is")
eq(r"e_{i,t+1}=(1+r_{i,t+1})(1+i_{i,t}\Delta_t)-(1+i_{USD,t}\Delta_t)\ \approx\ r_{i,t+1}+(i_{i,t}-i_{USD,t})\Delta_t")
P("This means: borrow 1 USD at the USD rate, convert to currency i, earn the foreign rate, and convert back at t+1. We "
  "use the exact product form, not the first-order approximation, which drops the small cross term "
  "r<sub>i,t+1</sub>i<sub>i,t</sub>&Delta;<sub>t</sub>. A <b>short</b> position (negative weight) is the mirror trade: "
  "borrow currency i at i<sub>i,t</sub>, convert to USD and lend at i<sub>USD,t</sub>. Its P&amp;L is exactly "
  "&minus;e<sub>i,t+1</sub>. For USD itself e<sub>USD</sub> = 0. Carry relative to USD and the portfolio return are")
eq(r"c_{i,t}=i_{i,t}-i_{USD,t},\qquad R_{t+1}=\sum_{i}w_{i,t}\,e_{i,t+1},\qquad \sum_{w_{i,t}>0}w_{i,t}=1,\ \ \sum_{w_{i,t}<0}w_{i,t}=-1")
P("Because the weights sum to zero, R<sub>t+1</sub> equals the USD return of the long-short book. It is a zero-cost "
  "excess return; no interest on posted collateral is added in any comparison. <b>Timing:</b> weights are set from rates "
  "known at the close of t and earn the return over (t, t+1]; volatility signals likewise use values known at t. "
  "Carry ranking (which currencies have the highest / lowest c) determines exposure direction (long / short); the "
  "magnitude rule only distributes the +1 and &minus;1 within each side.")

h2("1.4 Treatment of USD")
P("We treat USD as one of the 10 candidate currencies (primary interpretation). The assignment asks for portfolios "
  "&ldquo;based on the USD-based basket of currencies&rdquo; and the workbook supplies USD 1M and 3M rates in the same "
  "blocks as the other nine, plus a USD ROR column of zeros. Both point to a 10-currency G-10 ranking universe in which "
  "returns are measured in USD. When USD ranks in the top or bottom K it is simply held, earning zero excess return. "
  "Excluding USD (a 9-currency universe) is reported as a sensitivity check in &sect;7; it changes no conclusion.")

h2("1.5 Allocation rules")
P("No lecture definitions were supplied with the assignment, so these are our stated conventions, not definitions "
  "specified by the professor. Order each side from most to least extreme carry (j = 1 is the highest-carry long or the "
  "lowest-carry short), and let m<sub>t</sub> be the cross-sectional median carry of the universe:")
eq(r"\mathrm{EQ}:\ |w_{(j)}|=\frac{1}{K};\qquad \mathrm{RANK}:\ |w_{(j)}|=\frac{K+1-j}{K(K+1)/2};\qquad "
   r"\mathrm{PROP}:\ |w_{i}|=\frac{|c_{i,t}-m_t|}{\sum_{l\in\mathrm{side}}|c_{l,t}-m_t|}")
P("RANK gives weights 2/3, 1/3 (K = 2) and 1/2, 1/3, 1/6 (K = 3). PROP measures carry relative to the cross-sectional "
  "median rather than to USD. With K &le; 3 of 10 currencies, every long has carry at or above the median and every "
  "short at or below it, so magnitudes are always non-negative, even on days when all selected currencies have "
  "positive (or all negative) carry versus USD. Normalising signed USD carry instead would flip positions whenever, "
  "for example, USD is the highest-yielding currency. If every selected currency on a side equals the median "
  "(zero denominator), the side falls back to equal weights; this never happens in the sample. Ties in carry are "
  "broken by the fixed universe order.")

h2("1.6 Performance statistics")
eq(r"W_T=\prod_{t}(1+R_t),\quad \bar g=W_T^{1/N}-1,\quad \mathrm{CAGR}=(1+\bar g)^{260}-1")
eq(r"\mu_{ann}=260\,\bar R,\quad \sigma_{ann}=\sqrt{260}\,s_R,\quad SR=\frac{\mu_{ann}}{\sigma_{ann}}")
eq(r"DD_t=\frac{W_t}{\max(1,\max_{s\leq t}W_s)}-1,\quad \mathrm{VaR}_{95}=-q_{0.05}(R),\quad "
   r"\mathrm{ES}_{95}=-\mathrm{E}[R\,|\,R\leq q_{0.05}(R)]")
P("260 observations per year (the sample averages 260.8 weekdays a year). The Sharpe numerator is the annualised "
  "arithmetic mean of the daily <i>excess</i> return: the USD funding cost is already inside e<sub>i,t+1</sub>, so no "
  "further risk-free rate is subtracted. The geometric return comes from compounding the arithmetic daily returns. "
  "Annualising the arithmetic mean by 260 (simple) but the geometric mean by compounding means the annualised CAGR can "
  "slightly exceed 260&times;mean when volatility is low. The like-for-like comparison is the daily arithmetic versus "
  "daily geometric mean, both reported. VaR and ES are positive numbers meaning losses (fraction of capital per day). "
  "Skewness and excess kurtosis are bias-adjusted sample moments. Calmar = CAGR / |maximum drawdown|.")
story.append(PageBreak())

# ============================================================================= 2. part (a)
h1("2. Part (a): the 12 carry portfolios")
P("The 12 portfolios are all combinations of tenor (1M, 3M) &times; K (2, 3 currencies per side) &times; allocation "
  "(EQ, RANK, PROP). They are named <font face='Courier'>tenor-K-method</font>, e.g. "
  "<font face='Courier'>3M-K3-EQ</font>. Each day the steps are: (1) compute c<sub>i,t</sub> for the 10 currencies from "
  "the tenor's rates; (2) select the K highest (long) and K lowest (short); (3) assign magnitudes with the allocation "
  "rule; (4) apply the weights to the return over (t, t+1]. Holdings therefore change daily whenever the carry ranking "
  "changes, and PROP magnitudes change every day. Table 3 shows one clearly dated example.")
ex = tab("a_example_weights_2007-06-29", index_col=0)[["long", "short"]]
ex.index.name = "portfolio"
table(ex, col_widths=[70, 225, 230], header=["long side (signed weight)", "short side (signed weight)"],
      caption=f"Table 3. Signed weights of all 12 portfolios set at the close of {res['example_date']} (applied to the return "
              f"from {res['example_date']} to {res['example_next']}). Long weights sum to +1 and short weights to -1 in every row.")
exr = tab("a_example_rates_2007-06-29", header=[0, 1], index_col=0)
exr.columns = ["1M rate (%)", "3M rate (%)", "1M carry vs USD (%)", "3M carry vs USD (%)"]
exr.index.name = "currency"
table(exr, col_widths=[55, 75, 75, 95, 95], digits=5,
      caption=f"Table 4. Inputs on {res['example_date']}. NZD, AUD and GBP have the highest carry, and JPY, CHF and SEK the lowest. "
              "USD (5.32%) ranks 4th, so it is not selected on this date.")
last = tab(f"a_example_weights_{res['last_signal_date']}", index_col=0)
usd_w = pd.read_csv(OUT / "histories" / "baseline_weights_daily.csv", index_col=0, parse_dates=True)["3M-K3-EQ|USD"]
usd_short_2010 = (usd_w.loc["2010"] < 0).mean()
P(f"For comparison, on the last signal date ({res['last_signal_date']}) the 3M-K3-EQ portfolio is long "
  f"{last.loc['3M-K3-EQ','long']} and short {last.loc['3M-K3-EQ','short']}. In 2010, with US rates near zero, USD "
  f"enters and leaves the short basket: it is shorted on {pct(usd_short_2010, 0)} of 2010 signal dates. This is why "
  "holdings must be read as daily histories, not as one fixed list.")
sel_p = sel.copy()
sel_p["cfg"] = sel_p["tenor"] + " K=" + sel_p["K"].astype(str)
lo = sel_p.pivot(index="currency", columns="cfg", values="pct_days_long")
sh = sel_p.pivot(index="currency", columns="cfg", values="pct_days_short")
selt = pd.concat({"% days long": lo, "% days short": sh}, axis=1)
selt.columns = [f"{a[:-1].replace('% days ','')} {b}" for a, b in selt.columns]
selt = selt.loc[["EUR", "JPY", "GBP", "CAD", "AUD", "NZD", "CHF", "NOK", "SEK", "USD"]]
selt.index.name = "currency"
table(selt, digits=2, col_widths=[45] + [60] * 8,
      header=[c.replace("lon ", "long ").replace("shor ", "short ") for c in selt.columns],
      caption="Table 5. Selection frequency (% of signal dates on which a currency is in the long or short set; it is the "
              "same for all three allocation methods). JPY and CHF are almost always funding currencies and NZD/AUD almost "
              "always investment currencies. USD is an investment (long) currency in 1999-2000 and 2006, when US rates were high, "
              "and a funding (short) currency in 2001-2004 and from late 2008.")
comp_t = comp.copy()
comp_t.index = comp_t["tenor"] + " K=" + comp_t["K"].astype(str)
comp_t = comp_t.drop(columns=["tenor", "K"])
comp_t.index.name = "config"
table(comp_t, digits=2, col_widths=[50, 85, 75, 90, 75, 75],
      header=["days with composition change", "% of days", "avg days between changes", "distinct long sets", "distinct short sets"],
      caption="Table 6. How often the selected currencies change. 3M rates are noisier, so the 3M ranking changes about three times as often.")
figure("a_selection_frequency.png", "Figure 1. Selection frequency by currency (green = long, red = short).")
figure("a_weights_through_time_3M_K3.png", "Figure 2. Daily signed weights for the three 3M-K3 allocation variants. "
       "EQ and RANK change only when the ranking changes; PROP moves continuously with the carry spreads.")
P("The file outputs/histories/baseline_weights_daily.csv holds every signed weight (columns "
  "<font face='Courier'>strategy|currency</font>, indexed by signal date) and baseline_returns_daily.csv holds the "
  "12 return streams (indexed by return date). Every day and every portfolio was validated: exactly K longs and K shorts, "
  "long weights sum to +1 and short weights to &minus;1, every long has carry at or above every unselected currency, "
  "and every short at or below (outputs/tables/weight_validation.csv).", SMALL)
story.append(PageBreak())

# ============================================================================= 3. part (b)
h1("3. Part (b): comparing the return distributions")
P(f"All 12 portfolios are evaluated on the same {int(T['n_obs'].iloc[0])} daily returns ({T['start'].iloc[0]} to "
  f"{T['end'].iloc[0]}).")
t1 = pd.DataFrame({
    "mean/day (%)": T["mean_daily"] * 100, "geo mean/day (%)": T["geo_mean_daily"] * 100,
    "ann. arith. mean": T["ann_arith_mean"], "CAGR": T["ann_geo_return"], "ann. vol": T["ann_vol"],
    "Sharpe": T["sharpe"], "Sortino": T["sortino"], "skew": T["skew"], "excess kurt.": T["excess_kurt"],
    "final wealth": T["final_wealth"]})
t1.index.name = "portfolio"
table(t1, font=7, col_widths=[60] + [46.5] * 10, bold_rows=[best],
      caption="Table 7. Return moments (decimals unless marked %). The highlighted row is the selected best baseline.")
t2 = pd.DataFrame({
    "max DD": T["max_drawdown"], "DD peak": T["dd_peak"], "DD trough": T["dd_trough"], "recovered": T["dd_recovery"],
    "VaR95/day": T["var95_daily"], "ES95/day": T["es95_daily"], "worst day": T["worst_day"],
    "worst date": T["worst_day_date"], "% days > 0": T["frac_positive"] * 100, "Calmar": T["calmar"]})
t2.index.name = "portfolio"
table(t2, font=7, col_widths=[60, 44, 52, 52, 50, 44, 44, 44, 52, 38, 40], bold_rows=[best],
      caption="Table 8. Downside statistics. Every drawdown peaks in July 2007 and troughs on 2009-02-02; none has recovered "
              "by the end of the sample. VaR/ES are positive loss fractions per day.")
figure("b_wealth_12.png", "Figure 3. Cumulative wealth (log scale), grouped by tenor and K; colours are allocation methods.")
figure("b_drawdowns_12.png", "Figure 4. Drawdowns. The 2008 carry crash dominates every portfolio; K = 3 books lose about 31-39% versus 38-45% for K = 2.")
figure("b_distribution.png", "Figure 5. Left: daily return distributions. Right: left-tail quantiles. The K = 3 portfolios (solid) "
       "have uniformly smaller tail losses than K = 2 (dashed).")

h2("3.1 Matched comparisons")
P("To isolate one factor we compare portfolios that differ <i>only</i> in that factor (6 tenor pairs, 6 K pairs, "
  "4 allocation triplets) and average the within-group range (max &minus; min) of each statistic.")
fi = fimp[["n_groups", "avg_range_sharpe", "avg_range_ann_arith_mean", "avg_range_ann_geo_return", "avg_range_ann_vol",
           "avg_range_max_drawdown", "avg_range_es95_daily", "avg_min_pair_corr"]].copy()
fi.index = ["rate tenor (1M vs 3M)", "number of currencies (K=2 vs 3)", "allocation (EQ/RANK/PROP)"]
fi.index.name = "factor varied"
table(fi, col_widths=[115, 38, 52, 52, 52, 52, 52, 52, 60],
      header=["groups", "Sharpe", "arith. mean", "CAGR", "ann. vol", "max DD", "ES95", "min. return corr"],
      caption="Table 9. Average matched range per factor. Larger = the factor matters more. The last column is the lowest "
              "correlation between daily returns within a matched group.")
mk = []
for t in ["1M", "3M"]:
    for m in ["EQ", "RANK", "PROP"]:
        a, b = f"{t}-K2-{m}", f"{t}-K3-{m}"
        mk.append({"pair": f"{t}-{m}", "Sharpe K2": T.loc[a, "sharpe"], "Sharpe K3": T.loc[b, "sharpe"],
                   "vol K2": T.loc[a, "ann_vol"], "vol K3": T.loc[b, "ann_vol"], "CAGR K2": T.loc[a, "ann_geo_return"],
                   "CAGR K3": T.loc[b, "ann_geo_return"], "maxDD K2": T.loc[a, "max_drawdown"], "maxDD K3": T.loc[b, "max_drawdown"],
                   "ES95 K2": T.loc[a, "es95_daily"], "ES95 K3": T.loc[b, "es95_daily"]})
mk = pd.DataFrame(mk).set_index("pair")
table(mk, font=7, col_widths=[50] + [47.5] * 10, caption="Table 10. K = 2 versus K = 3, holding tenor and allocation fixed.")
mt = []
for k in [2, 3]:
    for m in ["EQ", "RANK", "PROP"]:
        a, b = f"1M-K{k}-{m}", f"3M-K{k}-{m}"
        mt.append({"pair": f"K{k}-{m}", "Sharpe 1M": T.loc[a, "sharpe"], "Sharpe 3M": T.loc[b, "sharpe"],
                   "CAGR 1M": T.loc[a, "ann_geo_return"], "CAGR 3M": T.loc[b, "ann_geo_return"],
                   "maxDD 1M": T.loc[a, "max_drawdown"], "maxDD 3M": T.loc[b, "max_drawdown"]})
mt = pd.DataFrame(mt).set_index("pair")
table(mt, col_widths=[50] + [68] * 6, caption="Table 11. 1M versus 3M, holding K and allocation fixed.")
figure("b_factor_sharpe.png", "Figure 6. Sharpe ratios grouped by each construction factor.", width=W_AVAIL * 0.95)

h2("3.2 Selecting the best baseline")
P("Decision rule, fixed before looking at rankings: average the ranks of (i) the Sharpe ratio (risk-adjusted "
  "return), (ii) the Calmar ratio (compounded return per unit of worst drawdown) and (iii) the daily 95% expected "
  "shortfall (tail loss), with ties broken by Sharpe. Cumulative return alone is not used.")
rk = ranks.copy()
rk.index.name = "portfolio"
table(rk, col_widths=[70, 70, 70, 70, 70, 70], header=["Sharpe rank", "Calmar rank", "ES95 rank", "average rank", "Sharpe"],
      bold_rows=[best], caption="Table 12. Ranking under the decision rule.")
P(f"<b>{best}</b> ranks first, with the highest Sharpe ({Bm.sharpe:.5f}), the lowest volatility ({Bm.ann_vol:.5f}) and "
  f"the smallest tail loss (ES95 {Bm.es95_daily:.5f} per day). 1M-K3-EQ is a near tie (Sharpe "
  f"{T.loc['1M-K3-EQ','sharpe']:.5f}, best Calmar {T.loc['1M-K3-EQ','calmar']:.5f}); the two are 98% correlated, so the "
  f"choice of tenor is immaterial. {best} does <i>not</i> have the highest raw return: 3M-K2-RANK earns "
  f"{T.loc['3M-K2-RANK','ann_arith_mean']:.5f} a year arithmetically, but with 35% more volatility and a "
  f"{T.loc['3M-K2-RANK','max_drawdown']:.5f} drawdown. The difference in Sharpe between the top portfolios is small "
  "relative to its sampling error (roughly 1/&radic;11.4 &asymp; 0.3 for an 11-year sample). So the ranking says which "
  "construction was more efficient in this sample, not that it will win in the future.")
h2("3.3 Discussion")
bullets(prose=True, items=[
    f"<b>Number of currencies is the dominant factor.</b> Moving from K = 2 to K = 3 raises Sharpe in {k_wins}/6 pairs, "
    f"lowers volatility by about {fimp.loc['K','avg_range_ann_vol']:.5f} a year and improves maximum drawdown by about "
    f"{fimp.loc['K','avg_range_max_drawdown']:.5f}. The mean return barely changes. The 3rd-ranked currencies (GBP/NOK long, "
    "USD/SEK/EUR short) carry less than the extremes, but adding them diversifies the currency-specific crash risk of "
    "NZD, AUD and JPY. A K = 2 book is mostly a single NZD/AUD-vs-JPY/CHF bet. This is intuitive: carry's premium comes "
    "from a common factor, so extra names dilute idiosyncratic noise faster than they dilute the premium.",
    f"<b>Allocation matters second.</b> RANK and PROP tilt towards the most extreme carry (NZD and JPY), which raises the "
    f"mean slightly (PROP has the highest CAGR within most groups) but raises volatility and tail risk more. EQ has the "
    f"best Sharpe for K = 3 at both tenors{'' if eq_k3_best else ' (not uniformly)'}. Concentrating on the highest-carry "
    "pair concentrates exactly the exposure that crashes together in risk-off episodes.",
    f"<b>Tenor matters least</b> (average Sharpe range {fimp.loc['tenor','avg_range_sharpe']:.5f}, return correlations "
    f"&ge; {fimp.loc['tenor','avg_min_pair_corr']:.3f}). 1M and 3M rates rank G-10 currencies almost identically; 3M "
    "rankings flip more often (Table 6) but produce nearly identical P&amp;L. This is intuitive because cross-country rate "
    "differentials are persistent and similar across the short end of the curve.",
    "<b>Common to all 12:</b> negative skewness (about &minus;0.45 to &minus;0.64), excess kurtosis of 7.5 to 11.6, "
    "worst days on 2008-10-24 or 2008-10-08, and a single drawdown episode (July 2007 to February 2009) that erases "
    "about four years of gains. This is the classic &ldquo;up the stairs, down the elevator&rdquo; carry profile. It "
    "motivates the volatility conditioning in parts (c) to (e).",
])
figure("b_rolling_sharpe.png", "Figure 7. Rolling one-year Sharpe ratio of the best baseline versus two K = 2 portfolios.")
story.append(PageBreak())

# ============================================================================= 4. part (c)
h1(f"4. Part (c): the best baseline ({best}) and volatility")
P("We separate two questions. <b>Contemporaneous association:</b> does the carry return over (t, t+1] move with the "
  "<i>change</i> in volatility over the same interval? <b>Predictive conditioning:</b> does the volatility <i>level</i> "
  "known at t say anything about the carry return after t? Only the second can justify a trading rule. Volatility "
  "regressors are standardised (beta per one standard deviation), and t-statistics use Newey-West standard errors "
  "(5 lags daily, 21 lags for overlapping 21-day sums).")
ct = contemp[["corr_R_dVol", "corr_R_dlogVol", "beta_per_1sd_dVol", "t_HAC", "R2", "n", "frac_zero_dVol"]].copy()
ct["beta_per_1sd_dVol"] = ct["beta_per_1sd_dVol"] * 1e4
ct.index.name = "vol series"
table(ct, col_widths=[60, 62, 62, 75, 55, 55, 50, 75],
      header=["corr(R, dVol)", "corr(R, dlogVol)", "beta (bp per 1 sd dVol)", "HAC t", "R2", "n", "share dVol = 0"],
      caption=f"Table 13. Contemporaneous: R<sub>t+1</sub> of {best} vs the change in volatility from t to t+1.")
pt = pred[["corr_level_t_R_t+1", "beta1d_level_bp", "t1d_level", "corr_level_t_R_21d", "beta21d_level_pct", "t21d_level",
           "corr_dVol_t_R_t+1", "t1d_dVol"]].copy()
pt.index.name = "vol series"
table(pt, col_widths=[60, 58, 58, 52, 58, 62, 52, 62, 52],
      header=["corr(v<sub>t</sub>, R<sub>t+1</sub>)", "beta 1d (bp/sd)", "HAC t", "corr(v<sub>t</sub>, R 21d)",
              "beta 21d (%/sd)", "HAC t (21d)", "corr(dv<sub>t</sub>, R<sub>t+1</sub>)", "HAC t"],
      caption="Table 14. Predictive (linear): future returns regressed on the volatility level v<sub>t</sub> (and its change dv<sub>t</sub>). "
              "R 21d is the sum of the next 21 daily returns.")
rg = regs.copy()
rg["vol"] = pd.Categorical(rg["vol"], ["AY1wk", "AY3M", "AY6M", "JPY3M", "AUD3M", "VIX"])
rg = rg.sort_values(["vol", "regime"], key=lambda s: s.map({"low": 0, "medium": 1, "high": 2}) if s.name == "regime" else s)
rg.index = rg["vol"].astype(str) + " " + rg["regime"]
rg = rg[["n", "v_min", "v_max", "ann_mean", "ann_vol", "sharpe", "es95_daily", "worst_day"]]
rg.index.name = "vol / tercile of v<sub>t</sub>"
table(rg, col_widths=[80, 35, 50, 50, 60, 55, 55, 60, 60],
      header=["n", "v min", "v max", "ann. mean R<sub>t+1</sub>", "ann. vol", "Sharpe", "ES95/day", "worst day"],
      caption="Table 15. Regimes: next-day carry statistics by full-sample tercile of the volatility level at t. Terciles use "
              "the whole sample, so this is a descriptive in-sample split, not a tradable rule.")
figure("c_scatter_contemporaneous.png", "Figure 8. Contemporaneous scatter: daily carry return vs the same-day change in each volatility series, with OLS fit.")
figure("c_regime_sharpe.png", "Figure 9. Next-day Sharpe ratio by volatility tercile.", width=W_AVAIL * 0.85)
figure("c_wealth_and_vols.png", "Figure 10. Best-baseline wealth and the six volatility series.")
cd = cand.copy()
cd.index.name = "vol series"
table(cd, col_widths=[60, 75, 60, 60, 75, 70, 75],
      header=["Sharpe low - high", "Sharpe low", "Sharpe high", "ES95 high regime", "HAC t (21d level)", "contemp. corr"],
      caption="Table 16. Candidate ranking. Primary criterion = low-minus-high tercile Sharpe spread of R<sub>t+1</sub> "
              "(predictive information only); tie-break = smaller high-regime ES. Other columns are shown for context.")
h2("4.1 Discussion")
bullets(prose=True, items=[
    "<b>Contemporaneously</b>, carry is clearly short volatility. All six correlations are negative and highly "
    f"significant (HAC |t| &ge; {contemp['t_HAC'].abs().min():.2f}). The VIX change has the strongest link "
    f"(corr {contemp.loc['VIX','corr_R_dVol']:.5f}, R<sup>2</sup> {contemp.loc['VIX','R2']:.5f}), followed by "
    f"JPY3M ({contemp.loc['JPY3M','corr_R_dVol']:.5f}). Days when equity and yen volatility jump are the days when funding "
    "currencies rally and investment currencies fall. The AUD/JPY 1-week series has a zero change on "
    f"{pct(contemp.loc['AY1wk','frac_zero_dVol'],0)} of days (stale quotes), which weakens its measured association.",
    "<b>Linear predictability is absent.</b> The level of volatility at t has essentially zero correlation with R<sub>t+1</sub> "
    f"(|corr| &le; {pred['corr_level_t_R_t+1'].abs().max():.3f}) and no significant 21-day slope (|t| &le; "
    f"{pred['t21d_level'].abs().max():.2f}). Mean returns are not linearly lower when volatility is high. The 2008 crash is "
    "followed by large 2009 rebounds while volatility was still very high, and these offset each other in a linear fit.",
    "<b>The regime split shows what matters for conditioning.</b> High volatility mainly means much higher <i>risk</i> "
    "(annualised volatility of R<sub>t+1</sub> roughly doubles, ES95 more than doubles) with a lower mean. "
    f"The Sharpe drop from the low to the high tercile is largest for <b>{candidate}</b> "
    f"({regp.loc[candidate,'low']:.5f} to {regp.loc[candidate,'high']:.5f}), then AY3M "
    f"({regp.loc['AY3M','low']:.5f} to {regp.loc['AY3M','high']:.5f}) and AY1wk. VIX and JPY3M are non-monotonic: "
    f"their high tercile has a <i>higher</i> Sharpe than the medium one (VIX {regp.loc['VIX','high']:.5f} vs "
    f"{regp.loc['VIX','medium']:.5f}), because extreme equity or yen volatility often marks the rebound phase after a crash.",
    f"<b>Candidate: {candidate}</b>, the 3-month AUD/USD implied vol. It is intuitive. AUD is held long on "
    f"{sel[(sel.tenor == best.split('-')[0]) & (sel.K == int(best.split('-')[1][1:])) & (sel.currency == 'AUD')]['pct_days_long'].iloc[0]:.0f}% "
    "of days and is the archetypal investment currency, so option-implied uncertainty about AUD measures the risk of "
    "the portfolio's largest long exposure directly. The AUD/JPY vols (AY3M in particular) are a close second; they price the "
    "canonical carry cross, long the top investment currency against the top funding currency. A 3-month tenor "
    "is a compromise between responsiveness and noise. The 1-week tenor reacts fastest but is stale before 2005, and "
    "the 6-month tenor is the smoothest and slowest. Equity vol (VIX) reflects the broad risk-off driver of carry "
    "unwinds, but it also spikes for equity-specific reasons (2001&ndash;2003) when carry did well.",
])
h2("4.2 Relationship to Jurek")
P("<b>Limitation:</b> the Jurek reading was not in the materials available to us, so we cannot verify its specific "
  "statements or numbers, and we attribute none to it. We test the relationship as the assignment frames it: FX carry "
  "returns versus FX implied volatility and equity (VIX) volatility. Our data <b>confirm a strong contemporaneous "
  "link</b>: carry losses coincide with jumps in FX implied vol and in the VIX, and the VIX co-moves most strongly. "
  "This fits the view that carry is exposed to global risk-off and crash risk. For our portfolio the <b>predictive</b> "
  "link is weaker and comes through risk rather than mean: high-volatility states are not reliably followed by "
  "lower average returns in a linear sense, but they are followed by far worse risk-adjusted returns and fatter left tails. "
  "On that dimension FX vol tied to the portfolio's own currencies beats equity vol.")
story.append(PageBreak())

# ============================================================================= 5. part (d)
h1("5. Part (d): conditioning on each implied-volatility series")
P("The same rule is applied to each of the six variables, with no optimisation:")
eq(r"q_t=Q_{0.75}\left(\{v_s:\ s<\tau(t)\}\right),\qquad s_t=\mathbf{1}\{v_{\tau(t)}\leq q_t\},\qquad R^{c}_{t+1}=s_t\,R_{t+1}")
P("Here &tau;(t) is the latest date &le; t on which the variable was observed. The threshold is the expanding 75th "
  "percentile of all <i>earlier</i> observations: the current observation enters the threshold only from the next day "
  "on, i.e. it is added after the signal is computed. At least 252 observations (about one year) are required before the "
  "first signal. When s<sub>t</sub> = 0 the carry book is flat and the excess return is exactly zero. The cut-off is the "
  "same round number for every variable, and the 75th percentile simply means &lsquo;volatility is in the top quarter of "
  "its own history so far&rsquo;. Because all six series are compared on the same dates, they and the baseline share "
  f"a common sample from {dstart} to {dend} ({int(Dm['n_obs'].iloc[0])} days), which starts after the VIX warm-up.")
dt = pd.DataFrame({
    "ann. mean": Dm["ann_arith_mean"], "CAGR": Dm["ann_geo_return"], "ann. vol": Dm["ann_vol"], "Sharpe": Dm["sharpe"],
    "skew": Dm["skew"], "ex. kurt": Dm["excess_kurt"], "max DD": Dm["max_drawdown"], "DD trough": Dm["dd_trough"],
    "ES95/day": Dm["es95_daily"], "worst day": Dm["worst_day"], "Calmar": Dm["calmar"]})
dt.index.name = "strategy"
table(dt, font=7, col_widths=[60] + [41] * 7 + [50] + [41] * 3, bold_rows=[best_d],
      caption=f"Table 17. {best} unconditioned vs conditioned on each implied-volatility series, {dstart} to {dend}.")
xt = pd.DataFrame({"% days invested": Dm["frac_invested"] * 100, "switches": Dm["n_switches"].astype(int).astype(str),
                   "off spells": Dm["n_off_spells"].astype(int).astype(str),
                   "avg off-spell (days)": Dm["avg_off_spell_days"], "% days > 0": Dm["frac_positive"] * 100,
                   "% invested days > 0": Dm["frac_positive_when_invested"] * 100, "DD peak": Dm["dd_peak"],
                   "DD recovery": Dm["dd_recovery"]})
xt.index.name = "strategy"
table(xt, font=7, digits=2, col_widths=[62, 55, 45, 45, 60, 50, 65, 60, 70], bold_rows=[best_d],
      caption="Table 18. Exposure behaviour. A &lsquo;switch&rsquo; is a change between invested and flat. "
              "% days > 0 counts flat days as non-positive.")
rd = ranks_d.copy()
rd.index.name = "strategy"
table(rd, col_widths=[70, 70, 70, 70, 70, 70], header=["Sharpe rank", "Calmar rank", "ES95 rank", "average rank", "Sharpe"],
      bold_rows=[best_d], caption="Table 19. Same decision rule as in part (b), applied to the six conditioned strategies.")
ep_t = epi.pivot(index="strategy", columns="episode", values="cum_return")[list(dict.fromkeys(epi.episode))]
ep_o = epi.pivot(index="strategy", columns="episode", values="first_off_signal_from_45d_before")[list(dict.fromkeys(epi.episode))]
ep_c = pd.concat([ep_t, ep_o.add_prefix("first off: ")], axis=1).loc[list(Dm.index)]
ep_c.index.name = "strategy"
table(ep_c.fillna(""), col_widths=[62, 68, 68, 68, 88, 88, 88],
      header=["Aug-2007 return", "Sep-Dec 2008 return", "Apr-Jun 2010 return", "first off (Aug-07)", "first off (Sep-08)", "first off (Apr-10)"],
      caption="Table 20. Stress episodes: cumulative return within each window and the first date the signal switched carry off "
              "(searching from 45 calendar days before the window).")
figure("d_wealth.png", "Figure 11. Wealth of the baseline and the six conditioned strategies on the common sample.")
figure("d_signals.png", "Figure 12. Each volatility series, its real-time expanding 75th-percentile threshold, and the periods "
       "with carry switched off (shaded).")
figure("d_drawdowns.png", "Figure 13. Drawdowns of the baseline and conditioned strategies.")
h2("5.1 Discussion")
order_d = list(ranks_d.index)
bullets(prose=True, items=[
    f"<b>Best: {best_d}</b> (the 3-month AUD/JPY implied vol). Sharpe {d_best.sharpe:.5f} vs {d_b.sharpe:.5f} for the "
    f"baseline, CAGR {d_best.ann_geo_return:.5f} vs {d_b.ann_geo_return:.5f}, volatility {d_best.ann_vol:.5f} vs "
    f"{d_b.ann_vol:.5f}, maximum drawdown {d_best.max_drawdown:.5f} vs {d_b.max_drawdown:.5f}, excess kurtosis "
    f"{d_best.excess_kurt:.5f} vs {d_b.excess_kurt:.5f}, invested {pct(d_best.frac_invested)} of days. It was "
    f"switched off from {ep(epi,'Sep-Dec 2008 crisis',best_d,'first_off_signal_from_45d_before')}, before Lehman, and "
    "sat out the whole autumn 2008 crash, so its maximum drawdown is a 2006 episode instead. The ranking is "
    f"{', '.join(o.replace('-cond','') for o in order_d)}. "
    "Every conditioned strategy beats the baseline on Sharpe, volatility and drawdown.",
    f"<b>(i) Agreement with (c).</b> Partial. Part (c) singled out {candidate} with AY3M a close second, and the AUD-linked "
    f"FX implied vols (AY3M, AY6M, AUD3M, AY1wk) do occupy the top four places. VIX and JPY3M, the two variables with "
    f"the strongest contemporaneous correlation, rank last, as the non-monotonic regime results predicted. But {candidate} "
    f"itself is third (Sharpe {d_cand.sharpe:.5f}). It has the lowest ES and the same drawdown as {best_d}, but it switches "
    f"{int(d_cand.n_switches)} times and is invested only {pct(d_cand.frac_invested)} of the time. Its real-time threshold "
    "flagged 2000&ndash;2001 and 2004 as high-vol periods when carry earned well (Figure 12). It also turned off only "
    f"after the worst August 2007 losses and partly missed the late-August rebound "
    f"({ep(epi,'Aug-2007 carry unwind',candidate + '-cond'):.5f} in that episode, worse than the baseline).",
    "<b>(ii) Agreement with intuition.</b> Yes, broadly. The 3-month AUD/JPY vol prices the exact risk of the carry book's "
    "core exposure (long AUD/NZD against JPY), at a tenor that is responsive but not noisy. Equity vol is a less "
    "specific signal: it was elevated in 2001&ndash;2003 (Figure 12, VIX switched off for long stretches) for "
    f"equity-specific reasons while carry did well, and it has the most switches ({int(Dm.loc['VIX-cond','n_switches'])}). "
    "JPY3M (USD/JPY vol) ranks last. Yen volatility mixes in USD/JPY-specific moves, and although it first switched "
    f"off early (July 2008) it whipsawed back in during September 2008 and lost "
    f"{ep(epi,'Sep-Dec 2008 crisis','JPY3M-cond'):.5f} in Sep&ndash;Dec 2008. The 1-week tenor carries little "
    "information before 2005 because its quotes are stale.",
    "<b>(iii) Why test all six.</b> Part (c) is descriptive and in-sample: it uses full-sample terciles, linear "
    "correlations and contemporaneous changes. A trading rule depends on things (c) does not measure: <i>when</i> the "
    "real-time threshold is crossed, how long the variable stays above it, how often it whipsaws, whether the off "
    "periods also miss the rebounds, and data quality such as staleness. Contemporaneous correlation is not "
    "predictability: VIX has the strongest association yet the second-worst rule. The ranking of conditioned strategies "
    "is therefore an empirical question that must be checked variable by variable. Picking the best of six is itself "
    "selection, so we also checked robustness (&sect;7, Table 26): AY3M stays first with an extra day of signal lag and "
    "(tied with AY1wk) with a 90th-percentile threshold, and the AY3M rule improves all 12 baselines, not just the one "
    "it was selected for.",
])
story.append(PageBreak())

# ============================================================================= 6. part (e)
h1("6. Part (e): conditioning on historical (realised) volatility")
P("For the historical measure I use the 21-observation realised volatility of AUD/JPY, the canonical carry cross "
  "(long the archetypal investment currency, short the archetypal funding currency). The best baseline is long AUD "
  "on most days and short JPY on every day. AUD/JPY is also the pair underlying the AY implied vols, so this is a direct "
  "realised-versus-implied comparison on the same asset. 21 trading days (about one month) roughly matches the "
  "30-day VIX horizon and the 1M rate tenor. The cross is built from the supplied spot data on standardised levels, "
  "and only past and current returns enter the estimate:")
eq(r"P_t=\frac{X_{AUD,t}}{X_{JPY,t}}\ (\mathrm{JPY\ per\ AUD}),\qquad RV_t=100\sqrt{260}\ \mathrm{sd}\left(\ln\frac{P_s}{P_{s-1}},\ s=t-20,\ldots,t\right)")
P("Log returns are used only to estimate volatility; strategy returns stay arithmetic. The conditioning rule is "
  "exactly the part (d) rule (expanding 75th percentile of prior RV values, 252-observation warm-up, flat when "
  f"above). The comparison sample from {estart} to {eend} is common to all three strategies and starts after the RV warm-up.")
et = pd.DataFrame({
    "ann. mean": Em["ann_arith_mean"], "CAGR": Em["ann_geo_return"], "ann. vol": Em["ann_vol"], "Sharpe": Em["sharpe"],
    "ex. kurt": Em["excess_kurt"], "max DD": Em["max_drawdown"], "DD trough": Em["dd_trough"],
    "ES95/day": Em["es95_daily"], "Calmar": Em["calmar"], "% invested": Em["frac_invested"] * 100,
    "switches": Em["n_switches"].astype(int).astype(str)})
et.index.name = "strategy"
table(et, font=6.6, col_widths=[76] + [36] * 6 + [46, 36, 36, 40, 32],
      caption=f"Table 21. Baseline vs best implied-vol rule ({best_d}) vs realised-vol rule, {estart} to {eend}.")
ee_t = epi_e.pivot(index="strategy", columns="episode", values="cum_return")[list(dict.fromkeys(epi_e.episode))]
ee_o = epi_e.pivot(index="strategy", columns="episode", values="first_off_signal_from_45d_before")[list(dict.fromkeys(epi_e.episode))]
ee_c = pd.concat([ee_t, ee_o.add_prefix("first off: ")], axis=1).loc[list(Em.index)]
ee_c.index.name = "strategy"
table(ee_c.fillna(""), col_widths=[94, 62, 62, 62, 76, 76, 76],
      header=["Aug-2007 return", "Sep-Dec 2008 return", "Apr-Jun 2010 return", "first off (Aug-07)", "first off (Sep-08)", "first off (Apr-10)"],
      caption="Table 22. Stress episodes for the part (e) comparison.")
ll = rv_iv["leadlag_corr_dRV_t_vs_dIV_t-lag"]
llt = pd.DataFrame({"corr(dRV<sub>t</sub>, d" + iv + "<sub>t-L</sub>)": [ll[str(L)] for L in range(-5, 6)]}, index=[str(L) for L in range(-5, 6)]).T
llt.index.name = "lag L (days)"
table(llt, col_widths=[95] + [39] * 11, digits=3,
      caption=f"Table 23. Lead-lag between daily changes in RV and in {iv}. Positive L = implied vol change L days earlier. "
              "Correlations at L > 0 exceed those at L < 0: implied vol tends to move first.")
figure("e_rv_vs_iv.png", f"Figure 14. Top: AUD/JPY 21-day realised vol, {iv} implied vol and the RV threshold. Bottom: wealth on the common sample.")
figure("e_episodes.png", "Figure 15. Cumulative returns within the three stress episodes.")
figure("e_drawdowns.png", "Figure 16. Drawdowns on the part (e) common sample.")
h2("6.1 Discussion")
bullets(prose=True, items=[
    f"<b>Realised-vol conditioning helps, but less than implied vol.</b> Versus the baseline, RV conditioning raises "
    f"Sharpe from {e_b.sharpe:.5f} to {e_rv.sharpe:.5f}, cuts volatility from {e_b.ann_vol:.5f} to {e_rv.ann_vol:.5f} and "
    f"maximum drawdown from {e_b.max_drawdown:.5f} to {e_rv.max_drawdown:.5f}. {best_d} does better on every headline "
    f"metric (Sharpe {e_iv.sharpe:.5f}, CAGR {e_iv.ann_geo_return:.5f} vs {e_rv.ann_geo_return:.5f}, maximum drawdown "
    f"{e_iv.max_drawdown:.5f}, ES95 {e_iv.es95_daily:.5f} vs {e_rv.es95_daily:.5f}). Both are invested about 74.5% "
    f"of the time, and their on/off states agree on {pct(rv_iv['agreement_RV_vs_best_IV'],1)} of days, so the "
    "difference comes from <i>timing</i>, not from how often each is invested.",
    f"<b>Backward-looking vs forward-looking.</b> RV is a backward-looking average of the last month's squared moves, so it "
    f"rises only after large moves have occurred. Implied vol is a forward-looking market price that also includes option "
    f"buyers' demand for crash protection (a risk premium), so it can rise before realised moves do. Table 23 supports "
    f"this: implied-vol changes lead RV changes. In 2008 the {iv} signal switched off on "
    f"{ep(epi_e,'Sep-Dec 2008 crisis',best_d,'first_off_signal_from_45d_before')}, while RV crossed its threshold on "
    f"{ep(epi_e,'Sep-Dec 2008 crisis',rv_name,'first_off_signal_from_45d_before')}, so the RV rule lost "
    f"{ep(epi_e,'Sep-Dec 2008 crisis',rv_name):.5f} in Sep&ndash;Dec 2008 versus {ep(epi_e,'Sep-Dec 2008 crisis',best_d):.5f}. "
    "RV also falls back quickly once daily moves calm down. The RV rule re-entered carry in August 2009, while the "
    f"{iv} rule stayed flat until March 2010 because option markets kept pricing elevated risk. The RV rule's maximum "
    f"drawdown runs from {e_rv.dd_peak} to {e_rv.dd_trough} and includes the September 2008 losses taken before it "
    "switched off.",
    f"<b>But not uniformly.</b> In the sudden August 2007 unwind the RV signal switched off earlier "
    f"({ep(epi_e,'Aug-2007 carry unwind',rv_name,'first_off_signal_from_45d_before')} vs "
    f"{ep(epi_e,'Aug-2007 carry unwind',best_d,'first_off_signal_from_45d_before')}) and lost less "
    f"({ep(epi_e,'Aug-2007 carry unwind',rv_name):.5f} vs {ep(epi_e,'Aug-2007 carry unwind',best_d):.5f}). Before 2007 "
    "AUD/JPY implied vol was low and sticky (partly stale), whereas a few large daily moves immediately lift a "
    "21-day RV. Responsiveness to stress therefore depends on the episode. RV reacts fast to realised shocks; implied "
    "vol can anticipate a build-up of risk. Over the whole sample the forward-looking measure produced better "
    "risk-adjusted results, which agrees with our intuition, but the evidence comes from a handful of episodes and "
    "should not be read as a guarantee.",
])
story.append(PageBreak())

# ============================================================================= 7. verification
h1("7. Verification, reconciliation and sensitivity")
h2("7.1 Manual reconciliation of one interval")
rc = recon.T.copy()
rc.columns = list(recon.index)
rows_keep = ["quote_convention", "quote_t", "quote_t+1", "X_t (USD/ccy)", "X_t+1 (USD/ccy)", "spot_return", "rate_t_pct",
             "usd_rate_t_pct", "delta", "excess_return", "weight", "contribution"]
rc = rc.loc[rows_keep]
for r in rows_keep[1:]:
    rc.loc[r] = rc.loc[r].astype(float)
rc.index.name = "item"
rc = rc.map(lambda v: v if isinstance(v, str) else float(v))
table(rc, font=7, digits=6, col_widths=[85] + [73] * len(recon.index),
      caption=f"Table 24. {best} from {res['recon']['t']} (Friday close) to {res['recon']['t1']} (Monday): weights from 3M carry at "
              f"{res['recon']['t']}; &Delta; = 3/360; e = (1+r)(1+i&Delta;) &minus; (1+i<sub>USD</sub>&Delta;); contribution = w &times; e. "
              f"Sum of contributions = {res['recon']['manual']:.10f}, identical to the pipeline value {res['recon']['pipeline']:.10f}.")
P("Check, e.g. JPY: the quote falls from 94.32 to 92.78 JPY per USD, so X<sub>JPY</sub> rises by 94.32/92.78 &minus; 1 = "
  "0.016598 (the yen appreciates). The accrual adds (1.016598)(1 + 0.0094&times;3/360) &minus; (1 + 0.0411&times;3/360), "
  "giving e = 0.016336. Weighted by &minus;1/3, this costs the short-yen book 0.005445 that day.", SMALL)
h2("7.2 Automated checks")
bullets([
    "Dates strictly increasing, no duplicates, weekdays only; worksheet headers match the assignment's column letters.",
    "Quote inversion: X<sub>JPY</sub> = 1/94.32 on 2008-10-24; computed spot returns equal the workbook's L&ndash;T columns "
    f"(max error {res['max_abs_diff_returns']:.1e}); negating raw-quote returns is shown to be wrong. The AUD/JPY cross equals "
    "the AUD quote &times; the JPY quote.",
    "Rates are decimals; Friday&rarr;Monday &Delta; = 3/360 and weekday &Delta; = 1/360; the excess-return formula is reproduced by hand.",
    "Weights for all 12 strategies on every day: K longs and K shorts, +1/&minus;1 sums, longs = top-K carry, shorts = bottom-K; "
    "PROP magnitudes stay non-negative when all carries share a sign; the zero-denominator fallback works; RANK = (3,2,1)/6.",
    "No look-ahead: changing all rates and spot data after a cut-off date leaves earlier weights and returns unchanged; "
    "changing volatility after a cut-off leaves earlier thresholds and signals unchanged; the threshold at date d equals the "
    "75th percentile of observations strictly before d; the first 251 thresholds are missing (warm-up).",
    "Timing: the conditioned return at t+1 uses the signal at t; the first conditioned return is missing, not zero.",
    "Compounding, geometric mean, annualisation and drawdown (peak, trough, recovery) checked on a hand-computed series.",
    f"No artificial zero returns: the 12 baselines have {res['baseline_zero_return_days']} exactly-zero days and no gaps. "
    "VIX is not filled before 1999-01-04. RV uses only the current and previous 20 returns.",
    "Common samples: part (b) uses one sample for all 12; part (d) uses one sample for the baseline and six rules; part (e) "
    "uses one sample for all three strategies.",
], SMALL)
h2("7.3 Sensitivity to conventions")
su = s_usd.copy()
su.index.name = "portfolio"
table(su, col_widths=[70, 100, 100, 100, 100],
      header=["Sharpe (USD in universe)", "Sharpe (USD excluded)", "max DD (USD in)", "max DD (USD excluded)"],
      caption=f"Table 25. USD as a constituent vs pure funding currency (9-currency ranking). The best baseline is unchanged "
              f"({sens['best_if_usd_excluded']}), and K = 3 still dominates.")
P(f"Switching the day count from ACT/360 to ACT/365 for all currencies moves the {best} Sharpe from "
  f"{sens['best_act360']['sharpe']:.5f} to {sens['best_act365']['sharpe']:.5f} and the CAGR from "
  f"{sens['best_act360']['ann_geo_return']:.5f} to {sens['best_act365']['ann_geo_return']:.5f}. This is immaterial.", SMALL)
rb = rob.copy()
rb.index.name = "signal variable"
table(rb, col_widths=[85, 110, 110, 110],
      header=["Sharpe: q75, no extra lag (primary)", "Sharpe: q75, signal lagged 1 more day", "Sharpe: q90, no extra lag"],
      caption="Table 26. Robustness of the conditioning conclusions (not used to choose the rule). The extra lag guards "
              "against the FX and VIX closes not being synchronous. AY3M remains first in each column (tied with AY1wk at q90), "
              f"and AUD/JPY-related FX vols stay ahead of VIX and JPY3M. Implied-vol rows and the baseline row use the part (d) "
              f"sample; the RV21_AUDJPY row uses the part (e) sample ({estart} onwards).")
gn = gen.copy()
gn.index.name = "baseline"
table(gn, col_widths=[70, 100, 100, 100, 100],
      header=["Sharpe unconditioned", f"Sharpe {iv}-conditioned", "max DD unconditioned", f"max DD {iv}-conditioned"],
      caption=f"Table 27. The {iv} rule applied to every baseline over {dstart} to {dend}. It improves all 12, so the result is not "
              "specific to the portfolio it was chosen for.")
story.append(PageBreak())

# ============================================================================= 8. assumptions + reproduction
h1("8. Assumptions and limitations")
bullets([
    "<b>Course definitions not available.</b> No lecture notes were in the workspace, so the definitions of rank-based and "
    "proportional-to-carry weights (&sect;1.5), the +1/&minus;1 exposure normalisation, ACT/360 accrual and the 260-day "
    "annualisation are our explicit assumptions, not conventions specified by the professor.",
    "<b>USD treated as a ranking constituent</b> (&sect;1.4); excluding it does not change conclusions (Table 25).",
    "<b>Rate tenor as an overnight rate.</b> We accrue the 1M or 3M annualised rate daily, as if the deposit could be "
    "rolled daily at that rate. A real 3M deposit locks its rate for three months, and forward points would differ slightly.",
    "<b>Excess returns only.</b> No collateral interest is added and no transaction costs are charged (as instructed). The "
    f"conditioned strategies switch {int(Dm['n_switches'].drop('baseline').min())}&ndash;{int(Dm['n_switches'].max())} times, and the baselines trade whenever rankings or PROP weights change, "
    "so costs would reduce all results, the conditioned ones somewhat more.",
    "<b>Timing.</b> The FX, rate and volatility data are treated as known at the same daily close. The sources' exact "
    "fixing times are not documented in the workbook (VIX is a US close). An extra day of lag (Table 26) leaves the "
    "conclusions unchanged.",
    "<b>Data quality.</b> AY1wk is stale for most of 1999&ndash;2004 and AY3M/AY6M are partly stale before 2008. We used "
    "the data as supplied and did not clean or replace any series.",
    "<b>In-sample evidence.</b> One 11.4-year sample dominated by a single crash (2008). Choosing the best of 12 baselines "
    "and 6 rules is in-sample selection; Sharpe differences among top candidates are well within sampling error. "
    "Nothing here implies out-of-sample superiority.",
    "<b>Jurek reading unavailable;</b> no specific claims are attributed to it (&sect;4.2).",
], SMALL)
h1("9. Reproduction")
P("From the hw3 directory (Python 3.12; package versions pinned in requirements.txt):", SMALL)
for cmd in ["python3.12 -m venv .venv &amp;&amp; .venv/bin/pip install -r requirements.txt",
            ".venv/bin/python scripts/run_all.py &nbsp;&nbsp;# analysis: tables, histories, figures, results.json",
            ".venv/bin/python -m pytest -q tests &nbsp;&nbsp;# 16 verification tests",
            ".venv/bin/python scripts/build_report.py &nbsp;&nbsp;# this PDF"]:
    P(f"<font face='Courier' size='7.5'>{cmd}</font>", SMALL)
P("Code: src/fxcarry/data.py (loading, layout check, quote standardisation), returns.py (spot and excess returns), "
  "portfolios.py (ranking, weights, validation), metrics.py (statistics, drawdowns, decision rule), volanalysis.py "
  "(part c), conditioning.py (thresholds, signals, realised vol). Outputs: outputs/tables (all tables as CSV plus "
  "hw3_tables.xlsx), outputs/histories (daily weights, returns, signals), outputs/figures.", SMALL)


# ----------------------------------------------------------------------------- build
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    if doc.page > 1:
        canvas.drawString(MARGIN, letter[1] - 0.55 * inch, f"{AUTHOR}")
        canvas.drawRightString(letter[0] - MARGIN, letter[1] - 0.55 * inch, "B9339 Homework 3")
    canvas.drawCentredString(letter[0] / 2, 0.5 * inch, str(doc.page))
    canvas.restoreState()


PDF.parent.mkdir(exist_ok=True)
doc = SimpleDocTemplate(str(PDF), pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN,
                        topMargin=0.8 * inch, bottomMargin=0.8 * inch, title="B9339 Homework 3: FX Carry",
                        author=AUTHOR)
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"wrote {PDF}")
