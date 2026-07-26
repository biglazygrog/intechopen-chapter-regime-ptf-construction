# Regime-Aware Portfolio Construction — Reproduction Package

This folder accompanies *Regime-Aware Portfolio Construction: A Gaussian Mixture Model Approach to Multi-Asset Allocation* (Bevza & Wyse). It contains the code and data needed to reproduce every figure and table in the chapter.

> **Read this first:** the package produces **two** families of regime-probability series — a *retrospective* one for descriptive figures and a *one-sided* one for everything forecast- or backtest-related. They are not interchangeable. See [Retrospective vs one-sided regime probabilities](#retrospective-vs-one-sided-regime-probabilities) below. `REVISION_LOG.md` records the revision history and scope decisions.

## Repository layout

```
submission/
├── README.md                              # this file
├── requirements.txt                       # pinned Python dependencies
├── core/
│   ├── config.py                          # all hyperparameters & file paths
│   └── utils.py                           # shared helpers (hard labels, sig stars, etc.)
├── data/
│   ├── reader1.py                         # DataReader (produces log returns)
│   ├── raw_data.xlsx / raw_data_eur.xlsx       # core 6-asset universe (USD / EUR)
│   ├── raw_data2.xlsx / raw_data2_eur.xlsx     # expanded 15-asset universe
│   └── data_master_v1.csv / data_master2.csv   # ticker → name maps
├── models/
│   ├── gmm.py                             # MyGMM + rolling Optimiser
│   ├── cov_models.py                      # mclust covariance families
│   ├── forecasting.py                     # AR / random-walk forecasters
│   ├── portfolio.py                       # PortfolioOptimizer + backtest engines
│   ├── regime_analysis.py                 # RegimeDistributionalAnalysis (Tables 3 + 4)
│   └── plotting.py                        # shared plotting helpers
└── research/
    ├── analysis/
    │   ├── pipeline.py                    # main driver — fit GMM + intermediate CSVs
    │   ├── moments.py                     # Table 3 formatter
    │   ├── correlations.py                # Table 4 formatter
    │   ├── stability.py                   # Table 5 (universe-expansion analysis)
    │   ├── stability_analyzer.py          # underlying analyser class
    │   ├── robustness.py                  # Table A.1 (window-size grid)
    │   ├── shrinkage.py                   # Appendix κ₀ sensitivity
    │   ├── forecast_eval.py               # multi-horizon AR forecast (feeds Fig 2)
    │   ├── backtest.py                    # Figure 4 backtest
    │   └── validate_onesided.py           # no-look-ahead validation suite (T1/T2/T3/T5)
    ├── figures/
    │   ├── figure1_regime_probabilities.py
    │   ├── figure2_forecast_accuracy.py
    │   ├── figure3_rolling_hit_rate.py
    │   └── figure4_backtest.py
    └── output_charts/
        ├── figures/                       # produced PNG / PDF files
        └── tables/                        # produced CSV / TSV files
```

## Retrospective vs one-sided regime probabilities

The pipeline emits four probability files. **Which one an artefact consumes is a
substantive methodological choice, not a detail.**

### Retrospective (smoothed, full-sample) — descriptive use only

`Optimiser.get_daily_probabilities()` assigns each window's *entire in-sample*
responsibility matrix to that window's date range, then averages across all
overlapping windows. With `WINDOW_SIZE = 1250` and `STEP = 63`, about **20
windows cover every date**, and the latest of them is fitted on returns running
roughly **1200 observations past that date**. Two distinct leaks are present:
within a window, responsibilities are posteriors evaluated on the same data that
produced the parameters (and K is chosen by BIC over the whole window); across
windows, dates are averaged over fits that *begin* after them.

Measured on this dataset, every row except the last carries look-ahead:

| Date | Windows averaged | Latest window ends | Look-ahead |
|---|---|---|---|
| 2000-10-31 | 1 | 2006-01-27 | 1249 obs |
| 2006-01-27 | 20 | 2010-11-02 | 1197 obs |
| 2013-01-16 | 20 | 2017-11-07 | 1210 obs |

This series is the right object for **describing the estimated regime history**,
which is what Figure 1 and Tables 3-4 do, and for the estimation-stability
diagnostics in Tables A.1 and A.2. It is **not** an investable signal.

### One-sided (filtered) — primary series

`Optimiser.get_filtered_probabilities()` serves each date from exactly one
model, whose estimation window ended **strictly before** that date:

```
p_t  = model_w.predict_proba(x_t)
w(t) = argmax_w { e_w : e_w < t }
```

Model *w*, trained on `[s_w, e_w]`, serves dates `e_w + 1 … e_{w+1}`. Only the
E-step is evaluated on those dates — no refitting — so no return dated after *t*
touches `p_t`. This is stricter than the usual filtered convention, which would
let the training window include `x_t` itself.

Because no admissible model exists before the first window closes, the one-sided
series **starts at observation index 1250 (2006-01-30)** and earlier dates are
omitted, not back-filled. It runs to 2026-01-23 — further than the retrospective
series, which stops at the last window end. The evaluation samples behind
Figures 2-4 are correspondingly shorter, and each figure annotates its realised
sample.

### File map

| File | Series | Consumed by |
|---|---|---|
| `daily_regime_probabilities.csv` | retrospective, K-varying | Figure 1, Tables 3 & 4 |
| `daily_regime_probabilities_forecast.csv` | retrospective, K=2 | nothing — retained for reference |
| `daily_regime_probabilities_onesided.csv` | one-sided, K-varying | nothing — **for descriptive/reviewer use only** |
| `daily_regime_probabilities_forecast_onesided.csv` | one-sided, K=2 | **Figures 2, 3, 4** |

The one-sided files carry `#`-prefixed provenance headers; read them with
`pd.read_csv(path, index_col=0, parse_dates=True, comment="#")`.

Forecast and backtest inputs use the **fixed-K=2** fit. In the K-varying series
the trace-ordered `p_1` is the *middle* of three regimes, not the high-vol one
`TARGET_REGIME` documents; `forecast_eval.py` and `backtest.py` both raise if
handed a series whose regime count differs from `FORECAST_K`.

### Validating the one-sided series

```bash
python -m research.analysis.validate_onesided          # T1, T3, T5
python -m research.analysis.validate_onesided --t2     # adds future-corruption test
python -m research.analysis.validate_onesided --full   # T1 at the paper spec
```

T1 (truncation invariance) is the decisive test: fitting on `X[:t]` alone must
reproduce the full-sample one-sided series exactly over the dates both can
serve. Any leakage — through responsibilities, BIC/K-selection, the EWMA
initialiser or the previous-epoch prior — breaks it.

## Paper-artefact → producer-script map

| Paper item | Output file | Producer | Probability series |
|---|---|---|---|
| Figure 1 — Posterior regime probabilities | `figures/figure1_regime_probabilities.png` | `research.figures.figure1_regime_probabilities` | retrospective, K-varying |
| Figure 2 — Forecast accuracy by horizon  | `figures/figure2_forecast_accuracy.png`    | `research.figures.figure2_forecast_accuracy`    | **one-sided, K=2** |
| Figure 3 — Rolling hit rate AR vs RW     | `figures/figure3_rolling_hit_rate.png`     | `research.figures.figure3_rolling_hit_rate`     | **one-sided, K=2** |
| Figure 4 — Backtest (4-panel)            | `figures/figure4_backtest.png`             | `research.figures.figure4_backtest`             | **one-sided, K=2** |
| Table 1 — Baseline Asset Universe        | (hand-written)                              | `data/data_master_v1.csv`                       | — |
| Table 2 — Expanded Asset Universe        | (hand-written)                              | `data/data_master2.csv`                         | — |
| Table 3 — Regime-conditional moments     | `tables/table3_regime_moments.tsv`         | `research.analysis.moments`                     | retrospective, K-varying |
| Table 4 — Cross-asset correlations       | `tables/table4_correlations.tsv`           | `research.analysis.correlations`                | retrospective, K-varying |
| Table 5 — Universe-expansion stability   | `tables/table5_universe_stability.tsv`     | `research.analysis.stability`                   | retrospective (estimation stability) |
| Table A.1 — Window-size robustness       | `tables/tableA1_window_robustness.tsv`     | `research.analysis.robustness`                  | retrospective (estimation stability) |
| Appendix — κ₀ prior-weight sensitivity   | `tables/tableA2_kappa_sensitivity.tsv`     | `research.analysis.shrinkage`                   | retrospective (estimation stability) |

## Reproducing the paper

```bash
cd submission
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 1) Core pipeline — produces all intermediate CSVs (5-15 min)
python -m research.analysis.pipeline

# 2) Multi-horizon AR forecast (feeds Figure 2)
python -m research.analysis.forecast_eval

# 3) Portfolio backtest (Figure 4, also auto-saves the figure)
python -m research.analysis.backtest

# 4) Tables
python -m research.analysis.moments         # Table 3
python -m research.analysis.correlations    # Table 4
python -m research.analysis.stability       # Table 5 (~15 min)
python -m research.analysis.robustness      # Table A.1 (~15 min)
python -m research.analysis.shrinkage       # Appendix κ₀ (~20 min)

# 5) Figures
python -m research.figures.figure1_regime_probabilities
python -m research.figures.figure2_forecast_accuracy
python -m research.figures.figure3_rolling_hit_rate
# (figure4 is produced by research.analysis.backtest above)
```

All numerical configuration (window size, κ₀, transaction cost, etc.) lives in `core/config.py`. The default values reproduce the headline numbers in the paper.

## Methodological notes

- **Returns are log returns** (`np.log(P).diff()`). Portfolio backtests use exact log-return compounding (`exp(cumsum)`) and wealth-conserving aggregation (`np.log(w · exp(r))`).
- The rolling GMM uses **previous-epoch shrinkage** for the mean prior with **κ₀ = 63.1** (target prior weight p ≈ 10 %). See Appendix κ₀ sensitivity for robustness.
- The forecast-only K = 2 model is fit separately from the K-varying selection used for regime identification. All forecast and backtest artefacts consume the **one-sided** K = 2 series.

### The synchronous-trading filter

`DataReader` forward-fills prices onto a daily grid, so an index that did not
update yields a log return of exactly zero. Such a row is not a real synchronous
cross-section: the joint return vector carries a spurious zero, which would
corrupt the cross-asset covariance structure the GMM relies on to identify
regimes. The filter therefore keeps only rows on which every **market index**
actually traded.

The single definition lives in `core.utils.filter_synchronous_trading` — it was
previously duplicated at five call sites that could silently drift apart.

**Cash is exempt.** `high_yield` is `LD12TRUU Index`, a Bloomberg US Short
Treasury 1-12 Month index (displayed as "Cash"). At a pinned policy rate its true
daily accrual falls below the stored price precision (~2.6bp on a level of 192
held to 2 d.p.), so a recorded zero is a rounding artefact of a genuinely
near-zero return, not a stale quote. Testing it alongside the market indices
accounted for **96.8%** of all dropped rows and removed most of 2009-2016 and
2020-2021 — including 85.2% of the EU sovereign debt crisis and 55.1% of the
COVID drawdown, leaving 13 observations in 2021 and 26 in 2013. That earlier
6-asset specification is superseded; see `REVISION_LOG.md`, items 8 and 9.

Under the corrected filter:

- 12018 raw rows → 6605 after `dropna` → **6269** after the filter.
- Every full year retains **249-253** observations; the residual 3-5% shortfall
  against 252 is genuine non-synchronous market holidays across five markets.
- `WINDOW_SIZE = 1250` spans **4.96-5.24 calendar years** — so "approximately
  five years" is accurate. `STEP = 63` and the `*_MIN_TRAIN` constants remain
  observation counts, not calendar days.
- `ANN_FACTOR = 248.48`, derived as 6269 observations / 25.229 calendar years
  (2000-10-31 to 2026-01-23), replaces the hardcoded 252.

Figure A.1 (`research.figures.figureA1_observations_per_year`) documents the
grid.

### Reading the forecast hit rates

The high-vol regime is rare, so a constant "always calm" classifier already
scores around 87 %. `forecast_eval.py` reports `hit_rate_always_calm` alongside
every model hit rate, and `ar_change_dissertation_table.tsv` carries both plus
the gap. Hit rates must be read against that baseline, not against 50 %.
