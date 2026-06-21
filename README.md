# Regime-Aware Portfolio Construction — Reproduction Package

This folder accompanies *Regime-Aware Portfolio Construction: A Gaussian Mixture Model Approach to Multi-Asset Allocation* (Bevza & Wyse). It contains the code and data needed to reproduce every figure and table in the chapter.

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
    │   └── backtest.py                    # Figure 4 backtest
    ├── figures/
    │   ├── figure1_regime_probabilities.py
    │   ├── figure2_forecast_accuracy.py
    │   ├── figure3_rolling_hit_rate.py
    │   └── figure4_backtest.py
    └── output_charts/
        ├── figures/                       # produced PNG / PDF files
        └── tables/                        # produced CSV / TSV files
```

## Paper-artefact → producer-script map

| Paper item | Output file | Producer |
|---|---|---|
| Figure 1 — Posterior regime probabilities | `figures/figure1_regime_probabilities.png` | `research.figures.figure1_regime_probabilities` |
| Figure 2 — Forecast accuracy by horizon  | `figures/figure2_forecast_accuracy.png`    | `research.figures.figure2_forecast_accuracy`    |
| Figure 3 — Rolling hit rate AR vs RW     | `figures/figure3_rolling_hit_rate.png`     | `research.figures.figure3_rolling_hit_rate`     |
| Figure 4 — Backtest (4-panel)            | `figures/figure4_backtest.png`             | `research.figures.figure4_backtest`             |
| Table 1 — Baseline Asset Universe        | (hand-written)                              | `data/data_master_v1.csv`                       |
| Table 2 — Expanded Asset Universe        | (hand-written)                              | `data/data_master2.csv`                         |
| Table 3 — Regime-conditional moments     | `tables/table3_regime_moments.tsv`         | `research.analysis.moments`                     |
| Table 4 — Cross-asset correlations       | `tables/table4_correlations.tsv`           | `research.analysis.correlations`                |
| Table 5 — Universe-expansion stability   | `tables/table5_universe_stability.tsv`     | `research.analysis.stability`                   |
| Table A.1 — Window-size robustness       | `tables/tableA1_window_robustness.tsv`     | `research.analysis.robustness`                  |
| Appendix — κ₀ prior-weight sensitivity   | `tables/tableA2_kappa_sensitivity.tsv`     | `research.analysis.shrinkage`                   |

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
- The forecast-only K = 2 model is fit separately from the K-varying selection used for regime identification.
