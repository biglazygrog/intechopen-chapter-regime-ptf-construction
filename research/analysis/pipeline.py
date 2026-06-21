"""
Main pipeline — fits the rolling GMM and saves the intermediate CSVs that
feed every downstream artefact:

- daily_regime_probabilities.csv          (K-varying model, used by AR forecast + backtest)
- paper_profiles_raw.csv                  (regime-conditional moments — feeds Table 3)
- correlation_pairwise_bootstrap.csv      (correlation differences — feeds Table 4)
- dominant_regime_forecast_eval.csv       (in-sample regime-prob comparison — feeds Figure 3)

Run:  python -m research.analysis.pipeline
Runtime: 5-15 minutes.
"""
import numpy as np
import pandas as pd

from data.reader1 import DataReader
from models.gmm import Optimiser
from models.regime_analysis import (
    hard_labels_from_daily_probs,
    RegimeDistributionalAnalysis,
)
from core.config import (
    K_CANDIDATES, WINDOW_SIZE, STEP, SCALE_METHOD,
    MAX_ITER, TOL, REG_COVAR, RANDOM_STATE,
    SHRINKAGE_TARGET, KAPPA_0, NU_0_EXTRA, LAMBDA_SCALE,
    ALPHA_DIRICHLET, EWMA_DECAY,
    FORECAST_K, OUTPUT_DIR,
)


def _build_optimiser(K_candidates):
    return Optimiser(
        K_candidates=list(K_candidates),
        window_size=WINDOW_SIZE,
        step=STEP,
        scale_method=SCALE_METHOD,
        max_iter=MAX_ITER,
        tol=TOL,
        reg_covar=REG_COVAR,
        random_state=RANDOM_STATE,
        regularise=True,
        kappa_0=KAPPA_0,
        nu_0_extra=NU_0_EXTRA,
        lambda_scale=LAMBDA_SCALE,
        init_decay=None,
        ewma_decay=EWMA_DECAY,
        allow_partial_last_window=False,
        alpha_dirichlet=ALPHA_DIRICHLET,
        order_components=True,
        order_mode="trace",
        shrinkage_target=SHRINKAGE_TARGET,
    )


def main():
    print("=" * 70)
    print("PIPELINE — rolling GMM + paper tables")
    print("=" * 70)
    print()

    # ----- Load data -----
    df = DataReader().read_retns().dropna()
    df = df[(df != 0).all(axis=1)]
    df = df[np.isfinite(df).all(axis=1)]
    X = df.values
    dates = df.index
    print(f"Data shape: {df.shape}  range: {dates[0].date()} → {dates[-1].date()}")
    print(f"Features: {df.columns.tolist()}")
    print()

    # ----- 1. K-varying rolling GMM -----
    print("Fitting K-varying rolling GMM (K in {2..7})...")
    opt = _build_optimiser(K_CANDIDATES)
    opt.fit(X, index=dates)

    daily = opt.get_daily_probabilities()
    daily.to_csv(OUTPUT_DIR / "daily_regime_probabilities.csv")
    print(f"  Saved daily_regime_probabilities.csv (K-varying, columns={daily.columns.tolist()})")

    # ----- 2. Forecast-only K=2 rolling GMM (for AR forecasting + backtest) -----
    print(f"\nFitting forecast-only K={FORECAST_K} rolling GMM...")
    opt_k2 = _build_optimiser([FORECAST_K])
    opt_k2.fit(X, index=dates)
    daily_k2 = opt_k2.get_daily_probabilities()
    daily_k2.to_csv(OUTPUT_DIR / "daily_regime_probabilities_forecast.csv")
    print(f"  Saved daily_regime_probabilities_forecast.csv (K={FORECAST_K})")

    # ----- 3. Regime-conditional moments (feeds Table 3) -----
    print("\nComputing regime-conditional moments (paper_profiles_raw)...")
    regime_labels = hard_labels_from_daily_probs(daily)
    regime_labels = pd.Series(regime_labels, index=daily.index, name="regime")

    dist = RegimeDistributionalAnalysis(
        quantiles=(0.05, 0.10, 0.25),
        var_levels=(0.05, 0.10),
        min_n=50,
        bootstrap_B=1000,
        random_state=42,
    )
    tables = dist.paper_tables(
        df_returns=df,
        regime_labels=regime_labels,
        pair_mode="ordered",
        alpha_main=0.05,
    )
    tables["paper_profiles_raw"].to_csv(OUTPUT_DIR / "paper_profiles_raw.csv", index=False)
    print(f"  Saved paper_profiles_raw.csv (feeds Table 3)")

    # ----- 4. Correlation difference tests (feeds Table 4) -----
    print("\nComputing pairwise correlation differences (bootstrap B=1000)...")
    corr_tests = dist.correlation_difference_tests(
        df_returns=df,
        regime_labels=regime_labels,
        bootstrap_B=1000,
        ci_level=0.95,
    )
    corr_tests["correlation_pairwise"].to_csv(
        OUTPUT_DIR / "correlation_pairwise_bootstrap.csv", index=False
    )
    print(f"  Saved correlation_pairwise_bootstrap.csv (feeds Table 4)")

    # ----- 5. Dominant-regime forecast eval (feeds Figure 3) -----
    print("\nBuilding dominant_regime_forecast_eval.csv (feeds Figure 3)...")
    _save_dominant_forecast_eval(daily_k2)

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE — outputs in {OUTPUT_DIR}")
    print("=" * 70)


def _save_dominant_forecast_eval(daily_k2: pd.DataFrame) -> None:
    """One-step dominant-regime persistence vs Markov-style RW baseline.

    For each day t we compare:
      - p_ar_t1 : AR-style 1-day-ahead forecast (here: persistence at t-1)
      - p_rw_t1 : random-walk forecast (just yesterday's level)
      - z_t1     : realised hard label at t+1
    """
    p1 = daily_k2["p_1"].astype(float)
    # Both naive forecasters use the previous day's value; we keep this consistent
    # with the legacy CSV the dissertation figure script reads.
    forecast_ar = p1.shift(1)
    forecast_rw = p1.shift(1)
    realised = (p1 > 0.5).astype(int)

    out = pd.DataFrame({
        "date_t":  p1.index.shift(-1),
        "date_t1": p1.index,
        "p_ar_t1": forecast_ar.values,
        "p_rw_t1": forecast_rw.values,
        "z_t1":    realised.values,
    }).dropna()
    out.to_csv(OUTPUT_DIR / "dominant_regime_forecast_eval.csv", index=False)
    print(f"  Saved dominant_regime_forecast_eval.csv (n={len(out)})")


if __name__ == "__main__":
    main()
