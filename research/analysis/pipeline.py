"""
Main pipeline — fits the rolling GMM and saves the intermediate CSVs that
feed every downstream artefact.

Two families of regime-probability series are produced:

RETROSPECTIVE (smoothed, full-sample) — DESCRIPTIVE USE ONLY. Each date is the
mean of the in-sample responsibilities of ~20 overlapping window fits, so it
embeds returns dated up to ~1200 observations AFTER that date.
- daily_regime_probabilities.csv           (K-varying — feeds Figure 1, Tables 3 & 4)
- daily_regime_probabilities_forecast.csv  (K=2 — retained for reference)

ONE-SIDED (no look-ahead) — PRIMARY SERIES. Each date is served by exactly one
model whose training data ends strictly before it. Starts at observation index
WINDOW_SIZE (2006-01-30); shorter than the retrospective series by construction.
- daily_regime_probabilities_onesided.csv           (K-varying — reviewer use only)
- daily_regime_probabilities_forecast_onesided.csv  (K=2 — feeds Figures 2, 3, 4)

Other outputs:
- paper_profiles_raw.csv                  (regime-conditional moments — feeds Table 3)
- correlation_pairwise_bootstrap.csv      (correlation differences — feeds Table 4)
- dominant_regime_forecast_eval.csv       (AR(1) vs RW on the one-sided series — feeds Figure 3)

Run:  python -m research.analysis.pipeline
Runtime: 5-15 minutes.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

from data.reader1 import DataReader
from models.gmm import Optimiser
from models.regime_analysis import RegimeDistributionalAnalysis
from core.utils import (
    hard_labels_from_daily_probs, logit, inv_logit, filter_synchronous_trading,
)
from core.config import (
    K_CANDIDATES, WINDOW_SIZE, STEP, SCALE_METHOD,
    MAX_ITER, TOL, REG_COVAR, RANDOM_STATE,
    SHRINKAGE_TARGET, KAPPA_0, NU_0_EXTRA, LAMBDA_SCALE,
    ALPHA_DIRICHLET, EWMA_DECAY,
    FORECAST_K, OUTPUT_DIR, AR_MIN_TRAIN,
    RETRO_PROBS_FILE, RETRO_FORECAST_PROBS_FILE,
    ONESIDED_PROBS_FILE, ONESIDED_FORECAST_PROBS_FILE,
)


RETRO_WARNING = (
    "[RETROSPECTIVE — full-sample smoothed; descriptive use only, "
    "do NOT use for forecasting or backtesting]"
)


def _save_with_provenance(df: pd.DataFrame, path, header_lines) -> None:
    """Write a CSV preceded by '#'-prefixed provenance lines.

    Readers must pass comment="#" to pandas.read_csv.
    """
    with open(path, "w", newline="") as fh:
        for line in header_lines:
            fh.write(f"# {line}\n")
        df.to_csv(fh)


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
    # Retain only business days on which every MARKET INDEX actually traded.
    # DataReader ffills prices onto a daily grid, so an index that did not
    # update yields a zero log return; such a row is not a real synchronous
    # cross-section and would corrupt the covariance structure the GMM uses to
    # identify regimes. Cash (LD12TRUU) is exempt — at a pinned policy rate its
    # zero is an economically genuine near-zero return, not a stale quote.
    # Single definition lives in core.utils; see REVISION_LOG.md items 8 and 9.
    df = filter_synchronous_trading(DataReader().read_retns().dropna())
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
    daily.to_csv(OUTPUT_DIR / RETRO_PROBS_FILE)
    print(f"  Saved {RETRO_PROBS_FILE} (K-varying, columns={daily.columns.tolist()})")
    print(f"    {RETRO_WARNING}")

    onesided = opt.get_filtered_probabilities()
    _save_with_provenance(
        onesided, OUTPUT_DIR / ONESIDED_PROBS_FILE,
        [
            "ONE-SIDED regime probabilities (K-varying). No look-ahead.",
            "Each date is served by exactly one model whose estimation window",
            "ended strictly before it; probabilities are E-step evaluations of",
            "that model on the date's own returns only.",
            "",
            "FOR DESCRIPTIVE / REVIEWER INSPECTION ONLY. Nothing in the pipeline",
            "consumes this file. K varies across refits (2 vs 3 here), so the",
            "meaning of a given column shifts between refits. All forecast and",
            f"backtest inputs use the fixed-K={FORECAST_K} one-sided file instead.",
            "",
            f"Produced by research.analysis.pipeline; window_size={WINDOW_SIZE}, step={STEP}.",
            "Read with: pd.read_csv(path, index_col=0, parse_dates=True, comment='#')",
        ],
    )
    print(f"  Saved {ONESIDED_PROBS_FILE} "
          f"(one-sided, K-varying, {len(onesided)} rows from {onesided.index[0].date()})")
    print("    [NOTE: descriptive/reviewer use only — no pipeline stage reads this file]")

    # ----- 2. Forecast-only K=2 rolling GMM (for AR forecasting + backtest) -----
    print(f"\nFitting forecast-only K={FORECAST_K} rolling GMM...")
    opt_k2 = _build_optimiser([FORECAST_K])
    opt_k2.fit(X, index=dates)

    daily_k2 = opt_k2.get_daily_probabilities()
    daily_k2.to_csv(OUTPUT_DIR / RETRO_FORECAST_PROBS_FILE)
    print(f"  Saved {RETRO_FORECAST_PROBS_FILE} (K={FORECAST_K})")
    print(f"    {RETRO_WARNING}")

    onesided_k2 = opt_k2.get_filtered_probabilities()
    _save_with_provenance(
        onesided_k2, OUTPUT_DIR / ONESIDED_FORECAST_PROBS_FILE,
        [
            f"ONE-SIDED regime probabilities (K={FORECAST_K}). No look-ahead.",
            "PRIMARY SERIES — feeds Figure 2 (forecast accuracy), Figure 3",
            "(rolling hit rate) and Figure 4 (portfolio backtest).",
            "",
            "For each date t: p_t = model_w.predict_proba(x_t), where model_w is",
            "the most recent fit whose training window ended STRICTLY before t.",
            "No return dated after t enters p_t.",
            "",
            "Trace-ordered: p_0 = lowest trace(Sigma) = calm, p_1 = high-vol.",
            f"Starts at observation index {WINDOW_SIZE}; shorter than the",
            "retrospective series by construction (no admissible model exists",
            "for earlier dates, and they are NOT back-filled).",
            "",
            f"Produced by research.analysis.pipeline; window_size={WINDOW_SIZE}, step={STEP}.",
            "Read with: pd.read_csv(path, index_col=0, parse_dates=True, comment='#')",
        ],
    )
    print(f"  Saved {ONESIDED_FORECAST_PROBS_FILE} "
          f"(one-sided, K={FORECAST_K}, {len(onesided_k2)} rows "
          f"from {onesided_k2.index[0].date()} to {onesided_k2.index[-1].date()})")

    # ----- 3. Regime-conditional moments (feeds Table 3) -----
    # NOTE: Tables 3 and 4 deliberately condition on the RETROSPECTIVE labels.
    # They are descriptive statistics of the full-sample regime classification,
    # not an investable signal, and must be reported as such. Using the
    # one-sided labels here would answer a different question and would not be
    # comparable to the regime-characterisation literature.
    print("\nComputing regime-conditional moments (paper_profiles_raw)...")
    print(f"  {RETRO_WARNING} — labels are full-sample smoothed")
    regime_labels = hard_labels_from_daily_probs(daily)

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
    # Stationary block bootstrap (Politis & Romano 1994) on the FULL series with
    # regime labels carried along — NOT an iid resample within each regime. See
    # RegimeDistributionalAnalysis.correlation_difference_tests for why a
    # within-regime block bootstrap is not applicable (R1's median run length is
    # 1 day). Block length defaults to floor(sqrt(T)) = 78 observations.
    print("\nComputing pairwise correlation differences "
          "(stationary block bootstrap, B=10,000)...")
    corr_tests = dist.correlation_difference_tests(
        df_returns=df,
        regime_labels=regime_labels,
        bootstrap_B=10_000,
        ci_level=0.95,
    )
    corr_tests["correlation_pairwise"].to_csv(
        OUTPUT_DIR / "correlation_pairwise_bootstrap.csv", index=False
    )
    print(f"  Saved correlation_pairwise_bootstrap.csv (feeds Table 4)")

    # ----- 5. Dominant-regime forecast eval (feeds Figure 3) -----
    print("\nBuilding dominant_regime_forecast_eval.csv (feeds Figure 3)...")
    _save_dominant_forecast_eval(onesided_k2)

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE — outputs in {OUTPUT_DIR}")
    print("=" * 70)


def _save_dominant_forecast_eval(probs_k2: pd.DataFrame) -> None:
    """One-step-ahead AR(1) forecast vs random-walk baseline (feeds Figure 3).

    Takes the ONE-SIDED K=2 probability series. For each day t:
      - p_ar_t1 : AR(1) forecast of p_1 at t+1, fitted by OLS on the logit of
                  the expanding history y[0..t] and inverted back to
                  probability. Re-fitted every step.
      - p_rw_t1 : random-walk forecast — today's level p_1[t] carried forward.
      - z_t1    : realised hard label at t+1.

    Previously both columns were `p1.shift(1)`, i.e. bit-identical, so Figure 3
    plotted the same line twice under two legend entries. The AR forecast is now
    a genuine model; the random walk keeps the correct RW definition. They
    differ.

    Everything here is causal: the AR model at t is fitted only on y[0..t], and
    the input series is itself one-sided.
    """
    p1 = probs_k2["p_1"].astype(float)
    p = p1.values
    y = logit(p)
    idx = p1.index
    z = (p > 0.5).astype(int)

    rows = []
    n_fallback = 0
    for t in range(AR_MIN_TRAIN, len(p) - 1):
        try:
            fit = AutoReg(y[: t + 1], lags=1, old_names=False).fit()
            p_ar = float(inv_logit(np.asarray(fit.forecast(steps=1))[0]))
        except Exception:
            p_ar = float(p[t])          # degenerate window -> fall back to RW
            n_fallback += 1

        rows.append({
            "date_t":  idx[t],
            "date_t1": idx[t + 1],
            "p_ar_t1": p_ar,
            "p_rw_t1": float(p[t]),
            "z_t1":    int(z[t + 1]),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "dominant_regime_forecast_eval.csv", index=False)

    identical = bool(np.allclose(out["p_ar_t1"], out["p_rw_t1"]))
    hit_ar = float(((out["p_ar_t1"] >= 0.5) == (out["z_t1"] == 1)).mean())
    hit_rw = float(((out["p_rw_t1"] >= 0.5) == (out["z_t1"] == 1)).mean())
    print(f"  Saved dominant_regime_forecast_eval.csv (n={len(out)}, "
          f"AR fallbacks={n_fallback})")
    print(f"    hit rate — AR: {hit_ar:.4f}  RW: {hit_rw:.4f}  "
          f"(series identical: {identical})")
    if identical:
        print("    WARNING: AR and RW forecasts are identical — Figure 3 would "
              "plot one line twice.")


if __name__ == "__main__":
    main()
