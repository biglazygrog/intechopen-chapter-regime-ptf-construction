"""
Table A.1 — Rolling window size selection: Robustness Summary.

Runs the rolling GMM across the (window_size, step) grid and reports
probability correlation/MAE vs the (1250, 63) baseline plus conditioning
diagnostics. 5 separate fits — runtime 10-20 min.

Run:  python -m research.analysis.robustness
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from data.reader1 import DataReader
from models.gmm import Optimiser
from models.regime_analysis import (
    hard_labels_from_daily_probs,
    regime_turnover_stats,
    cov_stability_diagnostics,
)
from core.utils import filter_synchronous_trading
from core.config import (
    SHRINKAGE_TARGET, KAPPA_0, K_CANDIDATES,
    ROBUSTNESS_GRID as GRID, ROBUSTNESS_BASELINE_SPEC as BASELINE_SPEC,
    OUTPUT_DIR, TABLES_DIR,
)


def _build_optimiser(window_size: int, step: int) -> Optimiser:
    return Optimiser(
        K_candidates=K_CANDIDATES,
        window_size=window_size,
        step=step,
        scale_method="none",
        max_iter=200,
        tol=1e-4,
        reg_covar=1e-6,
        random_state=0,
        regularise=True,
        kappa_0=KAPPA_0,
        nu_0_extra=2,
        lambda_scale=1.0,
        init_decay=None,
        ewma_decay=0.97,
        allow_partial_last_window=False,
        alpha_dirichlet=0.5,
        order_components=True,
        order_mode="trace",
        shrinkage_target=SHRINKAGE_TARGET,
    )


def _run_one(X, dates, window_size, step, baseline_probs):
    opt = _build_optimiser(window_size, step)
    opt.fit(X, index=dates)
    daily = opt.get_daily_probabilities()
    labels = hard_labels_from_daily_probs(daily)

    mins = np.array(opt.Neff_min_, dtype=float)
    neff = {
        "Neff_min_min":   float(np.min(mins)),
        "Neff_min_p05":   float(np.quantile(mins, 0.05)),
        "Neff_min_median": float(np.median(mins)),
    }
    cov_diag = cov_stability_diagnostics(opt)
    turn = regime_turnover_stats(labels)

    if baseline_probs is not None:
        common = daily.index.intersection(baseline_probs.index)
        cols = sorted(set(daily.columns).intersection(baseline_probs.columns))
        if len(common) > 50 and cols:
            a = daily.loc[common, cols].values
            b = baseline_probs.loc[common, cols].values
            corr = float(np.corrcoef(a.flatten(), b.flatten())[0, 1])
            mae = float(np.mean(np.abs(a - b)))
        else:
            corr = float("nan")
            mae = float("nan")
    else:
        corr = 1.0
        mae = 0.0

    return {
        "window_size": int(window_size),
        "step": int(step),
        "n_windows": int(len(opt.fits)),
        "prob_corr_vs_baseline": corr,
        "prob_mae_vs_baseline": mae,
        **neff,
        **cov_diag,
        **turn,
    }, daily


def main():
    print("Reading data...")
    df = filter_synchronous_trading(DataReader().read_retns().dropna())
    X = df.values
    dates = df.index
    print(f"  shape={df.shape}  range={dates[0].date()}..{dates[-1].date()}")

    # 1. Baseline first to capture daily probabilities for comparison.
    print(f"\n--- baseline {BASELINE_SPEC} ---")
    baseline_row, baseline_probs = _run_one(
        X, dates, BASELINE_SPEC[0], BASELINE_SPEC[1], baseline_probs=None
    )

    rows: List[Dict] = []
    for ws, st in GRID:
        if (ws, st) == BASELINE_SPEC:
            rows.append(baseline_row)
            continue
        print(f"\n--- spec ({ws}, {st}) ---")
        row, _ = _run_one(X, dates, ws, st, baseline_probs=baseline_probs)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    # Reorder columns to match legacy file
    legacy_cols = [
        "window_size", "step", "n_windows",
        "prob_corr_vs_baseline", "prob_mae_vs_baseline",
        "Neff_min_min", "Neff_min_p05", "Neff_min_median",
        "min_eigenvalue", "max_condition_number", "near_singular_count",
    ]
    keep = [c for c in legacy_cols if c in df_out.columns]
    extra = [c for c in df_out.columns if c not in keep]
    df_out = df_out[keep + extra]

    csv_path  = OUTPUT_DIR / "robustness_summary.csv"
    xlsx_path = OUTPUT_DIR / "robustness_summary.xlsx"

    df_out.to_csv(csv_path, index=False)
    df_out.to_excel(xlsx_path, index=False)
    print(f"\nWritten: {csv_path}")
    print(f"Written: {xlsx_path}")

    # Dissertation-friendly TSV (matches the paper column labels).
    fmt = df_out.copy()
    fmt = fmt[[
        "window_size", "step", "n_windows",
        "prob_corr_vs_baseline", "prob_mae_vs_baseline",
        "Neff_min_min", "Neff_min_p05", "Neff_min_median",
        "min_eigenvalue", "max_condition_number", "near_singular_count",
    ]]
    fmt = fmt.rename(columns={
        "window_size":                "Window size",
        "step":                       "Step size",
        "n_windows":                  "No. of windows",
        "prob_corr_vs_baseline":      "Probability correlation vs baseline",
        "prob_mae_vs_baseline":       "Probability mean absolute error vs baseline",
        "Neff_min_min":               "Minimum effective sample size (min)",
        "Neff_min_p05":               "Minimum effective sample size (5th pct.)",
        "Neff_min_median":            "Minimum effective sample size (median)",
        "min_eigenvalue":             "Minimum eigenvalue",
        "max_condition_number":       "Maximum condition number",
        "near_singular_count":        "Near-singular matrices",
    })
    formatters = {
        "Probability correlation vs baseline": lambda x: f"{x:.3f}",
        "Probability mean absolute error vs baseline": lambda x: f"{x:.3f}",
        "Minimum effective sample size (min)": lambda x: f"{x:.2f}",
        "Minimum effective sample size (5th pct.)": lambda x: f"{x:.2f}",
        "Minimum effective sample size (median)": lambda x: f"{x:.2f}",
        "Minimum eigenvalue": lambda x: f"{x:.7f}",
        "Maximum condition number": lambda x: f"{x:.2f}",
    }
    for col, fn in formatters.items():
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(fn)

    tsv_path = TABLES_DIR / "tableA1_window_robustness.tsv"
    fmt.to_csv(tsv_path, sep="\t", index=False)
    print(f"Written: {tsv_path}")
    print()
    print(fmt.to_string(index=False))


if __name__ == "__main__":
    main()
