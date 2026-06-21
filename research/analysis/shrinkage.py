"""
Appendix table — Sensitivity to mu_0 prior weight (kappa_0).

Sweeps three shrinkage modes x six kappa_0 values = 18 rolling GMM fits.
Produces shrinkage_comparison_summary.csv and the dissertation appendix TSV.

Run:  python -m research.analysis.shrinkage
Runtime: 15-25 minutes.
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from models.gmm import Optimiser
from core.config import OUTPUT_DIR as _BASE_OUT, TABLES_DIR

OUTPUT_DIR = _BASE_OUT / "shrinkage_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# diagnostics
# -------------------------------------------------------------------

def _aligned_mean_drift(opt: Optimiser) -> float:
    """Average ||mu_k^(t) - mu_k^(t-1)|| across components and windows,
    after Hungarian alignment between consecutive windows."""
    fits = opt.fits
    if len(fits) < 2:
        return float("nan")

    drifts: List[float] = []
    for t in range(1, len(fits)):
        prev = fits[t - 1]["means"]
        curr = fits[t]["means"]
        Kp, Kc = prev.shape[0], curr.shape[0]
        cost = np.linalg.norm(curr[:, None, :] - prev[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            drifts.append(float(np.linalg.norm(curr[r] - prev[c])))
    return float(np.mean(drifts)) if drifts else float("nan")


def _probability_autocorr(opt: Optimiser) -> float:
    """Mean lag-1 autocorrelation of the dominant regime's daily probability.

    A higher value indicates a smoother / more persistent regime-probability
    series — i.e. less label noise across the rolling fit boundary.
    """
    probs = opt.get_daily_probabilities()
    if probs.empty:
        return float("nan")
    dominant = probs.max(axis=1)
    s = dominant.values
    if len(s) < 3:
        return float("nan")
    return float(np.corrcoef(s[:-1], s[1:])[0, 1])


def _k_stability(opt: Optimiser) -> float:
    ks = np.asarray(opt.selected_Ks_, dtype=int)
    if len(ks) < 2:
        return float("nan")
    return float(np.mean(ks[1:] == ks[:-1]))


def _max_regime_age(opt: Optimiser) -> int:
    if not opt.window_regime_ages_:
        return 0
    return int(max(int(a.max()) for a in opt.window_regime_ages_ if len(a)))


def _before_after_diff_stats(
    probs_before: pd.DataFrame, probs_after: pd.DataFrame
) -> Dict[str, object]:
    """Per-regime numerical diff between two regime-probability tables."""
    common = sorted(set(probs_before.columns).intersection(probs_after.columns))
    b = probs_before[common]
    a = probs_after[common]

    diff = (a - b).abs()
    per_regime_mean_abs = diff.mean().to_dict()
    per_regime_max_abs = diff.max().to_dict()

    correlations: Dict[str, float] = {}
    for c in common:
        bb = b[c].dropna()
        aa = a[c].reindex(bb.index).dropna()
        idx = bb.index.intersection(aa.index)
        if len(idx) > 5:
            correlations[c] = float(np.corrcoef(bb.loc[idx], aa.loc[idx])[0, 1])

    # Dominant-regime label flips.
    dom_b = b.idxmax(axis=1)
    dom_a = a.idxmax(axis=1)
    valid = dom_b.notna() & dom_a.notna()
    flips = ((dom_b != dom_a) & valid).sum()
    n_total = int(valid.sum())
    flip_rate = float(flips / n_total) if n_total else float("nan")
    flip_dates = b.index[((dom_b != dom_a) & valid).values][:10].tolist()

    return {
        "regimes_compared": common,
        "per_regime_mean_abs_diff": per_regime_mean_abs,
        "per_regime_max_abs_diff": per_regime_max_abs,
        "per_regime_corr": correlations,
        "n_days_compared": n_total,
        "dominant_flip_count": int(flips),
        "dominant_flip_rate": flip_rate,
        "dominant_flip_first_dates": [str(d.date()) for d in flip_dates],
    }


def _print_diff_stats(stats: Dict[str, object]) -> None:
    print(f"\n  days compared:                 {stats['n_days_compared']}")
    print(f"  regimes compared:              {stats['regimes_compared']}")
    print(f"  dominant flip count:           {stats['dominant_flip_count']} "
          f"({100 * stats['dominant_flip_rate']:.2f}%)")
    print(f"  first flip dates:              {stats['dominant_flip_first_dates'][:5]}")
    print("  per-regime mean |before-after|:")
    for k, v in stats["per_regime_mean_abs_diff"].items():
        print(f"     {k}:  {v:.6f}")
    print("  per-regime max  |before-after|:")
    for k, v in stats["per_regime_max_abs_diff"].items():
        print(f"     {k}:  {v:.6f}")
    print("  per-regime corr(before, after):")
    for k, v in stats["per_regime_corr"].items():
        print(f"     {k}:  rho = {v:.6f}")


def _summarise(opt: Optimiser, mode: str, kappa_0: float) -> Dict[str, float]:
    bic = np.array(opt.window_bic_, dtype=float)
    neff_min = np.array(opt.Neff_min_, dtype=float)
    coverage = np.array(opt.window_prior_coverage_, dtype=float)
    median_Nk = float(np.median(neff_min)) if neff_min.size else float("nan")
    prior_weight = (
        kappa_0 / (kappa_0 + median_Nk) if not np.isnan(median_Nk) else float("nan")
    )
    return {
        "mode": mode,
        "kappa_0": kappa_0,
        "n_windows": int(len(opt.fits)),
        "mean_K": float(np.mean(opt.selected_Ks_)) if opt.selected_Ks_ else float("nan"),
        "K_stability": _k_stability(opt),
        "mean_drift": _aligned_mean_drift(opt),
        "dominant_prob_autocorr": _probability_autocorr(opt),
        "mean_BIC": float(np.mean(bic)) if bic.size else float("nan"),
        "median_Neff_min": median_Nk,
        "implied_prior_weight": prior_weight,
        "mean_prior_coverage": float(np.mean(coverage)) if coverage.size else 0.0,
        "max_regime_age": _max_regime_age(opt),
    }


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------

def _build_optimiser(shrinkage_target: str, kappa_age_scale: bool, kappa_0: float) -> Optimiser:
    return Optimiser(
        K_candidates=[2, 3, 4, 5, 6, 7],
        window_size=1250,
        step=63,
        scale_method="none",
        max_iter=200,
        tol=1e-4,
        reg_covar=1e-6,
        random_state=0,
        regularise=True,
        kappa_0=kappa_0,
        nu_0_extra=2,
        lambda_scale=1.0,
        init_decay=None,
        ewma_decay=0.97,
        allow_partial_last_window=False,
        alpha_dirichlet=0.5,
        order_components=True,
        order_mode="trace",
        shrinkage_target=shrinkage_target,
        kappa_age_scale=kappa_age_scale,
    )


# (mode_label, shrinkage_target, kappa_age_scale)
MODES = [
    ("grand_mean",            "grand_mean",     False),
    ("previous_epoch",        "previous_epoch", False),
    ("previous_epoch_aged",   "previous_epoch", True),
]

# --- choice of kappa_0 grid -----------------------------------------------
#
# kappa_0 is the *effective sample size of the prior* in the Normal-Wishart
# MAP update: mu_k = (N_k * y_bar + kappa_0 * mu_prior) / (N_k + kappa_0).
# So the prior's posterior weight at a regime with N_k samples is
#   w_prior(N_k, kappa_0) = kappa_0 / (kappa_0 + N_k).
# Inverting for kappa_0:
#   kappa_0(N_k, p) = N_k * p / (1 - p).
#
# We therefore parametrise the sweep in *target prior weight* terms — the
# semantically meaningful quantity — and back out kappa_0 at the typical
# effective sample size per regime. With window_size=1250 and mean_K~2.2,
# N_k_typical ~ window_size / mean_K ~ 568.
#
# Primary (a priori) choice for the headline result: 10% prior weight.
# This is fixed BEFORE looking at any in-sample metric, so it is not subject
# to look-ahead bias. The 0% row (kappa_0=0.01) is the legacy/no-prior
# baseline; the 5%-25% rows are the meaningful sensitivity range; the 50%
# row is included to document over-shrinkage / degradation at the high end.
N_K_TYPICAL = 568.0  # window_size=1250, mean_K~2.2 → N_k ~ 568 per regime

def _kappa_from_prior_weight(p: float, n_k: float = N_K_TYPICAL) -> float:
    """kappa_0 such that the prior carries weight p at sample size n_k."""
    return float(n_k * p / (1.0 - p))

PRIOR_WEIGHT_GRID = [0.00, 0.05, 0.10, 0.15, 0.25, 0.50]
KAPPA_GRID = [
    0.01 if p == 0.0 else round(_kappa_from_prior_weight(p), 1)
    for p in PRIOR_WEIGHT_GRID
]
KAPPA_PRIMARY = round(_kappa_from_prior_weight(0.10), 1)  # the 10% row


def main():
    try:
        from data.reader1 import DataReader
    except ImportError as e:
        print(f"Could not import DataReader: {e}")
        return

    print("Reading data...")
    df = DataReader().read_retns().dropna()
    df = df[(df != 0).all(axis=1)]
    df = df[np.isfinite(df).all(axis=1)]
    X_raw = df.values
    dates = df.index
    print(f"  shape={df.shape}  range={dates[0].date()} to {dates[-1].date()}")

    rows = []
    daily_probs_by_cell: Dict[Tuple[str, float], pd.DataFrame] = {}

    for mode_label, shrinkage_target, kappa_age_scale in MODES:
        # The kappa sweep is meaningless for the grand_mean baseline (kappa_0
        # only changes how strongly each regime is dragged toward the SAME
        # grand mean; it doesn't introduce per-regime structure). We still run
        # one baseline per kappa to confirm the data-dominated behaviour, but
        # the regime structure won't move much.
        for kappa_0 in KAPPA_GRID:
            print(
                f"\n========== {mode_label} | kappa_0={kappa_0} "
                f"(shrinkage_target={shrinkage_target}, kappa_age_scale={kappa_age_scale}) =========="
            )
            opt = _build_optimiser(shrinkage_target, kappa_age_scale, kappa_0)
            opt.fit(X_raw, index=dates)
            rows.append(_summarise(opt, mode_label, kappa_0))
            daily_probs_by_cell[(mode_label, kappa_0)] = opt.get_daily_probabilities()

    summary = pd.DataFrame(rows)
    summary["primary"] = summary["kappa_0"] == KAPPA_PRIMARY
    summary_path = OUTPUT_DIR / "shrinkage_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\n========== SIDE-BY-SIDE SUMMARY ==========")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(summary.to_string(index=False))
    print(f"\nWritten: {summary_path}")

    print(
        f"\n========== PRIMARY (a priori) CHOICE: kappa_0 = {KAPPA_PRIMARY} =========="
    )
    print("Chosen from Bayesian prior-weight semantics, not from any in-sample metric.")
    print("All other kappa_0 rows are robustness analysis.\n")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(summary[summary["primary"]].to_string(index=False))

    # Compact diagnostic table: pivot mean_drift and dominant_prob_autocorr
    # across mode x kappa_0 so the effect of kappa is obvious at a glance.
    for metric in ("mean_drift", "dominant_prob_autocorr", "K_stability", "mean_BIC"):
        pv = summary.pivot(index="mode", columns="kappa_0", values=metric)
        pv_path = OUTPUT_DIR / f"pivot_{metric}.csv"
        pv.to_csv(pv_path)
        print(f"\n{metric} (mode x kappa_0):")
        print(pv.to_string())
        print(f"  -> {pv_path}")

    # ------------------------------------------------------------------
    # BEFORE / AFTER — the only chart we keep.
    # ------------------------------------------------------------------
    # Before = legacy default (grand_mean shrinkage, near-zero kappa_0).
    # After  = a priori recommendation (previous_epoch shrinkage,
    #          kappa_0 = N_k_typical * 0.10 / 0.90 ≈ KAPPA_PRIMARY).
    before_key = ("grand_mean", KAPPA_GRID[0])           # p = 0%, kappa_0 = 0.01
    after_key  = ("previous_epoch", KAPPA_PRIMARY)        # p = 10%

    probs_before = daily_probs_by_cell[before_key]
    probs_after  = daily_probs_by_cell[after_key]

    before_csv = OUTPUT_DIR / "daily_probs_before_legacy.csv"
    after_csv  = OUTPUT_DIR / "daily_probs_after_primary.csv"
    probs_before.to_csv(before_csv)
    probs_after.to_csv(after_csv)

    print("\n========== BEFORE / AFTER ==========")
    print(f"BEFORE: mode={before_key[0]}, kappa_0={before_key[1]} (legacy, no prior)")
    print(f"        -> {before_csv}")
    print(f"AFTER : mode={after_key[0]}, kappa_0={after_key[1]} (a priori, p=10%)")
    print(f"        -> {after_csv}")

    diff_stats = _before_after_diff_stats(probs_before, probs_after)
    _print_diff_stats(diff_stats)
    diff_path = OUTPUT_DIR / "before_after_diff_stats.csv"
    diff_rows = []
    for c in diff_stats["regimes_compared"]:
        diff_rows.append({
            "regime": c,
            "mean_abs_diff": diff_stats["per_regime_mean_abs_diff"].get(c, np.nan),
            "max_abs_diff":  diff_stats["per_regime_max_abs_diff"].get(c, np.nan),
            "corr":          diff_stats["per_regime_corr"].get(c, np.nan),
        })
    diff_rows.append({
        "regime": "_summary_",
        "mean_abs_diff": np.nan,
        "max_abs_diff": np.nan,
        "corr": np.nan,
        "n_days_compared": diff_stats["n_days_compared"],
        "dominant_flip_count": diff_stats["dominant_flip_count"],
        "dominant_flip_rate": diff_stats["dominant_flip_rate"],
        "first_flip_dates": ",".join(diff_stats["dominant_flip_first_dates"][:10]),
    })
    pd.DataFrame(diff_rows).to_csv(diff_path, index=False)
    print(f"\n        diff stats: {diff_path}")

    # Stacked-area two-panel comparison PNG. Top = before, bottom = after.
    # Stackplot fills the regime probabilities as solid bands that sum to 1,
    # giving a solid-coloured view with no white gaps.
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
    for ax, (key, title) in zip(
        axes,
        [
            (before_key, f"BEFORE  —  grand_mean,  kappa_0 = {before_key[1]}  (legacy)"),
            (after_key,  f"AFTER  —  previous_epoch,  kappa_0 = {after_key[1]}  (a priori, p = 10%)"),
        ],
    ):
        probs = daily_probs_by_cell[key]
        cols = list(probs.columns)
        series = [probs[c].values for c in cols]
        ax.stackplot(
            probs.index,
            *series,
            labels=[f"P(regime {c})" for c in cols],
            edgecolor="none",
        )
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("P(regime)")
        ax.set_ylim(0.0, 1.0)
        ax.margins(x=0)
        ax.legend(loc="upper right", ncol=min(len(cols), 4), fontsize=8, framealpha=0.85)
    axes[-1].set_xlabel("date")
    fig.suptitle("Regime probabilities — before vs after temporal-prior shrinkage", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    before_after_path = OUTPUT_DIR / "regime_probs_before_after.png"
    fig.savefig(before_after_path, dpi=140)
    plt.close(fig)
    print(f"        plot: {before_after_path}")

    # Dissertation appendix table
    _save_appendix_table()


def _save_appendix_table() -> None:
    """Format shrinkage_comparison_summary.csv into the dissertation appendix TSV."""
    from core.config import KAPPA_GRID, KAPPA_PRIMARY, KAPPA_P_LABEL

    df = pd.read_csv(OUTPUT_DIR / "shrinkage_comparison_summary.csv")
    rows = []
    base = df[(df["mode"] == "grand_mean") & (np.isclose(df["kappa_0"], 0.01))].iloc[0]
    rows.append({
        "Configuration": "Baseline (grand mean)",
        "Target prior weight p": "0%",
        "kappa_0": "0.01",
        "K stability": f"{base['K_stability']:.3f}",
        "Mean drift (x10^-3)": f"{base['mean_drift'] * 1000:.3f}",
        "Probability autocorrelation": f"{base['dominant_prob_autocorr']:.3f}",
        "Mean BIC": f"{base['mean_BIC']:,.1f}",
        "Mean prior coverage": "n/a",
    })
    pe = df[df["mode"] == "previous_epoch"].copy()
    for k in KAPPA_GRID:
        r = pe[np.isclose(pe["kappa_0"], k)].iloc[0]
        label = "Previous-epoch prior (PRIMARY)" if np.isclose(k, KAPPA_PRIMARY) else "Previous-epoch prior"
        kappa_disp = "0.0" if k == 0.01 else f"{k:.1f}"
        rows.append({
            "Configuration": label,
            "Target prior weight p": KAPPA_P_LABEL[k],
            "kappa_0": kappa_disp,
            "K stability": f"{r['K_stability']:.3f}",
            "Mean drift (x10^-3)": f"{r['mean_drift'] * 1000:.3f}",
            "Probability autocorrelation": f"{r['dominant_prob_autocorr']:.3f}",
            "Mean BIC": f"{r['mean_BIC']:,.1f}",
            "Mean prior coverage": f"{r['mean_prior_coverage'] * 100:.1f}%",
        })
    out = pd.DataFrame(rows)
    dst = TABLES_DIR / "tableA2_kappa_sensitivity.tsv"
    out.to_csv(dst, sep="\t", index=False)
    print(f"        appendix TSV: {dst}")


if __name__ == "__main__":
    main()
