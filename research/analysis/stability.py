"""
Table 5 — GMM Regime Stability Under Asset Universe Expansion.

Tests whether regimes remain stable as the asset universe expands tier by tier.
Produces stability_summary.csv + breaking_points.csv + diagnostic plots.

Run:  python -m research.analysis.stability
Runtime: 10-20 minutes.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")

from research.analysis.stability_analyzer import RegimeStabilityAnalyzer
from core.config import (
    K_CANDIDATES, WINDOW_SIZE, STEP, SCALE_METHOD,
    MAX_ITER, TOL, REG_COVAR, EXTENDED_ADD_ORDER, OUTPUT_DIR as _BASE_OUT,
)

OUTPUT_DIR = _BASE_OUT / "stability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("="*70)
    print("REGIME STABILITY ANALYSIS")
    print("Testing RQ4 & RQ5: Asset universe expansion effects")
    print("="*70)

    analyzer = RegimeStabilityAnalyzer(
        K_candidates=tuple(K_CANDIDATES),
        window_size=WINDOW_SIZE,
        step=STEP,
        scale_method=SCALE_METHOD,
        max_iter=MAX_ITER,
        tol=TOL,
        reg_covar=REG_COVAR,
        random_state=42,
        frequency="daily",
        start_date="2000-01-01",
        end_date=None,
        corr_threshold=0.7,
        Neff_threshold=50,
    )
    add_order = list(EXTENDED_ADD_ORDER)

    print(f"\nCore assets: {analyzer.core_assets}")
    print(f"Assets to add: {add_order}")
    print(f"Window size: {analyzer.window_size} days")
    print(f"Step: {analyzer.step} days")
    print(f"K candidates: {analyzer.K_candidates}")
    print()

    # Run analysis
    summary = analyzer.run(add_order=add_order)

    # Save summary
    summary_path = OUTPUT_DIR / "stability_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    # Identify and save breaking points
    breaking = analyzer.identify_breaking_points()
    breaking_path = OUTPUT_DIR / "breaking_points.csv"
    breaking.to_csv(breaking_path, index=False)
    print(f"Saved: {breaking_path}")

    # Print results
    print("\n" + "="*70)
    print("FULL SUMMARY")
    print("="*70)
    print(summary.to_string())

    print("\n" + "="*70)
    print("BREAKING POINTS")
    print("="*70)
    if len(breaking) > 0:
        print(breaking.to_string())
    else:
        print("No breaking points identified - regimes stable across all expansions!")

    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)

    analyzer.plot_stability(save=True)
    analyzer.plot_K_evolution(save=True)
    analyzer.plot_prob_correlation_over_time(save=True)

    # Summary statistics for dissertation
    print("\n" + "="*70)
    print("KEY STATISTICS FOR DISSERTATION")
    print("="*70)

    print(f"\n1. Core universe: {analyzer.core_assets}")
    print(f"   - Number of assets: {len(analyzer.core_assets)}")

    print(f"\n2. Full universe: {len(analyzer.core_assets) + len(add_order)} assets")

    print(f"\n3. Regime count (K) stability:")
    print(f"   - K match rate range: {summary['K_match_rate'].min():.1%} - {summary['K_match_rate'].max():.1%}")

    print(f"\n4. Regime probability concordance:")
    print(f"   - Mean correlation range: {summary['prob_corr_mean'].min():.3f} - {summary['prob_corr_mean'].max():.3f}")
    print(f"   - Min correlation observed: {summary['prob_corr_min'].min():.3f}")

    print(f"\n5. Breaking points identified: {len(breaking)}")
    if len(breaking) > 0:
        print(f"   - Assets: {breaking['added_asset'].tolist()}")

    print(f"\n6. Model fit (BIC):")
    print(f"   - Core: {summary.loc[0, 'bic_mean']:.0f}")
    print(f"   - Full: {summary.iloc[-1]['bic_mean']:.0f}")
    print(f"   - Improvement: {summary.iloc[-1]['bic_mean'] - summary.loc[0, 'bic_mean']:.0f}")

    # Dissertation-formatted TSV (Table 5)
    _save_dissertation_table(summary)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print(f"Output saved to: {OUTPUT_DIR}")
    print("="*70)


def _save_dissertation_table(summary: pd.DataFrame) -> None:
    """Format stability_summary.csv into the dissertation TSV layout."""
    from core.config import ASSET_DISPLAY, TABLES_DIR
    from core.utils import format_pct, format_corr

    out = pd.DataFrame({
        "Tier":             summary["tier"],
        "N Assets":         summary["n_assets"],
        "Added Asset":      summary["added_asset"].map(
            lambda a: "Core" if a == "core" else ASSET_DISPLAY.get(a, a),
        ),
        "Windows":          summary["n_windows"],
        "K (Mode)":         summary["K_mode"],
        "K Range":          summary.apply(lambda r: f"{r['K_min']}-{r['K_max']}", axis=1),
        "K Match Rate":     summary["K_match_rate"].map(format_pct),
        "Prob Corr (Mean)": summary["prob_corr_mean"].map(format_corr),
        "Prob Corr (Min)":  summary["prob_corr_min"].map(format_corr),
        "Unstable Windows": summary["n_unstable_windows"],
        "Neff Min (Mean)":  summary["Neff_min_mean"].map(lambda x: f"{x:.1f}"),
        "BIC (Mean)":       summary["bic_mean"].map(lambda x: f"{x:,.0f}"),
    })
    dst = TABLES_DIR / "table5_universe_stability.tsv"
    out.to_csv(dst, sep="\t", index=False)
    print(f"Saved: {dst}")


if __name__ == "__main__":
    main()