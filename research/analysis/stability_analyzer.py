"""
Iterative Regime Stability Analysis (RQ4 & RQ5)

Integrates with Optimiser to test regime stability:
- Across asset universes (expanding from core to full)
- Over time (using rolling/expanding windows)

For each asset tier, runs full Optimiser analysis, then compares
regime probabilities across tiers at each time window.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Sequence
from dataclasses import dataclass, field
from scipy.stats import spearmanr
import warnings

from data.reader1 import DataReader
from models.gmm import Optimiser
from core.utils import filter_synchronous_trading
from core.config import OUTPUT_DIR as _BASE_OUT

OUTPUT_DIR = _BASE_OUT / "stability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TierResult:
    """Results from Optimiser run on one asset tier."""
    tier_id: int
    n_assets: int
    asset_names: List[str]
    optimiser: Optimiser
    daily_probs: pd.DataFrame      # (T, max_K) regime probabilities
    selected_K_series: pd.Series   # K selected at each window
    window_bics: List[float]
    window_Neff_min: List[float]


@dataclass
class CrossTierComparison:
    """Comparison metrics between two consecutive tiers."""
    tier_from: int
    tier_to: int
    added_assets: List[str]

    # Per-window metrics
    K_match_rate: float            # Fraction of windows with same K
    prob_corr_by_window: pd.Series # Spearman correlation per window
    prob_corr_mean: float
    prob_corr_min: float

    # Aggregate metrics
    K_changes: int                 # Number of windows where K differs
    regime_flip_windows: List      # Windows where correlation < threshold


class RegimeStabilityAnalyzer:
    """
    Analyzes regime stability across expanding asset universes using
    the Optimiser class for rolling/expanding window GMM fits.

    Workflow:
    1. Define asset tiers (core -> extended)
    2. Run Optimiser on each tier
    3. Compare daily regime probabilities across tiers
    4. Identify breaking points where regimes destabilize
    """

    def __init__(
        self,
        # Optimiser parameters
        K_candidates: Sequence[int] = (2, 3, 4),
        window_size: int = 252 * 3,   # 3 years
        step: int = 252,               # 1 year
        scale_method: str = "rolling",
        max_iter: int = 200,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: int = 42,

        # Data parameters
        frequency: str = "daily",
        start_date: str = None,
        end_date: str = None,

        # Analysis parameters
        corr_threshold: float = 0.7,   # Below this = unstable
        Neff_threshold: float = 50,    # Below this = poor identification
    ):
        self.K_candidates = tuple(K_candidates)
        self.window_size = window_size
        self.step = step
        self.scale_method = scale_method
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        self.frequency = frequency
        self.start_date = start_date
        self.end_date = end_date

        self.corr_threshold = corr_threshold
        self.Neff_threshold = Neff_threshold

        # Load data
        self.dr_core = DataReader(version=1)
        self.dr_extended = DataReader(version=2)

        self.r_core = self.dr_core.read_retns(frequency, start_date, end_date)
        self.r_extended = self.dr_extended.read_retns(frequency, start_date, end_date)

        # Align dates (keep all data - cleaning done per-tier in _run_tier)
        common_idx = self.r_core.index.intersection(self.r_extended.index)
        self.r_core = self.r_core.loc[common_idx]
        self.r_extended = self.r_extended.loc[common_idx]

        # Define assets
        self.core_assets = list(self.r_core.columns)
        self.extended_assets = [c for c in self.r_extended.columns
                                if c not in self.core_assets]

        # Results storage
        self.tier_results: List[TierResult] = []
        self.comparisons: List[CrossTierComparison] = []

    def _create_optimiser(self) -> Optimiser:
        """Create a fresh Optimiser instance with configured parameters.

        Uses the per-regime temporal prior (shrinkage_target="previous_epoch")
        at kappa_0=63.1 (~10% target prior weight) consistent with the
        dissertation's primary methodology.
        """
        return Optimiser(
            K_candidates=self.K_candidates,
            window_size=self.window_size,
            step=self.step,
            scale_method=self.scale_method,
            max_iter=self.max_iter,
            tol=self.tol,
            reg_covar=self.reg_covar,
            random_state=self.random_state,
            regularise=True,
            kappa_0=63.1,
            alpha_dirichlet=0.5,
            order_components=True,
            order_mode="trace",
            shrinkage_target="previous_epoch",
        )

    def _run_tier(self, tier_id: int, assets: List[str], df: pd.DataFrame) -> TierResult:
        """Run Optimiser on a single asset tier."""
        print(f"\n{'='*70}")
        print(f"TIER {tier_id}: {len(assets)} assets")
        print(f"Assets: {assets}")
        print(f"{'='*70}")

        # Synchronous-trading filter, per tier. Same single definition as the
        # rest of the package: market indices must have traded; cash-like series
        # are exempt (a zero there is a genuine near-zero return, not a stale
        # quote). Exempt names absent from a given tier are ignored.
        df_tier = filter_synchronous_trading(df[assets].dropna())
        X = df_tier.values

        # Run Optimiser
        opt = self._create_optimiser()

        # Suppress verbose output from MyGMM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt.fit(X, index=df_tier.index)

        # Extract results
        daily_probs = opt.get_daily_probabilities()
        K_series = opt.get_selected_K_series()

        result = TierResult(
            tier_id=tier_id,
            n_assets=len(assets),
            asset_names=assets.copy(),
            optimiser=opt,
            daily_probs=daily_probs,
            selected_K_series=K_series,
            window_bics=opt.window_bic_.copy(),
            window_Neff_min=opt.Neff_min_.copy(),
        )

        print(f"Windows fitted: {len(opt.fits)}")
        print(f"K range: {K_series.min()} - {K_series.max()}")
        print(f"BIC range: {min(opt.window_bic_):.0f} - {max(opt.window_bic_):.0f}")

        return result

    def _compare_tiers(self, prev: TierResult, curr: TierResult) -> CrossTierComparison:
        """Compare regime probabilities between two consecutive tiers."""

        added = [a for a in curr.asset_names if a not in prev.asset_names]

        # Align indices
        common_dates = prev.daily_probs.index.intersection(curr.daily_probs.index)
        prev_probs = prev.daily_probs.loc[common_dates]
        curr_probs = curr.daily_probs.loc[common_dates]

        # Align K series by window end dates
        common_windows = prev.selected_K_series.index.intersection(
            curr.selected_K_series.index
        )
        K_prev = prev.selected_K_series.loc[common_windows]
        K_curr = curr.selected_K_series.loc[common_windows]

        # K match rate
        K_matches = (K_prev == K_curr).sum()
        K_match_rate = K_matches / len(common_windows) if len(common_windows) > 0 else 0
        K_changes = len(common_windows) - K_matches

        # Per-window probability correlation
        # Group daily probs by window and compute correlation
        corrs = []
        corr_dates = []
        flip_windows = []

        for win_date in common_windows:
            K_p = K_prev.loc[win_date]
            K_c = K_curr.loc[win_date]

            # Find dates in this window (approximate: use window_size before win_date)
            win_start = win_date - pd.Timedelta(days=self.window_size * 1.5)
            mask = (common_dates >= win_start) & (common_dates <= win_date)

            if mask.sum() < 10:
                continue

            if K_p == K_c:
                # Same K: compute correlation for each regime
                K = K_p
                regime_corrs = []
                for k in range(K):
                    col = f"p_{k}"
                    if col in prev_probs.columns and col in curr_probs.columns:
                        p1 = prev_probs.loc[mask, col].values
                        p2 = curr_probs.loc[mask, col].values
                        if len(p1) > 5 and np.std(p1) > 0 and np.std(p2) > 0:
                            rho, _ = spearmanr(p1, p2)
                            regime_corrs.append(rho)

                if regime_corrs:
                    avg_corr = np.mean(regime_corrs)
                    corrs.append(avg_corr)
                    corr_dates.append(win_date)

                    if avg_corr < self.corr_threshold:
                        flip_windows.append(win_date)
            else:
                # K differs: mark as low correlation
                corrs.append(np.nan)
                corr_dates.append(win_date)
                flip_windows.append(win_date)

        corr_series = pd.Series(corrs, index=corr_dates, name="prob_correlation")

        return CrossTierComparison(
            tier_from=prev.tier_id,
            tier_to=curr.tier_id,
            added_assets=added,
            K_match_rate=K_match_rate,
            prob_corr_by_window=corr_series,
            prob_corr_mean=np.nanmean(corrs) if corrs else np.nan,
            prob_corr_min=np.nanmin(corrs) if corrs else np.nan,
            K_changes=K_changes,
            regime_flip_windows=flip_windows,
        )

    def run(self, add_order: List[str] = None) -> pd.DataFrame:
        """
        Run full stability analysis.

        Parameters
        ----------
        add_order : list of str, optional
            Order in which to add extended assets. If None, uses default order.

        Returns
        -------
        summary : DataFrame with metrics per tier
        """
        if add_order is None:
            add_order = self.extended_assets

        self.tier_results.clear()
        self.comparisons.clear()

        # Build combined data source
        df_all = pd.concat([self.r_core, self.r_extended], axis=1)
        df_all = df_all.loc[:, ~df_all.columns.duplicated()]

        # Tier 0: Core assets
        result0 = self._run_tier(0, self.core_assets, df_all)
        self.tier_results.append(result0)

        # Iteratively add assets
        current_assets = self.core_assets.copy()

        for i, asset in enumerate(add_order):
            if asset not in df_all.columns:
                print(f"Warning: {asset} not found, skipping")
                continue
            if asset in current_assets:
                print(f"Warning: {asset} already in universe, skipping")
                continue

            current_assets.append(asset)

            result = self._run_tier(i + 1, current_assets, df_all)
            self.tier_results.append(result)

            # Compare with previous tier
            comparison = self._compare_tiers(self.tier_results[-2], result)
            self.comparisons.append(comparison)

            print(f"\n--- Comparison with Tier {i} ---")
            print(f"Added: {comparison.added_assets}")
            print(f"K match rate: {comparison.K_match_rate:.1%}")
            print(f"Prob correlation (mean): {comparison.prob_corr_mean:.3f}")
            if comparison.regime_flip_windows:
                print(f"*** UNSTABLE WINDOWS: {len(comparison.regime_flip_windows)} ***")

        return self.summary()

    def summary(self) -> pd.DataFrame:
        """Return summary DataFrame of all tiers."""
        rows = []

        for i, tier in enumerate(self.tier_results):
            row = {
                "tier": tier.tier_id,
                "n_assets": tier.n_assets,
                "added_asset": tier.asset_names[-1] if i > 0 else "core",
                "n_windows": len(tier.window_bics),
                "K_mode": tier.selected_K_series.mode().iloc[0] if len(tier.selected_K_series) > 0 else np.nan,
                "K_min": tier.selected_K_series.min(),
                "K_max": tier.selected_K_series.max(),
                "bic_mean": np.mean(tier.window_bics),
                "Neff_min_mean": np.mean(tier.window_Neff_min),
            }

            if i > 0 and i - 1 < len(self.comparisons):
                comp = self.comparisons[i - 1]
                row["K_match_rate"] = comp.K_match_rate
                row["prob_corr_mean"] = comp.prob_corr_mean
                row["prob_corr_min"] = comp.prob_corr_min
                row["n_unstable_windows"] = len(comp.regime_flip_windows)
            else:
                row["K_match_rate"] = np.nan
                row["prob_corr_mean"] = np.nan
                row["prob_corr_min"] = np.nan
                row["n_unstable_windows"] = 0

            rows.append(row)

        return pd.DataFrame(rows)

    def identify_breaking_points(self) -> pd.DataFrame:
        """
        Identify assets that cause regime instability.

        Breaking criteria:
        - K match rate < 80%
        - Prob correlation drops below threshold
        - Multiple unstable windows
        """
        summary = self.summary()

        breaking = []
        for i, row in summary.iterrows():
            if i == 0:
                continue

            reasons = []

            if row["K_match_rate"] < 0.8:
                reasons.append(f"K_match={row['K_match_rate']:.0%}")

            if row["prob_corr_min"] < self.corr_threshold:
                reasons.append(f"low_corr={row['prob_corr_min']:.2f}")

            if row["n_unstable_windows"] > 2:
                reasons.append(f"unstable_wins={row['n_unstable_windows']}")

            if row["Neff_min_mean"] < self.Neff_threshold:
                reasons.append(f"low_Neff={row['Neff_min_mean']:.0f}")

            if reasons:
                breaking.append({
                    "tier": row["tier"],
                    "added_asset": row["added_asset"],
                    "n_assets": row["n_assets"],
                    "reasons": ", ".join(reasons),
                    "K_match_rate": row["K_match_rate"],
                    "prob_corr_mean": row["prob_corr_mean"],
                    "n_unstable_windows": row["n_unstable_windows"],
                })

        return pd.DataFrame(breaking)

    def plot_stability(self, save: bool = True):
        """Plot regime stability metrics across tiers."""
        summary = self.summary()
        n_tiers = len(summary)

        fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)

        x = summary["n_assets"].values
        labels = [f"{row['added_asset']}\n({row['n_assets']})"
                  for _, row in summary.iterrows()]

        # 1. K range per tier
        ax = axes[0]
        ax.fill_between(x, summary["K_min"], summary["K_max"], alpha=0.3, label="K range")
        ax.plot(x, summary["K_mode"], marker="o", linewidth=2, label="K mode")
        ax.set_ylabel("Number of Regimes (K)")
        ax.set_title("Regime Count Stability Across Asset Universes")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)

        # 2. K match rate
        ax = axes[1]
        match_rates = summary["K_match_rate"].values[1:]  # Skip first (baseline)
        ax.bar(x[1:], match_rates, width=0.6, alpha=0.7)
        ax.axhline(0.8, color="orange", linestyle="--", label="Threshold (80%)")
        ax.set_ylabel("K Match Rate")
        ax.set_title("Fraction of Windows with Same K as Previous Tier")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. Probability correlation
        ax = axes[2]
        ax.plot(x[1:], summary["prob_corr_mean"].values[1:], marker="o",
                linewidth=2, label="Mean")
        ax.plot(x[1:], summary["prob_corr_min"].values[1:], marker="s",
                linewidth=2, label="Min", alpha=0.7)
        ax.axhline(self.corr_threshold, color="red", linestyle="--",
                   alpha=0.5, label=f"Threshold ({self.corr_threshold})")
        ax.set_ylabel("Spearman ρ")
        ax.set_title("Regime Probability Concordance (vs. Previous Tier)")
        ax.set_ylim(-0.5, 1.05)
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)

        # 4. Unstable windows
        ax = axes[3]
        ax.bar(x[1:], summary["n_unstable_windows"].values[1:], width=0.6,
               alpha=0.7, color="coral")
        ax.set_ylabel("# Unstable Windows")
        ax.set_title("Number of Windows with Low Correlation or K Mismatch")
        ax.grid(alpha=0.3)

        # 5. BIC and Neff
        ax = axes[4]
        ax2 = ax.twinx()

        l1, = ax.plot(x, summary["bic_mean"], marker="o", linewidth=2,
                      color="blue", label="BIC (mean)")
        l2, = ax2.plot(x, summary["Neff_min_mean"], marker="s", linewidth=2,
                       color="green", label="Neff min (mean)")
        ax2.axhline(self.Neff_threshold, color="green", linestyle="--", alpha=0.5)

        ax.set_ylabel("BIC (higher = better)", color="blue")
        ax2.set_ylabel("Min Effective Sample Size", color="green")
        ax.set_xlabel("Number of Assets")
        ax.set_title("Model Fit Quality")
        ax.legend(handles=[l1, l2], loc="upper left")
        ax.grid(alpha=0.3)

        # X-axis labels
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

        plt.tight_layout()

        if save:
            path = OUTPUT_DIR / "regime_stability_optimiser.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
        plt.show()

    def plot_K_evolution(self, save: bool = True):
        """Plot K selection over time for each tier."""
        n_tiers = len(self.tier_results)

        fig, axes = plt.subplots(n_tiers, 1, figsize=(14, 3 * n_tiers), sharex=True)
        if n_tiers == 1:
            axes = [axes]

        for i, tier in enumerate(self.tier_results):
            ax = axes[i]
            K_series = tier.selected_K_series

            ax.step(K_series.index, K_series.values, where="mid", linewidth=1.5)
            ax.set_ylabel("K")
            ax.set_title(f"Tier {tier.tier_id}: {tier.n_assets} assets ({tier.asset_names[-1]})")
            ax.set_ylim(0, max(self.K_candidates) + 1)
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel("Date")
        plt.tight_layout()

        if save:
            path = OUTPUT_DIR / "K_evolution_by_tier.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
        plt.show()

    def plot_prob_correlation_over_time(self, save: bool = True):
        """Plot probability correlation with previous tier over time."""
        if len(self.comparisons) == 0:
            print("No comparisons available. Run analysis first.")
            return

        n_comps = len(self.comparisons)
        fig, axes = plt.subplots(n_comps, 1, figsize=(14, 3 * n_comps), sharex=True)
        if n_comps == 1:
            axes = [axes]

        for i, comp in enumerate(self.comparisons):
            ax = axes[i]
            corr_series = comp.prob_corr_by_window.dropna()

            if len(corr_series) > 0:
                ax.plot(corr_series.index, corr_series.values, marker=".", linewidth=1)
                ax.axhline(self.corr_threshold, color="red", linestyle="--", alpha=0.5)
                ax.axhline(0, color="gray", linestyle="-", alpha=0.3)

            ax.set_ylabel("ρ")
            ax.set_title(f"Tier {comp.tier_from} → {comp.tier_to}: +{comp.added_assets}")
            ax.set_ylim(-1, 1.1)
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel("Window End Date")
        plt.tight_layout()

        if save:
            path = OUTPUT_DIR / "prob_correlation_over_time.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
        plt.show()


if __name__ == "__main__":
    # Settings aligned with models/gmm/main.py so the stability table is
    # directly comparable to the headline dissertation regime fit:
    #   - window_size=1250 (≈ 5 years), step=63 (≈ 3 months) → 38 core windows
    #   - K_candidates=(2,3) — broad-market regime structure
    #   - scale_method="none" — raw returns (same as main.py)
    analyzer = RegimeStabilityAnalyzer(
        K_candidates=(2, 3),
        window_size=1250,
        step=63,
        scale_method="none",
        frequency="daily",
        start_date="2005-01-01",
    )

    # Define order to add assets
    add_order = [
        "us_equities",
        "eu_equities",
        "em_equities",
        "gov_long",
        "gov_short",
        "inflation_linked",
        "em_debt",
        "real_estate",
        "energy",
        "cash",
    ]

    summary = analyzer.run(add_order=add_order)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(summary.to_string())

    summary_path = OUTPUT_DIR / "stability_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    print("\n" + "="*70)
    print("BREAKING POINTS")
    print("="*70)
    breaking = analyzer.identify_breaking_points()
    if len(breaking) > 0:
        print(breaking.to_string())
    else:
        print("No breaking points - regimes stable across all expansions!")

    breaking_path = OUTPUT_DIR / "breaking_points.csv"
    breaking.to_csv(breaking_path, index=False)
    print(f"Saved: {breaking_path}")

    analyzer.plot_stability()
    analyzer.plot_K_evolution()
    analyzer.plot_prob_correlation_over_time()