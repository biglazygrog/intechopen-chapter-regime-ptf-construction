"""
Regime analysis and statistical testing.

Contains:
- RegimeDistributionalAnalysis: Full regime comparison methodology
- QuantileTestRow: Dataclass for quantile test results
- compare_means_vols_across_regimes_hard: Statistical comparisons
- Helper functions for moments, labels, diagnostics
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple
from scipy import stats

from core.utils import hard_labels_from_daily_probs as _hard_labels
from core.utils import benjamini_hochberg


@dataclass
class QuantileTestRow:
    asset: str
    q: float
    regime_1: int
    regime_2: int
    diff_q1_minus_q2: float
    p_value: float
    ci_low: float
    ci_high: float
    n1: int
    n2: int


def make_hard_labels_from_probs(daily_probs_df: pd.DataFrame) -> pd.Series:
    """Hard regime label per day using argmax of p_k columns.

    Delegates to :func:`core.utils.hard_labels_from_daily_probs` so that the
    p_-column ordering is numeric rather than lexicographic.
    """
    return _hard_labels(daily_probs_df)


def hard_labels_from_daily_probs(daily_probs_df: pd.DataFrame) -> np.ndarray:
    """Hard regime label per day, as a plain ndarray.

    Thin ndarray-returning wrapper over
    :func:`core.utils.hard_labels_from_daily_probs`. Both implementations used
    to exist independently and differed: this one selected p_ columns in
    DataFrame order (lexicographic once written to CSV), which silently
    diverges from numeric order at p_10 and beyond.

    The ndarray return type is load-bearing — :func:`regime_turnover_stats`
    compares ``labels[1:] != labels[:-1]``, which on a pandas Series would align
    on the index instead of position and produce nonsense.
    """
    return _hard_labels(daily_probs_df).to_numpy()


def compare_means_vols_across_regimes_hard(
    df_returns: pd.DataFrame,
    regime_labels: pd.Series,
    min_n: int = 30,
    alpha: float = 0.05
) -> dict:
    """
    Compare means and vols across regimes for each asset using hard labels.

    Returns dict with:
      - summary tables (ANOVA, Levene)
      - pairwise tables (Welch t-tests, pairwise Levene)
    """
    common_idx = df_returns.index.intersection(regime_labels.index)
    X = df_returns.loc[common_idx]
    z = regime_labels.loc[common_idx].astype(int)

    regimes = sorted(z.unique())
    assets = X.columns.tolist()

    anova_rows = []
    levene_rows = []
    ttest_rows = []
    plevene_rows = []

    for a in assets:
        groups = []
        sizes = []
        for k in regimes:
            g = X.loc[z == k, a].dropna().values
            if len(g) >= min_n:
                groups.append(g)
                sizes.append(len(g))
            else:
                groups.append(None)
                sizes.append(len(g))

        valid = [g for g in groups if g is not None]
        if len(valid) < 2:
            continue

        F, pF = stats.f_oneway(*valid)
        anova_rows.append({
            "asset": a,
            "F": F,
            "p_value": pF,
            "regimes_tested": len(valid),
            "min_group_n": min([len(g) for g in valid]),
        })

        W, pW = stats.levene(*valid, center='median')
        levene_rows.append({
            "asset": a,
            "W": W,
            "p_value": pW,
            "regimes_tested": len(valid),
            "min_group_n": min([len(g) for g in valid]),
        })

        for (k1, k2) in combinations(regimes, 2):
            g1 = X.loc[z == k1, a].dropna().values
            g2 = X.loc[z == k2, a].dropna().values
            if len(g1) < min_n or len(g2) < min_n:
                continue

            tstat, ptt = stats.ttest_ind(g1, g2, equal_var=False)
            ttest_rows.append({
                "asset": a,
                "regime_1": k1,
                "regime_2": k2,
                "t": tstat,
                "p_value": ptt,
                "mean_1": float(np.mean(g1)),
                "mean_2": float(np.mean(g2)),
                "vol_1": float(np.std(g1, ddof=1)),
                "vol_2": float(np.std(g2, ddof=1)),
                "n1": len(g1),
                "n2": len(g2),
            })

            W12, pW12 = stats.levene(g1, g2, center='median')
            plevene_rows.append({
                "asset": a,
                "regime_1": k1,
                "regime_2": k2,
                "W": W12,
                "p_value": pW12,
                "n1": len(g1),
                "n2": len(g2),
            })

    out = {
        "anova": pd.DataFrame(anova_rows).sort_values("p_value") if anova_rows else pd.DataFrame(),
        "levene": pd.DataFrame(levene_rows).sort_values("p_value") if levene_rows else pd.DataFrame(),
        "pairwise_ttests": pd.DataFrame(ttest_rows).sort_values("p_value") if ttest_rows else pd.DataFrame(),
        "pairwise_levene": pd.DataFrame(plevene_rows).sort_values("p_value") if plevene_rows else pd.DataFrame(),
    }
    return out


def compute_regime_moments_long(
    opt,
    X_raw: np.ndarray,
    feature_names: list,
    use_model_implied: bool = True,
) -> pd.DataFrame:
    """
    Returns a long-form DataFrame with regime mean/vol by asset, per window.

    Columns:
      window_idx, window_end, K, regime, asset, mean, vol, weight
    """
    rows = []
    for w, fit in enumerate(opt.fits):
        model = fit["model"]
        K = fit["K"]
        start = opt.window_start_indices_[w]
        end = opt.window_end_indices_[w] + 1
        Xw = X_raw[start:end]
        resp = fit["resp"]

        wts = resp.mean(axis=0)

        if use_model_implied:
            means_kd = model.means_
            vols_kd = np.sqrt(np.diagonal(model.covariances_, axis1=1, axis2=2))
        else:
            Nk = resp.sum(axis=0) + 1e-15
            means_kd = (resp.T @ Xw) / Nk[:, None]
            vols_kd = np.zeros_like(means_kd)
            for k in range(K):
                diff = Xw - means_kd[k]
                var = (resp[:, k][:, None] * (diff ** 2)).sum(axis=0) / Nk[k]
                vols_kd[k] = np.sqrt(np.maximum(var, 0.0))

        window_end_time = opt.window_end_times_[w]

        for k in range(K):
            for j, name in enumerate(feature_names):
                rows.append({
                    "window_idx": w,
                    "window_end": window_end_time,
                    "K": K,
                    "regime": k,
                    "asset": name,
                    "mean": float(means_kd[k, j]),
                    "vol": float(vols_kd[k, j]),
                    "weight": float(wts[k]),
                })

    return pd.DataFrame(rows)


def pooled_regime_summary(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pooled regime mean/vol by asset, weighting each window's regime moments by
    that window's regime probability (weight).
    """
    g = df_long.groupby(["regime", "asset"], as_index=False)

    out = g.apply(lambda x: pd.Series({
        "mean_wavg": np.average(x["mean"], weights=x["weight"]),
        "vol_wavg": np.average(x["vol"], weights=x["weight"]),
        "avg_regime_weight": x["weight"].mean(),
        "n_windows": x["window_idx"].nunique(),
    })).reset_index(drop=True)

    return out.sort_values(["regime", "asset"])


def regime_turnover_stats(labels: np.ndarray) -> dict:
    """Compute regime turnover statistics."""
    changes = (labels[1:] != labels[:-1])
    turnover_rate = float(np.mean(changes)) if len(labels) > 1 else np.nan

    run_lengths = []
    if len(labels) > 0:
        run = 1
        for t in range(1, len(labels)):
            if labels[t] == labels[t-1]:
                run += 1
            else:
                run_lengths.append(run)
                run = 1
        run_lengths.append(run)

    avg_run = float(np.mean(run_lengths)) if run_lengths else np.nan
    med_run = float(np.median(run_lengths)) if run_lengths else np.nan

    return {"turnover_rate": turnover_rate, "avg_run_length": avg_run, "med_run_length": med_run}


def cov_stability_diagnostics(opt) -> dict:
    """
    Check eigenvalues + condition numbers for each window/regime covariance.
    """
    min_eig_global = np.inf
    max_cond_global = 0.0
    n_near_singular = 0

    for fit in opt.fits:
        covs = fit["covariances"]
        for k in range(covs.shape[0]):
            C = covs[k]
            eigs = np.linalg.eigvalsh(C)
            min_eig = float(np.min(eigs))
            min_eig_global = min(min_eig_global, min_eig)

            if min_eig <= 0:
                cond = np.inf
            else:
                cond = float(np.max(eigs) / min_eig)
            max_cond_global = max(max_cond_global, cond)

            if min_eig < 1e-8:
                n_near_singular += 1

    return {
        "min_eigenvalue": float(min_eig_global) if np.isfinite(min_eig_global) else np.nan,
        "max_condition_number": float(max_cond_global) if np.isfinite(max_cond_global) else np.inf,
        "near_singular_count": int(n_near_singular),
    }


class RegimeDistributionalAnalysis:
    """
    Regime comparison & distributional inference after GMM regime identification.

    Implements methodology:

      (1) Moment-based comparisons:
          - regime-conditional mean/variance/vol tables
          - ANOVA (means) + Brown-Forsythe/Levene-median (vols)

      (2) Non-parametric distributional comparisons:
          - pairwise Kolmogorov-Smirnov tests (KS)
          - Anderson-Darling k-sample (AD)
          - Kruskal-Wallis (KW)

      (3) Quantile + downside-risk comparisons:
          - regime-conditional quantiles
          - regime-conditional VaR and ES
          - bootstrap pairwise quantile equality tests
    """

    def __init__(
        self,
        quantiles: Tuple[float, ...] = (0.05, 0.10, 0.25),
        var_levels: Tuple[float, ...] = (0.05, 0.10),
        min_n: int = 50,
        bootstrap_B: int = 1000,
        random_state: int = 42,
    ):
        self.quantiles = tuple(float(q) for q in quantiles)
        self.var_levels = tuple(float(a) for a in var_levels)
        self.min_n = int(min_n)
        self.bootstrap_B = int(bootstrap_B)
        self.random_state = int(random_state)
        self.rng = np.random.default_rng(random_state)

        for q in self.quantiles:
            if not (0.0 < q < 1.0):
                raise ValueError("All quantiles must be in (0,1).")
        for a in self.var_levels:
            if not (0.0 < a < 1.0):
                raise ValueError("All VaR/ES levels must be in (0,1).")

    def _align(self, df_returns: pd.DataFrame, regime_labels: pd.Series):
        common_idx = df_returns.index.intersection(regime_labels.index)
        X = df_returns.loc[common_idx].copy()
        z = regime_labels.loc[common_idx].astype(int).copy()
        return X, z

    def _stationary_bootstrap_indices(
        self,
        T: int,
        expected_block: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Politis & Romano (1994) stationary bootstrap index sequence.

        Blocks start at uniform random positions and have geometric length with
        mean ``expected_block`` (restart probability p = 1/expected_block).
        Indices wrap circularly, which is what makes the resampled series
        stationary. Returns ``T`` indices into the original series.
        """
        if expected_block < 1:
            raise ValueError(f"expected_block must be >= 1, got {expected_block}.")

        rng = self.rng if rng is None else rng
        p = 1.0 / float(expected_block)
        ar = np.arange(T)

        new_block = rng.random(T) < p
        new_block[0] = True

        # Position of the most recent block start, for every position.
        block_start_pos = np.maximum.accumulate(np.where(new_block, ar, 0))
        offset = ar - block_start_pos
        starts = rng.integers(0, T, size=T)

        return (starts[block_start_pos] + offset) % T

    def _valid_regime_samples(self, X: pd.DataFrame, z: pd.Series, asset: str) -> Dict[int, np.ndarray]:
        samples: Dict[int, np.ndarray] = {}
        for k in sorted(z.unique()):
            vals = X.loc[z == k, asset].dropna().values
            if vals.size >= self.min_n:
                samples[int(k)] = vals
        return samples

    def _var_es(self, x: np.ndarray, alpha: float) -> Tuple[float, float, int]:
        var = float(np.quantile(x, alpha))
        tail = x[x <= var]
        es = float(np.mean(tail)) if tail.size > 0 else float(np.nan)
        return var, es, int(tail.size)

    @staticmethod
    def _fmt_p(p: float) -> str:
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return ""
        if p < 0.001:
            return "<0.001"
        return f"{p:.3f}"

    @staticmethod
    def _fmt_num(x: float, nd: int = 4) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"{x:.{nd}f}"

    def regime_moments_table(self, df_returns: pd.DataFrame, regime_labels: pd.Series) -> pd.DataFrame:
        """Regime-conditional mean/variance/vol per asset."""
        X, z = self._align(df_returns, regime_labels)
        assets = X.columns.tolist()
        regimes = sorted(z.unique())

        rows = []
        for k in regimes:
            Xk = X.loc[z == k]
            for a in assets:
                vals = Xk[a].dropna().values
                n = int(vals.size)
                if n < self.min_n:
                    continue
                mu = float(np.mean(vals))
                var = float(np.var(vals, ddof=1)) if n >= 2 else float(np.nan)
                vol = float(np.sqrt(var)) if np.isfinite(var) else float(np.nan)
                rows.append({"regime": int(k), "asset": a, "mean": mu, "var": var, "vol": vol, "n": n})

        return pd.DataFrame(rows).sort_values(["asset", "regime"])

    def moment_tests(self, df_returns: pd.DataFrame, regime_labels: pd.Series) -> Dict[str, pd.DataFrame]:
        """ANOVA and Brown-Forsythe tests."""
        tests = compare_means_vols_across_regimes_hard(
            df_returns=df_returns,
            regime_labels=regime_labels,
            min_n=self.min_n
        )

        levene_df = tests["levene"].copy()
        if len(levene_df) > 0:
            levene_df = levene_df.rename(columns={"W": "BF_stat", "p_value": "BF_p_value"})

        anova_df = tests["anova"].copy()
        if len(anova_df) > 0:
            anova_df = anova_df.rename(columns={"F": "ANOVA_F", "p_value": "ANOVA_p_value"})

        return {
            "anova_means": anova_df.sort_values("ANOVA_p_value") if len(anova_df) > 0 else anova_df,
            "brown_forsythe_vols": levene_df.sort_values("BF_p_value") if len(levene_df) > 0 else levene_df,
            "pairwise_welch_means": tests["pairwise_ttests"].sort_values("p_value") if len(tests["pairwise_ttests"]) > 0 else tests["pairwise_ttests"],
            "pairwise_bf_vols": tests["pairwise_levene"].sort_values("p_value") if len(tests["pairwise_levene"]) > 0 else tests["pairwise_levene"],
        }

    def ks_pairwise_tests(self, df_returns: pd.DataFrame, regime_labels: pd.Series) -> pd.DataFrame:
        """Pairwise Kolmogorov-Smirnov tests across regimes."""
        X, z = self._align(df_returns, regime_labels)
        assets = X.columns.tolist()

        rows = []
        for a in assets:
            samples = self._valid_regime_samples(X, z, a)
            regs = sorted(samples.keys())
            if len(regs) < 2:
                continue

            for k1, k2 in combinations(regs, 2):
                res = stats.ks_2samp(samples[k1], samples[k2], alternative="two-sided", mode="auto")
                rows.append({
                    "asset": a,
                    "regime_1": int(k1),
                    "regime_2": int(k2),
                    "ks_stat": float(res.statistic),
                    "p_value": float(res.pvalue),
                    "n1": int(samples[k1].size),
                    "n2": int(samples[k2].size),
                })

        return pd.DataFrame(rows).sort_values(["p_value", "asset", "regime_1", "regime_2"]) if rows else pd.DataFrame()

    def anderson_darling_k_sample_tests(self, df_returns: pd.DataFrame, regime_labels: pd.Series) -> pd.DataFrame:
        """Omnibus Anderson-Darling k-sample test."""
        X, z = self._align(df_returns, regime_labels)
        assets = X.columns.tolist()

        rows = []
        for a in assets:
            samples = self._valid_regime_samples(X, z, a)
            if len(samples) < 2:
                continue

            arrs = [samples[k] for k in sorted(samples.keys())]
            try:
                res = stats.anderson_ksamp(arrs)
                stat = float(getattr(res, "statistic", np.nan))
                sig_level = getattr(res, "significance_level", None)
                p_approx = float(sig_level) / 100.0 if sig_level is not None else np.nan
            except Exception:
                stat, p_approx = np.nan, np.nan

            rows.append({
                "asset": a,
                "ad_stat": float(stat) if np.isfinite(stat) else np.nan,
                "p_value_approx": float(p_approx) if np.isfinite(p_approx) else np.nan,
                "regimes_tested": int(len(samples)),
                "min_group_n": int(min(len(v) for v in samples.values())),
            })

        return pd.DataFrame(rows).sort_values(["p_value_approx", "asset"]) if rows else pd.DataFrame()

    def kruskal_wallis_tests(self, df_returns: pd.DataFrame, regime_labels: pd.Series) -> pd.DataFrame:
        """Omnibus Kruskal-Wallis test."""
        X, z = self._align(df_returns, regime_labels)
        assets = X.columns.tolist()

        rows = []
        for a in assets:
            samples = self._valid_regime_samples(X, z, a)
            if len(samples) < 2:
                continue

            arrs = [samples[k] for k in sorted(samples.keys())]
            H, p = stats.kruskal(*arrs)
            rows.append({
                "asset": a,
                "kw_H": float(H),
                "p_value": float(p),
                "regimes_tested": int(len(samples)),
                "min_group_n": int(min(len(v) for v in samples.values())),
            })

        return pd.DataFrame(rows).sort_values(["p_value", "asset"]) if rows else pd.DataFrame()

    def regime_quantiles_table(self, df_returns: pd.DataFrame, regime_labels: pd.Series) -> pd.DataFrame:
        """Regime-conditional empirical quantiles per asset."""
        X, z = self._align(df_returns, regime_labels)
        assets = X.columns.tolist()
        regimes = sorted(z.unique())

        rows = []
        for k in regimes:
            Xk = X.loc[z == k]
            for a in assets:
                vals = Xk[a].dropna().values
                n = int(vals.size)
                if n < self.min_n:
                    continue
                for q in self.quantiles:
                    rows.append({
                        "regime": int(k),
                        "asset": a,
                        "q": float(q),
                        "quantile": float(np.quantile(vals, q)),
                        "n": n,
                    })

        return pd.DataFrame(rows).sort_values(["asset", "q", "regime"])

    def regime_var_es_table(self, df_returns: pd.DataFrame, regime_labels: pd.Series) -> pd.DataFrame:
        """Regime-conditional empirical VaR and ES."""
        X, z = self._align(df_returns, regime_labels)
        assets = X.columns.tolist()
        regimes = sorted(z.unique())

        rows = []
        for k in regimes:
            Xk = X.loc[z == k]
            for a in assets:
                vals = Xk[a].dropna().values
                n = int(vals.size)
                if n < self.min_n:
                    continue

                for alpha in self.var_levels:
                    var, es, n_tail = self._var_es(vals, alpha)
                    rows.append({
                        "regime": int(k),
                        "asset": a,
                        "alpha": float(alpha),
                        "var": float(var),
                        "es": float(es),
                        "n": n,
                        "n_tail": int(n_tail),
                    })

        return pd.DataFrame(rows).sort_values(["asset", "alpha", "regime"])

    def quantile_equality_tests(self, df_returns: pd.DataFrame, regime_labels: pd.Series) -> pd.DataFrame:
        """Pairwise bootstrap tests of quantile equality.

        NOT TIME-SERIES VALID, AND NOT USED FOR ANY REPORTED RESULT.

        The resample below is ``rng.choice(x_k, size=n_k, replace=True)`` — iid
        over rows within each regime, exactly the procedure
        ``correlation_difference_tests`` was rewritten to remove. Daily returns
        cluster in volatility, so treating days as independent overstates the
        effective sample size: the intervals are too narrow and the p-values too
        small. Read the numbers accordingly.

        It is retained, unfixed, because nothing in the reproduction package
        reports it. Its output reaches ``paper_tables`` as the ``Q*_p`` columns
        of ``paper_comparison_raw``, which ``research/analysis/pipeline.py``
        never writes to disk — only ``paper_profiles_raw`` is saved. Table 3
        (``research/analysis/moments.py``) carries point estimates only: no
        confidence intervals and no p-values. Nothing else calls this method.

        If any future artefact does come to report these p-values, replace the
        resample with the stationary block bootstrap in
        ``_stationary_bootstrap_indices`` first — see
        ``correlation_difference_tests`` for the procedure and for why a
        within-regime block bootstrap is not applicable to this data.
        """
        X, z = self._align(df_returns, regime_labels)
        assets = X.columns.tolist()

        out: List[QuantileTestRow] = []

        for a in assets:
            samples = self._valid_regime_samples(X, z, a)
            regs = sorted(samples.keys())
            if len(regs) < 2:
                continue

            for (k1, k2) in combinations(regs, 2):
                x1 = samples[k1]
                x2 = samples[k2]
                n1, n2 = len(x1), len(x2)

                for q in self.quantiles:
                    diff = float(np.quantile(x1, q) - np.quantile(x2, q))

                    B = self.bootstrap_B
                    boots = np.empty(B, dtype=float)
                    for b in range(B):
                        b1 = self.rng.choice(x1, size=n1, replace=True)
                        b2 = self.rng.choice(x2, size=n2, replace=True)
                        boots[b] = np.quantile(b1, q) - np.quantile(b2, q)

                    ci_low = float(np.quantile(boots, 0.025))
                    ci_high = float(np.quantile(boots, 0.975))

                    p = float(2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))
                    p = min(p, 1.0)

                    out.append(QuantileTestRow(
                        asset=a, q=float(q),
                        regime_1=int(k1), regime_2=int(k2),
                        diff_q1_minus_q2=diff,
                        p_value=p,
                        ci_low=ci_low, ci_high=ci_high,
                        n1=int(n1), n2=int(n2)
                    ))

        return pd.DataFrame([r.__dict__ for r in out]).sort_values(["p_value", "asset", "q"]) if out else pd.DataFrame()

    def select_regime_pair(
        self,
        df_returns: pd.DataFrame,
        regime_labels: pd.Series,
        mode: str = "ordered",
    ) -> Tuple[int, int]:
        """Choose two regimes to compare."""
        X, z = self._align(df_returns, regime_labels)
        regs = sorted(z.unique())

        if len(regs) < 2:
            raise ValueError("Need at least 2 regimes to compare.")

        if mode == "ordered":
            return int(regs[0]), int(regs[-1])

        if mode == "realised_vol":
            vols = {}
            for k in regs:
                Xk = X.loc[z == k]
                if len(Xk) < self.min_n:
                    continue
                v = Xk.std(axis=0, ddof=1).replace([np.inf, -np.inf], np.nan).dropna()
                if len(v) > 0:
                    vols[int(k)] = float(v.mean())
            if len(vols) < 2:
                return int(regs[0]), int(regs[-1])
            return int(min(vols, key=vols.get)), int(max(vols, key=vols.get))

        raise ValueError("mode must be 'ordered' or 'realised_vol'.")

    def paper_tables(
        self,
        df_returns: pd.DataFrame,
        regime_labels: pd.Series,
        pair_mode: str = "ordered",
        alpha_main: float = 0.05,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate paper-ready summary tables.

        Returns:
          meta_selected_pair
          paper_profiles_raw, paper_comparison_raw, paper_omnibus_raw
          paper_profiles_fmt, paper_comparison_fmt, paper_omnibus_fmt
        """
        X, z = self._align(df_returns, regime_labels)
        assets = X.columns.tolist()

        r_low, r_high = self.select_regime_pair(df_returns, regime_labels, mode=pair_mode)

        def _stats_for_regime(k: int) -> pd.DataFrame:
            rows = []
            Xk = X.loc[z == k]
            for a in assets:
                vals = Xk[a].dropna().values
                n = int(vals.size)
                if n < self.min_n:
                    continue

                mu = float(np.mean(vals))
                vol = float(np.std(vals, ddof=1)) if n >= 2 else np.nan

                var_a = float(np.quantile(vals, alpha_main))
                tail = vals[vals <= var_a]
                es_a = float(np.mean(tail)) if tail.size > 0 else np.nan

                row = {
                    "asset": a,
                    "n": n,
                    "mean": mu,
                    "vol": vol,
                    f"VaR{int(alpha_main*100)}": var_a,
                    f"ES{int(alpha_main*100)}": es_a,
                }
                for q in self.quantiles:
                    row[f"Q{int(100*q)}"] = float(np.quantile(vals, q))
                rows.append(row)

            dfk = pd.DataFrame(rows).set_index("asset")
            dfk = dfk.add_prefix(f"R{k}_")
            return dfk

        prof_low = _stats_for_regime(r_low)
        prof_high = _stats_for_regime(r_high)
        paper_profiles = prof_low.join(prof_high, how="outer").reset_index()

        profiles_idx = paper_profiles.set_index("asset")
        comp = pd.DataFrame(index=profiles_idx.index)

        base_cols = ["mean", "vol", f"VaR{int(alpha_main*100)}", f"ES{int(alpha_main*100)}"]
        q_cols = [f"Q{int(100*q)}" for q in self.quantiles]

        for c in base_cols + q_cols:
            comp[f"delta_{c}_R{r_high}-R{r_low}"] = (
                profiles_idx.get(f"R{r_high}_{c}") - profiles_idx.get(f"R{r_low}_{c}")
            )

        # KS pairwise for selected pair
        ks = self.ks_pairwise_tests(df_returns, regime_labels)
        if len(ks) > 0:
            ks_pair = ks[
                ((ks["regime_1"] == r_low) & (ks["regime_2"] == r_high)) |
                ((ks["regime_1"] == r_high) & (ks["regime_2"] == r_low))
            ].set_index("asset")[["ks_stat", "p_value"]].rename(columns={"p_value": "KS_p_value"})
            comp = comp.join(ks_pair, how="left")

        # KW omnibus p-value
        kw = self.kruskal_wallis_tests(df_returns, regime_labels)
        if len(kw) > 0:
            kw = kw.set_index("asset")[["p_value"]].rename(columns={"p_value": "KW_p_value"})
            comp = comp.join(kw, how="left")

        # Bootstrap quantile equality p-values
        qeq = self.quantile_equality_tests(df_returns, regime_labels)
        if len(qeq) > 0:
            qeq_pair = qeq[
                ((qeq["regime_1"] == r_low) & (qeq["regime_2"] == r_high)) |
                ((qeq["regime_1"] == r_high) & (qeq["regime_2"] == r_low))
            ].copy()

            if len(qeq_pair) > 0:
                qeq_pair["q_label"] = qeq_pair["q"].map(lambda x: f"Q{int(100*x)}_p")
                qeq_wide = qeq_pair.pivot_table(index="asset", columns="q_label", values="p_value", aggfunc="min")
                comp = comp.join(qeq_wide, how="left")

        paper_comparison = comp.reset_index()

        # Omnibus tests per asset
        mt = self.moment_tests(df_returns, regime_labels)
        anova = mt["anova_means"]
        if len(anova) > 0:
            anova = anova.set_index("asset")[["ANOVA_p_value"]]
        else:
            anova = pd.DataFrame()

        bf = mt["brown_forsythe_vols"]
        if len(bf) > 0:
            bf = bf.set_index("asset")[["BF_p_value"]]
        else:
            bf = pd.DataFrame()

        kw2 = self.kruskal_wallis_tests(df_returns, regime_labels)
        if len(kw2) > 0:
            kw2 = kw2.set_index("asset")[["p_value"]].rename(columns={"p_value": "KW_p_value"})
        else:
            kw2 = pd.DataFrame()

        ad = self.anderson_darling_k_sample_tests(df_returns, regime_labels)
        if len(ad) > 0:
            ad = ad.set_index("asset")[["p_value_approx"]].rename(columns={"p_value_approx": "AD_p_value_approx"})
        else:
            ad = pd.DataFrame()

        paper_omnibus = anova.join(bf, how="outer").join(kw2, how="outer").join(ad, how="outer").reset_index()

        def format_profiles(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            for c in out.columns:
                if c == "asset":
                    continue
                if c.endswith("_n"):
                    out[c] = out[c].apply(lambda x: "" if pd.isna(x) else str(int(x)))
                else:
                    out[c] = out[c].apply(lambda x: self._fmt_num(x, 4))
            return out

        def format_comparison(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            for c in out.columns:
                if c == "asset":
                    continue
                if c in ["KS_p_value", "KW_p_value"] or c.endswith("_p"):
                    out[c] = out[c].apply(self._fmt_p)
                elif c == "ks_stat" or c.startswith("delta_"):
                    out[c] = out[c].apply(lambda x: self._fmt_num(x, 4))
                else:
                    out[c] = out[c].apply(lambda x: self._fmt_num(x, 4))
            return out

        def format_omnibus(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            for c in out.columns:
                if c == "asset":
                    continue
                out[c] = out[c].apply(self._fmt_p)
            return out

        return {
            "meta_selected_pair": pd.DataFrame([{
                "regime_low": r_low,
                "regime_high": r_high,
                "alpha_main": alpha_main,
                "pair_mode": pair_mode
            }]),
            "paper_profiles_raw": paper_profiles,
            "paper_comparison_raw": paper_comparison,
            "paper_omnibus_raw": paper_omnibus,
            "paper_profiles_fmt": format_profiles(paper_profiles),
            "paper_comparison_fmt": format_comparison(paper_comparison),
            "paper_omnibus_fmt": format_omnibus(paper_omnibus),
        }

    def correlation_difference_tests(
            self,
            df_returns: pd.DataFrame,
            regime_labels: pd.Series,
            regime_low: Optional[int] = None,
            regime_high: Optional[int] = None,
            bootstrap_B: Optional[int] = None,
            ci_level: float = 0.95,
            block_length: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Stationary-block bootstrap test for regime differences in correlation.

        The resampling unit is a BLOCK OF CONSECUTIVE CALENDAR DAYS drawn from
        the full return series with the regime labels carried along, not an
        individual day drawn from within a regime. Each replicate resamples the
        whole series, then splits the resampled rows by their carried labels and
        recomputes both regime correlation matrices.

        Why not a within-regime block bootstrap
        ---------------------------------------
        The obvious reading of "block bootstrap by regime" — draw blocks of
        ``floor(sqrt(N_k))`` from each regime's own observations — is not
        applicable to this data. The regime-conditional subsamples are not
        contiguous time series. Under the retrospective labels R1's 788
        observations arrive as 519 separate runs with MEDIAN LENGTH 1 DAY and a
        maximum of 23; not one run reaches the ``floor(sqrt(788)) = 28`` block
        that rule would prescribe. Blocks drawn from the concatenated subsample
        would splice together days separated by months or years — preserving no
        real serial dependence while carrying the appearance of a time-series
        method. Resampling the intact series instead keeps blocks on the real
        time axis, and additionally propagates uncertainty in the regime
        labels, which the previous procedure conditioned away entirely.

        The superseded procedure drew ``rng.choice(n_k, size=n_k, replace=True)``
        independently within each regime. Daily returns cluster in volatility,
        so treating days as independent overstates the effective sample size and
        understates the variance of the correlation estimator: intervals came
        out too narrow and p-values too small.

        Parameters
        ----------
        block_length : int, optional
            EXPECTED block length for the stationary bootstrap (Politis &
            Romano, 1994), whose block lengths are geometric with mean
            ``block_length``. Defaults to ``floor(sqrt(T))`` on the full series
            — 78 observations on the 6,269-row corrected grid — as a
            rule-of-thumb choice.

        Notes
        -----
        ``p_value_boot`` is the raw two-sided bootstrap p-value; ``q_value_bh``
        is the Benjamini-Hochberg adjustment across all C(d,2) pairwise tests
        (15 for the six-asset core universe). Table 4's stars are assigned on
        ``q_value_bh``.

        The pairwise frame carries TWO significance columns and they are not
        expected to agree. ``significant_ci_excludes_zero`` reads each pair's
        percentile CI on its own and is UNADJUSTED for multiple testing;
        ``significant_bh_q05`` is ``q_value_bh < 0.05`` and is the adjusted
        decision the reported stars follow. The ``ci_low``/``ci_high`` columns —
        and the CI columns printed in Table 4 — are likewise per-pair and
        unadjusted, so a pair can show a CI excluding zero while carrying fewer
        than two stars.

        The aggregate frame keeps ``significant_ci_excludes_zero`` only: it is a
        single test, so there is no multiplicity to adjust for.
        """
        X, z = self._align(df_returns, regime_labels)
        assets = X.columns.tolist()
        d = len(assets)

        if d < 2:
            raise ValueError("Need at least 2 assets to compute correlations.")

        regs = sorted(z.unique())
        if len(regs) < 2:
            raise ValueError("Need at least 2 regimes in regime_labels.")

        if regime_low is None or regime_high is None:
            r_low, r_high = self.select_regime_pair(df_returns, regime_labels, mode="ordered")
        else:
            r_low, r_high = int(regime_low), int(regime_high)

        X_low = X.loc[z == r_low].dropna(how="any")
        X_high = X.loc[z == r_high].dropna(how="any")

        n0, n1 = len(X_low), len(X_high)
        if n0 < self.min_n or n1 < self.min_n:
            raise ValueError(
                f"Not enough observations: n_low={n0}, n_high={n1}, min_n={self.min_n}."
            )

        B = int(bootstrap_B) if bootstrap_B is not None else int(self.bootstrap_B)
        alpha = 1.0 - ci_level
        q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

        def corr_mat(arr2d: np.ndarray) -> np.ndarray:
            C = np.corrcoef(arr2d, rowvar=False)
            return np.clip(C, -1.0, 1.0)

        C0 = corr_mat(X_low.values)
        C1 = corr_mat(X_high.values)
        D_hat = C1 - C0

        iu = np.triu_indices(d, k=1)

        def mean_abs_corr(C: np.ndarray) -> float:
            return float(np.mean(np.abs(C[iu])))

        rho_bar0 = mean_abs_corr(C0)
        rho_bar1 = mean_abs_corr(C1)
        delta_rho_bar_hat = rho_bar1 - rho_bar0

        # ---- Stationary block bootstrap on the FULL series ------------------
        # Rows are resampled in blocks of consecutive calendar days, carrying
        # their regime labels; each replicate is then split by the carried
        # labels. See the docstring for why a within-regime block bootstrap is
        # not applicable here.
        keep = X.notna().all(axis=1).values
        Xv = X.values[keep]
        zv = z.values[keep]
        T_full = len(Xv)

        L = int(block_length) if block_length is not None else int(np.floor(np.sqrt(T_full)))

        boot_pair = np.empty((B, len(iu[0])), dtype=float)
        boot_delta_rhobar = np.empty(B, dtype=float)

        # A replicate is rejected if either regime draws too few rows to give a
        # well-conditioned correlation matrix, or if a resampled column is
        # constant (corrcoef -> NaN). Rejections are redrawn, counted, and
        # reported: a silently dropped replicate would bias the interval.
        min_rows = max(int(self.min_n), d + 1)
        max_attempts = 100
        n_rejected = 0

        # Dedicated generator, seeded from random_state, so this test reproduces
        # identically whether it is run inside the full pipeline (after
        # paper_tables has already consumed draws from self.rng) or standalone.
        # The previous code drew from the shared self.rng, which made the result
        # depend on call order.
        boot_rng = np.random.default_rng(self.random_state)

        b = 0
        while b < B:
            attempts = 0
            while True:
                attempts += 1
                idx = self._stationary_bootstrap_indices(T_full, L, rng=boot_rng)
                zb = zv[idx]
                m0 = zb == r_low
                m1 = zb == r_high

                if m0.sum() >= min_rows and m1.sum() >= min_rows:
                    C0b = corr_mat(Xv[idx][m0])
                    C1b = corr_mat(Xv[idx][m1])
                    if np.isfinite(C0b).all() and np.isfinite(C1b).all():
                        break

                n_rejected += 1
                if attempts >= max_attempts:
                    raise RuntimeError(
                        f"Stationary bootstrap could not draw a usable replicate "
                        f"in {max_attempts} attempts (need >= {min_rows} rows in "
                        f"each of regimes {r_low}/{r_high}). Check the regime "
                        f"balance or lower min_n."
                    )

            Db = (C1b - C0b)
            boot_pair[b, :] = Db[iu]
            boot_delta_rhobar[b] = mean_abs_corr(C1b) - mean_abs_corr(C0b)
            b += 1

        self.last_bootstrap_info_ = {
            "method": "stationary_block",
            "block_length": L,
            "T_full": T_full,
            "B": B,
            "n_rejected": n_rejected,
        }

        rows = []
        for col, (i, j) in enumerate(zip(iu[0], iu[1])):
            boots = boot_pair[:, col]
            ci_low_val = float(np.quantile(boots, q_lo))
            ci_high_val = float(np.quantile(boots, q_hi))

            p = float(2.0 * min(np.mean(boots <= 0.0), np.mean(boots >= 0.0)))
            p = min(p, 1.0)

            rows.append({
                "asset_i": assets[i],
                "asset_j": assets[j],
                "regime_low": r_low,
                "regime_high": r_high,
                "rho_low": float(C0[i, j]),
                "rho_high": float(C1[i, j]),
                "delta_rho_high_minus_low": float(D_hat[i, j]),
                "ci_low": ci_low_val,
                "ci_high": ci_high_val,
                "p_value_boot": p,
                "significant_ci_excludes_zero": bool((ci_low_val > 0.0) or (ci_high_val < 0.0)),
                "n_low": int(n0),
                "n_high": int(n1),
                "B": int(B),
                "ci_level": float(ci_level),
            })

        pairwise_df = pd.DataFrame(rows)

        # BH adjustment across all C(d,2) pairwise tests — 15 for the six-asset
        # core universe. Table 4's stars are assigned on q, not p.
        pairwise_df["q_value_bh"] = benjamini_hochberg(pairwise_df["p_value_boot"].values)
        # Two significance columns, deliberately kept side by side and NOT
        # reconciled. `significant_ci_excludes_zero` is the per-pair percentile
        # CI read on its own, unadjusted for multiple testing;
        # `significant_bh_q05` is the multiplicity-adjusted decision and is the
        # one Table 4's stars follow. They disagree on pairs whose raw p is
        # below 0.05 but whose q is not — on the corrected grid that is
        # World Equities ~ Cash (p = 0.019, q = 0.057), so the CI flags 5 pairs
        # where BH flags 4. That disagreement is the multiplicity correction
        # doing its job, not an inconsistency to be papered over.
        pairwise_df["significant_bh_q05"] = pairwise_df["q_value_bh"] < 0.05
        pairwise_df["n_tests_bh"] = len(pairwise_df)
        pairwise_df["bootstrap_method"] = "stationary_block"
        pairwise_df["block_length"] = L

        pairwise_df = pairwise_df.sort_values(["p_value_boot", "asset_i", "asset_j"])

        ci_low_agg = float(np.quantile(boot_delta_rhobar, q_lo))
        ci_high_agg = float(np.quantile(boot_delta_rhobar, q_hi))
        p_agg = float(2.0 * min(np.mean(boot_delta_rhobar <= 0.0), np.mean(boot_delta_rhobar >= 0.0)))
        p_agg = min(p_agg, 1.0)

        aggregate_df = pd.DataFrame([{
            "regime_low": r_low,
            "regime_high": r_high,
            "rho_bar_low": float(rho_bar0),
            "rho_bar_high": float(rho_bar1),
            "delta_rho_bar_high_minus_low": float(delta_rho_bar_hat),
            "ci_low": ci_low_agg,
            "ci_high": ci_high_agg,
            "p_value_boot": p_agg,
            "significant_ci_excludes_zero": bool((ci_low_agg > 0.0) or (ci_high_agg < 0.0)),
            "n_low": int(n0),
            "n_high": int(n1),
            "B": int(B),
            "ci_level": float(ci_level),
            "bootstrap_method": "stationary_block",
            "block_length": int(L),
        }])

        return {
            "correlation_pairwise": pairwise_df,
            "correlation_aggregate": aggregate_df,
        }

    def run_all(self, df_returns: pd.DataFrame, regime_labels: pd.Series) -> Dict[str, pd.DataFrame]:
        """Full set of outputs."""
        out: Dict[str, pd.DataFrame] = {}

        out["regime_moments"] = self.regime_moments_table(df_returns, regime_labels)
        out.update(self.moment_tests(df_returns, regime_labels))

        out["ks_pairwise"] = self.ks_pairwise_tests(df_returns, regime_labels)
        out["anderson_darling_k_sample"] = self.anderson_darling_k_sample_tests(df_returns, regime_labels)
        out["kruskal_wallis"] = self.kruskal_wallis_tests(df_returns, regime_labels)

        out["regime_quantiles"] = self.regime_quantiles_table(df_returns, regime_labels)
        out["regime_var_es"] = self.regime_var_es_table(df_returns, regime_labels)

        out["tests_quantile_equality"] = self.quantile_equality_tests(df_returns, regime_labels)

        return out
