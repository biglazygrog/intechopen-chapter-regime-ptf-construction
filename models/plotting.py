"""
Plotting functions for GMM regime analysis.

Contains all visualization functions for:
- BIC results, log-likelihood convergence
- Regime probabilities
- Parameter drift
- Diagnostics
- Correlation matrices
- Effective sample size
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_chart(filename: str, dpi: int = 150):
    """Save current figure to output directory and close it."""
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def plot_bic_results(results, title="GMM Model Selection by BIC", save_as="bic_results.png"):
    Ks = [r["K"] for r in results]
    bics = [r["bic"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(Ks, bics, marker="o")
    plt.xlabel("Number of regimes (components)")
    plt.ylabel("BIC (mclust-style, higher is better)")
    plt.title(title)
    plt.grid(True)

    best_idx = int(np.argmax(bics))
    best_K = Ks[best_idx]
    best_BIC = bics[best_idx]

    plt.scatter([best_K], [best_BIC], zorder=10)
    plt.annotate(
        f"Best K = {best_K}",
        xy=(best_K, best_BIC),
        xytext=(
            best_K + 0.2,
            best_BIC + 0.2 * (max(bics) - min(bics) + 1.0),
        ),
        arrowprops=dict(arrowstyle="->"),
        fontsize=10,
    )

    plt.tight_layout()
    save_chart(save_as)


def plot_log_likelihood(gmm, save_as="log_likelihood.png"):
    ll = gmm.history_["log_likelihood"]
    iters = np.arange(1, len(ll) + 1)

    plt.figure()
    plt.plot(iters, ll, marker="o")
    plt.xlabel("EM iteration")
    plt.ylabel("Log-likelihood")
    plt.title("GMM log-likelihood convergence")
    plt.grid(True)
    save_chart(save_as)


def plot_component_weight(gmm, component_index, save_as=None):
    if save_as is None:
        save_as = f"component_weight_{component_index}.png"
    W = gmm.history_["weights"]
    iters = np.arange(1, W.shape[0] + 1)

    plt.figure()
    plt.plot(iters, W[:, component_index], marker="o")
    plt.xlabel("EM iteration")
    plt.ylabel(f"Weight of component {component_index}")
    plt.title(f"Weight trajectory for component {component_index}")
    plt.grid(True)
    save_chart(save_as)


def plot_feature_mean(gmm, component_index, feature_index, feature_name=None, save_as=None):
    if save_as is None:
        save_as = f"feature_mean_c{component_index}_f{feature_index}.png"
    M = gmm.history_["means"]
    iters = np.arange(1, M.shape[0] + 1)

    if feature_name is None:
        feature_name = f"feature {feature_index}"

    plt.figure()
    plt.plot(iters, M[:, component_index, feature_index], marker="o")
    plt.xlabel("EM iteration")
    plt.ylabel(f"Mean of {feature_name}")
    plt.title(f"Mean trajectory for {feature_name} (component {component_index})")
    plt.grid(True)
    save_chart(save_as)


def plot_responsibilities_for_sample(gmm, sample_index, components=None, save_as=None):
    if save_as is None:
        save_as = f"responsibilities_sample_{sample_index}.png"
    resp_history = gmm.history_["resp"]
    T = len(resp_history)
    K = gmm.n_components

    if components is None:
        components = list(range(K))

    R = np.zeros((T, K))
    for t in range(T):
        R[t, :] = resp_history[t][sample_index, :]

    iters = np.arange(1, T + 1)

    plt.figure()
    for k in components:
        plt.plot(iters, R[:, k], marker="o", label=f"Component {k}")

    plt.xlabel("EM iteration")
    plt.ylabel("Responsibility")
    plt.title(f"Responsibilities for sample {sample_index} over EM iterations")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    save_chart(save_as)


def plot_regime_probabilities(
    probs_df: pd.DataFrame,
    normalize: bool = True,
    stacked: bool = True,
    colors: Optional[List[str]] = None,
    alpha: float = 0.85,
    title: str = "Regime probabilities",
    save_as: str = "regime_probabilities.png",
):
    cols = [c for c in probs_df.columns if c.startswith("p_")]
    if not cols:
        raise ValueError("probs_df must contain columns named p_0, p_1, ...")

    cols = sorted(cols, key=lambda c: int(c.split("_")[1]))
    probs = probs_df[cols].copy()

    if normalize:
        probs = probs.div(
            probs.sum(axis=1).replace(0, np.nan), axis=0
        ).fillna(0.0)

    K = probs.shape[1]
    if colors is not None and len(colors) != K:
        raise ValueError(f"colors must be a list of length {K}")

    plt.figure(figsize=(12, 4))

    if stacked:
        plt.stackplot(
            probs.index,
            probs.values.T,
            labels=cols,
            colors=colors,
            alpha=alpha,
        )
    else:
        for k, col in enumerate(cols):
            plt.plot(
                probs.index,
                probs[col].values,
                label=col,
                color=None if colors is None else colors[k],
            )

    plt.ylim(0, 1)
    plt.ylabel("Probability")
    plt.xlabel("Time")
    plt.title(title)

    try:
        plt.legend(loc="lower right", ncols=min(K, 4), frameon=False)
    except TypeError:
        plt.legend(loc="lower right", frameon=False)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_chart(save_as)


def plot_selected_regimes(opt, title="Number of regimes selected per window", save_as="selected_regimes.png"):
    """Plot the number of regimes (K) selected in each window."""
    if hasattr(opt, "selected_Ks_") and len(opt.selected_Ks_) > 0:
        Kvals = opt.selected_Ks_
    elif hasattr(opt, "chosen_K") and len(opt.chosen_K) > 0:
        Kvals = opt.chosen_K
    else:
        raise ValueError("No selected regimes found. Did you run opt.fit()?")

    if hasattr(opt, "window_end_times_") and len(opt.window_end_times_) == len(Kvals):
        x = opt.window_end_times_
    elif hasattr(opt, "window_end_indices_") and len(opt.window_end_indices_) == len(Kvals):
        x = opt.window_end_indices_
    elif hasattr(opt, "window_ends_") and len(opt.window_ends_) == len(Kvals):
        if getattr(opt, "index_", None) is not None:
            x = [opt.index_[i] for i in opt.window_ends_]
        else:
            x = opt.window_ends_
    else:
        x = list(range(len(Kvals)))

    plt.figure(figsize=(10, 4))
    plt.step(x, Kvals, where="mid", marker="o")
    plt.ylim(0, max(Kvals) + 1)

    plt.title(title)
    plt.xlabel("Window end")
    plt.ylabel("Selected K")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_chart(save_as)


def plot_gmm_diagnostics(opt, title_prefix="GMM rolling", save_as="gmm_diagnostics.png"):
    """Plots EM iterations, BIC, and Log-likelihood per window."""
    if hasattr(opt, "window_end_times_") and len(opt.window_end_times_) > 0:
        x = opt.window_end_times_
    elif hasattr(opt, "window_end_indices_") and len(opt.window_end_indices_) > 0:
        x = opt.window_end_indices_
    elif hasattr(opt, "window_ends_") and len(opt.window_ends_) > 0:
        x = [opt.index_[i] for i in opt.window_ends_] if getattr(opt, "index_", None) is not None else opt.window_ends_
    else:
        n = len(getattr(opt, "window_n_iter_", []))
        x = list(range(n))

    iters = getattr(opt, "window_n_iter_", [])
    bics = getattr(opt, "window_bic_", [])
    logliks = getattr(opt, "window_loglik_", [])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(x, iters, marker="o")
    axes[0].set_title(title_prefix + ": EM iterations per window")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, bics, marker="o")
    axes[1].set_title(title_prefix + ": BIC per window")
    axes[1].grid(alpha=0.3)

    axes[2].plot(x, logliks, marker="o")
    axes[2].set_title(title_prefix + ": Log-likelihood per window")
    axes[2].grid(alpha=0.3)
    axes[2].set_xlabel("Window end")

    plt.tight_layout()
    save_chart(save_as)


def plot_parameter_drift(opt, feature_names=None, title_prefix="GMM rolling: ", save_prefix="param_drift"):
    """Plot the drift of regime means, variances, and weights across windows."""
    fits = opt.fits
    if len(fits) == 0:
        raise ValueError("No fits found. Run opt.fit() first.")

    models = [f["model"] for f in fits]
    n_windows = len(models)
    d = models[0].means_.shape[1]
    K_max = max(m.n_components for m in models)

    x = [opt.index_[i] for i in opt.window_ends_] if getattr(opt, "index_", None) is not None else list(range(n_windows))

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(d)]

    for j in range(d):
        plt.figure(figsize=(10, 5))
        for k in range(K_max):
            vals = [(m.means_[k, j] if k < m.n_components else np.nan) for m in models]
            plt.plot(x, vals, marker="o", label=f"Regime {k}")
        plt.title(title_prefix + f"Mean drift - {feature_names[j]}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        save_chart(f"{save_prefix}_mean_{feature_names[j]}.png")

    for j in range(d):
        plt.figure(figsize=(10, 5))
        for k in range(K_max):
            vals = [(m.covariances_[k][j, j] if k < m.n_components else np.nan) for m in models]
            plt.plot(x, vals, marker="o", label=f"Regime {k}")
        plt.title(title_prefix + f"Variance drift - {feature_names[j]}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        save_chart(f"{save_prefix}_var_{feature_names[j]}.png")

    plt.figure(figsize=(10, 5))
    for k in range(K_max):
        vals = [(m.weights_[k] if k < m.n_components else np.nan) for m in models]
        plt.plot(x, vals, marker="o", label=f"Regime {k}")
    plt.title(title_prefix + "Regime weights")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_chart(f"{save_prefix}_weights.png")


def plot_scaling_drift(opt, feature_names=None, save_prefix="scaling_drift"):
    """Plot expanding mean/std that Optimiser stored during scaling."""
    if len(getattr(opt, "raw_means_", [])) == 0:
        raise RuntimeError("raw_means_ empty - enable scaling stats recording.")

    means = np.vstack(opt.raw_means_)
    stds = np.vstack(opt.raw_stds_)
    n_windows, d = means.shape

    feature_names = feature_names or [f"feature_{i}" for i in range(d)]
    x = [opt.index_[i] for i in opt.window_ends_] if getattr(opt, "index_", None) is not None else list(range(n_windows))

    fig, axes = plt.subplots(d, 1, figsize=(12, 4 * d), sharex=True)
    if d == 1:
        axes = [axes]
    for j in range(d):
        axes[j].plot(x, means[:, j], marker="o")
        axes[j].set_title(f"Expanding mean - {feature_names[j]}")
        axes[j].grid(alpha=0.3)
    plt.tight_layout()
    save_chart(f"{save_prefix}_means.png")

    fig, axes = plt.subplots(d, 1, figsize=(12, 4 * d), sharex=True)
    if d == 1:
        axes = [axes]
    for j in range(d):
        axes[j].plot(x, stds[:, j], marker="o")
        axes[j].set_title(f"Expanding std - {feature_names[j]}")
        axes[j].grid(alpha=0.3)
    plt.tight_layout()
    save_chart(f"{save_prefix}_stds.png")


def plot_regime_correlations(corr_mats, feature_names, window_index=0, save_as=None):
    """Plot correlation matrices for a specific window."""
    if save_as is None:
        save_as = f"regime_correlations_win{window_index}.png"
    corrs = corr_mats[window_index]
    K = len(corrs)

    fig, axes = plt.subplots(1, K, figsize=(4 * K, 4))
    if K == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        im = ax.imshow(corrs[k], vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(f"Regime {k}")
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=90)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    save_chart(save_as)


def plot_window_regime_correlations(
    corr_list,
    feature_names,
    window_idx,
    title_prefix="Correlation matrices by regime",
    save_as=None,
):
    if save_as is None:
        save_as = f"window_regime_corr_win{window_idx}.png"
    window_corrs = corr_list[window_idx]
    K = len(window_corrs)

    fig, axes = plt.subplots(1, K, figsize=(4.5 * K + 0.8, 4.5))
    if K == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        corr = window_corrs[k]
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(f"{title_prefix}\nWindow {window_idx}, Regime {k}")
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=90)
        ax.set_yticks(range(len(feature_names)))
        if k == 0:
            ax.set_yticklabels(feature_names)
        else:
            ax.set_yticklabels([])

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.08)
    plt.subplots_adjust(right=0.88)
    save_chart(save_as)


def plot_corr_over_windows(
    corr_list,
    window_ends,
    i,
    j,
    regime_idx,
    feature_names,
    save_as=None,
):
    """Track correlation(i,j) for one regime over all windows."""
    if save_as is None:
        save_as = f"corr_over_windows_{feature_names[i]}_{feature_names[j]}_r{regime_idx}.png"

    vals = []
    xs = []

    for x, window_corrs in zip(window_ends, corr_list):
        if isinstance(window_corrs, dict):
            corr = window_corrs.get(regime_idx, None)
        else:
            corr = window_corrs[regime_idx] if regime_idx < len(window_corrs) else None

        vals.append(corr[i, j] if corr is not None else np.nan)
        xs.append(x)

    plt.figure(figsize=(10, 4))
    plt.plot(xs, vals, marker="o")
    plt.title(f"Correlation over windows: {feature_names[i]} vs {feature_names[j]} (Regime {regime_idx})")
    plt.xlabel("Window end")
    plt.ylabel("Correlation")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_chart(save_as)


def export_corr_long(opt, feature_names, path=None):
    """Export correlations to CSV."""
    if path is None:
        path = OUTPUT_DIR / "regime_correlations_long.csv"
    corr_list = opt.get_correlation_matrices()
    rows = []

    for w, window_corrs in enumerate(corr_list):
        end_date = opt.window_end_times_[w]
        for k, C in enumerate(window_corrs):
            d = C.shape[0]
            for i in range(d):
                for j in range(d):
                    rows.append(
                        {
                            "window_idx": w,
                            "end_date": end_date,
                            "regime": k,
                            "asset_i": feature_names[i],
                            "asset_j": feature_names[j],
                            "correlation": C[i, j],
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Saved long-form correlations to {path}")
    return df


def plot_regime_means_by_asset(summary_df: pd.DataFrame, title="Regime means by asset", save_as="regime_means_by_asset.png"):
    assets = summary_df["asset"].unique().tolist()
    regimes = sorted(summary_df["regime"].unique().tolist())

    plt.figure(figsize=(10, 4))
    for k in regimes:
        sub = summary_df[summary_df["regime"] == k].set_index("asset").reindex(assets)
        plt.plot(assets, sub["mean_wavg"].values, marker="o", label=f"Regime {k}")
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Mean (daily)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_chart(save_as)


def plot_regime_vols_by_asset(summary_df: pd.DataFrame, title="Regime volatility by asset", save_as="regime_vols_by_asset.png"):
    assets = summary_df["asset"].unique().tolist()
    regimes = sorted(summary_df["regime"].unique().tolist())

    plt.figure(figsize=(10, 4))
    for k in regimes:
        sub = summary_df[summary_df["regime"] == k].set_index("asset").reindex(assets)
        plt.plot(assets, sub["vol_wavg"].values, marker="o", label=f"Regime {k}")
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Volatility (daily)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_chart(save_as)


def plot_effective_sample_size(opt, title="Effective sample size per regime", save_as="effective_sample_size.png"):
    """Plot Neff_k = sum_t gamma_hat_{kt} per window."""
    Neff_list = getattr(opt, "Neff_by_window_", None)
    if not Neff_list:
        raise ValueError("No Neff_by_window_ found. Run opt.fit() after adding diagnostics.")

    n_windows = len(Neff_list)
    K_max = max(len(x) for x in Neff_list)

    Neff_mat = np.full((n_windows, K_max), np.nan, dtype=float)
    for w, neff in enumerate(Neff_list):
        neff = np.asarray(neff, dtype=float).ravel()
        Neff_mat[w, : len(neff)] = neff

    if hasattr(opt, "window_end_times_") and len(opt.window_end_times_) == n_windows:
        x = opt.window_end_times_
    elif hasattr(opt, "window_end_indices_") and len(opt.window_end_indices_) == n_windows:
        x = opt.window_end_indices_
    elif hasattr(opt, "window_ends_") and len(opt.window_ends_) == n_windows:
        x = [opt.index_[i] for i in opt.window_ends_] if getattr(opt, "index_", None) is not None else opt.window_ends_
    else:
        x = list(range(n_windows))

    plt.figure(figsize=(12, 5))
    for k in range(K_max):
        plt.plot(x, Neff_mat[:, k], marker="o", label=f"Regime {k}")

    min_neff = np.nanmin(Neff_mat, axis=1)
    plt.plot(x, min_neff, linestyle="--", linewidth=2, label="min across regimes")

    plt.title(title)
    plt.xlabel("Window end")
    plt.ylabel(r"$\tilde{N}_k=\sum_t \hat{\gamma}_{kt}$")
    plt.grid(alpha=0.3)
    plt.legend(ncols=min(K_max + 1, 4))
    plt.tight_layout()
    save_chart(save_as)


def plot_min_effective_sample_size(opt, title="Minimum effective sample size per window", save_as="min_effective_sample_size.png"):
    """Plot min_k Neff_k per window."""
    Neff_list = getattr(opt, "Neff_by_window_", None)
    if not Neff_list:
        raise ValueError("No Neff_by_window_ found. Run opt.fit() after adding diagnostics.")

    mins = [float(np.min(np.asarray(neff))) for neff in Neff_list]
    n_windows = len(mins)

    if hasattr(opt, "window_end_times_") and len(opt.window_end_times_) == n_windows:
        x = opt.window_end_times_
    elif hasattr(opt, "window_end_indices_") and len(opt.window_end_indices_) == n_windows:
        x = opt.window_end_indices_
    else:
        x = list(range(n_windows))

    plt.figure(figsize=(12, 4))
    plt.plot(x, mins, marker="o")
    plt.title(title)
    plt.xlabel("Window end")
    plt.ylabel(r"$\min_k \tilde{N}_k$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_chart(save_as)
