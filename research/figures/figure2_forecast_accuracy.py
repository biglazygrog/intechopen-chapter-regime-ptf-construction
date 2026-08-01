"""
Figure 2 — Forecast Accuracy by Horizon (3-panel: Brier Skill / AUC-ROC / High-Vol Detection).

Reads ar_change_multihorizon_summary.csv from research.analysis.forecast_eval,
which is computed on the ONE-SIDED K=2 regime-probability series.

Because the one-sided series starts at observation index WINDOW_SIZE
(2006-01-30), the evaluation sample is shorter than under the retrospective
series — the figure annotates the realised n_forecasts.

Run:  python -m research.figures.figure2_forecast_accuracy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.config import OUTPUT_DIR, FIGURES_DIR, AR_HORIZONS


def main():
    ar_change = pd.read_csv(OUTPUT_DIR / "ar_change_multihorizon_summary.csv", index_col=0)
    horizons = AR_HORIZONS
    x = np.arange(len(horizons))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) Brier Skill Score
    ax1 = axes[0]
    skill_change = [ar_change.loc[h, "brier_skill_change_vs_rw"] * 100 for h in horizons]
    skill_base = [ar_change.loc[h, "brier_skill_baseline_vs_rw"] * 100 for h in horizons]
    bars1 = ax1.bar(x - width/2, skill_change, width, label="AR Change", color="steelblue")
    bars2 = ax1.bar(x + width/2, skill_base, width, label="AR Baseline", color="orange")
    ax1.set_xticks(x); ax1.set_xticklabels([f"{h}-day" for h in horizons])
    ax1.set_ylabel("Brier Skill Score (%)")
    ax1.set_title("(a) Brier Skill Score vs Random Walk", fontweight="bold")
    ax1.legend(loc="upper right"); ax1.set_ylim(0, 30)
    ax1.axhline(0, color="red", linestyle="--", linewidth=0.5)
    for bar, val in zip(bars1, skill_change):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, skill_base):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=8)

    # (b) AUC-ROC
    ax2 = axes[1]
    auc_change = [ar_change.loc[h, "auc_change"] for h in horizons]
    auc_base = [ar_change.loc[h, "auc_baseline"] for h in horizons]
    auc_rw = [ar_change.loc[h, "auc_rw"] for h in horizons]
    ax2.bar(x - width, auc_change, width, label="AR Change", color="steelblue")
    ax2.bar(x, auc_base, width, label="AR Baseline", color="orange")
    ax2.bar(x + width, auc_rw, width, label="Random Walk", color="gray")
    ax2.set_xticks(x); ax2.set_xticklabels([f"{h}-day" for h in horizons])
    ax2.set_ylabel("AUC-ROC")
    ax2.set_title("(b) AUC-ROC by Horizon", fontweight="bold")
    ax2.legend(loc="upper right"); ax2.set_ylim(0.5, 0.9)
    ax2.axhline(0.5, color="red", linestyle="--", linewidth=0.5)

    # (c) High-Vol Detection
    ax3 = axes[2]
    hv_change = [ar_change.loc[h, "highvol_change"] * 100 for h in horizons]
    hv_base = [ar_change.loc[h, "highvol_baseline"] * 100 for h in horizons]
    hv_rw = [ar_change.loc[h, "highvol_rw"] * 100 for h in horizons]
    ax3.bar(x - width, hv_change, width, label="AR Change", color="steelblue")
    ax3.bar(x, hv_base, width, label="AR Baseline", color="orange")
    ax3.bar(x + width, hv_rw, width, label="Random Walk", color="gray")
    ax3.set_xticks(x); ax3.set_xticklabels([f"{h}-day" for h in horizons])
    ax3.set_ylabel("High-Vol Detection Rate (%)")
    ax3.set_title(r"(c) High-Volatility Detection ($\theta$ = 0.5)", fontweight="bold")
    ax3.legend(loc="upper right"); ax3.set_ylim(0, 50)

    # Panel (c) is a raw detection rate; annotate the trivial-classifier
    # baseline so it cannot be read as skill on its own.
    if "hit_rate_always_calm" in ar_change.columns:
        calm = [ar_change.loc[h, "hit_rate_always_calm"] * 100 for h in horizons]
        gains = [ar_change.loc[h, "hit_gain_change_vs_always_calm"] * 100 for h in horizons]
        calm_note = ("  Always-calm hit-rate baseline: "
                     + ", ".join(f"{h}d {c:.1f}% (AR Change {g:+.1f}pp)"
                                 for h, c, g in zip(horizons, calm, gains)) + ".")
    else:
        calm_note = ""

    n_fc = int(ar_change.loc[horizons[0], "n_forecasts"])
    fig.text(0.5, -0.04,
             f"Note: computed on the one-sided K=2 regime-probability series "
             f"(no look-ahead); n_forecasts = {n_fc} per horizon.{calm_note}",
             ha="center", fontsize=8, style="italic")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        dst = FIGURES_DIR / f"figure2_forecast_accuracy.{ext}"
        fig.savefig(dst, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: {dst}")
    plt.close()


if __name__ == "__main__":
    main()
