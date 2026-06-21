"""
Figure 1 — Posterior Regime Probabilities, Rolling GMM Estimation (2000-2024).

Reads daily_regime_probabilities.csv produced by pipeline.py and emits a
stacked-area chart in trace order (p_0 = calm, p_{K-1} = crisis).

Run:  python -m research.figures.figure1_regime_probabilities
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from core.config import OUTPUT_DIR, FIGURES_DIR


PALETTE = {
    2: ["#3a7d44", "#c0392b"],
    3: ["#3a7d44", "#e0a82e", "#c0392b"],
    4: ["#3a7d44", "#e0a82e", "#d8642c", "#c0392b"],
}


def main():
    probs = pd.read_csv(OUTPUT_DIR / "daily_regime_probabilities.csv",
                        index_col=0, parse_dates=True)
    prob_cols = sorted(
        [c for c in probs.columns if c.startswith("p_")],
        key=lambda c: int(c.split("_")[1]),
    )
    P = probs[prob_cols].div(probs[prob_cols].sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    K = P.shape[1]
    colors = PALETTE.get(K)

    print(f"Date range: {P.index.min().date()} → {P.index.max().date()}")
    print(f"K={K}, avg weights: {P.mean().round(4).to_dict()}")

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.stackplot(P.index, P.values.T, labels=prob_cols, colors=colors, alpha=0.88)
    ax.set_ylim(0, 1)
    ax.set_xlim(P.index.min(), P.index.max())
    ax.set_ylabel("Posterior probability")
    ax.set_xlabel("Date")
    ax.set_title("Posterior Regime Probabilities, Rolling GMM Estimation (2000–2024)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", ncols=K, frameon=False)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        dst = FIGURES_DIR / f"figure1_regime_probabilities.{ext}"
        plt.savefig(dst, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: {dst}")
    plt.close()


if __name__ == "__main__":
    main()
