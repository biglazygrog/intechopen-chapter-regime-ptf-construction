"""
Figure A.1 — Observations per Year After the Synchronous-Trading Filter.

Appendix chart documenting the estimation grid. Under the corrected filter
(applied to the five market indices only, with cash exempt — see
core.utils.filter_synchronous_trading) the grid is near-regular: every full year
retains 249-253 observations, and the residual ~3-5% shortfall against 252 is
genuine non-synchronous market holidays across the five markets.

For context on why this matters, the superseded 6-asset filter — which tested
the cash series too — left 13 observations in 2021 and 26 in 2013, because a
1-12 month short-Treasury index returns zero at the stored price precision
whenever the policy rate is pinned. See REVISION_LOG.md, Open questions 8 and 9.

Run:  python -m research.figures.figureA1_observations_per_year
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.reader1 import DataReader
from core.utils import filter_synchronous_trading
from core.config import FIGURES_DIR, ANN_FACTOR

# Sample starts 2000-10-31 and ends 2026-01-23, so the end years are partial.
PARTIAL_YEARS = {2000, 2026}


def main():
    post = filter_synchronous_trading(DataReader().read_retns().dropna())
    counts = post.groupby(post.index.year).size()
    years = list(range(int(counts.index.min()), int(counts.index.max()) + 1))
    vals = [int(counts.get(y, 0)) for y in years]

    print(f"n = {len(post):,}   {post.index[0].date()} -> {post.index[-1].date()}")
    full = [v for y, v in zip(years, vals) if y not in PARTIAL_YEARS]
    print(f"full years: min {min(full)}  max {max(full)}  mean {np.mean(full):.1f}")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    bars = ax.bar(years, vals, color="steelblue", edgecolor="white",
                  linewidth=0.8, zorder=3)
    for b, y in zip(bars, years):        # partial years hatched, not recoloured
        if y in PARTIAL_YEARS:
            b.set_hatch("//")
            b.set_alpha(0.55)

    ax.axhline(ANN_FACTOR, color="#c0392b", linestyle="--", linewidth=1.2, zorder=4)
    ax.text(0.5, 0.93, f"mean = {ANN_FACTOR:.1f} obs/yr", color="#c0392b",
            transform=ax.transAxes, va="bottom", ha="center", fontsize=8, zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    # Selective direct labels only — the extremes of the full years.
    lo_y = min((y for y in years if y not in PARTIAL_YEARS), key=lambda y: vals[years.index(y)])
    hi_y = max((y for y in years if y not in PARTIAL_YEARS), key=lambda y: vals[years.index(y)])
    for b, y, v in zip(bars, years, vals):
        if y in (lo_y, hi_y):
            ax.text(b.get_x() + b.get_width() / 2, v + 5, str(v),
                    ha="center", va="bottom", fontsize=8, color="#333333")

    ax.set_ylabel("Observations")
    ax.set_xlabel("Year")
    ax.set_title("Observations per Year After the Synchronous-Trading Filter "
                 f"(n = {len(post):,})", fontsize=11)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=8)
    ax.set_xlim(years[0] - 0.7, years[-1] + 0.7)
    ax.set_ylim(0, 285)
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.text(0.5, -0.06,
             "Hatched bars (2000, 2026) are partial years. Dashed line is the mean "
             f"of {ANN_FACTOR:.1f} observations/year used as the annualisation "
             "factor. All window and horizon\nreferences in the paper are "
             "observation counts, not calendar days.",
             ha="center", fontsize=8, style="italic")
    plt.tight_layout()

    for ext in ("png", "pdf"):
        dst = FIGURES_DIR / f"figureA1_observations_per_year.{ext}"
        fig.savefig(dst, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: {dst}")
    plt.close()


if __name__ == "__main__":
    main()
