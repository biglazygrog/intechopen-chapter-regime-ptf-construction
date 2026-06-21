"""
Figure 3 — Rolling 252-day Hit Rate: AR Model vs Random Walk.

Reads dominant_regime_forecast_eval.csv from research.analysis.pipeline.

Run:  python -m research.figures.figure3_rolling_hit_rate
"""
import pandas as pd
import matplotlib.pyplot as plt

from core.config import OUTPUT_DIR, FIGURES_DIR


def main():
    details = pd.read_csv(OUTPUT_DIR / "dominant_regime_forecast_eval.csv",
                          parse_dates=["date_t", "date_t1"])
    details = details.sort_values("date_t1").set_index("date_t1")

    details["hit_ar"] = ((details["p_ar_t1"] >= 0.5) == (details["z_t1"] == 1)).astype(int)
    details["hit_rw"] = ((details["p_rw_t1"] >= 0.5) == (details["z_t1"] == 1)).astype(int)

    window = 252
    rolling_ar = details["hit_ar"].rolling(window).mean()
    rolling_rw = details["hit_rw"].rolling(window).mean()

    print(f"Overall hit rate — AR: {details['hit_ar'].mean():.1%}, "
          f"RW: {details['hit_rw'].mean():.1%}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling_ar.index, rolling_ar * 100, label="AR Model", color="#1f77b4", linewidth=1.5)
    ax.plot(rolling_rw.index, rolling_rw * 100, label="Random Walk", color="gray", linewidth=1.5)
    ax.axhline(50, color="red", linestyle="--", linewidth=1, label="Random (50%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Hit Rate (%)")
    ax.set_title(f"Rolling {window}-Day Hit Rate: AR Model vs Random Walk")
    ax.legend(loc="lower right")
    ax.set_ylim(40, 100)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        dst = FIGURES_DIR / f"figure3_rolling_hit_rate.{ext}"
        plt.savefig(dst, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: {dst}")
    plt.close()


if __name__ == "__main__":
    main()
