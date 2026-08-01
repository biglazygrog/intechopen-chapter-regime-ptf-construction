"""
Table 3 — Regime-Conditional Moments and Downside Risk.

Reads paper_profiles_raw.csv produced by pipeline.py and emits the
dissertation-formatted table (annualised mean & vol, per-day VaR/ES/quantiles).

Two things to note when reading this table:

1. Regime labels are RETROSPECTIVE (full-sample smoothed). This is deliberate —
   the table is a descriptive characterisation of the estimated regimes, not an
   investable signal. See REVISION_LOG.md.
2. Annualisation uses ANN_FACTOR = 248.48 observations/year, derived from the
   actual filtered grid (6269 observations / 25.229 calendar years), not the
   252 business days previously assumed. Annualised means and volatilities
   therefore differ from earlier versions of this table.

Run:  python -m research.analysis.moments
"""
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import ASSET_DISPLAY, TABLES_DIR, ANN_FACTOR

DISPLAY_ORDER = ["Equities", "Bonds", "IG Credit", "Commodities", "Cash", "Gold"]
RAW_PATH = Path(__file__).resolve().parent.parent.parent / "research" / "output_charts" / "paper_profiles_raw.csv"


def build_table(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["asset_display"] = raw["asset"].map(ASSET_DISPLAY)
    raw = raw.set_index("asset_display").loc[DISPLAY_ORDER].reset_index()

    sqrt_ann = np.sqrt(ANN_FACTOR)
    rows = []
    for _, r in raw.iterrows():
        for reg in ("R0", "R1"):
            rows.append({
                "Asset": r["asset_display"] if reg == "R0" else "",
                "Regime": reg,
                "N": int(r[f"{reg}_n"]),
                "Mean (%/yr)":       r[f"{reg}_mean"] * ANN_FACTOR * 100,
                "Volatility (%/yr)": r[f"{reg}_vol"]  * sqrt_ann   * 100,
                "VaR(5%) (%)":       r[f"{reg}_VaR5"] * 100,
                "ES(5%) (%)":        r[f"{reg}_ES5"]  * 100,
                "Q5 (%)":            r[f"{reg}_Q5"]   * 100,
                "Q10 (%)":           r[f"{reg}_Q10"]  * 100,
                "Q25 (%)":           r[f"{reg}_Q25"]  * 100,
            })
    return pd.DataFrame(rows)


def main():
    raw = pd.read_csv(RAW_PATH)
    table = build_table(raw)

    # Full CSV (forward-fill asset col for machine consumption)
    full = table.copy()
    full.loc[full["Asset"] == "", "Asset"] = pd.NA
    full["Asset"] = full["Asset"].ffill()
    full.to_csv(TABLES_DIR / "table3_regime_moments.csv", index=False, float_format="%.4f")

    # Display TSV (blank Asset cell on R1 rows for dissertation layout)
    fmt = table.copy()
    for col in ["Mean (%/yr)", "Volatility (%/yr)",
                "VaR(5%) (%)", "ES(5%) (%)", "Q5 (%)", "Q10 (%)", "Q25 (%)"]:
        fmt[col] = fmt[col].map(lambda x: f"{x:.2f}")
    fmt.to_csv(TABLES_DIR / "table3_regime_moments.tsv", sep="\t", index=False)

    print("Table 3 — Regime-Conditional Moments and Downside Risk")
    print(f"Annualisation: ANN_FACTOR = {ANN_FACTOR} observations/year "
          f"(near-uniform grid). Regime labels are retrospective/full-sample.")
    print(fmt.to_string(index=False))


if __name__ == "__main__":
    main()
