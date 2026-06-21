"""
Table 4 — Cross-Asset Correlation Analysis Across Regimes (Selected Pairs).

Reads correlation_pairwise_bootstrap.csv from pipeline.py and emits the
dissertation-formatted table sorted by significance.

Run:  python -m research.analysis.correlations
"""
from pathlib import Path

import pandas as pd

from core.config import CORRELATION_DISPLAY, TABLES_DIR
from core.utils import sig_stars, format_corr

RAW_PATH = Path(__file__).resolve().parent.parent.parent / "research" / "output_charts" / "correlation_pairwise_bootstrap.csv"


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def build_table(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["Asset i"] = raw["asset_i"].map(CORRELATION_DISPLAY)
    raw["Asset j"] = raw["asset_j"].map(CORRELATION_DISPLAY)

    table = pd.DataFrame({
        "Asset i":         raw["Asset i"],
        "Asset j":         raw["Asset j"],
        "Regime (low)":    raw["regime_low"].astype(int),
        "Regime (high)":   raw["regime_high"].astype(int),
        "ρ (low)":         raw["rho_low"].round(3),
        "ρ (high)":        raw["rho_high"].round(3),
        "Δρ (high - low)": raw["delta_rho_high_minus_low"].round(3),
        "CI lower":        raw["ci_low"].round(3),
        "CI upper":        raw["ci_high"].round(3),
        "p-value":         raw["p_value_boot"].map(_fmt_p),
        "Sig.":            raw["p_value_boot"].map(sig_stars),
        "n (low)":         raw["n_low"].astype(int),
        "n (high)":        raw["n_high"].astype(int),
    })

    # Sort by raw p-value ascending (most significant first)
    order = raw["p_value_boot"].argsort(kind="stable")
    return table.iloc[order].reset_index(drop=True)


def main():
    raw = pd.read_csv(RAW_PATH)
    table = build_table(raw)
    table.to_csv(TABLES_DIR / "table4_correlations.csv", index=False)

    fmt = table.copy()
    for c in ["ρ (low)", "ρ (high)", "Δρ (high - low)", "CI lower", "CI upper"]:
        fmt[c] = fmt[c].map(format_corr)
    fmt.to_csv(TABLES_DIR / "table4_correlations.tsv", sep="\t", index=False)

    print("Table 4 — Cross-Asset Correlation Analysis Across Regimes")
    print(fmt.to_string(index=False))


if __name__ == "__main__":
    main()
