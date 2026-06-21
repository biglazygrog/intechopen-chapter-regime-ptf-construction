"""
Shared utility functions used across analysis and figure scripts.
"""
import numpy as np
import pandas as pd
from typing import Tuple


def hard_labels_from_daily_probs(probs_df: pd.DataFrame) -> pd.Series:
    """Argmax over p_0/p_1/... columns -> integer regime label per day."""
    cols = sorted(
        [c for c in probs_df.columns if c.startswith("p_")],
        key=lambda c: int(c.split("_")[1]),
    )
    P = probs_df[cols].values
    labels = np.argmax(P, axis=1)
    return pd.Series(labels, index=probs_df.index, name="regime")


def logit(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def inv_logit(y: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-y))


def regime_split_n(probs_df: pd.DataFrame, target_regime: str = "p_1",
                   threshold: float = 0.5) -> Tuple[int, int]:
    """Return (n_low, n_high) under hard-thresholding the target regime."""
    z = (probs_df[target_regime] > threshold).astype(int)
    return int((z == 0).sum()), int((z == 1).sum())


def format_pct(x, decimals: int = 1) -> str:
    """Format a fraction as a percentage string ('-' for NaN)."""
    if pd.isna(x):
        return "-"
    return f"{x * 100:.{decimals}f}%"


def format_corr(x, decimals: int = 3) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:.{decimals}f}"


def sig_stars(p_value: float) -> str:
    """Three-tier significance stars used in the paper correlation table."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.05:
        return "**"
    return "-"
