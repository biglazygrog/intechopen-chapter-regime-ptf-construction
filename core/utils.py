"""
Shared utility functions used across analysis and figure scripts.
"""
import numpy as np
import pandas as pd
from typing import Iterable, Optional, Tuple


# ---------------------------------------------------------------------
# Synchronous-trading filter
# ---------------------------------------------------------------------

# Assets EXEMPT from the zero-return filter.
#
# `high_yield` is LD12TRUU Index — a Bloomberg US Short Treasury 1-12 Month
# index, displayed as "Cash" in ASSET_DISPLAY. Its zero-return stretches track
# zero-interest-rate policy, not non-trading: at a pinned policy rate the true
# daily accrual falls below the stored price precision (~2.6bp on a level of 192
# held to 2 d.p.), so the recorded zero is a rounding artefact of a genuinely
# near-zero return. Between 2009 and 2016 the level sat in [191.19, 192.23] with
# only 105 distinct values; 2021 had 10 distinct values in the whole year.
#
# Filtering on it was responsible for 96.8% of all dropped rows and removed most
# of 2009-2016 and 2020-2021 — including 85.2% of the EU sovereign debt crisis
# and 55.1% of the COVID drawdown. See REVISION_LOG.md, Open questions 8 and 9.
CASH_LIKE_ASSETS: Tuple[str, ...] = ("high_yield",)


def filter_synchronous_trading(
    df: pd.DataFrame,
    exempt: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Keep only rows on which every *market index* asset actually traded.

    ``DataReader`` forward-fills prices onto a daily grid, so an index that did
    not update yields a log return of exactly zero. Such a row is not a real
    synchronous cross-section: the joint return vector carries a spurious zero,
    which would corrupt the cross-asset covariance structure the GMM relies on
    to identify regimes.

    Cash-like series are exempt (see :data:`CASH_LIKE_ASSETS`) — for those a
    zero is an economically genuine near-zero return, not a stale quote.

    This is the SINGLE definition of the filter. It previously existed as five
    independent copies (pipeline, robustness, shrinkage, stability_analyzer,
    validate_onesided) which could silently drift apart.

    Parameters
    ----------
    df : DataFrame
        Log returns, one column per asset. NaNs should already be dropped.
    exempt : iterable of str, optional
        Column names exempt from the zero test. Defaults to
        :data:`CASH_LIKE_ASSETS`. Names not present in ``df`` are ignored, so
        this is safe across the core and extended universes.

    Returns
    -------
    DataFrame with non-synchronous and non-finite rows removed.
    """
    exempt = set(CASH_LIKE_ASSETS if exempt is None else exempt)
    tested = [c for c in df.columns if c not in exempt]
    if not tested:
        raise ValueError("No columns left to test — `exempt` covers every column.")

    out = df[(df[tested] != 0).all(axis=1)]
    return out[np.isfinite(out).all(axis=1)]


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
