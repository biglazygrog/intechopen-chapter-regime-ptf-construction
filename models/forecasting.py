"""
Regime probability forecasting.

Contains:
- RegimeProbForecaster: ARIMA-based forecasting of regime probabilities
- Snail trail forecasting functions
- Forecast evaluation and comparison
- Forecast visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict
from statsmodels.tsa.arima.model import ARIMA

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_chart(filename: str, dpi: int = 150):
    """Save current figure to output directory and close it."""
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


# -------------------------------------------------------------------
# AR Order Selection and Statistical Tests
# -------------------------------------------------------------------

def select_ar_order_aic(y: np.ndarray, max_lag: int = 5, criterion: str = "aic") -> int:
    """
    Select optimal AR order using information criterion.

    Parameters
    ----------
    y : array
        Time series (logit-transformed regime probabilities)
    max_lag : int
        Maximum lag to consider (default: 5, reduced from 10 to prevent overfitting)
    criterion : str
        "aic" (Akaike) or "bic" (Bayesian, more parsimonious)

    Returns
    -------
    int : optimal AR order (p)
    """
    from statsmodels.tsa.ar_model import AutoReg

    n = len(y)
    if n < max_lag + 10:
        max_lag = max(1, n - 10)

    best_ic = np.inf
    best_order = 1

    for p in range(1, max_lag + 1):
        try:
            model = AutoReg(y, lags=p, old_names=False).fit()
            if criterion.lower() == "bic":
                ic = model.bic
            else:
                ic = model.aic
            if ic < best_ic:
                best_ic = ic
                best_order = p
        except Exception:
            continue

    return best_order


def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    loss: str = "squared"
) -> Dict[str, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    Tests H0: E[L(e1)] = E[L(e2)] (equal expected loss)

    Parameters
    ----------
    e1 : array
        Forecast errors from model 1 (e.g., AR model)
    e2 : array
        Forecast errors from model 2 (e.g., random walk)
    h : int
        Forecast horizon (for HAC variance adjustment)
    loss : str
        Loss function: "squared" (MSE) or "absolute" (MAE)

    Returns
    -------
    dict with:
        - dm_stat: Diebold-Mariano test statistic
        - p_value: two-sided p-value
        - mean_loss_diff: mean difference in losses (negative = model 1 better)

    References
    ----------
    Diebold, F.X. and Mariano, R.S. (1995). "Comparing Predictive Accuracy"
    Journal of Business & Economic Statistics, 13(3), 253-263.
    """
    from scipy import stats

    e1 = np.asarray(e1).flatten()
    e2 = np.asarray(e2).flatten()

    if len(e1) != len(e2):
        raise ValueError("Error arrays must have same length")

    T = len(e1)

    # Compute loss differential
    if loss == "squared":
        d = e1**2 - e2**2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    d_mean = d.mean()

    # HAC variance estimator (Newey-West style for h-step ahead)
    # For h-step ahead forecasts, errors are MA(h-1), so we need h-1 autocovariances
    gamma_0 = np.var(d, ddof=1)

    gamma_sum = 0.0
    for k in range(1, h):
        if k < T:
            gamma_k = np.cov(d[:-k], d[k:], ddof=1)[0, 1]
            gamma_sum += 2 * gamma_k

    # Long-run variance
    var_d = (gamma_0 + gamma_sum) / T

    # Handle numerical issues
    if var_d <= 0:
        var_d = gamma_0 / T

    # DM statistic (asymptotically N(0,1) under H0)
    dm_stat = d_mean / np.sqrt(var_d)

    # Two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "mean_loss_diff": float(d_mean),
        "model1_better": d_mean < 0,
        "n_forecasts": int(T),
        "horizon": int(h),
    }


def diebold_mariano_test_from_forecasts(
    forecasts_1: np.ndarray,
    forecasts_2: np.ndarray,
    realized: np.ndarray,
    h: int = 1,
    loss: str = "squared"
) -> Dict[str, float]:
    """
    Diebold-Mariano test from forecast and realized values.

    Parameters
    ----------
    forecasts_1 : array
        Forecasts from model 1
    forecasts_2 : array
        Forecasts from model 2
    realized : array
        Realized values
    h : int
        Forecast horizon
    loss : str
        Loss function: "squared" or "absolute"

    Returns
    -------
    dict : DM test results
    """
    e1 = np.asarray(forecasts_1) - np.asarray(realized)
    e2 = np.asarray(forecasts_2) - np.asarray(realized)
    return diebold_mariano_test(e1, e2, h=h, loss=loss)


class RegimeProbForecaster:
    """
    Forecast a single regime probability series p_k using logit-ARIMA.
    For K=2, you can forecast p_1 and derive p_0 = 1 - p_1.
    """

    def __init__(self, order=(1, 0, 0), target_regime: str = "p_1"):
        self.order = order
        self.target_regime = target_regime
        self.model_ = None
        self.regime_col_ = None

    def _logit(self, p):
        eps = 1e-8
        p = np.clip(p, eps, 1-eps)
        return np.log(p / (1 - p))

    def _inv_logit(self, y):
        return 1 / (1 + np.exp(-y))

    def fit(self, probs_df: pd.DataFrame):
        if self.target_regime not in probs_df.columns:
            raise ValueError(f"target_regime={self.target_regime} not found in probs_df columns.")

        self.regime_col_ = self.target_regime
        p_series = probs_df[self.regime_col_].astype(float)
        y = self._logit(p_series.values)

        self.model_ = ARIMA(y, order=self.order).fit()
        return self

    def forecast(self, steps=1, last_date=None, freq=None):
        """Returns DataFrame indexed by future dates if last_date+freq provided."""
        y_fore = self.model_.forecast(steps=steps)
        p_fore = self._inv_logit(np.asarray(y_fore))

        if last_date is not None and freq is not None:
            future_index = pd.date_range(last_date, periods=steps+1, freq=freq)[1:]
        else:
            future_index = pd.RangeIndex(steps)

        return pd.DataFrame({self.regime_col_: p_fore}, index=future_index)


def evaluate_ar1_vs_random_walk(
    daily_probs_df: pd.DataFrame,
    target_regime: str = "p_1",
    arima_order=(1, 0, 0),
    min_train: int = 250,
):
    """
    Walk-forward comparison: AR(1)-logit vs Random Walk on dominant regime prob.
    Returns:
      - summary dict
      - per-date dataframe with forecasts + scores
    """
    pcols = sorted([c for c in daily_probs_df.columns if c.startswith("p_")],
                   key=lambda c: int(c.split("_")[1]))
    P = daily_probs_df[pcols].copy()

    if target_regime not in P.columns:
        raise ValueError(f"target_regime={target_regime} not in daily_probs_df columns.")
    dom_col = target_regime

    hard = P.values.argmax(axis=1)
    dom_k = int(dom_col.split("_")[1])
    z = (hard == dom_k).astype(int)

    eps = 1e-8
    def logit(p):
        p = np.clip(p, eps, 1-eps)
        return np.log(p/(1-p))
    def inv_logit(y):
        return 1/(1+np.exp(-y))

    p = P[dom_col].values
    y = logit(p)

    rows = []
    idx = P.index

    for t in range(min_train, len(P)-1):
        y_train = y[:t+1]

        try:
            fit = ARIMA(y_train, order=arima_order).fit()
            yhat = float(fit.forecast(steps=1)[0])
            p_ar1 = float(inv_logit(yhat))
        except Exception:
            p_ar1 = float(p[t])

        p_rw = float(p[t])
        z_next = int(z[t+1])

        brier_ar1 = (z_next - p_ar1)**2
        brier_rw = (z_next - p_rw)**2

        p_ar1c = np.clip(p_ar1, eps, 1-eps)
        p_rwc = np.clip(p_rw, eps, 1-eps)

        log_ar1 = -(z_next*np.log(p_ar1c) + (1-z_next)*np.log(1-p_ar1c))
        log_rw = -(z_next*np.log(p_rwc) + (1-z_next)*np.log(1-p_rwc))

        rows.append({
            "date_t": idx[t],
            "date_t1": idx[t+1],
            "p_t": float(p[t]),
            "p_ar1_t1": p_ar1,
            "p_rw_t1": p_rw,
            "z_t1": z_next,
            "brier_ar1": brier_ar1,
            "brier_rw": brier_rw,
            "logloss_ar1": log_ar1,
            "logloss_rw": log_rw,
        })

    res = pd.DataFrame(rows)

    summary = {
        "dominant_regime_col": dom_col,
        "dominant_regime_id": dom_k,
        "n_forecasts": int(len(res)),
        "brier_mean_ar1": float(res["brier_ar1"].mean()),
        "brier_mean_rw": float(res["brier_rw"].mean()),
        "logloss_mean_ar1": float(res["logloss_ar1"].mean()),
        "logloss_mean_rw": float(res["logloss_rw"].mean()),
        "brier_skill_vs_rw": float(1 - res["brier_ar1"].mean() / res["brier_rw"].mean()) if res["brier_rw"].mean() > 0 else np.nan,
        "logloss_diff_ar1_minus_rw": float(res["logloss_ar1"].mean() - res["logloss_rw"].mean()),
        "hit_rate_ar1": float(((res["p_ar1_t1"] >= 0.5) == (res["z_t1"] == 1)).mean()),
        "hit_rate_rw": float(((res["p_rw_t1"] >= 0.5) == (res["z_t1"] == 1)).mean()),
    }
    return summary, res


def evaluate_ar_vs_random_walk(
    daily_probs_df: pd.DataFrame,
    target_regime: str = "p_1",
    max_ar_lag: int = 5,
    min_train: int = 250,
    refit_frequency: int = 21,
    criterion: str = "bic",
) -> Dict:
    """
    Walk-forward evaluation: AR(p) with information criterion selection vs Random Walk.

    Includes Diebold-Mariano test for statistical significance.

    Parameters
    ----------
    daily_probs_df : DataFrame
        Daily regime probabilities with columns p_0, p_1, ...
    target_regime : str
        Column to forecast (default: "p_1")
    max_ar_lag : int
        Maximum AR lag for selection (default: 5, reduced to prevent overfitting)
    min_train : int
        Minimum training observations before first forecast (default: 250)
    refit_frequency : int
        Re-select AR order every N days (default: 21, ~monthly)
    criterion : str
        "aic" or "bic" for AR order selection (default: "bic" for parsimony)

    Returns
    -------
    dict with:
        - summary: dict of aggregate metrics + DM test results
        - details: DataFrame with per-day forecasts and errors
        - ar_orders: Series of selected AR orders over time
    """
    pcols = sorted([c for c in daily_probs_df.columns if c.startswith("p_")],
                   key=lambda c: int(c.split("_")[1]))
    P = daily_probs_df[pcols].copy()

    if target_regime not in P.columns:
        raise ValueError(f"target_regime={target_regime} not in columns.")
    dom_col = target_regime

    # Binary outcome: is target regime the argmax?
    hard = P.values.argmax(axis=1)
    dom_k = int(dom_col.split("_")[1])
    z = (hard == dom_k).astype(int)

    eps = 1e-8
    def logit(p):
        p = np.clip(p, eps, 1-eps)
        return np.log(p/(1-p))
    def inv_logit(y):
        return 1/(1+np.exp(-y))

    p = P[dom_col].values
    y = logit(p)

    rows = []
    idx = P.index
    current_ar_order = 1
    ar_order_history = []

    for t in range(min_train, len(P)-1):
        y_train = y[:t+1]

        # Re-select AR order periodically using information criterion
        if (t - min_train) % refit_frequency == 0:
            current_ar_order = select_ar_order_aic(y_train, max_lag=max_ar_lag, criterion=criterion)

        ar_order_history.append({"date": idx[t], "ar_order": current_ar_order})

        # AR(p) forecast
        try:
            fit = ARIMA(y_train, order=(current_ar_order, 0, 0)).fit()
            yhat = float(fit.forecast(steps=1)[0])
            p_ar = float(inv_logit(yhat))
        except Exception:
            p_ar = float(p[t])  # fallback to random walk

        # Random walk forecast
        p_rw = float(p[t])

        # Realized values
        p_realized = float(p[t+1])
        z_realized = int(z[t+1])

        # Forecast errors (for probability)
        e_ar = p_ar - p_realized
        e_rw = p_rw - p_realized

        # Brier scores (for binary outcome)
        brier_ar = (z_realized - p_ar)**2
        brier_rw = (z_realized - p_rw)**2

        # Log loss
        p_ar_c = np.clip(p_ar, eps, 1-eps)
        p_rw_c = np.clip(p_rw, eps, 1-eps)
        logloss_ar = -(z_realized*np.log(p_ar_c) + (1-z_realized)*np.log(1-p_ar_c))
        logloss_rw = -(z_realized*np.log(p_rw_c) + (1-z_realized)*np.log(1-p_rw_c))

        rows.append({
            "date_t": idx[t],
            "date_t1": idx[t+1],
            "ar_order": current_ar_order,
            "p_t": float(p[t]),
            "p_ar_t1": p_ar,
            "p_rw_t1": p_rw,
            "p_realized_t1": p_realized,
            "z_t1": z_realized,
            "error_ar": e_ar,
            "error_rw": e_rw,
            "brier_ar": brier_ar,
            "brier_rw": brier_rw,
            "logloss_ar": logloss_ar,
            "logloss_rw": logloss_rw,
        })

    details = pd.DataFrame(rows)
    ar_orders = pd.DataFrame(ar_order_history)

    # Aggregate metrics
    n = len(details)
    brier_mean_ar = float(details["brier_ar"].mean())
    brier_mean_rw = float(details["brier_rw"].mean())
    logloss_mean_ar = float(details["logloss_ar"].mean())
    logloss_mean_rw = float(details["logloss_rw"].mean())

    # Diebold-Mariano tests
    dm_brier = diebold_mariano_test(
        details["error_ar"].values,
        details["error_rw"].values,
        h=1,
        loss="squared"
    )

    dm_absolute = diebold_mariano_test(
        details["error_ar"].values,
        details["error_rw"].values,
        h=1,
        loss="absolute"
    )

    # Modal AR order selected
    modal_ar_order = int(details["ar_order"].mode().iloc[0])
    mean_ar_order = float(details["ar_order"].mean())

    summary = {
        "target_regime": dom_col,
        "n_forecasts": n,
        # AR order selection
        "selection_criterion": criterion.upper(),
        "modal_ar_order": modal_ar_order,
        "mean_ar_order": mean_ar_order,
        "max_ar_lag_considered": max_ar_lag,
        # Brier scores
        "brier_mean_ar": brier_mean_ar,
        "brier_mean_rw": brier_mean_rw,
        "brier_skill_vs_rw": float(1 - brier_mean_ar / brier_mean_rw) if brier_mean_rw > 0 else np.nan,
        # Log loss
        "logloss_mean_ar": logloss_mean_ar,
        "logloss_mean_rw": logloss_mean_rw,
        "logloss_diff_ar_minus_rw": logloss_mean_ar - logloss_mean_rw,
        # Hit rates
        "hit_rate_ar": float(((details["p_ar_t1"] >= 0.5) == (details["z_t1"] == 1)).mean()),
        "hit_rate_rw": float(((details["p_rw_t1"] >= 0.5) == (details["z_t1"] == 1)).mean()),
        # Diebold-Mariano test (squared loss)
        "dm_stat_squared": dm_brier["dm_stat"],
        "dm_pvalue_squared": dm_brier["p_value"],
        "dm_ar_better_squared": dm_brier["model1_better"],
        # Diebold-Mariano test (absolute loss)
        "dm_stat_absolute": dm_absolute["dm_stat"],
        "dm_pvalue_absolute": dm_absolute["p_value"],
        "dm_ar_better_absolute": dm_absolute["model1_better"],
    }

    return {
        "summary": summary,
        "details": details,
        "ar_orders": ar_orders,
    }


def generate_snail_trail_forecasts(
    daily_probs_df: pd.DataFrame,
    forecast_main_regime: bool = True,
    arima_order=(1, 0, 0),
    min_train: int = 250,
    horizon: int = 21,
    forecast_step: int = 20,
):
    """
    Generates rolling multi-step forecasts (snail trail).
    """
    pcols = sorted([c for c in daily_probs_df.columns if c.startswith("p_")],
                   key=lambda c: int(c.split("_")[1]))
    P = daily_probs_df[pcols].copy()

    if forecast_main_regime:
        dom_col = P.mean().idxmax()
    else:
        dom_col = "p_1" if "p_1" in P.columns else P.columns[0]

    eps = 1e-8
    def logit(p):
        p = np.clip(p, eps, 1-eps)
        return np.log(p/(1-p))

    def inv_logit(y):
        return 1/(1+np.exp(-y))

    p = P[dom_col].values
    y = logit(p)

    idx = P.index
    rows = []

    for t in range(min_train, len(P)-horizon, forecast_step):
        y_train = y[:t+1]

        try:
            fit = ARIMA(y_train, order=arima_order).fit()
            yhat = fit.forecast(steps=horizon)
            phat = inv_logit(yhat)
        except Exception:
            phat = np.repeat(p[t], horizon)

        origin_date = idx[t]

        for h in range(horizon):
            rows.append({
                "origin_date": origin_date,
                "step": h+1,
                "forecast_date": idx[t+h+1],
                "forecast_prob": float(phat[h])
            })

    return pd.DataFrame(rows)


def generate_regime_aware_forecasts(
    daily_probs_df: pd.DataFrame,
    forecast_main_regime: bool = True,
    arima_order_normal: tuple = (1, 0, 0),
    arima_order_stress: tuple = (1, 0, 0),
    min_train: int = 250,
    horizon: int = 21,
    forecast_step: int = 20,
    regime_threshold: float = 0.5,
    blend_forecasts: bool = True,
):
    """Regime-aware AR forecasting: use different AR models for different regimes."""
    pcols = sorted([c for c in daily_probs_df.columns if c.startswith("p_")],
                   key=lambda c: int(c.split("_")[1]))
    P = daily_probs_df[pcols].copy()

    if forecast_main_regime:
        dom_col = P.mean().idxmax()
    else:
        dom_col = "p_1" if "p_1" in P.columns else P.columns[0]

    stress_col = "p_1" if dom_col == "p_0" else "p_0"

    eps = 1e-8
    def logit(p):
        p = np.clip(p, eps, 1-eps)
        return np.log(p/(1-p))

    def inv_logit(y):
        return 1/(1+np.exp(-y))

    p = P[dom_col].values
    y = logit(p)

    hard_labels = P[pcols].values.argmax(axis=1)
    dom_k = int(dom_col.split("_")[1])
    is_normal = (hard_labels == dom_k)

    idx = P.index
    rows = []

    for t in range(min_train, len(P)-horizon, forecast_step):
        y_train = y[:t+1]
        is_normal_train = is_normal[:t+1]

        current_p_normal = p[t]
        current_regime = "normal" if current_p_normal >= regime_threshold else "stress"

        y_normal = y_train[is_normal_train]
        y_stress = y_train[~is_normal_train]

        try:
            if len(y_normal) >= 30:
                fit_normal = ARIMA(y_normal, order=arima_order_normal).fit()
                yhat_normal = fit_normal.forecast(steps=horizon)
            else:
                yhat_normal = np.repeat(y[t], horizon)
        except Exception:
            yhat_normal = np.repeat(y[t], horizon)

        try:
            if len(y_stress) >= 30:
                fit_stress = ARIMA(y_stress, order=arima_order_stress).fit()
                yhat_stress = fit_stress.forecast(steps=horizon)
            else:
                yhat_stress = np.repeat(y[t], horizon)
        except Exception:
            yhat_stress = np.repeat(y[t], horizon)

        if blend_forecasts:
            w_normal = current_p_normal
            w_stress = 1 - current_p_normal
            yhat = w_normal * yhat_normal + w_stress * yhat_stress
            method = "blended"
        else:
            if current_regime == "normal":
                yhat = yhat_normal
                method = "normal_ar"
            else:
                yhat = yhat_stress
                method = "stress_ar"

        phat = inv_logit(yhat)
        origin_date = idx[t]

        for h in range(horizon):
            rows.append({
                "origin_date": origin_date,
                "step": h+1,
                "forecast_date": idx[t+h+1],
                "forecast_prob": float(phat[h]),
                "current_regime": current_regime,
                "p_normal_at_origin": float(current_p_normal),
                "method": method,
            })

    return pd.DataFrame(rows)


def build_snail_trail_forecasts(
    daily_probs_df: pd.DataFrame,
    target_regime: str = "p_1",
    arima_order=(1, 0, 0),
    horizon: int = 5,
    min_train: int = 250,
):
    """Build snail trail forecasts for a target regime."""
    p = daily_probs_df[target_regime].astype(float).values
    idx = daily_probs_df.index

    eps = 1e-8
    def logit(x):
        x = np.clip(x, eps, 1-eps)
        return np.log(x/(1-x))
    def inv_logit(y):
        return 1/(1+np.exp(-y))

    y = logit(p)

    rows = []
    for t in range(min_train, len(p) - 1):
        y_train = y[:t+1]
        try:
            fit = ARIMA(y_train, order=arima_order).fit()
            yhat = fit.forecast(steps=horizon)
            phat = inv_logit(np.asarray(yhat))
        except Exception:
            phat = np.full(horizon, p[t], dtype=float)

        for h in range(1, horizon+1):
            if t + h < len(idx):
                rows.append({
                    "origin_date": idx[t],
                    "step_ahead": h,
                    "forecast_date": idx[t+h],
                    "p_forecast": float(phat[h-1]),
                })

    return pd.DataFrame(rows)


def _get_forecast_vs_realized(snail_df: pd.DataFrame, daily_probs_df: pd.DataFrame):
    """Merge forecasts with realized values for evaluation."""
    pcols = [c for c in daily_probs_df.columns if c.startswith("p_")]
    dom_col = daily_probs_df[pcols].mean().idxmax()

    realized = daily_probs_df[[dom_col]].rename(columns={dom_col: "realized_prob"})

    merged = snail_df.merge(
        realized,
        left_on="forecast_date",
        right_index=True,
        how="inner"
    )
    merged["error"] = merged["forecast_prob"] - merged["realized_prob"]
    merged["abs_error"] = merged["error"].abs()
    merged["squared_error"] = merged["error"] ** 2

    return merged, dom_col


def plot_snail_trail(
    snail_df,
    daily_probs_df,
    alpha=0.3,
    mode="trails",
    show_realised=True,
    title=None,
    save_as="snail_trail.png",
):
    """Plot snail trail forecasts."""
    plt.figure(figsize=(14, 6))

    horizon = snail_df["step"].max() if len(snail_df) > 0 else 5

    if mode == "fan":
        fan = snail_df.groupby("forecast_date")["forecast_prob"].quantile([0.1, 0.5, 0.9]).unstack()
        fan.columns = ["p10", "p50", "p90"]
        fan = fan.sort_index()

        plt.fill_between(
            fan.index, fan["p10"], fan["p90"],
            alpha=alpha, color="steelblue", label="10th-90th percentile"
        )
        plt.plot(
            fan.index, fan["p50"],
            color="darkblue", linewidth=2, label="Median forecast"
        )
    else:
        for origin_date, group in snail_df.groupby("origin_date"):
            plt.plot(
                group["forecast_date"],
                group["forecast_prob"],
                alpha=alpha,
                linewidth=1,
                color="steelblue"
            )

    if show_realised:
        pcols = [c for c in daily_probs_df.columns if c.startswith("p_")]
        dom_col = daily_probs_df[pcols].mean().idxmax()

        plt.plot(
            daily_probs_df.index,
            daily_probs_df[dom_col],
            color="black",
            linewidth=1.5,
            label=f"Realised {dom_col}"
        )

    if title is None:
        title = f"{horizon}-Day Regime Forecast (Snail Trail)"
    plt.title(title)
    plt.ylabel("Probability")
    plt.xlabel("Date")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    save_chart(save_as)


def plot_snail_trail_twopanel(
    daily_probs_df,
    snail_df,
    target_regime="p_1",
    title="Snail-trail: AR(1)-logit forecasts",
    alpha=0.10,
    thin_origins=3,
    save_as="snail_trail_twopanel.png",
):
    """Two-panel snail-trail plot."""
    pcols = sorted(
        [c for c in daily_probs_df.columns if c.startswith("p_")],
        key=lambda c: int(c.split("_")[1])
    )

    hard = daily_probs_df[pcols].values.argmax(axis=1)
    target_k = int(target_regime.split("_")[1])
    realised = (hard == target_k).astype(int)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(13, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    unique_origins = snail_df["origin_date"].unique()
    unique_origins = unique_origins[::thin_origins]

    for origin in unique_origins:
        sub = snail_df[snail_df["origin_date"] == origin]
        ax1.plot(
            sub["forecast_date"],
            sub["p_forecast"],
            color="steelblue",
            alpha=alpha
        )

    ax1.plot(
        daily_probs_df.index,
        daily_probs_df[target_regime],
        color="darkblue",
        linewidth=2.5,
        label=f"Realised probability ({target_regime})"
    )

    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Probability")
    ax1.set_title(title)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    for i in range(len(realised) - 1):
        if realised[i] == 1:
            ax2.axvspan(
                daily_probs_df.index[i],
                daily_probs_df.index[i+1],
                color="black",
                alpha=0.35
            )

    ax2.set_ylim(0, 1)
    ax2.set_ylabel("High-vol regime")
    ax2.set_xlabel("Date")
    ax2.set_yticks([])
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    save_chart(save_as)


def plot_forecast_scatter(
    snail_df: pd.DataFrame,
    daily_probs_df: pd.DataFrame,
    save_as: str = "forecast_scatter.png",
):
    """Scatter plot: Forecast vs Realized with 45 degree line."""
    merged, dom_col = _get_forecast_vs_realized(snail_df, daily_probs_df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    ax1.scatter(merged["realized_prob"], merged["forecast_prob"], alpha=0.1, s=10)
    ax1.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect forecast")
    ax1.set_xlabel(f"Realized ({dom_col})")
    ax1.set_ylabel("Forecast")
    ax1.set_title("Forecast vs Realized (all horizons)")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(alpha=0.3)

    corr = merged["forecast_prob"].corr(merged["realized_prob"])
    ax1.text(0.05, 0.95, f"Corr: {corr:.3f}", transform=ax1.transAxes,
             fontsize=12, verticalalignment='top')

    ax2 = axes[1]
    horizons = sorted(merged["step"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(horizons)))

    for h, color in zip(horizons, colors):
        subset = merged[merged["step"] == h]
        ax2.scatter(subset["realized_prob"], subset["forecast_prob"],
                   alpha=0.3, s=15, c=[color], label=f"h={h}")

    ax2.plot([0, 1], [0, 1], "r--", linewidth=2)
    ax2.set_xlabel(f"Realized ({dom_col})")
    ax2.set_ylabel("Forecast")
    ax2.set_title("Forecast vs Realized (by horizon)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(loc="lower right", fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_chart(save_as)

    return merged


def plot_forecast_error_over_time(
    snail_df: pd.DataFrame,
    daily_probs_df: pd.DataFrame,
    horizon_to_plot: int = 1,
    save_as: str = "forecast_error_time.png",
):
    """Forecast error over time plot."""
    merged, dom_col = _get_forecast_vs_realized(snail_df, daily_probs_df)

    subset = merged[merged["step"] == horizon_to_plot].copy()
    subset = subset.sort_values("forecast_date")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1 = axes[0]
    ax1.plot(subset["forecast_date"], subset["forecast_prob"], alpha=0.7, label="Forecast")
    ax1.plot(subset["forecast_date"], subset["realized_prob"], alpha=0.7, label="Realized")
    ax1.set_ylabel("Probability")
    ax1.set_title(f"h={horizon_to_plot} Forecast vs Realized")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.bar(subset["forecast_date"], subset["error"], alpha=0.7, width=1)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Error (Forecast - Realized)")
    ax2.set_xlabel("Date")
    ax2.set_title(f"h={horizon_to_plot} Forecast Error")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_chart(save_as)


def plot_calibration(
    snail_df: pd.DataFrame,
    daily_probs_df: pd.DataFrame,
    n_bins: int = 10,
    save_as: str = "calibration.png",
):
    """Calibration plot."""
    merged, dom_col = _get_forecast_vs_realized(snail_df, daily_probs_df)

    pcols = [c for c in daily_probs_df.columns if c.startswith("p_")]
    dom_k = int(dom_col.split("_")[1])
    hard = daily_probs_df[pcols].values.argmax(axis=1)
    realized_binary = pd.Series((hard == dom_k).astype(int), index=daily_probs_df.index)

    merged = merged.merge(
        realized_binary.rename("realized_binary"),
        left_on="forecast_date",
        right_index=True,
        how="inner"
    )

    merged["bin"] = pd.cut(merged["forecast_prob"], bins=n_bins, labels=False)

    cal = merged.groupby("bin").agg({
        "forecast_prob": "mean",
        "realized_binary": "mean",
        "step": "count"
    }).rename(columns={"step": "count"})

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.scatter(cal["forecast_prob"], cal["realized_binary"], s=cal["count"]/10, alpha=0.7)
    ax.plot(cal["forecast_prob"], cal["realized_binary"], "o-", alpha=0.7, label="Model")

    ax.set_xlabel("Mean Forecast Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration Plot")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_chart(save_as)


def plot_accuracy_by_horizon(
    snail_df: pd.DataFrame,
    daily_probs_df: pd.DataFrame,
    save_as: str = "accuracy_by_horizon.png",
):
    """Plot forecast accuracy metrics by horizon."""
    merged, dom_col = _get_forecast_vs_realized(snail_df, daily_probs_df)

    metrics = merged.groupby("step").agg({
        "abs_error": "mean",
        "squared_error": "mean",
        "forecast_prob": lambda x: x.corr(merged.loc[x.index, "realized_prob"])
    }).rename(columns={
        "abs_error": "MAE",
        "squared_error": "MSE",
        "forecast_prob": "Correlation"
    })
    metrics["RMSE"] = np.sqrt(metrics["MSE"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(metrics.index, metrics["MAE"], marker="o")
    axes[0].set_xlabel("Forecast Horizon (days)")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Mean Absolute Error by Horizon")
    axes[0].grid(alpha=0.3)

    axes[1].plot(metrics.index, metrics["RMSE"], marker="o")
    axes[1].set_xlabel("Forecast Horizon (days)")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Root Mean Squared Error by Horizon")
    axes[1].grid(alpha=0.3)

    axes[2].plot(metrics.index, metrics["Correlation"], marker="o")
    axes[2].set_xlabel("Forecast Horizon (days)")
    axes[2].set_ylabel("Correlation")
    axes[2].set_title("Forecast-Realized Correlation by Horizon")
    axes[2].set_ylim(0, 1)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    save_chart(save_as)

    return metrics


def plot_rolling_hit_rate(
    snail_df: pd.DataFrame,
    daily_probs_df: pd.DataFrame,
    horizon_to_plot: int = 1,
    window: int = 60,
    save_as: str = "rolling_hit_rate.png",
):
    """Plot rolling hit rate over time."""
    merged, dom_col = _get_forecast_vs_realized(snail_df, daily_probs_df)

    pcols = [c for c in daily_probs_df.columns if c.startswith("p_")]
    dom_k = int(dom_col.split("_")[1])
    hard = daily_probs_df[pcols].values.argmax(axis=1)
    realized_binary = pd.Series((hard == dom_k).astype(int), index=daily_probs_df.index)

    merged = merged.merge(
        realized_binary.rename("realized_binary"),
        left_on="forecast_date",
        right_index=True,
        how="inner"
    )

    subset = merged[merged["step"] == horizon_to_plot].copy()
    subset = subset.sort_values("forecast_date")

    subset["forecast_binary"] = (subset["forecast_prob"] >= 0.5).astype(int)
    subset["hit"] = (subset["forecast_binary"] == subset["realized_binary"]).astype(int)
    subset["rolling_hit_rate"] = subset["hit"].rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(14, 5))
    plt.plot(subset["forecast_date"], subset["rolling_hit_rate"], linewidth=2)
    plt.axhline(0.5, color="red", linestyle="--", label="Random (50%)")
    plt.xlabel("Date")
    plt.ylabel(f"Rolling {window}-day Hit Rate")
    plt.title(f"Rolling Hit Rate (h={horizon_to_plot})")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_chart(save_as)


def compare_forecast_methods(
    daily_probs_df: pd.DataFrame,
    horizon: int = 21,
    forecast_step: int = 20,
    min_train: int = 250,
    save_as: str = "forecast_method_comparison.png",
):
    """Compare standard AR vs regime-aware AR forecasting."""
    print("Generating forecasts with different methods...")

    print("  Standard AR(1)...")
    snail_standard = generate_snail_trail_forecasts(
        daily_probs_df,
        horizon=horizon,
        forecast_step=forecast_step,
        min_train=min_train,
        arima_order=(1, 0, 0),
    )
    snail_standard["method"] = "Standard AR(1)"

    print("  Regime-aware AR (blended)...")
    snail_blended = generate_regime_aware_forecasts(
        daily_probs_df,
        horizon=horizon,
        forecast_step=forecast_step,
        min_train=min_train,
        blend_forecasts=True,
    )
    snail_blended_clean = snail_blended[["origin_date", "step", "forecast_date", "forecast_prob"]].copy()
    snail_blended_clean["method"] = "Regime-aware (blended)"

    print("  Regime-aware AR (hard switch)...")
    snail_hard = generate_regime_aware_forecasts(
        daily_probs_df,
        horizon=horizon,
        forecast_step=forecast_step,
        min_train=min_train,
        blend_forecasts=False,
    )
    snail_hard_clean = snail_hard[["origin_date", "step", "forecast_date", "forecast_prob"]].copy()
    snail_hard_clean["method"] = "Regime-aware (hard)"

    pcols = [c for c in daily_probs_df.columns if c.startswith("p_")]
    dom_col = daily_probs_df[pcols].mean().idxmax()
    realized = daily_probs_df[[dom_col]].rename(columns={dom_col: "realized_prob"})

    results = []
    for name, df in [("Standard AR(1)", snail_standard),
                     ("Regime-aware (blended)", snail_blended_clean),
                     ("Regime-aware (hard)", snail_hard_clean)]:

        merged = df.merge(realized, left_on="forecast_date", right_index=True, how="inner")
        merged["abs_error"] = (merged["forecast_prob"] - merged["realized_prob"]).abs()
        merged["squared_error"] = (merged["forecast_prob"] - merged["realized_prob"]) ** 2

        for h in merged["step"].unique():
            subset = merged[merged["step"] == h]
            results.append({
                "method": name,
                "horizon": h,
                "MAE": subset["abs_error"].mean(),
                "RMSE": np.sqrt(subset["squared_error"].mean()),
                "correlation": subset["forecast_prob"].corr(subset["realized_prob"]),
                "n": len(subset),
            })

    results_df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = results_df["method"].unique()
    colors = {"Standard AR(1)": "gray", "Regime-aware (blended)": "steelblue", "Regime-aware (hard)": "orange"}

    ax1 = axes[0]
    for m in methods:
        sub = results_df[results_df["method"] == m]
        ax1.plot(sub["horizon"], sub["MAE"], marker="o", label=m, color=colors.get(m, "black"), linewidth=2)
    ax1.set_xlabel("Forecast Horizon (days)")
    ax1.set_ylabel("MAE")
    ax1.set_title("Mean Absolute Error by Horizon")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    for m in methods:
        sub = results_df[results_df["method"] == m]
        ax2.plot(sub["horizon"], sub["RMSE"], marker="o", label=m, color=colors.get(m, "black"), linewidth=2)
    ax2.set_xlabel("Forecast Horizon (days)")
    ax2.set_ylabel("RMSE")
    ax2.set_title("Root Mean Squared Error by Horizon")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    for m in methods:
        sub = results_df[results_df["method"] == m]
        ax3.plot(sub["horizon"], sub["correlation"], marker="o", label=m, color=colors.get(m, "black"), linewidth=2)
    ax3.set_xlabel("Forecast Horizon (days)")
    ax3.set_ylabel("Correlation")
    ax3.set_title("Forecast-Realized Correlation by Horizon")
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    save_chart(save_as)

    print("\n" + "="*60)
    print("FORECAST METHOD COMPARISON SUMMARY")
    print("="*60)

    summary = results_df.groupby("method").agg({
        "MAE": "mean",
        "RMSE": "mean",
        "correlation": "mean",
    }).round(4)
    print(summary)

    best_mae = summary["MAE"].idxmin()
    print(f"\nBest method (lowest MAE): {best_mae}")

    return results_df, snail_blended


def plot_forecast_evaluation_summary(
    snail_df: pd.DataFrame,
    daily_probs_df: pd.DataFrame,
    save_prefix: str = "forecast_eval",
):
    """Generate all forecast evaluation plots."""
    print("Generating forecast evaluation plots...")

    print("  1. Scatter plot...")
    plot_forecast_scatter(snail_df, daily_probs_df, save_as=f"{save_prefix}_scatter.png")

    print("  2. Error over time...")
    plot_forecast_error_over_time(snail_df, daily_probs_df, horizon_to_plot=1,
                                   save_as=f"{save_prefix}_error_h1.png")

    print("  3. Calibration plot...")
    plot_calibration(snail_df, daily_probs_df, save_as=f"{save_prefix}_calibration.png")

    print("  4. Accuracy by horizon...")
    metrics = plot_accuracy_by_horizon(snail_df, daily_probs_df,
                                        save_as=f"{save_prefix}_by_horizon.png")

    print("  5. Rolling hit rate...")
    plot_rolling_hit_rate(snail_df, daily_probs_df, horizon_to_plot=1,
                          save_as=f"{save_prefix}_hit_rate.png")

    print("\nForecast evaluation complete!")
    return metrics
