"""
Figure 4 — Portfolio backtest (4-panel: cumulative returns, Sharpe bars, drawdown, summary table).

Five strategies:
- Unconditional max-Sharpe (expanding)
- Filtered Oracle (uses future filtered probabilities — infeasible, upper bound)
- AR Baseline (logit-level AR forecast of the high-vol probability)
- AR Change (delta-logit AR forecast)
- Random Walk

Run:  python -m research.analysis.backtest
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from scipy.optimize import minimize

from core.config import OUTPUT_DIR, FIGURES_DIR, TRANSACTION_COST


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def logit(p, eps=1e-8):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def inv_logit(y):
    return 1 / (1 + np.exp(-y))


def select_ar_order_bic(y, max_lag=5):
    best_bic = np.inf
    best_order = 1
    for p in range(1, max_lag + 1):
        try:
            model = AutoReg(y, lags=p, old_names=False).fit()
            if model.bic < best_bic:
                best_bic = model.bic
                best_order = p
        except:
            continue
    return best_order


def optimize_max_sharpe(mu, cov, rf=0.0, max_weight=0.40):
    n = len(mu)
    if np.any(np.isnan(mu)) or np.any(np.isnan(cov)):
        return np.ones(n) / n

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -(ret - rf) / vol if vol > 1e-10 else 0

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0, max_weight) for _ in range(n)]
    w0 = np.ones(n) / n

    try:
        result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        w_opt = result.x
        w_opt = np.maximum(w_opt, 0)
        w_opt = w_opt / w_opt.sum()
        return w_opt
    except:
        return np.ones(n) / n


def compute_metrics(net_returns, turnover_list):
    """net_returns are LOG returns."""
    ann_factor = 252
    total_return = np.exp(net_returns.sum()) - 1
    n_years = len(net_returns) / ann_factor
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = net_returns.std() * np.sqrt(ann_factor)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    cumulative = np.exp(net_returns.cumsum())
    rolling_max = cumulative.expanding().max()
    drawdown = cumulative / rolling_max - 1
    max_dd = drawdown.min()

    downside = net_returns[net_returns < 0]
    downside_std = downside.std() * np.sqrt(ann_factor) if len(downside) > 0 else 0
    sortino = ann_return / downside_std if downside_std > 0 else 0

    return {
        "ann_return": ann_return,
        "ann_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
    }


# =============================================================================
# BACKTEST FUNCTIONS
# =============================================================================

def backtest_unconditional_expanding(returns, min_train, transaction_cost):
    dates = returns.index
    n_assets = returns.shape[1]
    rebalance_dates = returns.index.to_series().groupby(returns.index.to_period("M")).last().values
    rebalance_set = set(pd.DatetimeIndex(rebalance_dates))

    current_weights = np.ones(n_assets) / n_assets
    port_returns = []
    tc_list = []
    turnover_list = []

    for t, date in enumerate(dates):
        daily_ret = returns.iloc[t].values
        new_values = current_weights * np.exp(daily_ret)
        port_ret = np.log(new_values.sum() / current_weights.sum())
        port_returns.append(port_ret)
        current_weights = new_values / new_values.sum()

        if date in rebalance_set and t >= min_train:
            past_returns = returns.iloc[:t+1]
            mu = past_returns.mean().values * 252
            cov = past_returns.cov().values * 252
            target_weights = optimize_max_sharpe(mu, cov)

            turnover = np.abs(target_weights - current_weights).sum() / 2
            tc = turnover * transaction_cost
            turnover_list.append(turnover)
            tc_list.append(tc)
            current_weights = target_weights.copy()
        else:
            turnover_list.append(0.0)
            tc_list.append(0.0)

    port_ret_series = pd.Series(port_returns, index=dates)
    tc_series = pd.Series(tc_list, index=dates)
    net_returns = port_ret_series - tc_series
    cumulative = np.exp(net_returns.cumsum())
    metrics = compute_metrics(net_returns, turnover_list)

    return {"net_returns": net_returns, "cumulative": cumulative, "metrics": metrics}


def compute_regime_portfolios_expanding(returns, probs, end_idx):
    ret_slice = returns.iloc[:end_idx]
    prob_slice = probs.iloc[:end_idx]
    regime_portfolios = {}

    for k in [0, 1]:
        weights = prob_slice[f"p_{k}"].values
        if weights.sum() < 10:
            regime_portfolios[k] = np.ones(ret_slice.shape[1]) / ret_slice.shape[1]
            continue

        weighted_returns = ret_slice.values * weights[:, np.newaxis]
        mu = weighted_returns.sum(axis=0) / weights.sum()
        centered = ret_slice.values - mu
        cov = (centered.T * weights) @ centered / weights.sum()
        mu_ann = mu * 252
        cov_ann = cov * 252
        regime_portfolios[k] = optimize_max_sharpe(mu_ann, cov_ann)

    return regime_portfolios


def backtest_regime_aware_expanding(returns, probs, forecast_probs, prob_column, min_train, transaction_cost, recompute_freq=63):
    common_dates = returns.index.intersection(forecast_probs.index)
    returns = returns.loc[common_dates]
    probs = probs.loc[common_dates]
    forecasts = forecast_probs.loc[common_dates]

    dates = returns.index
    n_assets = returns.shape[1]
    rebalance_dates = dates.to_series().groupby(dates.to_period("M")).last().values
    rebalance_set = set(pd.DatetimeIndex(rebalance_dates))

    current_weights = np.ones(n_assets) / n_assets
    regime_portfolios = {0: current_weights.copy(), 1: current_weights.copy()}

    port_returns = []
    tc_list = []
    turnover_list = []
    last_recompute = 0

    for t, date in enumerate(dates):
        orig_idx = returns.index.get_loc(date)
        daily_ret = returns.iloc[t].values
        new_values = current_weights * np.exp(daily_ret)
        port_ret = np.log(new_values.sum() / current_weights.sum())
        port_returns.append(port_ret)
        current_weights = new_values / new_values.sum()

        if date in rebalance_set and orig_idx >= min_train:
            if t - last_recompute >= recompute_freq or t == 0:
                regime_portfolios = compute_regime_portfolios_expanding(returns, probs, orig_idx)
                last_recompute = t

            p1 = forecasts.loc[date, prob_column]
            p0 = 1 - p1
            target_weights = p0 * regime_portfolios[0] + p1 * regime_portfolios[1]
            target_weights = target_weights / target_weights.sum()

            turnover = np.abs(target_weights - current_weights).sum() / 2
            tc = turnover * transaction_cost
            turnover_list.append(turnover)
            tc_list.append(tc)
            current_weights = target_weights.copy()
        else:
            turnover_list.append(0.0)
            tc_list.append(0.0)

    port_ret_series = pd.Series(port_returns, index=dates)
    tc_series = pd.Series(tc_list, index=dates)
    net_returns = port_ret_series - tc_series
    cumulative = np.exp(net_returns.cumsum())
    metrics = compute_metrics(net_returns, turnover_list)

    return {"net_returns": net_returns, "cumulative": cumulative, "metrics": metrics}


def generate_ar_forecasts(probs_df, horizon=21, min_train=252, refit_freq=21, max_lag=5):
    p = probs_df["p_1"].values
    y = logit(p)
    dy = np.diff(y)
    dy = np.concatenate([[0], dy])
    dates = probs_df.index
    n = len(p)
    results = []

    ar_order_baseline = 1
    ar_order_change = 1

    for t in range(min_train, n - horizon):
        y_train = y[:t+1]
        dy_train = dy[1:t+1]

        if (t - min_train) % refit_freq == 0:
            ar_order_baseline = select_ar_order_bic(y_train, max_lag)
            if len(dy_train) > 10:
                ar_order_change = select_ar_order_bic(dy_train, max_lag)

        try:
            fit_base = AutoReg(y_train, lags=ar_order_baseline, old_names=False).fit()
            yhat_base = fit_base.forecast(steps=horizon)
            p_baseline = inv_logit(np.asarray(yhat_base)[-1])
        except:
            p_baseline = p[t]

        try:
            fit_change = AutoReg(dy_train, lags=ar_order_change, old_names=False).fit()
            dyhat = fit_change.forecast(steps=horizon)
            y_forecast = y[t] + np.sum(dyhat)
            p_change = inv_logit(y_forecast)
        except:
            p_change = p[t]

        target_date = dates[t + horizon]
        # Oracle = average realised p_1 over the upcoming holding period
        # (forward 'horizon' trading days from target_date, truncated at data end)
        fwd_end = min(t + 2 * horizon, n)
        oracle_fwd = float(np.mean(p[t + horizon : fwd_end])) if fwd_end > t + horizon else float(p[t + horizon])
        results.append({
            "date": target_date,
            "p_1_filtered": oracle_fwd,
            "p_1_current": p[t],
            "p_1_ar_baseline": p_baseline,
            "p_1_ar_change": p_change,
        })

    return pd.DataFrame(results).set_index("date")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PORTFOLIO BACKTEST WITH ORACLE")
    print("=" * 70)
    print()

    # Load regime probabilities and asset returns (log)
    probs_path = OUTPUT_DIR / "daily_regime_probabilities.csv"
    probs_df = pd.read_csv(probs_path, index_col=0, parse_dates=True)
    print(f"Loaded {len(probs_df)} days of regime probabilities")

    from data.reader1 import DataReader
    returns_df = DataReader().read_retns().dropna()
    returns_df = returns_df[np.isfinite(returns_df).all(axis=1)]
    print(f"Loaded {len(returns_df)} days of log returns")

    common_dates = probs_df.index.intersection(returns_df.index)
    probs_df = probs_df.loc[common_dates]
    returns_df = returns_df.loc[common_dates]
    print(f"Common dates: {len(common_dates)}")
    print()

    # Generate forecasts
    horizon = 21
    min_train = 252
    tc = TRANSACTION_COST  # 5 bps

    print("Generating AR forecasts...")
    forecast_df = generate_ar_forecasts(probs_df, horizon=horizon, min_train=min_train)
    print(f"Generated {len(forecast_df)} forecasts")
    print()

    # Run backtests
    print("Running backtests...")

    print("  Unconditional (Expanding)...")
    res_uncond = backtest_unconditional_expanding(returns_df, min_train=min_train, transaction_cost=tc)
    mask = res_uncond["net_returns"].index.isin(forecast_df.index)
    res_uncond["net_returns"] = res_uncond["net_returns"][mask]
    res_uncond["cumulative"] = np.exp(res_uncond["net_returns"].cumsum())
    res_uncond["metrics"] = compute_metrics(res_uncond["net_returns"], [0.01] * len(res_uncond["net_returns"]))

    print("  Filtered (Oracle)...")
    res_oracle = backtest_regime_aware_expanding(
        returns_df, probs_df, forecast_df, "p_1_filtered", min_train, tc
    )

    print("  AR Baseline...")
    res_ar_base = backtest_regime_aware_expanding(
        returns_df, probs_df, forecast_df, "p_1_ar_baseline", min_train, tc
    )

    print("  AR Change...")
    res_ar_change = backtest_regime_aware_expanding(
        returns_df, probs_df, forecast_df, "p_1_ar_change", min_train, tc
    )

    print("  Random Walk...")
    res_rw = backtest_regime_aware_expanding(
        returns_df, probs_df, forecast_df, "p_1_current", min_train, tc
    )

    # Compile results
    results = {
        "Unconditional": res_uncond,
        "Oracle": res_oracle,
        "AR Baseline": res_ar_base,
        "AR Change": res_ar_change,
        "Random Walk": res_rw,
    }

    # Colors and styles
    colors = {
        "Unconditional": "green",
        "Oracle": "dodgerblue",
        "AR Baseline": "orange",
        "AR Change": "steelblue",
        "Random Walk": "gray",
    }

    # ==========================================================================
    # CREATE FIGURE
    # ==========================================================================

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Cumulative Returns (line chart)
    ax1 = axes[0, 0]
    for name, res in results.items():
        linestyle = "--" if name == "Oracle" else "-"
        ax1.plot(res["cumulative"], label=name, color=colors[name],
                 linestyle=linestyle, linewidth=1.5)
    ax1.set_ylabel("Cumulative Return")
    ax1.set_title("Panel A: Cumulative Returns", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel B: Sharpe Ratio Comparison (bar chart)
    ax2 = axes[0, 1]
    names = list(results.keys())
    sharpes = [results[n]["metrics"]["sharpe_ratio"] for n in names]
    bar_colors = [colors[n] for n in names]
    bars = ax2.barh(names, sharpes, color=bar_colors, edgecolor="black", linewidth=1)
    ax2.set_xlabel("Sharpe Ratio")
    ax2.set_title("Panel B: Sharpe Ratio Comparison", fontsize=11, fontweight="bold")
    ax2.axvline(x=results["Unconditional"]["metrics"]["sharpe_ratio"], color="green",
                linestyle="--", alpha=0.5, linewidth=2)

    for bar, val in zip(bars, sharpes):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
                 va="center", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, max(sharpes) * 1.15)

    # Panel C: Drawdown Over Time (line chart)
    ax3 = axes[1, 0]
    for name, res in results.items():
        cumulative = res["cumulative"]
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative / rolling_max - 1) * 100
        linestyle = "--" if name == "Oracle" else "-"
        ax3.plot(drawdown, label=name, color=colors[name],
                 linestyle=linestyle, linewidth=1)
    ax3.set_ylabel("Drawdown (%)")
    ax3.set_title("Panel C: Drawdown Over Time", fontsize=11, fontweight="bold")
    ax3.legend(loc="lower left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel D: Performance Summary Table
    ax4 = axes[1, 1]
    ax4.axis("off")

    table_data = [["Strategy", "Return", "Vol", "Sharpe", "Max DD", "Sortino"]]
    for name in names:
        m = results[name]["metrics"]
        table_data.append([
            name,
            f'{m["ann_return"]*100:.2f}%',
            f'{m["ann_volatility"]*100:.2f}%',
            f'{m["sharpe_ratio"]:.3f}',
            f'{m["max_drawdown"]*100:.1f}%',
            f'{m["sortino_ratio"]:.3f}',
        ])

    table = ax4.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
        colWidths=[0.22, 0.12, 0.12, 0.12, 0.12, 0.12]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor("lightgray")
        table[(0, i)].set_text_props(fontweight="bold")

    # Highlight best Sharpe
    best_idx = np.argmax(sharpes)
    table[(best_idx + 1, 0)].set_facecolor("lightgreen")
    table[(best_idx + 1, 3)].set_facecolor("lightgreen")

    ax4.set_title("Panel D: Performance Summary", fontsize=11, fontweight="bold", y=0.95)

    # Add note
    fig.text(0.5, 0.02,
             "Note: Oracle uses future regime probabilities (infeasible). TC = 5bps.",
             ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    fig.savefig(FIGURES_DIR / "figure4_backtest.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "figure4_backtest.pdf", bbox_inches="tight")
    print(f"\nSaved: {FIGURES_DIR / 'figure4_backtest.png'}")

    plt.close()

    # Print summary
    print()
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print()

    for name in names:
        m = results[name]["metrics"]
        print(f"{name:15} | Sharpe: {m['sharpe_ratio']:.3f} | Return: {m['ann_return']*100:.2f}% | Max DD: {m['max_drawdown']*100:.1f}%")


if __name__ == "__main__":
    main()
