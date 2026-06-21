"""
Regime-aware portfolio construction and backtesting.

Contains:
- RegimePortfolioOptimizer: Mean-variance optimization per regime
- StrategicPortfolio: Regime-weighted allocation using long-run probabilities
- PortfolioBacktester: Monthly rebalancing with transaction costs
- Performance metrics and comparison functions

References:
- Markowitz (1952) - Mean-variance portfolio optimization
- Hamilton (1989) - Regime-switching models
- Guidolin & Timmermann (2007) - Asset allocation under regime switching
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings
from pathlib import Path

# Output directory (can be patched by calling code)
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PortfolioResult:
    """Result of portfolio optimization."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    regime: Optional[int] = None


@dataclass
class BacktestResult:
    """Result of portfolio backtest."""
    returns: pd.Series
    weights_history: pd.DataFrame
    turnover: pd.Series
    transaction_costs: pd.Series
    net_returns: pd.Series
    metrics: Dict[str, float]


# =============================================================================
# MEAN-VARIANCE OPTIMIZER
# =============================================================================

class RegimePortfolioOptimizer:
    """
    Mean-variance portfolio optimization with regime-specific moments.

    Solves:
        min_w  w'Σw
        s.t.   w'μ = μ* (target return)
               1'w = 1   (fully invested)
               w >= 0    (no short sales)

    Parameters
    ----------
    risk_free_rate : float
        Annual risk-free rate for Sharpe ratio (default: 0.0)
    allow_short : bool
        Allow short positions (default: False)
    max_weight : float
        Maximum weight per asset (default: 1.0)
    min_weight : float
        Minimum weight per asset (default: 0.0)
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        allow_short: bool = False,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
    ):
        self.risk_free_rate = risk_free_rate
        self.allow_short = allow_short
        self.max_weight = max_weight
        self.min_weight = min_weight if not allow_short else -max_weight

    def _portfolio_volatility(self, w: np.ndarray, cov: np.ndarray) -> float:
        """Portfolio volatility (standard deviation)."""
        return np.sqrt(w @ cov @ w)

    def _portfolio_return(self, w: np.ndarray, mu: np.ndarray) -> float:
        """Portfolio expected return."""
        return w @ mu

    def optimize_min_variance(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        target_return: Optional[float] = None,
    ) -> PortfolioResult:
        """
        Minimum variance portfolio, optionally with target return constraint.

        Parameters
        ----------
        mu : array
            Expected returns (n_assets,)
        cov : array
            Covariance matrix (n_assets, n_assets)
        target_return : float, optional
            Target portfolio return. If None, finds global minimum variance.

        Returns
        -------
        PortfolioResult
        """
        n = len(mu)
        w0 = np.ones(n) / n  # equal weight initial guess

        # Objective: minimize variance
        def objective(w):
            return w @ cov @ w

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # fully invested
        ]

        if target_return is not None:
            constraints.append({
                "type": "eq",
                "fun": lambda w, mu=mu, t=target_return: w @ mu - t
            })

        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000},
        )

        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")

        w_opt = result.x
        ret = self._portfolio_return(w_opt, mu)
        vol = self._portfolio_volatility(w_opt, cov)

        # Annualize for Sharpe (assuming daily returns, 252 days)
        sharpe = (ret * 252 - self.risk_free_rate) / (vol * np.sqrt(252)) if vol > 0 else 0.0

        return PortfolioResult(
            weights=w_opt,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
        )

    def optimize_max_sharpe(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
    ) -> PortfolioResult:
        """
        Maximum Sharpe ratio portfolio (tangency portfolio).

        Parameters
        ----------
        mu : array
            Expected returns (n_assets,)
        cov : array
            Covariance matrix (n_assets, n_assets)

        Returns
        -------
        PortfolioResult
        """
        n = len(mu)
        w0 = np.ones(n) / n

        # Daily risk-free rate
        rf_daily = self.risk_free_rate / 252

        # Objective: minimize negative Sharpe ratio
        def neg_sharpe(w):
            ret = w @ mu
            vol = np.sqrt(w @ cov @ w)
            if vol < 1e-10:
                return 1e10
            return -(ret - rf_daily) / vol

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]

        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]

        result = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000},
        )

        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")

        w_opt = result.x
        ret = self._portfolio_return(w_opt, mu)
        vol = self._portfolio_volatility(w_opt, cov)
        sharpe = (ret * 252 - self.risk_free_rate) / (vol * np.sqrt(252)) if vol > 0 else 0.0

        return PortfolioResult(
            weights=w_opt,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
        )

    def optimize_target_volatility(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        target_vol: float,
    ) -> PortfolioResult:
        """
        Maximum return portfolio for a given target volatility.

        Parameters
        ----------
        mu : array
            Expected returns (n_assets,)
        cov : array
            Covariance matrix (n_assets, n_assets)
        target_vol : float
            Target portfolio volatility (daily)

        Returns
        -------
        PortfolioResult
        """
        n = len(mu)
        w0 = np.ones(n) / n

        # Objective: maximize return (minimize negative return)
        def objective(w):
            return -w @ mu

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: target_vol**2 - w @ cov @ w},  # vol <= target
        ]

        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000},
        )

        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")

        w_opt = result.x
        ret = self._portfolio_return(w_opt, mu)
        vol = self._portfolio_volatility(w_opt, cov)
        sharpe = (ret * 252 - self.risk_free_rate) / (vol * np.sqrt(252)) if vol > 0 else 0.0

        return PortfolioResult(
            weights=w_opt,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
        )

    def compute_efficient_frontier(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        n_points: int = 50,
    ) -> pd.DataFrame:
        """
        Compute the efficient frontier.

        Parameters
        ----------
        mu : array
            Expected returns
        cov : array
            Covariance matrix
        n_points : int
            Number of points on the frontier

        Returns
        -------
        DataFrame with columns: target_return, volatility, sharpe, weights
        """
        # Find return range
        gmv = self.optimize_min_variance(mu, cov)
        max_sharpe = self.optimize_max_sharpe(mu, cov)

        min_ret = gmv.expected_return
        max_ret = max(mu)  # maximum individual asset return

        target_returns = np.linspace(min_ret, max_ret, n_points)

        results = []
        for target in target_returns:
            try:
                result = self.optimize_min_variance(mu, cov, target_return=target)
                results.append({
                    "target_return": target,
                    "volatility": result.volatility,
                    "sharpe": result.sharpe_ratio,
                    "weights": result.weights,
                })
            except Exception:
                continue

        return pd.DataFrame(results)


# =============================================================================
# REGIME-CONDITIONAL PORTFOLIOS
# =============================================================================

def compute_regime_portfolios(
    regime_means: Dict[int, np.ndarray],
    regime_covs: Dict[int, np.ndarray],
    optimizer: Optional[RegimePortfolioOptimizer] = None,
    method: str = "max_sharpe",
    target_return: Optional[float] = None,
    target_vol: Optional[float] = None,
) -> Dict[int, PortfolioResult]:
    """
    Compute optimal portfolios for each regime.

    Parameters
    ----------
    regime_means : dict
        {regime_id: mean_vector} for each regime
    regime_covs : dict
        {regime_id: covariance_matrix} for each regime
    optimizer : RegimePortfolioOptimizer, optional
        Optimizer instance. Created with defaults if None.
    method : str
        "max_sharpe", "min_variance", "target_return", or "target_vol"
    target_return : float, optional
        Target return for "target_return" method
    target_vol : float, optional
        Target volatility for "target_vol" method

    Returns
    -------
    dict : {regime_id: PortfolioResult}
    """
    if optimizer is None:
        optimizer = RegimePortfolioOptimizer()

    portfolios = {}

    for k in regime_means.keys():
        mu = regime_means[k]
        cov = regime_covs[k]

        if method == "max_sharpe":
            result = optimizer.optimize_max_sharpe(mu, cov)
        elif method == "min_variance":
            result = optimizer.optimize_min_variance(mu, cov)
        elif method == "target_return":
            if target_return is None:
                raise ValueError("target_return required for method='target_return'")
            result = optimizer.optimize_min_variance(mu, cov, target_return=target_return)
        elif method == "target_vol":
            if target_vol is None:
                raise ValueError("target_vol required for method='target_vol'")
            result = optimizer.optimize_target_volatility(mu, cov, target_vol=target_vol)
        else:
            raise ValueError(f"Unknown method: {method}")

        result.regime = k
        portfolios[k] = result

    return portfolios


def compute_strategic_portfolio(
    regime_portfolios: Dict[int, PortfolioResult],
    regime_probabilities: Dict[int, float],
) -> np.ndarray:
    """
    Compute strategic regime-aware portfolio weights.

    Combines regime-conditional portfolios using long-run regime probabilities:
        w_strategic = sum_k pi_k * w_k

    Parameters
    ----------
    regime_portfolios : dict
        {regime_id: PortfolioResult} from compute_regime_portfolios
    regime_probabilities : dict
        {regime_id: probability} long-run regime probabilities

    Returns
    -------
    array : strategic portfolio weights
    """
    # Validate probabilities sum to 1
    total_prob = sum(regime_probabilities.values())
    if abs(total_prob - 1.0) > 1e-6:
        warnings.warn(f"Regime probabilities sum to {total_prob}, normalizing...")
        regime_probabilities = {k: v / total_prob for k, v in regime_probabilities.items()}

    # Get number of assets from first portfolio
    first_portfolio = next(iter(regime_portfolios.values()))
    n_assets = len(first_portfolio.weights)

    # Weighted combination
    w_strategic = np.zeros(n_assets)
    for k, portfolio in regime_portfolios.items():
        if k in regime_probabilities:
            w_strategic += regime_probabilities[k] * portfolio.weights

    # Normalize to ensure sum = 1 (should be close already)
    w_strategic = w_strategic / w_strategic.sum()

    return w_strategic


def compute_unconditional_portfolio(
    returns: np.ndarray,
    optimizer: Optional[RegimePortfolioOptimizer] = None,
    method: str = "max_sharpe",
) -> PortfolioResult:
    """
    Compute portfolio using unconditional (full-sample) moments.

    Parameters
    ----------
    returns : array
        Return matrix (T, n_assets)
    optimizer : RegimePortfolioOptimizer, optional
    method : str
        Optimization method

    Returns
    -------
    PortfolioResult
    """
    if optimizer is None:
        optimizer = RegimePortfolioOptimizer()

    mu = returns.mean(axis=0)
    cov = np.cov(returns, rowvar=False)

    if method == "max_sharpe":
        return optimizer.optimize_max_sharpe(mu, cov)
    elif method == "min_variance":
        return optimizer.optimize_min_variance(mu, cov)
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# PORTFOLIO BACKTESTER
# =============================================================================

class PortfolioBacktester:
    """
    Backtest portfolios with monthly rebalancing and transaction costs.

    Parameters
    ----------
    rebalance_freq : str
        Rebalancing frequency: "M" (monthly), "W" (weekly), "Q" (quarterly)
    transaction_cost : float
        Proportional transaction cost (e.g., 0.001 = 10 bps)
    """

    def __init__(
        self,
        rebalance_freq: str = "M",
        transaction_cost: float = 0.001,
    ):
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Get dates when rebalancing occurs."""
        if self.rebalance_freq == "M":
            # Last trading day of each month
            arr = dates.to_series().groupby(dates.to_period("M")).last().values
        elif self.rebalance_freq == "W":
            arr = dates.to_series().groupby(dates.to_period("W")).last().values
        elif self.rebalance_freq == "Q":
            arr = dates.to_series().groupby(dates.to_period("Q")).last().values
        else:
            raise ValueError(f"Unknown rebalance_freq: {self.rebalance_freq}")
        return pd.DatetimeIndex(arr)

    def backtest_static(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
    ) -> BacktestResult:
        """
        Backtest a static (fixed weight) portfolio with rebalancing.

        Parameters
        ----------
        returns : DataFrame
            Daily returns (T, n_assets)
        weights : array
            Target portfolio weights

        Returns
        -------
        BacktestResult
        """
        dates = returns.index
        n_assets = len(weights)
        rebalance_dates = set(self._get_rebalance_dates(dates))

        # Initialize
        current_weights = weights.copy()
        portfolio_returns = []
        weights_history = []
        turnover_list = []
        tc_list = []

        for t, date in enumerate(dates):
            # Record weights at start of day
            weights_history.append(current_weights.copy())

            # Daily log return (exact wealth update under log returns)
            daily_ret = returns.iloc[t].values
            new_values = current_weights * np.exp(daily_ret)
            port_ret = np.log(new_values.sum() / current_weights.sum())
            portfolio_returns.append(port_ret)
            current_weights = new_values / new_values.sum()

            # Rebalance if needed
            if date in rebalance_dates:
                turnover = np.abs(weights - current_weights).sum() / 2
                tc = turnover * self.transaction_cost
                turnover_list.append(turnover)
                tc_list.append(tc)
                current_weights = weights.copy()
            else:
                turnover_list.append(0.0)
                tc_list.append(0.0)

        # Convert to Series/DataFrame
        port_ret_series = pd.Series(portfolio_returns, index=dates, name="return")
        weights_df = pd.DataFrame(weights_history, index=dates, columns=returns.columns)
        turnover_series = pd.Series(turnover_list, index=dates, name="turnover")
        tc_series = pd.Series(tc_list, index=dates, name="transaction_cost")
        net_returns = port_ret_series - tc_series

        # Compute metrics
        metrics = self._compute_metrics(net_returns)

        return BacktestResult(
            returns=port_ret_series,
            weights_history=weights_df,
            turnover=turnover_series,
            transaction_costs=tc_series,
            net_returns=net_returns,
            metrics=metrics,
        )

    def backtest_regime_aware(
        self,
        returns: pd.DataFrame,
        regime_probs: pd.DataFrame,
        regime_portfolios: Dict[int, PortfolioResult],
        blend_weights: bool = True,
    ) -> BacktestResult:
        """
        Backtest regime-aware portfolio that adjusts to current regime probabilities.

        Parameters
        ----------
        returns : DataFrame
            Daily returns (T, n_assets)
        regime_probs : DataFrame
            Daily regime probabilities with columns p_0, p_1, ...
        regime_portfolios : dict
            {regime_id: PortfolioResult}
        blend_weights : bool
            If True, blend portfolio weights by regime probabilities.
            If False, use portfolio of dominant regime.

        Returns
        -------
        BacktestResult
        """
        dates = returns.index
        common_dates = dates.intersection(regime_probs.index)
        returns = returns.loc[common_dates]
        regime_probs = regime_probs.loc[common_dates]
        dates = common_dates

        rebalance_dates = set(self._get_rebalance_dates(dates))

        # Get regime columns
        prob_cols = sorted([c for c in regime_probs.columns if c.startswith("p_")])

        # Initialize with blended weights
        current_weights = self._get_regime_weights(
            regime_probs.iloc[0], prob_cols, regime_portfolios, blend_weights
        )

        portfolio_returns = []
        weights_history = []
        turnover_list = []
        tc_list = []

        for t, date in enumerate(dates):
            # Record weights
            weights_history.append(current_weights.copy())

            # Daily log return (exact wealth update under log returns)
            daily_ret = returns.iloc[t].values
            new_values = current_weights * np.exp(daily_ret)
            port_ret = np.log(new_values.sum() / current_weights.sum())
            portfolio_returns.append(port_ret)
            current_weights = new_values / new_values.sum()

            # Rebalance if needed
            if date in rebalance_dates:
                target_weights = self._get_regime_weights(
                    regime_probs.iloc[t], prob_cols, regime_portfolios, blend_weights
                )
                turnover = np.abs(target_weights - current_weights).sum() / 2
                tc = turnover * self.transaction_cost
                turnover_list.append(turnover)
                tc_list.append(tc)
                current_weights = target_weights.copy()
            else:
                turnover_list.append(0.0)
                tc_list.append(0.0)

        # Convert to Series/DataFrame
        port_ret_series = pd.Series(portfolio_returns, index=dates, name="return")
        weights_df = pd.DataFrame(weights_history, index=dates, columns=returns.columns)
        turnover_series = pd.Series(turnover_list, index=dates, name="turnover")
        tc_series = pd.Series(tc_list, index=dates, name="transaction_cost")
        net_returns = port_ret_series - tc_series

        metrics = self._compute_metrics(net_returns)

        return BacktestResult(
            returns=port_ret_series,
            weights_history=weights_df,
            turnover=turnover_series,
            transaction_costs=tc_series,
            net_returns=net_returns,
            metrics=metrics,
        )

    def backtest_regime_conditional(
        self,
        returns: pd.DataFrame,
        regime_labels: pd.Series,
        regime_portfolios: Dict[int, PortfolioResult],
    ) -> BacktestResult:
        """
        Backtest regime-conditional portfolio that uses the portfolio
        corresponding to the current regime (hard assignment).

        Parameters
        ----------
        returns : DataFrame
            Daily returns (T, n_assets)
        regime_labels : Series
            Hard regime labels for each day
        regime_portfolios : dict
            {regime_id: PortfolioResult}

        Returns
        -------
        BacktestResult
        """
        dates = returns.index
        common_dates = dates.intersection(regime_labels.index)
        returns = returns.loc[common_dates]
        regime_labels = regime_labels.loc[common_dates]
        dates = common_dates

        rebalance_dates = set(self._get_rebalance_dates(dates))

        # Get initial regime
        current_regime = regime_labels.iloc[0]
        current_weights = regime_portfolios[current_regime].weights.copy()

        portfolio_returns = []
        weights_history = []
        turnover_list = []
        tc_list = []

        for t, date in enumerate(dates):
            weights_history.append(current_weights.copy())

            # Daily log return (exact wealth update under log returns)
            daily_ret = returns.iloc[t].values
            new_values = current_weights * np.exp(daily_ret)
            port_ret = np.log(new_values.sum() / current_weights.sum())
            portfolio_returns.append(port_ret)
            current_weights = new_values / new_values.sum()

            # Rebalance
            if date in rebalance_dates:
                current_regime = regime_labels.iloc[t]
                target_weights = regime_portfolios[current_regime].weights
                turnover = np.abs(target_weights - current_weights).sum() / 2
                tc = turnover * self.transaction_cost
                turnover_list.append(turnover)
                tc_list.append(tc)
                current_weights = target_weights.copy()
            else:
                turnover_list.append(0.0)
                tc_list.append(0.0)

        port_ret_series = pd.Series(portfolio_returns, index=dates, name="return")
        weights_df = pd.DataFrame(weights_history, index=dates, columns=returns.columns)
        turnover_series = pd.Series(turnover_list, index=dates, name="turnover")
        tc_series = pd.Series(tc_list, index=dates, name="transaction_cost")
        net_returns = port_ret_series - tc_series

        metrics = self._compute_metrics(net_returns)

        return BacktestResult(
            returns=port_ret_series,
            weights_history=weights_df,
            turnover=turnover_series,
            transaction_costs=tc_series,
            net_returns=net_returns,
            metrics=metrics,
        )

    def _get_regime_weights(
        self,
        probs_row: pd.Series,
        prob_cols: List[str],
        regime_portfolios: Dict[int, PortfolioResult],
        blend: bool,
    ) -> np.ndarray:
        """Get portfolio weights based on regime probabilities."""
        if blend:
            # Blend weights by probabilities
            n_assets = len(regime_portfolios[0].weights)
            weights = np.zeros(n_assets)
            for col in prob_cols:
                k = int(col.split("_")[1])
                if k in regime_portfolios:
                    weights += probs_row[col] * regime_portfolios[k].weights
            return weights / weights.sum()
        else:
            # Use dominant regime
            dominant = probs_row[prob_cols].idxmax()
            k = int(dominant.split("_")[1])
            return regime_portfolios[k].weights.copy()

    def _compute_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Compute performance metrics. `returns` are LOG returns."""
        ann_factor = 252

        total_return = np.exp(returns.sum()) - 1
        ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(ann_factor)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(ann_factor) if len(downside_returns) > 0 else 0.0
        sortino = ann_return / downside_std if downside_std > 0 else 0.0

        # Maximum drawdown
        cumulative = np.exp(returns.cumsum())
        rolling_max = cumulative.expanding().max()
        drawdown = cumulative / rolling_max - 1
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        # Skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # VaR and CVaR (5%)
        var_5 = returns.quantile(0.05)
        cvar_5 = returns[returns <= var_5].mean()

        return {
            "total_return": float(total_return),
            "ann_return": float(ann_return),
            "ann_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "var_5pct": float(var_5),
            "cvar_5pct": float(cvar_5),
            "n_days": int(len(returns)),
        }


# =============================================================================
# RISK CONTRIBUTION ANALYSIS
# =============================================================================

def compute_risk_contributions(
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """
    Compute marginal risk contributions for each asset.

    Risk contribution of asset i: RC_i = w_i * (Σw)_i / σ_p

    Parameters
    ----------
    weights : array
        Portfolio weights
    cov : array
        Covariance matrix

    Returns
    -------
    array : risk contributions (sum to portfolio volatility)
    """
    port_var = weights @ cov @ weights
    port_vol = np.sqrt(port_var)

    # Marginal contribution to risk
    mcr = cov @ weights / port_vol

    # Risk contribution
    rc = weights * mcr

    return rc


def compute_risk_contribution_pct(
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """
    Compute percentage risk contributions (sum to 100%).

    Parameters
    ----------
    weights : array
        Portfolio weights
    cov : array
        Covariance matrix

    Returns
    -------
    array : percentage risk contributions
    """
    rc = compute_risk_contributions(weights, cov)
    return rc / rc.sum()


def risk_contribution_stability(
    weights_history: pd.DataFrame,
    returns: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """
    Compute rolling risk contributions over time.

    Parameters
    ----------
    weights_history : DataFrame
        Portfolio weights over time
    returns : DataFrame
        Asset returns
    window : int
        Rolling window for covariance estimation (default: 63 = quarterly)

    Returns
    -------
    DataFrame : risk contributions over time
    """
    dates = weights_history.index
    rc_history = []

    for t in range(window, len(dates)):
        date = dates[t]
        w = weights_history.loc[date].values

        # Rolling covariance
        ret_window = returns.iloc[t-window:t]
        cov = ret_window.cov().values

        rc_pct = compute_risk_contribution_pct(w, cov)
        rc_history.append(rc_pct)

    rc_df = pd.DataFrame(
        rc_history,
        index=dates[window:],
        columns=returns.columns,
    )

    return rc_df


# =============================================================================
# COMPARISON AND HYPOTHESIS TESTING
# =============================================================================

def compare_portfolios(
    results: Dict[str, BacktestResult],
    returns_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare multiple portfolio strategies.

    Parameters
    ----------
    results : dict
        {strategy_name: BacktestResult}
    returns_df : DataFrame
        Original asset returns (for risk contribution analysis)

    Returns
    -------
    DataFrame : comparison table
    """
    comparison = []

    for name, result in results.items():
        metrics = result.metrics.copy()
        metrics["strategy"] = name

        # Add turnover statistics
        metrics["avg_turnover"] = float(result.turnover.mean())
        metrics["total_turnover"] = float(result.turnover.sum())
        metrics["total_tc"] = float(result.transaction_costs.sum())

        comparison.append(metrics)

    df = pd.DataFrame(comparison)
    df = df.set_index("strategy")

    return df


def test_sharpe_difference(
    returns1: pd.Series,
    returns2: pd.Series,
    method: str = "bootstrap",
    n_bootstrap: int = 10000,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Test whether two strategies have significantly different Sharpe ratios.

    Uses bootstrap method (Ledoit & Wolf, 2008 style).

    Parameters
    ----------
    returns1 : Series
        Returns of strategy 1
    returns2 : Series
        Returns of strategy 2
    method : str
        "bootstrap" (default)
    n_bootstrap : int
        Number of bootstrap samples
    random_state : int, optional
        Random seed

    Returns
    -------
    dict with:
        - sharpe_1, sharpe_2: Sharpe ratios
        - diff: difference (sharpe_1 - sharpe_2)
        - p_value: two-sided p-value
        - ci_lower, ci_upper: 95% confidence interval for difference
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Align series
    common = returns1.index.intersection(returns2.index)
    r1 = returns1.loc[common].values
    r2 = returns2.loc[common].values
    n = len(r1)

    def sharpe(r):
        return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0

    sharpe1 = sharpe(r1)
    sharpe2 = sharpe(r2)
    diff_obs = sharpe1 - sharpe2

    # Bootstrap
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        s1 = sharpe(r1[idx])
        s2 = sharpe(r2[idx])
        diffs.append(s1 - s2)

    diffs = np.array(diffs)

    # Two-sided p-value (proportion of bootstrap diffs on opposite side of 0)
    if diff_obs >= 0:
        p_value = 2 * (diffs <= 0).mean()
    else:
        p_value = 2 * (diffs >= 0).mean()

    p_value = min(p_value, 1.0)

    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)

    return {
        "sharpe_1": float(sharpe1),
        "sharpe_2": float(sharpe2),
        "diff": float(diff_obs),
        "p_value": float(p_value),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "significant_5pct": p_value < 0.05,
    }


def sensitivity_to_transaction_costs(
    returns: pd.DataFrame,
    weights: np.ndarray,
    rebalance_freq: str = "M",
    tc_range: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Analyze sensitivity of portfolio performance to transaction costs.

    Parameters
    ----------
    returns : DataFrame
        Asset returns
    weights : array
        Portfolio weights
    rebalance_freq : str
        Rebalancing frequency
    tc_range : list, optional
        Transaction cost levels to test (default: 1-10 bps)

    Returns
    -------
    DataFrame : performance metrics for each tc level
    """
    if tc_range is None:
        tc_range = [0.0001 * i for i in range(1, 11)]  # 1-10 bps

    results = []

    for tc in tc_range:
        backtester = PortfolioBacktester(
            rebalance_freq=rebalance_freq,
            transaction_cost=tc,
        )
        result = backtester.backtest_static(returns, weights)

        row = result.metrics.copy()
        row["tc_bps"] = tc * 10000
        results.append(row)

    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_portfolio_comparison(
    results: Dict[str, BacktestResult],
    save_as: Optional[str] = None,
):
    """
    Plot cumulative returns comparison.

    Parameters
    ----------
    results : dict
        {strategy_name: BacktestResult}
    save_as : str, optional
        Filename to save plot
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Cumulative returns
    ax1 = axes[0, 0]
    for name, result in results.items():
        cum_ret = np.exp(result.net_returns.cumsum())
        ax1.plot(cum_ret.index, cum_ret.values, label=name, linewidth=1.5)
    ax1.set_title("Cumulative Returns (net of TC)")
    ax1.set_ylabel("Growth of $1")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    # 2. Drawdowns
    ax2 = axes[0, 1]
    for name, result in results.items():
        cum_ret = np.exp(result.net_returns.cumsum())
        rolling_max = cum_ret.expanding().max()
        drawdown = cum_ret / rolling_max - 1
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, label=name)
    ax2.set_title("Drawdowns")
    ax2.set_ylabel("Drawdown")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    # 3. Rolling Sharpe (252-day)
    ax3 = axes[1, 0]
    for name, result in results.items():
        rolling_sharpe = (
            result.net_returns.rolling(252).mean() /
            result.net_returns.rolling(252).std() * np.sqrt(252)
        )
        ax3.plot(rolling_sharpe.index, rolling_sharpe.values, label=name, linewidth=1)
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax3.set_title("Rolling 252-day Sharpe Ratio")
    ax3.set_ylabel("Sharpe Ratio")
    ax3.legend(loc="upper left")
    ax3.grid(alpha=0.3)

    # 4. Turnover
    ax4 = axes[1, 1]
    for name, result in results.items():
        # Monthly aggregated turnover
        monthly_to = result.turnover.resample("ME").sum()
        ax4.plot(monthly_to.index, monthly_to.values, label=name, linewidth=1)
    ax4.set_title("Monthly Turnover")
    ax4.set_ylabel("Turnover")
    ax4.legend(loc="upper right")
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    if save_as:
        OUTPUT_DIR.mkdir(exist_ok=True)
        plt.savefig(OUTPUT_DIR / save_as, dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / save_as}")

    plt.close()


def plot_risk_contributions(
    rc_df: pd.DataFrame,
    save_as: Optional[str] = None,
):
    """
    Plot risk contributions over time.

    Parameters
    ----------
    rc_df : DataFrame
        Risk contributions over time (from risk_contribution_stability)
    save_as : str, optional
        Filename to save plot
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Stacked area chart
    ax1 = axes[0]
    ax1.stackplot(rc_df.index, rc_df.T.values, labels=rc_df.columns, alpha=0.8)
    ax1.set_title("Risk Contributions Over Time")
    ax1.set_ylabel("Percentage")
    ax1.legend(loc="upper left", fontsize=8, ncol=min(5, len(rc_df.columns)))
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    # Stability metrics
    ax2 = axes[1]
    rc_std = rc_df.std()
    ax2.bar(range(len(rc_std)), rc_std.values)
    ax2.set_xticks(range(len(rc_std)))
    ax2.set_xticklabels(rc_std.index, rotation=45, ha="right")
    ax2.set_title("Risk Contribution Volatility (stability measure)")
    ax2.set_ylabel("Std Dev of Risk Contribution")
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_as:
        OUTPUT_DIR.mkdir(exist_ok=True)
        plt.savefig(OUTPUT_DIR / save_as, dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / save_as}")

    plt.close()


def plot_efficient_frontiers(
    regime_means: Dict[int, np.ndarray],
    regime_covs: Dict[int, np.ndarray],
    unconditional_mu: np.ndarray,
    unconditional_cov: np.ndarray,
    save_as: Optional[str] = None,
):
    """
    Plot efficient frontiers for each regime and unconditional.

    Parameters
    ----------
    regime_means : dict
        {regime_id: mean_vector}
    regime_covs : dict
        {regime_id: covariance_matrix}
    unconditional_mu : array
        Unconditional mean returns
    unconditional_cov : array
        Unconditional covariance matrix
    save_as : str, optional
        Filename to save plot
    """
    import matplotlib.pyplot as plt

    optimizer = RegimePortfolioOptimizer()

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.tab10.colors

    # Plot unconditional frontier
    ef_uncond = optimizer.compute_efficient_frontier(unconditional_mu, unconditional_cov)
    ax.plot(
        ef_uncond["volatility"] * np.sqrt(252) * 100,
        ef_uncond["target_return"] * 252 * 100,
        "k--",
        linewidth=2,
        label="Unconditional",
    )

    # Plot regime-specific frontiers
    for i, (k, mu) in enumerate(regime_means.items()):
        cov = regime_covs[k]
        ef = optimizer.compute_efficient_frontier(mu, cov)
        ax.plot(
            ef["volatility"] * np.sqrt(252) * 100,
            ef["target_return"] * 252 * 100,
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"Regime {k}",
        )

    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title("Efficient Frontiers by Regime")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_as:
        OUTPUT_DIR.mkdir(exist_ok=True)
        plt.savefig(OUTPUT_DIR / save_as, dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / save_as}")

    plt.close()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_regime_moments_from_optimiser(opt, feature_names: Optional[List[str]] = None):
    """
    Extract regime-specific means and covariances from a fitted Optimiser.

    Uses the last window's GMM parameters.

    Parameters
    ----------
    opt : Optimiser
        Fitted GMM optimiser
    feature_names : list, optional
        Asset names

    Returns
    -------
    tuple : (regime_means, regime_covs, long_run_probs)
        regime_means: {regime_id: mean_vector}
        regime_covs: {regime_id: covariance_matrix}
        long_run_probs: {regime_id: probability}
    """
    # Get last window's GMM (fits is a list of dicts)
    last_fit = opt.fits[-1]

    K = last_fit["K"]

    regime_means = {}
    regime_covs = {}
    long_run_probs = {}

    for k in range(K):
        regime_means[k] = last_fit["means"][k]
        regime_covs[k] = last_fit["covariances"][k]
        long_run_probs[k] = last_fit["weights"][k]

    return regime_means, regime_covs, long_run_probs


def extract_pooled_regime_moments(opt, X_raw: np.ndarray):
    """
    Extract pooled regime moments across all windows.

    Averages regime-specific means and covariances across windows,
    weighted by number of observations.

    Parameters
    ----------
    opt : Optimiser
        Fitted GMM optimiser
    X_raw : array
        Original return data

    Returns
    -------
    tuple : (regime_means, regime_covs, long_run_probs)
    """
    from collections import defaultdict

    # Collect moments from all windows (fits is a list of dicts)
    K = opt.fits[-1]["K"]

    means_sum = defaultdict(lambda: np.zeros_like(opt.fits[0]["means"][0]))
    covs_sum = defaultdict(lambda: np.zeros_like(opt.fits[0]["covariances"][0]))
    weights_sum = defaultdict(float)
    total_weight = 0.0

    for fit in opt.fits:
        n_obs = 1  # equal weight per window
        fit_K = fit["K"]
        for k in range(min(K, fit_K)):
            means_sum[k] += fit["means"][k] * n_obs
            covs_sum[k] += fit["covariances"][k] * n_obs
            weights_sum[k] += fit["weights"][k] * n_obs
        total_weight += n_obs

    regime_means = {k: means_sum[k] / total_weight for k in range(K)}
    regime_covs = {k: covs_sum[k] / total_weight for k in range(K)}

    total_weights = sum(weights_sum.values())
    long_run_probs = {k: weights_sum[k] / total_weights for k in range(K)}

    return regime_means, regime_covs, long_run_probs


# =============================================================================
# RISK-ADJUSTED ANALYSIS (for reframing H3 around risk characteristics)
# =============================================================================

def compute_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Compute historical Value-at-Risk (VaR).

    Parameters
    ----------
    returns : Series
        Portfolio returns
    confidence : float
        Confidence level (default: 95%)

    Returns
    -------
    float : VaR (positive number representing loss)
    """
    return -np.percentile(returns.dropna(), (1 - confidence) * 100)


def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Compute Conditional VaR (Expected Shortfall).

    Parameters
    ----------
    returns : Series
        Portfolio returns
    confidence : float
        Confidence level (default: 95%)

    Returns
    -------
    float : CVaR (positive number representing expected loss)
    """
    var = compute_var(returns, confidence)
    return -returns[returns <= -var].mean()


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdown series from LOG returns."""
    cum_ret = np.exp(returns.cumsum())
    rolling_max = cum_ret.expanding().max()
    drawdown = cum_ret / rolling_max - 1
    return drawdown


def compute_max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown."""
    drawdown = compute_drawdown_series(returns)
    return -drawdown.min()


def compute_avg_drawdown(returns: pd.Series) -> float:
    """Compute average drawdown (when in drawdown)."""
    drawdown = compute_drawdown_series(returns)
    in_drawdown = drawdown[drawdown < 0]
    return -in_drawdown.mean() if len(in_drawdown) > 0 else 0.0


def compute_max_drawdown_duration(returns: pd.Series) -> int:
    """
    Compute maximum drawdown duration (days to recovery).

    Returns
    -------
    int : Number of days in longest drawdown
    """
    cum_ret = np.exp(returns.cumsum())
    rolling_max = cum_ret.expanding().max()

    # Find periods where we're below the peak
    in_drawdown = cum_ret < rolling_max

    # Count consecutive drawdown days
    durations = []
    current_duration = 0

    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
            current_duration = 0

    # Don't forget the last period if still in drawdown
    if current_duration > 0:
        durations.append(current_duration)

    return max(durations) if durations else 0


def compute_calmar_ratio(returns: pd.Series) -> float:
    """
    Compute Calmar ratio (annualized return / max drawdown).

    Returns
    -------
    float : Calmar ratio
    """
    ann_return = returns.mean() * 252
    max_dd = compute_max_drawdown(returns)
    return ann_return / max_dd if max_dd > 0 else np.nan


def compute_ulcer_index(returns: pd.Series) -> float:
    """
    Compute Ulcer Index (RMS of drawdowns).

    Measures downside risk accounting for both depth and duration.

    Returns
    -------
    float : Ulcer Index
    """
    drawdown = compute_drawdown_series(returns)
    return np.sqrt((drawdown ** 2).mean())


def compute_pain_index(returns: pd.Series) -> float:
    """
    Compute Pain Index (mean of absolute drawdowns).

    Returns
    -------
    float : Pain Index
    """
    drawdown = compute_drawdown_series(returns)
    return -drawdown.mean()


def compute_downside_deviation(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Compute downside deviation.

    Parameters
    ----------
    returns : Series
        Portfolio returns
    threshold : float
        Target return (default: 0)

    Returns
    -------
    float : Annualized downside deviation
    """
    downside = returns[returns < threshold] - threshold
    return np.sqrt((downside ** 2).mean()) * np.sqrt(252)


def compute_risk_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Compute comprehensive risk metrics.

    Parameters
    ----------
    returns : Series
        Portfolio returns

    Returns
    -------
    dict : Risk metrics
    """
    returns = returns.dropna()

    if len(returns) < 30:
        return {key: np.nan for key in [
            "var_95", "var_99", "cvar_95", "cvar_99",
            "max_drawdown", "avg_drawdown", "max_dd_duration",
            "calmar_ratio", "ulcer_index", "pain_index",
            "downside_dev", "skewness", "kurtosis"
        ]}

    return {
        "var_95": compute_var(returns, 0.95),
        "var_99": compute_var(returns, 0.99),
        "cvar_95": compute_cvar(returns, 0.95),
        "cvar_99": compute_cvar(returns, 0.99),
        "max_drawdown": compute_max_drawdown(returns),
        "avg_drawdown": compute_avg_drawdown(returns),
        "max_dd_duration": compute_max_drawdown_duration(returns),
        "calmar_ratio": compute_calmar_ratio(returns),
        "ulcer_index": compute_ulcer_index(returns),
        "pain_index": compute_pain_index(returns),
        "downside_dev": compute_downside_deviation(returns),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
    }


def compare_risk_profiles(
    results: Dict[str, BacktestResult],
    returns_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare risk profiles across strategies.

    Provides risk-focused comparison (H3 reframed around risk characteristics).

    Parameters
    ----------
    results : dict
        {strategy_name: BacktestResult}
    returns_df : DataFrame
        Original returns for benchmark

    Returns
    -------
    DataFrame : Risk comparison with significance indicators
    """
    rows = []

    for name, result in results.items():
        risk_metrics = compute_risk_metrics(result.net_returns)

        # Standard performance metrics
        ann_ret = result.net_returns.mean() * 252
        ann_vol = result.net_returns.std() * np.sqrt(252)

        row = {
            "strategy": name,
            "ann_return": ann_ret,
            "ann_volatility": ann_vol,
            "sharpe_ratio": ann_ret / ann_vol if ann_vol > 0 else np.nan,
            "sortino_ratio": ann_ret / risk_metrics["downside_dev"] if risk_metrics["downside_dev"] > 0 else np.nan,
            **risk_metrics,
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("strategy")

    # Add ranks (lower is better for risk metrics, higher for return metrics)
    risk_cols = ["var_95", "var_99", "cvar_95", "cvar_99", "max_drawdown",
                 "avg_drawdown", "max_dd_duration", "ulcer_index", "pain_index", "downside_dev"]
    return_cols = ["ann_return", "sharpe_ratio", "sortino_ratio", "calmar_ratio"]

    for col in risk_cols:
        if col in df.columns:
            df[f"{col}_rank"] = df[col].rank(ascending=True)  # lower is better

    for col in return_cols:
        if col in df.columns:
            df[f"{col}_rank"] = df[col].rank(ascending=False)  # higher is better

    return df


def test_drawdown_difference(
    returns_1: pd.Series,
    returns_2: pd.Series,
    n_bootstrap: int = 10000,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap test for max drawdown difference.

    Parameters
    ----------
    returns_1 : Series
        Returns from strategy 1
    returns_2 : Series
        Returns from strategy 2
    n_bootstrap : int
        Number of bootstrap samples
    random_state : int
        Random seed

    Returns
    -------
    dict : Test results including p-value
    """
    rng = np.random.RandomState(random_state)

    # Align series
    common_idx = returns_1.dropna().index.intersection(returns_2.dropna().index)
    r1 = returns_1.loc[common_idx].values
    r2 = returns_2.loc[common_idx].values

    n = len(r1)

    dd1_obs = compute_max_drawdown(pd.Series(r1))
    dd2_obs = compute_max_drawdown(pd.Series(r2))
    diff_obs = dd1_obs - dd2_obs

    # Block bootstrap (block size ~ sqrt(n))
    block_size = max(5, int(np.sqrt(n)))
    n_blocks = n // block_size

    boot_diffs = []
    for _ in range(n_bootstrap):
        # Sample blocks with replacement
        block_starts = rng.randint(0, n - block_size + 1, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_size) for s in block_starts])[:n]

        r1_boot = r1[indices]
        r2_boot = r2[indices]

        dd1_boot = compute_max_drawdown(pd.Series(r1_boot))
        dd2_boot = compute_max_drawdown(pd.Series(r2_boot))
        boot_diffs.append(dd1_boot - dd2_boot)

    boot_diffs = np.array(boot_diffs)

    # Two-sided p-value
    p_value = 2 * min(
        (boot_diffs >= abs(diff_obs)).mean(),
        (boot_diffs <= -abs(diff_obs)).mean()
    )
    p_value = min(1.0, max(0.0, p_value))

    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)

    return {
        "max_dd_1": dd1_obs,
        "max_dd_2": dd2_obs,
        "diff": diff_obs,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "strategy_1_better": diff_obs < 0,  # Lower drawdown is better
        "significant_5pct": p_value < 0.05,
    }


def test_cvar_difference(
    returns_1: pd.Series,
    returns_2: pd.Series,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap test for CVaR difference.

    Parameters
    ----------
    returns_1 : Series
        Returns from strategy 1
    returns_2 : Series
        Returns from strategy 2
    confidence : float
        VaR confidence level
    n_bootstrap : int
        Number of bootstrap samples
    random_state : int
        Random seed

    Returns
    -------
    dict : Test results including p-value
    """
    rng = np.random.RandomState(random_state)

    common_idx = returns_1.dropna().index.intersection(returns_2.dropna().index)
    r1 = returns_1.loc[common_idx].values
    r2 = returns_2.loc[common_idx].values

    n = len(r1)

    cvar1_obs = compute_cvar(pd.Series(r1), confidence)
    cvar2_obs = compute_cvar(pd.Series(r2), confidence)
    diff_obs = cvar1_obs - cvar2_obs

    boot_diffs = []
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        r1_boot = r1[indices]
        r2_boot = r2[indices]

        cvar1_boot = compute_cvar(pd.Series(r1_boot), confidence)
        cvar2_boot = compute_cvar(pd.Series(r2_boot), confidence)
        boot_diffs.append(cvar1_boot - cvar2_boot)

    boot_diffs = np.array(boot_diffs)

    p_value = 2 * min(
        (boot_diffs >= abs(diff_obs)).mean(),
        (boot_diffs <= -abs(diff_obs)).mean()
    )
    p_value = min(1.0, max(0.0, p_value))

    return {
        "cvar_1": cvar1_obs,
        "cvar_2": cvar2_obs,
        "diff": diff_obs,
        "p_value": p_value,
        "ci_lower": np.percentile(boot_diffs, 2.5),
        "ci_upper": np.percentile(boot_diffs, 97.5),
        "strategy_1_better": diff_obs < 0,  # Lower CVaR is better
        "significant_5pct": p_value < 0.05,
    }


# =============================================================================
# OUT-OF-SAMPLE EXPANDING WINDOW TESTING
# =============================================================================

def expanding_window_backtest(
    returns: pd.DataFrame,
    regime_probs: pd.DataFrame,
    regime_portfolios: Dict[int, PortfolioResult],
    optimizer: RegimePortfolioOptimizer,
    initial_train_period: int = 1250,
    reoptimize_frequency: int = 63,
    transaction_cost: float = 0.0005,
    blend_weights: bool = True,
) -> Dict[str, BacktestResult]:
    """
    Out-of-sample expanding window backtest.

    Avoids look-ahead bias by re-estimating portfolios using only
    data available up to each rebalancing point.

    Parameters
    ----------
    returns : DataFrame
        Asset returns
    regime_probs : DataFrame
        Daily regime probabilities
    regime_portfolios : dict
        Initial regime portfolios (will be re-estimated)
    optimizer : RegimePortfolioOptimizer
        Portfolio optimizer
    initial_train_period : int
        Initial training period (days)
    reoptimize_frequency : int
        Re-estimate portfolios every N days
    transaction_cost : float
        Transaction cost rate
    blend_weights : bool
        If True, blend regime portfolios by probability

    Returns
    -------
    dict : {strategy_name: BacktestResult}
    """
    dates = returns.index
    n_assets = returns.shape[1]

    if len(dates) <= initial_train_period:
        raise ValueError("Insufficient data for expanding window backtest")

    # Initialize tracking
    all_returns = {"unconditional": [], "regime_aware": [], "strategic": []}
    all_weights = {"unconditional": [], "regime_aware": [], "strategic": []}
    all_dates = []

    prev_weights = {
        "unconditional": np.ones(n_assets) / n_assets,
        "regime_aware": np.ones(n_assets) / n_assets,
        "strategic": np.ones(n_assets) / n_assets,
    }

    # Expanding window loop
    for t in range(initial_train_period, len(dates)):
        current_date = dates[t]

        # Check if we need to re-optimize
        if (t - initial_train_period) % reoptimize_frequency == 0:
            # Use only data up to t-1 (no look-ahead)
            train_returns = returns.iloc[:t]

            # Re-estimate unconditional portfolio
            try:
                uncond_port = optimizer.max_sharpe(
                    train_returns.mean().values,
                    train_returns.cov().values
                )
                prev_weights["unconditional"] = uncond_port.weights
            except Exception:
                pass  # Keep previous weights

            # Re-estimate regime portfolios using training data
            # (In practice, you'd re-fit the GMM here; for simplicity, we use regime probs)
            try:
                train_regime_probs = regime_probs.loc[:dates[t-1]]

                # Estimate regime means/covs from training data
                for k in regime_portfolios.keys():
                    prob_col = f"p_{k}"
                    if prob_col in train_regime_probs.columns:
                        regime_weight = train_regime_probs[prob_col].values[:, None]
                        weighted_returns = train_returns.values * regime_weight[:len(train_returns)]

                        mu_k = np.average(train_returns.values, weights=regime_weight[:len(train_returns)].flatten(), axis=0)
                        # Weighted covariance
                        centered = train_returns.values - mu_k
                        cov_k = np.cov(centered.T, aweights=regime_weight[:len(train_returns)].flatten())

                        port_k = optimizer.max_sharpe(mu_k, cov_k)
                        regime_portfolios[k] = port_k
            except Exception:
                pass

            # Strategic portfolio (blend of regime portfolios by long-run probs)
            train_probs = regime_probs.loc[:dates[t-1]]
            long_run = {k: train_probs[f"p_{k}"].mean() for k in regime_portfolios.keys()}
            strategic_w = np.zeros(n_assets)
            for k, port in regime_portfolios.items():
                strategic_w += long_run[k] * port.weights
            prev_weights["strategic"] = strategic_w / strategic_w.sum()

        # Get current regime probabilities for regime-aware portfolio
        if current_date in regime_probs.index:
            current_probs = regime_probs.loc[current_date]
            if blend_weights:
                regime_w = np.zeros(n_assets)
                for k, port in regime_portfolios.items():
                    prob_k = current_probs.get(f"p_{k}", 0)
                    regime_w += prob_k * port.weights
                prev_weights["regime_aware"] = regime_w / regime_w.sum() if regime_w.sum() > 0 else prev_weights["regime_aware"]
            else:
                # Hard switch to dominant regime
                dom_k = int(current_probs[[f"p_{k}" for k in regime_portfolios.keys()]].idxmax().split("_")[1])
                prev_weights["regime_aware"] = regime_portfolios[dom_k].weights

        # Record returns (log; exact portfolio log return from per-asset log returns)
        day_return = returns.iloc[t].values
        gross_factors = np.exp(day_return)
        for strategy in all_returns.keys():
            w = prev_weights[strategy]
            port_return = np.log(np.dot(w, gross_factors) / w.sum())
            all_returns[strategy].append(port_return)
            all_weights[strategy].append(w.copy())

        all_dates.append(current_date)

    # Build BacktestResults
    results = {}
    for strategy in all_returns.keys():
        ret_series = pd.Series(all_returns[strategy], index=all_dates)
        weights_df = pd.DataFrame(all_weights[strategy], index=all_dates, columns=returns.columns)

        # Compute turnover
        turnover = weights_df.diff().abs().sum(axis=1)
        turnover.iloc[0] = 0

        # Transaction costs
        tc_series = turnover * transaction_cost
        net_returns = ret_series - tc_series

        # Compute metrics
        ann_ret = net_returns.mean() * 252
        ann_vol = net_returns.std() * np.sqrt(252)

        metrics = {
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": ann_ret / ann_vol if ann_vol > 0 else np.nan,
            "max_drawdown": compute_max_drawdown(net_returns),
            "calmar_ratio": compute_calmar_ratio(net_returns),
            "avg_turnover": turnover.mean(),
            "total_tc": tc_series.sum(),
        }

        results[strategy] = BacktestResult(
            returns=ret_series,
            weights_history=weights_df,
            turnover=turnover,
            transaction_costs=tc_series,
            net_returns=net_returns,
            metrics=metrics,
        )

    return results


def compare_oos_performance(
    results: Dict[str, BacktestResult],
) -> pd.DataFrame:
    """
    Compare out-of-sample performance across strategies.

    Parameters
    ----------
    results : dict
        {strategy_name: BacktestResult}

    Returns
    -------
    DataFrame : Performance comparison
    """
    rows = []
    for name, result in results.items():
        risk_metrics = compute_risk_metrics(result.net_returns)

        row = {
            "strategy": name,
            "ann_return": result.metrics.get("annualized_return", result.net_returns.mean() * 252),
            "ann_volatility": result.metrics.get("annualized_volatility", result.net_returns.std() * np.sqrt(252)),
            "sharpe_ratio": result.metrics.get("sharpe_ratio", np.nan),
            "max_drawdown": risk_metrics["max_drawdown"],
            "calmar_ratio": risk_metrics["calmar_ratio"],
            "cvar_95": risk_metrics["cvar_95"],
            "sortino_ratio": result.net_returns.mean() * 252 / risk_metrics["downside_dev"] if risk_metrics["downside_dev"] > 0 else np.nan,
            "avg_turnover": result.metrics.get("avg_turnover", result.turnover.mean()),
            "total_tc": result.metrics.get("total_tc", result.transaction_costs.sum()),
        }
        rows.append(row)

    return pd.DataFrame(rows).set_index("strategy")


def generate_h3_risk_report(
    results: Dict[str, BacktestResult],
    returns_df: pd.DataFrame,
    n_bootstrap: int = 5000,
) -> Dict[str, pd.DataFrame]:
    """
    Generate comprehensive H3 report focused on risk characteristics.

    Parameters
    ----------
    results : dict
        {strategy_name: BacktestResult}
    returns_df : DataFrame
        Original returns
    n_bootstrap : int
        Bootstrap samples for significance tests

    Returns
    -------
    dict : {
        "performance": Overall performance comparison,
        "risk_metrics": Detailed risk metrics,
        "sharpe_tests": Sharpe ratio difference tests,
        "drawdown_tests": Max drawdown difference tests,
        "cvar_tests": CVaR difference tests,
        "summary": Text summary of findings
    }
    """
    # Performance comparison
    performance_df = compare_risk_profiles(results, returns_df)

    # Risk metrics detail
    risk_metrics = {}
    for name, result in results.items():
        risk_metrics[name] = compute_risk_metrics(result.net_returns)
    risk_df = pd.DataFrame(risk_metrics).T

    # Statistical tests (compare to unconditional as baseline)
    if "Unconditional" not in results and "unconditional" not in results:
        baseline_name = list(results.keys())[0]
    else:
        baseline_name = "Unconditional" if "Unconditional" in results else "unconditional"

    baseline_returns = results[baseline_name].net_returns

    sharpe_tests = []
    drawdown_tests = []
    cvar_tests = []

    for name, result in results.items():
        if name == baseline_name:
            continue

        # Sharpe test
        sharpe_test = test_sharpe_difference(
            result.net_returns, baseline_returns,
            n_bootstrap=n_bootstrap
        )
        sharpe_test["comparison"] = f"{name} vs {baseline_name}"
        sharpe_tests.append(sharpe_test)

        # Drawdown test
        dd_test = test_drawdown_difference(
            result.net_returns, baseline_returns,
            n_bootstrap=n_bootstrap
        )
        dd_test["comparison"] = f"{name} vs {baseline_name}"
        drawdown_tests.append(dd_test)

        # CVaR test
        cvar_test = test_cvar_difference(
            result.net_returns, baseline_returns,
            n_bootstrap=n_bootstrap
        )
        cvar_test["comparison"] = f"{name} vs {baseline_name}"
        cvar_tests.append(cvar_test)

    sharpe_df = pd.DataFrame(sharpe_tests) if sharpe_tests else pd.DataFrame()
    drawdown_df = pd.DataFrame(drawdown_tests) if drawdown_tests else pd.DataFrame()
    cvar_df = pd.DataFrame(cvar_tests) if cvar_tests else pd.DataFrame()

    # Generate summary
    summary_lines = [
        "H3 Risk Analysis Summary",
        "=" * 50,
        "",
        "Key Findings:",
    ]

    # Best Sharpe
    best_sharpe = performance_df["sharpe_ratio"].idxmax()
    summary_lines.append(f"  - Best Sharpe Ratio: {best_sharpe} ({performance_df.loc[best_sharpe, 'sharpe_ratio']:.4f})")

    # Lowest Max Drawdown
    best_dd = performance_df["max_drawdown"].idxmin()
    summary_lines.append(f"  - Lowest Max Drawdown: {best_dd} ({performance_df.loc[best_dd, 'max_drawdown']:.2%})")

    # Best Calmar
    best_calmar = performance_df["calmar_ratio"].idxmax()
    summary_lines.append(f"  - Best Calmar Ratio: {best_calmar} ({performance_df.loc[best_calmar, 'calmar_ratio']:.4f})")

    # Lowest CVaR
    best_cvar = performance_df["cvar_95"].idxmin()
    summary_lines.append(f"  - Lowest CVaR (95%): {best_cvar} ({performance_df.loc[best_cvar, 'cvar_95']:.4%})")

    summary_lines.append("")
    summary_lines.append("Statistical Significance (vs Unconditional):")

    for _, row in sharpe_df.iterrows():
        sig = "***" if row["p_value"] < 0.01 else "**" if row["p_value"] < 0.05 else "*" if row["p_value"] < 0.10 else ""
        summary_lines.append(f"  - {row['comparison']}: Sharpe diff = {row['diff']:.4f}, p = {row['p_value']:.4f} {sig}")

    for _, row in drawdown_df.iterrows():
        sig = "***" if row["p_value"] < 0.01 else "**" if row["p_value"] < 0.05 else "*" if row["p_value"] < 0.10 else ""
        summary_lines.append(f"  - {row['comparison']}: DD diff = {row['diff']:.4%}, p = {row['p_value']:.4f} {sig}")

    summary = "\n".join(summary_lines)

    return {
        "performance": performance_df,
        "risk_metrics": risk_df,
        "sharpe_tests": sharpe_df,
        "drawdown_tests": drawdown_df,
        "cvar_tests": cvar_df,
        "summary": summary,
    }
