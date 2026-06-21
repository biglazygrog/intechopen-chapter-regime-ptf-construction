import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


MASTER_PATH = Path(__file__).parent / "data_master_v1.csv"
RAW_PATH  = Path(__file__).parent / "raw_data.xlsx"

MASTER_PATH_2 = Path(__file__).parent / "data_master2.csv"
RAW_PATH_2 = Path(__file__).parent / "raw_data2.xlsx"

RAW_PATH_EUR = Path(__file__).parent / "raw_data_eur.xlsx"
RAW_PATH_2_EUR = Path(__file__).parent / "raw_data2_eur.xlsx"

class DataReader:
    """
    Reads:
      - data_master.xlsx (columns: bbg_ticker, name)
      - raw_data.xlsx    (columns: 'date', plus one column per ticker)

    Replaces tickers (column headers) with names from master, and provides:
        read_retns(frequency='daily'|'weekly'|'monthly', start=None, end=None)
    """

    def __init__(self, version: int = 1, use_eur: bool = False):
        """
        version: 1 = original 6 assets, 2 = extended 13 assets
        use_eur: if True, use EUR return data files (raw_data_eur.xlsx, raw_data2_eur.xlsx)
        """
        if version == 2:
            self.master = pd.read_csv(MASTER_PATH_2)
            self.raw = pd.read_excel(RAW_PATH_2_EUR if use_eur else RAW_PATH_2)
        else:
            self.master = pd.read_csv(MASTER_PATH)
            self.raw = pd.read_excel(RAW_PATH_EUR if use_eur else RAW_PATH)

        self.prices = self._prepare_prices()

    def read_retns(self, frequency: str = "daily", start=None, end=None) -> pd.DataFrame:
        """
        Compute log returns at given frequency and date range.
        frequency: 'daily', 'weekly', or 'monthly'
        start/end: date-like or None
        """
        prices = self._filter_and_resample(self.prices, frequency, start, end)
        rets = np.log(prices).diff()
        return rets.dropna(how="all")

    def _prepare_prices(self) -> pd.DataFrame:
        df = self.raw.copy()

        if "date" not in df.columns:
            raise ValueError("raw_data.xlsx must have a 'date' column")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()

        # Drop unnamed/error columns
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

        if not {"bbg_ticker", "name"}.issubset(self.master.columns):
            raise ValueError("data_master.xlsx must contain 'bbg_ticker' and 'name' columns")

        mapping = self.master.set_index("bbg_ticker")["name"].to_dict()
        # Keep only columns that exist in master
        valid_tickers = set(mapping.keys())
        df = df[[c for c in df.columns if c in valid_tickers]]
        df = df.rename(columns=mapping)

        # Convert to numeric
        df = df.apply(pd.to_numeric, errors="coerce")

        return df

    def _filter_and_resample(self, df: pd.DataFrame, frequency: str, start, end) -> pd.DataFrame:

        if start is not None:
            df = df[df.index >= pd.to_datetime(start)]
        if end is not None:
            df = df[df.index <= pd.to_datetime(end)]

        freq_map = {"daily": None, "weekly": "W-FRI", "monthly": "M"}
        rule = freq_map.get(frequency.lower())

        if rule is None:
            return df.ffill()
        return df.resample(rule).last().ffill()


def summarize_returns(r_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize key statistics for a daily returns DataFrame.

    r_daily: pd.DataFrame
        - index: datetime-like (dates)
        - columns: asset identifiers
        - values: daily returns

    Returns: pd.DataFrame
        Summary table with statistics per asset
    """
    summary = pd.DataFrame({
        "start_date": r_daily.index.min(),
        "end_date": r_daily.index.max(),
        "n_obs": r_daily.count(),
        "mean": r_daily.mean(),
        "vol": r_daily.std(),
        "min": r_daily.min(),
        "max": r_daily.max(),
    })
    return summary


def check_normality_hist_qq(df, method="jarque_bera", bins=60, figsize_per_asset=(10, 4)):
    """
    For each asset (column) in df:
      - test normality
      - plot histogram + fitted normal PDF
      - plot Q–Q plot vs N(0,1) (after standardizing)

    Parameters
    ----------
    df : pd.DataFrame
        Date index (ignored here) and columns = asset returns.
    method : str
        'jarque_bera', 'shapiro', or 'dagostino'.
    bins : int
        Number of bins in the histogram.
    figsize_per_asset : tuple
        Size of figure *per asset* (width, height). Total fig size is
        (width, height * n_assets).

    Returns
    -------
    results : pd.DataFrame
        Columns: statistic, pvalue, mean, std, skew, kurtosis
    """
    assert isinstance(df, pd.DataFrame)
    n_assets = df.shape[1]

    # Create a grid: one row per asset, 2 columns (hist + qq)
    fig, axes = plt.subplots(
        n_assets,
        2,
        figsize=(figsize_per_asset[0], figsize_per_asset[1] * n_assets),
        squeeze=False,
    )

    results = {}

    for i, col in enumerate(df.columns):
        x = df[col].dropna().values
        mu = np.mean(x)
        sigma = np.std(x, ddof=1)
        skew = stats.skew(x, bias=False)
        kurt = stats.kurtosis(x, fisher=False, bias=False)  # 3 for normal

        # --- choose normality test ---
        m = method.lower()
        if m in ["jarque_bera", "jb"]:
            stat, p = stats.jarque_bera(x)
            test_name = "Jarque–Bera"
        elif m in ["shapiro", "sw"]:
            stat, p = stats.shapiro(x)
            test_name = "Shapiro–Wilk"
        elif m in ["dagostino", "dago"]:
            stat, p = stats.normaltest(x)
            test_name = "D'Agostino K²"
        else:
            raise ValueError("method must be 'jarque_bera', 'shapiro', or 'dagostino'.")

        results[col] = {
            "statistic": stat,
            "pvalue": p,
            "mean": mu,
            "std": sigma,
            "skew": skew,
            "kurtosis": kurt,
        }

        # ========= LEFT: histogram + fitted normal =========
        ax_hist = axes[i, 0]
        ax_hist.hist(x, bins=bins, density=True, alpha=0.6, edgecolor="white")

        x_grid = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
        pdf = stats.norm.pdf(x_grid, mu, sigma)
        ax_hist.plot(x_grid, pdf, linestyle="--", linewidth=2)

        ax_hist.set_title(
            f"{col}\nμ={mu:.4f}, σ={sigma:.4f}, skew={skew:.2f}, kurt={kurt:.2f}\n"
            f"{test_name} p={p:.3g}"
        )
        ax_hist.set_ylabel("Density")
        ax_hist.grid(alpha=0.3)

        # ========= RIGHT: Q–Q plot =========
        ax_qq = axes[i, 1]
        # standardize for QQ
        z = (x - mu) / sigma if sigma > 0 else x * 0.0
        stats.probplot(z, dist="norm", plot=ax_qq)
        ax_qq.get_lines()[0].set_marker(".")  # q-q points
        ax_qq.get_lines()[1].set_color("red")  # reference line
        ax_qq.set_title(f"{col} – Q–Q plot (standardized)")
        ax_qq.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(results).T

if __name__ == "__main__":
    dr = DataReader()

    r_daily = dr.read_retns()
    r_daily_sum = summarize_returns(r_daily)
    print(r_daily_sum)
    # r_daily.to_csv('r_daily.csv')

    # r_weekly = dr.read_retns(frequency="weekly", start="2000-01-01")
    # r_monthly = dr.read_retns(frequency="monthly", start="2010-01-01", end="2020-12-31")

    # print(r_monthly.tail())

    # results = check_normality_hist_qq(r_monthly, method="jarque_bera")
    # print(results)
