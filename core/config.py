"""
Centralised configuration for the regime-aware portfolio paper.

All headline-result numbers in the paper come from the constants below.
Changing them invalidates the reproduction.
"""
from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "research" / "output_charts"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Rolling-window GMM
# ---------------------------------------------------------------------
WINDOW_SIZE = 1250        # 5 years of business days
STEP = 63                 # ~quarterly refit
K_CANDIDATES = [2, 3, 4, 5, 6, 7]
SCALE_METHOD = "none"     # raw log returns into the GMM
MAX_ITER = 200
TOL = 1e-4
REG_COVAR = 1e-6
RANDOM_STATE = 0

# Normal-Wishart prior + previous-epoch shrinkage
SHRINKAGE_TARGET = "previous_epoch"
KAPPA_0 = 63.1            # target prior weight p ~ 10%
NU_0_EXTRA = 2
LAMBDA_SCALE = 1.0
ALPHA_DIRICHLET = 0.5
EWMA_DECAY = 0.97

# ---------------------------------------------------------------------
# Forecast-only K=2 model (for AR forecasting + backtest)
# ---------------------------------------------------------------------
FORECAST_K = 2

# ---------------------------------------------------------------------
# AR forecasting (Figure 2, Figure 3)
# ---------------------------------------------------------------------
AR_HORIZONS = [1, 5, 21]
AR_MAX_LAG = 5
AR_MIN_TRAIN = 250
AR_REFIT_FREQ = 21
HV_THRESHOLD = 0.5
TARGET_REGIME = "p_1"     # high-vol regime in K=2 trace-ordered fit

# ---------------------------------------------------------------------
# Portfolio backtest (Figure 4)
# ---------------------------------------------------------------------
TRANSACTION_COST_BPS = 5
TRANSACTION_COST = TRANSACTION_COST_BPS / 10_000
PORT_MIN_TRAIN = 250
REGIME_RECOMPUTE_FREQ = 63
ANN_FACTOR = 252

# ---------------------------------------------------------------------
# Asset universes
# ---------------------------------------------------------------------
CORE_ASSETS = [
    "world_equities",
    "us_treasuries",
    "ig_credit",
    "commodities",
    "high_yield",
    "gold",
]

EXTENDED_ADD_ORDER = [
    "us_equities",
    "eu_equities",
    "em_equities",
    "gov_long",
    "gov_short",
    "inflation_linked",
    "em_debt",
    "real_estate",
    "energy",
]

# Display names used in dissertation tables
ASSET_DISPLAY = {
    "world_equities":   "Equities",
    "us_treasuries":    "Bonds",
    "ig_credit":        "IG Credit",
    "commodities":      "Commodities",
    "high_yield":       "Cash",
    "gold":             "Gold",
    "us_equities":      "US Equities",
    "eu_equities":      "EU Equities",
    "em_equities":      "EM Equities",
    "gov_long":         "Gov Long",
    "gov_short":        "Gov Short",
    "inflation_linked": "Inflation-Linked",
    "em_debt":          "EM Debt",
    "real_estate":      "Real Estate",
    "energy":           "Energy",
}

CORRELATION_DISPLAY = {
    "world_equities": "World Eq.",
    "us_treasuries":  "Govt Bonds",
    "ig_credit":      "IG Credit",
    "commodities":    "Commodities",
    "high_yield":     "Cash",
    "gold":           "Gold",
}

# ---------------------------------------------------------------------
# Window-size robustness grid (Table A.1)
# ---------------------------------------------------------------------
ROBUSTNESS_GRID = [
    (1000, 63),
    (1250, 21),
    (1250, 63),    # baseline spec
    (1250, 126),
    (1500, 63),
]
ROBUSTNESS_BASELINE_SPEC = (1250, 63)

# ---------------------------------------------------------------------
# Kappa sensitivity grid (Appendix kappa table)
# ---------------------------------------------------------------------
KAPPA_GRID = [0.01, 29.9, 63.1, 100.2, 189.3, 568.0]
KAPPA_PRIMARY = 63.1
KAPPA_P_LABEL = {0.01: "0%", 29.9: "5%", 63.1: "10%", 100.2: "15%", 189.3: "25%", 568.0: "50%"}
