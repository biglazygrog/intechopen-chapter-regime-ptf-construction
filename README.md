# intechopen-chapter-regime-ptf-construction
GMM-based market regime identification and regime-aware portfolio construction for multi-asset institutional portfolios.

This repository contains the Python code supporting the research presented in Regime-Aware Portfolio Construction: A Gaussian Mixture Model Approach to Multi-Asset Allocation (Bevza, 2026), published in the IntechOpen volume Investor Behavior and Investment Strategies.
The code implements a MAP-EM Gaussian Mixture Model with Normal-Wishart regularisation for identifying latent market regimes from multi-asset return data, and applies the recovered regime structure to portfolio construction and backtesting. The analysis covers a baseline universe of five core institutional asset classes over 2000–2026 and an expanded universe of fourteen indices.
Key features:

MAP-EM GMM estimation with Dirichlet, Normal, and Normal-Wishart priors
Rolling window regime identification with BIC-based model selection
Label-switching mitigation via trace-covariance ordering
Regime-conditional correlation and downside risk analysis
Portfolio backtesting: oracle, AR forecast, random walk, and unconditional strategies
Asset universe expansion and regime concordance analysis

Requirements: Python 3.9+, NumPy, pandas, scikit-learn, statsmodels, scipy, matplotlib.
