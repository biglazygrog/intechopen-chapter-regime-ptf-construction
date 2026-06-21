"""
Figure 4 — Portfolio Backtest Results (4-panel).

Thin wrapper around research.analysis.backtest.main, which runs the 5-strategy
expanding-window backtest and writes figure4_backtest.png inside the
research/output_charts/figures/ folder.

Run:  python -m research.figures.figure4_backtest
"""
from research.analysis.backtest import main

if __name__ == "__main__":
    main()
