# rfie — Regime-Filtered Investment Engine

- **Regime detection**: Uses Hidden Markov Models and macro indicators (via FRED) to identify distinct market regimes (risk-on, risk-off, transition) across the full data history.
- **Ensemble modeling**: Trains base learners (XGBoost, statsmodels, PyTorch) per regime and combines them with a meta-learner optimized via Optuna for walk-forward validation.
- **Evolutionary portfolio construction**: Applies DEAP genetic algorithms alongside CVXPY convex optimization to build regime-aware, risk-adjusted sector-rotation portfolios across 11 SPDR ETFs.
- **Robust backtesting**: Evaluates strategies on a strict train/val/test split (2015–2021 / 2022–mid-2023 / mid-2023–2024) using Empyrical and QuantStats for performance attribution and drawdown analysis.
- **Interactive dashboard**: Streams live regime signals, portfolio weights, and backtest tearsheets through a Streamlit + Plotly dashboard for rapid iteration and reporting.
