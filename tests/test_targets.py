"""
tests/test_targets.py
Correctness tests for TargetBuilder.

Tests use a synthetic close-price DataFrame so no external data files are
required.  All expected values are computed independently from the module
under test — numpy operations only — and compared to module output.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest

# Suppress the training-only import warning inside the test suite
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from data.targets import TargetBuilder

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MOCK_CONFIG = {
    "data": {
        "tickers": ["SPY", "XLK", "XLF"],
    },
    "paths": {
        "processed": "data/processed/",
    },
}

# Deterministic synthetic prices: 300 business days starting 2018-01-01
N_ROWS = 300
_IDX = pd.bdate_range(start="2018-01-01", periods=N_ROWS)
_RNG = np.random.default_rng(seed=42)
# Simulate a GBM-like price path so returns are sensible
_RAW = 100 * np.cumprod(1 + _RNG.normal(0, 0.01, size=(N_ROWS, 3)), axis=0)


@pytest.fixture
def df_close() -> pd.DataFrame:
    return pd.DataFrame(_RAW, index=_IDX, columns=["SPY", "XLK", "XLF"])


@pytest.fixture
def builder() -> TargetBuilder:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return TargetBuilder(MOCK_CONFIG)


# ---------------------------------------------------------------------------
# Test 1 — forward_return matches manual computation for 5 random SPY dates
# ---------------------------------------------------------------------------

def test_forward_return_spy_matches_manual(builder, df_close):
    """
    For 5 randomly sampled dates, assert that the stored forward_1d_return
    for SPY equals the manually computed value within 1e-10 tolerance.

    Manual formula: log(close[t + 1] / close[t])
    """
    rng = np.random.default_rng(seed=0)
    # Choose dates that have a valid t+1 (exclude the last row)
    candidate_positions = np.arange(0, N_ROWS - 1)
    sampled_positions = rng.choice(candidate_positions, size=5, replace=False)
    sampled_positions.sort()

    fwd_ret = builder.forward_return(df_close, "SPY", horizon_days=1)

    for pos in sampled_positions:
        date = df_close.index[pos]
        stored = fwd_ret.iloc[pos]
        expected = np.log(df_close["SPY"].iloc[pos + 1] / df_close["SPY"].iloc[pos])
        assert abs(stored - expected) < 1e-10, (
            f"forward_return mismatch at {date.date()} (row {pos}): "
            f"stored={stored:.12f}, expected={expected:.12f}, "
            f"diff={abs(stored - expected):.2e}"
        )


# ---------------------------------------------------------------------------
# Test 2 — forward_volatility uses only future returns
# ---------------------------------------------------------------------------

def test_forward_volatility_uses_future_returns(builder, df_close):
    """
    forward_vol[t] must equal std(log_ret[t+1], ..., log_ret[t+horizon])
    computed independently.  Verified at 5 randomly sampled dates.
    """
    horizon = 5
    rng = np.random.default_rng(seed=1)
    # Need at least horizon rows after sampled position
    candidate_positions = np.arange(0, N_ROWS - horizon)
    sampled_positions = rng.choice(candidate_positions, size=5, replace=False)

    fwd_vol = builder.forward_volatility(df_close, "SPY", horizon_days=horizon)
    log_prices = np.log(df_close["SPY"].values)
    log_rets = np.diff(log_prices, prepend=np.nan)  # log_rets[i] = log(p[i]/p[i-1])

    for pos in sampled_positions:
        stored = fwd_vol.iloc[pos]
        # Future returns: indices pos+1 through pos+horizon (inclusive)
        future_rets = log_rets[pos + 1 : pos + horizon + 1]
        expected = float(np.std(future_rets, ddof=1))
        assert abs(stored - expected) < 1e-10, (
            f"forward_volatility mismatch at row {pos}: "
            f"stored={stored:.12f}, expected={expected:.12f}, "
            f"diff={abs(stored - expected):.2e}"
        )


# ---------------------------------------------------------------------------
# Test 3 — trailing NaN structure: last MAX_HORIZON rows are all NaN
# ---------------------------------------------------------------------------

def test_trailing_nan_structure(builder, df_close):
    """
    For each target column, the last N rows must be NaN where N is the
    horizon encoded in the column name (e.g. _5d → 5, _21d → 21).

    Checking all columns against a single MAX_HORIZON is incorrect because
    a 1d-return target is only NaN in its final 1 row, not its final 21.
    """
    parts = []
    for ticker in ["SPY", "XLK", "XLF"]:
        parts.append(builder.forward_return(df_close, ticker, horizon_days=1))
        parts.append(builder.forward_return(df_close, ticker, horizon_days=5))
        parts.append(builder.forward_return(df_close, ticker, horizon_days=21))
        parts.append(builder.forward_volatility(df_close, ticker, horizon_days=5))
        parts.append(builder.forward_volatility(df_close, ticker, horizon_days=21))
        parts.append(builder.forward_max_drawdown(df_close, ticker, horizon_days=21))

    targets = pd.concat(parts, axis=1)

    bad_cols = []
    for col in targets.columns:
        match = re.search(r"_(\d+)d$", col)
        horizon = int(match.group(1)) if match else 1
        tail = targets[col].iloc[-horizon:]
        if not tail.isna().all():
            bad_cols.append((col, horizon))

    assert len(bad_cols) == 0, (
        f"{len(bad_cols)} column(s) are not fully NaN in their trailing horizon rows: "
        + "; ".join(f"{c} (last {h} rows)" for c, h in bad_cols[:5])
    )
