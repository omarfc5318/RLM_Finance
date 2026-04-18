"""
TRAINING USE ONLY.
Never import this module in any inference or prediction file.
Targets are future values used only for model training.

data/targets.py
Builds forward-looking training targets from close price data.

All methods in TargetBuilder use negative shifts (lookahead).  This is
intentional and correct — these values are targets, not features.  They
must NEVER appear in any feature DataFrame or be passed to a model at
inference time.

Safeguards:
  - Module-level warning logged on import.
  - build_all() drops all rows from train_end onward before saving so the
    stored parquet cannot accidentally include out-of-sample targets.
  - validate_target_alignment() asserts that target columns are NaN for
    the final `max_horizon` rows, confirming the shift rolled off correctly.

Environment variables:
  RFIE_TRAINING_MODE=1  Must be set in any script that imports this module.
                        The default (unset) is safe — import is blocked.
                        Do NOT export this in your shell profile permanently.
"""

from __future__ import annotations

import os
if os.environ.get("RFIE_TRAINING_MODE") != "1":
    raise ImportError(
        "data.targets may only be imported when RFIE_TRAINING_MODE=1 "
        "is set. This module is for training use only."
    )

import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml
from loguru import logger

logger.warning(
    "data.targets is a TRAINING-ONLY module. "
    "Do not import in inference or prediction code."
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Target builder
# ---------------------------------------------------------------------------
class TargetBuilder:
    """
    Computes forward-looking training targets for all configured tickers.

    All methods use negative shifts — they look into the future deliberately.
    The resulting columns must only be used as model labels, never as input
    features.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.
    """

    # Maximum horizon across all targets — used to trim trailing NaN rows
    # in build_all() and in the alignment validator.
    MAX_HORIZON = 21

    def __init__(self, config: dict) -> None:
        self._tickers: List[str] = config["data"]["tickers"]
        self._processed_dir = PROJECT_ROOT / config["paths"]["processed"]

    # ------------------------------------------------------------------
    # Forward return
    # ------------------------------------------------------------------
    def forward_return(
        self,
        df_close: pd.DataFrame,
        ticker: str,
        horizon_days: int = 1,
    ) -> pd.Series:
        """
        Log return from today's close to the close horizon_days ahead.

        THIS IS LOOKAHEAD — it is the target, not a feature.
        That is intentional and correct for training.

        forward_ret[t] = log(close[t + horizon_days] / close[t])

        Column name: {ticker}_tgt_ret_{horizon_days}d
        """
        prices = df_close[ticker]
        fwd_ret = np.log(prices.shift(-horizon_days) / prices)
        fwd_ret.name = f"{ticker}_tgt_ret_{horizon_days}d"
        logger.debug(
            "forward_return {} horizon={}d: {} non-NaN values",
            ticker, horizon_days, fwd_ret.notna().sum(),
        )
        return fwd_ret

    # ------------------------------------------------------------------
    # Forward volatility
    # ------------------------------------------------------------------
    def forward_volatility(
        self,
        df_close: pd.DataFrame,
        ticker: str,
        horizon_days: int = 5,
    ) -> pd.Series:
        """
        Realised volatility of log returns over the NEXT horizon_days days.

        The rolling window is anchored to future dates:
          fwd_vol[t] = std(log_ret[t+1], ..., log_ret[t+horizon_days])

        Computed as:
          log_rets = diff of log prices
          fwd_vol[t] = rolling(horizon_days).std() applied to future returns

        Column name: {ticker}_tgt_vol_{horizon_days}d
        """
        log_rets = np.log(df_close[ticker]).diff()
        # shift(-horizon_days) moves the rolling window forward so that
        # the std at position t covers returns from t+1 … t+horizon_days.
        fwd_vol = log_rets.shift(-horizon_days).rolling(horizon_days).std()
        fwd_vol.name = f"{ticker}_tgt_vol_{horizon_days}d"
        logger.debug(
            "forward_volatility {} horizon={}d: {} non-NaN values",
            ticker, horizon_days, fwd_vol.notna().sum(),
        )
        return fwd_vol

    # ------------------------------------------------------------------
    # Forward maximum drawdown
    # ------------------------------------------------------------------
    def forward_max_drawdown(
        self,
        df_close: pd.DataFrame,
        ticker: str,
        horizon_days: int = 21,
    ) -> pd.Series:
        """
        Maximum drawdown in the NEXT horizon_days trading days,
        with correct peak-before-trough ordering.

        For each date t, the window is prices[t+1 … t+horizon_days] inclusive.
        MDD = min over all j in window of (prices[j] - running_max_up_to_j) / running_max_up_to_j

        The old shift+rolling max/min approach computed (global_min - global_max) / global_max
        which does not enforce that the trough occurs after the peak — it can
        produce an artificially large drawdown when the minimum precedes the
        maximum inside the window.

        sliding_window_view gives a zero-copy (n - horizon_days, horizon_days)
        view of the price array.  np.maximum.accumulate along axis=1 builds the
        running peak at each position within the window, so every drawdown ratio
        is correctly anchored to the highest price seen so far in that window.

        The conservative MAX_HORIZON trim in build_all() ensures no valid rows
        are accidentally exposed — short-horizon targets (1d, 5d) have valid
        values in rows [-20:-1] but are intentionally discarded as a safety margin.

        Column name: {ticker}_tgt_mdd_{horizon_days}d
        """
        from numpy.lib.stride_tricks import sliding_window_view

        prices = df_close[ticker]
        vals = prices.to_numpy(dtype=float)
        if np.any(vals <= 0):
            raise ValueError(
                f"Non-positive prices found for {ticker} — "
                "check raw data for data errors or corporate actions."
            )
        n = len(vals)
        if n <= horizon_days:
            series = pd.Series(np.nan, index=prices.index,
                               name=f"{ticker}_tgt_mdd_{horizon_days}d")
            logger.warning(
                "forward_max_drawdown {}: series length {} <= horizon {}, "
                "returning all-NaN series",
                ticker, n, horizon_days,
            )
            return series

        # windows shape: (n - horizon_days, horizon_days)
        # [1:] shifts the view forward by one so window at row t covers
        # prices[t+1 … t+horizon_days], not prices[t … t+horizon_days-1].
        windows = sliding_window_view(vals, horizon_days)[1:]
        running_maxes = np.maximum.accumulate(windows, axis=1)
        dd_matrix = (windows - running_maxes) / running_maxes
        mdd_vals = dd_matrix.min(axis=1)

        # Pad the tail with NaN so the result aligns with the original index.
        # horizon_days NaNs at the end: no full forward window exists there.
        n_nan = n - len(mdd_vals)
        result = np.concatenate([
            mdd_vals,
            np.full(n_nan, np.nan, dtype=np.float64)
        ])

        series = pd.Series(result, index=prices.index)
        series.name = f"{ticker}_tgt_mdd_{horizon_days}d"
        logger.debug(
            "forward_max_drawdown {} horizon={}d: {} non-NaN values",
            ticker, horizon_days, series.notna().sum(),
        )
        return series

    # ------------------------------------------------------------------
    # Build all targets
    # ------------------------------------------------------------------
    def build_all(self, df_close: pd.DataFrame) -> pd.DataFrame:
        """
        Build every target for every configured ticker and persist.

        Targets built per ticker:
          - forward_return at horizons 1, 5, 21 days
          - forward_volatility at horizons 5, 21 days
          - forward_max_drawdown at horizon 21 days

        The final MAX_HORIZON rows are dropped before saving because their
        target values are NaN by construction (no future prices exist).

        Parameters
        ----------
        df_close : pd.DataFrame
            Wide close-price DataFrame (columns = tickers).

        Returns
        -------
        pd.DataFrame saved to data/processed/targets.parquet.
        """
        logger.info(
            "Building targets for {} tickers × {} rows",
            len(self._tickers), len(df_close),
        )

        parts: List[pd.Series] = []
        for ticker in self._tickers:
            parts.append(self.forward_return(df_close, ticker, horizon_days=1))
            parts.append(self.forward_return(df_close, ticker, horizon_days=5))
            parts.append(self.forward_return(df_close, ticker, horizon_days=21))
            parts.append(self.forward_volatility(df_close, ticker, horizon_days=5))
            parts.append(self.forward_volatility(df_close, ticker, horizon_days=21))
            parts.append(self.forward_max_drawdown(df_close, ticker, horizon_days=21))

        targets = pd.concat(parts, axis=1)
        logger.info("Raw target matrix: shape {}", targets.shape)

        # Validate trailing NaN structure before trimming
        self._validate_target_alignment(targets)

        # Conservative trim: drop the final MAX_HORIZON rows as a safety margin.
        # Short-horizon targets (1d, 5d) have valid values in rows [-20:-1] but
        # are intentionally discarded to ensure no target column can ever be
        # carelessly sliced and exposed at inference time.
        targets = targets.iloc[: -self.MAX_HORIZON]
        logger.info(
            "After trimming {} trailing rows: shape {}",
            self.MAX_HORIZON, targets.shape,
        )

        # Persist
        self._processed_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._processed_dir / "targets.parquet"
        targets.to_parquet(out_path)
        logger.info(
            "Saved → {} ({} rows, {} targets)",
            out_path.relative_to(PROJECT_ROOT),
            *targets.shape,
        )

        return targets

    # ------------------------------------------------------------------
    # Internal validator
    # ------------------------------------------------------------------
    @staticmethod
    def _horizon_from_col(col_name: str) -> int:
        """
        Extract the horizon integer from a column name ending in _{n}d.

        e.g. 'SPY_tgt_ret_5d' → 5, 'XLK_tgt_mdd_21d' → 21.
        Falls back to 1 if the pattern is not found.
        """
        match = re.search(r"_(\d+)d$", col_name)
        return int(match.group(1)) if match else 1

    def _validate_target_alignment(self, targets: pd.DataFrame) -> None:
        """
        Assert that the last N rows for each column are NaN, where N is the
        horizon encoded in that column's name.

        A 1d-forward target only becomes NaN in the final 1 row; a 21d target
        in the final 21 rows.  Checking all columns against MAX_HORIZON would
        incorrectly flag short-horizon targets as misaligned.
        """
        bad_cols = []
        for col in targets.columns:
            horizon = self._horizon_from_col(col)
            tail = targets[col].iloc[-horizon:]
            if not tail.isna().all():
                bad_cols.append((col, horizon, "under-shifted"))
            pre_tail = targets[col].iloc[-(horizon + 1)]
            if pd.isna(pre_tail):
                bad_cols.append((col, horizon, "over-shifted"))

        if bad_cols:
            details = "; ".join(
                f"{c} (horizon={h}d, {direction})" for c, h, direction in bad_cols[:5]
            )
            raise ValueError(
                f"Target alignment error: {len(bad_cols)} column(s) failed alignment check "
                f"(under-shifted or over-shifted): {details}"
            )
        logger.info(
            "Target alignment validated — all {} columns are NaN in their "
            "respective trailing horizon rows",
            len(targets.columns),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "targets.log", rotation="10 MB", retention="30 days")

    cfg = _load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw"]

    close_path = raw_dir / "all_close_prices.parquet"
    if not close_path.exists():
        print(f"Missing {close_path} — run ingest_prices.py first.")
        sys.exit(1)

    df_close = pd.read_parquet(close_path)
    builder = TargetBuilder(cfg)
    targets = builder.build_all(df_close)

    print(f"\nTarget matrix: {targets.shape[0]} rows × {targets.shape[1]} columns")
    print(f"Date range:    {targets.index[0].date()} → {targets.index[-1].date()}")
    print(f"NaN count:     {targets.isna().sum().sum()}")
    print(f"\nColumn list ({len(targets.columns)}):")
    for col in targets.columns:
        print(f"  {col}")
    sys.exit(0)
