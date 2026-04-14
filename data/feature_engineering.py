"""
data/feature_engineering.py
Price-derived feature construction for all ETF tickers.

ANTI-LOOKAHEAD CONTRACT
-----------------------
Every method in PriceFeatureBuilder shifts its input by at least 1 period
before computing any rolling window.  The invariant is:

    feature[t] depends only on data available at the CLOSE of day t-1.

This is enforced by:
  1. Calling .shift(1) on prices / returns before every rolling operation.
  2. The assertion in build_all() that verifies the first non-NaN row of each
     feature column is at least 2 rows after the first row of the input.

Do NOT remove, weaken, or bypass these shifts.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------
class PriceFeatureBuilder:
    """
    Builds a wide feature DataFrame from OHLCV close prices and full OHLCV data.

    All features are point-in-time safe: the value at row t is computed
    exclusively from rows 0 … t-1 of the input.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.  Used for window sizes and output paths.
    """

    def __init__(self, config: dict) -> None:
        feat = config["features"]
        self.vol_windows: List[int] = feat["volatility_windows"]
        self.mom_windows: List[int] = feat["momentum_windows"]
        self._processed_dir = PROJECT_ROOT / config["paths"]["processed"]

    # ------------------------------------------------------------------
    # Log returns  (no shift here — returns at t naturally use t-1 data)
    # ------------------------------------------------------------------
    def log_returns(self, df_close: pd.DataFrame) -> pd.DataFrame:
        """
        Compute daily log returns.

        log_ret[t] = log(close[t]) - log(close[t-1])

        This is equivalent to np.log(close[t] / close[t-1]).  The differencing
        itself means ret[t] is already a function of close[t] and close[t-1],
        so no extra shift is needed here — but downstream callers (vol, mom)
        must still shift these returns before rolling.
        """
        log_ret = np.log(df_close).diff(1)
        log_ret.columns = [f"{c}_logret" for c in df_close.columns]
        logger.debug("log_returns: shape {}", log_ret.shape)
        return log_ret

    # ------------------------------------------------------------------
    # Rolling volatility
    # ------------------------------------------------------------------
    def rolling_volatility(
        self,
        df_close: pd.DataFrame,
        windows: List[int] | None = None,
    ) -> pd.DataFrame:
        """
        Annualised rolling standard deviation of log returns.

        ANTI-LOOKAHEAD: log returns are shifted by 1 before the rolling window
        so vol[t] uses returns up to and including day t-1.

        Column naming: {ticker}_vol_{w}d
        """
        if windows is None:
            windows = self.vol_windows

        log_ret = np.log(df_close).diff(1)
        # Shift returns: vol at t must not see return at t
        shifted_ret = log_ret.shift(1)

        parts = []
        for w in windows:
            vol = shifted_ret.rolling(w).std() * np.sqrt(252)
            vol.columns = [f"{c}_vol_{w}d" for c in df_close.columns]
            parts.append(vol)
            logger.debug("rolling_volatility window={}: shape {}", w, vol.shape)

        return pd.concat(parts, axis=1)

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------
    def momentum(
        self,
        df_close: pd.DataFrame,
        windows: List[int] | None = None,
    ) -> pd.DataFrame:
        """
        Rolling sum of log returns.

        ANTI-LOOKAHEAD: log returns are shifted by 1 before the rolling window
        so mom[t] is the cumulative return over the w days ending at t-1.

        Column naming: {ticker}_mom_{w}d
        """
        if windows is None:
            windows = self.mom_windows

        log_ret = np.log(df_close).diff(1)
        # Shift returns: momentum at t must not include the return at t
        shifted_ret = log_ret.shift(1)

        parts = []
        for w in windows:
            mom = shifted_ret.rolling(w).sum()
            mom.columns = [f"{c}_mom_{w}d" for c in df_close.columns]
            parts.append(mom)
            logger.debug("momentum window={}: shape {}", w, mom.shape)

        return pd.concat(parts, axis=1)

    # ------------------------------------------------------------------
    # RSI  (Wilder smoothing)
    # ------------------------------------------------------------------
    def rsi(self, df_close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Wilder Relative Strength Index.

        ANTI-LOOKAHEAD: close prices are shifted by 1 before computing
        gains/losses, so rsi[t] is derived entirely from prices ≤ t-1.

        A common implementation computes gains/losses from the raw close and
        then shifts the resulting RSI series — that is equivalent but obscures
        where the shift sits.  Here we shift prices first to make the
        no-lookahead guarantee explicit and auditable.

        Column naming: {ticker}_rsi_{window}
        """
        # Shift prices so that the gain/loss at position t reflects the move
        # from close[t-2] to close[t-1], i.e. fully known by the open of t.
        shifted = df_close.shift(1)
        delta = shifted.diff(1)

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Wilder smoothing: seed with simple mean of first `window` rows,
        # then apply exponential smoothing with alpha = 1/window.
        # pandas ewm(alpha=1/w, adjust=False) replicates Wilder's formula.
        # min_periods=window suppresses RSI output until a full window of
        # observations exists, preventing spurious values on sparse early data.
        avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_df = 100 - (100 / (1 + rs))

        # 0/0 edge case: both avg_gain and avg_loss are zero when prices are
        # completely flat over the window.  RS = 0/NaN = NaN → RSI = NaN, but
        # a flat market is neutral, so RSI = 50 is the correct convention.
        flat = (avg_gain == 0) & (avg_loss == 0)
        rsi_df[flat] = 50

        rsi_df.columns = [f"{c}_rsi_{window}" for c in df_close.columns]
        logger.debug("rsi window={}: shape {}", window, rsi_df.shape)
        return rsi_df

    # ------------------------------------------------------------------
    # Bollinger Band width
    # ------------------------------------------------------------------
    def bollinger_width(self, df_close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Normalised Bollinger Band width: (upper - lower) / mid = (4 * std) / mid.

        Standard definition: upper = mid + 2*std, lower = mid - 2*std,
        so (upper - lower) / mid = 4*std / mid.

        ANTI-LOOKAHEAD: prices are shifted by 1 so the band at time t is
        computed from the rolling window ending at t-1.

        Column naming: {ticker}_bb_width
        """
        shifted = df_close.shift(1)
        mid = shifted.rolling(window).mean()
        std = shifted.rolling(window).std()
        width = (4 * std) / mid
        width.columns = [f"{c}_bb_width" for c in df_close.columns]
        logger.debug("bollinger_width window={}: shape {}", window, width.shape)
        return width

    # ------------------------------------------------------------------
    # Average True Range
    # ------------------------------------------------------------------
    def atr(self, df_ohlcv: dict[str, pd.DataFrame], window: int = 14) -> pd.DataFrame:
        """
        Average True Range per ticker.

        ANTI-LOOKAHEAD: true range is computed from raw prices (prev_close is
        the only natural lag needed by the formula), and the final rolling mean
        is shifted by 1 so that atr[t] is the rolling mean of true ranges
        ending at t-1.  Shifting inputs instead would double-lag prev_close
        (effectively shift(2) from raw data) making ATR unnecessarily stale.

        true_range = max(high - low,
                         |high - prev_close|,
                         |low  - prev_close|)

        Column naming: {ticker}_atr_{window}

        Parameters
        ----------
        df_ohlcv : dict[str, pd.DataFrame]
            Mapping of ticker → OHLCV DataFrame with columns
            ['Open', 'High', 'Low', 'Close', 'Volume'].
        """
        parts = []
        for ticker, ohlcv in df_ohlcv.items():
            # Use raw prices; prev_close is one natural lag for the true-range formula.
            # The single shift(1) on the final ATR output is the sole no-lookahead
            # guard: atr[t] = rolling mean of true ranges ending at t-1.
            # The previous pattern shifted inputs by 1 and then shifted close again
            # for prev_close, introducing a shift(2) lag on prev_close that made
            # ATR unnecessarily stale.
            high = ohlcv["High"]
            low = ohlcv["Low"]
            close = ohlcv["Close"]
            prev_close = close.shift(1)

            hl = high - low
            hpc = (high - prev_close).abs()
            lpc = (low - prev_close).abs()

            true_range = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
            atr_series = true_range.rolling(window).mean().shift(1)
            atr_series.name = f"{ticker}_atr_{window}"
            parts.append(atr_series)
            logger.debug("atr ticker={} window={}: {} rows", ticker, window, len(atr_series))

        return pd.concat(parts, axis=1)

    # ------------------------------------------------------------------
    # Build all features
    # ------------------------------------------------------------------
    def build_all(
        self,
        df_close: pd.DataFrame,
        df_ohlcv: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Compute every feature family, concatenate, validate, and persist.

        Parameters
        ----------
        df_close : pd.DataFrame
            Wide close-price DataFrame (columns = tickers, index = DatetimeIndex).
        df_ohlcv : dict[str, pd.DataFrame]
            Per-ticker OHLCV DataFrames.

        Returns
        -------
        pd.DataFrame
            Combined feature DataFrame saved to data/processed/price_features.parquet.
        """
        logger.info("Building price features for {} tickers", df_close.shape[1])

        feature_blocks = [
            # shift(1) is required here: log_returns[t] = log(close[t]) - log(close[t-1]),
            # so the first valid value sits at row 1 and encodes close[t] — a lookahead
            # violation when used as a raw feature.  Shifting moves the first valid value
            # to row 2, meaning feature[t] = return from close[t-2] to close[t-1].
            self.log_returns(df_close).shift(1),
            self.rolling_volatility(df_close),
            self.momentum(df_close),
            self.rsi(df_close),
            self.bollinger_width(df_close),
            self.atr(df_ohlcv),
        ]

        combined = pd.concat(feature_blocks, axis=1)
        logger.info("Combined feature matrix: shape {}", combined.shape)

        # ------------------------------------------------------------------
        # Anti-lookahead assertion
        # All features must have their first non-NaN value no earlier than
        # row index 2 (i.e. at least 2 rows after the first input row).
        # A feature appearing at row 0 or row 1 would imply it was computed
        # from fewer than 2 input observations, which violates the shift(1)
        # contract for any feature that requires at least one lag.
        # ------------------------------------------------------------------
        first_input_pos = 0  # row position of the first row in df_close
        violations = []
        for col in combined.columns:
            first_valid_pos = combined[col].first_valid_index()
            if first_valid_pos is None:
                continue  # fully-NaN column — caught by downstream quality checks
            pos = combined.index.get_loc(first_valid_pos)
            if pos <= first_input_pos + 1:
                violations.append((col, pos))

        if violations:
            details = "; ".join(f"{c} @ row {p}" for c, p in violations[:10])
            raise ValueError(
                f"Anti-lookahead violation: {len(violations)} feature(s) have their "
                f"first non-NaN value within the first 2 rows of input data: {details}"
            )
        logger.info("Anti-lookahead assertion passed — all features start at row ≥ 2")

        # Persist
        self._processed_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._processed_dir / "price_features.parquet"
        combined.to_parquet(out_path)
        logger.info(
            "Saved → {} ({} rows, {} features)",
            out_path.relative_to(PROJECT_ROOT),
            len(combined),
            combined.shape[1],
        )

        return combined


# ---------------------------------------------------------------------------
# Macro feature builder
# ---------------------------------------------------------------------------
class MacroFeatureBuilder:
    """
    Builds macro-derived features from the aligned macro DataFrame produced
    by ingest_macro.py.

    All features are point-in-time safe: every method shifts its input by 1
    before computing any transformation, so feature[t] uses only macro values
    known at the close of day t-1.
    """

    def __init__(self, config: dict) -> None:
        self._processed_dir = PROJECT_ROOT / config["paths"]["processed"]

    # ------------------------------------------------------------------
    # Rate changes
    # ------------------------------------------------------------------
    def rate_changes(
        self, df_macro: pd.DataFrame, windows: List[int] | None = None
    ) -> pd.DataFrame:
        """
        N-day change in DFF and GS10 rates.

        ANTI-LOOKAHEAD: rates are shifted by 1 before differencing so that
        change[t] = rate[t-1] - rate[t-1-w], fully known before t opens.

        Column naming: dff_change_{w}d, gs10_change_{w}d
        """
        if windows is None:
            windows = [1, 5, 21]

        shifted = df_macro[["DFF", "GS10"]].shift(1)
        parts = []
        for w in windows:
            diff = shifted.diff(w)
            diff.columns = [f"dff_change_{w}d", f"gs10_change_{w}d"]
            parts.append(diff)
            logger.debug("rate_changes window={}: shape {}", w, diff.shape)

        return pd.concat(parts, axis=1)

    # ------------------------------------------------------------------
    # Inflation surprise
    # ------------------------------------------------------------------
    def inflation_surprise(self, df_macro: pd.DataFrame) -> pd.DataFrame:
        """
        Approximate month-over-month CPI change as a 21-day percent change.

        ANTI-LOOKAHEAD: CPIAUCSL already has a 1-month publication lag applied
        during ingestion (ingest_macro.py shifts the index forward by one month).
        Adding another shift(1) here would make inflation_surprise two periods
        stale — the value at t would reflect CPI from t-2, not t-1.  No extra
        shift is applied.

        Column naming: cpi_surprise
        """
        surprise = df_macro["CPIAUCSL"].pct_change(21).rename("cpi_surprise")
        logger.debug("inflation_surprise: shape {}", surprise.shape)
        return surprise.to_frame()

    # ------------------------------------------------------------------
    # VIX regime category
    # ------------------------------------------------------------------
    def vix_regime(self, df_macro: pd.DataFrame) -> pd.DataFrame:
        """
        Categorical VIX regime: 'low' (<15), 'medium' (15–25), 'high' (>25).

        ANTI-LOOKAHEAD: VIX is shifted by 1 so the regime at t reflects the
        closing VIX level from day t-1.

        Column naming: vix_regime
        """
        shifted_vix = df_macro["VIXCLS"].shift(1)
        regime = pd.cut(
            shifted_vix,
            bins=[-np.inf, 15, 25, np.inf],
            labels=["low", "medium", "high"],
        ).rename("vix_regime")
        logger.debug("vix_regime: {} categories", regime.nunique())
        return regime.to_frame()

    # ------------------------------------------------------------------
    # VIX change
    # ------------------------------------------------------------------
    def vix_change(
        self, df_macro: pd.DataFrame, windows: List[int] | None = None
    ) -> pd.DataFrame:
        """
        N-day change in VIX level.

        ANTI-LOOKAHEAD: VIX is shifted by 1 before differencing.

        Column naming: vix_change_{w}d
        """
        if windows is None:
            windows = [5, 21]

        shifted_vix = df_macro["VIXCLS"].shift(1)
        parts = []
        for w in windows:
            chg = shifted_vix.diff(w).rename(f"vix_change_{w}d")
            parts.append(chg)
            logger.debug("vix_change window={}: shape {}", w, chg.shape)

        return pd.concat(parts, axis=1)

    # ------------------------------------------------------------------
    # Yield curve slope
    # ------------------------------------------------------------------
    def yield_curve_slope(self, df_macro: pd.DataFrame) -> pd.DataFrame:
        """
        Yield curve slope (rate_spread) and its 21-day rolling mean.

        ANTI-LOOKAHEAD: rate_spread is shifted by 1; the rolling mean is
        applied to the already-shifted series.

        Column naming: yield_curve_slope, yield_curve_slope_21d_ma
        """
        shifted_spread = df_macro["rate_spread"].shift(1)
        slope = shifted_spread.rename("yield_curve_slope")
        slope_ma = shifted_spread.rolling(21).mean().rename("yield_curve_slope_21d_ma")
        result = pd.concat([slope, slope_ma], axis=1)
        logger.debug("yield_curve_slope: shape {}", result.shape)
        return result

    # ------------------------------------------------------------------
    # Build all macro features
    # ------------------------------------------------------------------
    def build_all(self, df_macro: pd.DataFrame) -> pd.DataFrame:
        """
        Compute every macro feature family, concatenate, and persist.

        Parameters
        ----------
        df_macro : pd.DataFrame
            Aligned macro DataFrame from ingest_macro.py.

        Returns
        -------
        pd.DataFrame
            Combined macro feature DataFrame saved to
            data/processed/macro_features.parquet.
        """
        logger.info("Building macro features — input shape {}", df_macro.shape)

        blocks = [
            self.rate_changes(df_macro),
            self.inflation_surprise(df_macro),
            self.vix_regime(df_macro),
            self.vix_change(df_macro),
            self.yield_curve_slope(df_macro),
        ]

        combined = pd.concat(blocks, axis=1)
        logger.info("Combined macro feature matrix: shape {}", combined.shape)

        self._processed_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._processed_dir / "macro_features.parquet"
        # vix_regime is categorical — store as string so parquet round-trips cleanly
        combined["vix_regime"] = combined["vix_regime"].astype(str)
        combined.to_parquet(out_path)
        logger.info(
            "Saved → {} ({} rows, {} features)",
            out_path.relative_to(PROJECT_ROOT),
            len(combined),
            combined.shape[1],
        )

        return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "feature_engineering.log", rotation="10 MB", retention="30 days")

    cfg = _load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw"]

    close_path = raw_dir / "all_close_prices.parquet"
    if not close_path.exists():
        print(f"Missing {close_path} — run ingest_prices.py first.")
        sys.exit(1)

    df_close = pd.read_parquet(close_path)
    tickers = df_close.columns.tolist()

    df_ohlcv = {}
    missing = []
    for ticker in tickers:
        p = raw_dir / f"{ticker}_ohlcv.parquet"
        if p.exists():
            df_ohlcv[ticker] = pd.read_parquet(p)
        else:
            missing.append(ticker)
    if missing:
        print(f"Missing OHLCV files for: {missing} — run ingest_prices.py first.")
        sys.exit(1)

    builder = PriceFeatureBuilder(cfg)
    features = builder.build_all(df_close, df_ohlcv)

    print(f"\nFeature matrix: {features.shape[0]} rows × {features.shape[1]} columns")
    print(f"Date range:     {features.index[0].date()} → {features.index[-1].date()}")
    print(f"NaN rows:       {features.isna().any(axis=1).sum()} "
          f"({features.isna().any(axis=1).mean():.1%} of total)")
    print("\nFirst 5 columns:")
    print(features.iloc[:3, :5].to_string())
    sys.exit(0)
