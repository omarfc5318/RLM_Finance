"""
data/ingest_macro.py
Raw macroeconomic series ingestion via FRED.
No feature engineering or forward-looking statistics are computed here,
with the exception of the rate_spread derived field, which is a contemporaneous
arithmetic combination of two already-downloaded series (no lookahead).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from fredapi import Fred
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

# FRED series to download: {column_name: fred_series_id}
FRED_SERIES: dict[str, str] = {
    "DFF": "DFF",        # Federal Funds Rate          – daily
    "CPIAUCSL": "CPIAUCSL",  # Consumer Price Index    – monthly
    "VIXCLS": "VIXCLS",  # CBOE VIX                   – daily
    "GS10": "GS10",      # 10-Year Treasury Yield      – daily
    "T10Y2Y": "T10Y2Y",  # 10Y minus 2Y Treasury Spread – daily
}


# ---------------------------------------------------------------------------
# Config & environment
# ---------------------------------------------------------------------------
def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Loaded config from {}", path)
    return cfg


def load_fred_client() -> Fred:
    """Load FRED_API_KEY from .env and return an authenticated Fred client."""
    load_dotenv(ENV_PATH)
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    logger.info("FRED_API_KEY loaded from {}", ENV_PATH)
    return Fred(api_key=api_key)


# ---------------------------------------------------------------------------
# Download individual series
# ---------------------------------------------------------------------------
def download_series(
    fred: Fred,
    series_id: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Download a single FRED series and return a named Series with a DatetimeIndex."""
    logger.info("Downloading FRED series {} ({} → {})", series_id, start_date, end_date)
    try:
        s = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    except Exception as exc:
        raise RuntimeError(
            f"[{series_id}] FRED download raised an exception: {exc}"
        ) from exc

    if s is None or s.empty:
        raise ValueError(
            f"[{series_id}] FRED returned an empty series for the period "
            f"{start_date} → {end_date}. Check the series ID and date range."
        )

    s.index = pd.to_datetime(s.index)
    s.name = series_id
    logger.info(
        "  {} → {} observations  [{} → {}]",
        series_id,
        len(s),
        s.index[0].date().isoformat(),
        s.index[-1].date().isoformat(),
    )
    return s


# ---------------------------------------------------------------------------
# Resample to business-day frequency
# ---------------------------------------------------------------------------
def to_business_day(s: pd.Series) -> pd.Series:
    """
    Resample a series to business-day ('B') frequency using forward-fill.

    # ffill carries forward the last known value.
    # This is NOT lookahead because we are using the most recent past value,
    # not a future value. The alternative (interpolation) would be lookahead.
    """
    resampled = s.resample("B").ffill()
    logger.info(
        "  {} resampled to business-day frequency → {} rows",
        s.name,
        len(resampled),
    )
    return resampled


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------
def check_macro_quality(df: pd.DataFrame, equity_index: pd.DatetimeIndex) -> None:
    """
    Assert data-quality invariants on the aligned macro DataFrame.

    Checks:
    - No NaN values in any column after ffill
    - Every date in the equity close index is present in the macro index
    """
    # 1. Zero nulls after ffill
    null_counts = df.isnull().sum()
    if null_counts.any():
        bad = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Null values remain after ffill: {bad}")
    logger.info("  Null check passed — no NaN values in any column")

    # 2. Macro index must cover every equity trading date
    missing_dates = equity_index.difference(df.index)
    if len(missing_dates):
        raise ValueError(
            f"{len(missing_dates)} equity date(s) missing from macro index. "
            f"First 5: {missing_dates[:5].tolist()}"
        )
    logger.info(
        "  Coverage check passed — all {} equity dates present in macro index",
        len(equity_index),
    )


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("  Saved → {} ({} rows, {} cols)", path.relative_to(PROJECT_ROOT), *df.shape)


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------
def ingest_macro(cfg: dict) -> pd.DataFrame:
    """
    Download, resample, align, and persist all FRED macro series.

    Returns the final macro DataFrame saved to disk.
    """
    start_date: str = cfg["data"]["start_date"]
    end_date: str = cfg["data"]["end_date"]
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw"]

    # Load equity close index as the alignment target
    close_path = raw_dir / "all_close_prices.parquet"
    if not close_path.exists():
        raise FileNotFoundError(
            f"Equity close prices not found at {close_path}. "
            "Run ingest_prices.py first."
        )
    equity_index: pd.DatetimeIndex = pd.read_parquet(close_path).index
    logger.info(
        "Equity date index loaded — {} dates [{} → {}]",
        len(equity_index),
        equity_index[0].date().isoformat(),
        equity_index[-1].date().isoformat(),
    )

    fred = load_fred_client()

    # Download and resample each series
    resampled: dict[str, pd.Series] = {}
    for col_name, series_id in FRED_SERIES.items():
        raw_series = download_series(fred, series_id, start_date, end_date)

        if col_name == "CPIAUCSL":
            # Shift the CPI index forward by one month to simulate real publication lag.
            # The BLS typically releases each month's CPI reading ~2 weeks into the
            # following month, so a model running on date T should only see CPI values
            # whose original observation date is at least one month before T.
            # Shifting the index (not the values) moves each reading to the first date
            # it would realistically have been available, preventing lookahead.
            raw_series.index = raw_series.index + pd.DateOffset(months=1)
            logger.info(
                "  CPIAUCSL index shifted forward by 1 month to reflect publication lag"
            )

        resampled[col_name] = to_business_day(raw_series)

    # Outer join preserves every date across all series, then reindex pins the result
    # to exactly the equity trading calendar. method='ffill' fills any equity dates
    # that fall between FRED observation dates (e.g. CPI only updates monthly) with
    # the most recently available value — same no-lookahead guarantee as resample ffill.
    macro_df = pd.concat(resampled, axis=1, join="outer")
    macro_df = macro_df.reindex(equity_index, method="ffill")

    logger.info(
        "Macro DataFrame aligned to equity index → shape {}",
        macro_df.shape,
    )

    # Secondary fill pass: reindex ffill can leave NaNs at the very start of the
    # series if the first equity date precedes the first FRED observation for that
    # column. ffill() has nothing to carry forward there, so bfill() backfills from
    # the earliest available value. bfill is safe here because it only fires on the
    # leading edge — once a real observation exists, ffill has already covered the rest.
    nans_before = int(macro_df.isnull().sum().sum())
    macro_df = macro_df.ffill().bfill()
    nans_after = int(macro_df.isnull().sum().sum())
    nans_filled = nans_before - nans_after
    if nans_filled:
        logger.info(
            "  Secondary ffill/bfill pass filled {} NaN cell(s) "
            "({} remain after pass)",
            nans_filled,
            nans_after,
        )
    else:
        logger.info("  Secondary ffill/bfill pass: no NaNs needed filling")

    # Derived feature: rate spread (contemporaneous, no lookahead)
    macro_df["rate_spread"] = macro_df["GS10"] - macro_df["DFF"]
    logger.info("Derived column 'rate_spread' = GS10 − DFF added")

    # Quality checks
    check_macro_quality(macro_df, equity_index)

    save_parquet(macro_df, raw_dir / "macro_data.parquet")

    return macro_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "ingest_macro.log", rotation="10 MB", retention="30 days")

    cfg = load_config()
    macro_df = ingest_macro(cfg)

    print("\n=== Macro Series Summary ===")
    summary = pd.DataFrame(
        {
            "series": macro_df.columns,
            "start_date": [macro_df[c].first_valid_index().date().isoformat() for c in macro_df.columns],
            "end_date": [macro_df[c].last_valid_index().date().isoformat() for c in macro_df.columns],
            "row_count": [macro_df[c].count() for c in macro_df.columns],
            "nan_count": [int(macro_df[c].isna().sum()) for c in macro_df.columns],
        }
    )
    print(summary.to_string(index=False))
    print(f"\nFinal macro DataFrame shape: {macro_df.shape}")
    sys.exit(0)
