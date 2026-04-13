"""
data/ingest_prices.py
Raw OHLCV ingestion for all configured ETF tickers.
No feature engineering or forward-looking statistics are computed here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Loaded config from {}", path)
    return cfg


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_ticker(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download OHLCV for a single ticker and return a tidy DataFrame."""
    logger.info("Downloading {} ({} → {})", ticker, start_date, end_date)
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    except Exception as exc:
        raise RuntimeError(f"[{ticker}] yfinance download raised an exception: {exc}") from exc

    if df.empty:
        raise ValueError(
            f"[{ticker}] yfinance returned an empty DataFrame for the period "
            f"{start_date} → {end_date}. Check the ticker symbol and date range."
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    logger.info("  {} → shape {}", ticker, df.shape)
    return df


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------
def check_data_quality(df: pd.DataFrame, ticker: str = "") -> None:
    """
    Assert basic data-quality invariants on a raw OHLCV DataFrame.

    Checks:
    - DataFrame is not empty
    - Minimum 1 000 rows (≈ 4 years of trading days)
    - All Close prices are strictly positive
    - No NaN values in the Close column
    - Date index is monotonically increasing
    - No duplicate dates
    - All dates fall on business days (Mon–Fri)
    """
    tag = f"[{ticker}] " if ticker else ""

    # 1. Not empty
    if df.empty:
        raise ValueError(f"{tag}DataFrame is empty")

    # 2. Minimum row count
    min_rows = 1_000
    if len(df) < min_rows:
        raise ValueError(
            f"{tag}Only {len(df)} rows — expected at least {min_rows} trading days"
        )

    # 3. All Close prices strictly positive
    non_positive = int((df["Close"] <= 0).sum())
    if non_positive:
        raise ValueError(
            f"{tag}Close column contains {non_positive} non-positive price(s)"
        )

    # 4. No NaN in Close
    nan_count = int(df["Close"].isna().sum())
    if nan_count:
        raise ValueError(f"{tag}Close column contains {nan_count} NaN value(s)")

    # 5. Monotonically increasing index
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{tag}Date index is not monotonically increasing")

    # 6. No duplicate dates
    dupes = int(df.index.duplicated().sum())
    if dupes:
        raise ValueError(f"{tag}Date index contains {dupes} duplicate date(s)")

    # 7. All dates are business days (weekday 0–4)
    non_bdays = df.index[df.index.dayofweek >= 5]
    if len(non_bdays):
        raise ValueError(
            f"{tag}{len(non_bdays)} date(s) fall on weekends: {non_bdays[:5].tolist()}"
        )

    logger.info("  {}quality checks passed", tag)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("  Saved → {} ({} rows)", path.relative_to(PROJECT_ROOT), len(df))


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------
def ingest_all(cfg: dict) -> pd.DataFrame:
    """
    Download, validate, and persist OHLCV data for every configured ticker.

    Returns a summary DataFrame with one row per ticker.
    """
    tickers: list[str] = cfg["data"]["tickers"]
    start_date: str = cfg["data"]["start_date"]
    end_date: str = cfg["data"]["end_date"]
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw"]

    all_close: dict[str, pd.Series] = {}
    summary_rows: list[dict] = []

    for ticker in tickers:
        df = download_ticker(ticker, start_date, end_date)
        check_data_quality(df, ticker)

        save_parquet(df, raw_dir / f"{ticker}_ohlcv.parquet")

        all_close[ticker] = df["Close"]

        summary_rows.append(
            {
                "ticker": ticker,
                "start_date": df.index[0].date().isoformat(),
                "end_date": df.index[-1].date().isoformat(),
                "row_count": len(df),
                "nan_count": int(df.isna().sum().sum()),
            }
        )

    # Combined close prices — inner join so only dates present in ALL tickers are kept
    close_df = pd.concat(all_close, axis=1, join="inner")
    close_df.columns = list(all_close.keys())
    close_df.index.name = "Date"

    # Warn about any tickers whose raw history started later than the shared start date
    shared_start = close_df.index[0]
    short_history = {
        t: all_close[t].index[0]
        for t in tickers
        if all_close[t].index[0] > shared_start
    }
    if short_history:
        details = ", ".join(
            f"{t} (starts {idx.date().isoformat()})"
            for t, idx in sorted(short_history.items(), key=lambda x: x[1])
        )
        logger.warning(
            "Ticker(s) with shorter history trimmed by inner join: {}. "
            "Shared date range starts {}.",
            details,
            shared_start.date().isoformat(),
        )
    else:
        logger.info("All {} tickers share an identical date index", len(tickers))

    save_parquet(close_df, raw_dir / "all_close_prices.parquet")
    logger.info("Combined close prices saved → shape {}", close_df.shape)

    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "ingest_prices.log", rotation="10 MB", retention="30 days")

    cfg = load_config()
    summary = ingest_all(cfg)

    print("\n=== Ingestion Summary ===")
    print(
        summary.to_string(
            index=False,
            columns=["ticker", "start_date", "end_date", "row_count", "nan_count"],
        )
    )
    print(f"\nTotal tickers ingested: {len(summary)}")
    sys.exit(0)
