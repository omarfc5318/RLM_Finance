"""
data/temporal_split.py
Temporal train / validation / test splitting with no-lookahead guarantees.
All split boundaries are derived from config.yaml; nothing is hardcoded here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator, Tuple

import pandas as pd
import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class TemporalSplitter:
    """
    Splits a time-indexed DataFrame into non-overlapping train / val / test
    partitions using boundaries defined in config.yaml.

    Attributes
    ----------
    train_end : pd.Timestamp
        Last date (inclusive) of the training set.
    val_end : pd.Timestamp
        Last date (inclusive) of the validation set.
    train_next : pd.Timestamp
        First business day after train_end — start of validation set.
    val_next : pd.Timestamp
        First business day after val_end — start of test set.
    """

    def __init__(self, config: dict) -> None:
        splits = config["splits"]
        self.train_end = pd.Timestamp(splits["train_end"])
        self.val_end = pd.Timestamp(splits["val_end"])

        # Advance to the next business day to prevent any single date appearing
        # in two splits.  pd.offsets.BDay(1) skips weekends but is NOT calendar-
        # aware: if train_end or val_end falls on a public holiday that is NOT a
        # weekend (e.g. Christmas on a Wednesday), BDay(1) will compute the next
        # Monday as the boundary start, silently creating a one-day gap in the
        # split coverage.  validate_no_overlap does not catch gaps — only overlaps
        # — so callers should verify total row counts sum to len(full_df) if this
        # matters for their use case.
        self.train_next = self.train_end + pd.offsets.BDay(1)
        self.val_next = self.val_end + pd.offsets.BDay(1)

        self._splits_dir = PROJECT_ROOT / config["paths"]["splits"]

        logger.info(
            "TemporalSplitter initialised — train ≤ {}  |  val {} → {}  |  test ≥ {}",
            self.train_end.date(),
            self.train_next.date(),
            self.val_end.date(),
            self.val_next.date(),
        )

    # ------------------------------------------------------------------
    # Primary splits
    # ------------------------------------------------------------------
    def get_train(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return all rows up to and including train_end."""
        split = df.loc[:self.train_end]
        logger.debug("Train split: {} rows [{} → {}]", len(split),
                     split.index[0].date(), split.index[-1].date())
        return split

    def get_val(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return rows from the first business day after train_end through val_end."""
        split = df.loc[self.train_next:self.val_end]
        logger.debug("Val split:   {} rows [{} → {}]", len(split),
                     split.index[0].date(), split.index[-1].date())
        return split

    def get_test(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return all rows from the first business day after val_end onwards."""
        split = df.loc[self.val_next:]
        logger.debug("Test split:  {} rows [{} → {}]", len(split),
                     split.index[0].date(), split.index[-1].date())
        return split

    # ------------------------------------------------------------------
    # Overlap validation
    # ------------------------------------------------------------------
    def validate_no_overlap(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """
        Assert that the three splits are strictly ordered with no shared dates.

        Raises
        ------
        ValueError
            If any input DataFrame is empty, or if any boundary condition is
            violated, with a message identifying which pair of splits overlap
            and the offending dates.
        """
        # Guard: empty inputs would cause index.min()/max() to return NaT,
        # producing silent incorrect comparisons rather than a meaningful error.
        empty = {
            "train": train_df.empty,
            "val": val_df.empty,
            "test": test_df.empty,
        }
        if any(empty.values()):
            empty_names = [name for name, is_empty in empty.items() if is_empty]
            raise ValueError(
                f"validate_no_overlap received empty DataFrame(s): {empty_names}. "
                "All three splits must contain at least one row."
            )

        train_max = train_df.index.max()
        val_min = val_df.index.min()
        val_max = val_df.index.max()
        test_min = test_df.index.min()

        if train_max >= val_min:
            overlap = train_df.index.intersection(val_df.index).tolist()
            raise ValueError(
                f"Train/val overlap detected: train ends {train_max.date()}, "
                f"val starts {val_min.date()}. "
                f"Overlapping dates ({len(overlap)}): {overlap[:5]}"
            )

        if val_max >= test_min:
            overlap = val_df.index.intersection(test_df.index).tolist()
            raise ValueError(
                f"Val/test overlap detected: val ends {val_max.date()}, "
                f"test starts {test_min.date()}. "
                f"Overlapping dates ({len(overlap)}): {overlap[:5]}"
            )

        logger.info(
            "validate_no_overlap passed — train max {} < val min {} < test min {}",
            train_max.date(), val_min.date(), test_min.date(),
        )

    # ------------------------------------------------------------------
    # Walk-forward rolling windows
    # ------------------------------------------------------------------
    def get_rolling_windows(
        self,
        df: pd.DataFrame,
        window_size_months: int = 12,
        step_months: int = 1,
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Yield (train_df, test_df) tuples for expanding-window walk-forward validation.

        Each test window is exactly one month wide; the training window grows by
        one month on each step.  Rows beyond the configured val_end are excluded
        so walk-forward never touches held-out test data.

        Parameters
        ----------
        df : pd.DataFrame
            Full time-indexed DataFrame (should be the training+val portion only).
        window_size_months : int
            Initial training window size in months.
        step_months : int
            How many months to advance on each iteration.

        Yields
        ------
        (train_df, test_df) : tuple of DataFrames
        """
        # Hard-clip to val_end regardless of what the caller passes in.
        # Without this, passing the full dataset would silently include test
        # dates in the walk-forward folds, leaking held-out information.
        df = df.loc[:self.val_end]

        start = df.index.min()
        end = self.val_end  # never leak into the held-out test set

        train_end = start + pd.DateOffset(months=window_size_months) - pd.offsets.BDay(1)
        step = 0

        while True:
            test_start = train_end + pd.offsets.BDay(1)
            test_end = test_start + pd.DateOffset(months=step_months) - pd.offsets.BDay(1)

            if test_end > end:
                break

            train_df = df.loc[:train_end]
            test_df = df.loc[test_start:test_end]

            if train_df.empty or test_df.empty:
                break

            logger.debug(
                "Rolling window {} — train [{} → {}]  test [{} → {}]",
                step,
                train_df.index[0].date(), train_df.index[-1].date(),
                test_df.index[0].date(), test_df.index[-1].date(),
            )
            yield train_df, test_df

            train_end += pd.DateOffset(months=step_months)
            step += 1

    # ------------------------------------------------------------------
    # Persist split metadata
    # ------------------------------------------------------------------
    def save_split_info(self, df: pd.DataFrame) -> Path:
        """
        Write split boundary dates to data/splits/split_info.json.

        Parameters
        ----------
        df : pd.DataFrame
            The full DataFrame used for splitting (needed to derive train_start
            and test_end from the actual index).

        Returns
        -------
        Path to the written JSON file.
        """
        self._splits_dir.mkdir(parents=True, exist_ok=True)

        info = {
            "train_start": df.index.min().date().isoformat(),
            "train_end": self.train_end.date().isoformat(),
            "val_start": self.train_next.date().isoformat(),
            "val_end": self.val_end.date().isoformat(),
            "test_start": self.val_next.date().isoformat(),
            "test_end": df.index.max().date().isoformat(),
        }

        out_path = self._splits_dir / "split_info.json"
        with open(out_path, "w") as fh:
            json.dump(info, fh, indent=2)

        logger.info("Split info saved → {}", out_path.relative_to(PROJECT_ROOT))
        for key, val in info.items():
            logger.debug("  {}: {}", key, val)

        return out_path


# ---------------------------------------------------------------------------
# Entry point — smoke test against all_close_prices if available
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    splitter = TemporalSplitter(cfg)

    close_path = PROJECT_ROOT / cfg["paths"]["raw"] / "all_close_prices.parquet"
    if close_path.exists():
        df = pd.read_parquet(close_path)
        train = splitter.get_train(df)
        val = splitter.get_val(df)
        test = splitter.get_test(df)
        splitter.validate_no_overlap(train, val, test)
        splitter.save_split_info(df)

        print(f"Train: {len(train):>5} rows  [{train.index[0].date()} → {train.index[-1].date()}]")
        print(f"Val:   {len(val):>5} rows  [{val.index[0].date()} → {val.index[-1].date()}]")
        print(f"Test:  {len(test):>5} rows  [{test.index[0].date()} → {test.index[-1].date()}]")
        print(f"Total: {len(train) + len(val) + len(test)} / {len(df)} rows")
    else:
        print(f"No parquet found at {close_path} — run ingest_prices.py first.")
