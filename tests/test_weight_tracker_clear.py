"""
tests/test_weight_tracker_clear.py
Unit tests for WeightTracker.clear().
"""

import pandas as pd
import pytest

from models.meta.weight_tracker import WeightTracker


def _log_row(wt: WeightTracker, date: str) -> None:
    wt.log_weights(
        pd.Timestamp(date),
        {"return": 0.25, "vol": 0.25, "regime": 0.25, "drawdown": 0.25},
    )


def _data_rows(path) -> int:
    return len(pd.read_csv(path))


def test_clear_resets_row_count(tmp_path):
    """After 3 rows, clear(), then 2 more rows — only 2 rows remain (not 5)."""
    wt = WeightTracker(path=str(tmp_path / "audit.csv"))

    _log_row(wt, "2023-01-03")
    _log_row(wt, "2023-01-04")
    _log_row(wt, "2023-01-05")
    assert _data_rows(wt.path) == 3, "Expected 3 rows before clear()"

    wt.clear()
    assert _data_rows(wt.path) == 0, "Expected 0 data rows immediately after clear()"

    _log_row(wt, "2023-01-06")
    _log_row(wt, "2023-01-09")
    assert _data_rows(wt.path) == 2, "Expected 2 rows after clear() + 2 appends, not 5"


def test_clear_preserves_header(tmp_path):
    """clear() leaves a readable CSV with the correct column names and zero data rows."""
    wt = WeightTracker(path=str(tmp_path / "audit.csv"))
    _log_row(wt, "2023-01-03")
    wt.clear()

    df = pd.read_csv(wt.path)
    assert list(df.columns) == ["date", "return_w", "vol_w", "regime_w", "drawdown_w"]
    assert len(df) == 0


def test_clear_then_log_preserves_values(tmp_path):
    """Rows logged after clear() contain the correct values."""
    wt = WeightTracker(path=str(tmp_path / "audit.csv"))
    _log_row(wt, "2023-01-03")
    wt.clear()

    wt.log_weights(
        pd.Timestamp("2023-01-10"),
        {"return": 0.30, "vol": 0.20, "regime": 0.28, "drawdown": 0.22},
    )
    df = pd.read_csv(wt.path, parse_dates=["date"])
    assert len(df) == 1
    assert abs(df.iloc[0]["return_w"] - 0.30) < 1e-9


def test_clear_on_new_tracker_creates_header(tmp_path):
    """clear() on a tracker whose file has never been written creates the header-only file."""
    wt = WeightTracker(path=str(tmp_path / "fresh.csv"))
    wt.clear()

    df = pd.read_csv(wt.path)
    assert list(df.columns) == ["date", "return_w", "vol_w", "regime_w", "drawdown_w"]
    assert len(df) == 0
