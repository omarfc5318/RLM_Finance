"""
tests/test_temporal_split.py
Unit tests for TemporalSplitter.
All tests use synthetic DataFrames — no external data or files required.
"""

from __future__ import annotations

import pytest
import pandas as pd

from data.temporal_split import TemporalSplitter

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MOCK_CONFIG = {
    "splits": {
        "train_end": "2021-12-31",
        "val_end": "2023-06-30",
    },
    "paths": {
        "splits": "data/splits/",
    },
}


@pytest.fixture
def splitter() -> TemporalSplitter:
    return TemporalSplitter(MOCK_CONFIG)


@pytest.fixture
def full_df() -> pd.DataFrame:
    """Business-day index spanning 2015-01-01 → 2024-12-31."""
    idx = pd.bdate_range(start="2015-01-01", end="2024-12-31")
    return pd.DataFrame({"value": range(len(idx))}, index=idx)


# ---------------------------------------------------------------------------
# Test 1 — No date overlap between any two splits
# ---------------------------------------------------------------------------

def test_no_date_overlap(splitter, full_df):
    """
    Each date must appear in exactly one split.
    Train ∩ Val, Train ∩ Test, and Val ∩ Test must all be empty.
    """
    train = splitter.get_train(full_df)
    val = splitter.get_val(full_df)
    test = splitter.get_test(full_df)

    train_val_overlap = train.index.intersection(val.index)
    train_test_overlap = train.index.intersection(test.index)
    val_test_overlap = val.index.intersection(test.index)

    assert len(train_val_overlap) == 0, (
        f"Train and val share {len(train_val_overlap)} date(s): "
        f"{train_val_overlap[:5].tolist()}"
    )
    assert len(train_test_overlap) == 0, (
        f"Train and test share {len(train_test_overlap)} date(s): "
        f"{train_test_overlap[:5].tolist()}"
    )
    assert len(val_test_overlap) == 0, (
        f"Val and test share {len(val_test_overlap)} date(s): "
        f"{val_test_overlap[:5].tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Train + val + test covers the full date range with no gaps
# ---------------------------------------------------------------------------

def test_full_coverage_no_gaps(splitter, full_df):
    """
    The union of all three splits must equal the full DataFrame index exactly —
    no dates dropped, no gaps introduced by the business-day boundary offsets.
    """
    train = splitter.get_train(full_df)
    val = splitter.get_val(full_df)
    test = splitter.get_test(full_df)

    combined_index = train.index.union(val.index).union(test.index)

    # Same length
    assert len(combined_index) == len(full_df), (
        f"Combined splits have {len(combined_index)} rows but full_df has "
        f"{len(full_df)}. Missing: "
        f"{full_df.index.difference(combined_index)[:5].tolist()}"
    )

    # Same dates
    assert combined_index.equals(full_df.index), (
        "Combined split index does not match full_df index. "
        f"In full_df but not splits: {full_df.index.difference(combined_index)[:5].tolist()} | "
        f"In splits but not full_df: {combined_index.difference(full_df.index)[:5].tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 3 — validate_no_overlap does NOT raise on the splitter's own output
# ---------------------------------------------------------------------------

def test_validate_no_overlap_passes_on_clean_splits(splitter, full_df):
    """
    validate_no_overlap must not raise when given the splits produced by the
    splitter itself.  This proves that get_train / get_val / get_test produce
    correctly ordered, non-overlapping partitions end-to-end.
    """
    train = splitter.get_train(full_df)
    val = splitter.get_val(full_df)
    test = splitter.get_test(full_df)

    # Should complete without raising
    splitter.validate_no_overlap(train, val, test)
