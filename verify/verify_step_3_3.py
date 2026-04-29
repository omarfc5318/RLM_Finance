"""
verify/verify_step_3_3.py
5 perturbation-based causality tests for FeedbackLoopEngine.

Tests:
  T1 — STEP A is independent of actual_return (same signal regardless)
  T2 — validate_causality() passes on a clean run
  T3 — validate_causality() detects an injected violation
  T4 — _return_history buffer grows by one per step with actual_return
  T5 — PnL pairs signal[t-1] * actual_return[t], not signal[t]
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from engine.feedback import FeedbackLoopEngine
from models.meta.weight_tracker import WeightTracker


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _mock_ensemble(return_pred: float = 0.01) -> MagicMock:
    ens = MagicMock()
    ens.predict_all.return_value = pd.DataFrame(
        {
            "return_pred":        [return_pred],
            "vol_pred":           [0.01],
            "regime_pred":        [0.0],
            "drawdown_risk_prob": [0.2],
        }
    )
    return ens


def _mock_meta_learner() -> MagicMock:
    ml = MagicMock()
    ml.models = None
    ml.feature_cols = None
    return ml


def _features_row() -> dict:
    row = pd.DataFrame({"x": [1.0]})
    return {"price": row, "joined": row}


def _make_engine(return_pred: float = 0.01, tmp_path=None) -> FeedbackLoopEngine:
    wt_path = str(tmp_path / "wt.csv") if tmp_path else "/tmp/wt_test.csv"
    return FeedbackLoopEngine(
        ensemble=_mock_ensemble(return_pred),
        meta_learner=_mock_meta_learner(),
        weight_tracker=WeightTracker(path=wt_path),
    )


# ---------------------------------------------------------------------------
# T1 — STEP A does not use actual_return
# ---------------------------------------------------------------------------

def test_step_a_independent_of_actual_return(tmp_path):
    """
    The same features_row must produce the same weighted signal whether
    actual_return is None, 0.0, or a large number.
    """
    date = pd.Timestamp("2023-01-03")
    feat = _features_row()

    signals = []
    for ar in [None, 0.0, 0.05, -9999.0]:
        eng = _make_engine(tmp_path=tmp_path)
        sig = eng.step(date, feat, actual_return=ar)
        signals.append(sig)

    assert all(s == signals[0] for s in signals), (
        f"Signal changed with actual_return: {signals}"
    )


# ---------------------------------------------------------------------------
# T2 — validate_causality() passes on a clean run
# ---------------------------------------------------------------------------

def test_validate_causality_passes_clean_run(tmp_path):
    """
    After stepping through 5 consecutive trading days, every performance_log
    entry must have feedback_date > prediction_date.
    """
    eng = _make_engine(tmp_path=tmp_path)
    dates = pd.date_range("2023-01-03", periods=5, freq="B")

    for i, t in enumerate(dates):
        ar = 0.005 * (i + 1) if i > 0 else None
        eng.step(t, _features_row(), actual_return=ar)

    assert eng.validate_causality() is True


# ---------------------------------------------------------------------------
# T3 — validate_causality() detects an injected violation
# ---------------------------------------------------------------------------

def test_validate_causality_detects_violation(tmp_path):
    """
    Manually append a log entry where feedback_date == prediction_date.
    validate_causality() must return False.
    """
    eng = _make_engine(tmp_path=tmp_path)
    eng.step(pd.Timestamp("2023-01-03"), _features_row(), actual_return=None)

    eng.performance_log.append({
        "prediction_date": pd.Timestamp("2023-01-05"),
        "feedback_date":   pd.Timestamp("2023-01-05"),  # equal — violation
        "signal": 0.01,
        "actual_return": 0.005,
        "pnl": 0.00005,
        "weights": {"return": 0.25, "vol": 0.25, "regime": 0.25, "drawdown": 0.25},
    })

    assert eng.validate_causality() is False


# ---------------------------------------------------------------------------
# T4 — _return_history buffer grows by one per step with actual_return
# ---------------------------------------------------------------------------

def test_return_history_grows_per_step(tmp_path):
    """
    Each step() with non-None actual_return appends exactly one entry to
    _return_history. Steps without actual_return must not append.
    """
    eng = _make_engine(tmp_path=tmp_path)
    dates = pd.date_range("2023-01-03", periods=6, freq="B")

    assert len(eng._return_history) == 0

    eng.step(dates[0], _features_row(), actual_return=None)
    assert len(eng._return_history) == 0, "No append expected without actual_return"

    for i, t in enumerate(dates[1:], start=1):
        eng.step(t, _features_row(), actual_return=0.001 * i)
        assert len(eng._return_history) == i, (
            f"Expected {i} entries after step {i}, got {len(eng._return_history)}"
        )


# ---------------------------------------------------------------------------
# T5 — PnL pairs signal[t-1] * actual_return[t], not signal[t]
# ---------------------------------------------------------------------------

def test_pnl_pairs_yesterday_signal_with_today_return(tmp_path):
    """
    On day t+1, the engine receives actual_return and logs a PnL entry.
    Contract: pnl = signal[t] * actual_return[t+1], not signal[t+1].

    prediction_date must be dates[0], feedback_date must be dates[1].
    """
    ret_pred   = 0.05
    actual_ret = 0.03

    eng = _make_engine(return_pred=ret_pred, tmp_path=tmp_path)
    dates = pd.date_range("2023-01-03", periods=3, freq="B")

    sig_t0 = eng.step(dates[0], _features_row(), actual_return=None)
    sig_t1 = eng.step(dates[1], _features_row(), actual_return=actual_ret)

    assert len(eng.performance_log) == 1, (
        f"Expected 1 PnL entry, got {len(eng.performance_log)}"
    )

    entry = eng.performance_log[0]
    assert entry["prediction_date"] == dates[0], (
        f"prediction_date should be dates[0]={dates[0]}, got {entry['prediction_date']}"
    )
    assert entry["feedback_date"] == dates[1], (
        f"feedback_date should be dates[1]={dates[1]}, got {entry['feedback_date']}"
    )

    expected_pnl = sig_t0 * actual_ret
    assert abs(entry["pnl"] - expected_pnl) < 1e-12, (
        f"pnl={entry['pnl']:.8f} != sig_t0*ar={expected_pnl:.8f}. "
        f"Engine used sig_t1={sig_t1:.8f} instead of sig_t0={sig_t0:.8f}?"
    )
