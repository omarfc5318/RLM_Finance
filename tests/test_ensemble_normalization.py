"""
tests/test_ensemble_normalization.py
Unit tests for BaseEnsemble prediction normalization.

Tests:
  1. fit_normalizer: train-period stats are correct; regime risk-remap applied
  2. normalize_predictions: z-scores each column; regime remap skipped if already normalized
  3. predict_all: returns normalized predictions when norm_stats loaded
"""

import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_predictions(n_total: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-02", periods=n_total, freq="B")
    return pd.DataFrame(
        {
            "return_pred":        rng.normal(0.001, 0.005, n_total),
            "vol_pred":           rng.uniform(0.005, 0.020, n_total),
            "regime_pred":        rng.choice([0.0, 1.0, 2.0], n_total).astype(float),
            "drawdown_risk_prob": rng.uniform(0.1, 0.9, n_total),
        },
        index=dates,
    )


def _make_ensemble_stub() -> object:
    """Return a BaseEnsemble instance with __init__ dependencies bypassed."""
    from models.base.ensemble import BaseEnsemble
    ens = BaseEnsemble.__new__(BaseEnsemble)
    ens._return_model = MagicMock()
    ens._drawdown_model = MagicMock()
    ens._vol_df = pd.DataFrame()
    ens._regime_df = pd.DataFrame()
    ens._norm_stats = None
    return ens


# ---------------------------------------------------------------------------
# Test 1: fit_normalizer uses train-period stats and risk-remaps regime
# ---------------------------------------------------------------------------
def test_fit_normalizer_train_stats(tmp_path):
    """fit_normalizer stores mean/std from the train period; regime is risk-remapped."""
    import models.base.ensemble as ens_mod

    norm_path = tmp_path / "norm.joblib"
    original = ens_mod._NORMALIZER_PATH
    ens_mod._NORMALIZER_PATH = norm_path

    try:
        raw = _make_raw_predictions(200)
        train_end = raw.index[99]  # first 100 rows = train

        ens = _make_ensemble_stub()
        ens.fit_normalizer(raw, train_end)

        stats = ens._norm_stats
        assert stats is not None
        assert "return_pred" in stats
        assert "drawdown_risk_prob" in stats
        assert "vol_pred" in stats

        # return_pred stats must match train-period values exactly
        train_ret = raw.loc[raw.index <= train_end, "return_pred"].dropna()
        mu, sigma = stats["return_pred"]
        assert abs(mu - float(train_ret.mean())) < 1e-9
        assert abs(sigma - float(train_ret.std())) < 1e-9

        # regime_pred was risk-remapped {0→0, 2→1, 1→2} before computing stats
        remapped = raw.loc[raw.index <= train_end, "regime_pred"].map(
            {0.0: 0.0, 2.0: 1.0, 1.0: 2.0}
        )
        expected_mu = float(remapped.dropna().mean())
        assert abs(stats["regime_pred"][0] - expected_mu) < 1e-9
    finally:
        ens_mod._NORMALIZER_PATH = original


# ---------------------------------------------------------------------------
# Test 2: normalize_predictions z-scores columns; skips remap if already normalized
# ---------------------------------------------------------------------------
def test_normalize_predictions_zscores_and_remap_guard(tmp_path):
    """normalize_predictions z-scores columns; regime remap skipped on already-normalized input."""
    import models.base.ensemble as ens_mod

    norm_path = tmp_path / "norm.joblib"
    original = ens_mod._NORMALIZER_PATH
    ens_mod._NORMALIZER_PATH = norm_path

    try:
        raw = _make_raw_predictions(200)
        train_end = raw.index[-1]

        ens = _make_ensemble_stub()
        ens.fit_normalizer(raw, train_end)
        normed = ens.normalize_predictions(raw)

        # return_pred should be near zero-mean, unit-std on the fit data
        assert abs(normed["return_pred"].mean()) < 0.1
        assert abs(normed["return_pred"].std() - 1.0) < 0.1

        # regime_pred should be continuous z-scores, not {0, 1, 2}
        unique_after = set(normed["regime_pred"].dropna().unique())
        assert not unique_after.issubset({0.0, 1.0, 2.0})

        # Calling normalize on already-normalized output should not raise or NaN
        normed2 = ens.normalize_predictions(normed)
        assert normed2["return_pred"].notna().all()
        assert normed2["drawdown_risk_prob"].notna().all()
    finally:
        ens_mod._NORMALIZER_PATH = original


# ---------------------------------------------------------------------------
# Test 3: predict_all normalizes output when norm_stats are loaded
# ---------------------------------------------------------------------------
def test_predict_all_normalizes_when_stats_loaded():
    """predict_all returns z-scored predictions (return_pred mean≈0) when norm_stats present."""
    from models.base.ensemble import BaseEnsemble

    ens = _make_ensemble_stub()
    ens._norm_stats = {
        "return_pred":        (0.001, 0.005),
        "vol_pred":           (0.010, 0.003),
        "regime_pred":        (1.0,   0.816),
        "drawdown_risk_prob": (0.5,   0.2),
    }

    dates = pd.date_range("2020-01-02", periods=5, freq="B")

    ens._vol_df = pd.DataFrame({"garch_vol_5d": [0.010] * 5}, index=dates)
    ens._regime_df = pd.DataFrame(
        {"regime_filtered": [0.0, 1.0, 2.0, 0.0, 1.0]}, index=dates
    )
    ens._return_model.feature_names_in_ = ["f1"]
    ens._return_model.predict.return_value = np.full(5, 0.001)
    ens._drawdown_model.feature_names_in_ = ["f1"]
    ens._drawdown_model.predict_proba.return_value = np.column_stack(
        [np.full(5, 0.5), np.full(5, 0.5)]
    )

    X_price = pd.DataFrame({"f1": np.zeros(5)}, index=dates)
    X_joined = pd.DataFrame({"f1": np.zeros(5)}, index=dates)

    result = ens.predict_all(X_price, X_joined)

    # return_pred: (0.001 - 0.001) / 0.005 = 0.0
    assert abs(result["return_pred"].mean()) < 1e-9
    # drawdown_risk_prob: (0.5 - 0.5) / 0.2 = 0.0
    assert abs(result["drawdown_risk_prob"].mean()) < 1e-9
    # regime_pred: risk-remapped then z-scored → not {0, 1, 2}
    unique_vals = set(result["regime_pred"].dropna().unique())
    assert not unique_vals.issubset({0.0, 1.0, 2.0})
