"""
models/base/ensemble.py
Aggregates predictions from all four base models into a single DataFrame.

Model inventory
---------------
  return_pred        : XGBRegressor (spy_ret1d.joblib) — 1-day forward return
                       Features: price_features_v1 only (99 cols)
  vol_pred           : pre-computed GARCH forecasts (vol_forecasts.parquet)
  regime_pred        : causal HMM labels (regimes.parquet, regime_filtered col)
  drawdown_risk_prob : XGBClassifier (drawdown_estimator.joblib) — P(MDD ≥ 5%)
                       Features: price + macro joined (111 cols, vix_regime ordinal-encoded)

Column semantic directions (for downstream use)
-----------------------------------------------
  return_pred        : +ve = bullish (risk-on)
  vol_pred           : +ve = high vol (risk-off)  [daily std, not annualized]
  regime_pred        : 0=bull, 1=bear, 2=sideways (categorical, NOT risk-ordered)
  drawdown_risk_prob : [0, 1], +ve = high drawdown risk (risk-off)

Prediction normalization
------------------------
  fit_normalizer(full_predictions_df, train_end) computes per-column mean/std
  on the TRAIN period only (causal-safe). regime_pred is first risk-remapped
  (bull=0, sideways=1, bear=2) before statistics are computed.

  vol_pred has no train-period data (GARCH was fit only on val+test in Step 2.3).
  Fallback: use the first ≤252 non-NaN rows of vol_pred for normalizer stats,
  regardless of date.

  After fitting, predict_all() calls normalize_predictions() which z-scores
  each raw prediction using the stored stats. The normalized regime_pred is
  a continuous z-score, no longer in {0, 1, 2}.

  Normalizer stats are persisted to models/base/prediction_normalizer.joblib
  so FeedbackLoopEngine can reload them without re-fitting.

Disagreement metric
-------------------
  Each column is standardised (zero mean, unit variance) using the VAL-period
  statistics only (no lookahead into test). The val period is used instead of
  train because vol_forecasts.parquet does not cover the train period — GARCH
  was evaluated on val+test only in Step 2.3. Regime is remapped to a risk-ordered integer
  (bull=0, sideways=1, bear=2) for the disagreement computation only; the
  regime_pred output column keeps its original HMM labeling (unless already
  normalized, in which case the remap is skipped — see compute_disagreement guard).

  Row-wise std of the four standardised signals = disagreement score.

  KNOWN LIMITATION: This metric measures dispersion-after-zscoring, not
  semantic disagreement. A day where all 4 models point risk-off at +2σ has
  near-zero disagreement even though return_pred=+2σ and vol_pred=+2σ are
  semantically contradictory. A principled unified risk-off-axis metric is
  planned for Phase 3 — see KNOWN_ISSUES.md.

Output
------
  data/processed/ensemble_predictions.parquet
  Columns: return_pred, vol_pred, regime_pred, drawdown_risk_prob, disagreement
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

_MODEL_DIR = PROJECT_ROOT / "models" / "base"
_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RETURN_MODEL_PATH   = _MODEL_DIR / "spy_ret1d.joblib"
DRAWDOWN_MODEL_PATH = _MODEL_DIR / "drawdown_estimator.joblib"
VOL_FORECASTS_PATH  = _PROCESSED_DIR / "vol_forecasts.parquet"
REGIMES_PATH        = _PROCESSED_DIR / "regimes.parquet"
OUTPUT_PATH         = _PROCESSED_DIR / "ensemble_predictions.parquet"
_NORMALIZER_PATH    = _MODEL_DIR / "prediction_normalizer.joblib"

# Risk-ordered remap for regime_pred: HMM labels 0=bull, 1=bear, 2=sideways
# → risk-ordered 0=low risk (bull), 1=medium (sideways), 2=high risk (bear)
_REGIME_RISK_MAP: Dict[float, float] = {0.0: 0.0, 2.0: 1.0, 1.0: 2.0}

_PRED_COLS = ["return_pred", "vol_pred", "regime_pred", "drawdown_risk_prob"]


def _load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
class BaseEnsemble:
    """
    Loads all four base models and produces a unified prediction DataFrame.

    The return predictor and drawdown estimator are loaded from joblib.
    Vol forecasts and regime labels are loaded from pre-computed parquets
    (produced by volatility_predictor.py and regime_classifier.py).
    """

    def __init__(self) -> None:
        missing = []
        for p in (RETURN_MODEL_PATH, DRAWDOWN_MODEL_PATH,
                  VOL_FORECASTS_PATH, REGIMES_PATH):
            if not p.exists():
                missing.append(str(p.relative_to(PROJECT_ROOT)))
        if missing:
            raise FileNotFoundError(
                "Missing required files — run each model's __main__ first:\n  "
                + "\n  ".join(missing)
            )

        self._return_model   = joblib.load(RETURN_MODEL_PATH)
        self._drawdown_model = joblib.load(DRAWDOWN_MODEL_PATH)
        self._vol_df         = pd.read_parquet(VOL_FORECASTS_PATH)
        self._regime_df      = pd.read_parquet(REGIMES_PATH)

        # Load normalizer stats if they exist (dict of {col: (mean, std)})
        self._norm_stats: dict | None = None
        if _NORMALIZER_PATH.exists():
            self._norm_stats = joblib.load(_NORMALIZER_PATH)
            logger.info(
                "BaseEnsemble: loaded normalizer from {} (cols: {})",
                _NORMALIZER_PATH.name,
                list(self._norm_stats.keys()),
            )

        logger.info(
            "BaseEnsemble loaded — return model: {}  drawdown model: {}  "
            "vol rows: {}  regime rows: {}",
            RETURN_MODEL_PATH.name,
            DRAWDOWN_MODEL_PATH.name,
            len(self._vol_df),
            len(self._regime_df),
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def fit_normalizer(
        self,
        full_predictions_df: pd.DataFrame,
        train_end: pd.Timestamp,
    ) -> None:
        """
        Compute per-column mean/std on the TRAIN period and persist to disk.

        regime_pred is risk-remapped before statistics are computed so the
        normalizer operates on a monotone risk scale (0/1/2) not arbitrary
        HMM labels.

        vol_pred has no train-period coverage (GARCH fit on val+test only).
        Fallback: use the first ≤252 non-NaN rows of vol_pred.

        Parameters
        ----------
        full_predictions_df : pd.DataFrame
            Output of predict_all() covering the full date range.
        train_end : pd.Timestamp
            Last inclusive date of the train period.
        """
        df = full_predictions_df.copy()

        # Risk-remap regime before computing stats
        if "regime_pred" in df.columns:
            df["regime_pred"] = df["regime_pred"].map(_REGIME_RISK_MAP)

        train_df = df.loc[df.index <= train_end]

        norm_stats: dict = {}
        for col in _PRED_COLS:
            if col not in df.columns:
                continue
            if col == "vol_pred":
                # vol_pred has no train data — fall back to first 252 non-NaN rows
                available = df[col].dropna()
                ref = available.iloc[:252] if len(available) >= 252 else available
                if len(ref) == 0:
                    logger.warning("fit_normalizer: vol_pred has no non-NaN rows — skip")
                    continue
                mu, sigma = float(ref.mean()), float(ref.std())
                logger.info(
                    "fit_normalizer: vol_pred fallback — {} rows (first non-NaN)",
                    len(ref),
                )
            else:
                col_data = train_df[col].dropna()
                if len(col_data) < 20:
                    logger.warning(
                        "fit_normalizer: {} has only {} train rows — skip", col, len(col_data)
                    )
                    continue
                mu, sigma = float(col_data.mean()), float(col_data.std())
            if sigma == 0:
                logger.warning("fit_normalizer: {} has zero std — skip", col)
                continue
            norm_stats[col] = (mu, sigma)
            logger.info(
                "fit_normalizer: {} → mean={:.6f}  std={:.6f}  n={}",
                col, mu, sigma,
                len(train_df[col].dropna()) if col != "vol_pred" else "fallback",
            )

        self._norm_stats = norm_stats
        _NORMALIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(norm_stats, _NORMALIZER_PATH)
        logger.info("Normalizer saved → {}", _NORMALIZER_PATH)

    def normalize_predictions(self, preds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Z-score each prediction column using the stored normalizer stats.

        regime_pred is risk-remapped to {0, 1, 2} before z-scoring only if its
        values are still raw HMM labels (subset of {0.0, 1.0, 2.0}); if already
        normalized (continuous z-scores), the remap is skipped.

        Parameters
        ----------
        preds_df : pd.DataFrame
            Raw predictions (output of predict_all before normalization).

        Returns
        -------
        pd.DataFrame with the same columns, each normalized.
        Columns without normalizer stats are passed through unchanged.

        Raises
        ------
        RuntimeError if fit_normalizer() has not been called (no stats loaded).
        """
        if self._norm_stats is None:
            raise RuntimeError(
                "normalize_predictions: no normalizer stats — call fit_normalizer() first "
                "or ensure prediction_normalizer.joblib exists."
            )

        out = preds_df.copy()

        # Risk-remap regime if still raw labels
        if "regime_pred" in out.columns:
            unique_vals = set(out["regime_pred"].dropna().unique())
            if unique_vals.issubset({0.0, 1.0, 2.0}):
                out["regime_pred"] = out["regime_pred"].map(_REGIME_RISK_MAP)

        for col, (mu, sigma) in self._norm_stats.items():
            if col in out.columns:
                out[col] = (out[col] - mu) / sigma

        return out

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_all(
        self,
        X_price: pd.DataFrame,
        X_joined: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run all four models on their respective feature matrices.

        Parameters
        ----------
        X_price : pd.DataFrame
            Date-indexed price features only (99 cols from price_features_v1.parquet).
            Consumed by the return model.
        X_joined : pd.DataFrame
            Date-indexed joined matrix (price + macro, 111 cols, vix_regime ordinal-encoded).
            Consumed by the drawdown model.

        Returns
        -------
        pd.DataFrame with columns:
          return_pred, vol_pred, regime_pred, drawdown_risk_prob
        Index = union of X_price.index and X_joined.index (typically they match).
        """
        # Align X_price columns to the saved return model's expected feature order
        expected_price_cols = list(self._return_model.feature_names_in_)
        missing = set(expected_price_cols) - set(X_price.columns)
        if missing:
            raise ValueError(
                f"X_price missing {len(missing)} features expected by return model: "
                f"{sorted(missing)[:10]}"
            )
        X_price_aligned = X_price[expected_price_cols]

        # Align X_joined columns to the saved drawdown model's expected feature order
        expected_joined_cols = list(self._drawdown_model.feature_names_in_)
        missing = set(expected_joined_cols) - set(X_joined.columns)
        if missing:
            raise ValueError(
                f"X_joined missing {len(missing)} features expected by drawdown model: "
                f"{sorted(missing)[:10]}"
            )
        X_joined_aligned = X_joined[expected_joined_cols]

        # Return prediction (XGBRegressor on price features)
        return_pred = pd.Series(
            self._return_model.predict(X_price_aligned),
            index=X_price_aligned.index,
            name="return_pred",
        )

        # Drawdown risk (XGBClassifier on joined features)
        drawdown_prob = pd.Series(
            self._drawdown_model.predict_proba(X_joined_aligned)[:, 1],
            index=X_joined_aligned.index,
            name="drawdown_risk_prob",
        )

        # Vol forecast — garch_vol_5d column, reindexed
        vol_col = "garch_vol_5d" if "garch_vol_5d" in self._vol_df.columns else self._vol_df.columns[0]
        vol_pred = self._vol_df[vol_col].reindex(X_price_aligned.index).rename("vol_pred")

        # Regime label — causal filtered column
        regime_col = "regime_filtered" if "regime_filtered" in self._regime_df.columns else self._regime_df.columns[0]
        regime_pred = self._regime_df[regime_col].reindex(X_price_aligned.index).rename("regime_pred")

        predictions = pd.concat(
            [return_pred, vol_pred, regime_pred, drawdown_prob], axis=1
        )

        nan_counts = predictions.isna().sum()
        if nan_counts.any():
            logger.warning(
                "predict_all: NaN counts per column — {}",
                nan_counts[nan_counts > 0].to_dict(),
            )

        if self._norm_stats is not None:
            predictions = self.normalize_predictions(predictions)
            logger.debug("predict_all: predictions normalized")

        logger.info(
            "predict_all: {} rows  [{} → {}]",
            len(predictions),
            predictions.index.min().date(),
            predictions.index.max().date(),
        )
        return predictions

    # ------------------------------------------------------------------
    # Disagreement
    # ------------------------------------------------------------------
    def compute_disagreement(
        self,
        predictions_df: pd.DataFrame,
        fit_start: str | pd.Timestamp | None = None,
        fit_end: str | pd.Timestamp | None = None,
    ) -> pd.Series:
        """
        Measure cross-model disagreement as row-wise std of standardised signals.

        The StandardScaler is fit on a fixed calibration window — typically the
        val period — to avoid lookahead into the test period. Rows outside
        [fit_start, fit_end] are standardised using the frozen fit-period
        statistics.

        Regime is temporarily remapped to a risk-ordered integer
        (bull=0, sideways=1, bear=2) for the scaler input only; the regime_pred
        output column is not modified.

        Parameters
        ----------
        predictions_df : pd.DataFrame
            Output of predict_all().
        fit_start : str | pd.Timestamp | None
            Inclusive start date for the scaler-fit window. If None, uses the
            earliest date in predictions_df where all 4 columns are non-NaN.
        fit_end : str | pd.Timestamp | None
            Inclusive end date for the scaler-fit window. If None, fits on all
            complete rows (lookahead — logs a warning).

        Returns
        -------
        pd.Series of disagreement scores, same index as predictions_df.
        Rows outside the complete-prediction window will be NaN.

        Known limitations: see module docstring — disagreement-in-z-score-space
        does not equal disagreement-in-semantic-direction.
        """
        cols = ["return_pred", "vol_pred", "regime_pred", "drawdown_risk_prob"]
        available = [c for c in cols if c in predictions_df.columns]
        df = predictions_df[available].copy()

        # Risk-ordered regime remap: bull=0, sideways=1, bear=2
        # (original HMM labels: 0=bull, 1=bear, 2=sideways)
        # Guard: skip remap if regime_pred is already normalized (z-scores, not {0,1,2})
        if "regime_pred" in df.columns:
            unique_vals = set(df["regime_pred"].dropna().unique())
            if unique_vals.issubset({0.0, 1.0, 2.0}):
                df["regime_pred"] = df["regime_pred"].map(_REGIME_RISK_MAP)

        scaler = StandardScaler()
        mask_complete = df.notna().all(axis=1)

        if mask_complete.sum() == 0:
            raise ValueError(
                "No rows have all 4 prediction columns non-NaN — cannot compute disagreement."
            )

        if fit_start is not None or fit_end is not None:
            fit_mask = mask_complete.copy()
            if fit_start is not None:
                fit_mask &= (df.index >= pd.Timestamp(fit_start))
            if fit_end is not None:
                fit_mask &= (df.index <= pd.Timestamp(fit_end))
            if fit_mask.sum() == 0:
                raise ValueError(
                    f"No complete rows in fit window [{fit_start}, {fit_end}] "
                    f"to fit the disagreement scaler."
                )
            scaler.fit(df[fit_mask])
            logger.info(
                "compute_disagreement: scaler fit on {} rows in [{}, {}]",
                int(fit_mask.sum()),
                pd.Timestamp(fit_start).date() if fit_start else "min",
                pd.Timestamp(fit_end).date() if fit_end else "max",
            )
        else:
            logger.warning(
                "compute_disagreement: no fit window provided — fitting scaler on "
                "ALL complete rows (lookahead). Pass fit_start/fit_end for production use."
            )
            scaler.fit(df[mask_complete])

        scaled_df = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
        scaled_df.loc[mask_complete] = scaler.transform(df[mask_complete])

        disagreement = scaled_df.std(axis=1, ddof=1).rename("disagreement")
        logger.info(
            "compute_disagreement: mean={:.4f}  std={:.4f}  max={:.4f}  non_nan={}",
            float(disagreement.mean()),
            float(disagreement.std()),
            float(disagreement.max()),
            int(disagreement.notna().sum()),
        )
        return disagreement

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_predictions(
        self,
        predictions_df: pd.DataFrame,
        fit_start: str | pd.Timestamp | None = None,
        fit_end: str | pd.Timestamp | None = None,
    ) -> Path:
        """
        Add disagreement column to predictions_df and save to parquet.
        fit_start/fit_end are forwarded to compute_disagreement for the scaler fit window.
        """
        disagreement = self.compute_disagreement(
            predictions_df, fit_start=fit_start, fit_end=fit_end,
        )
        out_df = predictions_df.copy()
        out_df["disagreement"] = disagreement

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(OUTPUT_PATH)
        logger.info(
            "Ensemble predictions saved → {} ({} rows, {} cols)",
            OUTPUT_PATH.relative_to(PROJECT_ROOT),
            len(out_df),
            len(out_df.columns),
        )
        return OUTPUT_PATH


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "ensemble.log", rotation="10 MB", retention="30 days")

    cfg = _load_config()
    processed_dir = PROJECT_ROOT / cfg["paths"]["processed"]

    features_path = processed_dir / "price_features_v1.parquet"
    macro_path    = processed_dir / "macro_features_v1.parquet"

    for p in (features_path, macro_path):
        if not p.exists():
            print(f"Missing {p} — run feature_engineering.py first.")
            sys.exit(1)

    # Load price features (for Model A: return predictor)
    X_price = pd.read_parquet(features_path)
    logger.info("Loaded price features: {} rows × {} cols", len(X_price), len(X_price.columns))

    # Build joined matrix exactly as drawdown_estimator.py does (for Model D)
    macro_df = pd.read_parquet(macro_path)
    X_joined = X_price.join(macro_df, how="left").ffill()
    if "vix_regime" in X_joined.columns:
        vix_map = {"low": 0, "medium": 1, "high": 2}
        X_joined["vix_regime"] = X_joined["vix_regime"].map(vix_map).astype("float64")
    logger.info(
        "Built joined matrix: {} rows × {} cols (99 price + {} macro)",
        len(X_joined), len(X_joined.columns), len(macro_df.columns),
    )

    try:
        ensemble = BaseEnsemble()
    except FileNotFoundError as exc:
        print(f"\nCannot build ensemble:\n{exc}")
        sys.exit(1)

    from data.temporal_split import TemporalSplitter
    splitter  = TemporalSplitter(cfg)
    train_end = splitter.train_end
    fit_start = splitter.train_next        # first business day after train_end
    fit_end   = splitter.val_end if hasattr(splitter, "val_end") else splitter.test_start - pd.Timedelta(days=1)

    # Produce raw (un-normalized) predictions first, then fit normalizer
    # on train period so that predict_all() can normalize on the second call.
    raw_predictions = ensemble.predict_all(X_price, X_joined)

    logger.info(
        "Fitting prediction normalizer on train period (up to {})",
        train_end.date(),
    )
    ensemble.fit_normalizer(raw_predictions, train_end)

    # Now predict_all() will auto-normalize (normalizer stats loaded in __init__)
    predictions = ensemble.predict_all(X_price, X_joined)

    # The disagreement scaler fits on the VAL period only — vol_forecasts
    # doesn't cover the train period (GARCH was only evaluated on val+test
    # in Step 2.3). Fitting on val avoids lookahead into test.
    logger.info(
        "Disagreement scaler fit window: [{}, {}] (val period)",
        fit_start.date(), pd.Timestamp(fit_end).date(),
    )

    out_path = ensemble.save_predictions(
        predictions, fit_start=fit_start, fit_end=fit_end,
    )

    # Reload saved file to show disagreement stats in the console summary
    saved = pd.read_parquet(out_path)

    print("\n=== Ensemble Predictions Summary ===")
    print(f"Rows:       {len(saved)}")
    print(f"Date range: {saved.index.min().date()} → {saved.index.max().date()}")
    print(f"\nPer-column stats:")
    print(saved.describe().round(4).to_string())
    print(f"\nDisagreement non-NaN: {saved['disagreement'].notna().sum()} / {len(saved)}")
    print(f"\nSaved → {out_path.relative_to(PROJECT_ROOT)}")
    sys.exit(0)
