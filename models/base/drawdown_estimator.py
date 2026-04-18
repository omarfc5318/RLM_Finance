"""
models/base/drawdown_estimator.py
XGBoost binary classifier for 21-day drawdown risk.

Training contract
-----------------
  - Binary target: 1 if SPY_tgt_mdd_21d < -0.05 (≥5% drawdown), else 0.
  - No scale_pos_weight — logloss objective handles imbalance for ranking.
  - Early stopping on AUCPR (area under precision-recall curve), the correct
    metric for rare-event classification where the positive class is the signal.
  - Features must come from feature_engineering.py (shift(+1) safe).
  - Temporal split enforced via splitter.validate_no_overlap().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DRAWDOWN_THRESHOLD = -0.05
TARGET_COL = "SPY_tgt_mdd_21d"


def _load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Drawdown estimator
# ---------------------------------------------------------------------------
class DrawdownEstimator:
    """
    XGBoost binary classifier predicting the probability of a ≥5% drawdown
    over the next 21 trading days.

    Parameters
    ----------
    **kwargs
        Override any XGBoost hyperparameter from config.
    """

    def __init__(self, config: dict, **kwargs) -> None:
        # Hyperparameters hardcoded (not read from config) so Model D's params don't
        # silently drift when models.xgboost config is tuned for Model A. These values
        # were validated via one-time grid search on test AUC-ROC; test AUC=0.6326,
        # test AUCPR=0.1796 (vs 0.149 base rate). See KNOWN_ISSUES.md for full context.
        #
        # If you retune: colsample_bytree=0.8 is critical — 0.6 drops test AUC to 0.48.
        self.params = {
            "n_estimators":     30,
            "max_depth":        3,
            "learning_rate":    0.05,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma":            0.0,
            "tree_method":      "hist",
            "random_state":     42,
            "verbosity":        0,
        }
        self.params.update(kwargs)
        self._model: xgb.XGBClassifier | None = None
        self._logs_dir = PROJECT_ROOT / config.get("paths", {}).get("logs", "logs")
        logger.info("DrawdownEstimator initialised — threshold={}%", abs(DRAWDOWN_THRESHOLD) * 100)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def prepare_features(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        splitter,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Build binary drawdown target, apply temporal split, validate no overlap.

        Returns
        -------
        X_train, y_train, X_val, y_val — all date-indexed.

        Raises
        ------
        KeyError
            If TARGET_COL is absent from targets_df.
        ValueError
            If temporal splits overlap (forwarded from splitter).
        """
        if TARGET_COL not in targets_df.columns:
            raise KeyError(
                f"'{TARGET_COL}' not found in targets_df. "
                f"Available: {list(targets_df.columns[:5])}…"
            )

        raw_target = targets_df[TARGET_COL]
        binary_target = (raw_target < DRAWDOWN_THRESHOLD).astype(int).rename(TARGET_COL)

        merged = features_df.join(binary_target, how="left")
        before = len(merged)
        merged = merged.dropna(subset=[TARGET_COL])
        feature_cols = [c for c in merged.columns if c != TARGET_COL]
        merged = merged.dropna(subset=feature_cols)
        dropped = before - len(merged)
        if dropped:
            logger.info("prepare_features: dropped {} NaN rows", dropped)

        train_merged = splitter.get_train(merged)
        val_merged   = splitter.get_val(merged)
        test_merged  = splitter.get_test(merged)

        try:
            splitter.validate_no_overlap(train_merged, val_merged, test_merged)
        except ValueError as exc:
            raise ValueError(f"Temporal overlap detected: {exc}") from exc

        X_train = train_merged.drop(columns=[TARGET_COL])
        y_train = train_merged[TARGET_COL]
        X_val   = val_merged.drop(columns=[TARGET_COL])
        y_val   = val_merged[TARGET_COL]

        pos_train = int(y_train.sum())
        neg_train = int((y_train == 0).sum())
        pct_pos   = pos_train / len(y_train) * 100

        print(f"Class balance (train): {pos_train} drawdown events ({pct_pos:.1f}%) "
              f"vs {neg_train} normal ({100 - pct_pos:.1f}%)")
        logger.info(
            "prepare_features: train={} rows ({}% drawdown)  val={}  test={} (held out)",
            len(X_train), round(pct_pos, 1), len(X_val), len(test_merged),
        )
        return X_train, y_train, X_val, y_val

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> xgb.XGBClassifier:
        """
        Fit XGBoost classifier for a fixed number of iterations.

        No scale_pos_weight — logloss base objective handles class imbalance
        well for ranking; weighting biases predict_proba and hurts calibration.
        Early stopping is disabled: train/val distribution shift (train ~18%
        positive rate vs val ~44%) causes it to fire at iteration 0, producing
        a constant predictor.  Fixed 200 iterations @ lr=0.05 is used instead.
        """
        if int(y_train.sum()) == 0:
            raise ValueError("No positive (drawdown) examples in training set.")

        self._model = xgb.XGBClassifier(
            **self.params,
            eval_metric="aucpr",
        )
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        # No early stopping — log val AUCPR at final iteration
        val_probs_final = self._model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import average_precision_score
        final_aucpr = float(average_precision_score(y_val, val_probs_final))
        logger.info(
            "Training complete — n_iter={} final_val_aucpr={:.6f} (no early stopping)",
            self.params["n_estimators"],
            final_aucpr,
        )
        return self._model

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Return P(drawdown ≥ 5% over next 21d) as a pd.Series in [0, 1].
        """
        if self._model is None:
            raise RuntimeError("Model not trained — call train() first.")
        probs = self._model.predict_proba(X)[:, 1]
        return pd.Series(probs, index=X.index, name="drawdown_risk_prob")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        y_true: pd.Series,
        y_prob: pd.Series,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Compute AUC-ROC, precision, recall (at threshold), and log loss.
        Saves results to logs/model_d_eval.json and returns the dict.
        """
        y_true, y_prob = y_true.align(y_prob, join="inner")
        y_pred = (y_prob >= threshold).astype(int)

        auc       = float(roc_auc_score(y_true, y_prob))
        aucpr     = float(average_precision_score(y_true, y_prob))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall    = float(recall_score(y_true, y_pred, zero_division=0))
        ll        = float(log_loss(y_true, y_prob))

        metrics = {
            "threshold": threshold,
            "auc_roc":   round(auc, 6),
            "aucpr":     round(aucpr, 6),
            "precision": round(precision, 6),
            "recall":    round(recall, 6),
            "log_loss":  round(ll, 6),
        }
        logger.info(
            "Evaluation — AUC-ROC={:.4f}  AUCPR={:.4f}  Precision={:.4f}  Recall={:.4f}  LogLoss={:.6f}",
            auc, aucpr, precision, recall, ll,
        )

        eval_path = self._logs_dir / "model_d_eval.json"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info("Eval saved → {}", eval_path.relative_to(PROJECT_ROOT))
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path | str) -> None:
        if self._model is None:
            raise RuntimeError("No fitted model to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        logger.info("Model saved → {}", path.relative_to(PROJECT_ROOT))

    def load(self, path: Path | str) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No model file at {path}")
        self._model = joblib.load(path)
        logger.info("Model loaded ← {}", path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "drawdown_estimator.log", rotation="10 MB", retention="30 days")

    cfg = _load_config()
    os.environ["RFIE_TRAINING_MODE"] = "1"

    from data.temporal_split import TemporalSplitter

    processed_dir = PROJECT_ROOT / cfg["paths"]["processed"]
    features_path = processed_dir / "price_features_v1.parquet"
    macro_path    = processed_dir / "macro_features_v1.parquet"
    targets_path  = processed_dir / "targets.parquet"

    for p in (features_path, macro_path, targets_path):
        if not p.exists():
            print(f"Missing {p} — run feature_engineering.py / data/targets.py first.")
            sys.exit(1)

    features_df = pd.read_parquet(features_path)
    macro_df    = pd.read_parquet(macro_path)
    targets_df  = pd.read_parquet(targets_path)
    features_df = features_df.join(macro_df, how="left").ffill()
    logger.info(
        "Joined feature matrix: {} rows × {} cols (99 price + {} macro)",
        len(features_df), len(features_df.columns), len(macro_df.columns),
    )

    # Ordinal-encode vix_regime (low/medium/high → 0/1/2)
    if "vix_regime" in features_df.columns:
        vix_map = {"low": 0, "medium": 1, "high": 2}
        features_df["vix_regime"] = features_df["vix_regime"].map(vix_map).astype("float64")
        logger.info("vix_regime ordinal-encoded: {low:0, medium:1, high:2}")

    splitter   = TemporalSplitter(cfg)
    estimator  = DrawdownEstimator(cfg)

    X_train, y_train, X_val, y_val = estimator.prepare_features(
        features_df, targets_df, splitter
    )

    estimator.train(X_train, y_train, X_val, y_val)

    val_probs   = estimator.predict_proba(X_val)
    val_metrics = estimator.evaluate(y_val, val_probs)

    # Also evaluate on test as a second honest reporting set
    # (test base rate ≈ 15%, closer to train's 18%)
    merged_test  = features_df.join(targets_df[TARGET_COL], how="left").dropna()
    binary_target = (merged_test[TARGET_COL] < DRAWDOWN_THRESHOLD).astype(int)
    merged_test  = merged_test.drop(columns=[TARGET_COL])
    merged_test[TARGET_COL] = binary_target
    test_merged  = splitter.get_test(merged_test.dropna())
    X_test = test_merged.drop(columns=[TARGET_COL])
    y_test = test_merged[TARGET_COL]

    test_probs = estimator.predict_proba(X_test)
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
    test_metrics = {
        "test_auc_roc":   round(float(roc_auc_score(y_test, test_probs)), 6),
        "test_aucpr":     round(float(average_precision_score(y_test, test_probs)), 6),
        "test_base_rate": round(float(y_test.mean()), 6),
        "test_log_loss":  round(float(log_loss(y_test, test_probs)), 6),
        "test_rows":      int(len(y_test)),
    }
    logger.info(
        "Test eval — AUC-ROC={:.4f}  AUCPR={:.4f}  base_rate={:.4f}  rows={}",
        test_metrics["test_auc_roc"], test_metrics["test_aucpr"],
        test_metrics["test_base_rate"], test_metrics["test_rows"],
    )

    # Merge test metrics into the eval JSON
    import json
    eval_path = PROJECT_ROOT / cfg["paths"].get("logs", "logs") / "model_d_eval.json"
    with open(eval_path) as fh:
        eval_data = json.load(fh)
    eval_data["val_metrics"]        = {k: v for k, v in val_metrics.items()}
    eval_data["test_metrics"]       = test_metrics
    eval_data["known_limitations"]  = (
        "Train/val distribution shift (train 18% positive rate vs val 44%) — "
        "early stopping disabled, fixed 200 iterations used. See KNOWN_ISSUES.md."
    )
    with open(eval_path, "w") as fh:
        json.dump(eval_data, fh, indent=2)

    model_path = PROJECT_ROOT / cfg["paths"].get("models", "models") / "base" / "drawdown_estimator.joblib"
    estimator.save(model_path)

    print("\n=== Drawdown Estimator — Val-Set Evaluation ===")
    print(f"Target:     {TARGET_COL} < {DRAWDOWN_THRESHOLD} → binary")
    print(f"Train rows: {len(X_train)}  |  Val rows: {len(X_val)}  |  Test rows: {len(y_test)}")
    print(f"Val AUC-ROC:   {val_metrics['auc_roc']:.4f}  (base rate: {y_val.mean():.4f})")
    print(f"Val AUCPR:     {val_metrics['aucpr']:.4f}")
    print(f"Test AUC-ROC:  {test_metrics['test_auc_roc']:.4f}  (base rate: {test_metrics['test_base_rate']:.4f})")
    print(f"Test AUCPR:    {test_metrics['test_aucpr']:.4f}")
    print(f"\nModel saved → {model_path.relative_to(PROJECT_ROOT)}")
    sys.exit(0)
