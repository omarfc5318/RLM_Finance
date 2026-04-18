"""
models/base/return_predictor.py
Per-ticker XGBoost return predictor with temporal-safe train/val splitting.

Training contract
-----------------
  - Features (X) must come from feature_engineering.py — all shift(+1) safe.
  - Targets (y) must come from data/targets.py — RFIE_TRAINING_MODE required.
  - Early stopping uses the VALIDATION set only, never the test set.
  - prepare_data() calls splitter.validate_no_overlap() before returning any
    split — a DataLeakageError is raised immediately if overlap is detected.
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
from scipy import stats
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class DataLeakageError(Exception):
    """Raised when temporal split validation detects date overlap."""


# ---------------------------------------------------------------------------
# Return predictor
# ---------------------------------------------------------------------------
class ReturnPredictor:
    """
    XGBoost-based forward return predictor for a single target column.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.  XGBoost hyperparameters are read from
        config['models']['xgboost']; any key can be overridden by passing
        kwargs to __init__.
    target_col : str
        Name of the target column to predict (e.g. 'SPY_tgt_ret_1d').
    **kwargs
        Override any XGBoost hyperparameter from config.
    """

    def __init__(self, config: dict, target_col: str = "SPY_tgt_ret_1d", **kwargs) -> None:
        self.target_col = target_col

        xgb_cfg = config.get("models", {}).get("xgboost", {})
        self.params = {
            "n_estimators":       xgb_cfg.get("n_estimators", 200),
            "max_depth":          xgb_cfg.get("max_depth", 4),
            "learning_rate":      xgb_cfg.get("learning_rate", 0.05),
            "subsample":          xgb_cfg.get("subsample", 0.8),
            "colsample_bytree":   xgb_cfg.get("colsample_bytree", 0.8),
            "min_child_weight":   xgb_cfg.get("min_child_weight", 10),
            "gamma":              xgb_cfg.get("gamma", 0.1),
            "early_stopping_rounds": 50,
            "tree_method":        "hist",
            "random_state":       42,
            "verbosity":          0,
        }
        self.params.update(kwargs)

        self._model: xgb.XGBRegressor | None = None
        self._logs_dir = PROJECT_ROOT / config.get("paths", {}).get("logs", "logs")

        logger.info(
            "ReturnPredictor initialised — target='{}' n_estimators={} lr={}",
            self.target_col, self.params["n_estimators"], self.params["learning_rate"],
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def prepare_data(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        splitter,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Merge features and targets, apply temporal split, validate no overlap.

        Parameters
        ----------
        features_df : pd.DataFrame
            Wide feature DataFrame (X), date-indexed, all columns shift(+1) safe.
        targets_df : pd.DataFrame
            Wide target DataFrame (y), date-indexed, must contain self.target_col.
        splitter : TemporalSplitter
            Pre-configured splitter instance.

        Returns
        -------
        X_train, y_train, X_val, y_val — all date-indexed.

        Raises
        ------
        DataLeakageError
            If splitter.validate_no_overlap() detects date overlap between splits.
        KeyError
            If self.target_col is not present in targets_df.
        """
        if self.target_col not in targets_df.columns:
            raise KeyError(
                f"Target column '{self.target_col}' not found in targets_df. "
                f"Available: {list(targets_df.columns[:5])}…"
            )

        # Left join: keep all feature dates; target NaNs will be dropped below
        merged = features_df.join(targets_df[[self.target_col]], how="left")

        # Drop rows where either features or target are NaN
        before = len(merged)
        merged = merged.dropna(subset=[self.target_col])
        # Drop rows where any feature is NaN (leading NaN from rolling windows)
        feature_cols = [c for c in merged.columns if c != self.target_col]
        merged = merged.dropna(subset=feature_cols)
        dropped = before - len(merged)
        if dropped:
            logger.info(
                "prepare_data: dropped {} rows with NaN (features or target)",
                dropped,
            )

        # Temporal splits
        train_merged = splitter.get_train(merged)
        val_merged = splitter.get_val(merged)
        # test split is held out — only train/val are returned here
        test_merged = splitter.get_test(merged)

        # Validate no overlap — raises ValueError on failure, wrapped as DataLeakageError
        try:
            splitter.validate_no_overlap(train_merged, val_merged, test_merged)
        except ValueError as exc:
            raise DataLeakageError(
                f"Date overlap detected in prepared splits: {exc}"
            ) from exc

        X_train = train_merged.drop(columns=[self.target_col])
        y_train = train_merged[self.target_col]
        X_val = val_merged.drop(columns=[self.target_col])
        y_val = val_merged[self.target_col]

        logger.info(
            "prepare_data: train={} rows, val={} rows, test={} rows (held out)",
            len(X_train), len(X_val), len(test_merged),
        )
        logger.info(
            "prepare_data: test window {} → {} (held out, never seen by model)",
            test_merged.index.min().date(), test_merged.index.max().date(),
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
    ) -> xgb.XGBRegressor:
        """
        Fit XGBoost with early stopping monitored on the validation set.

        Early stopping uses VALIDATION data only — test data must never be
        passed here.  Training halts when val RMSE does not improve for
        early_stopping_rounds consecutive boosting rounds.

        Returns
        -------
        Fitted XGBRegressor (also stored as self._model).
        """
        model_params = {k: v for k, v in self.params.items()
                        if k != "early_stopping_rounds"}

        self._model = xgb.XGBRegressor(
            **model_params,
            early_stopping_rounds=self.params["early_stopping_rounds"],
            eval_metric="rmse",
        )

        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_iter = self._model.best_iteration
        # best_score is val RMSE at the stopping round
        best_rmse = self._model.best_score

        logger.info(
            "Training complete — best_iteration={} val_RMSE={:.6f}",
            best_iter, best_rmse,
        )
        return self._model

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate predictions for the given feature DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame with the same columns used during training.

        Returns
        -------
        pd.Series with the same DatetimeIndex as X.
        """
        if self._model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        preds = self._model.predict(X)
        return pd.Series(preds, index=X.index, name=f"pred_{self.target_col}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """
        Compute IC (Spearman rank correlation), RMSE, and hit rate.

        Parameters
        ----------
        y_true : pd.Series  — realised targets.
        y_pred : pd.Series  — model predictions.

        Returns
        -------
        dict with keys: 'ic', 'rmse', 'hit_rate'
        """
        # Align on index in case of any date mismatch
        y_true, y_pred = y_true.align(y_pred, join="inner")

        _spearman = stats.spearmanr(y_true, y_pred)
        ic = float(_spearman.statistic if hasattr(_spearman, "statistic") else _spearman.correlation)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        hit_rate = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

        metrics = {"ic": ic, "rmse": rmse, "hit_rate": hit_rate}
        logger.info(
            "Evaluation — IC={:.4f}  RMSE={:.6f}  HitRate={:.4f}",
            ic, rmse, hit_rate,
        )
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path | str) -> None:
        """Serialise the fitted model to disk with joblib."""
        if self._model is None:
            raise RuntimeError("No fitted model to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        logger.info("Model saved → {}", path.relative_to(PROJECT_ROOT))

    def load(self, path: Path | str) -> None:
        """Deserialise a saved model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No model file found at {path}")
        self._model = joblib.load(path)
        logger.info("Model loaded ← {}", path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "return_predictor.log", rotation="10 MB", retention="30 days")

    cfg = _load_config()

    # Require training mode guard for target import
    os.environ["RFIE_TRAINING_MODE"] = "1"
    from data.targets import TargetBuilder
    from data.temporal_split import TemporalSplitter

    # Load combined features
    processed_dir = PROJECT_ROOT / cfg["paths"]["processed"]
    features_path = processed_dir / "price_features_v1.parquet"
    targets_path = processed_dir / "targets.parquet"

    if not features_path.exists():
        print(f"Missing {features_path} — run feature_engineering.py first.")
        sys.exit(1)
    if not targets_path.exists():
        print(f"Missing {targets_path} — run data/targets.py first.")
        sys.exit(1)

    features_df = pd.read_parquet(features_path)
    targets_df = pd.read_parquet(targets_path)

    # Use SPY 1-day forward return as the training target
    TARGET_COL = "SPY_tgt_ret_1d"

    splitter = TemporalSplitter(cfg)
    predictor = ReturnPredictor(cfg, target_col=TARGET_COL)

    X_train, y_train, X_val, y_val = predictor.prepare_data(
        features_df, targets_df, splitter
    )

    predictor.train(X_train, y_train, X_val, y_val)

    val_preds = predictor.predict(X_val)
    metrics = predictor.evaluate(y_val, val_preds)

    # Save model
    model_path = PROJECT_ROOT / cfg["paths"].get("models", "models") / "base" / "spy_ret1d.joblib"
    predictor.save(model_path)

    # Save evaluation metrics to JSON
    eval_path = log_dir / "model_a_eval.json"
    eval_output = {
        "target_col": TARGET_COL,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "best_iteration": int(predictor._model.best_iteration),
        **metrics,
    }
    with open(eval_path, "w") as fh:
        json.dump(eval_output, fh, indent=2)

    print("\n=== Val-Set Evaluation ===")
    print(f"Target:    {TARGET_COL}")
    print(f"Train rows: {len(X_train)}  |  Val rows: {len(X_val)}")
    print(f"IC:        {metrics['ic']:.4f}")
    print(f"RMSE:      {metrics['rmse']:.6f}")
    print(f"Hit Rate:  {metrics['hit_rate']:.4f}")
    print(f"\nMetrics saved → {eval_path.relative_to(PROJECT_ROOT)}")
    sys.exit(0)
