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

Disagreement metric
-------------------
  Each column is standardised (zero mean, unit variance) using the VAL-period
  statistics only (no lookahead into test). The val period is used instead of
  train because vol_forecasts.parquet does not cover the train period — GARCH
  was evaluated on val+test only in Step 2.3. Regime is remapped to a risk-ordered integer
  (bull=0, sideways=1, bear=2) for the disagreement computation only; the
  regime_pred output column keeps its original HMM labeling.

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

        logger.info(
            "BaseEnsemble loaded — return model: {}  drawdown model: {}  "
            "vol rows: {}  regime rows: {}",
            RETURN_MODEL_PATH.name,
            DRAWDOWN_MODEL_PATH.name,
            len(self._vol_df),
            len(self._regime_df),
        )

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
        if "regime_pred" in df.columns:
            regime_risk_order = {0: 0, 2: 1, 1: 2}
            df["regime_pred"] = df["regime_pred"].map(regime_risk_order)

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

    predictions = ensemble.predict_all(X_price, X_joined)

    # The disagreement scaler fits on the VAL period only — vol_forecasts
    # doesn't cover the train period (GARCH was only evaluated on val+test
    # in Step 2.3). Fitting on val avoids lookahead into test.
    from data.temporal_split import TemporalSplitter
    splitter  = TemporalSplitter(cfg)
    fit_start = splitter.train_next        # first business day after train_end
    fit_end   = splitter.val_end if hasattr(splitter, "val_end") else splitter.test_start - pd.Timedelta(days=1)
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
