"""
engine/feedback.py
Online feedback loop engine for the RFIE meta-learner.

Causality contract
------------------
  STEP A (prediction at date t):
    Uses only model predictions and meta-features computed from data
    observable at the close of t-1. actual_return is NOT touched in STEP A.

  STEP B (feedback processing at date t):
    actual_return is the return from t-1 → t, known after the close of t.
    It is used ONLY to update rolling buffers and the performance log.
    It never retroactively changes the prediction already recorded for t.

  validate_causality() enforces this contract post-hoc: every performance_log
  entry must have feedback_date > prediction_date.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Rolling window for meta-feature computation (matches MetaFeatureBuilder default)
_WINDOW = 21
# Minimum history rows before meta-features are computable:
#   window + max target shift (2 for 1d-return IC)
_WARMUP = _WINDOW + 2


class FeedbackLoopEngine:
    """
    Online feedback loop: at each date make a prediction, then incorporate
    the realized return as feedback without contaminating the prediction.

    Parameters
    ----------
    ensemble : BaseEnsemble
        Must expose predict_all(X_price, X_joined) → DataFrame with columns
        return_pred, vol_pred, regime_pred, drawdown_risk_prob.
        Optionally exposes compute_disagreement(predictions_df, ...) → Series.
    meta_learner : MetaLearner
        Must expose predict_weights(X: DataFrame) → DataFrame and have
        .feature_cols and .models populated (i.e., load() or train() called).
    weight_tracker : WeightTracker
        Must expose log_weights(date, weights_dict) where weights_dict keys
        are 'return', 'vol', 'regime', 'drawdown'.
    """

    _EQUAL_WEIGHTS: Dict[str, float] = {
        "return": 0.25, "vol": 0.25, "regime": 0.25, "drawdown": 0.25,
    }

    def __init__(self, ensemble, meta_learner, weight_tracker) -> None:
        self.ensemble       = ensemble
        self.meta_learner   = meta_learner
        self.weight_tracker = weight_tracker

        # Public outputs
        self.prediction_buffer: Dict[pd.Timestamp, float] = {}
        self.performance_log:   List[dict]                 = []

        # Rolling buffers — grown by step(), trimmed to 3×_WINDOW rows
        self._pred_history:   List[dict]                       = []
        self._return_history: List[Tuple[pd.Timestamp, float]] = []
        self._weights_buffer: Dict[pd.Timestamp, Dict[str, float]] = {}

        # Cached latest meta-feature row; None during warmup
        self._latest_meta_features: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(
        self,
        date: pd.Timestamp,
        features_row: Dict[str, pd.DataFrame],
        actual_return: Optional[float] = None,
    ) -> float:
        """
        Process a single trading date.

        Parameters
        ----------
        date : pd.Timestamp
        features_row : dict
            {'price':  single-row DataFrame of price features (99 cols),
             'joined': single-row DataFrame of joined features (111 cols)}
        actual_return : float, optional
            Log return from t-1 → t, known at close of t.
            Used ONLY in STEP B — never in STEP A.

        Returns
        -------
        float  — today's weighted directional signal.
        """
        # ── STEP A: predict for today ─────────────────────────────────────
        preds_df = self.ensemble.predict_all(
            features_row["price"], features_row["joined"]
        )
        preds = preds_df.iloc[-1]

        ret_pred      = float(preds["return_pred"])
        regime_val    = float(preds["regime_pred"])
        vol_pred_val  = float(preds["vol_pred"]) if pd.notna(preds.get("vol_pred")) else np.nan
        drawdown_prob = float(preds["drawdown_risk_prob"])

        # Stash raw prediction in history BEFORE computing weights (causal)
        self._pred_history.append({
            "date":               date,
            "return_pred":        ret_pred,
            "vol_pred":           vol_pred_val,
            "regime_pred":        regime_val,
            "drawdown_risk_prob": drawdown_prob,
        })

        weights = self._get_weights()

        # Unified directional signal — single source of truth shared with
        # MetaLearner.build_proxy_target / evaluate_lift / evaluate_meta_lift.
        # ensemble_predictions are normalized (z-score space), so each base
        # signal is mapped to a directional contribution via sign convention:
        #   d_return   = +return_pred           (raw direction)
        #   d_regime   = -regime_pred           (low z = bull → positive)
        #   d_vol      = -vol_pred              (high vol = risk-off → negative)
        #   d_drawdown = -drawdown_risk_prob    (high prob = risk-off → negative)
        # NaN risk signals during warm-up contribute 0 so the weighted sum
        # stays finite even before all 4 base predictions are populated.
        d_return   = ret_pred
        d_regime   = (-regime_val)    if not np.isnan(regime_val)    else 0.0
        d_vol      = (-vol_pred_val)  if not np.isnan(vol_pred_val)  else 0.0
        d_drawdown = (-drawdown_prob) if not np.isnan(drawdown_prob) else 0.0

        weighted_signal = (
            weights["return"]   * d_return
            + weights["vol"]      * d_vol
            + weights["regime"]   * d_regime
            + weights["drawdown"] * d_drawdown
        )

        self.prediction_buffer[date] = weighted_signal
        self._weights_buffer[date]   = weights
        self.weight_tracker.log_weights(date, weights)

        logger.debug(
            "step [{}]: signal={:.6f}  weights=ret:{:.3f} vol:{:.3f} "
            "reg:{:.3f} dd:{:.3f}",
            date.date(), weighted_signal,
            weights["return"], weights["vol"], weights["regime"], weights["drawdown"],
        )

        # ── STEP B: incorporate yesterday's feedback ──────────────────────
        # actual_return is NOT used above — only used here
        if actual_return is not None:
            self._update_rolling_performance(date, actual_return)
            self._recompute_meta_features()

        return weighted_signal

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(
        self,
        features_df,          # pd.DataFrame or dict {'price': df, 'joined': df}
        returns_df: pd.Series,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Backtest loop over [start_date, end_date].

        At each date t:
          - Pass features_df.loc[t] as today's features.
          - Pass returns_df.iloc[t_pos - 1] as feedback (YESTERDAY's return).
            Today's return is never used as a predictor — only as feedback
            after the prediction is already recorded.

        Parameters
        ----------
        features_df : pd.DataFrame or dict {'price': df, 'joined': df}
            If a plain DataFrame, it is used as both X_price and X_joined.
        returns_df : pd.Series
            Date-indexed SPY log returns (any range; sliced internally).
        start_date, end_date : pd.Timestamp
            Inclusive backtest window.

        Clears the weight-tracker audit at the start of each run to prevent
        accumulation across simulations.

        Returns
        -------
        predictions_s : pd.Series  — date-indexed weighted signals
        perf_df       : pd.DataFrame — one row per feedback day
        """
        self.weight_tracker.clear()

        if isinstance(features_df, dict):
            price_df  = features_df["price"].loc[start_date:end_date]
            joined_df = features_df["joined"].loc[start_date:end_date]
        else:
            price_df  = features_df.loc[start_date:end_date]
            joined_df = features_df.loc[start_date:end_date]

        sorted_ret = returns_df.sort_index()
        dates      = sorted_ret.loc[start_date:end_date].index

        for t in dates:
            if t not in price_df.index or t not in joined_df.index:
                logger.warning("run: missing features for {} — skipping", t.date())
                continue

            feat = {
                "price":  price_df.loc[[t]],
                "joined": joined_df.loc[[t]],
            }

            # Yesterday's return — causal feedback; None on the first date
            t_pos         = sorted_ret.index.get_loc(t)
            yesterday_ret = float(sorted_ret.iloc[t_pos - 1]) if t_pos > 0 else None

            self.step(t, feat, actual_return=yesterday_ret)

        predictions_s = pd.Series(
            self.prediction_buffer, name="weighted_signal"
        ).sort_index()

        perf_df = (
            pd.DataFrame(self.performance_log)
            if self.performance_log
            else pd.DataFrame(columns=[
                "prediction_date", "feedback_date",
                "signal", "actual_return", "pnl",
            ])
        )

        logger.info(
            "run complete: {} prediction days, {} feedback entries",
            len(predictions_s), len(perf_df),
        )
        return predictions_s, perf_df

    # ------------------------------------------------------------------
    # validate_causality
    # ------------------------------------------------------------------
    def validate_causality(self) -> bool:
        """
        Replay performance_log and verify feedback_date > prediction_date
        for every entry. Returns False (and logs each violation) if any
        entry has feedback_date <= prediction_date.
        """
        violations = 0
        for entry in self.performance_log:
            pd_date = entry.get("prediction_date")
            fb_date = entry.get("feedback_date")
            if pd_date is None or fb_date is None:
                continue
            if fb_date <= pd_date:
                logger.error(
                    "causality VIOLATION: prediction_date={} >= feedback_date={}",
                    pd_date, fb_date,
                )
                violations += 1

        if violations:
            logger.error("validate_causality: {} violation(s) found", violations)
            return False

        logger.info(
            "validate_causality: {} entries checked — PASS",
            len(self.performance_log),
        )
        return True

    # ------------------------------------------------------------------
    # get_performance_summary
    # ------------------------------------------------------------------
    def get_performance_summary(self) -> dict:
        """
        Compute Sharpe ratio, maximum drawdown, and weight trajectory from
        the accumulated performance log.

        Returns
        -------
        dict:
          sharpe            float   annualized (√252 scaling)
          max_drawdown      float   most negative cumulative PnL trough
          n_days            int     feedback days with valid PnL
          weight_trajectory pd.DataFrame  one row per feedback day
        """
        if not self.performance_log:
            return {
                "sharpe": float("nan"), "max_drawdown": float("nan"),
                "n_days": 0, "weight_trajectory": pd.DataFrame(),
            }

        df = pd.DataFrame(self.performance_log).dropna(subset=["pnl"])
        if df.empty:
            return {
                "sharpe": float("nan"), "max_drawdown": float("nan"),
                "n_days": 0, "weight_trajectory": pd.DataFrame(),
            }

        pnl = df["pnl"].values
        sharpe = (
            float(pnl.mean() / pnl.std() * np.sqrt(252))
            if pnl.std() > 0
            else float("nan")
        )

        cum          = np.cumsum(pnl)
        running_max  = np.maximum.accumulate(cum)
        max_drawdown = float((cum - running_max).min())

        weight_traj = (
            pd.DataFrame(
                df["weights"].tolist(),
                index=pd.Index(df["feedback_date"].values, name="date"),
            )
            if "weights" in df.columns
            else pd.DataFrame()
        )

        return {
            "sharpe":            sharpe,
            "max_drawdown":      max_drawdown,
            "n_days":            int(len(df)),
            "weight_trajectory": weight_traj,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _get_weights(self) -> Dict[str, float]:
        """
        Return meta-learner weights if warmed up, else equal weights.
        Falls back to equal weights on any prediction error.
        """
        if (
            self._latest_meta_features is None
            or self.meta_learner.models is None
            or self.meta_learner.feature_cols is None
        ):
            return dict(self._EQUAL_WEIGHTS)

        try:
            w_df = self.meta_learner.predict_weights(self._latest_meta_features)
            return {k: float(v) for k, v in w_df.iloc[0].items()}
        except Exception as exc:
            logger.warning("_get_weights: predict_weights failed ({}) — equal", exc)
            return dict(self._EQUAL_WEIGHTS)

    def _update_rolling_performance(
        self, date: pd.Timestamp, actual_return: float
    ) -> None:
        """
        Append today's return to the buffer and record a PnL entry using
        yesterday's signal (prediction_date = t-1).
        """
        self._return_history.append((date, actual_return))

        all_dates = list(self.prediction_buffer.keys())
        pred_date = all_dates[-2] if len(all_dates) >= 2 else None
        signal    = self.prediction_buffer.get(pred_date)
        pnl       = signal * actual_return if signal is not None else None

        self.performance_log.append({
            "prediction_date": pred_date,
            "feedback_date":   date,
            "signal":          signal,
            "actual_return":   actual_return,
            "pnl":             pnl,
            "weights":         self._weights_buffer.get(pred_date, dict(self._EQUAL_WEIGHTS)),
        })

    def _recompute_meta_features(self) -> None:
        """
        Rebuild self._latest_meta_features from rolling buffers using
        MetaFeatureBuilder's individual methods (not build_all, which
        writes to disk and would overwrite the training artefact).

        Only runs once both buffers have >= _WARMUP rows.
        Pads any column that cannot be computed with 0.0 (logged as warning)
        so the meta-learner always receives its full feature vector.
        Trims both buffers to 3×_WINDOW rows to bound memory use.
        """
        required_cols = self.meta_learner.feature_cols
        if required_cols is None:
            return
        if len(self._pred_history) < _WARMUP or len(self._return_history) < _WARMUP:
            return

        pred_df = pd.DataFrame(self._pred_history).set_index("date")
        pred_df.index = pd.DatetimeIndex(pred_df.index)

        ret_series = pd.Series(
            {d: r for d, r in self._return_history},
            name="SPY_logret",
        ).sort_index()
        ret_series.index = pd.DatetimeIndex(ret_series.index)

        # Proxy targets constructed from realized returns (causal):
        #   SPY_tgt_ret_1d[t] = ret[t+1]  (shift(-1) here; MetaFeatureBuilder
        #     applies shift(+2) internally, so IC at window position i uses
        #     tgt[i-2] = ret[i-1] — fully past at time i)
        #   SPY_tgt_vol_5d[t] = 5d realized vol from t+1 (same causal argument)
        tgt_df = pd.DataFrame(index=ret_series.index)
        tgt_df["SPY_tgt_ret_1d"] = ret_series.shift(-1)
        tgt_df["SPY_tgt_vol_5d"] = ret_series.shift(-1).rolling(5).std()

        try:
            from models.meta.meta_features import MetaFeatureBuilder
            mfb   = MetaFeatureBuilder(window=_WINDOW)
            parts = []

            if any(c.startswith("rolling_ic") for c in required_cols):
                parts.append(mfb.rolling_ic(pred_df, tgt_df))

            if any(c.startswith("rolling_pnl") for c in required_cols):
                parts.append(mfb.rolling_pnl(pred_df, ret_series))

            if "rolling_disagreement" in required_cols:
                try:
                    dis = self.ensemble.compute_disagreement(
                        pred_df,
                        fit_start=ret_series.index.min(),
                        fit_end=ret_series.index.max(),
                    )
                    pred_df_dis = pred_df.copy()
                    pred_df_dis["disagreement"] = dis
                    parts.append(mfb.model_disagreement(pred_df_dis))
                except Exception as dis_exc:
                    logger.warning(
                        "_recompute_meta_features: disagreement failed ({}) — padding 0",
                        dis_exc,
                    )
                    parts.append(
                        pd.DataFrame(
                            {"rolling_disagreement": 0.0}, index=pred_df.index
                        )
                    )

            if not parts:
                return

            meta = pd.concat(parts, axis=1, sort=False).sort_index()

        except Exception as exc:
            logger.warning("_recompute_meta_features: build failed — {}", exc)
            return

        if "rolling_regime_accuracy" in meta.columns:
            meta = meta.drop(columns=["rolling_regime_accuracy"])

        # Pad any feature the builder couldn't produce
        missing = set(required_cols) - set(meta.columns)
        if missing:
            logger.warning(
                "_recompute_meta_features: padding cols {} with 0.0", missing
            )
            for c in missing:
                meta[c] = 0.0

        latest = meta[required_cols].dropna(how="any")
        if latest.empty:
            return

        self._latest_meta_features = latest.iloc[[-1]]

        # Trim buffers to bound memory
        keep = _WINDOW * 3
        if len(self._pred_history)   > keep:
            self._pred_history   = self._pred_history[-keep:]
        if len(self._return_history) > keep:
            self._return_history = self._return_history[-keep:]
