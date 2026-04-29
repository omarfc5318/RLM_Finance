"""
models/meta/meta_learner.py
Phase 3.2 meta-learner: dynamically weights the 4 base models using rolling
meta-features from meta_features.parquet.

Audit fixes applied vs. initial scaffold:
  FIX 1 (temporal CV): RidgeCV uses TimeSeriesSplit(n_splits=5, gap=21) to
         embargo the 21-day autocorrelated label window from leaking into CV.
  FIX 2 (convex floor): Weight floor applied as a convex combination
         w_i = (1 - n*floor) * softmax(z)_i + floor, preserving softmax ranking
         and guaranteeing the floor exactly without clip-then-renormalize drift.
  FIX 3 (train/eval split): Val set split 70%/30% chronologically so lift
         is evaluated on rows the meta-learner never trained on (Layer 3 fix).

Known limitations: see KNOWN_ISSUES.md §Step 3.2.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy.special import softmax
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH  = PROJECT_ROOT / "config.yaml"

META_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "meta_features.parquet"
ENSEMBLE_PATH      = PROJECT_ROOT / "data" / "processed" / "ensemble_predictions.parquet"
TARGETS_PATH       = PROJECT_ROOT / "data" / "processed" / "targets.parquet"


def _load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Meta-learner
# ---------------------------------------------------------------------------
class MetaLearner:
    """
    Learns to weight the 4 base models using their rolling predictive skill.

    Parameters
    ----------
    alphas : tuple
        Ridge regularization candidates for RidgeCV.
    cv_splits : int
        Number of TimeSeriesSplit folds (FIX 1).
    cv_gap : int
        CV gap — rows embargoed between train and val fold (FIX 1).
        Set to future_window to avoid label autocorrelation leakage.
    weight_floor : float
        Minimum allocation to any base model (FIX 2 convex combination).
    n_models : int
        Number of base models.
    future_window : int
        Forward horizon (days) for proxy target construction.
    """

    def __init__(
        self,
        alphas: tuple = (0.1, 1.0, 10.0, 100.0),
        cv_splits: int = 5,
        cv_gap: int = 21,
        weight_floor: float = 0.05,
        n_models: int = 4,
        future_window: int = 21,
    ) -> None:
        self.alphas        = alphas
        self.cv_splits     = cv_splits
        self.cv_gap        = cv_gap
        self.weight_floor  = weight_floor
        self.n_models      = n_models
        self.future_window = future_window
        self.models: Optional[List[RidgeCV]]        = None
        self.model_names                             = ["return", "vol", "regime", "drawdown"]
        self.feature_cols: Optional[List[str]]      = None
        # H3 fix: scaler + saturation tracking (populated by train())
        self._feature_scaler: Optional[StandardScaler] = None
        self.saturated: dict                         = {}  # {model_name: bool}

    # ------------------------------------------------------------------
    # Proxy target construction
    # ------------------------------------------------------------------
    def build_proxy_target(
        self,
        meta_features_df: pd.DataFrame,
        ensemble_preds_df: pd.DataFrame,
        targets_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Unified directional-PnL proxy target (Option 3 redesign).

        Each base model carries an implicit directional signal d_i (in z-score
        space, since ensemble_predictions are already normalized):
          d_return[k]   = +return_pred[k]            (raw direction)
          d_regime[k]   = -regime_pred[k]            (low z = bull → positive)
          d_vol[k]      = -vol_pred[k]               (high vol = risk-off → negative)
          d_drawdown[k] = -drawdown_risk_prob[k]     (high prob = risk-off → negative)

        For each date t in meta_features_df.index, the proxy score for model i is
        the mean realized directional PnL of d_i over the forward window:

          proxy_i[t] = mean_{k in [t+1, t+W]} d_i[k] * actual_return[k]

        where actual_return[k] = SPY_tgt_ret_1d[k] (forward-stamped log return,
        i.e. the return earned from holding d_i[k] over k → k+1).

        After scoring: drop rows with any NaN, then z-score each column. All 4
        models compete on the same metric → softmax weights pick the model with
        the strongest realized directional contribution given the meta-features.

        NOTE: the forward window is the TRAINING TARGET — looking forward in y
        is standard supervised ML. The anti-lookahead contract applies to X.
        """
        ep  = ensemble_preds_df
        tgt = targets_df

        if "SPY_tgt_ret_1d" not in tgt.columns:
            logger.warning("build_proxy_target: SPY_tgt_ret_1d not in targets")
            return pd.DataFrame(columns=self.model_names)

        ret_tgt = tgt["SPY_tgt_ret_1d"].reindex(ep.index)

        # Per-model directional signals (same convention as evaluate_lift /
        # engine.feedback.step — single source of truth)
        signals = {
            "return":   ep["return_pred"]        if "return_pred"        in ep else None,
            "vol":      -ep["vol_pred"]          if "vol_pred"           in ep else None,
            "regime":   -ep["regime_pred"]       if "regime_pred"        in ep else None,
            "drawdown": -ep["drawdown_risk_prob"] if "drawdown_risk_prob" in ep else None,
        }

        ep_pos   = {d: i for i, d in enumerate(ep.index)}
        min_obs  = max(5, self.future_window // 4)
        rows: List[List[float]] = []

        for t in meta_features_df.index:
            if t not in ep_pos:
                rows.append([np.nan] * 4)
                continue
            pos = ep_pos[t]
            if pos + self.future_window >= len(ep):
                rows.append([np.nan] * 4)
                continue

            w  = slice(pos + 1, pos + self.future_window + 1)
            rt = ret_tgt.iloc[w]

            row_scores: List[float] = []
            for name in self.model_names:
                sig = signals.get(name)
                if sig is None:
                    row_scores.append(np.nan)
                    continue
                s_w  = sig.iloc[w]
                mask = s_w.notna() & rt.notna()
                if mask.sum() < min_obs:
                    row_scores.append(np.nan)
                    continue
                row_scores.append(float((s_w[mask].values * rt[mask].values).mean()))

            rows.append(row_scores)

        out = pd.DataFrame(rows, index=meta_features_df.index, columns=self.model_names)
        out = out.dropna()

        # Z-score each column so softmax operates on commensurable scales
        for col in out.columns:
            if out[col].std() > 0:
                out[col] = (out[col] - out[col].mean()) / out[col].std()

        logger.info(
            "build_proxy_target: {} complete rows (unified directional PnL, z-scored)",
            len(out),
        )
        return out

    # ------------------------------------------------------------------
    # Data preparation  (FIX 3: 70/30 val split)
    # ------------------------------------------------------------------
    def prepare_data(
        self,
        meta_features_df: pd.DataFrame,
        ensemble_preds_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        splitter,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        FIX 3: Split val into val_meta_train (first 70%) and val_meta_eval
        (last 30%) so lift is not evaluated on rows used for training.

        Steps:
          1. Slice all three inputs to val via splitter.get_val().
          2. Build proxy target y on val-sliced data.
          3. Inner-join X and y on shared index; drop NaN from X.
          4. Chronological 70/30 split.
          5. Store self.feature_cols.

        Returns (X_train, y_train, X_eval, y_eval).
        """
        # Step 1: slice to val
        X_val_raw = splitter.get_val(meta_features_df)
        ep_val    = splitter.get_val(ensemble_preds_df)
        tgt_val   = splitter.get_val(targets_df)

        # Drop rolling_regime_accuracy: 269/375 NaN in val set due to sideways
        # sparsity in the underlying regime classifier output. The new regime
        # directional-IC proxy target (EDIT 1) already captures regime skill.
        dropped_cols = []
        if "rolling_regime_accuracy" in X_val_raw.columns:
            X_val_raw = X_val_raw.drop(columns=["rolling_regime_accuracy"])
            dropped_cols.append("rolling_regime_accuracy")

        n_val_raw = len(X_val_raw)
        logger.info(
            "prepare_data: val_raw={} rows, dropped_cols={}",
            n_val_raw, dropped_cols,
        )

        # Step 2: proxy target on val slice
        y_val_raw = self.build_proxy_target(X_val_raw, ep_val, tgt_val)

        # Step 3: inner join (y may have fewer rows due to dropna in build_proxy_target)
        combined = X_val_raw.join(y_val_raw, how="inner").dropna()

        X_cols          = X_val_raw.columns.tolist()
        X_clean         = combined[X_cols]
        y_clean         = combined[self.model_names]
        self.feature_cols = X_cols

        n_clean = len(combined)

        # Step 4: chronological 70/30 split
        n_train = int(n_clean * 0.70)
        X_train = X_clean.iloc[:n_train]
        y_train = y_clean.iloc[:n_train]
        X_eval  = X_clean.iloc[n_train:]
        y_eval  = y_clean.iloc[n_train:]

        logger.info(
            "prepare_data: val_raw={}, after NaN drop={}, "
            "train={} ({} → {}), eval={} ({} → {})",
            n_val_raw, n_clean,
            len(X_train), X_train.index.min().date(), X_train.index.max().date(),
            len(X_eval),  X_eval.index.min().date(),  X_eval.index.max().date(),
        )
        return X_train, y_train, X_eval, y_eval

    # ------------------------------------------------------------------
    # Training  (FIX 1: TimeSeriesSplit with gap)
    # ------------------------------------------------------------------
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        FIX 1: RidgeCV uses TimeSeriesSplit(n_splits, gap=cv_gap) to embargo
        the future_window autocorrelated label window from leaking into CV folds.

        H3 fix: meta-features are StandardScaler-normalized before Ridge fit so
        that scale differences across rolling_ic / rolling_pnl columns don't
        inflate regularization pressure on small-magnitude columns.

        After fitting, records self.saturated = {name: bool} where True means
        the CV chose alpha = max(self.alphas) — indicating Ridge wanted more
        regularization than the grid allows. predict_weights() zeros out the
        raw scores for saturated models before softmax so they don't emit a
        near-constant ~0.25 that dilutes the signal from the fitted models.
        """
        # H3 fix: fit StandardScaler on X (train rows only — causal)
        self._feature_scaler = StandardScaler()
        X_scaled = self._feature_scaler.fit_transform(X.values)

        tscv = TimeSeriesSplit(n_splits=self.cv_splits, gap=self.cv_gap)
        self.models   = []
        self.saturated = {}
        alpha_max = max(self.alphas)
        for col in self.model_names:
            m = RidgeCV(alphas=self.alphas, cv=tscv)
            m.fit(X_scaled, y[col].values)
            self.models.append(m)
            at_max = bool(m.alpha_ >= alpha_max)
            self.saturated[col] = at_max
            logger.info(
                "train: model={}  alpha_chosen={}  {}",
                col, m.alpha_, "SATURATED" if at_max else "interior",
            )

        n_sat = sum(self.saturated.values())
        if n_sat > 0:
            logger.warning(
                "train: {}/{} models saturated at alpha={} — "
                "consider widening the alpha grid further",
                n_sat, len(self.model_names), alpha_max,
            )

    # ------------------------------------------------------------------
    # Weight floor  (FIX 2: convex combination)
    # ------------------------------------------------------------------
    def _apply_weight_floor(self, softmax_weights: np.ndarray) -> np.ndarray:
        """
        FIX 2: Convex-combination floor.
          w_i = (1 - n_models * floor) * softmax_w_i + floor

        Properties:
          - Row sums remain exactly 1.0 (sum_i w_i = scale*1 + n*floor = 1).
          - Every element >= floor exactly.
          - Softmax ranking preserved (no clip + renormalize distortion).

        Raises AssertionError if row sums deviate from 1.0 by > 1e-10.
        """
        scale   = 1.0 - self.n_models * self.weight_floor
        floored = scale * softmax_weights + self.weight_floor
        row_sums = floored.sum(axis=1)
        assert np.all(np.abs(row_sums - 1.0) < 1e-10), (
            f"_apply_weight_floor: row sums not 1.0 — "
            f"min={row_sums.min():.2e} max={row_sums.max():.2e}"
        )
        return floored

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_weights(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Produce softmax + convex-floor weights for each row.

        H3 fix:
          - X is transformed with the StandardScaler fit during train().
          - Saturated models (alpha == grid_max) have their raw score replaced
            by 0.0 before softmax — this prevents a near-constant Ridge output
            from polluting softmax with effectively-uniform contributions and
            ensures the floor controls their final weight.

        Assertions (hard failures):
          - weights_df.sum(axis=1) == 1.0 within 1e-10.
          - weights_df.min(axis=1) >= weight_floor within 1e-10.
        """
        if self.models is None:
            raise RuntimeError("MetaLearner not trained — call train() first.")

        # H3 fix: scale X with the train-fit scaler. v1 joblibs (no scaler) skip.
        X_scaled = (
            self._feature_scaler.transform(X.values)
            if self._feature_scaler is not None
            else X.values
        )

        raw_scores = np.column_stack([m.predict(X_scaled) for m in self.models])

        # H3 fix: zero out raw scores for saturated models so softmax weights
        # them by the floor only (no spurious variation from a near-constant
        # over-regularized Ridge).
        if self.saturated:
            for i, name in enumerate(self.model_names):
                if self.saturated.get(name, False):
                    raw_scores[:, i] = 0.0

        sm         = softmax(raw_scores, axis=1)
        floored    = self._apply_weight_floor(sm)

        weights_df = pd.DataFrame(floored, index=X.index, columns=self.model_names)

        assert (weights_df.sum(axis=1) - 1.0).abs().max() < 1e-10, \
            "predict_weights: row sums != 1.0"
        assert weights_df.min(axis=1).min() >= self.weight_floor - 1e-10, \
            f"predict_weights: weight below floor {self.weight_floor}"

        return weights_df

    # ------------------------------------------------------------------
    # Lift evaluation
    # ------------------------------------------------------------------
    def evaluate_lift(
        self,
        weights_df: pd.DataFrame,
        ensemble_preds_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        target_col: str = "SPY_tgt_ret_1d",
    ) -> dict:
        """
        Meta-weighted signal vs equal-weight signal, annualized Sharpe.

        Uses the unified 4-model directional combination (Option 3 redesign):
          d_return   = +return_pred         (raw direction)
          d_regime   = -regime_pred         (low z = bull → positive)
          d_vol      = -vol_pred            (high vol = risk-off → negative)
          d_drawdown = -drawdown_risk_prob  (high prob = risk-off → negative)

          meta_signal[t]  = sum_i  w_i[t] * d_i[t]
          equal_signal[t] = sum_i  0.25  * d_i[t]
          pnl[t]          = signal[t-1] * actual_return[t]
          Sharpe          = mean(pnl) / std(pnl) * sqrt(252)
          lift_pct        = (meta_sharpe - equal_sharpe) / |equal_sharpe| * 100
        """
        ep = ensemble_preds_df.reindex(weights_df.index)
        if target_col not in targets_df.columns:
            logger.warning("evaluate_lift: {} not found in targets_df", target_col)
            return {}

        # Convention A (causal, realization-date indexed): pair signal[t-1]
        # with log(close[t]/close[t-1]) = SPY_tgt_ret_1d[t-1]. SPY_tgt_ret_1d
        # is forward-stamped, so .shift(1) converts forward → backward.
        # Aligns with engine.feedback.step (live), build_proxy_target
        # (training objective), and verify/evaluate_meta_lift.py (gate).
        actual_ret = targets_df[target_col].shift(1).reindex(weights_df.index)

        # Per-model directional signals (consistent with build_proxy_target
        # and engine.feedback.step)
        d_return   =  ep["return_pred"]
        d_vol      = -ep["vol_pred"]            if "vol_pred"           in ep else 0.0
        d_regime   = -ep["regime_pred"]         if "regime_pred"        in ep else 0.0
        d_drawdown = -ep["drawdown_risk_prob"]  if "drawdown_risk_prob" in ep else 0.0

        # Risk-only signals are NaN before val (vol_pred warm-up, regime warm-up)
        # — fillna(0) so they contribute zero (not NaN) to the weighted sum.
        d_vol      = d_vol.fillna(0)      if hasattr(d_vol,    "fillna") else d_vol
        d_regime   = d_regime.fillna(0)   if hasattr(d_regime, "fillna") else d_regime
        d_drawdown = d_drawdown.fillna(0) if hasattr(d_drawdown, "fillna") else d_drawdown

        meta_signal = (
            weights_df["return"]   * d_return
            + weights_df["vol"]      * d_vol
            + weights_df["regime"]   * d_regime
            + weights_df["drawdown"] * d_drawdown
        )
        equal_signal = 0.25 * (d_return + d_vol + d_regime + d_drawdown)

        meta_pnl  = (meta_signal.shift(1)  * actual_ret).dropna()
        equal_pnl = (equal_signal.shift(1) * actual_ret).dropna()

        def annualized_sharpe(pnl: pd.Series) -> float:
            if len(pnl) < 2 or pnl.std() == 0:
                return float("nan")
            return float(pnl.mean() / pnl.std() * np.sqrt(252))

        meta_sharpe  = annualized_sharpe(meta_pnl)
        equal_sharpe = annualized_sharpe(equal_pnl)

        if not np.isnan(equal_sharpe) and equal_sharpe != 0:
            lift_pct = float((meta_sharpe - equal_sharpe) / abs(equal_sharpe) * 100)
        else:
            lift_pct = float("nan")

        metrics = {
            "meta_sharpe":  round(meta_sharpe, 4),
            "equal_sharpe": round(equal_sharpe, 4),
            "lift_pct":     round(lift_pct, 2),
            "n_days":       int(len(meta_pnl)),
            "eval_date_range": [
                str(weights_df.index.min().date()),
                str(weights_df.index.max().date()),
            ],
            "KNOWN_LIMITATIONS": [
                "Lift is NOT a valid OOS estimate. Three layers of leakage exist:",
                "(L1) Base models (A/B/C/D) were tuned on val set, so val-set",
                "     predictions are already optimistic.",
                "(L2) Meta-features encode val-set base-model performance.",
                "(L3) Meta-learner trained on val_meta_train (70%) and evaluated",
                "     on val_meta_eval (30%) — FIX 3 addresses Layer 3 only.",
                "True OOS evaluation happens in Step 5.2 walk-forward engine.",
                "Proxy-target construction uses independent per-model ridge and",
                "non-commensurable z-scored skill scores; this is a placeholder",
                "expected to be replaced by an end-to-end Sharpe objective in Phase 5.",
            ],
        }

        logger.info(
            "evaluate_lift: meta_sharpe={:.4f}  equal_sharpe={:.4f}  lift={:.2f}%  n={}",
            meta_sharpe, equal_sharpe, lift_pct, int(len(meta_pnl)),
        )

        eval_path = PROJECT_ROOT / "logs" / "meta_learner_eval.json"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info("Eval saved → {}", eval_path.relative_to(PROJECT_ROOT))
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str = "models/meta/meta_learner.joblib") -> None:
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "models":          self.models,
                "feature_cols":    self.feature_cols,
                "model_names":     self.model_names,
                "weight_floor":    self.weight_floor,
                "n_models":        self.n_models,
                "future_window":   self.future_window,
                "alphas":          self.alphas,
                "feature_scaler":  self._feature_scaler,
                "saturated":       self.saturated,
            },
            p,
        )
        logger.info("MetaLearner saved → {}", p.relative_to(PROJECT_ROOT))

    def load(self, path: str = "models/meta/meta_learner.joblib") -> None:
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        d = joblib.load(p)
        self.models           = d["models"]
        self.feature_cols     = d["feature_cols"]
        self.model_names      = d["model_names"]
        self.weight_floor     = d["weight_floor"]
        self.n_models         = d["n_models"]
        self.future_window    = d["future_window"]
        # v2 fields — fall back to v1-compatible defaults if absent
        self.alphas           = d.get("alphas", self.alphas)
        self._feature_scaler  = d.get("feature_scaler", None)
        self.saturated        = d.get("saturated", {})
        logger.info("MetaLearner loaded ← {}", p.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    os.environ.setdefault("RFIE_TRAINING_MODE", "1")

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "meta_learner.log", rotation="10 MB", retention="30 days")

    import yaml
    from data.temporal_split import TemporalSplitter
    from models.meta.weight_tracker import WeightTracker

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    for p, label in [
        (META_FEATURES_PATH, "meta_features.parquet"),
        (ENSEMBLE_PATH,      "ensemble_predictions.parquet"),
        (TARGETS_PATH,       "targets.parquet"),
    ]:
        if not p.exists():
            print(f"Missing {p.relative_to(PROJECT_ROOT)} — run prerequisite scripts first.")
            sys.exit(1)

    meta_feats = pd.read_parquet(META_FEATURES_PATH)
    ens_preds  = pd.read_parquet(ENSEMBLE_PATH)
    targets    = pd.read_parquet(TARGETS_PATH)
    splitter   = TemporalSplitter(config)

    ml = MetaLearner()
    X_tr, y_tr, X_ev, y_ev = ml.prepare_data(meta_feats, ens_preds, targets, splitter)
    ml.train(X_tr, y_tr)

    # Predict weights on held-out val_meta_eval slice (FIX 3)
    weights_eval = ml.predict_weights(X_ev)

    # ------------------------------------------------------------------
    # Verification prints
    # ------------------------------------------------------------------
    print("\n=== MetaLearner Step 3.2 Verification ===")

    print("\n[1] weights_eval row sums (min=max must equal 1.0):")
    print(weights_eval.sum(axis=1).describe().round(10).to_string())

    min_w = float(weights_eval.min(axis=1).min())
    print(f"\n[2] weights_eval min weight: {min_w:.8f}  "
          f"(floor={ml.weight_floor})  "
          f"{'PASS' if min_w >= ml.weight_floor - 1e-10 else 'FAIL'}")

    print("\n[3] Chosen Ridge alpha per model:")
    for name, model in zip(ml.model_names, ml.models):
        print(f"    {name:10s}: alpha={model.alpha_}")

    print("\n[4] weights_eval distribution:")
    print(weights_eval.describe().round(4).to_string())

    lift = ml.evaluate_lift(weights_eval, ens_preds, targets)
    print(f"\n[5] Lift (eval on val_meta_eval — Layer 3 fixed, L1/L2 remain):")
    print(f"    meta_sharpe:  {lift.get('meta_sharpe')}")
    print(f"    equal_sharpe: {lift.get('equal_sharpe')}")
    print(f"    lift_pct:     {lift.get('lift_pct')}%")
    print(f"    n_days:       {lift.get('n_days')}")
    print(f"    eval_range:   {lift.get('eval_date_range')}")

    ml.save()

    # Log full weight history (train + eval) for audit trail
    tracker  = WeightTracker()
    all_X    = pd.concat([X_tr, X_ev])
    all_w    = ml.predict_weights(all_X)
    tracker.log_weights_batch(all_w)
    n_rows   = len(tracker.load_weights())
    print(f"\n[6] weight_audit.csv rows: {n_rows}  "
          f"(expected: {len(X_tr) + len(X_ev)})  "
          f"{'PASS' if n_rows == len(X_tr) + len(X_ev) else 'FAIL'}")

    print(f"\nModel saved → models/meta/meta_learner.joblib")
    print(f"Eval  saved → logs/meta_learner_eval.json")
    sys.exit(0)
