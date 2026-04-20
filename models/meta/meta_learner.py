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
        self.models: Optional[List[RidgeCV]] = None
        self.model_names   = ["return", "vol", "regime", "drawdown"]
        self.feature_cols: Optional[List[str]] = None

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
        For each date t in meta_features_df.index, compute a 4-vector of
        forward skill scores over [t+1, t+future_window].

          score_return[t]   = Spearman(return_pred, SPY_tgt_ret_1d) over window
          score_vol[t]      = -RMSE(vol_pred, SPY_tgt_vol_5d)
          score_regime[t]   = Spearman IC between regime_directional and forward return
                              regime mapping: 0→+1 (bull), 1→-1 (bear), 2→0 (neutral)
                              Sideways days contribute 0 (not excluded) so Spearman
                              can compute over the full 21d window. Requires > 1
                              unique value in both series; else NaN.
          score_drawdown[t] = -Brier(drawdown_risk_prob,
                               (SPY_tgt_mdd_21d < -0.05))
                              Works for single-class windows (2022 bear persistence).

        After scoring: drop rows with any NaN, then z-score each column.

        NOTE: the forward window is the TRAINING TARGET — looking forward in y
        is standard supervised ML. The anti-lookahead contract applies to X.
        """
        from scipy.stats import spearmanr

        ep  = ensemble_preds_df
        tgt = targets_df

        ret_col = "SPY_tgt_ret_1d"  if "SPY_tgt_ret_1d"  in tgt.columns else None
        vol_col = "SPY_tgt_vol_5d"  if "SPY_tgt_vol_5d"  in tgt.columns else None
        mdd_col = "SPY_tgt_mdd_21d" if "SPY_tgt_mdd_21d" in tgt.columns else None

        ret_tgt = tgt[ret_col].reindex(ep.index) if ret_col else None
        vol_tgt = tgt[vol_col].reindex(ep.index) if vol_col else None
        mdd_tgt = tgt[mdd_col].reindex(ep.index) if mdd_col else None

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

            w = slice(pos + 1, pos + self.future_window + 1)

            # return: Spearman IC
            score_ret = np.nan
            if ret_tgt is not None and "return_pred" in ep.columns:
                rp   = ep["return_pred"].iloc[w]
                rt   = ret_tgt.iloc[w]
                mask = rp.notna() & rt.notna()
                if mask.sum() >= min_obs:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        rho, _ = spearmanr(rp[mask], rt[mask])
                    if not np.isnan(rho):
                        score_ret = float(rho)

            # vol: negative RMSE
            score_vol = np.nan
            if vol_tgt is not None and "vol_pred" in ep.columns:
                vp   = ep["vol_pred"].iloc[w]
                vt   = vol_tgt.iloc[w]
                mask = vp.notna() & vt.notna()
                if mask.sum() >= min_obs:
                    score_vol = -float(
                        np.sqrt(np.mean((vp[mask].values - vt[mask].values) ** 2))
                    )

            # regime: Spearman IC between regime_directional and forward return
            # Mapping: 0 (bull) -> +1, 1 (bear) -> -1, 2 (sideways) -> 0
            # Sideways days contribute zero signal but are NOT excluded from the
            # window — they just get a neutral directional value of 0. This lets
            # Spearman still compute over the full 21d window.
            #
            # Degenerate-window policy (0.0, NOT NaN):
            #   All-sideways: reg_dir all 0  → constant → Spearman undefined.
            #   All-same-dir: reg_dir all +1/-1 → constant → Spearman undefined.
            #   Both cases score 0.0 ("regime adds no discriminative signal in
            #   this window") so the row is retained in the proxy target.
            #   This prevents ~215/354 val windows from being dropped by dropna().
            score_reg = np.nan
            if ret_tgt is not None and "regime_pred" in ep.columns:
                rg      = ep["regime_pred"].iloc[w]
                rt      = ret_tgt.iloc[w]
                reg_dir = (rg == 0).astype(float) - (rg == 1).astype(float)
                mask    = reg_dir.notna() & rt.notna()
                if mask.sum() >= min_obs:
                    if reg_dir[mask].nunique() > 1 and rt[mask].nunique() > 1:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            rho, _ = spearmanr(reg_dir[mask], rt[mask])
                        score_reg = float(rho) if not np.isnan(rho) else 0.0
                    else:
                        # Constant reg_dir (all-sideways or all-same-direction):
                        # no informative signal in this window.
                        score_reg = 0.0

            # drawdown: negative Brier score (MSE between prob and binary label)
            # Brier score = mean((p - y)^2). Negative so higher = better.
            # Unlike log_loss, Brier score does NOT require both classes present
            # in the window. This prevents ~151/354 val windows from dropping
            # due to the 2022 persistent-bear period (all SPY_tgt_mdd_21d < -0.05
            # in windows entirely within the bear regime → all-positive binary).
            score_dd = np.nan
            if mdd_tgt is not None and "drawdown_risk_prob" in ep.columns:
                dp   = ep["drawdown_risk_prob"].iloc[w]
                mt   = mdd_tgt.iloc[w]
                mask = dp.notna() & mt.notna()
                if mask.sum() >= min_obs:
                    binary   = (mt[mask] < -0.05).astype(float)
                    probs    = dp[mask].clip(1e-7, 1 - 1e-7).values
                    score_dd = -float(np.mean((probs - binary.values) ** 2))

            rows.append([score_ret, score_vol, score_reg, score_dd])

        out = pd.DataFrame(rows, index=meta_features_df.index, columns=self.model_names)

        # Drop rows with any NaN score (window edges, insufficient data)
        out = out.dropna()

        # Z-score each column on the remaining (complete) rows
        for col in out.columns:
            if out[col].std() > 0:
                out[col] = (out[col] - out[col].mean()) / out[col].std()

        logger.info(
            "build_proxy_target: {} complete rows returned (all 4 scores non-NaN, z-scored)",
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
        """
        tscv = TimeSeriesSplit(n_splits=self.cv_splits, gap=self.cv_gap)
        self.models = []
        for col in self.model_names:
            m = RidgeCV(alphas=self.alphas, cv=tscv)
            m.fit(X.values, y[col].values)
            self.models.append(m)
            logger.info("train: model={} alpha_chosen={}", col, m.alpha_)

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

        Assertions (hard failures):
          - weights_df.sum(axis=1) == 1.0 within 1e-10.
          - weights_df.min(axis=1) >= weight_floor within 1e-10.
        """
        if self.models is None:
            raise RuntimeError("MetaLearner not trained — call train() first.")

        raw_scores = np.column_stack([m.predict(X.values) for m in self.models])
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

        Uses only return_pred and regime_pred for direction (vol_pred and
        drawdown_risk_prob are risk signals, not directional).

          regime_directional: {0: +1, 1: -1, 2: 0}
          meta_signal[t]  = w_return * return_pred + w_regime * regime_dir * |return_pred|
          equal_signal[t] = 0.5 * return_pred + 0.5 * regime_dir * |return_pred|
          pnl[t]          = signal[t-1] * actual_return[t]
          Sharpe          = mean(pnl) / std(pnl) * sqrt(252)
          lift_pct        = (meta_sharpe - equal_sharpe) / |equal_sharpe| * 100
        """
        ep = ensemble_preds_df.reindex(weights_df.index)
        if target_col not in targets_df.columns:
            logger.warning("evaluate_lift: {} not found in targets_df", target_col)
            return {}

        actual_ret  = targets_df[target_col].reindex(weights_df.index)
        ret_pred    = ep["return_pred"]
        regime_pred = ep["regime_pred"]
        regime_dir  = (regime_pred == 0).astype(float) - (regime_pred == 1).astype(float)

        meta_signal  = (
            weights_df["return"] * ret_pred
            + weights_df["regime"] * regime_dir * ret_pred.abs()
        )
        equal_signal = 0.5 * ret_pred + 0.5 * regime_dir * ret_pred.abs()

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
                "models":        self.models,
                "feature_cols":  self.feature_cols,
                "model_names":   self.model_names,
                "weight_floor":  self.weight_floor,
                "n_models":      self.n_models,
                "future_window": self.future_window,
            },
            p,
        )
        logger.info("MetaLearner saved → {}", p.relative_to(PROJECT_ROOT))

    def load(self, path: str = "models/meta/meta_learner.joblib") -> None:
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        d = joblib.load(p)
        self.models        = d["models"]
        self.feature_cols  = d["feature_cols"]
        self.model_names   = d["model_names"]
        self.weight_floor  = d["weight_floor"]
        self.n_models      = d["n_models"]
        self.future_window = d["future_window"]
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
