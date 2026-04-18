"""
models/meta/meta_features.py
Builds meta-features that describe each base model's recent predictive quality.

These features are consumed by the Phase 3 meta-learner to dynamically weight
the base models based on their recent track records.

Anti-lookahead contract
-----------------------
  All rolling computations use only information observable by the close of
  t-1 (for diagnostics excluding t) or t (for backtest PnL features):

  rolling_ic (excludes t):
      correlation over [t-window, t-1], using targets shifted by their
      full horizon + 1-day boundary margin (see TARGET_HORIZONS).
      - SPY_tgt_ret_1d:   shift(2)
      - SPY_tgt_vol_5d:   shift(6)
      - SPY_tgt_mdd_21d: shift(22)  (future use)

  rolling_pnl (includes t, backtest convention):
      daily_pnl[t] = sign(pred[t-1]) * return[t]
      return_pred is mean-centered via expanding().mean() before sign()
      to remove the model's non-zero baseline bias.

  rolling_disagreement (includes t):
      rolling mean of ensemble disagreement, which is already causal
      upstream (BaseEnsemble.compute_disagreement fits scaler on
      val period only, Step 2.5).

  regime_accuracy (includes t, backtest convention):
      rolling hit rate — fraction of days where sign(regime_signal[t-1])
      matches sign(return[t]). Sideways days (signal=0) excluded from
      the rolling denominator. Uses hit rate instead of Spearman IC
      because HMM regimes are highly persistent and Spearman is undefined
      on constant windows.

  Warmup policy: strict min_periods=window for all features. The first
  `window` rows per feature are NaN by design — per the Step 3.1 situation
  report's "first 21 rows will be NaN (warmup period)" spec.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import warnings

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy import stats
from scipy.stats import ConstantInputWarning

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH  = PROJECT_ROOT / "config.yaml"

ENSEMBLE_PATH = PROJECT_ROOT / "data" / "processed" / "ensemble_predictions.parquet"
OUTPUT_PATH   = PROJECT_ROOT / "data" / "processed" / "meta_features.parquet"

# Shift amounts per target column. A target with horizon H means the value
# at index t is realised over [t, t+H]. To make it fully past-at-index-t,
# shift by H (minimum) or H+1 (with boundary margin). We use H+1 for
# boundary margin — mirrors the defensive posture from Phase 1.
TARGET_HORIZONS = {
    "SPY_tgt_ret_1d":   2,   # 1-day forward return, +1 margin
    "SPY_tgt_vol_5d":   6,   # 5-day forward vol,    +1 margin
    "SPY_tgt_mdd_21d": 22,   # 21-day MDD,           +1 margin (future use)
}


def _causal_target(tgt: pd.Series, tgt_name: str) -> pd.Series:
    """
    Shift a target series so its value at index t is fully realised by
    the close of t-1 (no future contamination).
    """
    if tgt_name not in TARGET_HORIZONS:
        raise KeyError(
            f"Target {tgt_name!r} has no registered horizon. "
            f"Add it to TARGET_HORIZONS in meta_features.py before use."
        )
    return tgt.shift(TARGET_HORIZONS[tgt_name])


def _load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Meta-feature builder
# ---------------------------------------------------------------------------
class MetaFeatureBuilder:
    """
    Computes rolling meta-features from base-model predictions.

    Parameters
    ----------
    window : int
        Default rolling window in trading days (default 21).
    """

    def __init__(self, window: int = 21) -> None:
        self.window = window

    # ------------------------------------------------------------------
    # Rolling IC
    # ------------------------------------------------------------------
    def rolling_ic(
        self,
        predictions_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Rolling Spearman IC between model predictions and realised targets.

        For each date t, IC is computed over [t-window, t-1] — exactly `window`
        values, with t itself excluded. Target shifts are pulled from
        TARGET_HORIZONS so multi-horizon targets remain fully causal.
        """
        w = window or self.window

        ret_pred = predictions_df["return_pred"].dropna()
        vol_pred = predictions_df["vol_pred"].dropna()

        jobs = [
            ("rolling_ic_return", ret_pred, "SPY_tgt_ret_1d"),
            ("rolling_ic_vol",    vol_pred, "SPY_tgt_vol_5d"),
        ]

        results = {}
        for col, pred, tgt_name in jobs:
            if tgt_name not in targets_df.columns:
                logger.warning(
                    "rolling_ic: target column {} missing for {} — skipping",
                    tgt_name, col,
                )
                continue

            tgt_causal = _causal_target(targets_df[tgt_name], tgt_name)
            p, t_shifted = pred.align(tgt_causal, join="inner")

            ic_vals = np.full(len(p), np.nan)
            idx = p.index

            # For each t, use window [t-w, t-1] — exactly w values, t excluded.
            # i ranges from w (not w-1) so iloc[i-w:i] has w elements.
            for i in range(w, len(p)):
                p_win = p.iloc[i - w : i]
                t_win = t_shifted.iloc[i - w : i]
                mask = p_win.notna() & t_win.notna()
                if mask.sum() < w:  # strict warmup — any NaN in window → NaN IC
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConstantInputWarning)
                    rho, _ = stats.spearmanr(p_win[mask], t_win[mask])
                ic_vals[i] = rho

            results[col] = pd.Series(ic_vals, index=idx)

        out = pd.DataFrame(results)
        logger.info(
            "rolling_ic: {} rows, non-NaN per col: {}",
            len(out),
            {c: int(out[c].notna().sum()) for c in out.columns},
        )
        return out

    # ------------------------------------------------------------------
    # Rolling PnL
    # ------------------------------------------------------------------
    def rolling_pnl(
        self,
        predictions_df: pd.DataFrame,
        returns_df: pd.Series,
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Rolling PnL from naively following each model's directional signal.

        For return_pred:
          - MEAN-CENTERED sign: signal = sign(return_pred - expanding_mean)
            Model A's predictions are tightly clustered around a non-zero
            baseline (~0.0007). Raw sign() would return +1 on nearly all days
            and the rolling PnL would mirror SPY buy-and-hold — not a useful
            signal about Model A's directional skill. Mean-centering removes
            that baseline using an EXPANDING (causal) mean.
          - daily_pnl = signal.shift(1) * actual_return
          - rolling_pnl = daily_pnl.rolling(window).sum()

        For regime_pred:
          signal = (regime == 0) - (regime == 1)  — bull=+1, bear=-1, sideways=0
          Already three-valued, no mean-centering needed.

        NOTE: .sum() is scale-sensitive to window length. Fine for a single
        window, but if multiple windows are added later, .mean() may be
        preferable for scale invariance.
        """
        w = window or self.window

        results = {}

        # --- return_pred signal (mean-centered) ---
        rp = predictions_df["return_pred"]
        rp_centered = rp - rp.expanding().mean()
        ret_signal = np.sign(rp_centered)
        daily_pnl_ret = ret_signal.shift(1) * returns_df.reindex(ret_signal.index)
        results["rolling_pnl_return"] = daily_pnl_ret.rolling(w, min_periods=w).sum()

        # --- regime_pred signal ---
        if "regime_pred" in predictions_df.columns:
            regime = predictions_df["regime_pred"]
            reg_signal = (regime == 0).astype(float) - (regime == 1).astype(float)
            daily_pnl_reg = reg_signal.shift(1) * returns_df.reindex(reg_signal.index)
            results["rolling_pnl_regime"] = daily_pnl_reg.rolling(w, min_periods=w).sum()

        out = pd.DataFrame(results)
        logger.info(
            "rolling_pnl: {} rows, non-NaN per col: {}",
            len(out),
            {c: int(out[c].notna().sum()) for c in out.columns},
        )
        return out

    # ------------------------------------------------------------------
    # Model disagreement
    # ------------------------------------------------------------------
    def model_disagreement(
        self,
        predictions_df: pd.DataFrame,
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Rolling mean of the ensemble disagreement score.

        Loads the disagreement column from predictions_df (or falls back to
        reading ensemble_predictions.parquet if the column is absent).

        Column produced
        ---------------
        rolling_disagreement : window-day rolling mean of disagreement
        """
        w = window or self.window

        if "disagreement" in predictions_df.columns:
            dis = predictions_df["disagreement"]
        else:
            logger.warning(
                "model_disagreement: 'disagreement' not in predictions_df — "
                "loading from {}",
                ENSEMBLE_PATH.relative_to(PROJECT_ROOT),
            )
            dis = pd.read_parquet(ENSEMBLE_PATH)["disagreement"]
            dis = dis.reindex(predictions_df.index)

        rolling_dis = dis.rolling(w, min_periods=w).mean().rename("rolling_disagreement")
        out = rolling_dis.to_frame()
        logger.info(
            "model_disagreement: {} rows, {:.0f}% non-NaN",
            len(out), out["rolling_disagreement"].notna().mean() * 100,
        )
        return out

    # ------------------------------------------------------------------
    # Regime accuracy
    # ------------------------------------------------------------------
    def regime_accuracy(
        self,
        regime_pred: pd.Series,
        realized_returns: pd.Series,
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Rolling hit rate between the LAGGED regime signal and the
        contemporaneous return.

        regime_pred: 0=bull, 1=bear, 2=sideways (HMM labels)
        regime signal: +1 for bull, -1 for bear, 0 for sideways
        shift(1) applied so that at date t we use regime[t-1] paired with
        return[t] — causal backtest convention.

        HIT RATE DEFINITION:
          A day "hits" when sign(regime_signal[t-1]) == sign(return[t]).
          Days where regime_signal == 0 (sideways) are excluded from the
          rolling denominator — they're not a directional call.
          Value range: [0, 1]. Random ≈ 0.5.

        WHY HIT RATE, NOT IC:
          Spearman correlation is undefined on constant windows. HMM regimes
          are highly persistent (diagonal > 0.96 per Step 2.4), so 21-day
          windows are often mono-regime, making Spearman NaN for ~69% of rows.
          Hit rate is a cleaner metric for a categorical signal against a
          continuous return and produces a useful value even on stable regimes.

        Window includes the current index i (consistent with backtest PnL,
        differs from rolling_ic which excludes i). At t, both regime[t-1] and
        return[t] are known by the close of t — causal.
        """
        w = window or self.window

        reg_signal = (
            (regime_pred == 0).astype(float) - (regime_pred == 1).astype(float)
        ).shift(1)
        aligned_ret = realized_returns.reindex(reg_signal.index)

        # Per-day hit indicator: 1 if signs agree, 0 if disagree, NaN if
        # either value is NaN OR if signal is 0 (sideways — no directional call)
        sig_sign = np.sign(reg_signal)
        ret_sign = np.sign(aligned_ret)
        hit = (sig_sign == ret_sign).astype(float)
        # Mask out sideways days (signal=0) and any NaN rows
        mask = (reg_signal != 0) & reg_signal.notna() & aligned_ret.notna()
        hit = hit.where(mask)  # everywhere mask is False → NaN

        # Rolling mean ignoring NaN, but require at least w//2 directional calls
        # in the window to avoid spurious values during mostly-sideways periods
        rolling_hit = hit.rolling(w, min_periods=w // 2).mean()

        out = rolling_hit.rename("rolling_regime_accuracy").to_frame()
        logger.info(
            "regime_accuracy (hit rate): {} rows, {:.0f}% non-NaN, mean hit rate: {:.3f}",
            len(out),
            out["rolling_regime_accuracy"].notna().mean() * 100,
            float(out["rolling_regime_accuracy"].mean()) if out["rolling_regime_accuracy"].notna().any() else float("nan"),
        )
        return out

    # ------------------------------------------------------------------
    # Build all
    # ------------------------------------------------------------------
    def build_all(
        self,
        predictions_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        returns_df: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute all meta-features, concatenate, save, and return.

        Parameters
        ----------
        predictions_df : pd.DataFrame
            Output of BaseEnsemble.predict_all() with all 5 columns.
        targets_df : pd.DataFrame
            Targets from data/processed/targets.parquet.
        returns_df : pd.Series
            SPY daily log returns, date-indexed.

        Returns
        -------
        pd.DataFrame saved to data/processed/meta_features.parquet.
        """
        parts = []

        logger.info("Building rolling_ic…")
        parts.append(self.rolling_ic(predictions_df, targets_df))

        logger.info("Building rolling_pnl…")
        parts.append(self.rolling_pnl(predictions_df, returns_df))

        logger.info("Building model_disagreement…")
        parts.append(self.model_disagreement(predictions_df))

        if "regime_pred" in predictions_df.columns:
            logger.info("Building regime_accuracy…")
            parts.append(
                self.regime_accuracy(predictions_df["regime_pred"], returns_df)
            )

        meta = pd.concat(parts, axis=1, sort=False)
        meta.index.name = "Date"

        # Sort index — pd.concat with sort=False preserves first-appearance order,
        # which can leave out-of-order rows when input DataFrames have different
        # index coverage (rolling_ic drops trailing target-shift NaN rows while
        # rolling_pnl spans the full history).
        meta = meta.sort_index()

        # Sanity-check: index must be monotonic after sort
        assert meta.index.is_monotonic_increasing, (
            "meta_features index is still non-monotonic after sort_index() — "
            "check for duplicate dates in source DataFrames."
        )

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        meta.to_parquet(OUTPUT_PATH)
        logger.info(
            "Meta-features saved → {} ({} rows × {} cols)",
            OUTPUT_PATH.relative_to(PROJECT_ROOT),
            len(meta),
            len(meta.columns),
        )
        return meta


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "meta_features.log", rotation="10 MB", retention="30 days")

    os.environ.setdefault("RFIE_TRAINING_MODE", "1")

    cfg          = _load_config()
    processed_dir = PROJECT_ROOT / cfg["paths"]["processed"]
    raw_dir       = PROJECT_ROOT / cfg["paths"]["raw"]

    for p, label in [
        (ENSEMBLE_PATH,                             "ensemble_predictions.parquet"),
        (processed_dir / "targets.parquet",        "targets.parquet"),
        (raw_dir       / "all_close_prices.parquet","all_close_prices.parquet"),
    ]:
        if not p.exists():
            print(f"Missing {p} — run prerequisite scripts first.")
            sys.exit(1)

    predictions_df = pd.read_parquet(ENSEMBLE_PATH)
    targets_df     = pd.read_parquet(processed_dir / "targets.parquet")
    close_df       = pd.read_parquet(raw_dir / "all_close_prices.parquet")
    spy_returns    = np.log(close_df["SPY"]).diff().rename("SPY_logret")

    builder = MetaFeatureBuilder(window=21)
    meta    = builder.build_all(predictions_df, targets_df, spy_returns)

    print("\n=== Meta-Features Summary ===")
    print(f"Shape:      {meta.shape}")
    print(f"Date range: {meta.index.min().date()} → {meta.index.max().date()}")
    print(f"Columns:    {meta.columns.tolist()}")
    print(f"\nNon-NaN counts:")
    print(meta.notna().sum().to_string())
    print(f"\nSaved → {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    sys.exit(0)
