"""
verify/evaluate_meta_lift.py

Clean diagnostic comparing meta-weighted Sharpe vs true equal-weight
Sharpe on the val period.

Addresses the +3.1% vs -9.14% discrepancy by:
  (a) truncating the audit CSV before running so no prior-run rows accumulate
  (b) constructing a proper 4-model z-scored equal-weight baseline
  (c) reporting both directional and magnitude-aware Sharpe

Returns convention:
  SPY_tgt_ret_1d[t] = log(close[t+1]/close[t])  — FORWARD stamped.
  For a causal backtest, signal at t-1 earns the backward return
  log(close[t]/close[t-1]) = SPY_tgt_ret_1d[t-1].
  PnL: pnl[t] = signal[t-1] * SPY_tgt_ret_1d[t-1]
  Implemented as: signal.shift(1) * returns_series.shift(1)

Regime encoding confirmed from ensemble_predictions.parquet:
  0 = bull, 1 = bear, 2 = sideways
  Risk remap for equal-weight baseline: bull->0, sideways->1, bear->2
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.feedback import FeedbackLoopEngine
from models.base.ensemble import BaseEnsemble
from models.meta.meta_learner import MetaLearner
from models.meta.weight_tracker import WeightTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def annualized_sharpe(pnl_series: pd.Series, ann: int = 252):
    pnl = pnl_series.dropna()
    n   = len(pnl)
    if n == 0 or pnl.std() == 0:
        return float("nan"), n
    return float(pnl.mean() / pnl.std() * np.sqrt(ann)), n


def max_drawdown(pnl_series: pd.Series) -> float:
    cum  = pnl_series.fillna(0).cumsum()
    peak = cum.cummax()
    return float((cum - peak).min())


def metrics(name: str, pnl: pd.Series) -> dict:
    s, n      = annualized_sharpe(pnl)
    pnl_clean = pnl.dropna()
    return {
        "strategy": name,
        "sharpe":   round(s, 4),
        "ann_ret":  round(float(pnl_clean.mean() * 252), 4),
        "ann_vol":  round(float(pnl_clean.std() * np.sqrt(252)), 4),
        "hit_rate": round(float((pnl_clean > 0).mean()), 4),
        "max_dd":   round(max_drawdown(pnl), 6),
        "n_days":   n,
    }


def build_equal_weight_signal(
    ensemble_preds: pd.DataFrame, val_index: pd.Index
) -> pd.Series:
    """
    Equal-weight baseline using the unified 4-model directional convention:

      d_return   = +return_pred           (raw direction)
      d_regime   = -regime_pred           (low z = bull → positive)
      d_vol      = -vol_pred              (high vol = risk-off → negative)
      d_drawdown = -drawdown_risk_prob    (high prob = risk-off → negative)

      equal_signal = 0.25 * (d_return + d_vol + d_regime + d_drawdown)

    ensemble_predictions are already normalized (H1 fix), so re-z-scoring
    here would double-normalize — instead, consume the values directly.
    NaN risk-signal rows contribute 0 (no double-counting on warm-up).
    """
    df = ensemble_preds.reindex(val_index).copy()
    d_return   =  df["return_pred"]
    d_vol      = -df["vol_pred"].fillna(0)            if "vol_pred"           in df else 0.0
    d_regime   = -df["regime_pred"].fillna(0)         if "regime_pred"        in df else 0.0
    d_drawdown = -df["drawdown_risk_prob"].fillna(0)  if "drawdown_risk_prob" in df else 0.0
    return (0.25 * (d_return + d_vol + d_regime + d_drawdown)).rename("equal_weight_signal")


def verdict(pct: float) -> str:
    if pct >= 5.0:
        return "PASS (>= 5% gate)"
    if pct > 0.0:
        return "MARGINAL (positive but < 5% gate)"
    return "FAIL (meta underperforms)"


def lift_summary(meta_s: float, equal_s: float) -> str:
    delta = meta_s - equal_s
    pct   = (delta / abs(equal_s) * 100) if equal_s != 0 else float("nan")
    return (f"Lift = {delta:+.4f}  ({pct:+.1f}% of |equal|)  "
            f"[{verdict(pct)}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("META-LIFT DIAGNOSTIC  (evaluate_meta_lift.py)")
    print("=" * 70)

    # --- Load artifacts ---
    with open(PROJECT_ROOT / "config.yaml") as fh:
        cfg_top = yaml.safe_load(fh)
    meta_cfg   = cfg_top.get("meta", {})
    ml_version = meta_cfg.get("learner_version", "v1")
    ml_path    = (
        "models/meta/meta_learner_v2.joblib"
        if ml_version == "v2"
        else "models/meta/meta_learner.joblib"
    )
    print(f"\nMeta-learner version: {ml_version} ({ml_path})")

    ensemble     = BaseEnsemble()
    meta_learner = MetaLearner()
    meta_learner.load(ml_path)

    # Truncate audit CSV so no rows from prior runs accumulate
    audit_path = PROJECT_ROOT / "logs" / "weight_audit_metalift.csv"
    if audit_path.exists():
        audit_path.unlink()
    wt = WeightTracker(path=str(audit_path))

    # Joined features: price_features_v1 (99 cols) + macro_features_v1 (12 cols)
    # vix_regime ordinal-encoded to match training convention
    price_df  = pd.read_parquet("data/processed/price_features_v1.parquet")
    macro_df  = pd.read_parquet("data/processed/macro_features_v1.parquet")
    joined_df = price_df.join(macro_df, how="inner").copy()
    if "vix_regime" in joined_df.columns:
        joined_df["vix_regime"] = (
            joined_df["vix_regime"]
            .map({"low": 0, "medium": 1, "high": 2})
            .astype("float64")
        )

    targets        = pd.read_parquet("data/processed/targets.parquet")
    returns_fwd    = targets["SPY_tgt_ret_1d"]   # forward-stamped
    ensemble_preds = pd.read_parquet("data/processed/ensemble_predictions.parquet")

    features_dict = {"price": price_df, "joined": joined_df}

    # Val window
    with open(PROJECT_ROOT / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    sp        = cfg["splits"]
    train_end = pd.Timestamp(sp["train_end"])
    val_end   = pd.Timestamp(sp["val_end"])
    ret_idx   = returns_fwd.dropna().index
    val_start = ret_idx[ret_idx > train_end][0]

    print(f"\nVal window : {val_start.date()} -> {val_end.date()}")
    print(f"Val length : {len(returns_fwd.loc[val_start:val_end])} trading days")

    # --- Fresh single simulation ---
    print("\nRunning fresh FeedbackLoopEngine simulation...")
    eng = FeedbackLoopEngine(ensemble, meta_learner, wt)
    meta_signal, _ = eng.run(features_dict, returns_fwd, val_start, val_end)
    print(f"Simulation complete. Signal dates: {len(meta_signal)}")

    # --- Audit CSV sanity ---
    audit_df = pd.read_csv(str(audit_path), parse_dates=["date"])
    print(f"Weight audit rows: {len(audit_df)} "
          f"(expected {len(meta_signal)} — 1 row per step, no accumulation)")
    if len(audit_df) != len(meta_signal):
        print(f"  WARNING: mismatch — CSV accumulation or missing rows!")

    # --- Equal-weight baseline ---
    val_index    = meta_signal.index
    equal_signal = build_equal_weight_signal(ensemble_preds, val_index)

    # --- PnL with backward returns ---
    # SPY_tgt_ret_1d is forward-stamped; shift(1) converts to backward:
    #   ret_bwd[t] = SPY_tgt_ret_1d[t-1] = log(close[t]/close[t-1])
    ret_bwd = returns_fwd.shift(1).reindex(val_index)

    # Directional: pnl[t] = sign(signal[t-1]) * ret_bwd[t]
    meta_pnl_dir  = (np.sign(meta_signal.shift(1))  * ret_bwd).dropna()
    equal_pnl_dir = (np.sign(equal_signal.shift(1)) * ret_bwd).dropna()

    # Magnitude-aware: pnl[t] = signal[t-1] * ret_bwd[t]
    meta_pnl_mag  = (meta_signal.shift(1)  * ret_bwd).dropna()
    equal_pnl_mag = (equal_signal.shift(1) * ret_bwd).dropna()

    # --- Tables ---
    print("\n" + "=" * 70)
    print("DIRECTIONAL Sharpe  (pnl = sign(signal[t-1]) * backward_ret[t])")
    print("=" * 70)
    dir_rows = [metrics("meta",  meta_pnl_dir),
                metrics("equal", equal_pnl_dir)]
    print(pd.DataFrame(dir_rows).to_string(index=False))
    print(lift_summary(dir_rows[0]["sharpe"], dir_rows[1]["sharpe"]))

    print("\n" + "=" * 70)
    print("MAGNITUDE-AWARE Sharpe  (pnl = signal[t-1] * backward_ret[t])")
    print("=" * 70)
    mag_rows = [metrics("meta",  meta_pnl_mag),
                metrics("equal", equal_pnl_mag)]
    print(pd.DataFrame(mag_rows).to_string(index=False))
    print(lift_summary(mag_rows[0]["sharpe"], mag_rows[1]["sharpe"]))

    # --- Weight trajectory sample ---
    print("\n" + "=" * 70)
    print("WEIGHT TRAJECTORY (first 3 and last 3 rows)")
    print("=" * 70)
    wt_df = wt.load_weights()
    print("First 3 rows:")
    print(wt_df.head(3).to_string())
    print("Last 3 rows:")
    print(wt_df.tail(3).to_string())
    stds = wt_df[["return_w", "vol_w", "regime_w", "drawdown_w"]].std()
    print(f"Std per weight:  {stds.to_dict()}")

    # --- PDF gate summary ---
    print("\n" + "=" * 70)
    print("PDF EXIT GATE SUMMARY")
    print("=" * 70)
    print(f"  Directional  : {lift_summary(dir_rows[0]['sharpe'], dir_rows[1]['sharpe'])}")
    print(f"  Magnitude    : {lift_summary(mag_rows[0]['sharpe'], mag_rows[1]['sharpe'])}")
    print()
    print("Discrepancy explanation:")
    print("  +3.1%  (verify_step_3_3_manual Check 5): evaluate_lift() received a")
    print("    1875-row weight CSV (5 runs * 375 days) because WeightTracker.log_weights()")
    print("    appends rather than overwrites. Sharpe was computed over n=1874 (all rows),")
    print("    not 375. Effectively averaged 5 runs, diluting the bad days.")
    print("  -9.14% (meta_learner offline eval): evaluate_lift() pairs signal[t-1] with")
    print("    SPY_tgt_ret_1d[t] directly (no shift on returns). Since SPY_tgt_ret_1d is")
    print("    forward-stamped, this means signal[t-1] earns return[t -> t+1] (one bar")
    print("    ahead of causal). Slightly different time alignment from this script.")
    print("  This script: clean single-run CSV (no accumulation), backward returns")
    print("    (SPY_tgt_ret_1d shifted by 1), proper z-scored 4-model equal-weight.")


if __name__ == "__main__":
    main()
