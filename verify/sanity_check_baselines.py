"""
verify/sanity_check_baselines.py

Sanity check for Phase 3 exit gate: does meta_v2 beat FOUR independent
baselines on val by >= 5% directional Sharpe?

This rules out baseline-shopping: the +28.8% lift in evaluate_meta_lift.py
is measured against the v2-architecture equal-weight baseline. We need to
confirm v2 also beats:
  B1 — ORIGINAL Phase 2 baseline (0.25 * return_pred only; risk signals
       were never meant to be added directly to a directional return signal)
  B2 — Naive z-scored equal-weight (no sign flips, treat all 4 as
       directional return predictors)
  B3 — Z-scored equal-weight WITH v2 sign-flip convention (current baseline
       in evaluate_meta_lift.py — reproduced here)
  B4 — SPY buy-and-hold (signal = +1 every day)

PRELIMINARY (confirmed):
  - ensemble_predictions.parquet now stores z-scored predictions (H1 fix)
    so z_ret = ensemble_preds["return_pred"] directly.
  - Sharpe is scale-invariant under a constant multiplier, so B1 with
    z-scored ret_pred yields the same Sharpe as B1 with raw ret_pred.
  - Backward-returns convention: signal[t-1] earns ret_bwd[t] where
    ret_bwd = SPY_tgt_ret_1d.shift(1).

Read-only diagnostic. Does NOT modify any production file.
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

def _annualized_sharpe(pnl: pd.Series, ann: int = 252) -> float:
    p = pnl.dropna()
    if len(p) < 2 or p.std() == 0:
        return float("nan")
    return float(p.mean() / p.std() * np.sqrt(ann))


def _max_drawdown(pnl: pd.Series) -> float:
    cum  = pnl.fillna(0).cumsum()
    peak = cum.cummax()
    return float((cum - peak).min())


def _metrics(name: str, signal: pd.Series, ret_bwd: pd.Series) -> dict:
    """Compute directional + magnitude Sharpe for a signal."""
    pnl_dir = (np.sign(signal.shift(1)) * ret_bwd).dropna()
    pnl_mag = (signal.shift(1)            * ret_bwd).dropna()
    return {
        "strategy":   name,
        "dir_sharpe": round(_annualized_sharpe(pnl_dir), 4),
        "mag_sharpe": round(_annualized_sharpe(pnl_mag), 4),
        "ann_ret":    round(float(pnl_dir.mean() * 252), 4),
        "ann_vol":    round(float(pnl_dir.std() * np.sqrt(252)), 4),
        "hit_rate":   round(float((pnl_dir > 0).mean()), 4),
        "max_dd":     round(_max_drawdown(pnl_dir), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("BASELINE SANITY CHECK  meta_v2 vs B1, B2, B3, B4")
    print("=" * 70)

    # --- Config + val window --------------------------------------------------
    with open(PROJECT_ROOT / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    sp        = cfg["splits"]
    train_end = pd.Timestamp(sp["train_end"])
    val_end   = pd.Timestamp(sp["val_end"])

    targets     = pd.read_parquet("data/processed/targets.parquet")
    returns_fwd = targets["SPY_tgt_ret_1d"]
    ret_idx     = returns_fwd.dropna().index
    val_start   = ret_idx[ret_idx > train_end][0]

    print(f"\nVal window : {val_start.date()} -> {val_end.date()}")

    # --- Confirm normalization status ----------------------------------------
    ens_preds = pd.read_parquet("data/processed/ensemble_predictions.parquet")
    val_ens   = ens_preds.loc[val_start:val_end]
    rp_std    = float(val_ens["return_pred"].std())
    is_z      = 0.5 < rp_std < 2.5
    print(f"return_pred std on val = {rp_std:.4f}  ->  "
          f"{'z-scored (H1 fix detected)' if is_z else 'RAW (no H1 fix)'}")
    if not is_z:
        print("WARNING: parquet appears un-normalized — B1 will use raw scale.")

    # --- Run meta_v2 simulation ----------------------------------------------
    price_df  = pd.read_parquet("data/processed/price_features_v1.parquet")
    macro_df  = pd.read_parquet("data/processed/macro_features_v1.parquet")
    joined_df = price_df.join(macro_df, how="inner").copy()
    if "vix_regime" in joined_df.columns:
        joined_df["vix_regime"] = (
            joined_df["vix_regime"]
            .map({"low": 0, "medium": 1, "high": 2})
            .astype("float64")
        )

    ml_version = cfg.get("meta", {}).get("learner_version", "v1")
    ml_path    = (
        "models/meta/meta_learner_v2.joblib"
        if ml_version == "v2"
        else "models/meta/meta_learner.joblib"
    )
    print(f"Meta-learner: {ml_version} ({ml_path})")

    ensemble = BaseEnsemble()
    meta     = MetaLearner()
    meta.load(ml_path)

    audit_path = PROJECT_ROOT / "logs" / "weight_audit_sanity.csv"
    if audit_path.exists():
        audit_path.unlink()
    wt = WeightTracker(path=str(audit_path))

    eng = FeedbackLoopEngine(ensemble, meta, wt)
    print("\nRunning meta_v2 FeedbackLoopEngine simulation...")
    meta_signal, _ = eng.run(
        {"price": price_df, "joined": joined_df},
        returns_fwd, val_start, val_end,
    )
    val_index = meta_signal.index
    print(f"  meta_signal: {len(meta_signal)} dates")

    # --- Backward returns ----------------------------------------------------
    # SPY_tgt_ret_1d is forward-stamped. signal[t-1] earns the return from
    # t-1 -> t, which is SPY_tgt_ret_1d[t-1] = ret_bwd[t].
    ret_bwd = returns_fwd.shift(1).reindex(val_index)
    ep      = ens_preds.reindex(val_index)

    # Fill warm-up NaNs with 0 so baselines stay finite
    z_ret      = ep["return_pred"].fillna(0)
    z_vol      = ep["vol_pred"].fillna(0)
    z_regime   = ep["regime_pred"].fillna(0)
    z_drawdown = ep["drawdown_risk_prob"].fillna(0)

    # --- Construct 4 baselines -----------------------------------------------
    # B1 — ORIGINAL Phase 2 baseline. Only return_pred drives direction.
    #      Sharpe is scale-invariant for a constant 0.25 multiplier, so this is
    #      identical whether we use raw or z-scored ret_pred.
    signal_B1 = 0.25 * z_ret

    # B2 — Naive z-scored equal-weight, no sign flips. Treats every column as
    #      a directional return predictor (a strawman that should lose to v2).
    signal_B2 = 0.25 * (z_ret + z_vol + z_regime + z_drawdown)

    # B3 — Z-scored equal-weight WITH v2's sign-flip convention.
    #      Reproduces evaluate_meta_lift.build_equal_weight_signal exactly.
    signal_B3 = 0.25 * (z_ret - z_vol - z_regime - z_drawdown)

    # B4 — SPY buy-and-hold (always long). Sharpe of holding SPY long.
    signal_B4 = pd.Series(1.0, index=val_index)

    # --- Compute metrics -----------------------------------------------------
    rows = [
        _metrics("meta_v2",        meta_signal, ret_bwd),
        _metrics("B1_ret_only",    signal_B1,   ret_bwd),
        _metrics("B2_naive_eq",    signal_B2,   ret_bwd),
        _metrics("B3_v2_eq",       signal_B3,   ret_bwd),
        _metrics("B4_spy_buyhold", signal_B4,   ret_bwd),
    ]
    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("METRICS TABLE")
    print("=" * 70)
    print(df.to_string(index=False))

    # --- Lift table ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("LIFT TABLE  meta_v2 directional Sharpe vs each baseline")
    print("=" * 70)
    print(f"{'Baseline':<18}  {'b_dir':>9}  {'lift':>10}  {'>=5%':>6}  verdict")

    meta_dir   = float(rows[0]["dir_sharpe"])
    lift_rows  = []
    for r in rows[1:]:
        b = float(r["dir_sharpe"])
        if b == 0 or np.isnan(b):
            lift_pct = float("nan")
            passes   = False
        else:
            lift_pct = (meta_dir - b) / abs(b) * 100
            passes   = lift_pct >= 5.0
        verdict = "PASS" if passes else "FAIL"
        check   = "Y" if passes else "N"
        lift_rows.append((r["strategy"], lift_pct, passes))
        print(f"{r['strategy']:<18}  {b:>+9.4f}  {lift_pct:>+9.1f}%  {check:>6}  {verdict}")

    # --- Final verdict -------------------------------------------------------
    n_pass  = sum(1 for _, _, p in lift_rows if p)
    n_total = len(lift_rows)

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    if n_pass == n_total:
        print(f"PASS — meta_v2 beats ALL {n_total} baselines by >= 5% directional")
    elif n_pass > 0:
        passed = [s for s, _, p in lift_rows if p]
        failed = [s for s, _, p in lift_rows if not p]
        print(f"PARTIAL — meta_v2 beats {n_pass}/{n_total} baselines")
        print(f"  Beaten : {passed}")
        print(f"  Lost to: {failed}")
    else:
        print(f"FAIL — meta_v2 fails to beat ANY baseline ({n_pass}/{n_total})")

    # --- Interpretation guide ------------------------------------------------
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print(
        "  If meta_v2 beats B1, B2, B3, AND B4: the +28.8% is genuinely real,\n"
        "    not baseline-shopping. Move to Phase 4 with confidence.\n"
        "  If meta_v2 beats B3 (current baseline) and B4 but loses to B1 or B2:\n"
        "    the win is at least partially architectural baseline-shopping.\n"
        "    The unified-directional design helps the META learner more than\n"
        "    it helps a naive equal-weighter, but the ORIGINAL-style baseline\n"
        "    was actually a stronger benchmark. Salvageable but worth\n"
        "    documenting honestly.\n"
        "  If meta_v2 loses to B4 (SPY buy-and-hold): the meta-learner is not\n"
        "    adding value over the dumbest possible strategy on this val\n"
        "    period. Stop and rethink before Phase 4."
    )


if __name__ == "__main__":
    main()
