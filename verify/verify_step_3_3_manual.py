"""
verify/verify_step_3_3_manual.py

Teal manual verification gate for Step 3.3 per RFIE Build Prompts v3.
Runs five checks in sequence and prints a clear pass/fail banner for
each. Exits with code 0 only if ALL checks pass.

Signatures confirmed against actual source:
  FeedbackLoopEngine(ensemble, meta_learner, weight_tracker)
  run(features_df, returns_df, start_date, end_date) -> (pd.Series, pd.DataFrame)
  performance_log cols: prediction_date, feedback_date, signal, actual_return, pnl, weights
  validate_causality() -> bool
  WeightTracker(path=...)               [NOT audit_path]
  WeightTracker.load_weights() -> date-indexed df, cols: return_w vol_w regime_w drawdown_w
  evaluate_lift(weights_df, ensemble_preds_df, targets_df) -> dict
    weights_df cols must be: return, vol, regime, drawdown
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

PASS_TAG = "\033[92m[PASS]\033[0m"
FAIL_TAG = "\033[91m[FAIL]\033[0m"
INFO_TAG = "\033[94m[INFO]\033[0m"

results: dict = {}


def banner(n: int, title: str) -> None:
    print(f"\n{'=' * 70}\nCHECK {n}: {title}\n{'=' * 70}")


# ------------------------------------------------------------------
# Setup: load all artifacts
# ------------------------------------------------------------------
print(f"{INFO_TAG} Loading Phase 2/3 artifacts...")

ensemble     = BaseEnsemble()
meta_learner = MetaLearner()
meta_learner.load("models/meta/meta_learner.joblib")

price_df  = pd.read_parquet("data/processed/price_features_v1.parquet")
macro_df  = pd.read_parquet("data/processed/macro_features_v1.parquet")
joined_df = price_df.join(macro_df, how="inner")  # 111 cols — matches drawdown model
# Ordinal-encode vix_regime to match training convention: low→0, medium→1, high→2
if "vix_regime" in joined_df.columns:
    vix_map   = {"low": 0, "medium": 1, "high": 2}
    joined_df = joined_df.copy()
    joined_df["vix_regime"] = joined_df["vix_regime"].map(vix_map).astype("float64")
targets   = pd.read_parquet("data/processed/targets.parquet")

# Returns series: SPY_tgt_ret_1d is forward-stamped.
# run() applies iloc[t_pos-1] shift internally so PnL pairing is causal.
returns_series = targets["SPY_tgt_ret_1d"]

features_dict = {"price": price_df, "joined": joined_df}

# Val window from config
with open(PROJECT_ROOT / "config.yaml") as fh:
    _cfg = yaml.safe_load(fh)
_sp = _cfg["splits"]
_train_end = pd.Timestamp(_sp["train_end"])
val_end    = pd.Timestamp(_sp["val_end"])

# Resolve val_start to the first actual trading date after train_end
_ret_idx  = returns_series.dropna().index
val_start = _ret_idx[_ret_idx > _train_end][0]

print(f"{INFO_TAG} Val window: {val_start.date()} -> {val_end.date()}")
print(f"{INFO_TAG} Val length: {len(returns_series.loc[val_start:val_end])} days")


# ------------------------------------------------------------------
# CHECK 1: validate_causality() returns True
# ------------------------------------------------------------------
banner(1, "validate_causality() must return True on completed simulation")

wt1   = WeightTracker(path="logs/weight_audit_manual_1.csv")
eng1  = FeedbackLoopEngine(ensemble, meta_learner, wt1)
preds1, perf1 = eng1.run(features_dict, returns_series, val_start, val_end)

causality_ok = eng1.validate_causality()
print(f"validate_causality() returned: {causality_ok}")
print(f"Performance log entries: {len(perf1)}")
print(f"Prediction buffer entries: {len(preds1)}")

if causality_ok is True:
    print(f"{PASS_TAG} Check 1")
    results["check_1_causality"] = True
else:
    print(f"{FAIL_TAG} Check 1 — validate_causality() returned {causality_ok}")
    results["check_1_causality"] = False


# ------------------------------------------------------------------
# CHECK 2: Weight trajectory varies over time (not frozen at 0.25)
# ------------------------------------------------------------------
banner(2, "Weight trajectory must vary (not frozen at uniform 0.25)")

wt1_df      = wt1.load_weights()   # cols: return_w, vol_w, regime_w, drawdown_w
weight_cols = ["return_w", "vol_w", "regime_w", "drawdown_w"]

print(f"Weight audit rows: {len(wt1_df)}")
print(f"\nFirst 5 rows:\n{wt1_df.head()}")
print(f"\nLast 5 rows:\n{wt1_df.tail()}")
stds = wt1_df[weight_cols].std()
print(f"\nWeight std across time (per model):\n{stds}")

MIN_STD = 1e-4
if (stds > MIN_STD).all():
    print(f"{PASS_TAG} Check 2 — all weights vary (min std = {stds.min():.6f})")
    results["check_2_varying"] = True
else:
    frozen = stds[stds <= MIN_STD].index.tolist()
    print(f"{FAIL_TAG} Check 2 — frozen weights: {frozen} (std <= {MIN_STD})")
    print("       Meta-learner may not have exited warmup or models are None.")
    results["check_2_varying"] = False


# ------------------------------------------------------------------
# CHECK 3: Buffer semantics — pnl=None on day 1, first pnl on day 2
# ------------------------------------------------------------------
banner(3, "Day 1: pnl=None (no paired signal yet). Day 2: first valid pnl.")
# run() passes returns_df.iloc[t_pos-1] as actual_return for every val date
# (t_pos > 0 since val_start is well into the full series). On day 1,
# prediction_buffer has only one entry so pred_date=None, signal=None, pnl=None.
# On day 2 prediction_buffer has two entries: pred_date = val_start,
# signal = prediction_buffer[val_start], pnl = signal * actual_return.

print(f"Performance log shape: {perf1.shape}")
print(f"Columns: {list(perf1.columns)}")
print(f"\nFirst 5 entries:\n{perf1.head()}")

if len(perf1) < 2:
    print(f"{FAIL_TAG} Check 3 — fewer than 2 log entries ({len(perf1)})")
    results["check_3_buffer"] = False
else:
    day1 = perf1.iloc[0]
    day2 = perf1.iloc[1]

    day1_pnl_none    = pd.isna(day1.get("pnl", np.nan))
    day1_signal_none = pd.isna(day1.get("signal", np.nan))
    day2_pnl_ok      = not pd.isna(day2.get("pnl", np.nan))
    day2_signal_ok   = not pd.isna(day2.get("signal", np.nan))

    print(f"\nDay 1 — prediction_date: {day1.get('prediction_date')}, "
          f"signal: {day1.get('signal')}, pnl: {day1.get('pnl')}")
    day2_pnl_val = day2.get('pnl')
    day2_sig_val = day2.get('signal')
    _d2_sig_str = f"{day2_sig_val:.6f}" if day2_signal_ok else str(day2_sig_val)
    _d2_pnl_str = f"{day2_pnl_val:.6f}" if day2_pnl_ok else str(day2_pnl_val)
    print(f"Day 2 — prediction_date: {day2.get('prediction_date')}, "
          f"signal: {_d2_sig_str}, pnl: {_d2_pnl_str}")

    if day1_pnl_none and day1_signal_none and day2_pnl_ok and day2_signal_ok:
        print(f"{PASS_TAG} Check 3 — 1-step buffer semantics correct")
        results["check_3_buffer"] = True
    else:
        print(f"{FAIL_TAG} Check 3 — unexpected buffer semantics")
        print(f"  day1_pnl_none={day1_pnl_none}, day1_signal_none={day1_signal_none}, "
              f"day2_pnl_ok={day2_pnl_ok}, day2_signal_ok={day2_signal_ok}")
        results["check_3_buffer"] = False


# ------------------------------------------------------------------
# CHECK 4: Determinism — two runs with same inputs produce identical results
# ------------------------------------------------------------------
banner(4, "Two runs with same inputs must be identical (determinism)")

wt2   = WeightTracker(path="logs/weight_audit_manual_2.csv")
eng2  = FeedbackLoopEngine(ensemble, meta_learner, wt2)
preds2, perf2 = eng2.run(features_dict, returns_series, val_start, val_end)

try:
    pd.testing.assert_series_equal(preds1, preds2, check_exact=False, atol=1e-10, rtol=0)
    print(f"{PASS_TAG} Check 4 — predictions identical across runs (atol=1e-10)")
    results["check_4_deterministic"] = True
except AssertionError as exc:
    max_diff = (preds1 - preds2).abs().max()
    print(f"{FAIL_TAG} Check 4 — predictions differ (max |diff| = {max_diff:.2e})")
    print(f"       Details: {exc}")
    print("       Fix: propagate a fixed random seed to all stochastic components.")
    results["check_4_deterministic"] = False


# ------------------------------------------------------------------
# CHECK 5: Exit gate — meta Sharpe > equal-weight Sharpe on val
# ------------------------------------------------------------------
banner(5, "EXIT GATE: meta-weighted Sharpe > equal-weight Sharpe (val)")
# Uses evaluate_lift() which applies:
#   meta_signal[t]  = w_return*ret_pred[t] + w_regime*regime_dir[t]*|ret_pred[t]|
#   equal_signal[t] = 0.5*ret_pred[t] + 0.5*regime_dir[t]*|ret_pred[t]|
#   pnl[t]          = signal[t-1] * SPY_tgt_ret_1d[t]   (backward convention)
# weight_tracker stores return_w etc.; evaluate_lift() expects return, vol, ...

ens_preds = pd.read_parquet("data/processed/ensemble_predictions.parquet")

weights_for_eval = wt1_df.rename(columns={
    "return_w":   "return",
    "vol_w":      "vol",
    "regime_w":   "regime",
    "drawdown_w": "drawdown",
})

lift_metrics = meta_learner.evaluate_lift(
    weights_df=weights_for_eval,
    ensemble_preds_df=ens_preds,
    targets_df=targets,
)

meta_sharpe  = lift_metrics.get("meta_sharpe",  float("nan"))
equal_sharpe = lift_metrics.get("equal_sharpe", float("nan"))
lift_pct     = lift_metrics.get("lift_pct",     float("nan"))
n_days       = lift_metrics.get("n_days",       0)
eval_range   = lift_metrics.get("eval_date_range", ["?", "?"])

print(f"Meta-weighted Sharpe : {meta_sharpe:.4f}")
print(f"Equal-weight Sharpe  : {equal_sharpe:.4f}")
print(f"Lift                 : {lift_pct:+.2f}%")
print(f"n_days               : {n_days}")
print(f"Eval window          : {eval_range[0]} -> {eval_range[1]}")

if meta_sharpe > equal_sharpe:
    print(f"{PASS_TAG} Check 5 — meta-learner adds value on val set")
    results["check_5_exit_gate"] = True
else:
    print(f"{FAIL_TAG} Check 5 — meta-learner NOT beating equal-weight on val")
    print("       Per PDF: this is the Phase 3 exit gate. Investigate")
    print("       meta-feature quality or Ridge regularization before Phase 4.")
    results["check_5_exit_gate"] = False


# ------------------------------------------------------------------
# FINAL BANNER
# ------------------------------------------------------------------
print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
for k, v in results.items():
    tag = PASS_TAG if v else FAIL_TAG
    print(f"{tag} {k}")

all_passed = all(results.values())
print(f"\n{'=' * 70}")
if all_passed:
    print(f"{PASS_TAG} PHASE 3 EXIT GATE: ALL CHECKS PASSED")
    sys.exit(0)
else:
    n_fail = sum(1 for v in results.values() if not v)
    print(f"{FAIL_TAG} PHASE 3 EXIT GATE: {n_fail} CHECK(S) FAILED — DO NOT PROCEED")
    sys.exit(1)
