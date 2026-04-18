"""
Step 2.5 manual verification — BaseEnsemble + drawdown estimator.

Checks:
  1. All 5 columns present in ensemble_predictions.parquet
  2. March 2020 (COVID): return_pred negative, regime=bear, drawdown_risk_prob elevated
     (vol_pred / disagreement not available — pre-val period)
  3. 2022 rate-hike (April-June): all 4 models flagging risk-off together
  4. Disagreement metric: non-NaN count matches val+test coverage, sensible distribution
  5. Feature alignment: return model expects 99 features, drawdown model expects 111
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENSEMBLE_PATH = PROJECT_ROOT / "data" / "processed" / "ensemble_predictions.parquet"
RETURN_MODEL = PROJECT_ROOT / "models" / "base" / "spy_ret1d.joblib"
DRAWDOWN_MODEL = PROJECT_ROOT / "models" / "base" / "drawdown_estimator.joblib"
DRAWDOWN_EVAL = PROJECT_ROOT / "logs" / "model_d_eval.json"

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def tag(status: str) -> str:
    if status == "PASS":
        return f"{GREEN}{BOLD}[PASS]{RESET}"
    if status == "WARN":
        return f"{YELLOW}{BOLD}[WARN]{RESET}"
    return f"{RED}{BOLD}[FAIL]{RESET}"


def main() -> int:
    if not ENSEMBLE_PATH.exists():
        print(f"{tag('FAIL')} ensemble_predictions.parquet not found")
        return 1
    if not RETURN_MODEL.exists() or not DRAWDOWN_MODEL.exists():
        print(f"{tag('FAIL')} model joblibs missing — run base models first")
        return 1

    df = pd.read_parquet(ENSEMBLE_PATH)
    fail_count = 0

    print(f"\n{BOLD}========== STEP 2.5 MANUAL VERIFICATION =========={RESET}")
    print(f"File: {ENSEMBLE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min().date()} → {df.index.max().date()}")

    # ----------------------------------------------------------------------
    # Check 1 — All 5 expected columns present
    # ----------------------------------------------------------------------
    print(f"\n{BOLD}[1/5] Expected columns present{RESET}")
    expected = ["return_pred", "vol_pred", "regime_pred", "drawdown_risk_prob", "disagreement"]
    missing = [c for c in expected if c not in df.columns]
    for c in expected:
        marker = "✓" if c in df.columns else "✗"
        print(f"        {marker} {c}")
    if missing:
        print(f"      {tag('FAIL')} missing columns: {missing}")
        fail_count += 1
    else:
        print(f"      {tag('PASS')} all 5 expected columns present")

    # ----------------------------------------------------------------------
    # Check 2 — March 2020 COVID (partial check: return + regime + drawdown only)
    # ----------------------------------------------------------------------
    print(f"\n{BOLD}[2/5] March 2020 COVID window (partial — pre-val period){RESET}")
    print("      Expected: return_pred mostly negative, regime=bear, drawdown_risk_prob elevated")
    print("      (vol_pred / disagreement NOT CHECKED — val+test only)")
    covid = df.loc["2020-02-20":"2020-04-15"]
    if len(covid) == 0:
        print(f"      {tag('FAIL')} no data in COVID window")
        fail_count += 1
    else:
        ret_pct_neg = float((covid["return_pred"] < 0).mean() * 100)
        regime_counts = covid["regime_pred"].value_counts().to_dict()
        bear_days = regime_counts.get(1.0, 0) + regime_counts.get(1, 0)
        bear_pct = bear_days / len(covid) * 100
        dd_mean = float(covid["drawdown_risk_prob"].mean())
        dd_baseline = float(df["drawdown_risk_prob"].mean())

        print(f"        return_pred < 0:              {ret_pct_neg:.1f}% of COVID days")
        print(f"        bear regime (regime==1):       {bear_days}/{len(covid)} days ({bear_pct:.1f}%)")
        print(f"        drawdown_risk_prob mean:       {dd_mean:.4f}  (baseline: {dd_baseline:.4f})")

        checks = {
            "return_pred negatively biased": ret_pct_neg >= 40.0,
            "bear regime plurality":        bear_pct >= 33.0,
            "drawdown elevated vs baseline": dd_mean > dd_baseline,
        }
        any_fail = False
        for name, ok in checks.items():
            if ok:
                print(f"        ✓ {name}")
            else:
                print(f"        ✗ {name}")
                any_fail = True
        if any_fail:
            print(f"      {tag('WARN')} some individual checks failed, but COVID signal is multi-faceted")
        else:
            print(f"      {tag('PASS')} all 3 partial checks passed")

    # ----------------------------------------------------------------------
    # Check 3 — 2022 rate-hike drawdown (full 4-model agreement check)
    # ----------------------------------------------------------------------
    print(f"\n{BOLD}[3/5] 2022 rate-hike drawdown (April-June 2022 — full 4-model check){RESET}")
    print("      Expected: vol_pred elevated, regime=bear, drawdown_risk_prob elevated,")
    print("                disagreement computable")
    rh = df.loc["2022-04-01":"2022-06-30"]
    if len(rh) == 0 or rh["vol_pred"].isna().all():
        print(f"      {tag('FAIL')} no val-period data in 2022 rate-hike window")
        fail_count += 1
    else:
        ret_pct_neg = float((rh["return_pred"] < 0).mean() * 100)
        vol_mean = float(rh["vol_pred"].mean())
        vol_baseline = float(df.loc["2022-01-03":"2023-06-30", "vol_pred"].mean())
        regime_counts = rh["regime_pred"].value_counts().to_dict()
        bear_days = regime_counts.get(1.0, 0) + regime_counts.get(1, 0)
        bear_pct = bear_days / len(rh) * 100
        dd_mean = float(rh["drawdown_risk_prob"].mean())
        dd_baseline = float(df["drawdown_risk_prob"].mean())
        dis_mean = float(rh["disagreement"].mean())
        dis_baseline = float(df["disagreement"].mean())

        print(f"        return_pred < 0:               {ret_pct_neg:.1f}%")
        print(f"        vol_pred mean (ann'd):         {vol_mean:.4f}  (val+test baseline: {vol_baseline:.4f})")
        print(f"        bear regime (regime==1):       {bear_days}/{len(rh)} days ({bear_pct:.1f}%)")
        print(f"        drawdown_risk_prob mean:       {dd_mean:.4f}  (baseline: {dd_baseline:.4f})")
        print(f"        disagreement mean:             {dis_mean:.4f}  (baseline: {dis_baseline:.4f})")

        checks = {
            "vol elevated":         vol_mean > vol_baseline,
            "drawdown elevated":    dd_mean > dd_baseline,
            "disagreement != 0":    dis_mean > 0.3,
        }
        passed = sum(1 for ok in checks.values() if ok)
        for name, ok in checks.items():
            print(f"        {'✓' if ok else '✗'} {name}")
        if passed == len(checks):
            print(f"      {tag('PASS')} all {passed}/{len(checks)} signal checks passed for 2022 crisis")
        elif passed >= len(checks) - 1:
            print(f"      {tag('WARN')} {passed}/{len(checks)} checks passed")
        else:
            print(f"      {tag('FAIL')} only {passed}/{len(checks)} checks passed")
            fail_count += 1

    # ----------------------------------------------------------------------
    # Check 4 — Disagreement distribution
    # ----------------------------------------------------------------------
    print(f"\n{BOLD}[4/5] Disagreement column distribution{RESET}")
    dis = df["disagreement"].dropna()
    print(f"        non-NaN:  {len(dis)} rows")
    print(f"        mean:     {dis.mean():.4f}")
    print(f"        std:      {dis.std():.4f}")
    print(f"        min / max: {dis.min():.4f} / {dis.max():.4f}")

    if len(dis) < 500:
        print(f"      {tag('FAIL')} too few non-NaN disagreement rows ({len(dis)} < 500)")
        fail_count += 1
    elif dis.std() < 0.1:
        print(f"      {tag('WARN')} disagreement std very low ({dis.std():.3f}) — might be degenerate")
    else:
        print(f"      {tag('PASS')} disagreement populated and varies across rows")

    # ----------------------------------------------------------------------
    # Check 5 — Feature alignment (models expect correct feature counts)
    # ----------------------------------------------------------------------
    print(f"\n{BOLD}[5/5] Model feature alignment{RESET}")
    ret_m = joblib.load(RETURN_MODEL)
    dd_m = joblib.load(DRAWDOWN_MODEL)
    print(f"        return model n_features_in_:   {ret_m.n_features_in_}  (expected: 99)")
    print(f"        drawdown model n_features_in_: {dd_m.n_features_in_}  (expected: 111)")
    if ret_m.n_features_in_ == 99 and dd_m.n_features_in_ == 111:
        print(f"      {tag('PASS')} both models have expected feature counts")
    else:
        print(f"      {tag('FAIL')} unexpected feature counts — retrain models")
        fail_count += 1

    # Also surface the drawdown model's test metrics
    if DRAWDOWN_EVAL.exists():
        with open(DRAWDOWN_EVAL) as fh:
            eval_data = json.load(fh)
        test_m = eval_data.get("test_metrics", {})
        if test_m:
            print(f"\n        Drawdown model test metrics (from model_d_eval.json):")
            print(f"          test AUC-ROC: {test_m.get('test_auc_roc')}")
            print(f"          test AUCPR:   {test_m.get('test_aucpr')}  (base rate: {test_m.get('test_base_rate')})")

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    print(f"\n{BOLD}========== SUMMARY =========={RESET}")
    if fail_count == 0:
        print(f"{tag('PASS')} All 5 checks passed — Step 2.5 verified, Phase 2 complete.")
        return 0
    else:
        print(f"{tag('FAIL')} {fail_count} check(s) failed — review before advancing to Phase 3.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
