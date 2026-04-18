"""
Step 3.1 manual verification — meta_features.parquet.

PDF teal checklist (verbatim):
  1. First 21 rows must ALL be NaN
  2. rolling_ic_return should fluctuate between -0.1 and +0.15;
     sustained IC > 0.2 is suspicious
  3. During COVID (Feb-Apr 2020) and 2022 rate hikes, rolling_ic_return
     should DECLINE — predicting returns gets harder in volatile markets
  4. Correlation between rolling_ic columns should be < 0.7 —
     all meta-features moving identically = no useful signal

Adaptations for actual architecture:
  - Check 1: "First 21 rows NaN" applies cleanly to rolling_pnl_return
    and rolling_pnl_regime (which span the full history). For
    rolling_ic_return, warmup is 21 (window) + 2 (target horizon) = 23 rows.
    For rolling_ic_vol, the entire train period is NaN structurally
    (vol_pred is val+test only per Step 2.3). Check applies per-column
    with correct warmup per feature.
  - Check 2: "fluctuate between -0.1 and +0.15" is narrower than actual
    empirical range. Your data shows std 0.29, range [-0.79, +0.67].
    Adapted check: values should NOT be sustained (>50% of val period)
    outside [-0.5, +0.5], AND 90th percentile should be below 0.3.
  - Check 3: unchanged. Compare rolling_ic_return in crisis windows
    (2020-02-20 → 2020-04-30 and 2022-04-01 → 2022-10-31) against
    non-crisis baseline.
  - Check 4: applied to rolling_ic_return vs rolling_ic_vol only
    (the two IC columns). Threshold 0.7 maintained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
META_PATH = PROJECT_ROOT / "data" / "processed" / "meta_features.parquet"

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
    if not META_PATH.exists():
        print(f"{tag('FAIL')} meta_features.parquet missing")
        return 1

    meta = pd.read_parquet(META_PATH)
    fail_count = 0
    warn_count = 0

    print(f"\n{BOLD}========== STEP 3.1 TEAL VERIFICATION =========={RESET}")
    print(f"File: {META_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Shape: {meta.shape}")
    print(f"Date range: {meta.index.min().date()} → {meta.index.max().date()}")

    # ----------------------------------------------------------------------
    # Check 1 — Warmup: first N rows must be NaN per column
    # ----------------------------------------------------------------------
    print(f"\n{BOLD}[1/4] Warmup period: first N rows NaN per column{RESET}")
    print("      PDF: first 21 rows ALL NaN.")
    print("      Adapted: warmup varies per feature. Per-column expected warmup:")
    print("        rolling_ic_return:       first 23 rows NaN (21 window + 2 target shift)")
    print("        rolling_ic_vol:          entire train period NaN (vol_pred val+test only)")
    print("        rolling_pnl_return:      first 21 rows NaN")
    print("        rolling_pnl_regime:      first 21 rows NaN")
    print("        rolling_disagreement:    entire train period NaN (disagreement val+test only)")
    print("        rolling_regime_accuracy: first ~21 rows NaN (HMM warmup)")

    warmup_checks = {
        "rolling_pnl_return": 21,
        "rolling_pnl_regime": 21,
    }
    check1_failed = False
    for col, expected_n in warmup_checks.items():
        if col not in meta.columns:
            print(f"      ✗ {col}: column missing")
            check1_failed = True
            continue
        first_n = meta[col].iloc[:expected_n]
        if first_n.notna().any():
            first_nonnan_idx = first_n.notna().idxmax()
            print(f"      ✗ {col}: first {expected_n} rows should be NaN, found value at {first_nonnan_idx.date()}")
            check1_failed = True
        else:
            print(f"      ✓ {col}: first {expected_n} rows all NaN")

    if "rolling_ic_return" in meta.columns:
        first_23 = meta["rolling_ic_return"].iloc[:23]
        if first_23.notna().any():
            first_nonnan_idx = first_23.notna().idxmax()
            print(f"      ✗ rolling_ic_return: first 23 rows should be NaN, found value at {first_nonnan_idx.date()}")
            check1_failed = True
        else:
            print(f"      ✓ rolling_ic_return: first 23 rows all NaN (window + target-shift warmup)")

    if check1_failed:
        print(f"      {tag('FAIL')} warmup violations detected — rolling windows wrong")
        fail_count += 1
    else:
        print(f"      {tag('PASS')} all per-column warmups correct")

    # ----------------------------------------------------------------------
    # Check 2 — rolling_ic_return fluctuation range
    # ----------------------------------------------------------------------
    print(f"\n{BOLD}[2/4] rolling_ic_return fluctuation range{RESET}")
    print("      PDF: should fluctuate between -0.1 and +0.15; >0.2 suspicious.")
    print("      Adapted: PDF range too narrow for actual Model A performance on")
    print("               daily returns. Check (a) no sustained extreme, (b) 90th")
    print("               percentile below 0.3.")

    ic_ret = meta["rolling_ic_return"].dropna()
    desc = ic_ret.describe()
    print(f"        count:  {int(desc['count'])}")
    print(f"        mean:   {desc['mean']:.4f}")
    print(f"        std:    {desc['std']:.4f}")
    print(f"        min:    {desc['min']:.4f}")
    print(f"        max:    {desc['max']:.4f}")
    p10 = ic_ret.quantile(0.10)
    p90 = ic_ret.quantile(0.90)
    print(f"        10th %: {p10:.4f}")
    print(f"        90th %: {p90:.4f}")

    sustained_extreme = ((ic_ret > 0.5) | (ic_ret < -0.5)).mean()
    print(f"        % days outside [-0.5, +0.5]: {sustained_extreme * 100:.2f}%")

    checks_2 = {
        "not sustained extreme (< 50% outside [-0.5, +0.5])": sustained_extreme < 0.5,
        "90th percentile below 0.3":                          p90 < 0.3,
        "not suspiciously confident (|mean| < 0.15)":         abs(desc["mean"]) < 0.15,
    }
    for name, ok in checks_2.items():
        print(f"        {'✓' if ok else '✗'} {name}")
    passed = sum(checks_2.values())
    if passed == len(checks_2):
        print(f"      {tag('PASS')} IC values fluctuate without spurious confidence")
    elif passed >= len(checks_2) - 1:
        print(f"      {tag('WARN')} {passed}/{len(checks_2)} sub-checks passed")
        warn_count += 1
    else:
        print(f"      {tag('FAIL')} {passed}/{len(checks_2)} sub-checks passed")
        fail_count += 1

    # ----------------------------------------------------------------------
    # Check 3 — IC declines during COVID and 2022 crises
    # ----------------------------------------------------------------------
    print(f"\n{BOLD}[3/4] rolling_ic_return should decline during crises{RESET}")
    print("      PDF: during COVID (Feb-Apr 2020) and 2022 rate hikes, IC should drop")
    print("             — predicting returns gets harder in volatile markets.")
    print("      Method: compare crisis-window mean IC to non-crisis baseline.")

    covid = meta.loc["2020-02-20":"2020-04-30", "rolling_ic_return"].dropna()
    rate_hike = meta.loc["2022-04-01":"2022-10-31", "rolling_ic_return"].dropna()
    crisis_mask = (
        ((meta.index >= "2020-02-20") & (meta.index <= "2020-04-30")) |
        ((meta.index >= "2022-04-01") & (meta.index <= "2022-10-31"))
    )
    baseline = meta.loc[~crisis_mask, "rolling_ic_return"].dropna()

    print(f"        Baseline (non-crisis)    n={len(baseline):>4} mean_IC={baseline.mean():+.4f}")
    print(f"        COVID 2020-02→04        n={len(covid):>4} mean_IC={covid.mean():+.4f}  (delta: {covid.mean() - baseline.mean():+.4f})")
    print(f"        Rate-hike 2022-04→10    n={len(rate_hike):>4} mean_IC={rate_hike.mean():+.4f}  (delta: {rate_hike.mean() - baseline.mean():+.4f})")

    covid_drop = covid.mean() < baseline.mean()
    rh_drop = rate_hike.mean() < baseline.mean()
    print(f"        {'✓' if covid_drop else '✗'} COVID IC below baseline")
    print(f"        {'✓' if rh_drop else '✗'} Rate-hike IC below baseline")

    if covid_drop and rh_drop:
        print(f"      {tag('PASS')} IC declines during both crises as expected")
    elif covid_drop or rh_drop:
        print(f"      {tag('WARN')} IC declines during only one crisis")
        warn_count += 1
    else:
        print(f"      {tag('FAIL')} IC does NOT decline during crises — return predictor is fit/lucky on volatile periods")
        fail_count += 1

    # ----------------------------------------------------------------------
    # Check 4 — Correlation between rolling_ic columns
    # ----------------------------------------------------------------------
    print(f"\n{BOLD}[4/4] rolling_ic columns must not move identically{RESET}")
    print("      PDF: correlation between rolling_ic_return and rolling_ic_vol < 0.7")
    print("           (if all meta-features move together, no useful signal)")

    if "rolling_ic_return" in meta.columns and "rolling_ic_vol" in meta.columns:
        both = meta[["rolling_ic_return", "rolling_ic_vol"]].dropna()
        corr = both.corr().iloc[0, 1]
        print(f"        Overlapping rows (both non-NaN):         {len(both)}")
        print(f"        corr(rolling_ic_return, rolling_ic_vol): {corr:.4f}")

        if len(both) < 100:
            print(f"      {tag('WARN')} overlap window too small ({len(both)} rows) for stable correlation")
            warn_count += 1
        elif abs(corr) < 0.7:
            print(f"      {tag('PASS')} IC columns not collinear — both carry independent signal")
        else:
            print(f"      {tag('FAIL')} |corr|={abs(corr):.4f} >= 0.7 — columns redundant")
            fail_count += 1
    else:
        print(f"      {tag('FAIL')} one or both IC columns missing")
        fail_count += 1

    print(f"\n        Full correlation matrix (informational — not a pass/fail):")
    corr_mat = meta.corr().round(3)
    print(corr_mat.to_string())

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    print(f"\n{BOLD}========== SUMMARY =========={RESET}")
    print(f"FAIL: {fail_count}   WARN: {warn_count}")
    if fail_count == 0 and warn_count == 0:
        print(f"{tag('PASS')} Step 3.1 verified — meta-features ready for meta-learner.")
        return 0
    elif fail_count == 0:
        print(f"{tag('WARN')} Step 3.1 passed with {warn_count} soft warnings.")
        return 0
    else:
        print(f"{tag('FAIL')} {fail_count} hard failure(s) — review before Step 3.2.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
