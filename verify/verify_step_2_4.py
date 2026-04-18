"""
Step 2.4 manual verification — HMM regime classifier.

Runs the four teal checks from the situation report:
  1. Regime distribution: each regime >= 10% of days (warn at <10%, fail at <5%)
  2. COVID check: Feb 20 – Mar 23, 2020 should be mostly regime 1 (bear)
  3. Persistence: avg consecutive days per regime > 30
  4. Transition matrix: all diagonals > 0.85 (fail at <0.7)

Runs checks on regime_filtered (causal column, safe for features).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGIMES_PATH = PROJECT_ROOT / "data" / "processed" / "regimes.parquet"
EVAL_PATH = PROJECT_ROOT / "logs" / "regime_classifier_eval.json"

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
    if not REGIMES_PATH.exists():
        print(f"{tag('FAIL')} regimes.parquet not found at {REGIMES_PATH}")
        return 1
    if not EVAL_PATH.exists():
        print(f"{tag('FAIL')} regime_classifier_eval.json not found at {EVAL_PATH}")
        return 1

    df = pd.read_parquet(REGIMES_PATH)
    with open(EVAL_PATH) as fh:
        eval_data = json.load(fh)

    if "regime_filtered" not in df.columns:
        print(f"{tag('FAIL')} regime_filtered column missing from regimes.parquet")
        return 1

    regime = df["regime_filtered"]
    total = len(regime)
    label_names = {0: "bull", 1: "bear", 2: "sideways"}
    fail_count = 0

    print(f"\n{BOLD}========== STEP 2.4 MANUAL VERIFICATION =========={RESET}")
    print(f"File: {REGIMES_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Column under test: regime_filtered")
    print(f"Total days: {total}")
    print(f"Date range: {regime.index.min().date()} → {regime.index.max().date()}")

    # ----------------------------------------------------------------------
    # Check 1 — Regime distribution (each regime >= 10%; <5% = fail)
    # ----------------------------------------------------------------------
    print(f"\n{BOLD}[1/4] Regime distribution{RESET}")
    print("      Expect: each regime >= 10% of days")
    print("      FAIL if any regime < 5% (HMM collapsed to degenerate solution)")
    counts = regime.value_counts().sort_index()
    pcts = (counts / total * 100).round(2)
    worst_pct = float(pcts.min())
    for lbl in sorted(counts.index):
        name = label_names.get(int(lbl), "?")
        print(f"        regime {int(lbl)} ({name:>8}): {int(counts[lbl]):>5} days  ({pcts[lbl]:>5.2f}%)")
    if worst_pct < 5.0:
        print(f"      {tag('FAIL')} smallest regime is {worst_pct:.2f}% (< 5% threshold)")
        fail_count += 1
    elif worst_pct < 10.0:
        print(f"      {tag('WARN')} smallest regime is {worst_pct:.2f}% (below 10% target but above 5% floor)")
    else:
        print(f"      {tag('PASS')} smallest regime is {worst_pct:.2f}% (>= 10%)")

    # ----------------------------------------------------------------------
    # Check 2 — COVID window should be mostly bear (regime 1)
    # ----------------------------------------------------------------------
    print(f"\n{BOLD}[2/4] COVID check (2020-02-20 → 2020-03-23){RESET}")
    print("      Expect: regime 1 (bear) should be the plurality")
    print("      FAIL if bear is not the most common label in window")
    covid = regime.loc["2020-02-20":"2020-03-23"]
    if len(covid) == 0:
        print(f"      {tag('FAIL')} no data in COVID window — check date alignment")
        fail_count += 1
    else:
        covid_counts = covid.value_counts().sort_index()
        covid_pcts = (covid_counts / len(covid) * 100).round(2)
        for lbl in sorted(covid_counts.index):
            name = label_names.get(int(lbl), "?")
            print(f"        regime {int(lbl)} ({name:>8}): {int(covid_counts[lbl]):>3} days  ({covid_pcts[lbl]:>5.2f}%)")
        dominant = int(covid_counts.idxmax())
        bear_pct = float(covid_pcts.get(1, 0.0))
        if dominant == 1 and bear_pct >= 50.0:
            print(f"      {tag('PASS')} bear is dominant at {bear_pct:.2f}% of COVID window")
        elif dominant == 1:
            print(f"      {tag('WARN')} bear is dominant but only at {bear_pct:.2f}% (plurality, not majority)")
        else:
            dom_name = label_names.get(dominant, "?")
            print(f"      {tag('FAIL')} dominant regime is {dominant} ({dom_name}), not bear (1)")
            print(f"              → label_regimes_by_vix mapping may be broken, or HMM didn't learn COVID as bear")
            fail_count += 1

    # ----------------------------------------------------------------------
    # Check 3 — Persistence (avg consecutive days per regime > 30)
    # ----------------------------------------------------------------------
    print(f"\n{BOLD}[3/4] Persistence — avg consecutive days per regime{RESET}")
    print("      Expect: avg run length > 30 days for each regime")
    print("      FAIL if any regime has avg < 5 days (regimes flipping uselessly)")
    avg_days = eval_data["filtered"]["avg_days_per_regime"]
    avg_days = {int(k): float(v) for k, v in avg_days.items()}
    for lbl in sorted(avg_days):
        name = label_names.get(lbl, "?")
        print(f"        regime {lbl} ({name:>8}): avg {avg_days[lbl]:>6.2f} consecutive days")
    worst_run = min(avg_days.values())
    if worst_run < 5.0:
        print(f"      {tag('FAIL')} shortest avg run = {worst_run:.2f} days (< 5)")
        fail_count += 1
    elif worst_run < 30.0:
        print(f"      {tag('WARN')} shortest avg run = {worst_run:.2f} days (below 30-day target but above 5-day floor)")
    else:
        print(f"      {tag('PASS')} shortest avg run = {worst_run:.2f} days (>= 30)")

    # ----------------------------------------------------------------------
    # Check 4 — Transition matrix diagonals > 0.85
    # ----------------------------------------------------------------------
    print(f"\n{BOLD}[4/4] Transition matrix diagonal persistence{RESET}")
    print("      Expect: all diagonal values > 0.85")
    print("      FAIL if any diagonal < 0.70 (model too noisy)")
    diag = eval_data["filtered"]["diag_persistence"]
    diag = {int(k): float(v) for k, v in diag.items()}
    for lbl in sorted(diag):
        name = label_names.get(lbl, "?")
        print(f"        regime {lbl} ({name:>8}): P(stay) = {diag[lbl]:.4f}")
    worst_diag = min(diag.values())
    if worst_diag < 0.70:
        print(f"      {tag('FAIL')} lowest diagonal = {worst_diag:.4f} (< 0.70)")
        fail_count += 1
    elif worst_diag < 0.85:
        print(f"      {tag('WARN')} lowest diagonal = {worst_diag:.4f} (below 0.85 target but above 0.70 floor)")
    else:
        print(f"      {tag('PASS')} lowest diagonal = {worst_diag:.4f} (>= 0.85)")

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    print(f"\n{BOLD}========== SUMMARY =========={RESET}")
    if fail_count == 0:
        print(f"{tag('PASS')} All 4 checks passed — Step 2.4 verified, ready to advance.")
        return 0
    else:
        print(f"{tag('FAIL')} {fail_count} check(s) failed — review before advancing.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
