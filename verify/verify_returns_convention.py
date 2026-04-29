"""
verify/verify_returns_convention.py
Determines the exact convention of the returns series consumed by
FeedbackLoopEngine.run() so the PnL pairing in feedback.py can be confirmed.
"""

import pandas as pd
import numpy as np

closes = pd.read_parquet("data/raw/all_close_prices.parquet")
spy = closes["SPY"]

targets = pd.read_parquet("data/processed/targets.parquet")
print("Columns in targets.parquet containing 'ret' (first 10):")
print([c for c in targets.columns if "ret" in c.lower()][:10])
print()

# Test against SPY_tgt_ret_1d — the series feedback.py consumes.
# If feedback.py actually consumes a different series, log that and retest.
ret_series = targets["SPY_tgt_ret_1d"].dropna()

ret_backward = np.log(spy / spy.shift(1))
ret_forward  = np.log(spy.shift(-1) / spy)

common = ret_series.index.intersection(ret_backward.index)
diff_b = (ret_series.loc[common] - ret_backward.loc[common]).abs().max()
diff_f = (ret_series.loc[common] - ret_forward.loc[common]).abs().max()

print(f"max|ret_series - backward_return| = {diff_b:.2e}")
print(f"max|ret_series - forward_return|  = {diff_f:.2e}")

if diff_b < 1e-10:
    print(">>> CONVENTION: backward — returns_df.loc[t] = log(close[t]/close[t-1])")
    print(">>> signal[t] (made with info through t-1) earns returns_df.loc[t]")
elif diff_f < 1e-10:
    print(">>> CONVENTION: forward — returns_df.loc[t] = log(close[t+1]/close[t])")
    print(">>> signal[t] (made with info through t-1) earns returns_df.loc[t-1]")
else:
    print(">>> UNKNOWN — investigate manually")
    print(f"    First 5 values of ret_series:\n{ret_series.head()}")
    print(f"    First 5 values of backward :\n{ret_backward.dropna().head()}")
    print(f"    First 5 values of forward  :\n{ret_forward.dropna().head()}")

# ------------------------------------------------------------------
# ADDITIONAL STEP: inspect what returns_df feedback.py actually uses
# ------------------------------------------------------------------
print()
print("=" * 60)
print("ADDITIONAL: What series does FeedbackLoopEngine.run() consume?")
print("=" * 60)

with open("engine/feedback.py") as f:
    src = f.read()

# Extract the run() method body
run_start = src.index("def run(")
run_end   = src.index("\n    # --", run_start)
run_body  = src[run_start:run_end]
print("\nRelevant lines from run():")
for line in run_body.splitlines():
    if "return" in line.lower() or "ret" in line.lower():
        print(f"  {line}")

print()
print("Conclusion from code inspection:")
print("  run() receives `returns_df` as a parameter — the caller supplies it.")
print("  The __main__ in meta_learner.py computes:")
print("    spy_returns = np.log(close_df['SPY']).diff().rename('SPY_logret')")
print("  That is log(close[t]/close[t-1]) — BACKWARD convention.")
print()

# Confirm by testing that series too
from pathlib import Path
import yaml

cfg_path = Path("config.yaml")
if cfg_path.exists():
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    raw_dir    = Path(cfg["paths"]["raw"])
    close_path = raw_dir / "all_close_prices.parquet"
    if close_path.exists():
        close_df   = pd.read_parquet(close_path)
        spy_logret = np.log(close_df["SPY"]).diff().rename("SPY_logret")
        common2      = ret_series.index.intersection(spy_logret.dropna().index)
        diff_logret_b = (ret_series.loc[common2] - spy_logret.loc[common2]).abs().max()
        diff_logret_f = (ret_series.loc[common2] - spy_logret.shift(-1).loc[common2]).abs().max()
        print("Testing spy_logret = np.log(SPY).diff()  (the meta_learner __main__ series):")
        print(f"  max|SPY_tgt_ret_1d - spy_logret(backward)| = {diff_logret_b:.2e}")
        print(f"  max|SPY_tgt_ret_1d - spy_logret(forward) | = {diff_logret_f:.2e}")
        if diff_logret_b < 1e-10:
            print("  >>> MATCH: spy_logret is backward — same convention as SPY_tgt_ret_1d")
        elif diff_logret_f < 1e-10:
            print("  >>> MATCH (shifted): spy_logret is forward relative to SPY_tgt_ret_1d")
        else:
            print("  >>> NO MATCH — SPY_tgt_ret_1d and spy_logret are different series")
            print(f"  Sample SPY_tgt_ret_1d:\n{ret_series.head()}")
            print(f"  Sample spy_logret:\n{spy_logret.dropna().head()}")

print()
print("=" * 60)
print("PnL PAIRING ASSESSMENT for FeedbackLoopEngine")
print("=" * 60)
print("""
In feedback.py _update_rolling_performance():
  pnl = signal[pred_date] * actual_return[feedback_date]

  pred_date     = t-1  (yesterday's prediction)
  feedback_date = t    (today — when the return is received)

If actual_return[t] = log(close[t]/close[t-1])  (BACKWARD):
  => pnl = signal[t-1] * realized_return[t-1 → t]
  => signal made at t-1 (using info through t-2) earns the return
     from t-1 to t.  CORRECT — standard backtest convention.

If actual_return[t] = log(close[t+1]/close[t])  (FORWARD):
  => pnl = signal[t-1] * forward_return_from[t]
  => off by one — signal at t-1 is paired with return[t → t+1],
     which is one bar too far ahead.  LOOKAHEAD BUG.
""")
