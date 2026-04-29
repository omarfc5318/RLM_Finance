"""
verify/diagnose_meta_learner.py

Four-hypothesis diagnostic for why the meta-learner underperforms
equal-weight on val. Read-only. Does not modify any production file.

H1: Signal-scale mismatch in weighted_sum = sum(w_i * pred_i)
H2: Meta-features are constant/weak on val
H3: RidgeCV alpha is too high (over-regularized to max of grid)
H4: Softmax / proxy target is biased toward drawdown

Signature notes (confirmed from source):
  ml.models        -- List[RidgeCV], one per model_name (not ml.model)
  ml.model_names   -- ["return", "vol", "regime", "drawdown"]
  ml.feature_cols  -- 5 features (rolling_regime_accuracy was dropped)
  ml.alphas        -- (0.1, 1.0, 10.0, 100.0)
  predict_weights  -- column_stack([m.predict(X) for m in ml.models])
                      -> softmax -> convex-floor
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import softmax
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.meta.meta_learner import MetaLearner

PASS_TAG = "\033[92m[PASS]\033[0m"
FAIL_TAG = "\033[91m[FAIL]\033[0m"
WARN_TAG = "\033[93m[WARN]\033[0m"


def banner(n: int, h: str, title: str) -> None:
    print(f"\n{'=' * 70}\nH{n}: {h}\n{title}\n{'=' * 70}")


# ------------------------------------------------------------------
# Load once, reuse across all 4 hypotheses
# ------------------------------------------------------------------
print("Loading artifacts for meta-learner diagnostic...")

ensemble_preds = pd.read_parquet("data/processed/ensemble_predictions.parquet")
meta_features  = pd.read_parquet("data/processed/meta_features.parquet")
targets        = pd.read_parquet("data/processed/targets.parquet")
returns_series = targets["SPY_tgt_ret_1d"]

ml = MetaLearner()
ml.load("models/meta/meta_learner.joblib")

# Val window from config (matches evaluate_meta_lift convention)
with open(PROJECT_ROOT / "config.yaml") as fh:
    cfg = yaml.safe_load(fh)
sp        = cfg["splits"]
train_end = pd.Timestamp(sp["train_end"])
val_end   = pd.Timestamp(sp["val_end"])
ret_idx   = returns_series.dropna().index
val_start = ret_idx[ret_idx > train_end][0]

val_idx = ensemble_preds.loc[val_start:val_end].index

# Meta-features on val: use only the trained feature_cols, drop NaN rows
mf_val_raw = meta_features.loc[val_start:val_end, ml.feature_cols]
meta_idx   = mf_val_raw.dropna(how="any").index

print(f"Val window  : {val_start.date()} -> {val_end.date()}")
print(f"Val days    : {len(val_idx)} prediction dates")
print(f"Meta rows   : {len(meta_idx)} non-NaN meta rows (of {len(mf_val_raw)} total)")
print(f"feature_cols: {ml.feature_cols}")
print(f"model_names : {ml.model_names}")
print(f"weight_floor: {ml.weight_floor}")

results: dict = {}

PRED_COLS = ["return_pred", "vol_pred", "regime_pred", "drawdown_risk_prob"]


# ==================================================================
# H1: SIGNAL-SCALE MISMATCH
# ==================================================================
banner(1, "Signal-scale mismatch",
       "Measure scale of each of the 4 predictions. If any column is "
       "100x+ larger than another, the weighted sum is numerically "
       "dominated regardless of w_i.")

preds_val = ensemble_preds.loc[val_idx, PRED_COLS]
stats = pd.DataFrame({
    "mean":     preds_val.mean(),
    "std":      preds_val.std(),
    "abs_mean": preds_val.abs().mean(),
    "abs_max":  preds_val.abs().max(),
})
print("Per-model prediction statistics on val:")
print(stats.round(6))

contributions = (0.25 * preds_val.abs()).mean()
total_contrib  = contributions.sum()
shares = (contributions / total_contrib).round(4)
print(f"\nFractional contribution to |uniform-weighted sum| per model:")
print(shares)

max_share = float(shares.max())
max_col   = shares.idxmax()

if max_share > 0.50:
    finding = (f"{max_col} contributes {max_share:.1%} of signal magnitude "
               f"under uniform weights — scale-dominated, not weight-dominated.")
    verdict = f"{FAIL_TAG} H1: SCALE MISMATCH — {finding}"
    severity = 3
elif max_share > 0.35:
    finding = f"{max_col} at {max_share:.1%} — mild imbalance."
    verdict = f"{WARN_TAG} H1: mild scale imbalance — {finding}"
    severity = 1
else:
    finding = "All 4 models contribute 25-40% of weighted sum — no scale issue."
    verdict = f"{PASS_TAG} H1: no scale mismatch — {finding}"
    severity = 0

results["H1"] = {"severity": severity, "verdict": verdict, "finding": finding}
print(verdict)


# ==================================================================
# H2: META-FEATURES CONSTANT / WEAK ON VAL
# ==================================================================
banner(2, "Meta-features constant or weak on val",
       "If rolling ICs / PnLs don't vary, Ridge has no signal to "
       "condition on and weights will be intercept-dominated.")

mf_val = mf_val_raw.loc[meta_idx]
mf_stats = pd.DataFrame({
    "mean": mf_val.mean(),
    "std":  mf_val.std(),
    "min":  mf_val.min(),
    "max":  mf_val.max(),
    "iqr":  mf_val.quantile(0.75) - mf_val.quantile(0.25),
})
print("Meta-feature statistics on val (non-NaN rows only):")
print(mf_stats.round(4))

cv = (mf_val.std() / mf_val.abs().mean().replace(0, np.nan)).fillna(np.inf)
print(f"\nCoefficient of variation (std / |mean|) per feature:")
print(cv.round(3))

zero_cols  = mf_val.columns[mf_val.std() < 1e-6].tolist()
low_signal = cv[(cv > 0) & (cv < 0.10)].index.tolist()

if zero_cols:
    finding = f"Zero-variance features on val: {zero_cols}"
    verdict = f"{FAIL_TAG} H2: zero-variance features — {finding}"
    severity = 3
elif low_signal:
    finding = f"Near-constant features on val (CV < 0.10): {low_signal}"
    verdict = f"{WARN_TAG} H2: low-signal features — {finding}"
    severity = 1
else:
    finding = "Meta-features vary meaningfully on val (all CV >= 0.10)."
    verdict = f"{PASS_TAG} H2: meta-features informative — {finding}"
    severity = 0

results["H2"] = {"severity": severity, "verdict": verdict, "finding": finding}
print(verdict)


# ==================================================================
# H3: RIDGECV ALPHA OVER-REGULARIZATION
# ==================================================================
banner(3, "RidgeCV over-regularization",
       "If the selected alpha is the largest in the CV grid for any "
       "model, Ridge wanted even more regularization than available — "
       "weights will be near-uniform for that sub-model.")

# ml.models is List[RidgeCV], one per model_name
alpha_grid = ml.alphas
print(f"Alpha grid  : {alpha_grid}")
print(f"Max of grid : {max(alpha_grid)}")
print()

at_max = []
at_min = []
for name, m in zip(ml.model_names, ml.models):
    chosen = m.alpha_
    status = ("MAX" if chosen >= max(alpha_grid) else
              "min" if chosen <= min(alpha_grid) else "interior")
    print(f"  {name:>10} model: alpha_chosen = {chosen:8.3f}  [{status}]")
    if chosen >= max(alpha_grid):
        at_max.append(name)
    elif chosen <= min(alpha_grid):
        at_min.append(name)

print()
print(f"Models at grid MAX ({max(alpha_grid)}): {at_max}")
print(f"Models at grid min ({min(alpha_grid)}): {at_min}")

# Measure actual weight dispersion on val meta rows
X_val       = mf_val.values
raw_scores  = np.column_stack([m.predict(X_val) for m in ml.models])
sm_weights  = softmax(raw_scores, axis=1)
row_std     = sm_weights.std(axis=1)
col_std     = sm_weights.std(axis=0)

print(f"\nSoftmax weight statistics on val meta rows ({len(X_val)} rows):")
print(f"  Per-row std (across 4 models):  mean={row_std.mean():.5f}  "
      f"max={row_std.max():.5f}  min={row_std.min():.5f}")
print(f"  Per-model std (across val rows):")
for name, s in zip(ml.model_names, col_std):
    print(f"    {name:>10}: {s:.5f}")
print(f"  (uniform 0.25 weights have per-row std = 0.000)")

print(f"\nRaw Ridge score stats (before softmax):")
for i, name in enumerate(ml.model_names):
    col = raw_scores[:, i]
    print(f"  {name:>10}: mean={col.mean():.6f}  std={col.std():.6f}  "
          f"range=[{col.min():.4f}, {col.max():.4f}]")

if len(at_max) >= 3:
    finding = (f"{len(at_max)}/4 models ({at_max}) chose alpha={max(alpha_grid)} "
               f"(grid max). Ridge wanted MORE regularization than the grid allows. "
               f"These models produce near-constant scores; only the model(s) at "
               f"low alpha drive variation. Per-row weight std = {row_std.mean():.5f}.")
    verdict = f"{FAIL_TAG} H3: ALPHA SATURATED FOR {len(at_max)}/4 MODELS — {finding}"
    severity = 3
elif len(at_max) >= 1:
    finding = f"{at_max} model(s) at grid max — partial over-regularization."
    verdict = f"{WARN_TAG} H3: partial over-regularization — {finding}"
    severity = 2
else:
    finding = "All alphas interior or at min — no over-regularization."
    verdict = f"{PASS_TAG} H3: alpha healthy — {finding}"
    severity = 0

results["H3"] = {"severity": severity, "verdict": verdict, "finding": finding}
print(verdict)


# ==================================================================
# H4: PROXY TARGET BIAS TOWARD DRAWDOWN
# ==================================================================
banner(4, "Proxy target / softmax bias toward drawdown",
       "Reconstruct proxy target on val — which model had the highest "
       "next-window Spearman IC most often? If drawdown wins most of "
       "the time, Ridge correctly learned to favor it, but the training "
       "target is misaligned with directional PnL generation.")

WINDOW = 21
preds_for_target = ensemble_preds.loc[val_idx, PRED_COLS].copy()
# Remap regime to risk-ordered numeric so Spearman is monotone with risk
preds_for_target["regime_pred"] = (
    preds_for_target["regime_pred"].map({0.0: 0, 2.0: 1, 1.0: 2})
)

ic_wins  = []
ic_rows  = []
dates_list = list(preds_for_target.index)

for i, t in enumerate(dates_list[:-WINDOW]):
    fwd_dates = [d for d in dates_list if d > t][:WINDOW]
    if len(fwd_dates) < WINDOW:
        continue
    fwd_ret = returns_series.reindex(fwd_dates).dropna()
    if len(fwd_ret) < WINDOW // 2:
        continue

    row = {"date": t}
    for col in PRED_COLS:
        pred_w = preds_for_target[col].reindex(fwd_ret.index)
        if pred_w.std() < 1e-9 or fwd_ret.std() < 1e-9:
            row[col] = np.nan
            continue
        rho, _ = spearmanr(pred_w.values, fwd_ret.values)
        row[col] = float(rho) if not np.isnan(rho) else np.nan

    ic_rows.append(row)
    valid = {k: v for k, v in row.items() if k != "date" and not np.isnan(v)}
    if valid:
        winner = max(valid, key=lambda k: valid[k])
        ic_wins.append(winner)

ic_df = pd.DataFrame(ic_rows).set_index("date")
print(f"Proxy target IC matrix computed for {len(ic_wins)} val dates.")
print(f"\nMean Spearman IC per model over val forward windows:")
print(ic_df[PRED_COLS].mean().round(4))
print(f"\nFraction each model wins the proxy target:")
winner_dist = pd.Series(ic_wins).value_counts(normalize=True).round(3)
print(winner_dist)

if len(winner_dist) == 0:
    finding = "Could not compute proxy target — insufficient val data."
    verdict = f"{WARN_TAG} H4: inconclusive"
    severity = 1
else:
    top_winner = winner_dist.index[0]
    top_share  = float(winner_dist.iloc[0])

    if top_winner == "drawdown_risk_prob" and top_share > 0.35:
        finding = (f"drawdown_risk_prob wins proxy target {top_share:.1%} of val days. "
                   f"Ridge correctly learned to over-weight it per training target, "
                   f"but drawdown_risk_prob is a binary risk signal, not a return "
                   f"predictor — proxy target is misaligned with directional PnL.")
        verdict = f"{FAIL_TAG} H4: proxy target rewards drawdown disproportionately — {finding}"
        severity = 2
    elif top_share > 0.55:
        finding = f"{top_winner} dominates proxy target at {top_share:.1%}."
        verdict = f"{WARN_TAG} H4: concentrated proxy target on {top_winner}"
        severity = 1
    else:
        finding = f"Winners spread across models — target is balanced."
        verdict = f"{PASS_TAG} H4: proxy target balanced — {finding}"
        severity = 0

results["H4"] = {"severity": severity, "verdict": verdict, "finding": finding}
print(verdict)


# ==================================================================
# FINAL SUMMARY
# ==================================================================
print(f"\n{'=' * 70}\nFINAL RANKING  (severity 0=healthy, 3=primary cause)\n{'=' * 70}")
ranked = sorted(results.items(), key=lambda x: -x[1]["severity"])
for h, r in ranked:
    print(f"  {h} (severity {r['severity']}): {r['verdict']}")

print(f"\n{'=' * 70}\nRECOMMENDED ACTION\n{'=' * 70}")
primary = ranked[0][0]
sev     = ranked[0][1]["severity"]

if sev == 0:
    print("All hypotheses passed — the problem is elsewhere.")
elif primary == "H3":
    print("FIX H3 FIRST: widen RidgeCV alpha grid. Current grid (0.1, 1, 10, 100)")
    print("is too narrow — 3/4 models saturated at the max. Proposed grid:")
    print("  (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)")
    print("Also normalize meta-features before Ridge (StandardScaler on train)")
    print("to avoid scale-driven regularization pressure.")
elif primary == "H1":
    print("FIX H1: z-score each prediction column BEFORE the weighted sum.")
elif primary == "H2":
    print("FIX H2: rebuild meta-features with more signal.")
elif primary == "H4":
    print("FIX H4: redesign proxy target to use signal * actual_return PnL.")
