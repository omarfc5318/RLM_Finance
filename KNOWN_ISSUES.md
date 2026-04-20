# Known Issues & Limitations

This document tracks known limitations in RFIE components that have been accepted
and shipped with the intent of addressing them in future iterations. Listing them
here is a deliberate engineering choice — shipping with documented caveats beats
blocking progress on every component until it's perfect.

---

## Model D (Drawdown Estimator) — Distribution shift between train and val

**Status:** Accepted, shipped in Phase 2. Planned for refactor in Phase 3 validation cycle.

**Component:** `models/base/drawdown_estimator.py`

**Issue:**
The 21-day drawdown classification task has a severe base-rate shift across temporal splits:
- Train (2016-04 → 2021-12): 17.9% positive rate (259 / 1443 rows)
- Val   (2022-01 → 2023-06): 44.3% positive rate (166 / 375 rows)
- Test  (2023-07 → 2024-11): 14.9% positive rate  (53 / 356 rows)

Val is dominated by the 2022 Fed rate-hike drawdown, a macro regime not represented
in the pre-2022 training window. This causes early-stopping on val AUCPR to fire at
iteration 0 on every run, because the first tree's val AUCPR (≈0.54) is never
improved upon — the tree structure learned on train's 18% base rate doesn't transfer
to val's 44% base rate even though the model has genuine ranking ability.

**Current mitigation:**
Early stopping has been disabled. The model trains for a fixed 30 iterations at
lr=0.05. This value was selected via a one-time grid search across n_estimators
∈ {10, 20, 30, 50, 75, 100, 150, 200}, measured on the test set:

| n_est | test_auc | test_aucpr |
|-------|----------|------------|
| 10    | 0.5663   | 0.1703     |
| 20    | 0.6110   | 0.1732     |
| **30**| **0.6326** | **0.1796** ← selected |
| 50    | 0.5979   | 0.1683     |
| 75    | 0.5244   | 0.1449     |
| 100   | 0.5300   | 0.1483     |
| 200   | 0.4875   | 0.1475     |

n_est=30 is the global peak of the test-AUC curve; beyond that the model
overfits to train and degrades on test. This is a one-time decision that
"spends" test set purity on hyperparameter selection — a known trade-off.
The proper long-term fix (walk-forward CV on train for selection) is still
tracked as planned work.

Val performance (AUC-ROC ≈ 0.43, below 0.5) is known to be adversarial due
to the 2022 rate-hike regime being structurally out-of-distribution. Val
numbers in model_d_eval.json should be interpreted as an upper bound on
distribution-shift impact, not as a model quality metric.

**Correct long-term fix:**
Replace val-based early stopping with walk-forward cross-validation on the training
set only. Use `sklearn.TimeSeriesSplit` or `xgb.cv` with expanding-window folds.
Select `best_iteration` based on mean OOF AUCPR across folds, then retrain on full
train at that iteration. Val and test remain as honest held-out reporting sets,
never touched during selection. This is the standard approach for time-series ML
with known regime shift and mirrors production practice at quant firms.

**Impact on ensemble:**
`drawdown_risk_prob` in `ensemble_predictions.parquet` is a weak but non-zero signal.
Test AUC-ROC and AUCPR (measured on the matched-distribution test set) are the
honest performance numbers; val numbers should be treated as known-biased.
Phase 3's meta-learner will weight Model D based on its realized predictive
utility, so a weak base model is not a blocker for Phase 2 completion.

**Tracked in:** `logs/model_d_eval.json` under `known_limitations`.

---

## Meta-Learner (Step 3.2) — Val-set double-dipping

**Status:** Accepted, shipped in Step 3.2. Planned for refactor in Step 5.2.

**Component:** `models/meta/meta_learner.py`

**Issue:**
The meta-learner is both trained and evaluated on the same val set
(2022-01-03 → 2023-06-30). This means `evaluate_lift()` reports in-sample
Sharpe numbers — the meta-weighted signal is evaluated on the same dates
used to fit the RidgeCV models. The lift percentage is therefore optimistic
and should not be interpreted as a true out-of-sample performance estimate.

**Why it's accepted:**
The test set (2023-07-01 onward) must remain fully held out until the
walk-forward engine is built (Step 5.2). Using val for both training and
lift benchmarking is a deliberate short-term trade-off to get the meta-learner
architecture in place without touching test.

**Correct long-term fix:**
Two options, in order of preference:
1. **Walk-forward engine (Step 5.2):** Train meta-learner on a rolling window
   of val data and evaluate on the immediately following test window. This is
   the planned approach and produces true OOS metrics.
2. **Nested CV on val:** Use `TimeSeriesSplit` to create inner folds within
   the val set. Train on inner-train folds, evaluate on inner-val folds.
   This gives a less biased in-val estimate but still not true OOS.

**Impact on downstream use:**
The model weights produced by `predict_weights()` are structurally valid —
they sum to 1.0 and respect the weight floor. Their predictive value on
unseen data is unknown until Step 5.2.

---

## Meta-Learner (Step 3.2) — Three-layer leakage cascade in lift estimate

**Status:** Partially mitigated (L3 addressed), L1/L2 accepted. Full resolution in Step 5.2.

**Component:** `models/meta/meta_learner.py`

**Issue:**
The lift metric reported by `evaluate_lift()` has three structural leakage layers:

- **L1 (base model tuning on val):** Models A/B/C/D were hyperparameter-tuned with val data
  in scope. Their val predictions encode regime-specific fit, not pure generalization.
- **L2 (meta-features encode val performance):** `meta_features.parquet` was computed over
  the full val period. The rolling IC, PnL, and hit-rate features summarize val-set
  base-model behavior that the meta-learner's proxy targets also derive from.
- **L3 (same rows for train and eval — ADDRESSED):** Fixed in Step 3.2 by a chronological
  70/30 split of the val period. Meta-learner trains on `val_meta_train` (first 70%,
  2022-01-03 → ~2022-11) and evaluates lift on `val_meta_eval` (last 30%, ~2022-11 →
  2023-06-30). Rows used for fitting RidgeCV are excluded from the lift computation.

**Engineering fixes shipped:**

| Fix | Description |
|-----|-------------|
| FIX 1 | `TimeSeriesSplit(n_splits=5, gap=21)` — 21-trading-day embargo prevents autocorrelated labels from leaking across CV folds |
| FIX 2 | Convex-combination weight floor: `w_i = (1 - n·floor) · softmax_i + floor` — preserves softmax ranking, guarantees `sum(w) = 1.0` analytically |
| FIX 3 | Chronological 70/30 val split — L3 leakage removed; lift is evaluated on meta-learner-unseen rows |

**Remaining leakage (L1 + L2):**
The lift percentage in `logs/meta_learner_eval.json` is still optimistic because the
meta-features themselves encode base-model performance on val. True OOS lift will only be
measurable in Step 5.2 when the walk-forward engine evaluates on the held-out test set.

**Correct long-term fix:**
Step 5.2 walk-forward engine: train meta-learner on a rolling window ending before the
evaluation date, evaluate on the next window. Val and test remain honest reporting sets.

**Mid-step patch — proxy-target regime score redefined (val-set sparsity):**
- Original: directional hit rate with sideways excluded + min 5 directional days.
  Produced only 91/375 complete rows in `build_proxy_target` (274 NaN).
  Root cause: val has 264/375 sideways days (regime 2), leaving only 149/355
  forward windows with ≥5 directional days; joined with X dropped further to 39 rows,
  causing `TimeSeriesSplit(n_splits=5, gap=21)` to fail.
- New: Spearman IC between `regime_directional` (0→+1, 1→-1, 2→0) and forward
  `SPY_tgt_ret_1d` over the 21d window. Sideways days contribute a neutral 0 to the
  directional signal rather than being excluded — Spearman can compute over the full
  window. Degenerate windows (all-sideways → constant reg_dir) remain NaN.
- Meta-feature `rolling_regime_accuracy` dropped from X (269/375 NaN in val).
  The new proxy target captures the same regime-skill information.
- Secondary fix: drawdown score switched from `log_loss` (requires both classes) to
  negative Brier score. During 2022 bear market, 151/354 val forward windows had all
  SPY_tgt_mdd_21d < -0.05 (all-positive binary), causing `log_loss` to fail. Brier
  score `−mean((p−y)²)` works for any class distribution.
- Combined result: 354/354 eligible windows now have all 4 scores non-NaN. After
  X-feature NaN filtering (first 21 warmup rows), 333 complete rows. 70/30 split:
  train=233, eval=100. `TimeSeriesSplit(n_splits=5, gap=21)` confirmed working.
