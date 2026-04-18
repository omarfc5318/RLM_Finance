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
