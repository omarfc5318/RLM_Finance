# RFIE Design Log

This document records architectural changes that diverge from the
canonical PDF spec (RFIE_Build_Prompts_v3.pdf). The PDF remains the
project's original blueprint; this log is the source of truth for
what was actually built and why.

For known limitations and accepted caveats of the shipped system,
see KNOWN_ISSUES.md (the project's "what could go wrong" doc — there
is no separate KNOWN_LIMITATIONS.md file in this repo). This document
focuses on what changed and the reasoning, not on what's still
imperfect.

## Change 001: H1 — z-score normalization of base ensemble outputs

- **Date:** 2026-04-29
- **Files touched:**
  - `models/base/ensemble.py`
    - new module-level constants `_NORMALIZER_PATH`,
      `_REGIME_RISK_MAP = {0.0: 0.0, 2.0: 1.0, 1.0: 2.0}`, `_PRED_COLS`
    - `BaseEnsemble.__init__` auto-loads
      `models/base/prediction_normalizer.joblib` if present, populating
      `self._norm_stats`
    - new method `BaseEnsemble.fit_normalizer(full_predictions_df, train_end)`
    - new method `BaseEnsemble.normalize_predictions(preds_df)`
    - `BaseEnsemble.predict_all` calls `normalize_predictions` at the
      end when `self._norm_stats is not None` (`ensemble.py:343–345`)
    - `BaseEnsemble.compute_disagreement` guards the regime risk-remap
      so it only fires when `regime_pred` values are still raw HMM
      labels ⊆ {0.0, 1.0, 2.0}; skipped when input is already z-scored
    - `__main__` calls `predict_all` once raw, then `fit_normalizer`,
      then `predict_all` again to emit normalized predictions
  - `tests/test_ensemble_normalization.py` — 3 new unit tests
- **Reason:** `verify/diagnose_meta_learner.py` H1 hypothesis showed
  that under the original Phase 2 design, `regime_pred` (raw HMM
  labels in {0, 1, 2}) had `abs_mean = 1.627` while `return_pred` had
  `abs_mean = 0.000792` — a ~2000× scale gap. Under uniform 0.25
  weights, `regime_pred` contributed **79%** of the signal magnitude
  in the weighted sum (fractional contribution to
  |uniform-weighted sum|). A meta-learner cannot rebalance a sum that
  is already scale-dominated, so any reweighting was numerically
  irrelevant.
- **What changed:** Per-column mean/std are computed on the train
  period only (causal-safe). `regime_pred` is risk-remapped via
  `_REGIME_RISK_MAP` (HMM 0=bull → 0, HMM 2=sideways → 1, HMM 1=bear
  → 2) **before** statistics are computed, so the normalizer operates
  on a monotone risk scale, not arbitrary HMM labels. `vol_pred` has
  no train-period coverage (GARCH was fit on val+test only in
  Step 2.3), so its normalizer fallback uses the first ≤252 non-NaN
  rows of `vol_pred` regardless of date. Stats are persisted as a
  joblib dict `{col: (mean, std)}` at
  `models/base/prediction_normalizer.joblib`. After `fit_normalizer`,
  every subsequent `predict_all` call z-scores each prediction column;
  the normalized regime is a continuous z-score, no longer in {0, 1, 2}.
  Diagnostic confirms the fix: `verify/diagnose_meta_learner.py` H1
  dropped from FAIL severity 3 → WARN severity 1 (vol_pred 37%, an
  acceptable mild imbalance from val-period sparsity).
- **Backward compatibility:** `BaseEnsemble` works without the
  normalizer joblib — `self._norm_stats` stays `None`, `predict_all`
  emits raw predictions unchanged. `compute_disagreement` is
  idempotent: if regime is already normalized, the risk-remap step is
  silently skipped via the unique-value guard.

## Change 002: H3 — saturation gate in MetaLearner

- **Date:** 2026-04-29
- **Files touched:**
  - `models/meta/meta_learner.py`
    - new import `from sklearn.preprocessing import StandardScaler`
    - `MetaLearner.__init__` adds `self._feature_scaler` (Optional
      StandardScaler) and `self.saturated: dict` (model_name → bool)
    - `MetaLearner.train` fits a `StandardScaler` on `X.values` before
      fitting RidgeCV; populates
      `self.saturated[name] = bool(m.alpha_ >= alpha_max)` for each
      model; emits a warning when ≥1 model saturates
    - `MetaLearner.predict_weights` transforms `X` with the stored
      scaler (no-op when the scaler is `None`, for v1 joblibs), zeros
      the raw score column for any saturated model before softmax,
      then applies the convex floor as before
      (`meta_learner.py:356–374`)
    - `MetaLearner.save` / `MetaLearner.load` add `alphas`,
      `feature_scaler`, `saturated` to the persisted dict; `load()`
      uses `.get(..., default)` so v1 joblibs round-trip without
      schema errors
  - `verify/train_meta_learner_v2.py` — new training script that
    instantiates `MetaLearner` with `V2_ALPHAS = (0.001, 0.01, 0.1,
    1.0, 10.0, 100.0, 1000.0, 10000.0)` and writes
    `models/meta/meta_learner_v2.joblib`
- **Reason:** The original `MetaLearner.__init__` default
  `alphas = (0.1, 1.0, 10.0, 100.0)` caused **3 of 4** RidgeCV fits
  to saturate at the grid max (`return`, `vol`, `regime` all chose
  `alpha = 100.0`). Saturation means the CV wanted *more*
  regularization than the grid allowed; the resulting Ridge predictions
  are near-constant, but a softmax over near-constant scores still
  introduces small spurious variation in the weights. Only `drawdown`
  (chose `alpha = 0.1`) produced meaningful score variation, so
  `drawdown_w` was effectively the only weight with signal — and on
  val, `drawdown_risk_prob` anti-correlates with directional returns,
  so weighting it heavily *hurt* the meta-learner.
- **What changed:**
  - **Widened alpha grid** to
    `(0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)`,
    instantiated as `V2_ALPHAS` in `verify/train_meta_learner_v2.py`.
  - **StandardScaler on meta-features** inside `MetaLearner.train` so
    scale differences across `rolling_ic_*` and `rolling_pnl_*`
    columns don't inflate regularization pressure on small-magnitude
    columns. The same scaler is applied at inference inside
    `predict_weights`.
  - **Per-model `saturated` boolean** recorded in `self.saturated`,
    persisted in the joblib.
  - **Raw-score bypass:** in `predict_weights`, `raw_scores[:, i] = 0.0`
    for any `i` where `self.saturated[name] == True`, before the
    softmax. The convex floor then governs that model's final weight
    via `w_i = (1 − n·floor)·(1/n) + floor` for the bypassed dimensions
    (with `floor = 0.05`, `n = 4` → bypassed weight ≈ 0.20 per row,
    constant across the val window).
- **Result after retrain on the unified proxy target (Change 004):**
  2 of 4 still saturate at the new max `alpha = 10000.0`: `vol` and
  `regime`. `return` (chose `alpha = 10.0`) and `drawdown` (chose
  `alpha = 100.0`) are interior and adaptive. The two adaptive models
  are now the ones whose realized PnL contribution is most predictable
  from the meta-features. Observed weight stds on the
  `verify/evaluate_meta_lift.py` val run:
  `return_w = 0.112`, `drawdown_w = 0.110` (adaptive),
  `vol_w = 0.034`, `regime_w = 0.034` (floored).

## Change 003: Unified directional signal fusion

- **Date:** 2026-04-29
- **Files touched:**
  - `engine/feedback.py` — `FeedbackLoopEngine.step()`
    (`feedback.py:124–146`)
  - `models/meta/meta_learner.py` — `MetaLearner.evaluate_lift()`
    rewritten to use the same 4-model formula
  - `verify/evaluate_meta_lift.py` — `build_equal_weight_signal()`
    rewritten to consume normalized predictions directly. The previous
    body called `df["regime_pred"].map({0.0: 0, 2.0: 1, 1.0: 2})`,
    which silently produced all-NaN once `regime_pred` was z-scored
    (Change 001) — the equal-weight baseline had been quietly running
    on only 3 of 4 models.
- **Reason:** The PDF spec implicitly treated `vol_pred` and
  `drawdown_risk_prob` as if they were directional return predictors
  that could be added to `return_pred` in a weighted sum. They are
  not — they are **risk** signals, monotone in expected risk-off
  intensity, not in expected return. Adding them with a positive sign
  to a return signal is anti-correlated with intent: a high vol
  forecast should *subtract* directional exposure, not add to it.
  Worse, the original `step()` formula
  `signal = w_return·ret_pred + w_regime·regime_dir·|ret_pred|` used
  only 2 of the 4 weights — `vol_w` and `drawdown_w` were computed by
  `predict_weights` and audit-logged by `WeightTracker`, but
  **never entered the trading signal**. The meta-learner was solving
  a 4-class softmax against a downstream signal that could only see 2
  dimensions.
- **What changed.** The exact formula now used by `step()`
  (`feedback.py:136–146`):
  ```
  d_return   = +ret_pred
  d_regime   = -regime_val      (z-scored; NaN → 0 during warm-up)
  d_vol      = -vol_pred_val    (z-scored; NaN → 0 during warm-up)
  d_drawdown = -drawdown_prob   (z-scored; NaN → 0 during warm-up)

  weighted_signal = weights["return"]   * d_return
                  + weights["vol"]      * d_vol
                  + weights["regime"]   * d_regime
                  + weights["drawdown"] * d_drawdown
  ```
  Equivalently, expanding the signs against the **raw** weights and
  the **normalized** base predictions:
  ```
  weighted_signal = w_ret      * z_ret_pred
                  − w_vol      * z_vol_pred
                  − w_regime   * z_regime_pred
                  − w_drawdown * z_drawdown_pred
  ```
  The same formula now appears in three places — a single source of
  truth: `engine/feedback.py:step`, `MetaLearner.evaluate_lift`, and
  `verify/evaluate_meta_lift.py:build_equal_weight_signal`. NaN risk
  signals during warm-up contribute 0 to the weighted sum: the floor
  still gives the warm-up dimensions weight, but with `d_i = 0` they
  do not move the signal.
- **Backward compatibility:** `meta_v1` is preserved at
  `models/meta/meta_learner.joblib`. The redesigned
  `engine.feedback.step` uses the new directional formula
  unconditionally; v1 joblibs are still loadable (their
  `_feature_scaler` and `saturated` fields are absent and default to
  `None` / `{}` via `.get()` in `MetaLearner.load`). v1 weights
  evaluated under the new architecture are intentionally what the
  baseline-shopping check in `verify/sanity_check_baselines.py`
  measures — see KNOWN_ISSUES.md for the lift attribution.

## Change 004: Unified proxy target

- **Date:** 2026-04-29
- **Files touched:** `models/meta/meta_learner.py` —
  `MetaLearner.build_proxy_target` (`meta_learner.py:101` onward)
- **Reason:** The original `build_proxy_target` rewarded each model
  for **independent per-model skill** on a hand-crafted, model-specific
  metric: Spearman IC for `return_pred`, negative RMSE for `vol_pred`,
  Spearman IC against `regime_directional` for `regime_pred`,
  negative Brier score for `drawdown_risk_prob`. These metrics are not
  commensurable: a high Brier score for drawdown does not imply
  drawdown contributes positively to directional PnL — and on val it
  specifically didn't. The meta-learner was being trained to pick the
  model with the best per-model skill on its own metric, not the
  model whose signal contributed most to actual realized PnL under
  the unified directional fusion of Change 003.
- **What changed:** Each base model carries a directional signal `d_i`
  using the same sign convention as Change 003. For each date `t` in
  `meta_features_df.index`, the proxy score for model `i` is the
  realized directional PnL contribution of `d_i` over the forward
  21-day window:
  ```
  proxy_i[t] = mean over k in [t+1, t+W] of  d_i[k] * actual_return[k]
  ```
  where `actual_return[k] = SPY_tgt_ret_1d[k]` (forward-stamped log
  return — the return earned from holding `d_i[k]` over `k → k+1`).
  After scoring, drop rows where any model's score is NaN, then
  z-score each column on the remaining rows so softmax operates on
  commensurable scales. All 4 RidgeCV models now fit against the same
  scalar metric: realized directional PnL contribution, not per-model
  skill.

## Change 005: WeightTracker.clear() + auto-clear on run()

- **Date:** 2026-04-29
- **Files touched:**
  - `models/meta/weight_tracker.py` — new method
    `WeightTracker.clear()` writes a header-only CSV
    (`weight_tracker.py:58–65`)
  - `engine/feedback.py` — `FeedbackLoopEngine.run()` calls
    `self.weight_tracker.clear()` at the top
  - `tests/test_weight_tracker_clear.py` — 4 new unit tests
    (clear truncates rows, preserves header, post-clear logs work,
    fresh-tracker clear creates header-only file)
- **Reason:** `WeightTracker.log_weights` opens the audit CSV in
  append mode (`mode="a"`). When the Step 3.3 manual verification ran
  the simulation 5 times to check determinism, the audit CSV ballooned
  to 1875 rows (5 × 375 days). `evaluate_lift()` then computed Sharpe
  over `n = 1874` — effectively a 5-run average that diluted bad days
  and produced a misleading +3.1% lift number that did not reproduce
  on a single clean run. The accumulation was a silent bug; nothing
  detected the row-count anomaly until Check 5 of the Step 3.3 manual
  gate cross-checked single-run vs accumulated audit lengths.
- **What changed:** `WeightTracker.clear()` overwrites the audit CSV
  with `pd.DataFrame(columns=self.columns).to_csv(self.path,
  index=False)` — header row only, zero data rows.
  `FeedbackLoopEngine.run()` invokes `self.weight_tracker.clear()` as
  its first action so every backtest starts from a clean audit trail.
  The `log_weights_batch()` path (used by the offline
  `MetaLearner.__main__` post-train) was already overwrite-mode and
  is unchanged.

## Change 006: meta.learner_version config flag

- **Date:** 2026-04-29
- **Files touched:**
  - `config.yaml` — new top-level `meta:` block with key
    `learner_version: v2` (default)
  - `verify/evaluate_meta_lift.py` — reads
    `cfg["meta"]["learner_version"]` and routes `MetaLearner.load()`
    to either `models/meta/meta_learner.joblib` (v1) or
    `models/meta/meta_learner_v2.joblib` (v2)
- **Reason:** The v1 meta-learner (alpha grid 0.1–100, no
  StandardScaler, no saturation gate, original per-model proxy target)
  is preserved on disk so it can be loaded as an honest baseline
  against the redesigned v2 stack. Hard-coding the v2 path would have
  erased the ability to compare. The flag is also forward-compatible:
  a future v3 joblib can be added by extending the routing block
  without touching the trading code.
- **What changed.** `config.yaml` adds:
  ```
  meta:
    learner_version: v2
  ```
  `verify/evaluate_meta_lift.py:main()` resolves the version up front
  and prints it, so any verification report makes the active learner
  version explicit. v1 remains at the original path unchanged. Note:
  `engine/feedback.py` itself does not read this flag — it accepts
  whatever `MetaLearner` instance the caller injects. The flag is
  consumed at the boundary (verify scripts and, in future Phase 5
  work, the walk-forward driver).

## Reconciling with the PDF

The PDF spec defines the system at a conceptual level: four base
models (return predictor, volatility forecaster, regime classifier,
drawdown estimator), a meta-learner that produces dynamic weights,
a weight tracker for audit, a feedback-loop engine that consumes
predictions causally, and an exit gate of "meta Sharpe > equal weight
on val." Every one of those concepts survives in the actual system,
and the file/class names map 1:1: `BaseEnsemble`, `MetaLearner`,
`WeightTracker`, `FeedbackLoopEngine`. What the PDF got wrong was the
**interaction model**: it implicitly assumed the four base models
could be combined as if they were directional return predictors on a
common scale, and that a softmax over Ridge regressions of per-model
skill scores would pick the best of them on each day. Both
assumptions failed on val. The base outputs are on wildly different
scales (Change 001), only `return_pred` carries directional intent —
vol and drawdown are risk signals (Change 003) — and per-model skill
metrics don't translate to contributions to realized PnL under the
unified fusion (Change 004). Treating them as linear-additive parts
of the same return signal anti-correlated with the trading objective.

The redesigned system replaces the PDF's fusion math with a unified
directional signal in z-score space, retrains the meta-learner
against realized PnL contribution rather than per-model skill, fixes
the saturation pathology in the Ridge alpha grid, and adds the
audit-trail and config plumbing that production use exposed as
missing. None of the PDF's components were removed; they were
rewired. Future PDF revisions should reflect this; until then, this
log is the canonical description of what the repo actually does.
KNOWN_ISSUES.md documents what the redesigned system still cannot
guarantee — most notably that the in-val +28.8% directional-Sharpe
lift is overwhelmingly architectural (Change 003 is responsible for
≈+1.18 Sharpe; meta-learning adds ≈+0.14 on top) and that true
out-of-sample validation must wait for the Phase 5.2 walk-forward
engine.
