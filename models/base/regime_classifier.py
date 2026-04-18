"""
models/base/regime_classifier.py
HMM-based market regime classifier (bull / bear / sideways).

Regime labels are pinned to VIX levels after fitting so they remain
interpretable across different random seeds and re-runs:
  0 = bull     (lowest median VIX)
  1 = bear     (highest median VIX)
  2 = sideways (remaining state)

Output schema (data/processed/regimes.parquet)
-----------------------------------------------
  regime_filtered : causal, forward-only argmax of P(s_t | o_1..t).
                    Safe to use as a feature in downstream predictive models.
  regime_smoothed : Viterbi over the full series; label at t depends on
                    observations at t+1..T. Uses future observations —
                    diagnostics only, never use as a predictive feature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml
from hmmlearn.hmm import GaussianHMM
from loguru import logger
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Regime classifier
# ---------------------------------------------------------------------------
class RegimeClassifier:
    """
    GaussianHMM-based market regime classifier.

    Parameters
    ----------
    n_states : int
        Number of hidden states (default 3: bull, bear, sideways).
    """

    def __init__(self, n_states: int = 3) -> None:
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        self._scaler = StandardScaler()
        self._fitted = False
        logger.info("RegimeClassifier initialised — n_states={}", n_states)

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------
    def prepare_features_raw(
        self,
        df_close: pd.DataFrame,
        df_macro: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build HMM input matrix with 4 features, ALL lagged by 1 day.

        Features
        --------
        1. SPY log return   — log(close).diff().shift(1)
        2. SPY 21d roll vol — computed on shifted returns, then rolled
        3. VIXCLS level     — shift(1)
        4. T10Y2Y spread    — shift(1)

        Returns an UNSCALED DataFrame with NaN rows dropped.
        Caller must pass X_train through fit() before calling transform().
        """
        spy_log = np.log(df_close["SPY"])
        spy_ret = spy_log.diff().shift(1)
        spy_vol = spy_ret.rolling(21).std().rename("spy_vol_21d")
        spy_ret = spy_ret.rename("spy_ret")

        vix = df_macro["VIXCLS"].shift(1).rename("vixcls")
        t10y2y = df_macro["T10Y2Y"].shift(1).rename("t10y2y")

        return pd.concat([spy_ret, spy_vol, vix, t10y2y], axis=1).dropna()

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Scale X_raw using the scaler fitted on training data only."""
        if not self._fitted:
            raise RuntimeError(
                "transform() called before fit() — scaler is not fitted."
            )
        return pd.DataFrame(
            self._scaler.transform(X_raw),
            index=X_raw.index,
            columns=X_raw.columns,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X_train_raw: pd.DataFrame) -> None:
        """Fit scaler on training data, then fit GaussianHMM on scaled training data."""
        X_train = pd.DataFrame(
            self._scaler.fit_transform(X_train_raw),
            index=X_train_raw.index,
            columns=X_train_raw.columns,
        )
        self.model.fit(X_train.values)
        self._fitted = True
        logger.info(
            "HMM fitted — converged={} train_log_likelihood={:.4f} n_iter={}",
            self.model.monitor_.converged,
            float(self.model.score(X_train.values)),
            len(self.model.monitor_.history),
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_regimes_smoothed(self, X: pd.DataFrame) -> pd.Series:
        """
        Viterbi (smoothed) regime sequence.

        WARNING: Viterbi backtracks over the full sequence, so the label at
        time t depends on observations at t+1..T. Do NOT use as a feature in
        any downstream predictive model — for diagnostics only.
        """
        states = self.model.predict(X.values)
        return pd.Series(states, index=X.index, name="regime_smoothed_raw")

    def predict_regimes_filtered(self, X: pd.DataFrame) -> pd.Series:
        """
        Forward-pass filtered regime sequence (causal, no lookahead).

        Runs a single forward pass in log-space and returns argmax of the
        filtered posteriors P(s_t | o_1..t). Safe to use as a feature in
        downstream predictive models.
        """
        obs = X.values
        T = obs.shape[0]
        K = self.n_states

        log_startprob = np.log(self.model.startprob_ + 1e-300)
        log_transmat = np.log(self.model.transmat_ + 1e-300)
        log_B = self.model._compute_log_likelihood(obs)  # (T, K)

        log_alpha = np.full((T, K), -np.inf)
        log_alpha[0] = log_startprob + log_B[0]
        for t in range(1, T):
            prev = log_alpha[t - 1][:, None] + log_transmat  # (K, K)
            log_alpha[t] = _logsumexp(prev, axis=0) + log_B[t]

        log_norm = _logsumexp(log_alpha, axis=1, keepdims=True)
        filtered = np.exp(log_alpha - log_norm)
        states = np.argmax(filtered, axis=1)
        return pd.Series(states, index=X.index, name="regime_filtered_raw")

    # ------------------------------------------------------------------
    # Label assignment
    # ------------------------------------------------------------------
    def label_regimes_by_vix(
        self,
        regime_series: pd.Series,
        vix_series: pd.Series,
        name: str = "regime",
    ) -> pd.Series:
        """
        Remap raw HMM state integers to interpretable labels using median VIX.

        Generalised for any n_states:
          sorted_states[0]  -> 0 (bull,  lowest VIX)
          sorted_states[-1] -> 1 (bear, highest VIX)
          middle states     -> 2, 3, ... ordered by ascending VIX

        Returns
        -------
        pd.Series renamed to `name`.
        """
        aligned_vix = vix_series.reindex(regime_series.index)
        median_vix: dict[int, float] = {}
        for state in range(self.n_states):
            mask = regime_series == state
            if mask.sum() == 0:
                logger.warning(
                    "Raw state {} has zero observations — label pinning may be unstable.",
                    state,
                )
                median_vix[state] = np.inf
            else:
                median_vix[state] = float(aligned_vix[mask].median())

        sorted_states = sorted(median_vix, key=lambda s: median_vix[s])
        remap: dict[int, int] = {}
        remap[sorted_states[0]] = 0   # lowest VIX  -> bull
        remap[sorted_states[-1]] = 1  # highest VIX -> bear
        for mid_label, state in enumerate(sorted_states[1:-1], start=2):
            remap[state] = mid_label

        labelled = regime_series.map(remap).rename(name)
        logger.info(
            "Regime label mapping ({}): raw→label {}  |  median VIX per raw state: {}",
            name,
            remap,
            {s: (round(v, 2) if np.isfinite(v) else None) for s, v in median_vix.items()},
        )
        return labelled

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        regime_series: pd.Series,
        vix_series: pd.Series | None = None,
        spy_ret_series: pd.Series | None = None,
    ) -> Dict:
        """
        Compute regime diagnostics: transition matrix, avg days per regime,
        diagonal persistence, and optional VIX / return stats per regime.

        Returns a dict with all metrics; also logs results.
        """
        labels = sorted(regime_series.dropna().unique())
        n = len(labels)
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

        # Transition matrix (labels may not be contiguous 0..n-1)
        trans = np.zeros((n, n), dtype=int)
        states = regime_series.dropna().values
        for i in range(len(states) - 1):
            src = label_to_idx[states[i]]
            dst = label_to_idx[states[i + 1]]
            trans[src, dst] += 1
        row_sums = trans.sum(axis=1, keepdims=True)
        trans_prob = np.zeros_like(trans, dtype=float)
        np.divide(trans, row_sums, where=row_sums > 0, out=trans_prob)

        # Diagonal persistence
        diag_persistence = {int(labels[i]): float(trans_prob[i, i]) for i in range(n)}
        min_diag = min(diag_persistence.values())

        # Average consecutive days per regime
        runs: dict[int, list[int]] = {int(lbl): [] for lbl in labels}
        count = 1
        for i in range(1, len(states)):
            if states[i] == states[i - 1]:
                count += 1
            else:
                runs[int(states[i - 1])].append(count)
                count = 1
        runs[int(states[-1])].append(count)
        avg_days = {int(lbl): float(np.mean(runs[lbl])) if runs[lbl] else 0.0
                    for lbl in runs}

        result: Dict = {
            "transition_matrix": trans_prob.tolist(),
            "diag_persistence": diag_persistence,
            "avg_days_per_regime": avg_days,
        }

        # Soft warnings
        if min_diag < 0.85:
            logger.warning(
                "Regime persistence low: min diagonal = {:.3f} (< 0.85). "
                "Regimes may be flipping too often.",
                min_diag,
            )
        if min(avg_days.values()) < 10:
            logger.warning(
                "Regime avg run length low: min = {:.1f} days (< 10). "
                "Consider whether fewer states or different features would help.",
                min(avg_days.values()),
            )

        # VIX stats per regime
        if vix_series is not None:
            aligned_vix = vix_series.reindex(regime_series.index)
            vix_stats = {}
            for lbl in labels:
                mask = regime_series == lbl
                v = aligned_vix[mask].dropna()
                vix_stats[int(lbl)] = {
                    "median": round(float(v.median()), 2),
                    "mean":   round(float(v.mean()), 2),
                    "std":    round(float(v.std()), 2),
                }
            result["vix_by_regime"] = vix_stats
            logger.info("VIX distribution per regime: {}", vix_stats)

        # Return stats per regime
        if spy_ret_series is not None:
            aligned_ret = spy_ret_series.reindex(regime_series.index)
            ret_stats = {}
            for lbl in labels:
                mask = regime_series == lbl
                r = aligned_ret[mask].dropna()
                cnt = len(r)
                if cnt < 2:
                    ret_stats[int(lbl)] = {"count": cnt}
                else:
                    mean_d = float(r.mean())
                    std_d = float(r.std())
                    ret_stats[int(lbl)] = {
                        "count":      cnt,
                        "mean_daily": round(mean_d, 6),
                        "std_daily":  round(std_d, 6),
                        "ann_return": round(mean_d * 252, 4),
                        "ann_vol":    round(std_d * np.sqrt(252), 4),
                        "sharpe_ann": round(mean_d / std_d * np.sqrt(252), 4) if std_d > 0 else 0.0,
                    }
            result["return_stats_by_regime"] = ret_stats
            logger.info("Return stats per regime: {}", ret_stats)

        logger.info("Avg days per regime: {}", avg_days)
        logger.info("Diagonal persistence: {}", diag_persistence)
        logger.info("Transition matrix:\n{}", trans_prob)
        return result


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def _to_jsonable(obj):
    """Recursively convert numpy scalar types in dict keys/values to Python primitives."""
    if isinstance(obj, dict):
        return {
            (int(k) if hasattr(k, "item") else k): _to_jsonable(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return obj


def _logsumexp(a: np.ndarray, axis: int = 0, keepdims: bool = False) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "regime_classifier.log", rotation="10 MB", retention="30 days")

    cfg = _load_config()
    hmm_cfg = cfg.get("models", {}).get("hmm", {})

    from data.temporal_split import TemporalSplitter

    processed_dir = PROJECT_ROOT / cfg["paths"]["processed"]
    close_path = PROJECT_ROOT / cfg["paths"]["raw"] / "all_close_prices.parquet"
    macro_path = PROJECT_ROOT / cfg["paths"]["raw"] / "macro_data.parquet"

    for p in (close_path, macro_path):
        if not p.exists():
            print(f"Missing {p} — run ingest scripts first.")
            sys.exit(1)

    df_close = pd.read_parquet(close_path)
    df_macro = pd.read_parquet(macro_path)

    clf = RegimeClassifier(n_states=hmm_cfg.get("n_states", 3))

    # Build UNSCALED feature matrix
    X_raw = clf.prepare_features_raw(df_close, df_macro)

    # Temporal split on RAW features (no leakage)
    splitter = TemporalSplitter(cfg)
    X_train_raw = splitter.get_train(X_raw)
    X_val_raw   = splitter.get_val(X_raw)
    X_test_raw  = splitter.get_test(X_raw)

    logger.info(
        "Feature splits (raw) — train={} val={} test={}",
        len(X_train_raw), len(X_val_raw), len(X_test_raw),
    )

    # Fit scaler + HMM on TRAIN ONLY
    clf.fit(X_train_raw)

    # Transform full/val/test using train-fitted scaler
    X_full = clf.transform(X_raw)
    X_val  = clf.transform(X_val_raw)  if len(X_val_raw)  else None
    X_test = clf.transform(X_test_raw) if len(X_test_raw) else None

    val_ll  = float(clf.model.score(X_val.values))  if X_val  is not None and len(X_val)  > 0 else None
    test_ll = float(clf.model.score(X_test.values)) if X_test is not None and len(X_test) > 0 else None
    logger.info("Val log-likelihood: {}  |  Test log-likelihood: {}", val_ll, test_ll)

    # Predict: smoothed + filtered
    raw_smoothed = clf.predict_regimes_smoothed(X_full)
    raw_filtered = clf.predict_regimes_filtered(X_full)

    vix_full = df_macro["VIXCLS"].reindex(X_full.index)

    # Shared label mapping — pin based on SMOOTHED state VIX medians (stable),
    # apply the same remap to filtered so labels agree across both columns.
    aligned_vix = vix_full.reindex(raw_smoothed.index)
    median_vix = {}
    for state in range(clf.n_states):
        mask = raw_smoothed == state
        median_vix[state] = float(aligned_vix[mask].median()) if mask.sum() > 0 else np.inf
    sorted_states = sorted(median_vix, key=lambda s: median_vix[s])
    remap = {sorted_states[0]: 0, sorted_states[-1]: 1}
    for idx, s in enumerate(sorted_states[1:-1]):
        remap[s] = 2 + idx

    regime_smoothed = raw_smoothed.map(remap).rename("regime_smoothed")
    regime_filtered = raw_filtered.map(remap).rename("regime_filtered")
    logger.info("Shared label mapping (raw→label): {}", remap)

    # SPY log return (for per-regime return stats)
    spy_ret_full = np.log(df_close["SPY"]).diff().reindex(X_full.index)

    logger.info("=== Evaluating FILTERED (causal) regimes ===")
    metrics_filt = clf.evaluate(regime_filtered, vix_series=vix_full, spy_ret_series=spy_ret_full)

    logger.info("=== Evaluating SMOOTHED (Viterbi) regimes ===")
    metrics_smooth = clf.evaluate(regime_smoothed, vix_series=vix_full, spy_ret_series=spy_ret_full)

    # Outputs: two-column parquet + eval JSON
    out_df = pd.concat([regime_filtered, regime_smoothed], axis=1)
    out_path = processed_dir / "regimes.parquet"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path)
    logger.info("Regimes saved → {}", out_path.relative_to(PROJECT_ROOT))

    import json
    eval_path = log_dir / "regime_classifier_eval.json"
    with open(eval_path, "w") as fh:
        json.dump(
            {
                "label_mapping_raw_to_label": {str(int(k)): int(v) for k, v in remap.items()},
                "train_log_likelihood": float(clf.model.score(clf.transform(X_train_raw).values)),
                "val_log_likelihood":   val_ll,
                "test_log_likelihood":  test_ll,
                "converged":            bool(clf.model.monitor_.converged),
                "n_iter":               int(len(clf.model.monitor_.history)),
                "filtered": _to_jsonable(metrics_filt),
                "smoothed": _to_jsonable(metrics_smooth),
            },
            fh,
            indent=2,
        )
    logger.info("Eval JSON saved → {}", eval_path.relative_to(PROJECT_ROOT))

    # Console summary
    label_names = {0: "bull", 1: "bear", 2: "sideways"}
    print("\n=== Regime Distribution (FILTERED — use this for features) ===")
    for lbl, cnt in regime_filtered.value_counts().sort_index().items():
        print(f"  {lbl} ({label_names.get(int(lbl), '?'):>8}): {cnt:>5} days")

    print("\n=== Regime Distribution (SMOOTHED — diagnostics only) ===")
    for lbl, cnt in regime_smoothed.value_counts().sort_index().items():
        print(f"  {lbl} ({label_names.get(int(lbl), '?'):>8}): {cnt:>5} days")

    agreement = (regime_filtered == regime_smoothed).mean()
    print(f"\nFiltered vs Smoothed agreement: {agreement:.3%}")
    print(f"\nRegimes saved → {out_path.relative_to(PROJECT_ROOT)}")
    print(f"Eval JSON    → {eval_path.relative_to(PROJECT_ROOT)}")
    sys.exit(0)
