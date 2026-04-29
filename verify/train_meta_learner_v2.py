"""
verify/train_meta_learner_v2.py

Train MetaLearner v2 with the H3 fixes:
  - widened RidgeCV alpha grid (0.001 → 10000) to avoid grid-max saturation
  - StandardScaler on meta-features (already integrated in MetaLearner.train)
  - saturation tracking + softmax bypass (already integrated)

Output: models/meta/meta_learner_v2.joblib

Run:
  PYTHONPATH=/Users/ofarhan/rfie venv/bin/python verify/train_meta_learner_v2.py
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.temporal_split import TemporalSplitter
from models.meta.meta_learner import (
    MetaLearner,
    META_FEATURES_PATH,
    ENSEMBLE_PATH,
    TARGETS_PATH,
)


V2_PATH    = PROJECT_ROOT / "models" / "meta" / "meta_learner_v2.joblib"
V2_ALPHAS  = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def main() -> None:
    print("=" * 70)
    print("MetaLearner v2 training  (H3 fix: wider alpha grid + feature scaling)")
    print("=" * 70)

    with open(PROJECT_ROOT / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    for p, label in [
        (META_FEATURES_PATH, "meta_features.parquet"),
        (ENSEMBLE_PATH,      "ensemble_predictions.parquet"),
        (TARGETS_PATH,       "targets.parquet"),
    ]:
        if not p.exists():
            print(f"Missing {label} — run prerequisite scripts first.")
            sys.exit(1)

    meta_feats = pd.read_parquet(META_FEATURES_PATH)
    ens_preds  = pd.read_parquet(ENSEMBLE_PATH)
    targets    = pd.read_parquet(TARGETS_PATH)
    splitter   = TemporalSplitter(cfg)

    ml = MetaLearner(alphas=V2_ALPHAS)
    print(f"\nAlpha grid: {V2_ALPHAS}")

    X_tr, y_tr, X_ev, y_ev = ml.prepare_data(meta_feats, ens_preds, targets, splitter)
    ml.train(X_tr, y_tr)

    weights_eval = ml.predict_weights(X_ev)

    print("\n[Saturation] per model:")
    for name, sat in ml.saturated.items():
        chosen = ml.models[ml.model_names.index(name)].alpha_
        print(f"  {name:>10}  alpha={chosen:<10}  saturated={sat}")

    print("\n[Weights eval] describe:")
    print(weights_eval.describe().round(4).to_string())

    print("\n[Weights eval] per-column std:")
    for col in weights_eval.columns:
        print(f"  {col:>10}: {weights_eval[col].std():.6f}")

    lift = ml.evaluate_lift(weights_eval, ens_preds, targets)
    print(f"\n[Lift on val_meta_eval]")
    print(f"  meta_sharpe : {lift.get('meta_sharpe')}")
    print(f"  equal_sharpe: {lift.get('equal_sharpe')}")
    print(f"  lift_pct    : {lift.get('lift_pct')}%")
    print(f"  n_days      : {lift.get('n_days')}")

    ml.save(str(V2_PATH))
    print(f"\nSaved → {V2_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
