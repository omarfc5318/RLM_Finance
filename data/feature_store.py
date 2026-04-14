"""
data/feature_store.py
Versioned persistence and retrieval for feature DataFrames.

FeatureStore handles:
  - Saving parquet files with a version suffix and a companion metadata JSON.
  - Loading by explicit version or auto-resolving 'latest'.
  - Merging price and macro feature files into a single combined DataFrame.
  - Asserting that no feature values leak past a given split date.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import pandas as pd
import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Feature store
# ---------------------------------------------------------------------------
class FeatureStore:
    """
    Versioned feature persistence backed by the data/processed/ directory.

    File layout
    -----------
    data/processed/{name}_v{version}.parquet
    data/processed/{name}_v{version}_meta.json
    """

    def __init__(self, config: dict) -> None:
        self._processed_dir = PROJECT_ROOT / config["paths"]["processed"]
        self._processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save(self, df: pd.DataFrame, name: str, version: int) -> Path:
        """
        Persist a feature DataFrame and its metadata.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame to save.
        name : str
            Logical name, e.g. 'price_features' or 'combined_features'.
        version : int
            Integer version number.

        Returns
        -------
        Path to the written parquet file.
        """
        # Collision guard: if the requested version already exists, auto-increment
        # to the next unused version rather than silently overwriting saved data.
        original_version = version
        while (self._processed_dir / f"{name}_v{version}.parquet").exists():
            version += 1
        if version != original_version:
            logger.warning(
                "FeatureStore.save: v{} already exists for '{}' — "
                "bumped to v{} to avoid overwrite",
                original_version, name, version,
            )

        parquet_path = self._processed_dir / f"{name}_v{version}.parquet"
        meta_path = self._processed_dir / f"{name}_v{version}_meta.json"

        df.to_parquet(parquet_path)

        meta = {
            "name": name,
            "version": version,
            "date_created": datetime.now(timezone.utc).isoformat(),
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
            "index_start": df.index.min().isoformat(),
            "index_end": df.index.max().isoformat(),
        }
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        logger.info(
            "FeatureStore.save: {} → {} ({} rows, {} cols)",
            name,
            parquet_path.relative_to(PROJECT_ROOT),
            *df.shape,
        )
        return parquet_path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(self, name: str, version: Union[int, str] = "latest") -> pd.DataFrame:
        """
        Load a versioned feature DataFrame.

        Parameters
        ----------
        name : str
            Logical feature name.
        version : int or 'latest'
            Explicit version number, or 'latest' to auto-resolve the highest
            available version for this name.

        Returns
        -------
        pd.DataFrame
        """
        if version == "latest":
            version = self._resolve_latest_version(name)
            logger.info("FeatureStore.load: resolved '{}' latest → v{}", name, version)

        parquet_path = self._processed_dir / f"{name}_v{version}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"No feature file found at {parquet_path}. "
                f"Available versions: {self._list_versions(name)}"
            )

        df = pd.read_parquet(parquet_path)

        # Read companion metadata and log as a sanity check
        meta_path = self._processed_dir / f"{name}_v{version}_meta.json"
        if meta_path.exists():
            with open(meta_path) as fh:
                meta = json.load(fh)
            logger.info(
                "FeatureStore.load: loaded {} v{} — shape {} | created {}",
                name, version, tuple(meta["shape"]), meta["date_created"],
            )
        else:
            logger.warning(
                "FeatureStore.load: loaded {} v{} — shape {} | no metadata file found",
                name, version, df.shape,
            )

        return df

    # ------------------------------------------------------------------
    # Merge all
    # ------------------------------------------------------------------
    def merge_all(self, version: Union[int, str] = "latest") -> pd.DataFrame:
        """
        Inner-join price features and macro features on their date index.

        Parameters
        ----------
        version : int or 'latest'
            Version to load for both feature files.

        Returns
        -------
        pd.DataFrame
            Combined feature DataFrame aligned to the intersection of both
            date indices.
        """
        price_df = self.load("price_features", version)
        macro_df = self.load("macro_features", version)

        combined = pd.concat([price_df, macro_df], axis=1, join="inner")
        logger.info(
            "FeatureStore.merge_all: price {} + macro {} → combined {}",
            price_df.shape, macro_df.shape, combined.shape,
        )

        # Warn about any dates dropped by the inner join
        price_only = price_df.index.difference(combined.index)
        macro_only = macro_df.index.difference(combined.index)
        if len(price_only):
            logger.warning(
                "{} date(s) in price_features not in macro_features — dropped by inner join",
                len(price_only),
            )
        if len(macro_only):
            logger.warning(
                "{} date(s) in macro_features not in price_features — dropped by inner join",
                len(macro_only),
            )

        return combined

    # ------------------------------------------------------------------
    # Validate no future leakage
    # ------------------------------------------------------------------
    def validate_no_future(self, df: pd.DataFrame, split_date: str) -> None:
        """
        Assert that the DataFrame contains no rows beyond split_date.

        Use this to confirm a training feature set has not accidentally
        included validation or test dates.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame whose index is a DatetimeIndex.
        split_date : str
            ISO date string, e.g. '2021-12-31'.

        Raises
        ------
        AssertionError
            If df.index.max() > pd.Timestamp(split_date).
        """
        cutoff = pd.Timestamp(split_date)
        actual_max = df.index.max()
        assert actual_max <= cutoff, (
            f"Future leakage detected: feature DataFrame extends to "
            f"{actual_max.date()} which is after split_date {cutoff.date()}. "
            f"Slice your DataFrame to [:{split_date}] before training."
        )
        logger.info(
            "validate_no_future passed — max date {} ≤ split_date {}",
            actual_max.date(), cutoff.date(),
        )

        # Secondary check: warn if any column's first non-NaN value appears
        # at row position 0 or 1, which would indicate the feature was built
        # without a proper shift(1) lag (consistent with PriceFeatureBuilder
        # anti-lookahead assertion).  This does not raise — it warns so callers
        # can investigate without blocking a pipeline run.
        early_cols = []
        for col in df.columns:
            first_valid = df[col].first_valid_index()
            if first_valid is None:
                continue
            pos = df.index.get_loc(first_valid)
            if pos <= 1:
                early_cols.append((col, pos))
        if early_cols:
            details = "; ".join(f"{c} @ row {p}" for c, p in early_cols[:10])
            logger.warning(
                "validate_no_future: {} column(s) have their first non-NaN value "
                "within the first 2 rows — possible missing shift(1): {}",
                len(early_cols), details,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _list_versions(self, name: str) -> list[int]:
        """Return sorted list of available integer versions for a feature name."""
        pattern = f"{name}_v*.parquet"
        versions = []
        for p in self._processed_dir.glob(pattern):
            # filename format: {name}_v{version}.parquet
            stem = p.stem  # e.g. price_features_v3
            suffix = stem[len(name) + 2:]  # strip '{name}_v'
            if suffix.isdigit():
                versions.append(int(suffix))
        return sorted(versions)

    def _resolve_latest_version(self, name: str) -> int:
        versions = self._list_versions(name)
        if not versions:
            raise FileNotFoundError(
                f"No saved versions found for feature '{name}' "
                f"in {self._processed_dir}. Run build_all() first."
            )
        return versions[-1]


# ---------------------------------------------------------------------------
# Entry point — build macro features, merge, save combined_features v1
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "feature_store.log", rotation="10 MB", retention="30 days")

    cfg = _load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw"]
    store = FeatureStore(cfg)

    # --- Build and save macro features -----------------------------------
    macro_path = raw_dir / "macro_data.parquet"
    if not macro_path.exists():
        print(f"Missing {macro_path} — run ingest_macro.py first.")
        sys.exit(1)

    from data.feature_engineering import MacroFeatureBuilder

    df_macro = pd.read_parquet(macro_path)
    macro_builder = MacroFeatureBuilder(cfg)
    macro_features = macro_builder.build_all(df_macro)
    store.save(macro_features, "macro_features", version=1)

    # --- Expect price_features to exist already --------------------------
    price_path = PROJECT_ROOT / cfg["paths"]["processed"] / "price_features.parquet"
    if not price_path.exists():
        print(f"Missing {price_path} — run feature_engineering.py first.")
        sys.exit(1)

    # Wrap existing price_features.parquet under versioned naming
    price_features = pd.read_parquet(price_path)
    store.save(price_features, "price_features", version=1)

    # --- Merge and save combined features --------------------------------
    combined = store.merge_all(version=1)
    out_path = store.save(combined, "combined_features", version=1)

    print(f"\nCombined feature matrix saved → {out_path.relative_to(PROJECT_ROOT)}")
    print(f"Shape:      {combined.shape[0]} rows × {combined.shape[1]} columns")
    print(f"Date range: {combined.index[0].date()} → {combined.index[-1].date()}")
    print(f"NaN cells:  {combined.isna().sum().sum()}")
    sys.exit(0)
