"""
models/meta/weight_tracker.py
Append-only audit trail of the meta-learner's daily model weights.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class WeightTracker:
    """
    Maintains an audit log of meta-learner weights.

    log_weights()       — appends a single-row snapshot (online use).
    log_weights_batch() — overwrites the file with a full batch snapshot
                          (backtest use; replaces any prior run's output).

    Parameters
    ----------
    path : str
        Path to the CSV file (relative to PROJECT_ROOT or absolute).
    """

    def __init__(self, path: str = "logs/weight_audit.csv") -> None:
        p = Path(path)
        self.path = p if p.is_absolute() else PROJECT_ROOT / p
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = ["date", "return_w", "vol_w", "regime_w", "drawdown_w"]

    def _header_needed(self) -> bool:
        return not self.path.exists() or self.path.stat().st_size == 0

    def log_weights(self, date, weights_dict: dict) -> None:
        """
        Append a single row.

        Parameters
        ----------
        date : date-like
            The date for this weight snapshot.
        weights_dict : dict
            Keys: 'return', 'vol', 'regime', 'drawdown'.
        """
        row = pd.DataFrame([{
            "date":       date,
            "return_w":   weights_dict.get("return"),
            "vol_w":      weights_dict.get("vol"),
            "regime_w":   weights_dict.get("regime"),
            "drawdown_w": weights_dict.get("drawdown"),
        }])
        row.to_csv(self.path, mode="a", header=self._header_needed(), index=False)

    def log_weights_batch(self, weights_df: pd.DataFrame) -> None:
        """
        Overwrite the audit file with a complete batch of weights.

        Used in backtest pipelines where the full weight history is
        recomputed from scratch each run. Overwrites any prior content
        so the file always reflects the most recent run's output.

        Parameters
        ----------
        weights_df : pd.DataFrame
            Index = date, columns = ['return', 'vol', 'regime', 'drawdown'].
        """
        out = weights_df.rename(columns={
            "return":   "return_w",
            "vol":      "vol_w",
            "regime":   "regime_w",
            "drawdown": "drawdown_w",
        }).copy()
        out.index.name = "date"
        out = out.reset_index()
        out.to_csv(self.path, index=False)  # full overwrite

    def load_weights(self) -> pd.DataFrame:
        """Load the full audit log as a date-indexed DataFrame."""
        return pd.read_csv(self.path, parse_dates=["date"], index_col="date")

    def plot_weights(self, save_path: str = "logs/weight_trajectory.png") -> None:
        """
        Stacked area chart of the 4 model weights over time.
        Saves to save_path (relative to PROJECT_ROOT or absolute).
        """
        import matplotlib.pyplot as plt

        if not self.path.exists():
            return

        df = self.load_weights()

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.stackplot(
            df.index,
            df["return_w"],
            df["vol_w"],
            df["regime_w"],
            df["drawdown_w"],
            labels=["return", "vol", "regime", "drawdown"],
            alpha=0.8,
        )
        ax.legend(loc="upper left")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Weight")
        ax.set_xlabel("Date")
        ax.set_title("Meta-Learner Model Weights Over Time")
        plt.tight_layout()

        sp = Path(save_path)
        if not sp.is_absolute():
            sp = PROJECT_ROOT / sp
        sp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(sp, dpi=150)
        plt.close()
