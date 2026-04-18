"""
models/base/volatility_predictor.py
GARCH(1,1) volatility predictor with a realized-vol benchmark.

Training contract
-----------------
  - fit() must only receive training-set returns.  Passing val or test data
    to fit() is a lookahead violation.
  - rolling_forecast() enforces the expanding-window discipline: at each
    step only data strictly before the current date is used to fit the model.
  - Returns are scaled by ×100 before GARCH fitting for numerical stability;
    all forecasts are unscaled before returning.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml
from arch import arch_model
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# GARCH volatility predictor
# ---------------------------------------------------------------------------
class VolatilityPredictor:
    """
    GARCH(1,1) 5-day-ahead annualised volatility forecaster.

    Parameters
    ----------
    p : int
        GARCH lag order for the squared error term (default 1).
    q : int
        GARCH lag order for the conditional variance term (default 1).
    """

    SCALE = 100          # multiply returns before fitting for numerical stability
    ANNUALISE = 252 ** 0.5

    def __init__(self, p: int = 1, q: int = 1) -> None:
        self.p = p
        self.q = q
        self.result = None
        self._last_fit_end: pd.Timestamp | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, return_series: pd.Series) -> None:
        """
        Fit GARCH(p, q) on the supplied return series.

        ANTI-LOOKAHEAD: only pass training-set returns here.

        Parameters
        ----------
        return_series : pd.Series
            Daily log or simple returns, date-indexed.
            Must NOT contain val or test observations.
        """
        scaled = return_series.dropna() * self.SCALE
        model = arch_model(scaled, vol="GARCH", p=self.p, q=self.q, rescale=False)
        self.result = model.fit(disp="off")
        self._last_fit_end = return_series.dropna().index[-1]
        logger.debug(
            "GARCH fit complete — {} obs, last date {}",
            len(scaled), self._last_fit_end.date(),
        )

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------
    def forecast(self, steps: int = 5) -> pd.Series:
        """
        Produce a multi-step ahead variance forecast.

        Returns annualised volatility (not variance) as a pd.Series of
        length `steps`.  Each value is sqrt(variance / SCALE²) * sqrt(252).

        Parameters
        ----------
        steps : int
            Forecast horizon in trading days.

        Returns
        -------
        pd.Series of annualised vol forecasts indexed 1 … steps.
        """
        if self.result is None:
            raise RuntimeError("Model not fitted — call fit() first.")

        fc = self.result.forecast(horizon=steps, reindex=False)
        # variance is in SCALE² units; undo scaling then annualise
        var_tail = fc.variance.iloc[-1].values          # shape (steps,)
        ann_vol = np.sqrt(var_tail / self.SCALE ** 2) * self.ANNUALISE
        return pd.Series(ann_vol, index=range(1, steps + 1), name="garch_vol_forecast")

    # ------------------------------------------------------------------
    # Rolling (expanding-window) forecast
    # ------------------------------------------------------------------
    def rolling_forecast(
        self,
        full_return_series: pd.Series,
        train_end_date: str,
        steps: int = 5,
    ) -> pd.Series:
        """
        Walk-forward expanding-window GARCH forecast from val_start to series end.

        For each date t in [val_start, series_end]:
          1. Fit GARCH on full_return_series up to and including t-1.
          2. Forecast `steps`-day annualised vol.
          3. Store the DAY-1 forecast value for date t.

        On GARCH convergence failure the model falls back to the 21-day
        realised vol as of t-1 (lagged, so no lookahead).  Every failure
        is logged with the failing date.

        Parameters
        ----------
        full_return_series : pd.Series
            The complete return series spanning train + val + test.
        train_end_date : str
            ISO date string matching config['splits']['train_end'].
            The first forecast date is the next business day after this.

        Returns
        -------
        pd.Series of annualised vol forecasts with DatetimeIndex.
        """
        train_end = pd.Timestamp(train_end_date)
        val_start = train_end + pd.offsets.BDay(1)

        forecast_dates: List[pd.Timestamp] = []
        forecast_vals: List[float] = []

        eval_dates = full_return_series.index[full_return_series.index >= val_start]
        n_dates = len(eval_dates)

        convergence_failures = 0

        for i, current_date in enumerate(eval_dates):
            # Expanding window: all returns strictly before current_date
            window = full_return_series.loc[:current_date].iloc[:-1]

            if len(window) < 60:
                # Too few observations for GARCH — use realised vol fallback
                rv = window.rolling(min(21, len(window))).std().iloc[-1] * self.ANNUALISE
                forecast_vals.append(float(rv) if not np.isnan(rv) else np.nan)
                forecast_dates.append(current_date)
                continue

            try:
                self.fit(window)
                fc = self.forecast(steps=steps)
                forecast_vals.append(float(fc.iloc[0]))  # 1-day-ahead point estimate
            except Exception as exc:
                convergence_failures += 1
                logger.warning(
                    "GARCH convergence failure on {} (failure #{}) — {}: {}. "
                    "Falling back to 21d realised vol.",
                    current_date.date(), convergence_failures, type(exc).__name__, exc,
                )
                # Fallback: 21-day realised vol lagged by 1 day (no lookahead)
                rv = window.rolling(21).std().iloc[-1] * self.ANNUALISE
                forecast_vals.append(float(rv) if not np.isnan(rv) else np.nan)

            forecast_dates.append(current_date)

            if (i + 1) % 50 == 0:
                logger.info(
                    "rolling_forecast progress: {}/{} dates processed "
                    "({} convergence failures so far)",
                    i + 1, n_dates, convergence_failures,
                )

        if convergence_failures:
            logger.warning(
                "rolling_forecast complete — {} total convergence failure(s) out of {} dates",
                convergence_failures, n_dates,
            )
        else:
            logger.info(
                "rolling_forecast complete — {} dates, 0 convergence failures",
                n_dates,
            )

        return pd.Series(
            forecast_vals,
            index=pd.DatetimeIndex(forecast_dates),
            name="garch_vol_5d",
        )


# ---------------------------------------------------------------------------
# Realised-vol benchmark
# ---------------------------------------------------------------------------
class RealizedVolPredictor:
    """
    Benchmark predictor: next-period volatility = lagged 21-day realised vol.

    predict(t) = std(returns[t-22 : t-1]) * sqrt(252)

    The shift(1) lag ensures no same-day return is used — fully consistent
    with the shift(+1) contract in feature_engineering.py.
    """

    ANNUALISE = 252 ** 0.5

    def predict(self, return_series: pd.Series) -> pd.Series:
        """
        Compute rolling 21-day annualised vol, lagged by 1 day.

        Parameters
        ----------
        return_series : pd.Series
            Full return series (train + val + test allowed; lagging handles safety).

        Returns
        -------
        pd.Series of annualised vol estimates with the same DatetimeIndex.
        """
        rv = return_series.shift(1).rolling(21).std() * self.ANNUALISE
        rv.name = "realized_vol_21d"
        return rv


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """RMSE after inner-joining on date index."""
    y_true, y_pred = y_true.align(y_pred, join="inner")
    mask = y_true.notna() & y_pred.notna()
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "volatility_predictor.log", rotation="10 MB", retention="30 days")

    cfg = _load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw"]
    processed_dir = PROJECT_ROOT / cfg["paths"]["processed"]

    # Load SPY close prices
    close_path = raw_dir / "all_close_prices.parquet"
    if not close_path.exists():
        print(f"Missing {close_path} — run ingest_prices.py first.")
        sys.exit(1)

    close = pd.read_parquet(close_path)[["SPY"]]
    spy_returns = np.log(close["SPY"]).diff().dropna()
    spy_returns.name = "SPY_logret"

    # Split dates from config
    train_end = cfg["splits"]["train_end"]
    val_end = cfg["splits"]["val_end"]
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    # --- GARCH rolling forecast ----------------------------------------
    logger.info("Starting GARCH rolling forecast for SPY (val + test dates)…")
    garch_predictor = VolatilityPredictor()
    garch_forecasts = garch_predictor.rolling_forecast(spy_returns, train_end)

    # --- Realised vol benchmark ----------------------------------------
    rv_predictor = RealizedVolPredictor()
    rv_forecasts = rv_predictor.predict(spy_returns)

    # Save all forecasts — align rv to the garch date range (val+test only)
    # so the saved file has no structural NaN from the training period.
    forecast_df = pd.DataFrame({
        "garch_vol_5d": garch_forecasts,
        "realized_vol_21d": rv_forecasts.reindex(garch_forecasts.index),
    })
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "vol_forecasts.parquet"
    forecast_df.to_parquet(out_path)
    logger.info("Saved → {} ({} rows)", out_path.relative_to(PROJECT_ROOT), len(forecast_df))

    # --- Val-set RMSE comparison ---------------------------------------
    # Eval target: 5-day forward realised vol (annualised) from targets.parquet.
    # This is the true out-of-sample quantity both models are trying to predict.
    #
    # Why NOT rolling(21).std() as target:
    #   RealizedVolPredictor outputs shift(1).rolling(21).std() — a 1-day-lagged
    #   copy of that same series.  Comparing a series to a 1-day-shifted version
    #   of itself yields near-zero RMSE (20/21 days overlap), making RV look
    #   artificially perfect and GARCH look ~4x worse than it really is.
    targets_path = processed_dir / "targets.parquet"
    if not targets_path.exists():
        logger.warning(
            "targets.parquet not found at {} — falling back to lagged-RV proxy target. "
            "Run data/targets.py first for a proper bakeoff.",
            targets_path,
        )
        # Fallback proxy: forward-shift the lagged RV by 1 day so it is at least
        # measuring the window that starts tomorrow, not the window ending yesterday.
        actual_tgt = spy_returns.shift(-1).rolling(21).std() * VolatilityPredictor.ANNUALISE
        actual_tgt.name = "proxy_fwd_rv_21d"
    else:
        import os
        os.environ.setdefault("RFIE_TRAINING_MODE", "1")
        tgt_df = pd.read_parquet(targets_path)
        tgt_col = "SPY_tgt_vol_5d"
        if tgt_col not in tgt_df.columns:
            raise KeyError(
                f"Expected column '{tgt_col}' in targets.parquet. "
                f"Available vol columns: {[c for c in tgt_df.columns if 'vol' in c]}"
            )
        # Annualise: targets.py stores raw std (not annualised)
        actual_tgt = tgt_df[tgt_col] * VolatilityPredictor.ANNUALISE
        actual_tgt.name = "tgt_vol_5d_ann"
        logger.info("Eval target: {} (5d forward realised vol, annualised)", tgt_col)

    val_mask = (actual_tgt.index > train_end_ts) & (actual_tgt.index <= val_end_ts)
    actual_val = actual_tgt[val_mask]

    garch_rmse = _rmse(actual_val, garch_forecasts)
    rv_rmse = _rmse(actual_val, rv_forecasts)

    logger.info("Val-set RMSE — GARCH: {:.6f}  |  RealizedVol: {:.6f}", garch_rmse, rv_rmse)

    winner = "GARCH" if garch_rmse < rv_rmse else "RealizedVol-21d"

    # Persist eval metrics so downstream scripts and CI can read them
    # convergence stats are logged inside rolling_forecast(); not re-derived here
    import json
    eval_output = {
        "eval_target": actual_tgt.name,
        "garch_rmse_val": garch_rmse,
        "realized_vol_rmse_val": rv_rmse,
        "winner": winner,
    }
    eval_path = log_dir / "model_b_eval.json"
    with open(eval_path, "w") as fh:
        json.dump(eval_output, fh, indent=2)
    logger.info("Eval metrics saved → {}", eval_path.relative_to(PROJECT_ROOT))

    print("\n=== Volatility Model Comparison (Val Set) ===")
    print(f"Eval target:           {actual_tgt.name}")
    print(f"GARCH(1,1) RMSE:       {garch_rmse:.6f}")
    print(f"RealizedVol-21d RMSE:  {rv_rmse:.6f}")
    print(f"Winner:                {winner}")
    print(f"\nForecasts saved → {out_path.relative_to(PROJECT_ROOT)}")
    print(f"Eval JSON  saved → {eval_path.relative_to(PROJECT_ROOT)}")
    sys.exit(0)
