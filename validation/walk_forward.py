"""
Walk-Forward / Out-of-Sample Analyse auf Trade-Ebene.

- Nimmt trades_df (Entry/Exit, PnL, Zeitstempel).
- Splittet die Historie in rollende Train/Test-Fenster.
- Berechnet Kennzahlen auf den Test-Fenstern (OOS).
"""

from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from backtest.metrics import calculate_metrics
from utils.logger import get_logger

logger = get_logger(__name__)


def _split_walk_forward(
    trades_df: pd.DataFrame,
    train_days: int = 180,
    test_days: int = 90,
    step_days: int = 30,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Erzeugt rollende Train/Test-Splits basierend auf entry_time.

    Args:
        trades_df: DataFrame mit Spalte 'entry_time'.
        train_days: Länge des Trainingsfensters in Tagen.
        test_days: Länge des Testfensters in Tagen.
        step_days: Schrittweite, um das Fenster nach vorne zu schieben.

    Returns:
        Liste von (train_df, test_df) Paaren.
    """
    df = trades_df.sort_values("entry_time").reset_index(drop=True).copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])

    if df.empty:
        return []

    start_date = df["entry_time"].min()
    end_date = df["entry_time"].max()

    splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []

    current_start = start_date

    while True:
        train_end = current_start + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)

        if train_end >= end_date:
            break  # kein sinnvolles Testfenster mehr

        train_mask = (df["entry_time"] >= current_start) & (df["entry_time"] < train_end)
        test_mask = (df["entry_time"] >= train_end) & (df["entry_time"] < test_end)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(test_df) > 0:
            splits.append((train_df, test_df))

        current_start = current_start + pd.Timedelta(days=step_days)
        if current_start >= end_date:
            break

    return splits


def run_walk_forward_analysis(
    trades_df: pd.DataFrame,
    initial_capital: float = 10_000.0,
    train_days: int = 180,
    test_days: int = 90,
    step_days: int = 30,
) -> Dict[str, Any]:
    """
    Führt Walk-Forward/OOS-Analyse auf den Trades durch.

    Args:
        trades_df: DataFrame mit Spalten 'entry_time', 'pnl'.
        initial_capital: Startkapital für Metrics.
        train_days: Trainingsfenster in Tagen.
        test_days: Testfenster in Tagen.
        step_days: Schrittweite in Tagen.

    Returns:
        Dict mit:
        - oos_sharpe: Durchschnittlicher Sharpe der Testfenster.
        - oos_profit_factor: Durchschnittlicher Profitfaktor der Testfenster.
        - oos_max_dd: Durchschnittlicher Max Drawdown der Testfenster.
        - n_windows: Anzahl der Testfenster.
        - window_metrics: Liste von Dicts mit Kennzahlen pro Fenster.
    """
    if "entry_time" not in trades_df.columns or "pnl" not in trades_df.columns:
        raise ValueError("trades_df must contain 'entry_time' and 'pnl' columns")

    splits = _split_walk_forward(
        trades_df,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
    )

    if not splits:
        logger.warning("No walk-forward windows could be constructed")
        return {
            "oos_sharpe": 0.0,
            "oos_profit_factor": 0.0,
            "oos_max_dd": 0.0,
            "n_windows": 0,
            "window_metrics": [],
        }

    window_metrics: List[Dict[str, Any]] = []

    for idx, (train_df, test_df) in enumerate(splits, start=1):
        # Wir interessieren uns hier vor allem für die Test-Performance.
        if len(test_df) == 0:
            continue

        metrics_test = calculate_metrics(test_df, initial_capital=initial_capital)
        wm = {
            "window_id": idx,
            "train_start": train_df["entry_time"].min() if len(train_df) > 0 else None,
            "train_end": train_df["entry_time"].max() if len(train_df) > 0 else None,
            "test_start": test_df["entry_time"].min(),
            "test_end": test_df["entry_time"].max(),
            "test_n_trades": len(test_df),
            "test_sharpe": metrics_test["sharpe_ratio"],
            "test_profit_factor": metrics_test["profit_factor"],
            "test_max_dd": metrics_test["max_drawdown"],
            "test_total_return": metrics_test["total_return"],
            # NEU: Indizes der Test-Trades im Original-DataFrame
            "test_indices": test_df.index.tolist(),
        }
        window_metrics.append(wm)

    if not window_metrics:
        logger.warning("No test windows with trades in walk-forward analysis")
        return {
            "oos_sharpe": 0.0,
            "oos_profit_factor": 0.0,
            "oos_max_dd": 0.0,
            "n_windows": 0,
            "window_metrics": [],
        }

    sharpe_vals = [w["test_sharpe"] for w in window_metrics]
    pf_vals = [w["test_profit_factor"] for w in window_metrics]
    dd_vals = [w["test_max_dd"] for w in window_metrics]

    oos_sharpe = float(np.mean(sharpe_vals))
    oos_profit_factor = float(np.mean(pf_vals))
    oos_max_dd = float(np.mean(dd_vals))

    result = {
        "oos_sharpe": oos_sharpe,
        "oos_profit_factor": oos_profit_factor,
        "oos_max_dd": oos_max_dd,
        "n_windows": len(window_metrics),
        "window_metrics": window_metrics,
    }

    logger.info("Walk-forward OOS results:")
    logger.info("  n_windows: %d", result["n_windows"])
    logger.info("  oos_sharpe: %.2f", result["oos_sharpe"])
    logger.info("  oos_profit_factor: %.2f", result["oos_profit_factor"])
    logger.info("  oos_max_dd: %.2f%%", result["oos_max_dd"] * 100)

    return result
