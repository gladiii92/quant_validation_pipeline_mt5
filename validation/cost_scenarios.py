"""
Kosten-/Slippage-Szenarien auf Trade-PnL.
"""

from typing import Dict, Any

import pandas as pd

from backtest.metrics import calculate_metrics
from utils.logger import get_logger

logger = get_logger(__name__)


def run_cost_scenarios(
    trades_df: pd.DataFrame,
    initial_capital: float,
    scenarios: Dict[str, float] = None,
) -> Dict[str, Any]:
    """
    Führt Kosten-/Slippage-Szenarien aus.

    Args:
        trades_df: DataFrame mit Spalte 'pnl'.
        initial_capital: Startkapital.
        scenarios: Dict Name -> Faktor, z.B.:
            {"base": 1.0, "cost_plus_25": 0.75, "cost_plus_50": 0.5}

    Returns:
        Dict name -> metrics_dict
    """
    if "pnl" not in trades_df.columns:
        raise ValueError("trades_df must contain 'pnl' column")

    if scenarios is None:
        scenarios = {
            "base": 1.0,
            "cost_plus_25": 0.75,
            "cost_plus_50": 0.5,
        }

    results: Dict[str, Any] = {}

    for name, factor in scenarios.items():
        df_scen = trades_df.copy()
        df_scen["pnl"] = trades_df["pnl"].astype(float) * float(factor)

        metrics = calculate_metrics(df_scen, initial_capital=initial_capital)
        results[name] = metrics

        logger.info("Cost scenario '%s': factor=%.2f", name, factor)
        logger.info(
            "  TotalReturn: %.2f%%, Sharpe: %.2f, MaxDD: %.2f%%, ProfitFactor: %.2f",
            metrics["total_return"] * 100,
            metrics["sharpe_ratio"],
            metrics["max_drawdown"] * 100,
            metrics["profit_factor"],
        )

    return results
