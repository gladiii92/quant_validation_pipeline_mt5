"""
Kelly-basiertes Positionssizing aus empirischen Trade-Daten.
"""

from typing import Dict, Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def estimate_kelly_from_trades(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Schätzt Kelly-Quote aus Trade-PnLs.

    Args:
        trades_df: DataFrame mit Spalte 'pnl'.

    Returns:
        Dict mit:
        - win_rate
        - avg_win
        - avg_loss (positiv)
        - payoff_ratio
        - kelly_full, kelly_half, kelly_quarter
    """
    if "pnl" not in trades_df.columns:
        raise ValueError("trades_df must contain 'pnl' column")

    pnl = trades_df["pnl"].astype(float)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    if len(pnl) == 0 or len(wins) == 0 or len(losses) == 0:
        logger.warning("Not enough wins/losses to estimate Kelly")
        return {
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "payoff_ratio": 0.0,
            "kelly_full": 0.0,
            "kelly_half": 0.0,
            "kelly_quarter": 0.0,
        }

    win_rate = len(wins) / len(pnl)
    loss_rate = 1.0 - win_rate

    avg_win = wins.mean()
    avg_loss = -losses.mean()  # positive Zahl

    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    if payoff_ratio > 0:
        kelly_full = win_rate - (loss_rate / payoff_ratio)
    else:
        kelly_full = 0.0

    kelly_full = float(np.clip(kelly_full, 0.0, 1.0))

    result = {
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "payoff_ratio": float(payoff_ratio),
        "kelly_full": kelly_full,
        "kelly_half": kelly_full * 0.5,
        "kelly_quarter": kelly_full * 0.25,
    }

    logger.info("Kelly estimation from trades:")
    logger.info("  win_rate: %.2f%%", result["win_rate"] * 100)
    logger.info("  avg_win: %.2f, avg_loss: %.2f", result["avg_win"], result["avg_loss"])
    logger.info("  payoff_ratio: %.2f", result["payoff_ratio"])
    logger.info("  kelly_full: %.2f%%", result["kelly_full"] * 100)
    logger.info("  kelly_half: %.2f%%", result["kelly_half"] * 100)
    logger.info("  kelly_quarter: %.2f%%", result["kelly_quarter"] * 100)

    return result
