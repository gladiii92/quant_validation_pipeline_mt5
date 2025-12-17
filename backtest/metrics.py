"""
Berechnet Performance-Metriken aus Trades.

- Input: Trade-Liste (z.B. aus MT5TradesLoader)
- Output: Sharpe, Sortino, MaxDD, Profit Factor, etc.
"""

from typing import Dict

import numpy as np
import pandas as pd


def calculate_metrics(trades_df: pd.DataFrame, initial_capital: float = 10000) -> Dict[str, float]:
    """
    Berechnet Performance-Metriken.

    Args:
        trades_df: DataFrame mit mindestens 'entry_time', 'exit_time', 'pnl' Spalten
        initial_capital: Startkapital (USD)

    Returns:
        Dictionary mit Metriken
    """
    if len(trades_df) == 0:
        raise ValueError("No trades provided")

    # Stelle sicher, dass pnl als float vorliegt
    pnl = trades_df["pnl"].astype(float).values

    # Basis-Statistik
    total_pnl = float(pnl.sum())
    total_return = total_pnl / float(initial_capital)
    total_trades = int(len(trades_df))

    wins = int((pnl > 0).sum())
    losses = int((pnl <= 0).sum())
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    # Returns für Sharpe/Sortino (hier: PnL relativ zum Startkapital pro Trade)
    returns = pnl / float(initial_capital)

    # Sharpe Ratio (angenommen ~252 "periods" pro Jahr; hier musst du ggf. auf Tages-Returns umstellen)
    if returns.std() > 0:
        sharpe = np.sqrt(252) * (returns.mean() / returns.std())
    else:
        sharpe = 0.0

    # Sortino Ratio (nur downside Volatilität)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() if len(downside_returns) > 0 else 0.0
    if downside_vol > 0:
        sortino = np.sqrt(252) * (returns.mean() / downside_vol)
    else:
        sortino = 0.0

    # Drawdown auf kumulativem PnL
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / float(initial_capital)
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

    # Profit Factor
    gross_profit = pnl[pnl > 0].sum() if (pnl > 0).any() else 0.0
    gross_loss = abs(pnl[pnl <= 0].sum()) if (pnl <= 0).any() else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Risk-Reward
    avg_win = pnl[pnl > 0].mean() if (pnl > 0).any() else 0.0
    avg_loss = abs(pnl[pnl <= 0].mean()) if (pnl <= 0).any() else 0.0
    reward_risk = avg_win / avg_loss if avg_loss > 0 else float("inf")

    return {
        "total_return": total_return,
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_drawdown),
        "profit_factor": float(profit_factor),
        "reward_risk": float(reward_risk),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
    }
