"""
Kelly-Analyse für Trade-Sequenzen.

Anpassung:
- Kelly-Fraction wird auf praxisnahe Maxima gecappt (Standard: 2 %).
- Rückgabe enthält bereits gerundete Werte (Basispunkte).
"""

from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd


def _kelly_from_winrate_payoff(p: float, b: float) -> float:
    """
    Klassische Kelly-Formel: f* = (p * (b + 1) - 1) / b.

    p : Gewinnwahrscheinlichkeit (0..1)
    b : Payoff-Ratio (avg_win / |avg_loss|)
    """
    if b <= 0.0:
        return 0.0
    f_star = (p * (b + 1.0) - 1.0) / b
    return max(f_star, 0.0)


def estimate_kelly_from_trades(
    trades_df: pd.DataFrame, max_fraction: float = 0.02
) -> Dict[str, float]:
    """
    Schätzt Kelly-Fraction aus Trade-PnLs.

    max_fraction: Obergrenze pro Trade (z.B. 0.01 = 1 %, 0.02 = 2 %).
    """
    if trades_df.empty or "pnl" not in trades_df.columns:
        return {
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "payoff_ratio": 0.0,
            "kelly_full": 0.0,
            "kelly_half": 0.0,
            "kelly_quarter": 0.0,
        }

    pnl = trades_df["pnl"].astype(float)
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]

    if wins.empty or losses.empty:
        return {
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "payoff_ratio": 0.0,
            "kelly_full": 0.0,
            "kelly_half": 0.0,
            "kelly_quarter": 0.0,
        }

    p = float(len(wins) / len(pnl))
    avg_win = float(wins.mean())
    avg_loss = float(losses.mean())  # negativ
    payoff = float(avg_win / abs(avg_loss))

    raw_kelly = _kelly_from_winrate_payoff(p, payoff)

    # Praxis: harte Obergrenze, z.B. 2 % pro Trade
    capped_kelly = min(raw_kelly, max_fraction)

    # Werte runden (Basispunkte)
    k_full = round(capped_kelly, 4)
    k_half = round(capped_kelly / 2.0, 4)
    k_quarter = round(capped_kelly / 4.0, 4)

    return {
        "win_rate": round(p, 4),
        "avg_win": avg_win,
        "avg_loss": abs(avg_loss),
        "payoff_ratio": round(payoff, 4),
        "kelly_full": k_full,
        "kelly_half": k_half,
        "kelly_quarter": k_quarter,
    }
