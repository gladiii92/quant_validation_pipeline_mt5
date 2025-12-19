"""
Monte-Carlo Simulationen auf Basis der MT5-Trade-Sequenz.
- Nutzt die empirische PnL-Verteilung deiner MT5-Trades
- Permutiert die PnL-Reihenfolge → viele Equity-Kurven
- Liefert: mc_positive_prob, Verteilung von Max Drawdown/Total Return + EQUITY PATHS!
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from backtest.metrics import calculate_metrics
from utils.logger import get_logger

logger = get_logger(__name__)

def run_monte_carlo_on_trades(
    trades_df: pd.DataFrame, 
    initial_capital: float = 10000.0, 
    n_sims: int = 5000, 
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Führt Monte-Carlo-Tests auf der PnL-Sequenz der Trades durch (Bootstrap mit Zurücklegen).
    
    Args:
        trades_df: DataFrame mit mindestens Spalte 'pnl'
        initial_capital: Startkapital für Equity-Berechnung
        n_sims: Anzahl Monte-Carlo-Pfade
        random_state: Seed für Reproduzierbarkeit
    
    Returns:
        Dict mit:
        - mc_positive_prob: Anteil Pfade mit positivem End-PnL
        - mc_median_return: Median Total Return über alle Pfade
        - mc_p5_return/mc_p95_return: 5/95-Quantile Total Return
        - mc_median_max_dd/mc_p95_max_dd: Median/95-Quantil Max Drawdown
        - total_returns/max_drawdowns: Arrays aller simulierten Werte
        - equity_paths: ALLE 5000 Equity Kurven für Plot! ✅
    """
    if "pnl" not in trades_df.columns:
        raise ValueError("trades_df muss eine 'pnl'-Spalte enthalten")
    
    pnl = trades_df["pnl"].astype(float).values
    n_trades = len(pnl)
    if n_trades == 0:
        raise ValueError("No trades provided for Monte Carlo simulation")
    
    rng = np.random.default_rng(random_state)
    
    total_returns = []
    max_drawdowns = []
    equity_paths = []  # ✅ NEU: Alle Equity Kurven speichern
    positive_count = 0
    
    logger.info("Running %d Monte Carlo simulations...", n_sims)
    
    for i in range(n_sims):
        # Bootstrap: Zufällige Reihenfolge mit Zurücklegen
        indices = rng.integers(0, n_trades, size=n_trades)
        boot_pnl = pnl[indices]
        
        # Equity Curve berechnen
        equity_curve = initial_capital + np.cumsum(boot_pnl)
        equity_paths.append(equity_curve.tolist())  # ✅ Speichern!
        
        # Metriken berechnen
        sim_df = pd.DataFrame({"pnl": boot_pnl})
        sim_metrics = calculate_metrics(sim_df, initial_capital=initial_capital)
        
        total_returns.append(sim_metrics["total_return"])
        max_drawdowns.append(sim_metrics["max_drawdown"])
        
        if sim_metrics["total_pnl"] > 0:
            positive_count += 1
    
    total_returns = np.array(total_returns)
    max_drawdowns = np.array(max_drawdowns)
    mc_positive_prob = positive_count / n_sims
    
    result: Dict[str, Any] = {
        "mc_positive_prob": float(mc_positive_prob),
        "mc_median_return": float(np.median(total_returns)),
        "mc_p5_return": float(np.quantile(total_returns, 0.05)),
        "mc_p95_return": float(np.quantile(total_returns, 0.95)),
        "mc_median_max_dd": float(np.median(max_drawdowns)),
        "mc_p95_max_dd": float(np.quantile(max_drawdowns, 0.95)),
        "total_returns": total_returns,
        "max_drawdowns": max_drawdowns,
        "equity_paths": equity_paths, 
    }
    
    logger.info("Monte Carlo results (trades-level, bootstrap):")
    logger.info("  mc_positive_prob: %.2f%%", result["mc_positive_prob"] * 100)
    logger.info("  mc_median_return: %.2f%%", result["mc_median_return"] * 100)
    logger.info("  mc_p5_return: %.2f%%", result["mc_p5_return"] * 100)
    logger.info("  mc_p95_return: %.2f%%", result["mc_p95_return"] * 100)
    logger.info("  mc_median_max_dd: %.2f%%", result["mc_median_max_dd"] * 100)
    logger.info("  mc_p95_max_dd: %.2f%%", result["mc_p95_max_dd"] * 100)
    
    return result
