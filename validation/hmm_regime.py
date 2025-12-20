"""
HMM-Regime-Analyse - EXAKT MONTE_CARLO.PY STRUKTUR
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class HMMAnalysisResult:
    regimestats: Dict[str, Any]
    stateseries: pd.Series

def analyze_hmm_regimes(
    tradesdf: pd.DataFrame, 
    pricesdf=None, 
    config: Dict[str, Any] = None,
    initialcapital: float = 100000.0
) -> HMMAnalysisResult:
    logger.info("Running HMM regime analysis...")
    
    if 'pnl' not in tradesdf.columns:
        logger.error("tradesdf muss eine pnl-Spalte enthalten")
        return HMMAnalysisResult({}, pd.Series(dtype=int))
    
    pnl = tradesdf['pnl'].astype(float).values
    n_trades = len(pnl)
    if n_trades == 0:
        logger.error("No trades provided for HMM analysis")
        return HMMAnalysisResult({}, pd.Series(dtype=int))
    
    n_states = int(config.get('n_states', 3) if config else 3)
    min_trades_per_regime = int(config.get('min_trades_per_regime', 20) if config else 20)
    logger.info(f"HMM regime analysis: trades-level, {n_states} states")
    
    # Equity curve als pandas Series
    equity_curve = pd.Series(initialcapital + np.cumsum(pnl))
    returns = equity_curve.pct_change().dropna()
    
    if len(returns) < 10:
        logger.warning("Insufficient returns for HMM analysis")
        return HMMAnalysisResult({}, pd.Series(dtype=int))
    
    try:
        hmm = MarketRegimeHMM(n_states=n_states)  # ← n_states mit Unterstrich!
        hmm.fit(returns)
        state_series = hmm.predict(returns)
    except Exception as e:
        logger.warning(f"HMM analysis failed: {e}")
        return HMMAnalysisResult({}, pd.Series(dtype=int))
    
    # Regimes zu trades zuweisen
    regime_assignments = np.tile(state_series.values, n_trades // len(state_series) + 1)[:n_trades]
    
    # Regime-Statistiken berechnen
    regimestats = {}
    unique_regimes = np.unique(regime_assignments)
    
    for state in unique_regimes:
        state_mask = regime_assignments == state
        regime_pnl = pnl[state_mask]
        
        if len(regime_pnl) < min_trades_per_regime:
            continue
            
        regime_equity = initialcapital + np.cumsum(regime_pnl)
        total_return = (regime_equity[-1] - initialcapital) / initialcapital
        
        regime_returns = regime_pnl / initialcapital
        sharpe_ratio = np.mean(regime_returns) / (np.std(regime_returns) + 1e-8) * np.sqrt(252)
        
        peak = np.maximum.accumulate(regime_equity)
        drawdowns = regime_equity - peak
        max_drawdown = np.min(drawdowns) / initialcapital
        
        wins = regime_pnl[regime_pnl > 0]
        losses = regime_pnl[regime_pnl < 0]
        profit_factor = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 else 999.0
        
        regimestats[str(int(state))] = {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'winrate': float(len(wins) / len(regime_pnl)),
            'profit_factor': float(profit_factor),
            'n_trades': int(len(regime_pnl))
        }
        logger.info(f"HMM Regime {int(state)}: n={len(regime_pnl)}, TR={total_return*100:.1f}%, Sharpe={sharpe_ratio:.2f}")
    
    logger.info(f"HMM regimes found: {len(regimestats)}")
    return HMMAnalysisResult(regimestats=regimestats, stateseries=state_series)
