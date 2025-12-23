from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HMMAnalysisResult:
    regime_stats: Dict[str, Any]
    state_series: pd.Series


def analyze_hmm_regimes(
    trades_df: pd.DataFrame,
    prices_df: pd.DataFrame | None,
    config: Dict[str, Any] | None = None,
    initial_capital: float = 100_000.0,
) -> HMMAnalysisResult | Tuple[Dict[str, Any], pd.Series]:
    """
    Führt eine HMM‑Regimeanalyse auf Basis der Trade‑P&L‑Serie durch.

    - Schätzt ein GaussianHMM auf annualisierten Equity‑Returns.
    - Weist jedem Trade ein Regime‑Label zu.
    - Berechnet Kennzahlen pro Regime (TR, Sharpe, MaxDD, Winrate, PF).
    """

    logger.info("Running HMM regime analysis...")

    if "pnl" not in trades_df.columns:
        logger.error("trades_df muss eine 'pnl'-Spalte enthalten.")
        return HMMAnalysisResult({}, pd.Series(dtype=int))

    pnl = trades_df["pnl"].astype(float).values
    n_trades = len(pnl)
    if n_trades == 0:
        logger.error("No trades provided for HMM analysis.")
        return HMMAnalysisResult({}, pd.Series(dtype=int))

    n_states = int(config.get("nstates", 3)) if config else 3
    min_trades_per_regime = int(config.get("mintradesperregime", 20)) if config else 20
    logger.info("HMM regime analysis trades-level, %d states", n_states)

    # Equity‑Kurve und Returns
    equity_curve = pd.Series(initial_capital + np.cumsum(pnl), index=trades_df.index)
    returns = equity_curve.pct_change().dropna()

    if len(returns) < 10:
        logger.warning("Insufficient returns (%d) for HMM analysis.", len(returns))
        return HMMAnalysisResult({}, pd.Series(dtype=int))

    X = returns.values.reshape(-1, 1)

    try:
        hmm = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
        hmm.fit(X)
        state_seq = hmm.predict(X)
        # auf Trades-Länge hochziehen (erste Return gehört zum zweiten Trade)
        state_series = pd.Series(state_seq, index=returns.index)
    except Exception as e:
        logger.warning("HMM analysis failed: %s", e)
        return HMMAnalysisResult({}, pd.Series(dtype=int))

    # Regime‑Stats
    regime_assignments = np.tile(state_series.values, n_trades // len(state_series) + 1)[:n_trades]

    regime_stats: Dict[str, Any] = {}
    unique_states = np.unique(regime_assignments)

    for state in unique_states:
        mask = regime_assignments == state
        regime_pnl = pnl[mask]
        if len(regime_pnl) < min_trades_per_regime:
            continue

        regime_equity = initial_capital + np.cumsum(regime_pnl)
        total_return = (regime_equity[-1] - initial_capital) / initial_capital

        regime_returns = regime_pnl / initial_capital
        vol = np.std(regime_returns) + 1e-8
        sharpe_ratio = np.mean(regime_returns) / vol * np.sqrt(252)

        peak = np.maximum.accumulate(regime_equity)
        drawdowns = regime_equity - peak
        max_drawdown = drawdowns.min() / initial_capital

        wins = regime_pnl[regime_pnl > 0]
        losses = regime_pnl[regime_pnl < 0]
        if len(losses) > 0:
            profit_factor = wins.sum() / np.abs(losses.sum())
        else:
            profit_factor = 999.0

        winrate = len(wins) / len(regime_pnl) if len(regime_pnl) > 0 else 0.0

        regime_stats[str(int(state))] = {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "winrate": float(winrate),
            "profit_factor": float(profit_factor),
            "n_trades": int(len(regime_pnl)),
        }

        logger.info(
            "HMM Regime %d: n=%d, TR=%.1f%%, Sharpe=%.2f",
            state,
            len(regime_pnl),
            total_return * 100.0,
            sharpe_ratio,
        )

    logger.info("HMM regimes found %d", len(regime_stats))
    return HMMAnalysisResult(regime_stats=regime_stats, state_series=state_series)
