"""
VIX-Regime-Ausrichtung der Strategie gemäss regime_policy.yaml.
"""

from typing import Dict, Any

import pandas as pd

from backtest.metrics import calculate_metrics
from utils.logger import get_logger
from utils.config import load_config

logger = get_logger(__name__)


def _load_regime_policy(policy_path: str) -> Dict[str, Any]:
    """
    Lädt regime_policy.yaml.
    """
    policy = load_config(policy_path)
    if "strategies" not in policy:
        raise ValueError("regime_policy.yaml must contain a 'strategies' section")
    return policy


def analyze_vix_regime_alignment(
    trades_df: pd.DataFrame,
    vix_regimes: pd.Series,
    initial_capital: float,
    policy_path: str = "regime_policy.yaml",
    strategy_key: str = "range_breakout",
) -> Dict[str, Any]:
    """
    Ordnet Trades VIX-Regimen zu und vergleicht Performance mit der Policy.
    """
    df = trades_df.copy()

    if "entry_time" not in df.columns:
        raise ValueError("trades_df must contain 'entry_time' column")
    if "pnl" not in df.columns:
        raise ValueError("trades_df must contain 'pnl' column")

    df["date"] = df["entry_time"].dt.floor("D")
    vix_regimes = vix_regimes.sort_index()

    df["vix_regime"] = vix_regimes.reindex(df["date"]).values

    regime_stats: Dict[str, Any] = {}
    for name in sorted(df["vix_regime"].dropna().unique()):
        sub = df[df["vix_regime"] == name]
        if len(sub) < 10:
            logger.info(
                "Skipping VIX regime %s due to low sample size (n=%d)",
                name,
                len(sub),
            )
            continue

        m = calculate_metrics(sub, initial_capital=initial_capital)
        m["n_trades"] = len(sub)
        regime_stats[name] = m

        logger.info(
            "VIX Regime %s: n=%d, TotalReturn=%.2f%%, Sharpe=%.2f, MaxDD=%.2f%%, PF=%.2f",
            name,
            m["n_trades"],
            m["total_return"] * 100,
            m["sharpe_ratio"],
            m["max_drawdown"] * 100,
            m["profit_factor"],
        )

    policy_full = _load_regime_policy(policy_path)
    if "strategies" not in policy_full or strategy_key not in policy_full["strategies"]:
        raise ValueError(f"Strategy '{strategy_key}' not found in regime_policy.yaml")

    strat_policy = policy_full["strategies"][strategy_key]

    # Decision Logik HINZUFÜGEN (wie du gemacht hast):
    regime_decision = {
        "allowed": True,
        "risk_multiplier": 1.0,
        "violations": []
    }
    
    for regime_name, stats in regime_stats.items():
        regime_rule = strat_policy.get(regime_name, {})
        if "min_sharpe" in regime_rule and stats["sharpe_ratio"] < regime_rule["min_sharpe"]:
            regime_decision["allowed"] = False
            regime_decision["violations"].append(f"{regime_name}: Sharpe {stats['sharpe_ratio']:.2f} < {regime_rule['min_sharpe']}")
            regime_decision["risk_multiplier"] *= 0.5

    # **WICHTIG: TUPLE zurückgeben** (nicht Dict!)
    return {
        "regime_stats": regime_stats,
        "policy": strat_policy,
    }, regime_decision

