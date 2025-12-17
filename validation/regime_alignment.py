# validation/regime_alignment.py
from typing import Dict, Any

import pandas as pd

from backtest.metrics import calculate_metrics
from utils.logger import get_logger
from utils.config import load_config  # nutzt du bereits für config.yaml

logger = get_logger(__name__)


def _load_regime_policy(policy_path: str) -> Dict[str, Any]:
    """
    Lädt regime_policy.yaml mit demselben Loader wie deine Hauptconfig.
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
    Ordnet Trades VIX-Regimen zu und vergleicht die Performance mit der Policy.

    Args:
        trades_df: MT5-Trades mit 'entry_time' und 'pnl'.
        vix_regimes: pd.Series mit Index=Date, Wert=Regime-Name (z.B. 'Low_Volatility').
        initial_capital: Startkapital.
        policy_path: Pfad zu regime_policy.yaml.
        strategy_key: Key in regime_policy.yaml (z.B. 'range_breakout').

    Returns:
        Dict mit:
        - regime_stats: Dict regime_name -> metrics (+ n_trades)
        - policy: Dict der Policy für strategy_key
    """
    df = trades_df.copy()

    if "entry_time" not in df.columns:
        raise ValueError("trades_df must contain 'entry_time' column")
    if "pnl" not in df.columns:
        raise ValueError("trades_df must contain 'pnl' column")

    # Auf Tagesbasis mappen
    df["date"] = df["entry_time"].dt.floor("D")
    vix_regimes = vix_regimes.sort_index()

    # Map per reindex auf 'date'
    df["vix_regime"] = vix_regimes.reindex(df["date"]).values

    # Regime-Statistiken
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
        raise ValueError(
            f"Strategy '{strategy_key}' not found in regime_policy.yaml"
        )

    strat_policy = policy_full["strategies"][strategy_key]

    return {
        "regime_stats": regime_stats,
        "policy": strat_policy,
    }
