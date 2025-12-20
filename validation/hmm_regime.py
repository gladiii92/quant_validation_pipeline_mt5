"""
HMM-basierte Regime-Analyse für Trading-Strategien.

- Schätzt Marktregime aus Renditen via Gaussian HMM
- Ordnet Trades diesen Regimen zu
- Berechnet Kennzahlen (Sharpe, MaxDD, PF etc.) je Regime
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from backtest.metrics import calculate_metrics
from utils.logger import get_logger

logger = get_logger("HMM_REGIMES")


# ---------------------------------------------------------------------------
# HMM-Regime-Detektor
# ---------------------------------------------------------------------------

@dataclass
class MarketRegimeHMM:
    n_states: int = 3
    n_iter: int = 1000
    random_state: int = 42

    def __post_init__(self) -> None:
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

    def fit(self, returns: pd.Series) -> None:
        clean = returns.dropna().values.reshape(-1, 1)
        if clean.shape[0] < self.n_states * 10:
            raise ValueError(
                f"Not enough data to fit HMM: got {clean.shape[0]} obs "
                f"for {self.n_states} states."
            )
        self.model.fit(clean)

    def predict(self, returns: pd.Series) -> pd.Series:
        clean = returns.dropna()
        X = clean.values.reshape(-1, 1)
        states = self.model.predict(X)
        # Index wie returns (nur auf non-NaN)
        return pd.Series(states, index=clean.index, name="regime")


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _max_drawdown_from_equity(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min()) * -1.0


def _attach_regimes_to_trades(
    trades_df: pd.DataFrame,
    regime_series: pd.Series,
) -> pd.DataFrame:
    """
    Mapped die HMM-Regime (Index = Zeit) auf Trades (entrytime).
    Nimmt das Regime am Entry-Zeitpunkt (vorheriges bekanntes Regime).
    """
    if trades_df.empty:
        raise ValueError("trades_df empty in _attach_regimes_to_trades")

    if "entrytime" not in trades_df.columns:
        raise ValueError("trades_df must contain 'entrytime' column")

    df = trades_df.copy()
    df["entrytime"] = pd.to_datetime(df["entrytime"])
    regime_series = regime_series.sort_index()

    # Für jeden Entry das letzte bekannte Regime vor oder gleich der Entry-Zeit
    # via reindex(method="ffill")
    df = df.sort_values("entrytime")
    regimes_for_trades = regime_series.reindex(df["entrytime"], method="ffill")
    df["hmm_regime"] = regimes_for_trades.values

    return df


# ---------------------------------------------------------------------------
# Hauptfunktion für die Pipeline
# ---------------------------------------------------------------------------

def analyze_hmm_regimes(
    trades_df: pd.DataFrame,
    prices_df: pd.DataFrame | None,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Führt HMM-basierte Regime-Analyse durch.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Erwartet Spalten ['entrytime', 'pnl'].
    prices_df : pd.DataFrame | None
        Optionaler OHLCV-Preis-DataFrame mit Spalte 'close' und DatetimeIndex.
        Falls None, wird aus Trades eine Equity-Kurve gebaut und daraus
        Returns abgeleitet.
    config : Dict[str, Any]
        Abschnitt 'hmmregime' aus config.yaml, z.B.:

        hmmregime:
          enabled: true
          nstates: 3
          niter: 1000
          use_price_returns: true

    Returns
    -------
    Dict[str, Any]
        {
          "regimestats": { "<regime_id>": metrics_dict, ... },
          "state_series": pd.Series (Regime über Zeit),
        }
    """
    if trades_df.empty:
        raise ValueError("analyze_hmm_regimes: trades_df is empty.")

    if "entrytime" not in trades_df.columns:
        raise ValueError("trades_df must contain 'entrytime' column.")
    if "pnl" not in trades_df.columns:
        raise ValueError("trades_df must contain 'pnl' column.")

    n_states = int(config.get("nstates", 3))
    n_iter = int(config.get("niter", 1000))
    use_price_returns = bool(config.get("use_price_returns", True))

    # ------------------------------------------------------------------
    # 1) Returns bestimmen, auf denen HMM trainiert wird
    # ------------------------------------------------------------------
    if use_price_returns:
        if prices_df is None or "close" not in prices_df.columns:
            raise ValueError(
                "prices_df with 'close' column required when use_price_returns=True."
            )
        prices = prices_df.sort_index()["close"].astype(float)
        returns = prices.pct_change().dropna()
        logger.info(
            "HMM on price returns: %d observations, n_states=%d",
            len(returns),
            n_states,
        )
    else:
        # HMM auf Equity-Returns der Strategie
        sorted_trades = trades_df.sort_values("entrytime")
        initial_capital = float(config.get("initial_capital", 10_000.0))
        equity = initial_capital + sorted_trades["pnl"].cumsum().values
        equity_series = pd.Series(equity, index=pd.to_datetime(sorted_trades["entrytime"]))
        returns = equity_series.pct_change().dropna()
        logger.info(
            "HMM on equity returns: %d observations, n_states=%d",
            len(returns),
            n_states,
        )

    # ------------------------------------------------------------------
    # 2) HMM fitten und Regime-Zeitreihe erzeugen
    # ------------------------------------------------------------------
    hmm = MarketRegimeHMM(n_states=n_states, n_iter=n_iter)
    hmm.fit(returns)
    state_series = hmm.predict(returns)

    # ------------------------------------------------------------------
    # 3) Trades den Regimen zuordnen
    # ------------------------------------------------------------------
    trades_with_regime = _attach_regimes_to_trades(trades_df, state_series)

    # ------------------------------------------------------------------
    # 4) Kennzahlen pro Regime berechnen
    # ------------------------------------------------------------------
    initial_capital = float(config.get("initial_capital", 10_000.0))
    regimestats: Dict[str, Any] = {}

    for state in sorted(trades_with_regime["hmm_regime"].dropna().unique()):
        sub = trades_with_regime[trades_with_regime["hmm_regime"] == state]
        n_trades = len(sub)
        if n_trades < 10:
            logger.info(
                "Skipping HMM regime %s due to low sample size (%d trades).",
                state,
                n_trades,
            )
            continue

        metrics = calculate_metrics(sub, initial_capital=initial_capital)
        metrics["n_trades"] = n_trades
        regimestats[str(state)] = metrics

        logger.info(
            "HMM Regime %s: n_trades=%d, TotalReturn=%.2f, Sharpe=%.2f, MaxDD=%.2f, PF=%.2f",
            state,
            n_trades,
            metrics.get("total_return", 0.0) * 100.0,
            metrics.get("sharpe_ratio", 0.0),
            metrics.get("max_drawdown", 0.0) * 100.0,
            metrics.get("profit_factor", 0.0),
        )

    result: Dict[str, Any] = {
        "regimestats": regimestats,
        "state_series": state_series,
    }
    return result
