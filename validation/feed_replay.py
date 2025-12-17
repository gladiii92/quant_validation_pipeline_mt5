"""
Feed-Replay: MT5-Trades auf unabhängigen Preisfeed (z.B. yfinance) abbilden.

Idee:
- MT5 liefert Trade-Liste (entry_time, exit_time, direction, volume, pnl_mt5).
- Wir laden unabhängige Preise (USDJPY M15) und berechnen,
  was bei denselben Entry/Exit-Zeitpunkten dort herausgekommen wäre.
- Ergebnis: replay_pnl, mt5_correlation, eigene Metriken auf Replay-PnL.
"""

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from backtest.metrics import calculate_metrics
from utils.logger import get_logger

logger = get_logger(__name__)


def load_external_prices(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str = "15m",
) -> pd.DataFrame:
    """
    Lädt Preis-Daten von yfinance für den angegebenen Zeitraum.

    Args:
        symbol: z.B. "USDJPY=X" für USDJPY FX.
        start: Start-Zeitstempel (wird auf Datum gerundet).
        end: End-Zeitstempel (wird auf Datum gerundet).
        interval: z.B. "15m", "1h", "1d".

    Returns:
        DataFrame mit Spalten ['Open', 'High', 'Low', 'Close'] und DatetimeIndex.
    """
    # Nur das Datum verwenden, Uhrzeit komplett ignorieren
    start_date_str = start.date().strftime("%Y-%m-%d")
    end_date_str = end.date().strftime("%Y-%m-%d")

    logger.info(
        "Loading external prices from yfinance: symbol=%s, start=%s, end=%s, interval=%s",
        symbol,
        start_date_str,
        end_date_str,
        interval,
    )

    df = yf.download(
        symbol,
        start=start_date_str,
        end=end_date_str,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No price data returned from yfinance for {symbol}")

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
        raise ValueError(f"Downloaded data missing OHLC columns for {symbol}")

    df = df[["Open", "High", "Low", "Close"]]

    logger.info("Loaded %d bars from yfinance", len(df))

    return df


def _get_price_for_time(price_df: pd.DataFrame, t: pd.Timestamp) -> float:
    """
    Findet den Close-Preis derjenigen Bar, deren Timestamp <= t ist.

    Falls kein Timestamp <= t existiert, wird die früheste verfügbare Bar genutzt.
    """
    # Sicherstellen, dass der Index sortiert ist
    price_df = price_df.sort_index()

    # Alle Times <= t
    mask = price_df.index <= t
    if mask.any():
        return float(price_df.loc[mask].iloc[-1]["Close"])
    else:
        # Fallback: erste Bar
        return float(price_df.iloc[0]["Close"])


def replay_trades_on_prices(
    trades_df: pd.DataFrame,
    price_df: pd.DataFrame,
    lot_size: float = 100_000.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Rechnet MT5-Trades auf externen Preisen nach (Feed-Replay).

    Args:
        trades_df: DataFrame aus MT5TradesLoader mit mindestens:
            - entry_time (datetime)
            - exit_time (datetime)
            - direction (int: 1=long, -1=short)
            - volume (float)
            - pnl (float)  # MT5-PnL
        price_df: Preis-DataFrame mit DatetimeIndex und 'Close'-Spalte.
        lot_size: Konversionsfaktor für 1.0 Volumen (z.B. 100k bei FX).

    Returns:
        (replay_df, stats)
        replay_df: trades_df mit zusätzlicher Spalte 'pnl_replay'.
        stats: Dict mit:
            - mt5_correlation: Korrelation MT5-PnL vs. Replay-PnL
            - metrics_replay: Performance-Metriken auf Replay-PnL
    """
    required_cols = ["entry_time", "exit_time", "direction", "volume", "pnl"]
    missing = [c for c in required_cols if c not in trades_df.columns]
    if missing:
        raise ValueError(f"trades_df missing required columns: {missing}")

    df = trades_df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])

    # Für jedes Trade-Paar Entry/Exit-Preis aus externem Feed holen
    entry_prices = []
    exit_prices = []
    pnl_replay = []

    for _, row in df.iterrows():
        entry_t = row["entry_time"]
        exit_t = row["exit_time"]
        direction = int(row["direction"])
        volume = float(row["volume"])

        p_entry = _get_price_for_time(price_df, entry_t)
        p_exit = _get_price_for_time(price_df, exit_t)

        entry_prices.append(p_entry)
        exit_prices.append(p_exit)

        # PnL in Preis-Einheiten
        if direction == 1:  # long
            dp = p_exit - p_entry
        elif direction == -1:  # short
            dp = p_entry - p_exit
        else:
            dp = 0.0

        # Konversion in USD (vereinfachtes FX-Modell):
        # PnL ≈ dp * volume * lot_size
        pnl = dp * volume * lot_size
        pnl_replay.append(pnl)

    df["entry_price_replay"] = entry_prices
    df["exit_price_replay"] = exit_prices
    df["pnl_replay"] = pnl_replay

    # Korrelation MT5 vs. Replay-PnL
    mt5_pnl = df["pnl"].astype(float).values
    replay_pnl = df["pnl_replay"].astype(float).values

    if np.std(mt5_pnl) > 0 and np.std(replay_pnl) > 0:
        mt5_corr = float(np.corrcoef(mt5_pnl, replay_pnl)[0, 1])
    else:
        mt5_corr = 0.0

    # Metriken auf Replay-PnL
    replay_df_for_metrics = df.copy()
    replay_df_for_metrics["pnl"] = df["pnl_replay"]
    metrics_replay = calculate_metrics(replay_df_for_metrics, initial_capital=10_000.0)

    stats = {
        "mt5_correlation": mt5_corr,
        "metrics_replay": metrics_replay,
    }

    logger.info("Feed replay results:")
    logger.info("  mt5_correlation: %.2f", mt5_corr)
    logger.info(
        "  Replay Sharpe: %.2f, MaxDD: %.2f%%, TotalReturn: %.2f%%",
        metrics_replay["sharpe_ratio"],
        metrics_replay["max_drawdown"] * 100,
        metrics_replay["total_return"] * 100,
    )

    return df, stats
