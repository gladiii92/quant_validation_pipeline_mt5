"""
MT5 Trades Loader
Erklärt:
- Liest MT5 CSV mit Trades
- Normalisiert Spalten und Datentypen
- Gibt ein standardisiertes DataFrame zurück
"""

import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Any

# Ordner eine Ebene nach oben zum Modulpfad hinzufügen
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)


class MT5TradesLoader:
    """Lädt Trades aus MT5 CSV-Export."""

    def __init__(self):
        """Initialisiert den Loader."""
        pass

    def load_trades(self, csv_path: str) -> pd.DataFrame:

        """
        Lädt Trades aus MT5 CSV.

        Args:
            csv_path: Pfad zur CSV-Datei (z.B. "data/raw/trades/strategy_x.csv")

        Returns:
            Normalisiertes DataFrame mit Spalten:
            - ticket (int): Trade ID
            - entry_time (datetime): Eröffnungszeit
            - symbol (str): z.B. "EURUSD"
            - direction (int): 1 = long, -1 = short
            - volume (float): Trade-Größe
            - entry_price (float): Eröffnungspreis
            - exit_time (datetime): Schließungszeit
            - exit_price (float): Schließungspreis
            - pnl (float): Gewinn/Verlust in USD
            - commission (float): Gebühren
        """

        # Überprüfe, dass Datei existiert
        path = Path(csv_path)

        if not path.exists():
            raise FileNotFoundError(f"Trades file not found: {csv_path}")
        
        logger.info(f"Loading trades from {csv_path}...")

        # Lade CSV
        df = pd.read_csv(path)

        # Normalisiere Spalten (häufige MT5 Spalten-Namen)
        column_mapping = {
            'Ticket': 'ticket',
            'OpenTime': 'entry_time',
            'CloseTime': 'exit_time',
            'Type': 'direction',
            'Symbol': 'symbol',
            'Volume': 'volume',
            'OpenPrice': 'entry_price',
            'ClosePrice': 'exit_price',
            'Profit': 'pnl',
            'Commission': 'commission',
        }

        # Wende Mapping an (nur existierende Spalten)
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Konvertiere Zeitstrings zu datetime
        for col in ['entry_time', 'exit_time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        # Type (0 = buy/long, 1 = sell/short) → Direction (1, -1)
        if 'direction' in df.columns:
            df['direction'] = df['direction'].map({0: 1, 1: -1})

        # Berechne fehlende Spalten, falls nötig
        if 'pnl' not in df.columns and 'entry_price' in df.columns:
            # PnL = (exit_price - entry_price) * volume * 10000 (für Pips)
            df['pnl'] = (df['exit_price'] - df['entry_price']) * df['volume'] * 10000

        logger.info(f"✅ Loaded {len(df)} trades from {csv_path}")
        logger.info(f"Date range: {df['entry_time'].min()} to {df['exit_time'].max()}")

        return df
    
    def validate_trades(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validiert die geladenen Trades.

        Args:
            df: Trades DataFrame


        Returns:
        Dictionary mit Validierungsergebnissen
        """
        
        results = {
            'total_trades': len(df),
            'date_range': (df['entry_time'].min(), df['exit_time'].max()),
            'symbols': df['symbol'].unique().tolist(),
            'total_pnl': df['pnl'].sum(),
            'win_rate': (df['pnl'] > 0).sum() / len(df),
            'avg_win': df[df['pnl'] > 0]['pnl'].mean(),
            'avg_loss': df[df['pnl'] <= 0]['pnl'].mean(),
            'has_missing_values': df.isnull().any().any(),
        }

        logger.info("Trade Validation Results:")
        logger.info(f" Total Trades: {results['total_trades']}")
        logger.info(f" Total PnL: ${results['total_pnl']:.2f}")
        logger.info(f" Win Rate: {results['win_rate']:.2%}")

        return results
    
# Beispiel-Nutzung-mit 1. Skript
if __name__ == "__main__":
    loader = MT5TradesLoader()

    # Absoluter Pfad zu deiner CSV
    csv_path = r"G:\DAVID\Desktop\GitHub\quant_validation_pipeline_mt5\data\processed\RangeBreakOut_USDJPY_trades.csv"

    # Lade Trades
    trades_df = loader.load_trades(csv_path)

    # Validiere Trades
    validation = loader.validate_trades(trades_df)

    # Optional: Speichere DataFrame für später
    trades_df.to_parquet(r"G:\DAVID\Desktop\GitHub\quant_validation_pipeline_mt5\storage\trades.parquet")
    
    print(validation)