"""
MT5 Trades Loader für die aus RangeBreakOut_USDJPY.xlsx erzeugte merged-CSV.

- Liest die merged CSV (Orders + Trades)
- Bildet aus in/out-Zeilen pro Trade ein standardisiertes, positionsbasiertes DataFrame
"""

import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd

# Ordner eine Ebene nach oben zum Modulpfad hinzufügen
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)


class MT5TradesLoader:
    """Lädt Trades aus der von mt5_xml_to_csv_converter erzeugten CSV."""

    def __init__(self):
        """Initialisiert den Loader."""
        pass

    def load_trades(self, csv_path: str) -> pd.DataFrame:
        """
        Lädt Trades aus der merged MT5 CSV.

        Erwartete Spalten in der CSV (aus deinem Converter):

        - Trade
        - Symbol_trade
        - Typ_trade
        - Richtung  (in / out)
        - Volumen_trade
        - Preis_trade
        - Gewinn
        - Kontostand
        - Eröffnungszeit
        - Typ_order
        - Preis_order
        - S/L
        - T/P
        - Kommentar_trade
        - Kommentar_order

        Returns:
            DataFrame mit Spalten (positionsbasiert):

            - ticket (int): Trade ID (dein "Trade")
            - entry_time (datetime): Einstiegszeit
            - exit_time (datetime): Ausstiegszeit
            - symbol (str)
            - direction (int): 1 = long, -1 = short
            - volume (float)
            - entry_price (float)
            - exit_price (float)
            - pnl (float)
            - commission (float, aktuell 0)
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Trades file not found: {csv_path}")

        logger.info(f"Loading merged MT5 trades from {csv_path}...")

        # CSV laden (Separator anpassen, wenn du ; verwendest)
        df = pd.read_csv(path, sep=";")

        # Balance-Zeile(n) entfernen (Trade == 1 mit Symbol NaN etc.)
        if "Symbol_trade" in df.columns:
            df = df[~((df["Trade"] == 1) & df["Symbol_trade"].isna())]

        # Sicherstellen, dass die nötigen Spalten existieren
        required_cols = [
            "Trade",
            "Symbol_trade",
            "Typ_trade",
            "Richtung",
            "Volumen_trade",
            "Preis_trade",
            "Gewinn",
            "Zeit",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        # Zeit-Spalte in datetime konvertieren
        df["Zeit"] = pd.to_datetime(df["Zeit"])

        # Entry-/Exit-Zeilen markieren
        # Annahme: Richtung == 'in' ist Entry, 'out' ist Exit
        df_entry = df[df["Richtung"] == "in"].copy()
        df_exit = df[df["Richtung"] == "out"].copy()

        # Umbenennen für klarere Merge-Keys
        df_entry = df_entry.rename(
            columns={
                "Zeit": "entry_time",
                "Preis_trade": "entry_price",
                "Volumen_trade": "entry_volume",
                "Typ_trade": "entry_type",
            }
        )
        df_exit = df_exit.rename(
            columns={
                "Zeit": "exit_time",
                "Preis_trade": "exit_price",
                "Volumen_trade": "exit_volume",
                "Typ_trade": "exit_type",
                "Gewinn": "pnl",
            }
        )

        # Entry- und Exit-Zeilen per Trade-ID mergen
        merged = pd.merge(
            df_entry,
            df_exit[
                [
                    "Trade",
                    "exit_time",
                    "exit_price",
                    "exit_volume",
                    "exit_type",
                    "pnl",
                ]
            ],
            on="Trade",
            how="left",
            suffixes=("_entry", "_exit"),
        )

        # Tradingrichtung: aus entry_type (buy/sell)
        # buy -> long (1), sell -> short (-1)
        merged["direction"] = merged["entry_type"].map(
            {"buy": 1, "sell": -1}
        )

        # Positionsvolumen: wir nehmen das Volumen der Entry-Zeile
        merged["volume"] = merged["entry_volume"]

        # Standardisierte Spalten zusammenbauen
        result = pd.DataFrame(
            {
                "ticket": merged["Trade"].astype(int),
                "entry_time": merged["entry_time"],
                "exit_time": merged["exit_time"],
                "symbol": merged["Symbol_trade"],
                "direction": merged["direction"],
                "volume": merged["volume"],
                "entry_price": merged["entry_price"],
                "exit_price": merged["exit_price"],
                "pnl": merged["pnl"],
                # Kommission aktuell nicht vorhanden -> 0
                "commission": 0.0,
            }
        )

        # Optional: Zeilen ohne Exit (z.B. noch offene Trades) entfernen
        result = result.dropna(subset=["exit_time", "exit_price", "pnl"])

        logger.info(f"✅ Loaded {len(result)} closed positions from {csv_path}")
        logger.info(
            f"Date range: {result['entry_time'].min()} to {result['exit_time'].max()}"
        )

        return result

    def validate_trades(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validiert die geladenen Trades.

        Args:
            df: Trades DataFrame (standardisiertes Format)

        Returns:
            Dictionary mit Validierungsergebnissen
        """
        results = {
            "total_trades": len(df),
            "date_range": (df["entry_time"].min(), df["exit_time"].max()),
            "symbols": df["symbol"].unique().tolist(),
            "total_pnl": df["pnl"].sum(),
            "win_rate": (df["pnl"] > 0).sum() / len(df) if len(df) > 0 else 0.0,
            "avg_win": df[df["pnl"] > 0]["pnl"].mean()
            if (df["pnl"] > 0).any()
            else 0.0,
            "avg_loss": df[df["pnl"] <= 0]["pnl"].mean()
            if (df["pnl"] <= 0).any()
            else 0.0,
            "has_missing_values": df.isnull().any().any(),
        }

        logger.info("Trade Validation Results:")
        logger.info(f" Total Trades: {results['total_trades']}")
        logger.info(f" Total PnL: ${results['total_pnl']:.2f}")
        logger.info(f" Win Rate: {results['win_rate']:.2%}")

        return results


if __name__ == "__main__":
    loader = MT5TradesLoader()

    # Pfad zu deiner merged CSV aus dem Converter
    csv_path = r"G:\DAVID\Desktop\GitHub\quant_validation_pipeline_mt5\data\processed\RangeBreakOut_USDJPY_trades_merged.csv"

    trades_df = loader.load_trades(csv_path)
    validation = loader.validate_trades(trades_df)

    # Optional: Speichern als Parquet
    storage_path = (
        r"G:\DAVID\Desktop\GitHub\quant_validation_pipeline_mt5\storage\trades.parquet"
    )
    Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_parquet(storage_path)

    print(validation)
