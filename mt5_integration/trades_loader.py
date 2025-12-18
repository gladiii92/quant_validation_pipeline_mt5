"""
MT5 Trades Loader für die aus RangeBreakoutUSDJPY.xlsx erzeugte merged-CSV.

- Liest die merged CSV (Orders + Trades)
- Nutzt den MT5-Trades-Block (Spalte 'Trade') als Trade-ID
- Baut pro Trade-ID eine abgeschlossene Position (Entry/Exit)
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

# Ordner eine Ebene nach oben zum Modulpfad hinzufügen
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


class MT5TradesLoader:
    """Lädt abgeschlossene Trades aus der merged MT5 CSV."""

    def __init__(self):
        """Initialisiert den Loader."""
        pass

    def load_trades(self, csv_path: str) -> pd.DataFrame:
        """
        Lädt Trades aus der merged MT5 CSV.

        Erwartete Spalten in der CSV:

        - Trade
        - Symbol_trade
        - Typ_trade       (buy/sell)
        - Richtung        (in/out)
        - Volumen_trade
        - Preis_trade
        - Gewinn
        - Kontostand
        - Eröffnungszeit  (Zeit aus Orders/Trades-Block)
        - Typ_order
        - Preis_order
        - S/L
        - T/P
        - Kommentar_trade
        - Kommentar_order

        Returns:
            DataFrame mit Spalten (pro MT5-Trade-ID):

            - ticket (int): MT5 Trade ID
            - entry_time (datetime): erste 'in'-Zeit (Fallback: erste Zeile)
            - exit_time (datetime): letzte 'out'-Zeit (Fallback: letzte Zeile)
            - symbol (str)
            - direction (int): 1 = long (buy), -1 = short (sell)
            - volume (float): Volumen der 'out'-Zeile (MT5-Tradegröße)
            - entry_price (float): Preis der ersten 'in'-Zeile
            - exit_price (float): Preis der letzten 'out'-Zeile
            - pnl (float): Gewinn (aus 'out'-Zeile)
            - commission (float): aktuell 0 (MT5-Kommission steckt separat)
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Trades file not found: {csv_path}")

        logger.info("Loading merged MT5 trades from %s...", csv_path)

        # Deine neue CSV ist mit Komma getrennt
        df = pd.read_csv(path, sep=",")

        logger.info("Columns in CSV: %s", list(df.columns))

        # Pflichtspalten prüfen
        required_cols = [
            "Trade",
            "Symbol_trade",
            "Typ_trade",
            "Richtung",
            "Volumen_trade",
            "Preis_trade",
            "Gewinn",
            "Eröffnungszeit",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        # Zeitspalte in datetime
        df["Eröffnungszeit"] = pd.to_datetime(df["Eröffnungszeit"])

        # Balances/sonstige Zeilen mit leerem Symbol rausfiltern
        df = df[~df["Symbol_trade"].isna()].copy()

        # Nach Trade-ID und Zeit sortieren
        df = df.sort_values(["Trade", "Eröffnungszeit"]).reset_index(drop=True)

        trades: List[Dict[str, Any]] = []

        # Gruppierung nach MT5-Trade-ID
        for trade_id, group in df.groupby("Trade"):
            group = group.sort_values("Eröffnungszeit")
            symbol = str(group["Symbol_trade"].iloc[0]).strip()

            # Entry-/Exit-Zeilen bestimmen
            entry_rows = group[group["Richtung"].str.lower() == "in"]
            exit_rows = group[group["Richtung"].str.lower() == "out"]

            if not entry_rows.empty:
                entry_row = entry_rows.iloc[0]
            else:
                # Fallback: erste Zeile der Gruppe
                entry_row = group.iloc[0]

            if not exit_rows.empty:
                exit_row = exit_rows.iloc[-1]
            else:
                # Wenn kein 'out' vorhanden ist, ignorieren wir den Trade
                logger.warning(
                    "Trade %s has no 'out' row; skipping (symbol=%s)", trade_id, symbol
                )
                continue

            # Zeiten
            entry_time = entry_row["Eröffnungszeit"]
            exit_time = exit_row["Eröffnungszeit"]

            # Preise
            entry_price = float(entry_row["Preis_trade"])
            exit_price = float(exit_row["Preis_trade"])

            # Volumen: aus der 'out'-Zeile (MT5 zeigt dort Tradegröße)
            volume = float(exit_row["Volumen_trade"])

            # PnL: Gewinn aus der 'out'-Zeile
            pnl_raw = exit_row["Gewinn"]
            pnl = float(pnl_raw) if not pd.isna(pnl_raw) else 0.0

            # Richtung: buy = long, sell = short, aus Entry-Zeile
            typ = str(entry_row["Typ_trade"]).strip().lower()
            if typ == "buy":
                direction = 1
            elif typ == "sell":
                direction = -1
            else:
                direction = 0

            trades.append(
                {
                    "ticket": int(trade_id),
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "symbol": symbol,
                    "direction": direction,
                    "volume": volume,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "commission": 0.0,
                }
            )

        result = pd.DataFrame(trades)

        if result.empty:
            logger.warning("No trades could be constructed from CSV")
        else:
            logger.info("Constructed %d trades (per MT5 Trade-ID)", len(result))
            logger.info(
                "Date range: %s to %s",
                result["entry_time"].min(),
                result["exit_time"].max(),
            )
            logger.info("Total PnL from MT5 trades: %.2f", result["pnl"].sum())

        return result

    def validate_trades(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validiert die geladenen Trades.

        Args:
            df: Trades DataFrame (standardisiertes Format)

        Returns:
            Dictionary mit Validierungsergebnissen
        """
        if df.empty:
            return {
                "total_trades": 0,
                "date_range": (None, None),
                "symbols": [],
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "has_missing_values": False,
            }

        results = {
            "total_trades": len(df),
            "date_range": (df["entry_time"].min(), df["exit_time"].max()),
            "symbols": df["symbol"].unique().tolist(),
            "total_pnl": df["pnl"].sum(),
            "win_rate": (df["pnl"] > 0).sum() / len(df),
            "avg_win": df[df["pnl"] > 0]["pnl"].mean()
            if (df["pnl"] > 0).any()
            else 0.0,
            "avg_loss": df[df["pnl"] <= 0]["pnl"].mean()
            if (df["pnl"] <= 0).any()
            else 0.0,
            "has_missing_values": df.isnull().any().any(),
        }

        logger.info("Trade Validation Results:")
        logger.info(" Total Trades: %d", results["total_trades"])
        logger.info(" Total PnL: %.2f", results["total_pnl"])
        logger.info(" Win Rate: %.2f%%", results["win_rate"] * 100)

        return results


if __name__ == "__main__":
    loader = MT5TradesLoader()
    csv_path = r"data\processed\RangeBreakoutUSDJPY__v4_USDJPY_M15_20240101_20250925_trades_merged.csv"
    trades_df = loader.load_trades(csv_path)
    validation = loader.validate_trades(trades_df)

    from pathlib import Path as _Path

    storage_path = r"storage\trades_mt5.parquet"
    _Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_parquet(storage_path)

    print(validation)
