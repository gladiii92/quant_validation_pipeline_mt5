"""
MT5 Trades Loader für die aus RangeBreakOut_USDJPY.xlsx erzeugte merged-CSV.

- Liest die merged CSV (Orders + Trades)
- Bildet aus in/out-Zeilen pro Symbol & Richtung eine Positionsliste
- Gibt ein standardisiertes DataFrame zurück
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

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
        Lädt Trades aus der merged MT5 CSV und baut abgeschlossene Positionen.

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
            DataFrame mit Spalten (positionsbasiert):

            - ticket (int): Laufende Positions-ID (nicht MT5-Trade-ID)
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

        # Deine CSV ist mit ; getrennt
        df = pd.read_csv(path, sep=";")

        # Balance-Zeilen entfernen (Trade=1, Symbol leer/balance)
        if "Symbol_trade" in df.columns:
            df = df[~((df["Trade"] == 1) & df["Symbol_trade"].isna())]

        # Pflichtspalten prüfen – ohne 'Zeit'
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

        # Zeitspalte: Eröffnungszeit in datetime
        df["Eröffnungszeit"] = pd.to_datetime(df["Eröffnungszeit"])

        # Nach Eröffnungszeit sortieren, damit wir Entries/Exits in chronologischer Reihenfolge sehen
        df = df.sort_values("Eröffnungszeit").reset_index(drop=True)

        positions: List[Dict[str, Any]] = []
        open_positions: List[Dict[str, Any]] = []
        next_ticket = 1

        # Hilfsfunktion: long/short aus Typ_trade bestimmen
        def side_from_type(typ: str) -> int:
            typ = str(typ).lower()
            if typ == "buy":
                return 1
            elif typ == "sell":
                return -1
            else:
                return 0

        for _, row in df.iterrows():
            richtung = str(row["Richtung"]).strip().lower()
            typ_trade = str(row["Typ_trade"]).strip().lower()
            symbol = str(row["Symbol_trade"]).strip()

            # Manche Zeilen sind leer / nur Balance etc.
            if symbol == "" or symbol.lower() == "balance":
                continue

            vol = float(row["Volumen_trade"])
            price = float(row["Preis_trade"])
            time = row["Eröffnungszeit"]
            pnl = float(row["Gewinn"]) if not pd.isna(row["Gewinn"]) else 0.0

            side = side_from_type(typ_trade)
            if side == 0:
                # Unbekannter Typ, überspringen
                continue

            if richtung == "in":
                # Neue offene Position anlegen
                open_positions.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "volume": vol,
                        "entry_time": time,
                        "entry_price": price,
                        "mt5_trade_id": int(row["Trade"]),
                    }
                )
            elif richtung == "out":
                # Passende offene Position suchen:
                # gleiche Richtung (side), gleiches Symbol, Volumen-Toleranz
                match_index = None
                for i in range(len(open_positions) - 1, -1, -1):
                    pos = open_positions[i]
                    if (
                        pos["symbol"] == symbol
                        and pos["side"] == side
                        and abs(pos["volume"] - vol) < 1e-6
                    ):
                        match_index = i
                        break

                if match_index is None:
                    # Kein passender Entry gefunden -> Warnung, aber nicht crashen
                    logger.warning(
                        "No matching open position for exit: Trade=%s Symbol=%s side=%s vol=%s time=%s",
                        row["Trade"],
                        symbol,
                        side,
                        vol,
                        time,
                    )
                    continue

                entry_pos = open_positions.pop(match_index)

                positions.append(
                    {
                        "ticket": next_ticket,
                        "entry_time": entry_pos["entry_time"],
                        "exit_time": time,
                        "symbol": symbol,
                        "direction": side,
                        "volume": vol,
                        "entry_price": entry_pos["entry_price"],
                        "exit_price": price,
                        "pnl": pnl,
                        "commission": 0.0,  # kannst du später aus Kommission-Spalte ziehen
                        "mt5_entry_trade_id": entry_pos["mt5_trade_id"],
                        "mt5_exit_trade_id": int(row["Trade"]),
                    }
                )
                next_ticket += 1
            else:
                # unbekannte Richtung -> ignorieren
                continue

        result = pd.DataFrame(positions)

        if result.empty:
            logger.warning("No closed positions could be constructed from CSV")
        else:
            logger.info("Constructed %d closed positions", len(result))
            logger.info(
                "Date range: %s to %s",
                result["entry_time"].min(),
                result["exit_time"].max(),
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
        logger.info(" Total PnL: $%.2f", results["total_pnl"])
        logger.info(" Win Rate: %.2f%%", results["win_rate"] * 100)

        return results


if __name__ == "__main__":
    loader = MT5TradesLoader()

    # Testlauf mit deiner Datei (Pfad ggf. anpassen)
    csv_path = r"G:\DAVID\Desktop\GitHub\quant_validation_pipeline_mt5\data\processed\RangeBreakOut_USDJPY_trades_merged.csv"

    trades_df = loader.load_trades(csv_path)
    validation = loader.validate_trades(trades_df)

    from pathlib import Path

    storage_path = (
        r"G:\DAVID\Desktop\GitHub\quant_validation_pipeline_mt5\storage\trades.parquet"
    )
    Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_parquet(storage_path)

    print(validation)
