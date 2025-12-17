"""Tests für MT5-Integration mit der merged MT5-CSV."""

import pytest
from pathlib import Path

from mt5_integration.trades_loader import MT5TradesLoader


@pytest.fixture
def sample_merged_trades_csv(tmp_path: Path) -> str:
    """Erstellt eine Test-CSV im Format der merged MT5-Trades."""
    csv_data = """Trade;Symbol_trade;Typ_trade;Richtung;Volumen_trade;Preis_trade;Gewinn;Kontostand;Zeit;Eröffnungszeit;Typ_order;Preis_order;S/L;T/P;Kommentar_trade;Kommentar_order
1;USDJPY;buy;in;0.95;147.09;0;100000;2024-02-01 07:00:00;2024-02-01 07:00:00;buy stop;147.089;146.63;151.985;TimeRangeEAPending;TimeRangeEAPending
1;USDJPY;sell;out;0.95;146.63;-298.03;99691.97;2024-02-01 15:35:39;2024-02-01 07:00:00;sell;0;;;sl 146.630;TimeRangeEAPending
2;USDJPY;buy;in;1.00;148.00;0;99691.97;2024-02-02 07:00:00;2024-02-02 07:00:00;buy stop;148.000;147.50;152.000;TimeRangeEAPending;TimeRangeEAPending
2;USDJPY;sell;out;1.00;149.00;500.00;100191.97;2024-02-02 21:20:00;2024-02-02 07:00:00;sell;0;;;tp;TimeRangeEAPending
"""
    csv_path = tmp_path / "test_merged_trades.csv"
    csv_path.write_text(csv_data, encoding="utf-8")
    return str(csv_path)


def test_load_trades(sample_merged_trades_csv: str):
    """Teste, dass der Loader aus merged-CSV positionsbasierte Trades baut."""
    loader = MT5TradesLoader()

    df = loader.load_trades(sample_merged_trades_csv)

    # Es gibt 2 Tickets (1 und 2) mit jeweils in/out -> 2 geschlossene Positionen
    assert len(df) == 2, "Sollte 2 geschlossene Trades haben"

    # Pflichtspalten
    for col in [
        "ticket",
        "entry_time",
        "exit_time",
        "symbol",
        "direction",
        "volume",
        "entry_price",
        "exit_price",
        "pnl",
        "commission",
    ]:
        assert col in df.columns, f"Sollte Spalte {col} haben"

    # Ticket 1: loser Trade
    t1 = df[df["ticket"] == 1].iloc[0]
    assert t1["symbol"] == "USDJPY"
    # buy -> long
    assert t1["direction"] == 1
    assert pytest.approx(t1["entry_price"], rel=1e-6) == 147.09
    assert pytest.approx(t1["exit_price"], rel=1e-6) == 146.63
    assert pytest.approx(t1["pnl"], rel=1e-6) == -298.03

    # Ticket 2: Gewinner
    t2 = df[df["ticket"] == 2].iloc[0]
    assert t2["direction"] == 1
    assert pytest.approx(t2["pnl"], rel=1e-6) == 500.0


def test_validate_trades(sample_merged_trades_csv: str):
    """Teste Validierung der geladenen Trades."""
    loader = MT5TradesLoader()
    df = loader.load_trades(sample_merged_trades_csv)

    validation = loader.validate_trades(df)

    assert validation["total_trades"] == 2
    # -298.03 + 500.00 = 201.97
    assert pytest.approx(validation["total_pnl"], rel=1e-6) == 201.97
    assert 0 <= validation["win_rate"] <= 1
