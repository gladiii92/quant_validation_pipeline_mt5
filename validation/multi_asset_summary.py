# validation/multi_asset_summary.py

from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, Any
import re

import pandas as pd

def extract_strategy_name_from_optimizer(xml_path: str) -> str:
    """
    Liest den <Title>-Tag der MT5-Optimizer-XML und extrahiert den Strategienamen.

    Beispiel-Titel:
    "RangeBreakoutUSDJPY -v4 USDJPY,M15 2024.01.01-2025.09.25"

    Rückgabe:
        "RangeBreakoutUSDJPY_v4" (normalisiert: Spaces/Minus -> Unterstrich)
    """
    path = Path(xml_path)
    tree = ET.parse(path)
    root = tree.getroot()

    # Namespace für Office/Excel-XML
    ns_office = {"o": "urn:schemas-microsoft-com:office:office"}

    # <DocumentProperties><Title>...</Title>
    title_elem = root.find(".//o:DocumentProperties/o:Title", ns_office)
    if title_elem is None or not title_elem.text:
        # Fallback: Dateiname ohne Suffix
        return path.stem

    title = title_elem.text.strip()

    # Heuristik: alles bis zum Symbol/Timeframe-Teil, z.B. "RangeBreakoutUSDJPY -v4"
    # Trenne am ersten Symbol/Periode-Block (z. B. " USDJPY," oder " EURUSD,")
    # Einfacher: nimm den Teil bis zum ersten " <SYMBOL>," oder " <SYMBOL> ".
    # Robustere Variante: nur den ersten Block vor dem ersten Symbol-Trenner.
    # Für deine Zwecke reicht: bis zum ersten " USDJPY" / Komma / Zeitangabe.
    # Wir nehmen den Teil bis zum ersten " USD" / " EUR" / " GBP" / " JPY" etc. wäre zu kompliziert;
    # hier einfacher: der Strategiename steht vor dem ersten Symbol-Token, also trennen an " -vX".
    # Du hast meist "Name -v4 ..." → wir extrahieren den Prefix inklusive v-Teil.
    m = re.match(r"(.+?)(\s+[-–]v\d+)?\s", title)
    if m:
        base = m.group(1)
        suffix = m.group(2) or ""
        strategy_raw = (base + suffix).strip()
    else:
        strategy_raw = title

    # Normalisieren: Spaces und Minus zu Unterstrich
    strategy_clean = strategy_raw.replace(" ", "_").replace("-", "_")
    return strategy_clean


def parse_mt5_optimizer_xml(xml_path: str) -> pd.DataFrame:
    """
    Parst den MT5-Optimizer-Export (Tester Optimizator Results) im XML/Excel-XML-Format.

    Erwartete Header:
    Symbol, Pass, Result, Profit, Expected Payoff, Profit Factor,
    Recovery Factor, Sharpe Ratio, Custom, Equity DD %, Trades
    """
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"Optimizer XML not found: {xml_path}")

    ns = {
        "ss": "urn:schemas-microsoft-com:office:spreadsheet",
    }

    tree = ET.parse(path)
    root = tree.getroot()

    # Worksheet "Tester Optimizator Results"
    ws = root.find(".//ss:Worksheet[@ss:Name='Tester Optimizator Results']", ns)
    if ws is None:
        raise ValueError("Worksheet 'Tester Optimizator Results' not found in XML.")

    table = ws.find("ss:Table", ns)
    if table is None:
        raise ValueError("Table not found in 'Tester Optimizator Results' worksheet.")

    rows = table.findall("ss:Row", ns)
    if not rows or len(rows) < 2:
        return pd.DataFrame()

    data_rows = rows[1:]  # erste Zeile ist Header

    parsed = []
    for row in data_rows:
        cells = row.findall("ss:Cell", ns)
        if not cells:
            continue
        values = []
        for c in cells:
            d = c.find("ss:Data", ns)
            values.append(d.text if d is not None else None)

        # Leere Zeilen überspringen
        if not values or values[0] is None:
            continue

        try:
            parsed.append(
                {
                    "symbol": values[0],
                    "pass": int(values[1]),
                    "result": float(values[2]),
                    "profit": float(values[3]),
                    "expected_payoff": float(values[4]),
                    "profit_factor": float(values[5]),
                    "recovery_factor": float(values[6]),
                    "sharpe": float(values[7]),
                    "custom": float(values[8]) if values[8] is not None else 0.0,
                    "equity_dd_pct": float(values[9]),
                    "trades": int(values[10]),
                }
            )
        except (IndexError, ValueError):
            # Falls MetaQuotes etwas am Format ändert, Zeile überspringen
            continue

    return pd.DataFrame(parsed)


def compute_multi_asset_stats(df: pd.DataFrame, sharpe_threshold: float = 1.0) -> Dict[str, Any]:
    """
    Berechnet Meta-Kennzahlen über alle Symbole.
    """
    if df.empty:
        return {
            "hit_rate": 0.0,
            "n_symbols": 0,
            "n_symbols_pass": 0,
            "sharpe_threshold": sharpe_threshold,
        }

    mask = df["sharpe"] > sharpe_threshold
    n_pass = int(mask.sum())
    n_total = int(len(df))
    hit_rate = n_pass / n_total if n_total > 0 else 0.0

    return {
        "hit_rate": hit_rate,
        "n_symbols": n_total,
        "n_symbols_pass": n_pass,
        "sharpe_threshold": sharpe_threshold,
    }


def load_and_score_optimizer(xml_path: str, sharpe_threshold: float = 1.0) -> Dict[str, Any]:
    """
    High-Level:
    - XML laden
    - Tabelle parsen
    - Multi-Asset-Hit-Rate + Details zurückgeben
    """
    df = parse_mt5_optimizer_xml(xml_path)
    stats = compute_multi_asset_stats(df, sharpe_threshold=sharpe_threshold)
    stats["details"] = df.to_dict(orient="records")
    return stats


if __name__ == "__main__":
    # Kleiner CLI-Test
    import argparse

    parser = argparse.ArgumentParser(description="Parse MT5 Optimizer XML and compute multi-asset stats.")
    parser.add_argument("xml_file", type=str, help="Path to MT5 optimizer XML file")
    parser.add_argument("--sharpe-threshold", type=float, default=1.0)
    args = parser.parse_args()

    info = load_and_score_optimizer(args.xml_file, sharpe_threshold=args.sharpe_threshold)
    print(f"Symbols total: {info['n_symbols']}")
    print(f"Symbols pass:  {info['n_symbols_pass']} (Sharpe > {info['sharpe_threshold']})")
    print(f"Hit rate:      {info['hit_rate']:.1%}")
