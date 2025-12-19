import re
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd


def extract_header_info(xlsx_path: str) -> Dict[str, Any]:
    """
    Liest den Kopf des MT5-Strategietesterberichts aus und extrahiert:
    - strategy_name (Expertenprogramm)
    - symbol
    - timeframe
    - date_from, date_to (aus Periode)
    """
    path = Path(xlsx_path)
    df = pd.read_excel(path, sheet_name=0, header=None)

    strategy_name: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None

    for i in range(min(80, len(df))):
        row = df.iloc[i].astype(str).tolist()
        line = " ".join(cell for cell in row if cell != "nan").strip()
        if not line:
            continue

        if "Expertenprogramm" in line and ":" in line:
            value = line.split(":", 1)[1].strip()
            strategy_name = value

        elif line.startswith("Symbol:"):
            value = line.split(":", 1)[1].strip()
            symbol = value

        elif line.startswith("Periode:"):
            value = line.split(":", 1)[1].strip()
            m = re.match(
                r"([A-Z0-9]+)\s*\((\d{4}\.\d{2}\.\d{2})\s*-\s*(\d{4}\.\d{2}\.\d{2})\)",
                value,
            )
            if m:
                timeframe = m.group(1)
                date_from = m.group(2).replace(".", "")
                date_to = m.group(3).replace(".", "")

    if strategy_name is None:
        strategy_name = path.stem
    if symbol is None:
        symbol = "UNKNOWN"
    if timeframe is None:
        timeframe = "TF"
    if date_from is None or date_to is None:
        date_from = "START"
        date_to = "END"

    strategy_clean = strategy_name.strip()
    strategy_clean = strategy_clean.replace(" ", "_")
    strategy_clean = strategy_clean.replace("-", "_")

    # NEU: generischer Strategie-Key, z.B. RangeBreakoutUSDJPY_v4
    # dazu nur den Teil vor Symbol/Timeframe nehmen
    # Angenommen Titel: "RangeBreakoutUSDJPY -v4 USDJPY,M15 2024.01.01-2025.09.25"
    # extract_header_info kennt den Titel nicht direkt, aber strategy_name_raw reicht meist
    strategy_key = strategy_clean
    # doppelte Unterstriche zu einfachen
    strategy_key = strategy_key.replace("__", "_")

    return {
        "strategy_name_raw": strategy_name,
        "strategy_name": strategy_clean,
        "strategy_key": strategy_key,  # NEU
        "symbol": symbol,
        "timeframe": timeframe,
        "date_from": date_from,
        "date_to": date_to,
    }


def build_output_path(header: Dict[str, Any], base_dir: str = "data/processed") -> Path:
    """
    Baut den Zielpfad für die CSV aus den Header-Infos.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    strategy = header["strategy_name"]
    symbol = header["symbol"]
    tf = header["timeframe"]
    d_from = header["date_from"]
    d_to = header["date_to"]

    filename = f"{strategy}_{symbol}_{tf}_{d_from}_{d_to}_trades_merged.csv"
    return base / filename


def convert_mt5_xlsx_to_csv(xlsx_path: str, output_csv: str) -> Path:
    """
    Konvertiert eine MT5-Strategietester-XLSX:
    - Orders-Block und Trades-Block werden gemerged
    - CSV wird unter output_csv gespeichert
    """
    inputfile = Path(xlsx_path)
    outputfile = Path(output_csv)

    df = pd.read_excel(inputfile, sheet_name=0, header=None)

    # Orders-Block
    orders_title_idx = df[df.iloc[:, 0] == "Orders"].index
    if len(orders_title_idx) == 0:
        raise ValueError(f"Orders-Block nicht gefunden in {inputfile}.")
    orders_title_idx = orders_title_idx[0]
    orders_start_idx = orders_title_idx + 1

    trades_title_idx = df[df.iloc[:, 0] == "Trades"].index
    if len(trades_title_idx) == 0:
        raise ValueError(f"Trades-Block nicht gefunden in {inputfile}.")
    trades_title_idx = trades_title_idx[0]

    orders_df_raw = df.iloc[orders_start_idx:trades_title_idx].copy()

    orders_headers_full = [
        "Eröffnungszeit",
        "Auftrag",
        "Symbol",
        "Typ",
        "Volumen",
        "Leer1",
        "Preis",
        "S/L",
        "T/P",
        "Zeit",
        "Leer2",
        "Status",
        "Kommentar",
    ]

    orders_df_raw = orders_df_raw.iloc[:, : len(orders_headers_full)]
    orders_df_raw.columns = orders_headers_full

    orders_df = orders_df_raw[
        [
            "Eröffnungszeit",
            "Auftrag",
            "Symbol",
            "Typ",
            "Volumen",
            "Preis",
            "S/L",
            "T/P",
            "Zeit",
            "Status",
            "Kommentar",
        ]
    ].copy()
    orders_df = orders_df.dropna(how="all")

    # Trades-Block
    trades_start_idx = trades_title_idx + 1
    trades_df_raw = df.iloc[trades_start_idx:].copy()

    trades_headers = [
        "Zeit",
        "Trade",
        "Symbol",
        "Typ",
        "Richtung",
        "Volumen",
        "Preis",
        "Auftrag",
        "Kommission",
        "Swap",
        "Gewinn",
        "Kontostand",
        "Kommentar",
    ]
    trades_df_raw = trades_df_raw.iloc[1:, : len(trades_headers)]
    trades_df_raw.columns = trades_headers

    trades_df = trades_df_raw.dropna(how="all")

    merged = pd.merge(
        trades_df,
        orders_df,
        left_on="Trade",
        right_on="Auftrag",
        how="left",
        suffixes=("_trade", "_order"),
    )

    output_cols = [
        "Zeit",           # aus Trades-Block
        "Trade",          # Trade-ID
        "Symbol_trade",   # Symbol aus Trades
        "Typ_trade",      # Typ aus Trades
        "Richtung",
        "Volumen_trade",  # Volumen aus Trades
        "Preis_trade",    # Preis aus Trades
        "Auftrag",        # Order-ID (gleich wie Trade-Verknüpfung)
        "Gewinn",
        "Kontostand",
        "Eröffnungszeit", # aus Orders
        "Typ_order",      # Typ aus Orders
        "Preis_order",    # Preis aus Orders
        "S/L",
        "T/P",
        "Kommentar_trade",
        "Kommentar_order"
    ]
    
    available_cols = [c for c in output_cols if c in merged.columns]
    merged = merged[available_cols].copy()

    outputfile.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(outputfile, index=False, sep=",", encoding="utf-8")
    print(f"{len(merged)} Zeilen gespeichert in {outputfile}")
    return outputfile


def auto_convert_raw(xlsx_path: str, base_output_dir: str = "data/processed") -> Path:
    """
    High-Level für einzelne Datei:
    - Header lesen
    - Output-Pfad bauen
    - CSV erzeugen (oder überschreiben)
    """
    header = extract_header_info(xlsx_path)
    output_path = build_output_path(header, base_dir=base_output_dir)
    csv_path = convert_mt5_xlsx_to_csv(xlsx_path, str(output_path))

    print("Strategie:", header["strategy_name_raw"])
    print("Symbol:", header["symbol"])
    print("Periode:", header["timeframe"], header["date_from"], "-", header["date_to"])
    print("CSV:", csv_path)
    return csv_path


def batch_convert_raw(
    raw_dir: str = "data/raw",
    base_output_dir: str = "processed",
    overwrite: bool = False,
) -> None:
    """
    Batch-Mode:
    - Nimmt alle .xlsx in raw_dir
    - Für jede Datei:
      * Header lesen → Output-Pfad bestimmen
      * Wenn CSV existiert und neuer/gleich alt: skip (außer overwrite=True)
    """
    raw_path = Path(raw_dir)
    xlsx_files = sorted(raw_path.glob("*.xlsx"))

    if not xlsx_files:
        print(f"Keine .xlsx in {raw_path} gefunden.")
        return

    for xlsx_file in xlsx_files:
        header = extract_header_info(str(xlsx_file))
        out_path = build_output_path(header, base_dir=base_output_dir)

        # Skip-Logik
        if out_path.exists() and not overwrite:
            if out_path.stat().st_mtime >= xlsx_file.stat().st_mtime:
                print(f"[SKIP] {xlsx_file.name} → {out_path.name} (bereits aktuell)")
                continue

        print(f"[CONVERT] {xlsx_file.name} → {out_path.name}")
        convert_mt5_xlsx_to_csv(str(xlsx_file), str(out_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MT5 Strategietester XLSX → auto-benannte merged CSV (Single oder Batch)"
    )
    parser.add_argument(
        "xlsx_file",
        type=str,
        nargs="?",
        help="Pfad zur einzelnen MT5-Strategietester-Exceldatei (.xlsx). "
             "Wenn leer → Batch-Mode über alle .xlsx in data/raw.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Verzeichnis mit MT5-XLSX-Dateien (für Batch-Mode, default: data/raw).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed",
        help="Basis-Output-Verzeichnis für die CSV (default: data/processed).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Existierende CSVs trotzdem neu schreiben.",
    )
    args = parser.parse_args()

    if args.xlsx_file:
        auto_convert_raw(args.xlsx_file, base_output_dir=args.out_dir)
    else:
        batch_convert_raw(raw_dir=args.raw_dir, base_output_dir=args.out_dir, overwrite=args.overwrite)
