# organize_raw.py

from pathlib import Path
from typing import Dict
import shutil

from mt5_xml_to_csv_converter import extract_header_info
from validation.multi_asset_summary import extract_strategy_name_from_optimizer


def normalize_strategy_key(key: str) -> str:
    """Normalisiert Strategienamen für Ordner:
    - Leerzeichen und Minus -> Unterstrich
    - doppelte Unterstriche -> einfache
    """
    key = key.replace(" ", "_").replace("-", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key


def organize_raw(raw_dir: str = "data/raw") -> None:
    """
    Ordnet alle XLSX- und Optimizer-XML-Dateien in strategie-spezifische Unterordner ein.

    Struktur danach:
    data/raw/<strategy_key>/
        *.xlsx
        ReportOptimizer-*.xml
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    # 1) XLSX-Dateien nach Strategie gruppieren
    strategy_to_files: Dict[str, Dict[str, list]] = {}

    for xlsx_file in raw_path.glob("*.xlsx"):
        header = extract_header_info(str(xlsx_file))
        raw_key = header.get("strategy_key") or header["strategy_name"]
        strategy_key = normalize_strategy_key(raw_key)

        strategy_to_files.setdefault(strategy_key, {"xlsx": [], "xml": []})
        strategy_to_files[strategy_key]["xlsx"].append(xlsx_file)

    # 2) Optimizer-XMLs nach Strategie zuordnen
    for xml_file in raw_path.glob("ReportOptimizer-*.xml"):
        opt_strategy = extract_strategy_name_from_optimizer(str(xml_file))
        opt_key = normalize_strategy_key(opt_strategy)

        # Bestes Match in strategy_to_files suchen
        matched_key = None
        for strategy_key in strategy_to_files.keys():
            if strategy_key in opt_key or opt_key in strategy_key:
                matched_key = strategy_key
                break

        if matched_key is None:
            matched_key = opt_key
            strategy_to_files.setdefault(matched_key, {"xlsx": [], "xml": []})

        strategy_to_files[matched_key]["xml"].append(xml_file)

    # 3) Ordner erstellen und Dateien verschieben
    for strategy_key, files in strategy_to_files.items():
        strat_folder = raw_path / strategy_key
        strat_folder.mkdir(parents=True, exist_ok=True)

        for f in files["xlsx"]:
            target = strat_folder / f.name
            if f.resolve() != target.resolve():
                shutil.move(str(f), str(target))

        for f in files["xml"]:
            target = strat_folder / f.name
            if f.resolve() != target.resolve():
                shutil.move(str(f), str(target))

        print(f"Organized strategy '{strategy_key}' into {strat_folder}")


if __name__ == "__main__":
    organize_raw()
