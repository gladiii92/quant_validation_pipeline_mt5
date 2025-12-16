"""
Logging-Utility für die Pipeline.
Erklärt:
- Ein Logger ist ein "Protokollschreiber"
- Alles, was die Pipeline macht, wird geloggt
- Du siehst in der Console UND in logs/ Datei, was passiert
"""
import logging
import sys
from pathlib import Path
import yaml


def get_logger(
    name: str,
    log_file: str | None = None,
    level: int = logging.INFO
    ) -> logging.Logger:
    """
    Erstellt einen Logger mit Console und optional File Handler.

    Args:
        name: Name des Loggers (z.B. "data_pipeline")
        log_file: Pfad zur Log-Datei (optional)
        level: Log-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Logger-Objekt, das du verwenden kannst

    Beispiel:
        logger = get_logger("my_module", log_file="logs/my_module.log")
        logger.info("Starting process...")
        logger.warning("This is a warning!")
        logger.error("Something went wrong!")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Falls Logger schon Handler hat, nicht doppelt hinzufügen
    if logger.handlers:
        return logger
   
    # Format für Log-Nachrichten
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler (ausgabe in PowerShell)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (ausgabe in Datei)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Lädt die config.yaml Datei.

    Args:
        config_path: Pfad zur config.yaml

    Returns:
        Dictionary mit allen Konfigurationen
        
    Beispiel:
        cfg = load_config()
        print(cfg['data']['symbols']) # ['EURUSD', 'XAUUSD', ...]
    """

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config