"""Tests für Basis-Setup."""

import yaml
from utils.logger import get_logger
from utils.config import load_config

def test_config_loads():
    """Teste, dass config.yaml geladen wird."""

    cfg = load_config("config.yaml")

    # Diese Assertions prüfen, dass die Struktur stimmt
    assert "project" in cfg, "config.yaml muss 'project' Sektion haben"
    assert "data" in cfg, "config.yaml muss 'data' Sektion haben"
    assert "validation" in cfg, "config.yaml muss 'validation' Sektion haben"

    print(f"✅ Config geladen: {cfg['project']['name']}")


def test_logger_works():
    """Teste, dass Logger funktioniert."""

    logger = get_logger("test_logger")

    # Wenn keine Exception fliegt, ist es ok
    logger.info("Test log message")
    logger.warning("Test warning")

    print("✅ Logger funktioniert")

def test_sample_data(sample_price_data):
    """Teste die Sample-Daten Fixture."""

    df = sample_price_data

    assert len(df) == 252, "Sample sollte 252 Zeilen haben"
    assert 'close' in df.columns, "Sample sollte 'close' Spalte haben"
    
    print(f"✅ Sample data: {len(df)} Zeilen, {list(df.columns)}")