"""
Pytest Fixtures für Tests.

Erklärt:
- Fixtures sind "Setup-Funktionen" für Tests
- Sie werden automatisch von pytest aufgerufen
- Damit kannst du Test-Daten erzeugen, ohne sie jedes Mal zu schreiben
"""

import pytest
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

@pytest.fixture
def config():

    """Lädt config.yaml für Tests."""
    with open("config.yaml", 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    
@pytest.fixture
def sample_price_data():
    """
    Erzeugt künstliche Preisdaten zum Testen.

    Das ist wichtig: In echten Tests willst du nicht mit echten Daten arbeiten, weil:
        1. Schneller
        2. Reproduzierbar
        3. Keine Abhängigkeit von externen APIs
    """

    np.random.seed(42) # Damit ist es immer gleich

    dates = pd.date_range('2024-01-01', periods=252, freq='D')

    # Simuliere tägliche Returns (ca. 2% Volatilität)
    returns = np.random.randn(252) * 0.02
    prices = 1.0 * np.exp(np.cumsum(returns)) # Startpreis = 1.0

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 252)
    })
    
    return df