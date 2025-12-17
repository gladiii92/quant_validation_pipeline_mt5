# vix_loader.py
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from utils.logger import get_logger

logger = get_logger(__name__)


def _download_vix_history(start: str = "2000-01-01") -> pd.DataFrame:
    logger.info("Downloading VIX history from Yahoo Finance (^VIX)...")
    data = yf.download("^VIX", start=start)
    if data.empty:
        raise RuntimeError("No VIX data downloaded from Yahoo Finance")

    # Falls MultiIndex-Spalten (z.B. ('Close','^VIX')):
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]["^VIX"]
    else:
        close = data["Close"]

    df = close.to_frame(name="Close")
    df.index.name = "Date"
    df.reset_index(inplace=True)
    return df


def _is_file_fresh(path: Path, max_age_days: int) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime < timedelta(days=max_age_days)


def load_vix_regimes(
    cache_path: str = "data/external/vix_daily.csv",
    max_age_days: int = 14,
) -> pd.Series:
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if _is_file_fresh(cache_file, max_age_days):
        logger.info("Loading VIX data from cache: %s", cache_file)
        df = pd.read_csv(cache_file, parse_dates=["Date"])
    else:
        df = _download_vix_history(start="2000-01-01")
        df.to_csv(cache_file, index=False)
        logger.info("VIX data downloaded and cached to: %s", cache_file)

    df = df.sort_values("Date").set_index("Date")
    vix = df["Close"].astype(float)

    # Regime-Serien-Index exakt wie vix
    regimes = pd.Series(index=vix.index, dtype="object")

    regimes[vix < 15] = "Low_Volatility"
    regimes[(vix >= 15) & (vix < 25)] = "Range"
    regimes[(vix >= 25) & (vix < 35)] = "High_Volatility"
    regimes[vix >= 35] = "Crash"

    regimes.name = "vix_regime"
    return regimes
