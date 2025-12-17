"""
Hidden Markov Model für Market Regime Detection.

- HMM ist ein statistisches Modell
- Es erkennt, in welchem "Regime" der Markt gerade ist
- Regime: Bull, Bear, Range, HighVol, etc.
- Wichtig, um zu wissen, welche Strategie gerade sinnvoll ist
"""

from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from utils.logger import get_logger

logger = get_logger(__name__)


class MarketRegimeDetector:
    """Erkennt Marktregime mit HMM."""

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        """
        Initialisiert den Regime Detector.

        Args:
            n_regimes: Anzahl der Regimes (z.B. 3 = Bull, Range, Bear)
            random_state: Seed für Reproduzierbarkeit
        """
        self.n_regimes = n_regimes
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            random_state=random_state,
        )
        self.is_fitted: bool = False

        logger.info("MarketRegimeDetector initialized with %d regimes", n_regimes)

    def fit(self, returns: pd.Series) -> None:
        """
        Trainiert das HMM auf historischen Returns.

        Args:
            returns: pandas Series mit (z.B. täglichen) Returns
        """
        returns = returns.dropna()
        if len(returns) == 0:
            raise ValueError("No returns provided to fit HMM")

        # HMM erwartet ein 2D-Array
        X = returns.values.reshape(-1, 1)
        self.model.fit(X)
        self.is_fitted = True

        logger.info("HMM fitted on %d observations", len(returns))

    def predict(self, returns: pd.Series) -> np.ndarray:
        """
        Sagt Regimes für neue Returns voraus.

        Args:
            returns: pandas Series mit Returns

        Returns:
            Array mit Regime IDs (0, 1, 2, ...)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        returns = returns.dropna()
        X = returns.values.reshape(-1, 1)
        regimes = self.model.predict(X)

        logger.info("Predicted regimes for %d observations", len(returns))
        return regimes

    def get_regime_name(self, regime_id: int) -> str:
        """
        Konvertiert Regime-ID zu Name (heuristisch).

        Heuristik:
        - Regime mit niedrigster Volatilität = "Range"
        - Regime mit positivem Drift = "Bull"
        - Regime mit negativem Drift = "Bear"
        """
        if not self.is_fitted:
            return f"Regime_{regime_id}"

        # Mittlere Volatilität pro Regime
        vols = [np.sqrt(self.model.covars_[i, 0, 0]) for i in range(self.n_regimes)]
        # Drift (Mittelwert) pro Regime
        drifts = [self.model.means_[i, 0] for i in range(self.n_regimes)]

        lowest_vol_regime = int(np.argmin(vols))

        if regime_id == lowest_vol_regime:
            return "Range"
        elif drifts[regime_id] > 0:
            return "Bull"
        else:
            return "Bear"

    def label_series(self, returns: pd.Series) -> pd.DataFrame:
        """
        Annotiert eine Returns-Serie mit Regime-ID und Regime-Name.

        Args:
            returns: pandas Series mit Returns (Index = Datum/Zeit)

        Returns:
            DataFrame mit Spalten:
            - return
            - regime_id
            - regime_name
        """
        regimes = self.predict(returns)
        df = pd.DataFrame(
            {
                "return": returns.values,
                "regime_id": regimes,
            },
            index=returns.index,
        )
        df["regime_name"] = df["regime_id"].apply(self.get_regime_name)
        return df


if __name__ == "__main__":
    # Simuliere Returns
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.02)

    detector = MarketRegimeDetector(n_regimes=3)
    detector.fit(returns)

    labeled = detector.label_series(returns)
    print(labeled.tail(10))
