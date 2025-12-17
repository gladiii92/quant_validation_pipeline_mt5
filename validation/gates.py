"""
Decision Gates für Strategie-Validierung.

- Ein Gate ist eine Entscheidungsregel
- Es nimmt alle Metriken und sagt: GO oder NO-GO?
- Das ist das Herzstück der Validierungs-Pipeline
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List

from utils.logger import get_logger
from utils.config import load_config

logger = get_logger(__name__)


class GateStatus(Enum):
    """Mögliche Status nach Validierung."""

    FAIL_FAST = "FAIL_FAST"          # Offensichtlich schlecht
    CONDITIONAL_PASS = "CONDITIONAL_PASS"  # Ok für Demo/Paper
    LIVE_ELIGIBLE = "LIVE_ELIGIBLE"  # Gut für Live
    KILLED = "KILLED"                # War ok, jetzt schlecht


@dataclass
class GateResult:
    """Ergebnis eines Gates."""

    status: GateStatus
    reason: str
    violated_criteria: List[str]   # Welche Kriterien nicht erfüllt
    metrics: Dict[str, Any]        # Alle geprüften Metriken
    confidence: float              # 0.0 - 1.0 (wie sicher ist das Urteil?)

    def __str__(self) -> str:
        """Schöne Ausgabe."""
        result = f"""
╔════════════════════════════════════════════════════════╗
║                 DECISION GATE RESULT                   ║
╠════════════════════════════════════════════════════════╣
║ Status:     {self.status.value:40}                     ║
║ Confidence: {self.confidence:.1%:40}                   ║
║ Reason:     {self.reason[:45]:40}                      ║
╠════════════════════════════════════════════════════════╣
║ Violated Criteria:                                     ║
"""
        if not self.violated_criteria:
            result += "║  ✅ None                                            \n"
        else:
            for criterion in self.violated_criteria:
                result += f"║  ❌ {criterion}\n"

        result += "╚════════════════════════════════════════════════════════╝\n"
        return result


class DecisionGate:
    """Hauptklasse für Entscheidungslogik."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialisiert das Gate mit Konfiguration.

        Args:
            config_path: Pfad zur config.yaml
        """
        self.config = load_config(config_path)
        # Section 'validation' in config.yaml muss folgende Keys liefern:
        # - min_oos_sharpe
        # - max_drawdown
        # - min_mc_positive_prob
        # - min_mt5_correlation
        self.thresholds = self.config["validation"]
        logger.info("DecisionGate initialized with thresholds: %s", self.thresholds)

    def evaluate(self, metrics: Dict[str, float]) -> GateResult:
        """
        Evaluiert Metriken und trifft Entscheidung.

        Erwartete Metriken (Beispiele):

        - oos_sharpe: Sharpe im Out-of-Sample
        - is_sharpe: Sharpe im In-Sample
        - max_drawdown: maximaler Drawdown (0–1)
        - win_rate: Trefferquote (0–1)
        - profit_factor: Profitfaktor
        - mc_positive_prob: Monte-Carlo-Wahrscheinlichkeit für positive Equity
        - mt5_correlation: Korrelation Backtest vs. MT5 (0–1)
        - return_pct: Gesamtrendite in %

        Args:
            metrics: Dictionary mit Metriken.

        Returns:
            GateResult mit Status und Begründung.
        """
        violated: List[str] = []

        logger.info("Evaluating metrics: %s", metrics)

        # 1) OOS Sharpe
        oos_sharpe = metrics.get("oos_sharpe", 0.0)
        if oos_sharpe < self.thresholds["min_oos_sharpe"]:
            violated.append(
                f"OOS Sharpe {oos_sharpe:.2f} < {self.thresholds['min_oos_sharpe']:.2f}"
            )

        # 2) Max Drawdown
        max_dd = metrics.get("max_drawdown", 1.0)
        if max_dd > self.thresholds["max_drawdown"]:
            violated.append(
                f"Max Drawdown {max_dd:.1%} > {self.thresholds['max_drawdown']:.1%}"
            )

        # 3) Monte Carlo Erfolgsrate
        mc_prob = metrics.get("mc_positive_prob", 0.0)
        if mc_prob < self.thresholds["min_mc_positive_prob"]:
            violated.append(
                f"MC Positive Prob {mc_prob:.1%} < "
                f"{self.thresholds['min_mc_positive_prob']:.1%}"
            )

        # 4) MT5 Korrelation
        mt5_corr = metrics.get("mt5_correlation", 0.0)
        if mt5_corr < self.thresholds["min_mt5_correlation"]:
            violated.append(
                f"MT5 Correlation {mt5_corr:.2f} < "
                f"{self.thresholds['min_mt5_correlation']:.2f}"
            )

        # Status bestimmen
        if len(violated) >= 2:
            status = GateStatus.FAIL_FAST
            reason = f"Multiple critical failures ({len(violated)} criteria)"
            confidence = 0.95
        elif len(violated) == 1:
            status = GateStatus.CONDITIONAL_PASS
            reason = f"One criterion violated: {violated[0]}"
            confidence = 0.75
        else:
            status = GateStatus.LIVE_ELIGIBLE
            reason = "All validation criteria passed ✅"
            confidence = 0.90

        result = GateResult(
            status=status,
            reason=reason,
            violated_criteria=violated,
            metrics=metrics,
            confidence=confidence,
        )

        logger.info("Gate result: %s", status.value)
        return result


if __name__ == "__main__":
    gate = DecisionGate()

    # Beispiel-Metriken (inklusive mt5_correlation aus deiner MT5-Pipeline)
    test_metrics = {
        "oos_sharpe": 1.2,
        "is_sharpe": 1.8,
        "max_drawdown": 0.12,
        "win_rate": 0.58,
        "profit_factor": 2.0,
        "mc_positive_prob": 0.85,
        "mt5_correlation": 0.88,
        "return_pct": 35.5,
    }

    result = gate.evaluate(test_metrics)
    print(result)
