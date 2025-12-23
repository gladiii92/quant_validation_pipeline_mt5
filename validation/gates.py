"""
Decision Gates für Strategie-Validierung - ELITE LEVEL!

Mehrstufige Kriterien-Logik:
- ELITE:    alle Kern-Kriterien erfüllt
- LIVE:     max. 1 leichte Verletzung, keine harten RED FLAGS
- WAIT:     mehrere Verletzungen, aber kein Totalausfall -> Paper
- FAIL:     deutliche Verstöße gegen Risk-Obergrenzen
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
    CONDITIONAL_PASS = "WAIT_AND_SEE"  # Ok für Demo/Paper
    LIVE_ELIGIBLE = "LIVE_ELIGIBLE"  # Gut für Live
    ELITE = "ELITE"                  # TOP 1 % - sofort Live!


@dataclass
class GateResult:
    """Ergebnis eines Gates."""
    status: GateStatus
    reason: str
    violated_criteria: List[str]
    metrics: Dict[str, Any]
    confidence: float  # 0.0 - 1.0

    def __str__(self) -> str:
        result = f"""
╔════════════════════════════════════════════════════════╗
║ 🚀 DECISION GATE RESULT                                ║
╠════════════════════════════════════════════════════════╣
║ Status:     {self.status.value:30}                 ║
║ Confidence: {self.confidence:.1%:<30}                 ║
║ Reason:     {self.reason[:45]:<45}║
╠════════════════════════════════════════════════════════╣
║ VIOLATED CRITERIA:                                    ║
"""
        if not self.violated_criteria:
            result += "║ ✅ NONE - ALL ELITE CRITERIA PASSED!           ║\n"
        else:
            for i, criterion in enumerate(self.violated_criteria, 1):
                result += f"║ {i:2d}. ❌ {criterion:<45}║\n"
        result += "╚════════════════════════════════════════════════════════╝\n"
        return result


class DecisionGate:
    """
    ELITE Decision Gate - Nur Strategien mit robusten Kennzahlen.

    Die Schwellen können in config.yaml unter `validation:` überschrieben werden.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = load_config(config_path)

        # Standard-Schwellen (werden von config.yaml überschrieben)
        self.thresholds = {
            "min_oos_sharpe": 1.8,          # anspruchsvoller OOS Sharpe
            "max_oos_drawdown": 0.12,       # 12 % OOS MaxDD
            "min_mc_positive_prob": 0.95,   # 95 % der MC-Szenarien positiv
            "min_mc_p95_return": 0.20,      # 95%-Quantil > 20 %
            "max_cvar5": -0.15,             # CVaR 5 % besser als -15 %
            "min_mt5_correlation": 0.90,    # hohe Korrelation Live vs. Backtest
            "min_multi_asset_hit_rate": 0.75,  # 75 % der Symbole Sharpe > 1
            "min_kelly_oos_full": 0.05,     # Kelly-OOS mindestens 5 %
            "regime_multiplier_min": 0.8,   # Regime-Freigabe muss >= 0.8 sein
        }

        self.thresholds.update(self.config.get("validation", {}))
        logger.info("🚀 DecisionGate initialized with thresholds: %s", self.thresholds)

    def evaluate(self, metrics: Dict[str, float]) -> GateResult:
        """
        Bewertet eine Strategie anhand der Metriken.

        `metrics` erwartet u.a.:
        - oos_sharpe
        - oos_max_drawdown
        - mc_positive_prob
        - mc_p95_return
        - cvar5
        - kelly_oos_full
        - multi_asset_hit_rate (optional)
        - mt5_correlation (optional)
        - regime_allowed / regime_risk_multiplier (optional)
        """
        violated: List[str] = []
        hard_red_flags: List[str] = []

        logger.info(
            "Evaluating DecisionGate with metrics: %s",
            {k: v for k, v in metrics.items()}
        )

        # 1) OOS Sharpe
        oos_sharpe = float(metrics.get("oos_sharpe", 0.0))
        if oos_sharpe < self.thresholds["min_oos_sharpe"]:
            msg = f"OOS Sharpe {oos_sharpe:.2f} < {self.thresholds['min_oos_sharpe']:.2f}"
            violated.append(msg)
            if oos_sharpe < 1.0:
                hard_red_flags.append("OOS Sharpe < 1.0")

        # 2) OOS Max Drawdown
        oos_max_dd = float(metrics.get("oos_max_drawdown", 1.0))
        if oos_max_dd > self.thresholds["max_oos_drawdown"]:
            msg = f"OOS MaxDD {oos_max_dd:.1%} > {self.thresholds['max_oos_drawdown']:.1%}"
            violated.append(msg)
            if oos_max_dd > 0.20:
                hard_red_flags.append("OOS MaxDD > 20 %")

        # 3) Monte-Carlo P95 Return
        mc_p95 = float(metrics.get("mc_p95_return", 0.0))
        if mc_p95 < self.thresholds["min_mc_p95_return"]:
            msg = f"MC P95 Return {mc_p95:.1%} < {self.thresholds['min_mc_p95_return']:.1%}"
            violated.append(msg)

        # 4) CVaR 5 % (Tail Risk)
        cvar5 = float(metrics.get("cvar5", 0.0))
        if cvar5 < self.thresholds["max_cvar5"]:
            msg = f"CVaR5 {cvar5:.1%} < {self.thresholds['max_cvar5']:.1%}"
            violated.append(msg)
            if cvar5 < -0.25:
                hard_red_flags.append("CVaR5 < -25 %")

        # 5) Kelly OOS (realistisch, nicht astronomisch)
        kelly_oos_full = float(metrics.get("kelly_oos_full", 0.0))
        if kelly_oos_full < self.thresholds["min_kelly_oos_full"]:
            msg = (
                f"OOS Kelly full {kelly_oos_full:.1%} < "
                f"{self.thresholds['min_kelly_oos_full']:.1%}"
            )
            violated.append(msg)

        # 6) Multi-Asset Hit-Rate
        if "multi_asset_hit_rate" in metrics:
            hit_rate = float(metrics["multi_asset_hit_rate"])
            if hit_rate < self.thresholds["min_multi_asset_hit_rate"]:
                msg = (
                    f"Multi-Asset hit-rate {hit_rate:.1%} < "
                    f"{self.thresholds['min_multi_asset_hit_rate']:.1%}"
                )
                violated.append(msg)
                if hit_rate < 0.4:
                    hard_red_flags.append("Multi-Asset hit-rate < 40 %")

        # 7) MC positive Szenarien
        mc_pos = float(metrics.get("mc_positive_prob", 0.0))
        if mc_pos < self.thresholds["min_mc_positive_prob"]:
            msg = (
                f"MC positive prob {mc_pos:.1%} < "
                f"{self.thresholds['min_mc_positive_prob']:.1%}"
            )
            violated.append(msg)

        # 8) MT5-Korrelation (falls vorhanden)
        if "mt5_correlation" in metrics:
            corr = float(metrics["mt5_correlation"])
            if corr < self.thresholds["min_mt5_correlation"]:
                msg = (
                    f"MT5 correlation {corr:.2f} < "
                    f"{self.thresholds['min_mt5_correlation']:.2f}"
                )
                violated.append(msg)
                if corr < 0.8:
                    hard_red_flags.append("MT5 correlation < 0.80")

        # 9) Regime-Filter (falls vorhanden)
        if metrics.get("regime_allowed") is False:
            violated.append("Regime policy forbids trading in current environment")
            hard_red_flags.append("Regime policy = FORBIDDEN")
        risk_mult = float(metrics.get("regime_risk_multiplier", 1.0))
        if risk_mult < self.thresholds["regime_multiplier_min"]:
            violated.append(
                f"Regime risk multiplier {risk_mult:.2f} < "
                f"{self.thresholds['regime_multiplier_min']:.2f}"
            )

        # ---- Status bestimmen (mehrkriteriell) ----
        n_viol = len(violated)
        n_hard = len(hard_red_flags)

        if n_viol == 0:
            status = GateStatus.ELITE
            reason = "🏆 Alle Kern-Kriterien erfüllt – sofort LIVE (ELITE)."
            confidence = 1.0
        elif n_hard == 0 and n_viol == 1:
            status = GateStatus.LIVE_ELIGIBLE
            reason = f"⚠️ 1 Minor-Verletzung ({violated[0]}) – Live möglich, eng überwachen."
            confidence = 0.85
        elif n_hard == 0 and 1 < n_viol <= 3:
            status = GateStatus.CONDITIONAL_PASS
            reason = (
                f"📊 {n_viol} Kriterien verletzt – erst Demo/Paper, "
                "Verbesserungen & Re-Validation nötig."
            )
            confidence = 0.60
        else:
            status = GateStatus.FAIL_FAST
            reason = (
                f"❌ {n_viol} Violations, davon {n_hard} harte RED FLAGS – "
                "kein Live, Setup überarbeiten."
            )
            confidence = 0.10

        result = GateResult(
            status=status,
            reason=reason,
            violated_criteria=violated,
            metrics=metrics,
            confidence=confidence,
        )

        logger.info("🚀 Gate result: %s (%.1f%%)", status.value, confidence * 100.0)
        if hard_red_flags:
            logger.warning("Hard RED FLAGS: %s", hard_red_flags)
        return result


if __name__ == "__main__":
    gate = DecisionGate()
    test_metrics = {
        "oos_sharpe": 2.4,
        "oos_max_drawdown": 0.09,
        "mc_positive_prob": 0.98,
        "mc_p95_return": 0.32,
        "cvar5": -0.10,
        "kelly_oos_full": 0.12,
        "multi_asset_hit_rate": 0.82,
        "mt5_correlation": 0.93,
        "regime_allowed": True,
        "regime_risk_multiplier": 1.0,
    }
    print(gate.evaluate(test_metrics))
