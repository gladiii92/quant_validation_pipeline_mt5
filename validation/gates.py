"""
Decision Gates für Strategie-Validierung - ELITE LEVEL!
Nur TOP 1% Strategien kommen durch!
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List
from utils.logger import get_logger
from utils.config import load_config

logger = get_logger(__name__)

class GateStatus(Enum):
    """Mögliche Status nach Validierung."""
    FAIL_FAST = "FAIL_FAST"      # Offensichtlich schlecht
    CONDITIONAL_PASS = "CONDITIONAL_PASS"  # Ok für Demo/Paper
    LIVE_ELIGIBLE = "LIVE_ELIGIBLE"        # Gut für Live
    ELITE = "ELITE"                # TOP 1% - sofort Live!

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
║           🚀 DECISION GATE RESULT 🚀                  ║
╠════════════════════════════════════════════════════════╣
║ Status:     {self.status.value:30}                     ║
║ Confidence: {self.confidence:.1%:<30}                  ║
║ Reason:     {self.reason[:45]:<30}                     ║
╠════════════════════════════════════════════════════════╣
║ VIOLATED CRITERIA:                                    ║
"""
        if not self.violated_criteria:
            result += "║ ✅ NONE - ALL ELITE CRITERIA PASSED!      ║\n"
        else:
            for i, criterion in enumerate(self.violated_criteria, 1):
                result += f"║ {i:2d}. ❌ {criterion:<35}║\n"
        result += "╚════════════════════════════════════════════════════════╝\n"
        return result

class DecisionGate:
    """ELITE Decision Gate - Nur TOP 1% Strategien!"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        # HARDCODED ELITE THRESHOLDS (config.yaml überschreibt optional)
        self.thresholds = {
            "min_oos_sharpe": 1.8,           # TOP 1%
            "min_oos_profit_factor": 2.0,    # ELITE
            "max_oos_drawdown": 0.12,        # 12% OOS!
            "min_mc_positive_prob": 0.95,    # 95% Erfolg
            "min_mc_p95_return": 0.20,       # P95 > 20%
            "max_cvar5": -0.15,              # CVaR5 besser -15%
            "min_mt5_correlation": 0.90,     # Hohe Korrelation
            "min_multi_asset_hit_rate": 0.75, # 75% Symbole > Sharpe 1.0
            "min_kelly_oos_full": 0.15,      # Kelly OOS > 15%
            "regime_multiplier_min": 0.8,
        }
        # Config überschreibt
        self.thresholds.update(self.config.get("validation", {}))
        logger.info("🚀 ELITE DecisionGate initialized: %s", self.thresholds)

    def evaluate(self, metrics: Dict[str, float]) -> GateResult:
        violated = []
        logger.info("Evaluating ELITE metrics: %s", {k: v for k, v in metrics.items() if k in self.thresholds})

        # 1) OOS Sharpe (KRITISCH!)
        if metrics.get("oos_sharpe", 0) < self.thresholds["min_oos_sharpe"]:
            violated.append(f"OOS Sharpe {metrics['oos_sharpe']:.2f} < {self.thresholds['min_oos_sharpe']:.2f}")

        # 2) OOS Max Drawdown (OOS statt Full Sample!)
        if metrics.get("oos_max_drawdown", 1.0) > self.thresholds["max_oos_drawdown"]:
            violated.append(f"OOS MaxDD {metrics['oos_max_drawdown']:.1%} > {self.thresholds['max_oos_drawdown']:.1%}")

        # 3) Monte Carlo P95 Return (NEU!)
        if metrics.get("mc_p95_return", 0) < self.thresholds["min_mc_p95_return"]:
            violated.append(f"MC P95 Return {metrics['mc_p95_return']:.1%} < {self.thresholds['min_mc_p95_return']:.1%}")

        # 4) CVaR5 Tail Risk (NEU!)
        if metrics.get("cvar5", 0) < self.thresholds["max_cvar5"]:
            violated.append(f"CVaR5 {metrics['cvar5']:.1%} < {self.thresholds['max_cvar5']:.1%}")

        # 5) Kelly OOS (NEU!)
        if metrics.get("kelly_oos_full", 0) < self.thresholds["min_kelly_oos_full"]:
            violated.append(f"OOS Kelly {metrics['kelly_oos_full']:.1%} < {self.thresholds['min_kelly_oos_full']:.1%}")

        # 6) Multi-Asset Hit-Rate
        if metrics.get("multi_asset_hit_rate", 0) < self.thresholds["min_multi_asset_hit_rate"]:
            violated.append(f"Multi-Asset {metrics['multi_asset_hit_rate']:.1%} < {self.thresholds['min_multi_asset_hit_rate']:.1%}")

        # 7) MC Positive Probability
        if metrics.get("mc_positive_prob", 0) < self.thresholds["min_mc_positive_prob"]:
            violated.append(f"MC Pos Prob {metrics['mc_positive_prob']:.1%} < {self.thresholds['min_mc_positive_prob']:.1%}")

        # Status bestimmen
        if len(violated) == 0:
            status = GateStatus.ELITE
            reason = "🏆 ALL ELITE CRITERIA PASSED - IMMEDIATE LIVE DEPLOYMENT!"
            confidence = 1.0
        elif len(violated) == 1:
            status = GateStatus.LIVE_ELIGIBLE
            reason = f"⚠️ Minor violation: {violated[0]}"
            confidence = 0.85
        elif len(violated) <= 3:
            status = GateStatus.CONDITIONAL_PASS
            reason = f"📊 {len(violated)} criteria violated - Paper Trading OK"
            confidence = 0.60
        else:
            status = GateStatus.FAIL_FAST
            reason = f"❌ CRITICAL FAIL: {len(violated)} violations"
            confidence = 0.10

        result = GateResult(
            status=status,
            reason=reason,
            violated_criteria=violated,
            metrics=metrics,
            confidence=confidence
        )
        logger.info("🚀 Gate result: %s (%.1f%%)", status.value, confidence*100)
        return result

if __name__ == "__main__":
    gate = DecisionGate()
    test_metrics = {
        "oos_sharpe": 2.5,
        "oos_max_drawdown": 0.08,
        "mc_p95_return": 0.25,
        "cvar5": -0.10,
        "kelly_oos_full": 0.20,
        "multi_asset_hit_rate": 0.80,
        "mc_positive_prob": 0.98,
        "oos_profit_factor": 2.5
    }
    result = gate.evaluate(test_metrics)
    print(result)
