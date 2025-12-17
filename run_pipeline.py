"""
HAUPT-PIPELINE: MT5 Backtest → Validierung → Urteil

Nutzung:
    python run_pipeline.py --trades-file data/processed/RangeBreakOut_USDJPY_trades_merged.csv

Diese Datei orchestriert die gesamte Pipeline:
1. MT5-Trades importieren
2. Metriken berechnen
3. Robustheitstests (optional)
4. Decision Gate
5. Report ausgeben
"""

import argparse
from pathlib import Path

from mt5_integration.trades_loader import MT5TradesLoader
from backtest.metrics import calculate_metrics
from validation.gates import DecisionGate
from utils.logger import get_logger
from utils.config import load_config

logger = get_logger("PIPELINE", log_file="logs/pipeline.log")


def run_pipeline(trades_csv_path: str, config_path: str = "config.yaml") -> None:
    """
    Hauptfunktion der Pipeline.

    Args:
        trades_csv_path: Pfad zur MT5-Trades CSV (merged CSV aus deinem Converter)
        config_path: Pfad zur config.yaml
    """
    logger.info("=" * 60)
    logger.info("🚀 QUANT VALIDATION PIPELINE STARTED")
    logger.info("=" * 60)

    # === SCHRITT 1: Konfiguration laden ===
    logger.info("\n[STEP 1] Loading configuration...")
    config = load_config(config_path)
    logger.info("✅ Config loaded: %s", config["project"]["name"])

    # === SCHRITT 2: Trades importieren ===
    logger.info("\n[STEP 2] Importing MT5 trades...")
    loader = MT5TradesLoader()

    try:
        trades_df = loader.load_trades(trades_csv_path)
        logger.info("✅ Loaded %d trades", len(trades_df))
    except FileNotFoundError as e:
        logger.error("❌ %s", e)
        return

    # === SCHRITT 3: Validiere Trades ===
    logger.info("\n[STEP 3] Validating trades...")
    validation = loader.validate_trades(trades_df)
    logger.info("Validation summary: %s", validation)

    # === SCHRITT 4: Berechne Metriken ===
    logger.info("\n[STEP 4] Calculating metrics...")
    initial_capital = config["backtest"]["initial_capital"]
    metrics = calculate_metrics(trades_df, initial_capital)

    # Ergänze mit Meta-Informationen
    metrics["strategy_name"] = Path(trades_csv_path).stem
    metrics["total_trades"] = len(trades_df)
    metrics["date_range"] = (
        trades_df["entry_time"].min(),
        trades_df["exit_time"].max(),
    )

    logger.info("✅ Metrics calculated:")
    logger.info(" Total Return: %.2f%%", metrics["total_return"] * 100)
    logger.info(" Sharpe Ratio: %.2f", metrics["sharpe_ratio"])
    logger.info(" Max Drawdown: %.2f%%", metrics["max_drawdown"] * 100)
    logger.info(" Win Rate: %.2f%%", metrics["win_rate"] * 100)

    # === SCHRITT 5: Decision Gate ===
    logger.info("\n[STEP 5] Running Decision Gate...")
    gate = DecisionGate(config_path)

    # Verwandle Metriken ins Gate-Format
    gate_metrics = {
        "oos_sharpe": metrics.get("sharpe_ratio", 0.0),  # später: echte OOS-Sharpe
        "max_drawdown": metrics.get("max_drawdown", 1.0),
        # Platzhalter, später aus Monte-Carlo / Heston etc.
        "mc_positive_prob": 0.80,
        # Platzhalter, später aus Korrelation Python-Backtest vs. MT5
        "mt5_correlation": 0.90,
    }

    result = gate.evaluate(gate_metrics)

    logger.info("\n%s", "=" * 60)
    logger.info("GATE RESULT: %s", result.status.value)
    logger.info("Confidence: %.1f%%", result.confidence * 100)
    logger.info("Reason: %s", result.reason)

    if result.violated_criteria:
        logger.warning("Violated criteria:")
        for criterion in result.violated_criteria:
            logger.warning(" ❌ %s", criterion)

    logger.info("%s\n", "=" * 60)

    # === SCHRITT 6: Report speichern ===
    logger.info("[STEP 6] Saving report...")
    Path("reports").mkdir(exist_ok=True)
    report_path = Path("reports") / f"{Path(trades_csv_path).stem}_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("QUANT VALIDATION PIPELINE REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Strategy: {metrics['strategy_name']}\n")
        f.write(
            f"Date Range: {metrics['date_range'][0]} to {metrics['date_range'][1]}\n\n"
        )

        f.write("METRICS:\n")
        f.write(f" Total Trades: {metrics['total_trades']}\n")
        f.write(f" Total Return: {metrics['total_return']:.2%}\n")
        f.write(f" Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n")
        f.write(f" Max Drawdown: {metrics['max_drawdown']:.2%}\n")
        f.write(f" Win Rate: {metrics['win_rate']:.2%}\n\n")

        f.write(f"DECISION: {result.status.value}\n")
        f.write(f"Reason: {result.reason}\n")

    logger.info("✅ Report saved to %s", report_path)
    logger.info("\n" + "=" * 60)
    logger.info("🎉 PIPELINE COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Validation Pipeline")

    parser.add_argument(
        "--trades-file",
        type=str,
        required=True,
        help="Path to MT5 trades CSV file (merged converter output)",
    )

    args = parser.parse_args()
    run_pipeline(args.trades_file)
