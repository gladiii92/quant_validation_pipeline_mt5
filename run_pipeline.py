"""
HAUPT-PIPELINE: MT5 Backtest → Validierung → Urteil

Nutzung:
    python run_pipeline.py --trades-file data/processed/RangeBreakOut_USDJPY_trades_merged.csv

Diese Datei orchestriert die gesamte Pipeline:
1. MT5-Trades importieren
2. Metriken berechnen
3. Monte-Carlo-Robustheitstests
4. Decision Gate
5. Report + Plots ausgeben
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mt5_integration.trades_loader import MT5TradesLoader
from backtest.metrics import calculate_metrics
from validation.cost_scenarios import run_cost_scenarios
from validation.gates import DecisionGate
from validation.monte_carlo import run_monte_carlo_on_trades
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


    # === SCHRITT 5: Kosten-/Slippage-Szenarien ===
    logger.info("\n[STEP 5] Running cost/slippage scenarios...")

    cost_scenarios = {
        "base": 1.0,
        "cost_plus_25": 0.75,
        "cost_plus_50": 0.5,
    }

    cost_results = run_cost_scenarios(
        trades_df,
        initial_capital=initial_capital,
        scenarios=cost_scenarios,
    )

    logger.info("✅ Cost scenarios finished")

    # === SCHRITT 6: Monte-Carlo auf Trades ===
    logger.info("\n[STEP 5] Running Monte Carlo on trade sequence...")

    mc_results = run_monte_carlo_on_trades(
        trades_df,
        initial_capital=initial_capital,
        n_sims=5000,
        random_state=42,
    )

    logger.info("✅ Monte Carlo finished")
    logger.info(" mc_positive_prob: %.2f%%", mc_results["mc_positive_prob"] * 100)
    logger.info(" mc_median_return: %.2f%%", mc_results["mc_median_return"] * 100)
    logger.info(" mc_p5_return: %.2f%%", mc_results["mc_p5_return"] * 100)
    logger.info(" mc_p95_return: %.2f%%", mc_results["mc_p95_return"] * 100)
    logger.info(" mc_median_max_dd: %.2f%%", mc_results["mc_median_max_dd"] * 100)
    logger.info(" mc_p95_max_dd: %.2f%%", mc_results["mc_p95_max_dd"] * 100)

    # === Visualisierung: Histogramm der MC-Returns ===
    logger.info("[PLOT] Saving Monte Carlo return distribution...")

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    returns = np.array(mc_results["total_returns"])
    unique_vals = np.unique(returns)

    plt.figure(figsize=(8, 5))

    data_min = float(returns.min())
    data_max = float(returns.max())
    data_range = data_max - data_min

    # Fall 1: Alle Werte identisch ODER Range extrem klein → Bar-Plot
    if len(unique_vals) <= 1 or data_range < 1e-6:
        center = float(returns.mean())
        height = len(returns)
        # Breite ein bisschen relativ zum Wert wählen, damit man etwas sieht
        width = 0.001 * max(1.0, abs(center))

        plt.bar(
            [center],
            [height],
            width=width,
            color="steelblue",
            edgecolor="black",
        )
        plt.axvline(center, color="red", linestyle="--", label="Return")
    else:
        # Fall 2: „normale“ Verteilung → Histogramm
        n_unique = len(unique_vals)
        bins = max(5, min(50, n_unique))
        plt.hist(returns, bins=bins, color="steelblue", edgecolor="black")
        plt.axvline(
            mc_results["mc_median_return"], color="red", linestyle="--", label="Median"
        )
        plt.axvline(
            mc_results["mc_p5_return"], color="orange", linestyle="--", label="5%"
        )
        plt.axvline(
            mc_results["mc_p95_return"], color="green", linestyle="--", label="95%"
        )

    plt.title("Monte Carlo Total Return Distribution (Trades-Level)")
    plt.xlabel("Total Return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    mc_plot_path = reports_dir / f"{Path(trades_csv_path).stem}_mc_returns.png"
    plt.savefig(mc_plot_path)
    plt.close()

    logger.info("✅ Monte Carlo return plot saved to %s", mc_plot_path)

    # === SCHRITT 7: Decision Gate ===
    logger.info("\n[STEP 6] Running Decision Gate...")
    gate = DecisionGate(config_path)

    gate_metrics = {
        "oos_sharpe": metrics.get("sharpe_ratio", 0.0),  # vorerst Gesamt-Sharpe
        "max_drawdown": metrics.get("max_drawdown", 1.0),
        "mc_positive_prob": mc_results["mc_positive_prob"],
        "mt5_correlation": 0.90,  # TODO: später durch echte Korrelation ersetzen
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

    # === SCHRITT 8: Report speichern ===
    logger.info("[STEP 7] Saving report...")
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"{Path(trades_csv_path).stem}_report.txt"

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

        f.write("MONTE CARLO:\n")
        f.write(f" mc_positive_prob: {mc_results['mc_positive_prob']:.2%}\n")
        f.write(f" mc_median_return: {mc_results['mc_median_return']:.2%}\n")
        f.write(f" mc_p5_return: {mc_results['mc_p5_return']:.2%}\n")
        f.write(f" mc_p95_return: {mc_results['mc_p95_return']:.2%}\n")
        f.write(f" mc_median_max_dd: {mc_results['mc_median_max_dd']:.2%}\n")
        f.write(f" mc_p95_max_dd: {mc_results['mc_p95_max_dd']:.2%}\n\n")

        f.write("COST SCENARIOS:\n")
        for name, m in cost_results.items():
            f.write(f" {name}:\n")
            f.write(f"   Total Return: {m['total_return']:.2%}\n")
            f.write(f"   Sharpe Ratio: {m['sharpe_ratio']:.2f}\n")
            f.write(f"   Max Drawdown: {m['max_drawdown']:.2%}\n")
            f.write(f"   Profit Factor: {m['profit_factor']:.2f}\n")
        f.write("\n")

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
