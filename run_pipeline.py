"""
HAUPT-PIPELINE: MT5 Backtest → Validierung → Visualisierung → Urteil
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mt5_integration.trades_loader import MT5TradesLoader
from backtest.metrics import calculate_metrics
from validation.gates import DecisionGate
from utils.logger import get_logger
from utils.config import load_config
from validation.cost_scenarios import run_cost_scenarios
from validation.kelly import estimate_kelly_from_trades
from vix_loader import load_vix_regimes
from validation.regime_alignment import analyze_vix_regime_alignment
from validation.monte_carlo import run_monte_carlo_on_trades
from validation.walk_forward import run_walk_forward_analysis
from validation.multi_asset_summary import load_and_score_optimizer

logger = get_logger("PIPELINE", log_file="logs/pipeline.log")


def find_latest_trades_csv(processed_dir: str = "data/processed") -> str:
    """Nimmt die neueste *_trades_merged.csv aus processed."""
    files = sorted(
        Path(processed_dir).glob("*_trades_merged.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"Keine *_trades_merged.csv in {processed_dir} gefunden.")
    return str(files[0])


def run_pipeline(trades_csv_path: str, config_path: str = "config.yaml") -> None:
    """
    Hauptfunktion der Pipeline.
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

    # === SCHRITT 4b: Multi-Asset-Optimizer (optional) ===
    multi_asset_info = None
    optimizer_candidates = sorted(
        Path("data/raw").glob("ReportOptimizer-*.xml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if optimizer_candidates:
        optimizer_xml = optimizer_candidates[0]
        logger.info(
            "\n[STEP 4b] Evaluating multi-asset optimizer results from %s...",
            optimizer_xml,
        )
        multi_asset_info = load_and_score_optimizer(
            str(optimizer_xml), sharpe_threshold=1.0
        )
        logger.info(
            "Multi-Asset hit-rate: %.1f%% (%d/%d, Sharpe > %.2f)",
            multi_asset_info["hit_rate"] * 100,
            multi_asset_info["n_symbols_pass"],
            multi_asset_info["n_symbols"],
            multi_asset_info["sharpe_threshold"],
        )

    # === SCHRITT 4: Berechne Metriken (Full Sample) ===
    logger.info("\n[STEP 4] Calculating metrics...")
    initial_capital = config["backtest"]["initial_capital"]
    metrics = calculate_metrics(trades_df, initial_capital)
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
        trades_df, initial_capital=initial_capital, scenarios=cost_scenarios
    )
    logger.info("✅ Cost scenarios finished")

    # === SCHRITT 6: Kelly-basiertes Sizing (Full Sample) ===
    logger.info("\n[STEP 6] Estimating Kelly sizing...")
    kelly_info = estimate_kelly_from_trades(trades_df)
    logger.info("✅ Kelly estimation finished")

    # === SCHRITT 7: VIX-Regime-Ausrichtung ===
    logger.info("\n[STEP 7] Loading VIX regimes and checking alignment...")
    vix_regimes = load_vix_regimes(
        cache_path="data/external/vix_daily.csv",
        max_age_days=14,
    )
    vix_alignment = analyze_vix_regime_alignment(
        trades_df=trades_df,
        vix_regimes=vix_regimes,
        initial_capital=initial_capital,
        policy_path="regime_policy.yaml",
        strategy_key="range_breakout",
    )
    logger.info("✅ VIX regime alignment finished")

    # === SCHRITT 9: Walk-Forward / OOS-Analyse ===
    logger.info("\n[STEP 9] Running walk-forward analysis on trades...")
    wf_config = config.get("walk_forward", {})
    wf_results = run_walk_forward_analysis(
        trades_df=trades_df,
        initial_capital=initial_capital,
        train_days=wf_config.get("train_window", 252),
        test_days=wf_config.get("test_window", 63),
        step_days=wf_config.get("step_size", 21),
    )
    logger.info(
        "✅ Walk-forward finished: n_windows=%d, oos_sharpe=%.2f, "
        "oos_profit_factor=%.2f, oos_max_dd=%.2f%%",
        wf_results["n_windows"],
        wf_results["oos_sharpe"],
        wf_results["oos_profit_factor"],
        wf_results["oos_max_dd"] * 100,
    )

    # === Kelly aus OOS-Trades (Variante A) ===
    logger.info("[STEP 9b] Estimating Kelly from OOS (walk-forward test windows)...")
    oos_indices = []
    for w in wf_results["window_metrics"]:
        oos_indices.extend(w.get("test_indices", []))
    oos_indices = sorted(set(oos_indices))

    if oos_indices:
        trades_df_oos = trades_df.loc[oos_indices].copy()
        kelly_oos_info = estimate_kelly_from_trades(trades_df_oos)
        logger.info(
            "✅ OOS Kelly: win_rate=%.2f%%, payoff=%.2f, kelly_full=%.2f%%",
            kelly_oos_info["win_rate"] * 100,
            kelly_oos_info["payoff_ratio"],
            kelly_oos_info["kelly_full"] * 100,
        )
    else:
        kelly_oos_info = {
            "win_rate": 0.0,
            "payoff_ratio": 0.0,
            "kelly_full": 0.0,
            "kelly_half": 0.0,
            "kelly_quarter": 0.0,
        }
        logger.info("⚠️ No OOS trades for Kelly estimation.")



    # === SCHRITT 8: Monte Carlo ===
    logger.info("\n[STEP 8] Running Monte Carlo on trade sequence...")
    mc_results = run_monte_carlo_on_trades(
        trades_df=trades_df,
        initial_capital=initial_capital,
        n_sims=1000,
        random_state=42,
    )
    logger.info("✅ Monte Carlo finished")
    logger.info(" mc_positive_prob: %.2f%%", mc_results["mc_positive_prob"] * 100)
    logger.info(" mc_median_return: %.2f%%", mc_results["mc_median_return"] * 100)
    logger.info(" mc_p5_return: %.2f%%", mc_results["mc_p5_return"] * 100)
    logger.info(" mc_p95_return: %.2f%%", mc_results["mc_p95_return"] * 100)
    logger.info(
        " mc_median_max_dd: %.2f%%", mc_results["mc_median_max_dd"] * 100
    )
    logger.info(" mc_p95_max_dd: %.2f%%", mc_results["mc_p95_max_dd"] * 100)

    # === PLOTS ===

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    stem = Path(trades_csv_path).stem

    # Monte Carlo Return Distribution
    logger.info("[PLOT] Saving Monte Carlo return distribution...")
    mc_plot_path = reports_dir / f"{stem}_mc_returns.png"
    if "returns" in mc_results:
        plt.figure(figsize=(6, 4))
        plt.hist(mc_results["returns"], bins=50, alpha=0.7)
        plt.axvline(
            np.median(mc_results["returns"]),
            color="red",
            linestyle="--",
            label="Median",
        )
        plt.title("Monte Carlo Returns")
        plt.legend()
        plt.tight_layout()
        plt.savefig(mc_plot_path, dpi=150)
        plt.close()
        logger.info("✅ Monte Carlo return plot saved to %s", mc_plot_path)

    # Equity Curve
    logger.info("[PLOT] Saving equity curve...")
    eq = trades_df["pnl"].cumsum() + initial_capital
    plt.figure(figsize=(8, 4))
    plt.plot(eq.index, eq.values)
    plt.title("Equity Curve")
    plt.ylabel("Equity")
    plt.tight_layout()
    eq_path = reports_dir / f"{stem}_equity.png"
    plt.savefig(eq_path, dpi=150)
    plt.close()
    logger.info("✅ Equity curve saved to %s", eq_path)

    # VIX Regime Sharpe
    logger.info("[PLOT] Saving VIX regime Sharpe barplot...")
    vix_stats = vix_alignment["regime_stats"]
    names = list(vix_stats.keys())
    sharpes = [vix_stats[n]["sharpe_ratio"] for n in names]
    plt.figure(figsize=(6, 4))
    plt.bar(names, sharpes)
    plt.title("Sharpe by VIX Regime")
    plt.ylabel("Sharpe")
    plt.tight_layout()
    vix_plot_path = reports_dir / f"{stem}_vix_regime_sharpe.png"
    plt.savefig(vix_plot_path, dpi=150)
    plt.close()
    logger.info("✅ VIX regime Sharpe plot saved to %s", vix_plot_path)

    # Walk-Forward Sharpe per Window
    logger.info("[PLOT] Saving Walk-Forward Sharpe by window...")
    wf_plot_path = reports_dir / f"{stem}_walk_forward_sharpe.png"
    if wf_results["window_metrics"]:
        wf_sharpes = [w["test_sharpe"] for w in wf_results["window_metrics"]]
        wf_ids = [w["window_id"] for w in wf_results["window_metrics"]]
        plt.figure(figsize=(8, 4))
        plt.bar(wf_ids, wf_sharpes)
        plt.axhline(
            wf_results["oos_sharpe"],
            color="red",
            linestyle="--",
            label=f"OOS mean Sharpe = {wf_results['oos_sharpe']:.2f}",
        )
        plt.title("Walk-Forward OOS Sharpe by Window")
        plt.xlabel("Window ID")
        plt.ylabel("Sharpe")
        plt.legend()
        plt.tight_layout()
        plt.savefig(wf_plot_path, dpi=150)
        plt.close()
        logger.info("✅ Walk-Forward Sharpe plot saved to %s", wf_plot_path)

    # Multi-Asset Sharpe per Symbol (falls vorhanden)
    if multi_asset_info is not None and multi_asset_info.get("details"):
        logger.info("[PLOT] Saving Multi-Asset Sharpe by symbol...")
        ma_plot_path = reports_dir / f"{stem}_multi_asset_sharpe.png"
        df_ma = pd.DataFrame(multi_asset_info["details"])
        df_ma = df_ma.sort_values("sharpe", ascending=False)
        plt.figure(figsize=(8, 4))
        plt.bar(df_ma["symbol"], df_ma["sharpe"])
        plt.axhline(
            1.0,
            color="red",
            linestyle="--",
            label="Sharpe threshold = 1.0",
        )
        plt.title("Multi-Asset Sharpe by Symbol")
        plt.ylabel("Sharpe")
        plt.tight_layout()
        plt.legend()
        plt.savefig(ma_plot_path, dpi=150)
        plt.close()
        logger.info("✅ Multi-Asset Sharpe plot saved to %s", ma_plot_path)

    # === SCHRITT 11: Decision Gate ===
    logger.info("\n[STEP 11] Running Decision Gate...")
    gate = DecisionGate(config_path)
    gate_metrics = {
        "oos_sharpe": metrics.get("sharpe_ratio", 0.0),
        "max_drawdown": metrics.get("max_drawdown", 1.0),
        "mc_positive_prob": mc_results.get("mc_positive_prob", 0.0),
        "mt5_correlation": 0.9,
    }
    if multi_asset_info is not None:
        gate_metrics["multi_asset_hit_rate"] = multi_asset_info["hit_rate"]

    result = gate.evaluate(gate_metrics)
    logger.info("\n" + "=" * 60)
    logger.info("GATE RESULT: %s", result.status.value)
    logger.info("Confidence: %.1f%%", result.confidence * 100)
    logger.info("Reason: %s", result.reason)
    if result.violated_criteria:
        logger.warning("Violated criteria:")
        for criterion in result.violated_criteria:
            logger.warning(" ❌ %s", criterion)
    logger.info("=" * 60 + "\n")

    # === Summary-JSON ===
    summary = {
        "metrics": metrics,
        "cost_results": cost_results,
        "kelly": kelly_info,
        "kelly_oos": kelly_oos_info,        # NEU
        "vix_alignment": vix_alignment,
        "mc_results": mc_results,
        "walk_forward": wf_results,
        "multi_asset": multi_asset_info,
        "gate_result": {
            "status": result.status.value,
            "confidence": result.confidence,
            "reason": result.reason,
            "violated_criteria": result.violated_criteria,
        },
    }

    summary_path = reports_dir / f"{stem}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, default=str, indent=2)

    # === SCHRITT 12: Report speichern (mit Tabellen) ===
    logger.info("[STEP 12] Saving report...")
    report_path = reports_dir / f"{stem}_report.txt"
    Path("reports").mkdir(exist_ok=True)

    base_df = pd.DataFrame(
        {
            "Total Return": [f"{metrics['total_return']:.2%}"],
            "Sharpe": [f"{metrics['sharpe_ratio']:.2f}"],
            "MaxDD": [f"{metrics['max_drawdown']:.2%}"],
            "WinRate": [f"{metrics['win_rate']:.2%}"],
            "Trades": [metrics["total_trades"]],
        },
        index=["Base"],
    )

    cost_rows = []
    for name, m in cost_results.items():
        cost_rows.append(
            {
                "Scenario": name,
                "Total Return": f"{m['total_return']:.2%}",
                "Sharpe": f"{m['sharpe_ratio']:.2f}",
                "MaxDD": f"{m['max_drawdown']:.2%}",
                "PF": f"{m['profit_factor']:.2f}",
            }
        )
    cost_df = pd.DataFrame(cost_rows)

    kelly_df = pd.DataFrame(
        {
            "win_rate": [f"{kelly_info['win_rate']:.2%}"],
            "payoff": [f"{kelly_info['payoff_ratio']:.2f}"],
            "full": [f"{kelly_info['kelly_full']:.2%}"],
            "half": [f"{kelly_info['kelly_half']:.2%}"],
            "quarter": [f"{kelly_info['kelly_quarter']:.2%}"],
        },
        index=["Kelly"],
    )

    vix_rows = []
    for r_name, m in vix_alignment["regime_stats"].items():
        vix_rows.append(
            {
                "Regime": r_name,
                "Trades": m["n_trades"],
                "Total Return": f"{m['total_return']:.2%}",
                "Sharpe": f"{m['sharpe_ratio']:.2f}",
                "MaxDD": f"{m['max_drawdown']:.2%}",
                "PF": f"{m['profit_factor']:.2f}",
            }
        )
    vix_df = pd.DataFrame(vix_rows)

    wf_rows = []
    for w in wf_results["window_metrics"]:
        wf_rows.append(
            {
                "Window": w["window_id"],
                "Train": f"{w['train_start']} → {w['train_end']}",
                "Test": f"{w['test_start']} → {w['test_end']}",
                "Trades": w["test_n_trades"],
                "Sharpe": f"{w['test_sharpe']:.2f}",
                "PF": f"{w['test_profit_factor']:.2f}",
                "MaxDD": f"{w['test_max_dd']:.2%}",
                "Return": f"{w['test_total_return']:.2%}",
            }
        )
    wf_df = pd.DataFrame(wf_rows)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("QUANT VALIDATION PIPELINE REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Strategy: {metrics['strategy_name']}\n")
        f.write(
            f"Date Range: {metrics['date_range'][0]} to {metrics['date_range'][1]}\n\n"
        )

        f.write("METRICS:\n")
        f.write(base_df.to_markdown() + "\n\n")

        f.write("COST SCENARIOS:\n")
        f.write(cost_df.to_markdown(index=False) + "\n\n")

        f.write("KELLY SIZING:\n")
        f.write(kelly_df.to_markdown() + "\n\n")

        f.write("KELLY SIZING (Full Sample):\n")
        f.write(kelly_df.to_markdown() + "\n\n")

        kelly_oos_df = pd.DataFrame(
            {
                "win_rate": [f"{kelly_oos_info['win_rate']:.2%}"],
                "payoff": [f"{kelly_oos_info['payoff_ratio']:.2f}"],
                "full": [f"{kelly_oos_info['kelly_full']:.2%}"],
                "half": [f"{kelly_oos_info['kelly_half']:.2%}"],
                "quarter": [f"{kelly_oos_info['kelly_quarter']:.2%}"],
            },
            index=["Kelly_OOS"],
        )
        f.write("KELLY SIZING (Walk-Forward OOS only):\n")
        f.write(kelly_oos_df.to_markdown() + "\n\n")

        f.write("VIX REGIME PERFORMANCE:\n")
        f.write(vix_df.to_markdown(index=False) + "\n\n")

        f.write("WALK-FORWARD OOS WINDOWS:\n")
        if not wf_df.empty:
            f.write(wf_df.to_markdown(index=False) + "\n\n")
        else:
            f.write("  (no walk-forward windows)\n\n")

        if multi_asset_info is not None:
            df_ma = pd.DataFrame(multi_asset_info["details"])
            df_ma = df_ma.sort_values("sharpe", ascending=False)
            f.write("MULTI-ASSET OPTIMIZER (SYMBOL STATS):\n")
            f.write(
                df_ma[
                    [
                        "symbol",
                        "sharpe",
                        "profit",
                        "profit_factor",
                        "equity_dd_pct",
                        "trades",
                    ]
                ]
                .to_markdown(index=False)
                + "\n\n"
            )

        f.write("MONTE CARLO SUMMARY:\n")
        f.write(f" mc_positive_prob: {mc_results['mc_positive_prob']:.2%}\n")
        f.write(f" mc_median_return: {mc_results['mc_median_return']:.2%}\n")
        f.write(f" mc_p5_return: {mc_results['mc_p5_return']:.2%}\n")
        f.write(f" mc_p95_return: {mc_results['mc_p95_return']:.2%}\n")
        f.write(
            f" mc_median_max_dd: {mc_results['mc_median_max_dd']:.2%}\n"
        )
        f.write(f" mc_p95_max_dd: {mc_results['mc_p95_max_dd']:.2%}\n\n")

        f.write("DECISION GATE:\n")
        f.write(f" Status: {result.status.value}\n")
        f.write(f" Confidence: {result.confidence:.2%}\n")
        f.write(f" Reason: {result.reason}\n")
        if result.violated_criteria:
            f.write(" Violated criteria:\n")
            for c in result.violated_criteria:
                f.write(f" - {c}\n")

    logger.info("✅ Report saved to %s", report_path)
    logger.info("\n" + "=" * 60)
    logger.info("🎉 PIPELINE COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    from mt5_xml_to_csv_converter import batch_convert_raw

    parser = argparse.ArgumentParser(description="Quant Validation Pipeline")
    parser.add_argument(
        "--trades-file",
        type=str,
        required=False,
        help=(
            "Path to MT5 trades CSV file "
            "(wenn leer → auto: MT5-raw konvertieren + neueste *_trades_merged.csv nutzen)"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    if args.trades_file:
        trades_path = args.trades_file
    else:
        processed_dir = "data/processed"
        raw_dir = "data/raw"
        try:
            trades_path = find_latest_trades_csv(processed_dir)
        except FileNotFoundError:
            logger.info(
                "No *_trades_merged.csv found, running MT5 converter on raw files..."
            )
            batch_convert_raw(raw_dir=raw_dir, base_output_dir=processed_dir, overwrite=False)
            trades_path = find_latest_trades_csv(processed_dir)

    run_pipeline(trades_path, config_path=args.config)
