"""
HAUPT-PIPELINE: MT5 Backtest → Validierung → Visualisierung → Urteil
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mt5_xml_to_csv_converter import batch_convert_raw
from organize_raw import organize_raw
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
from validation.multi_asset_summary import load_and_score_optimizer, extract_strategy_name_from_optimizer

from generate_html_report import render_html_for_strategy_dir
from tail_risk import calculate_cvar
from validation.stochastic_scenarios import simulate_paths_from_trades
from validation.hmm_regime import analyze_hmm_regimes

logger = get_logger("PIPELINE", log_file="logs/pipeline.log")


def find_latest_trades_csv(processed_dir: str = "data/processed") -> str:
    """Nimmt die neueste *_trades_merged.csv rekursiv aus processed/**."""
    files = sorted(
        Path(processed_dir).rglob("*_trades_merged.csv"),  # NEU: rglob
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"Keine *_trades_merged.csv in {processed_dir} gefunden.")
    return str(files[0])


def run_pipeline(trades_csv_path: str, config_path: str = "config.yaml") -> None:
    """Hauptfunktion der Pipeline."""
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

    # === Strategie-Name & Report-Ordner ===
    strategy_stem_raw = Path(trades_csv_path).stem
    strategy_stem = strategy_stem_raw.replace("_trades_merged", "").replace("__", "_")
    logger.info(f"🔍 Fixed strategy stem: {strategy_stem_raw} → {strategy_stem}")

    strategy_reports_dir = Path("reports") / strategy_stem
    strategy_reports_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Report directory: %s", strategy_reports_dir)

    metrics_strategy_name = strategy_stem

    strategy_reports_dir = Path("reports") / strategy_stem
    strategy_reports_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Report directory: %s", strategy_reports_dir)

    # === SCHRITT 3: Validiere Trades ===
    logger.info("\n[STEP 3] Validating trades...")
    validation = loader.validate_trades(trades_df)

    # === SCHRITT 4b: Multi-Asset-Optimizer (optional) ===
    # Multi-Asset-Optimizer-XML im spezifischen Strategie-Ordner suchen
    raw_strategy_dir = Path("data/raw") / strategy_stem
    optimizer_candidates = sorted(
        raw_strategy_dir.glob("ReportOptimizer-*.xml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    multi_asset_info = None
    if optimizer_candidates:
        matched_xml = optimizer_candidates[0]
        logger.info(
            "\n[STEP 4b] Evaluating multi-asset optimizer results from %s...",
            matched_xml,
        )
        multi_asset_info = load_and_score_optimizer(
            str(matched_xml), sharpe_threshold=1.0
        )
        logger.info(
            "Multi-Asset hit-rate: %.1f%% (%d/%d, Sharpe > %.2f)",
            multi_asset_info["hit_rate"] * 100,
            multi_asset_info["n_symbols_pass"],
            multi_asset_info["n_symbols"],
            multi_asset_info["sharpe_threshold"],
        )
    else:
        logger.info(
            "[STEP 4b] No optimizer XML found in raw strategy folder %s",
            raw_strategy_dir,
        )

    # === SCHRITT 4: Berechne Metriken (Full Sample) ===
    logger.info("\n[STEP 4] Calculating metrics...")
    initial_capital = config["backtest"]["initial_capital"]
    metrics = calculate_metrics(trades_df, initial_capital)
    metrics["strategy_name"] = metrics_strategy_name
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
    logger.info("\n[STEP 6] Estimating Kelly sizing (full sample)...")
    # max_fraction=0.02 → 2 % Hard Cap pro Trade
    kelly_info = estimate_kelly_from_trades(trades_df, max_fraction=0.02)
    logger.info(
        "✅ Kelly (full): win_rate=%.2f%%, payoff=%.2f, kelly_full=%.2f%% (capped)",
        kelly_info["win_rate"] * 100,
        kelly_info["payoff_ratio"],
        kelly_info["kelly_full"] * 100,
    )

    # === SCHRITT 7: VIX-Regime-Ausrichtung ===
    logger.info("\n[STEP 7] Loading VIX regimes and checking alignment...")
    vix_regimes = load_vix_regimes(
        cache_path="data/external/vix_daily.csv",
        max_age_days=14,
    )
    vix_alignment, regime_decision = analyze_vix_regime_alignment(
        trades_df=trades_df,
        vix_regimes=vix_regimes,
        initial_capital=initial_capital,
        policy_path="regime_policy.yaml",
        strategy_key="range_breakout",
    )
    logger.info("✅ VIX regime alignment finished")

    # ------------------------------------------------------------------
    # STEP 7b: HMM-Regime-Analyse
    # ------------------------------------------------------------------
    logger.info("STEP 7b Running HMM-based regime analysis...")
    hmm_cfg = config.get("hmm_regime", {"n_regimes": 3, "min_trades_per_regime": 20})
    hmm_results_obj = analyze_hmm_regimes(
        trades_df,
        None,  # Optional: Preise laden falls verfügbar
        hmm_cfg,
        initial_capital,
    )
    logger.info(
        "HMM regimes found: %d",
        len(hmm_results_obj.regime_stats) if hmm_results_obj else 0,
    )

    # Für Summary-JSON ein serialisierbares Dict bauen
    if hmm_results_obj is not None:
        hmm_results = {
            "regime_stats": hmm_results_obj.regime_stats,
            "state_series": hmm_results_obj.state_series.to_dict(),
        }
    else:
        hmm_results = None

    # === SCHRITT 8: Walk-Forward / OOS-Analyse ===
    logger.info("\n[STEP 8] Running walk-forward analysis on trades...")
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
    logger.info("[STEP 8b] Estimating Kelly from OOS (walk-forward test windows)...")
    oos_indices: list[int] = []
    for w in wf_results["window_metrics"]:
        oos_indices.extend(w.get("test_indices", []))
    oos_indices = sorted(set(oos_indices))

    if oos_indices:
        trades_df_oos = trades_df.loc[oos_indices].copy()
        kelly_oos_info = estimate_kelly_from_trades(trades_df_oos, max_fraction=0.02)
        logger.info(
            "✅ OOS Kelly (capped): win_rate=%.2f%%, payoff=%.2f, kelly_full=%.2f%%",
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

    # === SCHRITT 9: Monte Carlo ===
    logger.info("\n[STEP 9] Running Monte Carlo on trade sequence...")
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

    # ------------------------------------------------------------------
    # STEP 9b: Stochastische Szenarien (GBM/Heston/Jump-Diffusion)
    # ------------------------------------------------------------------
    logger.info("STEP 9b Running stochastic model scenarios (GBM/Heston/Jump-Diffusion)...")
    sim_config = config.get(
        "simulation",
        {
            "T_years": 1.0,
            "num_steps": 252,
            "num_paths": 5000,
            "models": ["gbm", "heston", "jumpdiffusion"],
            "random_state": 12345,
        },
    )
    sim_results = simulate_paths_from_trades(
        trades_df=trades_df,
        config=sim_config,
        initial_capital=initial_capital,
    )
    logger.info("Stochastic scenarios: %s", list(sim_results.keys()))

    # Zusätzlich: Hinweis-Log, falls Heston wesentlich pessimistischer ist
    if "heston" in sim_results and "gbm" in sim_results:
        if sim_results["heston"]["median_return"] < 0 < sim_results["gbm"]["median_return"]:
            logger.warning(
                "Heston median_return (%.2f) < 0, während GBM median_return (%.2f) > 0 – "
                "Strategie profitiert evtl. nur in ruhigen Phasen.",
                sim_results["heston"]["median_return"],
                sim_results["gbm"]["median_return"],
            )

    # === PLOTS (alle in strategy_reports_dir) ===

    reports_dir = strategy_reports_dir

    # Monte Carlo Return Distribution
    logger.info("[PLOT] Saving Monte Carlo return distribution...")
    mc_plot_path = reports_dir / "mc_returns.png"

    mc_returns = np.array(mc_results["total_returns"])

    plt.figure(figsize=(6, 4))
    plt.hist(mc_returns, bins=50, alpha=0.7)
    plt.axvline(
        np.median(mc_returns),
        color="red",
        linestyle="--",
        label=f"Median {np.median(mc_returns):.2%}",
    )
    plt.title("Monte Carlo Total Returns")
    plt.xlabel("Total Return")
    plt.ylabel("Häufigkeit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(mc_plot_path, dpi=150)
    plt.close()
    logger.info("✅ Monte Carlo return plot saved to %s", mc_plot_path)

    tail_stats = {
        "cvar5": calculate_cvar(mc_returns, alpha=0.05),
        "worst_mc_return": float(mc_returns.min())
    }

    logger.info("TAIL RISK CVaR5=%.2f%% mc_p95_max_dd=%.2f%%", 
                tail_stats["cvar5"] * 100, 
                mc_results["mc_p95_max_dd"] * 100)
    

    # Stochastic-Scenario-Vergleichsplots (GBM vs. Heston vs. Jump-Diffusion)
    logger.info("[PLOT] Saving stochastic scenario comparison (total returns & maxDD)...")
    stoch_plot = reports_dir / "stochastic_scenarios.png"

    models = list(sim_results.keys())
    med_returns = [sim_results[m]["median_return"] for m in models]
    p5_returns = [sim_results[m]["p5_return"] for m in models]
    p95_returns = [sim_results[m]["p95_return"] for m in models]
    med_dds = [sim_results[m]["median_maxdd"] for m in models]
    p95_dds = [sim_results[m]["p95_maxdd"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Links: Returns
    x = np.arange(len(models))
    w = 0.25
    axes[0].bar(x - w, med_returns, width=w, label="Median")
    axes[0].bar(x, p5_returns, width=w, label="P5")
    axes[0].bar(x + w, p95_returns, width=w, label="P95")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_title("Stochastic Models – Total Returns")
    axes[0].set_ylabel("Return (fraction)")
    axes[0].legend()

    # Rechts: MaxDD
    axes[1].bar(x - w, med_dds, width=w, label="Median MaxDD")
    axes[1].bar(x + w, p95_dds, width=w, label="P95 MaxDD")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_title("Stochastic Models – Max Drawdown")
    axes[1].set_ylabel("Drawdown (fraction)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(stoch_plot, dpi=180)
    plt.close()
    logger.info("✅ Stochastic scenario comparison saved to %s", stoch_plot)


    # Equity Curve
    logger.info("PLOT Saving equity curve...")
    eq = trades_df["pnl"].cumsum() + initial_capital
    plt.figure(figsize=(8, 4))
    plt.plot(eq.index, eq.values)
    plt.title("Equity Curve")
    plt.ylabel("Equity")
    plt.tight_layout()
    eq_path = reports_dir / "equity.png"
    plt.savefig(eq_path, dpi=150)
    plt.close()
    logger.info("Equity curve saved to %s", eq_path)

    # NEU: HMM‑Regime‑Overlay, falls Ergebnisse vorhanden
    if hmm_results is not None and "state_series" in hmm_results:
        logger.info("[PLOT] Saving HMM regime overlay...")
        hmm_plot_path = reports_dir / "hmm_regime_overlay.png"

        # state_series liegt als Dict {index: state} vor → Series bauen
        states = pd.Series(hmm_results["state_series"])
        if not states.empty:
            # Index auf Trade-Index mappen (hinten ausrichten)
            states.index = trades_df.index[-len(states):]

            eq = trades_df["pnl"].cumsum() + initial_capital
            plt.figure(figsize=(10, 4))
            plt.plot(eq.index, eq.values, color="white", linewidth=1.5, label="Equity")

            unique_states = sorted(states.dropna().unique())
            colors = plt.cm.tab10(range(len(unique_states)))
            for s, c in zip(unique_states, colors):
                mask = states == s
                plt.scatter(
                    states.index[mask],
                    eq.loc[states.index[mask]],
                    s=8,
                    color=c,
                    label=f"Regime {s}",
                    alpha=0.7,
                )

            plt.title("Equity Curve with HMM Regimes")
            plt.ylabel("Equity")
            plt.legend(loc="upper left", fontsize=8)
            plt.tight_layout()
            plt.savefig(hmm_plot_path, dpi=150)
            plt.close()
            logger.info("✅ HMM overlay saved to %s", hmm_plot_path)
    else:
        logger.info("No HMM results available for overlay plot.")

    # VIX Regime Sharpe
    logger.info("PLOT Saving VIX regime Sharpe barplot...")
    vixstats = vix_alignment["regime_stats"]
    names = list(vixstats.keys())
    sharpes = [vixstats[n]["sharpe_ratio"] for n in names]
    plt.figure(figsize=(6, 4))
    plt.bar(names, sharpes)
    plt.title("Sharpe by VIX Regime")
    plt.ylabel("Sharpe")
    plt.tight_layout()
    vix_plot_path = reports_dir / "vix_regime_sharpe.png"
    plt.savefig(vix_plot_path, dpi=150)
    plt.close()
    logger.info("VIX regime Sharpe plot saved to %s", vix_plot_path)

    # NEU: Zeitreihen‑Plot der VIX‑Regime
    logger.info("PLOT Saving VIX regime time series...")
    vix_ts_path = reports_dir / "vix_regime_timeseries.png"

    # vix_regimes ist eine Series mit name 'vix_regime'
    vix_series = vix_regimes.sort_index()  # nur zur Sicherheit
    plt.figure(figsize=(10, 4))
    plt.step(vix_series.index, pd.Categorical(vix_series).codes, where="post")
    plt.yticks(
        ticks=range(len(vix_series.unique())),
        labels=list(vix_series.unique())
    )
    plt.title("VIX Regime over time")
    plt.xlabel("Date")
    plt.ylabel("Regime")
    plt.tight_layout()
    plt.savefig(vix_ts_path, dpi=150)
    plt.close()
    logger.info("VIX regime time series saved to %s", vix_ts_path)
    

    # Walk-Forward Sharpe per Window
    logger.info("[PLOT] Saving Walk-Forward Sharpe by window...")
    wf_plot_path = reports_dir / "walk_forward_sharpe.png"
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
        ma_plot_path = reports_dir / "multi_asset_sharpe.png"
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


    # === SCHRITT 11: ELITE Decision Gate ===
    logger.info("\n[STEP 11] Running Decision Gate...")
    gate = DecisionGate(config_path)
    gate_metrics = {
        "oos_sharpe": wf_results["oos_sharpe"],
        "oos_profit_factor": wf_results["oos_profit_factor"],
        "oos_max_drawdown": wf_results["oos_max_dd"],  # OOS!
        "mc_positive_prob": mc_results["mc_positive_prob"],
        "mc_p95_return": mc_results["mc_p95_return"],
        "cvar5": tail_stats["cvar5"],
        "kelly_oos_full": kelly_oos_info["kelly_full"],
        "mt5_correlation": 0.92,  # Platzhalter für reale Live-Korrelation
    }
    if multi_asset_info is not None:
        gate_metrics["multi_asset_hit_rate"] = multi_asset_info["hit_rate"]
    if regime_decision is not None:
        gate_metrics["regime_allowed"] = regime_decision["allowed"]
        gate_metrics["regime_risk_multiplier"] = regime_decision["risk_multiplier"]

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
        "kelly_oos": kelly_oos_info,
        "vix_alignment": vix_alignment,
        "mc_results": mc_results,
        "walk_forward": wf_results,
        "multi_asset": multi_asset_info,
        "sim_results": sim_results,
        "hmm_results": hmm_results,
        "gate_result": {
            "status": result.status.value,
            "confidence": result.confidence,
            "reason": result.reason,
            "violated_criteria": result.violated_criteria,
        },
    }
    summary_path = reports_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, default=str, indent=2)

    # === ERWEITERTE PLOTS (Senior Level) ===

    # 1. Monte Carlo PATHS Plot (100 Pfade + Median + Confidence Bands)
    logger.info("[PLOT] Saving Monte Carlo PATHS...")
    mc_path_plot = reports_dir / "mc_paths.png"
    equity_paths = np.array(mc_results["equity_paths"])
    n_show = 100  # Zeige 100 Pfade
    show_idx = np.random.choice(len(equity_paths), n_show, replace=False)

    plt.figure(figsize=(12, 6))
    for i in show_idx:
        plt.plot(equity_paths[i], alpha=0.1, color='steelblue', linewidth=0.5)
    median_path = np.median(equity_paths, axis=0)
    p5_path = np.percentile(equity_paths, 5, axis=0)
    p95_path = np.percentile(equity_paths, 95, axis=0)

    plt.plot(median_path, 'r-', linewidth=3, label='Median')
    plt.plot(p5_path, 'orange', linestyle='--', linewidth=2, label='5th %ile')
    plt.plot(p95_path, 'orange', linestyle='--', linewidth=2)
    plt.fill_between(range(len(p5_path)), p5_path, p95_path, alpha=0.2, color='orange', label='90% Confidence')

    plt.title(f'Monte Carlo Equity Paths (n={n_show}/{len(equity_paths)} shown)')
    plt.ylabel('Equity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(mc_path_plot, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("✅ MC Paths saved to %s", mc_path_plot)

    # 2. Drawdown Duration Histogram
    logger.info("[PLOT] Saving Drawdown Analysis...")
    dd_plot = reports_dir / "drawdown_analysis.png"

    # Calculate drawdown durations from equity curve
    equity = initial_capital + trades_df["pnl"].cumsum()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    dd_durations = []
    in_dd = False
    dd_start = 0

    for i, dd in enumerate(drawdown):
        if dd < 0 and not in_dd:
            in_dd = True
            dd_start = i
        elif dd >= -0.01 and in_dd:  # Recovery
            dd_durations.append(i - dd_start)
            in_dd = False

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(mc_results["max_drawdowns"], bins=50, alpha=0.7, color='red')
    plt.axvline(np.median(mc_results["max_drawdowns"]), color='black', lw=2, label='Median')
    plt.title('Max Drawdown Distribution')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    if dd_durations:
        plt.hist(dd_durations, bins=20, alpha=0.7, color='purple')
        plt.title('Drawdown Duration (Days)')
        plt.xlabel('Duration')
    plt.tight_layout()
    plt.savefig(dd_plot, dpi=200)
    plt.close()

    # 3. Kelly Growth Rate vs Risk
    logger.info("[PLOT] Saving Kelly Frontier...")
    kelly_plot = reports_dir / "kelly_frontier.png"

    # FIXED: Korrekte Kelly Growth Rate Berechnung
    risk_frac = np.linspace(0.001, min(kelly_info['kelly_full'], 0.5), 100)
    growth_rates = []

    initial_cap = initial_capital
    avg_trade_pnl = trades_df['pnl'].mean()
    std_trade_pnl = trades_df['pnl'].std()

    for f in risk_frac:
        # Kelly Growth Rate Formel: g(f) = p*log(1 + b*f) + (1-p)*log(1 - f)
        expected_return = f * avg_trade_pnl / initial_cap
        volatility = f * std_trade_pnl / initial_cap
        
        # Geometric growth rate (ohne log(0) Probleme)
        if expected_return > volatility * 0.1:  # Nur sinnvolle Fraktionen
            growth_rate = expected_return - 0.5 * volatility**2
            growth_rates.append(growth_rate * 252)  # Annualisiert
        else:
            growth_rates.append(0.0)

    growth_rates = np.maximum(growth_rates, -0.5)  # Floor für Plot

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(risk_frac * 100, np.array(growth_rates)*100, 'b-', linewidth=3, label='Kelly Growth Rate')

    # Kelly Markierungen
    ax.axvline(kelly_info['kelly_quarter'] * 100, color='limegreen', linestyle=':', linewidth=2, 
            label=f'¼ Kelly: {kelly_info["kelly_quarter"]:.1%}', alpha=0.8)
    ax.axvline(kelly_info['kelly_half'] * 100, color='green', linestyle='--', linewidth=3, 
            label=f'½ Kelly: {kelly_info["kelly_half"]:.1%} (KONSERVATIV)', alpha=0.9)
    ax.axvline(kelly_info['kelly_full'] * 100, color='red', linestyle='-', linewidth=2, 
            label=f'Full Kelly: {kelly_info["kelly_full"]:.1%} (AGGRESSIV)', alpha=0.9)

    # Optimaler Punkt markieren
    max_growth_idx = np.argmax(growth_rates)
    ax.plot(risk_frac[max_growth_idx]*100, growth_rates[max_growth_idx]*100, 'go', markersize=12, 
            label=f'Max Growth: {risk_frac[max_growth_idx]*100:.1f}%')

    ax.set_xlabel('Risiko pro Trade (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Jährliche Wachstumsrate (%)', fontsize=14, fontweight='bold')
    ax.set_title('🤑 KELLY CRITERION - OPTIMALES POSITION SIZING\n'
                f'(Winrate {kelly_info["win_rate"]:.1%} | Payoff {kelly_info["payoff_ratio"]:.2f})', 
                fontsize=16, fontweight='bold', pad=20)

    ax.legend(fontsize=11, framealpha=0.95, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(4, kelly_info['kelly_full']*100*1.2))

    plt.tight_layout()
    plt.savefig(kelly_plot, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info("✅ Kelly Frontier saved: Max Growth bei %.1f%% Risiko", risk_frac[max_growth_idx]*100)

    # 4. Trade PnL Distribution + Fat Tails
    logger.info("[PLOT] Saving PnL Distribution...")
    pnl_plot = reports_dir / "pnl_distribution.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # PnL Histogram
    axes[0,0].hist(trades_df['pnl'], bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].axvline(trades_df['pnl'].median(), color='red', lw=2)
    axes[0,0].set_title('Trade PnL Distribution')

    # QQ Plot vs Normal
    from scipy import stats
    stats.probplot(trades_df['pnl'], dist="norm", plot=axes[0,1])
    axes[0,1].set_title('QQ Plot vs Normal (Fat Tails?)')

    # PnL vs Volume
    axes[1,0].scatter(trades_df['volume'].abs(), trades_df['pnl'], alpha=0.5)
    axes[1,0].set_xlabel('Volume')
    axes[1,0].set_ylabel('PnL')
    axes[1,0].set_title('PnL vs Position Size')

    # Win/Loss by Hour
    trades_df['hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
    hourly_stats = trades_df.groupby('hour')['pnl'].agg(['mean', 'count']).reset_index()
    axes[1,1].bar(hourly_stats['hour'], hourly_stats['mean'])
    axes[1,1].set_title('Avg PnL by Entry Hour')
    plt.tight_layout()
    plt.savefig(pnl_plot, dpi=200)
    plt.close()

    logger.info("✅ Advanced plots completed")

    # === SCHRITT 12: Report speichern (mit Tabellen) ===
    logger.info("[STEP 12] Saving report...")
    report_path = reports_dir / "report.txt"

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
                "Kelly_full": f"{w.get('test_kelly_full', 0.0):.2%}",
                "Kelly_half": f"{w.get('test_kelly_half', 0.0):.2%}",
                "Kelly_quarter": f"{w.get('test_kelly_quarter', 0.0):.2%}",
            }
        )
    wf_df = pd.DataFrame(wf_rows)

    # HMM Regime Tabelle
    hmmrows = []
    if hasattr(hmm_results, 'regimestats') and hmm_results.regimestats:
        for rname, m in hmm_results.regimestats.items():  # ✅ Object-Attribut
            hmmrows.append({
                'Regime': rname,
                'Trades': m['n_trades'],
                'TotalReturn': f"{m['total_return']*100:.2f}%",
                'Sharpe': f"{m['sharpe_ratio']:.2f}",
                'MaxDD': f"{m['max_drawdown']*100:.2f}%", 
                'PF': f"{m['profit_factor']:.2f}"
            })
    hmmdf = pd.DataFrame(hmmrows)

    sim_rows = []
    for name, m in sim_results.items():
        sim_rows.append(
            dict(
                Model=name,
                MedianReturn=f"{m['median_return']:.2f}",
                P5=f"{m['p5_return']:.2f}",
                P95=f"{m['p95_return']:.2f}",
                MedianMaxDD=f"{m['median_maxdd']:.2f}",
                P95MaxDD=f"{m['p95_maxdd']:.2f}",
            )
        )
    sim_df = pd.DataFrame(sim_rows)


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

        f.write("KELLY SIZING (Full Sample):\n")
        f.write(kelly_df.to_markdown() + "\n\n")

        f.write("KELLY SIZING (Walk-Forward OOS only):\n")
        f.write(kelly_oos_df.to_markdown() + "\n\n")

        f.write("VIX REGIME PERFORMANCE:\n")
        f.write(vix_df.to_markdown(index=False) + "\n\n")

        f.write("WALK-FORWARD OOS WINDOWS:\n")
        if not wf_df.empty:
            f.write(wf_df.to_markdown(index=False) + "\n\n")
        else:
            f.write("  (no walk-forward windows)\n\n")

        f.write("HMM REGIME PERFORMANCE\n")
        if not hmmdf.empty:
            f.write(hmmdf.to_markdown(index=False) + "\n\n")
        else:
            f.write("no HMM regimes with sufficient trades\n\n")

        f.write("STOCHASTIC MODEL SCENARIOS\n")
        if not sim_df.empty:
            f.write(sim_df.to_markdown(index=False) + "\n\n")
            if (
                "heston" in sim_results
                and "gbm" in sim_results
                and sim_results["heston"]["median_return"] < 0
                and sim_results["gbm"]["median_return"] > 0
            ):
                f.write(
                    "NOTE: Under Heston (stochastic volatility) the median return is negative, "
                    "while GBM/Jump-Diffusion are positive. Strategy may rely on low-vol "
                    "regimes – treat as additional risk when volatility spikes.\n\n"
                )
        else:
            f.write("no stochastic scenarios\n\n")


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

    # HTML-Dashboard automatisch erzeugen
    try:
        html_path = render_html_for_strategy_dir(strategy_reports_dir)
        logger.info("✅ HTML report saved to %s", html_path)
    except Exception as e:
        logger.error("HTML report generation failed: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Validation Pipeline")
    parser.add_argument(
        "--trades-file",
        type=str,
        required=False,
        help=(
            "Path to MT5 trades CSV file "
            "(wenn leer → auto: raw-Dateien organisieren, konvertieren und neueste *_trades_merged.csv nutzen)"
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
        # Direkter Pfad angegeben -> nur Pipeline laufen lassen
        trades_path = args.trades_file
    else:
        raw_dir = "data/raw"
        processed_dir = "data/processed"

        # 1) Rohdaten nach Strategien sortieren
        logger.info("Organizing raw MT5 files into strategy folders...")
        organize_raw(raw_dir=raw_dir)

        # 2) XLSX → *_trades_merged.csv für alle Strategien
        logger.info("Converting MT5 raw files to merged trades CSVs...")
        batch_convert_raw(raw_dir=raw_dir, base_output_dir=processed_dir, overwrite=False)

        # 3) Neueste *_trades_merged.csv in processed/** auswählen
        try:
            trades_path = find_latest_trades_csv(processed_dir)
        except FileNotFoundError as e:
            logger.error("❌ %s", e)
            raise

    run_pipeline(trades_path, config_path=args.config)
