"""
HAUPT-PIPELINE
MT5 Backtest  -> Validierung -> Visualisierung -> Urteil
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from mt5_xml_to_csv_converter import batch_convert_raw
from organize_raw import organize_raw
from trades_loader import MT5TradesLoader
from backtest.metrics import calculate_metrics
from validation.gates import DecisionGate
from utils.logger import get_logger
from utils.config import load_config
from validation.cost_scenarios import run_cost_scenarios
from kelly import estimate_kelly_from_trades
from vix_loader import load_vix_regimes
from regime_alignment import analyze_vix_regime_alignment
from monte_carlo import run_monte_carlo_on_trades
from walk_forward import run_walk_forward_analysis
from multi_asset_summary import (
    load_and_score_optimizer,
    extract_strategy_name_from_optimizer,
)
from generate_html_report import render_html_for_strategy_dir
from tail_risk import calculate_cvar

# NEU
from validation.stochastic_scenarios import simulate_paths_from_trades
from validation.hmm_regimes import analyze_hmm_regimes

logger = get_logger("PIPELINE", logfile="logs/pipeline.log")


def find_latest_trades_csv(processed_dir: str = "data/processed") -> str:
    """
    Nimmt die neueste trades_merged.csv rekursiv aus processed.
    """
    files = sorted(
        Path(processed_dir).rglob("trades_merged.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"Keine trades_merged.csv in {processed_dir} gefunden.")
    return str(files[0])


def run_pipeline(trades_csv_path: str, config_path: str = "config.yaml") -> None:
    """
    Hauptfunktion der Pipeline.
    """
    logger.info("=" * 60)
    logger.info("QUANT VALIDATION PIPELINE STARTED")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # STEP 1: Konfiguration laden
    # ------------------------------------------------------------------
    logger.info("STEP 1 Loading configuration...")
    config = load_config(config_path)
    logger.info("Config loaded: %s", config.get("project_name", "N/A"))

    initial_capital = float(config["backtest"]["initial_capital"])

    # ------------------------------------------------------------------
    # STEP 2: MT5 Trades importieren
    # ------------------------------------------------------------------
    logger.info("STEP 2 Importing MT5 trades...")
    loader = MT5TradesLoader()
    try:
        trades_df = loader.load_trades(trades_csv_path)
        logger.info("Loaded %d trades from %s", len(trades_df), trades_csv_path)
    except FileNotFoundError as e:
        logger.error("Error loading trades: %s", e)
        return

    # Strategie-Name + Report-Ordner
    strategy_stem_raw = Path(trades_csv_path).stem
    strategy_stem = strategy_stem_raw.replace("trades_merged", "").replace("_", "").strip()
    logger.info("Fixed strategy stem: raw=%s -> %s", strategy_stem_raw, strategy_stem)

    strategy_reports_dir = Path("reports") / strategy_stem
    strategy_reports_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Report directory: %s", strategy_reports_dir)

    metrics_strategy_name = strategy_stem

    # ------------------------------------------------------------------
    # STEP 3: Trades validieren (Loader-spezifisch)
    # ------------------------------------------------------------------
    logger.info("STEP 3 Validating trades...")
    loader.validate_trades(trades_df)

    # ------------------------------------------------------------------
    # STEP 4a: Multi-Asset-Optimizer-XML (optional)
    # ------------------------------------------------------------------
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
            "STEP 4b Evaluating multi-asset optimizer results from %s",
            matched_xml,
        )
        multi_asset_info = load_and_score_optimizer(
            str(matched_xml),
            sharpe_threshold=1.0,
        )
        logger.info(
            "Multi-Asset hit-rate %.1f%% (%d/%d >= Sharpe %.2f)",
            multi_asset_info["hit_rate"] * 100,
            multi_asset_info["n_symbols_pass"],
            multi_asset_info["n_symbols"],
            multi_asset_info["sharpe_threshold"],
        )
    else:
        logger.info(
            "STEP 4b No optimizer XML found in raw strategy folder %s",
            raw_strategy_dir,
        )

    # ------------------------------------------------------------------
    # STEP 4: Metriken Full-Sample
    # ------------------------------------------------------------------
    logger.info("STEP 4 Calculating metrics...")
    metrics = calculate_metrics(trades_df, initial_capital=initial_capital)
    metrics["strategy_name"] = metrics_strategy_name
    metrics["total_trades"] = len(trades_df)
    metrics["date_range"] = (
        trades_df["entry_time"].min(),
        trades_df["exit_time"].max(),
    )

    logger.info("Metrics calculated.")
    logger.info("Total Return: %.2f%%", metrics["total_return"] * 100)
    logger.info("Sharpe Ratio: %.2f", metrics["sharpe_ratio"])
    logger.info("Max Drawdown: %.2f%%", metrics["max_drawdown"] * 100)
    logger.info("Win Rate: %.2f%%", metrics["win_rate"] * 100)

    # ------------------------------------------------------------------
    # STEP 5: Kosten-/Slippage-Szenarien
    # ------------------------------------------------------------------
    logger.info("STEP 5 Running cost/slippage scenarios...")
    cost_scenarios = {"base": 1.0, "cost_plus25": 0.75, "cost_plus50": 0.5}
    cost_results = run_cost_scenarios(
        trades_df,
        initial_capital=initial_capital,
        scenarios=cost_scenarios,
    )
    logger.info("Cost scenarios finished.")

    # ------------------------------------------------------------------
    # STEP 6: Kelly Sizing Full Sample
    # ------------------------------------------------------------------
    logger.info("STEP 6 Estimating Kelly sizing full sample...")
    kelly_info = estimate_kelly_from_trades(trades_df)
    logger.info("Kelly estimation finished.")

    # ------------------------------------------------------------------
    # STEP 7: VIX-Regime-Ausrichtung
    # ------------------------------------------------------------------
    logger.info("STEP 7 Loading VIX regimes and checking alignment...")
    vix_regimes = load_vix_regimes(
        cache_path="data/external/vix_daily.csv",
        max_age_days=14,
    )
    vix_alignment, regime_decision = analyze_vix_regime_alignment(
        trades_df=trades_df,
        vix_regimes=vix_regimes,
        initial_capital=initial_capital,
        policy_path="regime_policy.yaml",
        strategy_key="rangebreakout",
    )
    logger.info("VIX regime alignment finished.")

    # ------------------------------------------------------------------
    # STEP 7b: HMM-Regime-Analyse
    # ------------------------------------------------------------------
    logger.info("STEP 7b Running HMM-based regime analysis...")
    hmm_cfg = config.get("hmmregime", {})
    # In dieser Version nutzen wir Equity-Returns, daher kein prices_df
    hmm_results = analyze_hmm_regimes(
        trades_df=trades_df,
        prices_df=None,
        config={
            **hmm_cfg,
            "initial_capital": initial_capital,
        },
    )
    logger.info("HMM regime analysis finished.")

    # ------------------------------------------------------------------
    # STEP 8: Monte Carlo auf Trade-Sequenz
    # ------------------------------------------------------------------
    logger.info("STEP 8 Running Monte Carlo on trade sequence...")
    mc_results = run_monte_carlo_on_trades(
        trades_df=trades_df,
        initial_capital=initial_capital,
        n_sims=config["validation"]["monte_carlo"]["num_simulations"],
        random_state=42,
    )
    logger.info("Monte Carlo finished.")
    logger.info("mc_positive_prob: %.2f%%", mc_results["mc_positive_prob"] * 100)
    logger.info("mc_median_return: %.2f%%", mc_results["mc_median_return"] * 100)
    logger.info("mc_p5_return: %.2f%%", mc_results["mc_p5_return"] * 100)
    logger.info("mc_p95_return: %.2f%%", mc_results["mc_p95_return"] * 100)
    logger.info("mc_median_maxdd: %.2f%%", mc_results["mc_median_maxdd"] * 100)
    logger.info("mc_p95_maxdd: %.2f%%", mc_results["mc_p95_maxdd"] * 100)

    # Tail-Risk (CVaR) auf MC-Returns
    mc_returns_array = np.array(mc_results["total_returns"])
    tail_stats = calculate_cvar(mc_returns_array, alpha=0.05)
    worst_mc_return = float(mc_returns_array.min())
    logger.info(
        "TAIL RISK CVaR5: %.2f%%, worst_MC_return: %.2f%%",
        tail_stats["cvar"] * 100,
        worst_mc_return * 100,
    )

    # ------------------------------------------------------------------
    # STEP 8b: Stochastische Szenarien (GBM/Heston/Jump-Diffusion)
    # ------------------------------------------------------------------
    logger.info(
        "STEP 8b Running stochastic model scenarios (GBM/Heston/Jump-Diffusion)..."
    )
    sim_config = {
        **config.get("simulation", {}),
        "initial_capital": initial_capital,
    }
    sim_results = simulate_paths_from_trades(
        trades_df=trades_df,
        config=sim_config,
    )
    logger.info("Stochastic scenarios finished.")

    # ------------------------------------------------------------------
    # STEP 9: Walk-Forward-Analyse
    # ------------------------------------------------------------------
    logger.info("STEP 9 Running walk-forward analysis on trades...")
    wf_cfg = config.get("walkforward", {})
    wf_results = run_walk_forward_analysis(
        trades_df=trades_df,
        initial_capital=initial_capital,
        train_days=wf_cfg.get("train_window", 252),
        test_days=wf_cfg.get("test_window", 63),
        step_days=wf_cfg.get("step_size", 21),
    )
    logger.info(
        "Walk-forward finished: n_windows=%d, OOS Sharpe=%.2f, PF=%.2f, MaxDD=%.2f%%",
        wf_results["n_windows"],
        wf_results["oos_sharpe"],
        wf_results["oos_profit_factor"],
        wf_results["oos_maxdd"] * 100,
    )

    # ------------------------------------------------------------------
    # STEP 9b: Kelly aus OOS Walk-Forward
    # ------------------------------------------------------------------
    logger.info("STEP 9b Estimating Kelly from OOS walk-forward test windows...")
    oos_indices = []
    for w in wf_results.get("window_metrics", []):
        oos_indices.extend(w.get("test_indices", []))
    oos_indices = sorted(set(oos_indices))

    if oos_indices:
        trades_df_oos = trades_df.loc[oos_indices].copy()
        kelly_oos_info = estimate_kelly_from_trades(trades_df_oos)
        logger.info(
            "OOS Kelly: win_rate=%.2f%%, payoff=%.2f, kelly_full=%.2f%%",
            kelly_oos_info["win_rate"] * 100,
            kelly_oos_info["payoff_ratio"],
            kelly_oos_info["kelly_full"] * 100,
        )
    else:
        kelly_oos_info = dict(
            win_rate=0.0,
            payoff_ratio=0.0,
            kelly_full=0.0,
            kelly_half=0.0,
            kelly_quarter=0.0,
        )
        logger.info("No OOS trades for Kelly estimation.")

    # ------------------------------------------------------------------
    # PLOTS (Monte Carlo, Equity Curve, Regime-Plots etc.)
    # ------------------------------------------------------------------
    reports_dir = strategy_reports_dir

    # Monte-Carlo Return Distribution
    logger.info("PLOT Saving Monte Carlo return distribution...")
    mc_plot_path = reports_dir / "mc_returns.png"
    plt.figure(figsize=(6, 4))
    plt.hist(mc_returns_array, bins=50, alpha=0.7)
    plt.axvline(np.median(mc_returns_array), color="red", linestyle="--",
                label=f"Median {np.median(mc_returns_array):.2f}")
    plt.title("Monte Carlo Total Returns")
    plt.xlabel("Total Return")
    plt.ylabel("Häufigkeit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(mc_plot_path, dpi=150)
    plt.close()
    logger.info("Monte Carlo return plot saved to %s", mc_plot_path)

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

    # VIX Regime Sharpe
    logger.info("PLOT Saving VIX regime Sharpe barplot...")
    vix_stats = vix_alignment["regime_stats"]
    names = list(vix_stats.keys())
    sharpes = [vix_stats[n]["sharpe_ratio"] for n in names]
    plt.figure(figsize=(6, 4))
    plt.bar(names, sharpes)
    plt.title("Sharpe by VIX Regime")
    plt.ylabel("Sharpe")
    plt.tight_layout()
    vix_plot_path = reports_dir / "vix_regime_sharpe.png"
    plt.savefig(vix_plot_path, dpi=150)
    plt.close()
    logger.info("VIX regime Sharpe plot saved to %s", vix_plot_path)

    # Walk-Forward Sharpe per Window
    logger.info("PLOT Saving Walk-Forward Sharpe by window...")
    wf_plot_path = reports_dir / "walkforward_sharpe.png"
    if wf_results.get("window_metrics"):
        wf_sharpes = [w["test_sharpe"] for w in wf_results["window_metrics"]]
        wf_ids = [w["window_id"] for w in wf_results["window_metrics"]]
        plt.figure(figsize=(8, 4))
        plt.bar(wf_ids, wf_sharpes)
        plt.axhline(
            wf_results["oos_sharpe"],
            color="red",
            linestyle="--",
            label=f"OOS mean Sharpe {wf_results['oos_sharpe']:.2f}",
        )
        plt.title("Walk-Forward OOS Sharpe by Window")
        plt.xlabel("Window ID")
        plt.ylabel("Sharpe")
        plt.legend()
        plt.tight_layout()
        plt.savefig(wf_plot_path, dpi=150)
        plt.close()
    logger.info("Walk-Forward Sharpe plot saved to %s", wf_plot_path)

    # Multi-Asset Sharpe per Symbol
    if multi_asset_info is not None and multi_asset_info.get("details"):
        logger.info("PLOT Saving Multi-Asset Sharpe by symbol...")
        ma_plot_path = reports_dir / "multiasset_sharpe.png"
        df_ma = pd.DataFrame(multi_asset_info["details"])
        df_ma = df_ma.sort_values("sharpe", ascending=False)
        plt.figure(figsize=(8, 4))
        plt.bar(df_ma["symbol"], df_ma["sharpe"])
        plt.axhline(
            multi_asset_info["sharpe_threshold"],
            color="red",
            linestyle="--",
            label=f"Sharpe threshold {multi_asset_info['sharpe_threshold']:.2f}",
        )
        plt.title("Multi-Asset Sharpe by Symbol")
        plt.ylabel("Sharpe")
        plt.tight_layout()
        plt.legend()
        plt.savefig(ma_plot_path, dpi=150)
        plt.close()
        logger.info("Multi-Asset Sharpe plot saved to %s", ma_plot_path)

    # MC Equity Paths
    logger.info("PLOT Saving Monte Carlo PATHS...")
    mc_path_plot = reports_dir / "mc_paths.png"
    equity_paths = np.array(mc_results["equity_paths"])
    n_show = min(100, len(equity_paths))
    show_idx = np.random.choice(len(equity_paths), n_show, replace=False)

    plt.figure(figsize=(12, 6))
    for i in show_idx:
        plt.plot(equity_paths[i], alpha=0.1, color="steelblue", linewidth=0.5)

    median_path = np.median(equity_paths, axis=0)
    p5_path = np.percentile(equity_paths, 5, axis=0)
    p95_path = np.percentile(equity_paths, 95, axis=0)

    plt.plot(median_path, "r-", linewidth=3, label="Median")
    plt.plot(p5_path, "orange", linestyle="--", linewidth=2, label="5th %ile")
    plt.plot(p95_path, "orange", linestyle="--", linewidth=2)
    plt.fill_between(range(len(p5_path)), p5_path, p95_path, alpha=0.2, color="orange",
                     label="90% Confidence")
    plt.title(f"Monte Carlo Equity Paths ({n_show}/{len(equity_paths)} shown)")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(mc_path_plot, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("MC Paths saved to %s", mc_path_plot)

    # Drawdown-Analyse
    logger.info("PLOT Saving Drawdown Analysis...")
    dd_plot = reports_dir / "drawdown_analysis.png"
    equity_curve = trades_df["pnl"].cumsum() + initial_capital
    peak = equity_curve.cummax()
    drawdown = equity_curve - peak
    peak_nonzero = peak.replace(0, np.nan)
    drawdown_pct = drawdown / peak_nonzero

    dd_durations = []
    in_dd = False
    dd_start = 0
    for i, dd in enumerate(drawdown_pct):
        if dd < 0 and not in_dd:
            in_dd = True
            dd_start = i
        elif dd >= -0.01 and in_dd:
            dd_durations.append(i - dd_start)
            in_dd = False

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(mc_results["max_drawdowns"], bins=50, alpha=0.7, color="red")
    plt.axvline(np.median(mc_results["max_drawdowns"]), color="black", lw=2,
                label="Median")
    plt.title("Max Drawdown Distribution")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    if dd_durations:
        plt.hist(dd_durations, bins=20, alpha=0.7, color="purple")
    plt.title("Drawdown Duration (Days)")
    plt.xlabel("Duration")
    plt.tight_layout()
    plt.savefig(dd_plot, dpi=200)
    plt.close()
    logger.info("Drawdown analysis saved to %s", dd_plot)

    # Kelly Frontier (Full-Sample)
    logger.info("PLOT Saving Kelly Frontier...")
    kelly_plot = reports_dir / "kelly_frontier.png"
    risk_frac = np.linspace(0.001, min(kelly_info["kelly_full"], 0.5), 100)
    growth_rates = []
    avg_trade_pnl = trades_df["pnl"].mean()
    std_trade_pnl = trades_df["pnl"].std(ddof=1)

    for f in risk_frac:
        expected_return = f * avg_trade_pnl / initial_capital
        volatility = f * std_trade_pnl / initial_capital
        if volatility > 0.1:
            growth_rate = (expected_return - 0.5 * volatility ** 2) * 252
        else:
            growth_rate = 0.0
        growth_rates.append(growth_rate)
    growth_rates = np.maximum(growth_rates, -0.5)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(risk_frac * 100, np.array(growth_rates) * 100, "b-", linewidth=3,
            label="Kelly Growth Rate")
    ax.axvline(kelly_info["kelly_quarter"] * 100, color="limegreen", linestyle="-",
               linewidth=2, label=f"¼ Kelly {kelly_info['kelly_quarter'] * 100:.1f}%")
    ax.axvline(kelly_info["kelly_half"] * 100, color="green", linestyle="--",
               linewidth=3, label=f"½ Kelly {kelly_info['kelly_half'] * 100:.1f}%")
    ax.axvline(kelly_info["kelly_full"] * 100, color="red", linestyle="-",
               linewidth=2, label=f"Full Kelly {kelly_info['kelly_full'] * 100:.1f}%")

    max_growth_idx = int(np.argmax(growth_rates))
    ax.plot(
        risk_frac[max_growth_idx] * 100,
        growth_rates[max_growth_idx] * 100,
        "go",
        markersize=12,
        label=f"Max Growth {risk_frac[max_growth_idx] * 100:.1f}%",
    )

    ax.set_xlabel("Risiko pro Trade (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Jährliche Wachstumsrate (%)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"KELLY CRITERION - OPTIMALES POSITION SIZING\n"
        f"Winrate {kelly_info['win_rate'] * 100:.1f}%, "
        f"Payoff {kelly_info['payoff_ratio']:.2f}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.legend(fontsize=11, framealpha=0.95, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(10, kelly_info["kelly_full"] * 100 * 1.2))
    plt.tight_layout()
    plt.savefig(kelly_plot, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(
        "Kelly Frontier saved. Max Growth at %.1f%% risk.",
        risk_frac[max_growth_idx] * 100,
    )

    # PnL Distribution & weitere Plots
    logger.info("PLOT Saving PnL Distribution...")
    pnl_plot = reports_dir / "pnl_distribution.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1) Trade PnL Histogram
    axes[0, 0].hist(trades_df["pnl"], bins=50, alpha=0.7, edgecolor="black")
    axes[0, 0].axvline(trades_df["pnl"].median(), color="red", lw=2)
    axes[0, 0].set_title("Trade PnL Distribution")

    # 2) QQ-Plot vs Normal
    stats.probplot(trades_df["pnl"], dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("QQ Plot vs Normal (Fat Tails?)")

    # 3) PnL vs Volume
    axes[1, 0].scatter(trades_df["volume"].abs(), trades_df["pnl"], alpha=0.5)
    axes[1, 0].set_xlabel("Volume")
    axes[1, 0].set_ylabel("PnL")
    axes[1, 0].set_title("PnL vs Position Size")

    # 4) Avg PnL by Entry Hour
    trades_df["hour"] = pd.to_datetime(trades_df["entry_time"]).dt.hour
    hourly_stats = trades_df.groupby("hour")["pnl"].agg(["mean", "count"]).reset_index()
    axes[1, 1].bar(hourly_stats["hour"], hourly_stats["mean"])
    axes[1, 1].set_title("Avg PnL by Entry Hour")

    plt.tight_layout()
    plt.savefig(pnl_plot, dpi=200)
    plt.close()
    logger.info("Advanced plots completed: %s", pnl_plot)

    # ------------------------------------------------------------------
    # STEP 11: Decision Gate
    # ------------------------------------------------------------------
    logger.info("STEP 11 Running ELITE Decision Gate...")
    gate = DecisionGate(config_path)

    gate_metrics = dict(
        oos_median_sharpe=wf_results["oos_sharpe"],
        oos_profit_factor=wf_results["oos_profit_factor"],
        oos_max_drawdown=wf_results["oos_maxdd"],
        mc_prob_positive=mc_results["mc_positive_prob"],
        mc_p95_return=mc_results["mc_p95_return"],
        cvar5=tail_stats["cvar"],
        kelly_oos_full=kelly_oos_info["kelly_full"],
        mt5_correlation=0.92,  # Platzhalter für echte MT5-Reconciliation
    )
    if multi_asset_info is not None:
        gate_metrics["multi_asset_hit_rate"] = multi_asset_info["hit_rate"]
    if regime_decision is not None:
        gate_metrics["regime_allowed"] = regime_decision["allowed"]
        gate_metrics["regime_risk_multiplier"] = regime_decision["risk_multiplier"]

    result = gate.evaluate(gate_metrics)

    logger.info("=" * 60)
    logger.info("GATE RESULT: %s", result.status.value)
    logger.info("Confidence: %.1f%%", result.confidence * 100)
    logger.info("Reason: %s", result.reason)
    if result.violated_criteria:
        logger.warning("Violated criteria:")
        for criterion in result.violated_criteria:
            logger.warning(" - %s", criterion)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # STEP 12: Summary JSON + Text-Report
    # ------------------------------------------------------------------
    summary = dict(
        metrics=metrics,
        cost_results=cost_results,
        kelly=kelly_info,
        kelly_oos=kelly_oos_info,
        vix_alignment=vix_alignment,
        mc_results=mc_results,
        walkforward=wf_results,
        multi_asset=multi_asset_info,
        sim_results=sim_results,
        hmm_results={
            "regimestats": hmm_results["regimestats"],
        },
        gate_result=dict(
            status=result.status.value,
            confidence=result.confidence,
            reason=result.reason,
            violated_criteria=result.violated_criteria,
        ),
    )

    summary_path = reports_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, default=str, indent=2)
    logger.info("Summary JSON saved to %s", summary_path)

    # Text-Report
    logger.info("STEP 12 Saving report...")
    report_path = reports_dir / "report.txt"

    base_df = pd.DataFrame(
        [
            dict(
                TotalReturn=f"{metrics['total_return']:.2f}",
                Sharpe=f"{metrics['sharpe_ratio']:.2f}",
                MaxDD=f"{metrics['max_drawdown']:.2f}",
                WinRate=f"{metrics['win_rate']:.2f}",
                Trades=metrics["total_trades"],
            )
        ],
        index=["Base"],
    )

    cost_rows = []
    for name, m in cost_results.items():
        cost_rows.append(
            dict(
                Scenario=name,
                TotalReturn=f"{m['total_return']:.2f}",
                Sharpe=f"{m['sharpe_ratio']:.2f}",
                MaxDD=f"{m['max_drawdown']:.2f}",
                PF=f"{m['profit_factor']:.2f}",
            )
        )
    cost_df = pd.DataFrame(cost_rows)

    kelly_df = pd.DataFrame(
        [
            dict(
                winrate=f"{kelly_info['win_rate']:.2f}",
                payoff=f"{kelly_info['payoff_ratio']:.2f}",
                full=f"{kelly_info['kelly_full']:.2f}",
                half=f"{kelly_info['kelly_half']:.2f}",
                quarter=f"{kelly_info['kelly_quarter']:.2f}",
            )
        ],
        index=["Kelly"],
    )

    kelly_oos_df = pd.DataFrame(
        [
            dict(
                winrate=f"{kelly_oos_info['win_rate']:.2f}",
                payoff=f"{kelly_oos_info['payoff_ratio']:.2f}",
                full=f"{kelly_oos_info['kelly_full']:.2f}",
                half=f"{kelly_oos_info['kelly_half']:.2f}",
                quarter=f"{kelly_oos_info['kelly_quarter']:.2f}",
            )
        ],
        index=["Kelly_OOS"],
    )

    vix_rows = []
    for rname, m in vix_alignment["regime_stats"].items():
        vix_rows.append(
            dict(
                Regime=rname,
                Trades=m["n_trades"],
                TotalReturn=f"{m['total_return']:.2f}",
                Sharpe=f"{m['sharpe_ratio']:.2f}",
                MaxDD=f"{m['max_drawdown']:.2f}",
                PF=f"{m['profit_factor']:.2f}",
            )
        )
    vix_df = pd.DataFrame(vix_rows)

    wf_rows = []
    for w in wf_results.get("window_metrics", []):
        wf_rows.append(
            dict(
                Window=w["window_id"],
                Train=f"{w['train_start']} - {w['train_end']}",
                Test=f"{w['test_start']} - {w['test_end']}",
                Trades=w["test_n_trades"],
                Sharpe=f"{w['test_sharpe']:.2f}",
                PF=f"{w['test_profit_factor']:.2f}",
                MaxDD=f"{w['test_maxdd']:.2f}",
                Return=f"{w['test_total_return']:.2f}",
                Kelly_full=f"{w.get('test_kelly_full', 0.0):.2f}",
                Kelly_half=f"{w.get('test_kelly_half', 0.0):.2f}",
                Kelly_quarter=f"{w.get('test_kelly_quarter', 0.0):.2f}",
            )
        )
    wf_df = pd.DataFrame(wf_rows)

    # HMM Regime Tabelle
    hmm_rows = []
    for rname, m in hmm_results["regimestats"].items():
        hmm_rows.append(
            dict(
                Regime=rname,
                Trades=m["n_trades"],
                TotalReturn=f"{m['total_return']:.2f}",
                Sharpe=f"{m['sharpe_ratio']:.2f}",
                MaxDD=f"{m['max_drawdown']:.2f}",
                PF=f"{m['profit_factor']:.2f}",
            )
        )
    hmm_df = pd.DataFrame(hmm_rows)

    # Stochastische Modelle Tabelle
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
        f.write("=" * 60 + "\n")
        f.write(f"Strategy: {metrics_strategy_name}\n")
        f.write(
            f"Date Range: {metrics['date_range'][0]} to {metrics['date_range'][1]}\n\n"
        )

        f.write("METRICS\n")
        f.write(base_df.to_markdown() + "\n\n")

        f.write("COST SCENARIOS\n")
        f.write(cost_df.to_markdown(index=False) + "\n\n")

        f.write("KELLY SIZING Full Sample\n")
        f.write(kelly_df.to_markdown() + "\n\n")

        f.write("KELLY SIZING Walk-Forward OOS only\n")
        f.write(kelly_oos_df.to_markdown() + "\n\n")

        f.write("VIX REGIME PERFORMANCE\n")
        f.write(vix_df.to_markdown(index=False) + "\n\n")

        f.write("HMM REGIME PERFORMANCE\n")
        if not hmm_df.empty:
            f.write(hmm_df.to_markdown(index=False) + "\n\n")
        else:
            f.write("no HMM regimes with sufficient trades\n\n")

        f.write("WALK-FORWARD OOS WINDOWS\n")
        if not wf_df.empty:
            f.write(wf_df.to_markdown(index=False) + "\n\n")
        else:
            f.write("no walk-forward windows\n\n")

        if multi_asset_info is not None and multi_asset_info.get("details"):
            df_ma = pd.DataFrame(multi_asset_info["details"]).sort_values(
                "sharpe", ascending=False
            )
            f.write("MULTI-ASSET OPTIMIZER SYMBOL STATS\n")
            f.write(
                df_ma[["symbol", "sharpe", "profit", "profit_factor", "equity_dd_pct", "trades"]]
                .to_markdown(index=False)
                + "\n\n"
            )

        f.write("MONTE CARLO SUMMARY\n")
        f.write(f" mc_positive_prob: {mc_results['mc_positive_prob']:.2f}\n")
        f.write(f" mc_median_return: {mc_results['mc_median_return']:.2f}\n")
        f.write(f" mc_p5_return: {mc_results['mc_p5_return']:.2f}\n")
        f.write(f" mc_p95_return: {mc_results['mc_p95_return']:.2f}\n")
        f.write(f" mc_median_maxdd: {mc_results['mc_median_maxdd']:.2f}\n")
        f.write(f" mc_p95_maxdd: {mc_results['mc_p95_maxdd']:.2f}\n\n")

        f.write("STOCHASTIC MODEL SCENARIOS\n")
        if not sim_df.empty:
            f.write(sim_df.to_markdown(index=False) + "\n\n")
        else:
            f.write("no stochastic scenarios\n\n")

        f.write("DECISION GATE\n")
        f.write(f" Status: {result.status.value}\n")
        f.write(f" Confidence: {result.confidence:.2f}\n")
        f.write(f" Reason: {result.reason}\n")
        if result.violated_criteria:
            f.write(" Violated criteria:\n")
            for c in result.violated_criteria:
                f.write(f"  - {c}\n")

    logger.info("Report saved to %s", report_path)
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED")
    logger.info("=" * 60)

    # HTML-Report
    try:
        html_path = render_html_for_strategy_dir(strategy_reports_dir)
        logger.info("HTML report saved to %s", html_path)
    except Exception as e:
        logger.error("HTML report generation failed: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Validation Pipeline")
    parser.add_argument(
        "--trades-file",
        type=str,
        required=False,
        help=(
            "Path to MT5 trades CSV file; wenn leer: "
            "raw-Dateien organisieren, konvertieren und neueste trades_merged.csv nutzen."
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
        # direkter Pfad angegeben
        trades_path = args.trades_file
    else:
        raw_dir = Path("data/raw")
        processed_dir = Path("data/processed")
        logger.info("Organizing raw MT5 files into strategy folders...")
        organize_raw(raw_dir, raw_dir)
        logger.info("Converting MT5 raw files to merged trades CSVs...")
        batch_convert_raw(raw_dir, raw_dir, base_output_dir=processed_dir, overwrite=False)
        try:
            trades_path = find_latest_trades_csv(str(processed_dir))
        except FileNotFoundError as e:
            logger.error("No trades_merged.csv found after conversion: %s", e)
            raise

    run_pipeline(trades_path, config_path=args.config)
