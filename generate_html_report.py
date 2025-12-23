#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 QUANT VALIDATION DASHBOARD v5.0
Komplett kompatibel mit run_pipeline.py, inkl. aller Plots und Layout-Anpassungen.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import pandas as pd

# Mapping der Plot-Namen auf Dateien im Strategy-Reports-Ordner
PLOT_FILES: Dict[str, str] = {
    "equity": "equity.png",
    "pnl_distribution": "pnl_distribution.png",
    "mc_returns": "mc_returns.png",
    "mc_paths": "mc_paths.png",
    "drawdown_analysis": "drawdown_analysis.png",
    "walk_forward_sharpe": "walk_forward_sharpe.png",
    "kelly_frontier": "kelly_frontier.png",             
    "vix_regime_sharpe": "vix_regime_sharpe.png",  
    "vix_regime_timeseries": "vix_regime_timeseries.png",
    "hmm_regime_overlay": "hmm_regime_overlay.png",
    "multi_asset_sharpe": "multiassetsharpe.png",
    "stochastic_scenarios": "stochastic_scenarios.png",
}


# -------------------------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------------------------

def load_summary(summary_path: Path) -> Dict[str, Any]:
    """Lädt summary.json sicher."""
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Fehler beim Laden von {summary_path}: {e}")
        return {}


def safe_get(d: Dict[str, Any], keys: List[str] | tuple, default=None):
    """Verschachtelte Keys sicher aus einem Dict lesen."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def df_to_table_html(df: Optional[pd.DataFrame]) -> str:
    """Erstellt HTML-Tabelle aus DataFrame (Card Design)."""
    if df is None or df.empty:
        return (
            '<div class="card-body">'
            '<p style="text-align:left;color:var(--text-muted);padding:20px">'
            'Keine Daten verfügbar.'
            '</p></div>'
        )

    html: List[str] = []
    html.append('<div class="card-body"><div class="table-responsive">')
    html.append('<table class="table table-sm table-striped table-dark">')
    html.append("<thead><tr>")
    for col in df.columns:
        html.append(f'<th style="text-align:left">{col}</th>')
    html.append("</tr></thead><tbody>")

    for _, row in df.iterrows():
        html.append("<tr>")
        for val in row:
            if pd.isna(val):
                html.append("<td>-</td>")
            elif isinstance(val, (int,)) or (
                isinstance(val, float) and float(val).is_integer()
            ):
                html.append(f"<td>{int(val)}</td>")
            elif isinstance(val, float):
                html.append(f"<td>{val:.3f}</td>")
            else:
                html.append(f"<td>{val}</td>")
        html.append("</tr>")
    html.append("</tbody></table></div></div>")
    return "".join(html)


def get_status_badge(status: str, confidence: float) -> str:
    """Status-Badge mit passender Klasse."""
    status_map = {
        "live": "status-live",
        "ok": "status-live",
        "warning": "status-warning",
        "fail": "status-fail",
        "block": "status-fail",
    }
    css_class = status_map.get(status.lower(), "status-warning")
    return (
        f'<div class="badge {css_class}">'
        f'<span class="badge-dot"></span>'
        f"<span>{status.upper()}</span>"
        f"<span> · Confidence {confidence:.0%}</span>"
        "</div>"
    )


def plot_exists(output_dir: Path, plot_name: str) -> str:
    """Prüft, ob Plot existiert und gibt <img>-Tag oder Platzhalter zurück."""
    filename = PLOT_FILES.get(plot_name, f"{plot_name}.png")
    plot_path = output_dir / filename
    if plot_path.exists():
        return f'<img src="{plot_path.name}" class="plot" alt="{plot_name}">'
    return (
        '<div style="height:400px;display:flex;align-items:left;justify-content:left;'
        'background:#020617;border:1px solid #1f2937;border-radius:10px;'
        'color:var(--text-muted);font-size:13px">'
        f"{plot_name}-Plot nicht gefunden."
        "</div>"
    )


def render_vix_table(vix_alignment: Dict[str, Any]) -> str:
    """VIX Regime Tabelle."""
    vix_stats = vix_alignment.get("regime_stats", {})
    if not vix_stats:
        return df_to_table_html(pd.DataFrame())

    data = []
    for regime, stats in vix_stats.items():
        data.append(
            {
                "Regime": regime,
                "Trades": stats.get("n_trades", 0),
                "TotalReturn": stats.get("total_return", 0.0),
                "Sharpe": stats.get("sharpe_ratio", 0.0),
                "MaxDD": stats.get("max_drawdown", 0.0),
                "PF": stats.get("profit_factor", 0.0),
            }
        )
    return df_to_table_html(pd.DataFrame(data))


def render_hmm_table(hmm_results: Any) -> str:
    """HMM-Regime-Tabelle inkl. kurzer Interpretation."""
    try:
        if isinstance(hmm_results, dict) and "regime_stats" in hmm_results:
            stats_dict = hmm_results["regime_stats"]
        elif hasattr(hmm_results, "regime_stats"):
            stats_dict = hmm_results.regime_stats
        else:
            stats_dict = {}
    except Exception:
        stats_dict = {}

    if not stats_dict:
        return df_to_table_html(pd.DataFrame())

    rows = []
    for regime, stats in stats_dict.items():
        rows.append(
            {
                "Regime": regime,
                "Trades": stats.get("n_trades", 0),
                "TotalReturn": stats.get("total_return", 0.0),
                "Sharpe": stats.get("sharpe_ratio", 0.0),
                "MaxDD": stats.get("max_drawdown", 0.0),
                "PF": stats.get("profit_factor", 0.0),
            }
        )
    table_html = df_to_table_html(pd.DataFrame(rows))

    explanation = (
        '<div class="explanation">'
        "<h4>Wie lesen?</h4>"
        "<p>Jedes HMM-Regime fasst ähnliche Marktphasen zusammen, "
        "basierend auf der Equity-Kurve dieser Strategie – nicht auf einem Index. "
        "<strong>Regime</strong> ist nur ein Label (0,1,2…), "
        "<strong>Trades</strong> zeigt, wie viele Trades in diesem Zustand liegen.</p>"
        "<p><strong>TotalReturn</strong> und <strong>Sharpe</strong> zeigen, "
        "in welchen Zuständen die Strategie Geld verdient und wie effizient. "
        "<strong>MaxDD</strong> und <strong>PF</strong> zeigen, wie schmerzhaft "
        "Drawdowns in diesem Zustand sind. Typischer Use‑Case: "
        "Regime mit schlechter Sharpe und hoher MaxDD als Warnsignal/Filter nutzen.</p>"
        "</div>"
    )
    return table_html + explanation


# -------------------------------------------------------------------
# Haupt-HTML-Renderer
# -------------------------------------------------------------------

def generate_html_report(summary_path: Path, output_dir: Path, strategy_name: str) -> Path:
    """Generiert das komplette HTML-Dashboard für einen Strategie-Ordner."""
    summary = load_summary(summary_path)
    if not summary:
        print("Keine Summary-Daten gefunden!")
        return output_dir / f"{strategy_name}_dashboard.html"

    metrics = summary.get("metrics", {})
    gate_result = summary.get("gate_result", summary.get("gateresult", {}))
    vix_alignment = summary.get("vix_alignment", summary.get("vixalignment", {}))
    hmm_results = summary.get("hmm_results", {})
    mc_results = summary.get("mc_results", summary.get("mcresults", {}))
    wf_results = summary.get("walkforward", summary.get("walk_forward", {}))
    kelly_info = summary.get("kelly_oos", summary.get("kellyoos", {}))
    multi_asset = summary.get("multi_asset", summary.get("multiasset", {}))
    sim_results = summary.get("sim_results", summary.get("simresults", {}))

    # Basis-Metrics
    total_return = metrics.get("total_return", 0.0)
    sharpe = metrics.get("sharpe_ratio", 0.0)
    max_dd = metrics.get("max_drawdown", 0.0)
    winrate = metrics.get("win_rate", 0.0)
    total_trades = metrics.get("total_trades", 0)
    date_range = metrics.get("date_range", ["NA", "NA"])
    date_range_str = f"{date_range[0]} – {date_range[1]}" if date_range[0] != "NA" else "NA"

    # Erweiterte Kennzahlen
    oos_sharpe = wf_results.get("oos_sharpe", 0.0)
    mc_success = mc_results.get("mc_positive_prob", 0.0)
    mc_p95 = mc_results.get("mc_p95_return", 0.0)
    tail_risk = safe_get(summary, ["tail_stats", "cvar_5"], default=0.0) or safe_get(
        summary, ["tailstats", "cvar5"], default=0.0
    )
    cvar5 = safe_get(summary, ("tail_stats", "cvar5"), default=None)
    if cvar5 is None:
        cvar5 = mc_results.get("cvar5", 0.0)

    multi_asset_hitrate = multi_asset.get("hitrate", 0.0) * 100.0

    status = gate_result.get("status", "warning")
    confidence = gate_result.get("confidence", 0.5)
    reason = gate_result.get("reason", "Keine Begründung hinterlegt.")
    violations = gate_result.get("violated_criteria", gate_result.get("violatedcriteria", []))

    # HTML skeleton & CSS
    css_style = """
    :root {
        --bg: #0f172a;
        --bg-alt: #020617;
        --card-bg: #111827;
        --card-border: #1f2937;
        --text: #e5e7eb;
        --text-muted: #9ca3af;
        --accent: #3b82f6;
        --accent-soft: rgba(59,130,246,0.1);
        --danger: #f97316;
        --success: #22c55e;
        --warning: #eab308;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        background: radial-gradient(circle at top, #1e293b 0, #020617 55%);
        color: var(--text);
        padding: 24px;
    }
    a { color: var(--accent); text-decoration: none; }
    .container { max-width: 1400px; margin: 0 auto 64px auto; }
    .header {
        display: flex; justify-content: space-between; align-items: left;
        margin-bottom: 24px; gap: 16px;
    }
    .header-title { display: flex; flex-direction: column; gap: 4px; }
    .header-title h1 { font-size: 26px; letter-spacing: 0.04em; }
    .header-title p { font-size: 13px; color: var(--text-muted); }
    .badge {
        display: inline-flex; align-items: left; gap: 6px;
        padding: 6px 10px; border-radius: 999px; font-size: 12px;
        background: var(--accent-soft); color: var(--accent);
    }
    .badge-dot { width: 8px; height: 8px; border-radius: 999px; background: var(--accent); }
    .status-live { background: rgba(34,197,94,0.12); color: var(--success); }
    .status-live .badge-dot { background: var(--success); }
    .status-warning { background: rgba(234,179,8,0.12); color: var(--warning); }
    .status-warning .badge-dot { background: var(--warning); }
    .status-fail { background: rgba(248,113,113,0.12); color: var(--danger); }
    .status-fail .badge-dot { background: var(--danger); }

    .grid { display: grid; gap: 16px; }
    .grid-4 { grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); }
    .grid-2 { grid-template-columns: repeat(auto-fit, minmax(320px,1fr)); }

    .card, .card-head {
        background: radial-gradient(circle at top left,#1f2937 0,#020617 65%);
        border-radius: 16px;
        border: 1px solid var(--card-border);
        padding: 16px 18px 18px 18px;
        box-shadow: 0 18px 45px rgba(15,23,42,0.8);
        display: flex; flex-direction: column; height: 100%; overflow: hidden;
    }
    .card-head { min-height: 0; }
    .card { min-height: 260px; }

    .card table {
        width: 100%;
        border-collapse: collapse;
    }

    .card th,
    .card td {
        text-align: left;      /* ← DAS ist der entscheidende Punkt */
        vertical-align: middle;
    }

    .card-header {
        padding: 8px 0 10px 0;
        border-bottom: 1px solid #1f2937;
        font-size: 14px; font-weight: 600;
        margin-bottom: 10px;
        display: flex; justify-content: space-between; align-items: left;
        color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em;
    }
    .card-title { font-size: 15px; font-weight: 600; color: #e5e7eb; text-transform:none; letter-spacing:0; }
    .card-body { font-size: 13px; display: flex; flex-direction: column; flex: 1; }

    .metric-value { font-size: 24px; font-weight: 600; }
    .metric-label { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
    .metric-sub { font-size: 11px; margin-top: 6px; color: var(--text-muted); }

    .section-title {
        margin: 32px 0 12px 0;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--text-muted);
    }

    img.plot { width: 100%; border-radius: 10px; border: 1px solid #1f2937; display: block; }
    .plot-caption { font-size: 12px; color: var(--text-muted); margin-top: 8px; line-height: 1.4; }

    .explanation {
        margin-top: auto;
        padding: 10px 12px;
        border-radius: 10px;
        background: #020617;
        border: 1px solid #1f2937;
        font-size: 12px;
        color: var(--text-muted);
        line-height: 1.7;
    }
    .explanation h4 { font-size: 13px; margin-bottom: 8px; color: var(--text); }

    .table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .table th, .table td { padding: 4px 8px; border-bottom: 1px solid #111827; }
    .table thead th { background: #020617; font-weight: 500; color: var(--text-muted); }
    .table tbody tr:nth-child(even) { background: #020617; }
    .table-responsive { overflow-x: auto; }

    .footer {
        margin-top: 32px; font-size: 11px;
        color: var(--text-muted); text-align: right;
    }

    @media (max-width: 768px) {
        body { padding: 12px; }
        .header { flex-direction: column; align-items: flex-start; }
    }
    """

    html: List[str] = []
    html.append("<!DOCTYPE html>")
    html.append('<html lang="de"><head>')
    html.append('<meta charset="UTF-8">')
    html.append(f"<title>{strategy_name} - Quant Dashboard</title>")
    html.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    html.append(f"<style>{css_style}</style>")
    html.append("</head><body>")
    html.append('<div class="container">')

    # Header
    html.append('<div class="header">')
    html.append(
        '<div class="header-title">'
        '<h1>QUANT VALIDATION DASHBOARD</h1>'
        f"<p>Strategie <strong>{strategy_name}</strong> · Zeitraum {date_range_str} · Trades {total_trades}</p>"
        "</div>"
    )
    html.append("<div>")
    html.append(get_status_badge(status, confidence))
    html.append("</div></div>")  # header

    # Grid mit Top-Metriken
    html.append('<div class="grid grid-4">')

    html.append(
        '<div class="card-head">'
        '<div class="card-header"><div class="card-title">Gesamt-Rendite</div></div>'
        '<div class="card-body">'
        f'<div class="metric-value">{total_return*100:.2f}%</div>'
        '<div class="metric-label">Netto-Performance über gesamten Zeitraum</div>'
        '<div class="metric-sub">Zeigt, wie stark das Konto insgesamt gewachsen ist.</div>'
        "</div></div>"
    )

    html.append(
        '<div class="card-head">'
        '<div class="card-header"><div class="card-title">Sharpe-Ratio</div></div>'
        '<div class="card-body">'
        f'<div class="metric-value">{sharpe:.2f}</div>'
        '<div class="metric-label">Rendite pro Risikoeinheit</div>'
        '<div class="metric-sub">Höher ist besser – 1 gut, 2 sehr gut, 3 exzellent.</div>'
        "</div></div>"
    )

    html.append(
        '<div class="card-head">'
        '<div class="card-header"><div class="card-title">Max-Drawdown</div></div>'
        '<div class="card-body">'
        f'<div class="metric-value">{max_dd*100:.2f}%</div>'
        '<div class="metric-label">Größter Einbruch vom Konto-Hoch</div>'
        '<div class="metric-sub">Zeigt, wie tief das Konto zwischenzeitlich fallen kann.</div>'
        "</div></div>"
    )

    html.append(
        '<div class="card-head">'
        '<div class="card-header"><div class="card-title">Win-Rate</div></div>'
        '<div class="card-body">'
        f'<div class="metric-value">{winrate*100:.2f}%</div>'
        '<div class="metric-label">Anteil gewinnender Trades</div>'
        '<div class="metric-sub">Wichtig im Zusammenspiel mit Gewinn/Verlust-Größe (Payoff).</div>'
        "</div></div>"
    )

    html.append("</div>")  # grid-4

    # OOS / Monte Carlo / Tail Risk / Kelly
    html.append('<div class="grid grid-4" style="margin-top:16px">')

    html.append(
        '<div class="card-head">'
        '<div class="card-header"><div class="card-title">OOS Sharpe</div></div>'
        '<div class="card-body">'
        f'<div class="metric-value">{oos_sharpe:.2f}</div>'
        '<div class="metric-label">Sharpe in Testfenstern (Out-of-Sample)</div>'
        '<div class="metric-sub">Nutzt ausschließlich Daten außerhalb des Trainings.</div>'
        "</div></div>"
    )

    html.append(
        '<div class="card-head">'
        '<div class="card-header"><div class="card-title">Monte-Carlo</div></div>'
        '<div class="card-body">'
        f'<div class="metric-value">{mc_success*100:.1f}%</div>'
        '<div class="metric-label">Anteil profitabler Szenarien</div>'
        '<div class="metric-sub">Wie oft die Simulation mit Gewinn endet.</div>'
        "</div></div>"
    )

    html.append(
        '<div class="card-head">'
        '<div class="card-header"><div class="card-title">Monte-Carlo P95</div></div>'
        '<div class="card-body">'
        f'<div class="metric-value">{mc_p95*100:.1f}%</div>'
        '<div class="metric-label">95%-Quantil der Gesamtrendite</div>'
        '<div class="metric-sub">In 95% der Fälle besser als dieser Wert.</div>'
        "</div></div>"
    )

    html.append(
        '<div class="card-head">'
        '<div class="card-header"><div class="card-title">Tail-Risk CVaR</div></div>'
        '<div class="card-body">'
        f'<div class="metric-value">{cvar5*100:.1f}%</div>'
        '<div class="metric-label">Verlust der schlechtesten 5%</div>'
        '<div class="metric-sub">Realistischer Extremverlust statt theoretischem MaxDD.</div>'
        "</div></div>"
    )

    html.append("</div>")  # grid-4

    # ------------------------------------------------------------------
    # Strategie-Überblick
    # ------------------------------------------------------------------
    html.append('<h2 class="section-title">Strategie-Überblick</h2>')
    html.append('<div class="grid grid-2">')

    # Equity
    html.append('<div class="card">')
    html.append(
        '<div class="card-header">'
        '<div class="card-title">Equity Curve</div>'
        '<span>Kontoverlauf</span>'
        "</div>"
    )
    html.append('<div class="card-body">')
    html.append(plot_exists(output_dir, "equity"))
    html.append(
        '<p class="plot-caption">'
        "Die Equity-Kurve zeigt, wie sich das Konto über die Zeit entwickelt. "
        "Ein stetig steigender Verlauf mit überschaubaren Rücksetzern deutet auf eine stabile Strategie hin."
        "</p>"
    )
    html.append(
        '<div class="explanation"><h4>Einfach erklärt</h4>'
        "<p>Die Linie repräsentiert deinen Kontostand nach jedem Trade. "
        "Größere Dellen sind Drawdowns, also Phasen, in denen das Konto vom letzten Hoch zurücksetzt. "
        "Viele kleine Dellen sind normal; tiefe und lange Dellen sind kritisch.</p>"
        "</div></div></div>"
    )

    # PnL Distribution
    html.append('<div class="card">')
    html.append(
        '<div class="card-header">'
        '<div class="card-title">PnL-Verteilung & Verhalten</div>'
        '<span>Wo gewinnt/verliert die Strategie?</span>'
        "</div>"
    )
    html.append('<div class="card-body">')
    html.append(plot_exists(output_dir, "pnl_distribution"))
    html.append(
        '<p class="plot-caption">'
        "Oben links: Verteilung der Trade-Gewinne/-Verluste, oben rechts: QQ-Plot vs. Normalverteilung, "
        "unten links: PnL vs. Positionsgröße, unten rechts: durchschnittlicher PnL nach Einstiegs-Stunde."
        "</p>"
    )
    html.append(
        '<div class="explanation"><h4>Einfach erklärt</h4>'
        "<p>Hier erkennst du, ob wenige große Gewinne viele kleine Verluste ausgleichen, "
        "ob die Verteilung schwere Ausreißer (Fat Tails) hat und zu welchen Uhrzeiten die Strategie "
        "tendenziell Geld verdient oder verliert. Stunden mit dauerhaft negativem Durchschnitts-PnL "
        "sind typische Schwachstellen.</p>"
        "</div></div></div>"
    )

    html.append("</div>")  # grid-2

    # ------------------------------------------------------------------
    # Risiko / Tail-Risk
    # ------------------------------------------------------------------
    html.append('<h2 class="section-title">Risiko & Tail-Risk</h2>')
    html.append('<div class="grid grid-2">')

    # MC Returns
    html.append('<div class="card">')
    html.append(
        '<div class="card-header">'
        '<div class="card-title">Monte Carlo Total Returns</div>'
        '<span>Verteilung der Gesamtrenditen</span>'
        "</div>"
    )
    html.append('<div class="card-body">')
    html.append(plot_exists(output_dir, "mc_returns"))
    html.append(
        '<p class="plot-caption">'
        "Histogramm der simulierten Gesamtrenditen über alle Monte-Carlo-Szenarien. "
        "Die vertikale Linie zeigt die Median-Rendite."
        "</p>"
    )
    html.append(
        '<div class="explanation"><h4>Einfach erklärt</h4>'
        "<p>Monte-Carlo-Simulation bedeutet: Wir mischen die bestehenden Trades zufällig neu "
        "und simulieren viele alternative Zukunftsverläufe. Je mehr der Balken rechts von 0 liegen, "
        "desto robuster ist die Strategie.</p>"
        "</div></div></div>"
    )

    # MC Paths
    html.append('<div class="card">')
    html.append(
        '<div class="card-header">'
        '<div class="card-title">Monte Carlo Equity Paths</div>'
        '<span>Stresstest des Kontoverlaufs</span>'
        "</div>"
    )
    html.append('<div class="card-body">')
    html.append(plot_exists(output_dir, "mc_paths"))
    html.append(
        '<p class="plot-caption">'
        "Mehrere simulierte Equity-Pfade, inklusive Median-Kurve und 90%-Konfidenzband."
        "</p>"
    )
    html.append(
        '<div class="explanation"><h4>Einfach erklärt</h4>'
        "<p>Jede dünne Linie ist ein möglicher Kontoverlauf in der Zukunft. "
        "Die farbige Bandbreite zeigt, wie weit gute und schlechte Verläufe typischerweise auseinanderliegen. "
        "Eine enge Bandbreite bedeutet stabilere Ergebnisse.</p>"
        "</div></div></div>"
    )

    html.append("</div>")  # grid-2

    # Drawdown-Analyse + Kelly
    html.append('<div class="grid grid-2" style="margin-top:16px">')

    html.append('<div class="card">')
    html.append(
        '<div class="card-header">'
        '<div class="card-title">Drawdown-Analyse</div>'
        '<span>MaxDD & Dauer</span>'
        "</div>"
    )
    html.append('<div class="card-body">')
    html.append(plot_exists(output_dir, "drawdown_analysis"))
    html.append(
        '<p class="plot-caption">'
        "Links: Verteilung der maximalen Drawdowns aus den Monte-Carlo-Simulationen. "
        "Rechts: Histogramm der Dauer realer Drawdowns in Tagen."
        "</p>"
    )
    html.append(
        '<div class="explanation"><h4>Einfach erklärt</h4>'
        "<p>Max Drawdown ist der schlimmste Einbruch vom Konto-Hoch. "
        "Die Dauer zeigt, wie lange eine Durststrecke typischerweise anhält. "
        "Viele lange Drawdowns bedeuten, dass du emotional viel aushalten musst.</p>"
        "</div></div></div>"
    )

    html.append('<div class="card">')
    html.append(
        '<div class="card-header">'
        '<div class="card-title">Kelly Frontier</div>'
        '<span>Optimale Positionsgröße</span>'
        "</div>"
    )
    html.append('<div class="card-body">')
    html.append(plot_exists(output_dir, "kelly_frontier"))
    html.append(
        '<p class="plot-caption">'
        "Erwartete Wachstumsrate des Kontos in Abhängigkeit von der gewählten Risiko-Fraktion pro Trade (Kelly-Kriterium)."
        "</p>"
    )
    html.append(
        '<div class="explanation"><h4>Einfach erklärt</h4>'
        "<p>Das Kelly-Kriterium berechnet die theoretisch optimale Positionsgröße für maximales Wachstum. "
        "Volle Kelly-Größe ist oft zu aggressiv für die Praxis. Viele Trader nutzen eher die Hälfte "
        "oder ein Viertel davon, um Drawdowns zu begrenzen.</p>"
        "</div></div></div>"
    )

    html.append("</div>")  # grid-2


    # Stochastische Szenarien - GBM / Heston / Jump-Diffusion
    html.append('<h2 class="section-title">Stochastische Szenarien</h2>')
    html.append('<div class="grid grid-2">')

    # Card 1: Modell-Kennzahlen (Tabelle)
    html.append('<div class="card">')
    html.append(
        '<div class="card-header">'
        '<div class="card-title">Modell-Kennzahlen</div>'
        '<span>GBM / Heston / Jump-Diffusion</span>'
        '</div>'
    )

    if sim_results:
        rows = []
        for name, m in sim_results.items():
            rows.append(
                {
                    "Modell": name,
                    "MedianReturn": m.get("median_return", 0.0),
                    "P5": m.get("p5_return", 0.0),
                    "P95": m.get("p95_return", 0.0),
                    "MedianMaxDD": m.get("median_maxdd", 0.0),
                    "P95MaxDD": m.get("p95_maxdd", 0.0),
                }
            )
        html.append(df_to_table_html(pd.DataFrame(rows)))
    else:
        html.append(
            '<div class="card-body"><p style="text-align:left;'
            'color:var(--text-muted);padding:20px">'
            'Keine stochastischen Szenarien vorhanden.'
            '</p></div>'
        )

    html.append(
        '<div class="explanation"><h4>So liest du die Tabelle</h4>'
        '<p><strong>MedianReturn</strong>, <strong>P5</strong> und '
        '<strong>P95</strong> zeigen, wie sich die Gesamtrendite deiner '
        'Strategie verhält, wenn man sie in das jeweilige Modell einbettet. '
        'Liegt z.&nbsp;B. der Heston-Median deutlich tiefer als der GBM-Median, '
        'reagiert die Strategie empfindlich auf Volatilitätsspitzen.</p>'
        '<p><strong>MedianMaxDD</strong> und <strong>P95MaxDD</strong> zeigen, '
        'wie tief und wie häufig Drawdowns in den Modellen werden. '
        'Sind die Heston-Drawdowns deutlich höher als bei GBM/Jump-Diffusion, '
        'musst du in turbulenten Märkten mit deutlich größeren Einbrüchen rechnen.</p>'
        '</div>'
    )
    html.append('</div>')  # Ende Card 1

    # Card 2: Plot-Übersicht der stochastischen Szenarien
    html.append('<div class="card">')
    html.append(
        '<div class="card-header">'
        '<div class="card-title">Stochastische Modelle – Übersicht</div>'
        '<span>GBM, Heston, Jump-Diffusion im Vergleich</span>'
        '</div>'
    )
    html.append('<div class="card-body">')
    html.append(plot_exists(output_dir, "stochastic_scenarios"))
    html.append(
        '<p class="plot-caption">'
        'Der Plot fasst Median, „Best Case“ (P95) und „Worst Case“ (P5) '
        'für Rendite und MaxDD der drei Modellwelten zusammen. '
        'So siehst du auf einen Blick, ob ein Modell (z.&nbsp;B. Heston) '
        'deutlich pessimistischer ist als die anderen.</p>'
        '<div class="explanation">'
        '<h4>So interpretierst du den Plot</h4>'
        '<p>Wenn die Balken für Heston klar schlechter sind als für GBM oder '
        'Jump-Diffusion, bedeutet das: Deine Strategie funktioniert vor allem '
        'in „ruhigeren“ Märkten und leidet bei starken Volatilitätssprüngen. '
        'Sind alle drei Modelle ähnlich, ist die Strategie gegenüber '
        'verschiedenen Marktannahmen robuster.</p>'
        '</div>'
    )
    html.append('</div>')  # Ende Card-Body 2
    html.append('</div>')  # Ende Card 2

    html.append('</div>')  # Ende grid-2


    # ------------------------------------------------------------------
    # Walk-Forward
    # ------------------------------------------------------------------
    html.append('<h2 class="section-title">Walk-Forward & OOS</h2>')
    html.append('<div class="grid grid-2">')

    html.append('<div class="card">')
    html.append(
        '<div class="card-header">'
        '<div class="card-title">Walk-Forward OOS Sharpe</div>'
        '<span>Robustheit über Zeit</span>'
        "</div>"
    )
    html.append('<div class="card-body">')
    html.append(plot_exists(output_dir, "walk_forward_sharpe"))
    html.append(
        '<p class="plot-caption">'
        "Sharpe Ratio in einzelnen Walk-Forward-Testfenstern. Die rote Linie zeigt den Durchschnitt "
        "der OOS-Sharpe-Werte."
        "</p>"
    )
    html.append(
        '<div class="explanation"><h4>Einfach erklärt</h4>'
        "<p>Beim Walk-Forward-Test wird die Strategie in einem Zeitraum trainiert und in einem späteren "
        "Zeitraum getestet (Out-of-Sample). So sieht man, ob die Strategie über verschiedene Marktphasen "
        "hinweg stabil bleibt oder nur in bestimmten Zeiträumen funktioniert.</p>"
        "</div></div></div>"
    )

    # Platz für zusätzliche OOS-Tabellen, falls gewünscht
    html.append('<div class="card">')
    html.append(
        '<div class="card-header">'
        '<div class="card-title">Walk-Forward Fenster</div>'
        '<span>OOS-Kennzahlen</span>'
        "</div>"
    )
    wf_windows = wf_results.get("window_metrics", [])
    if wf_windows:
        rows = []
        for w in wf_windows:
            rows.append(
                {
                    "Window": w.get("window_id"),
                    "Train": f"{w.get('train_start')} – {w.get('train_end')}",
                    "Test": f"{w.get('test_start')} – {w.get('test_end')}",
                    "Trades": w.get("test_n_trades", 0),
                    "Sharpe": w.get("test_sharpe", 0.0),
                    "PF": w.get("test_profit_factor", 0.0),
                    "MaxDD": w.get("test_max_dd", 0.0),
                    "Return": w.get("test_total_return", 0.0),
                }
            )
        html.append(df_to_table_html(pd.DataFrame(rows)))
    else:
        html.append(
            '<div class="card-body"><p style="text-align:left;'
            'color:var(--text-muted);padding:20px">'
            "Keine Walk-Forward-Fenster verfügbar."
            "</p></div>"
        )
    html.append("</div>")  # card
    html.append("</div>")  # grid-2

    # ------------------------------------------------------------------
    # Regime-Analyse
    # ------------------------------------------------------------------
    html.append('<h2 class="section-title">Regime-Analyse</h2>')
    html.append('<div class="grid grid-2">')

    # VIX Regime Performance
    html.append('<div class="card">')
    html.append(
        '<div class="card-header">'
        '<div class="card-title">VIX Regime Performance</div>'
        '<span>Volatilitäts-Phasen</span>'
        '</div>'
    )
    html.append('<div class="card-body">')
    html.append(plot_exists(output_dir, "vix_regime_sharpe"))
    html.append(
        '<p class="plot-caption">'
        'Balkendiagramm der Sharpe Ratio je VIX-Regime (z. B. Low Volatility, Range, High Volatility).'
        '</p>'
        '<div class="explanation"><h4>Einfach erklärt</h4>'
        '<p>Der VIX ist ein Volatilitätsindex („Angstbarometer“). Die Regime-Einteilung zeigt, '
        'in welchen Marktphasen (ruhig, normal, hektisch) die Strategie besonders gut oder '
        'schlecht läuft. Ein starkes Ungleichgewicht kann für eine Regime-Filterung genutzt werden.</p>'
        '</div>'
    )
    html.append('</div>')  # card-body
    html.append('</div>')  # card

    # VIX Regime Tabelle
    html.append('<div class="card">')
    html.append(
        '<div class="card-header">'
        '<div class="card-title">VIX Regime Tabelle</div>'
        '<span>Performance je Phase</span>'
        '</div>'
    )
    html.append(render_vix_table(vix_alignment))
    html.append(
        '<div class="explanation">'
        '<h4>Wie nutzen?</h4>'
        '<p>Die VIX-Regime basieren auf einem Marktindex (Volatilität des S&P 500), '
        'während die HMM-Regime direkt aus deiner Strategie-Eigenperformance kommen. '
        'Idee: Wenn ein HMM-Problem-Regime zeitlich hauptsächlich in '
        '„High_Volatility“ oder „Range“ fällt, kannst du diese Phasen mit einem '
        'einfachen VIX-Filter vorab ausblenden.</p>'
        '<p>Praktisch bedeutet das: '
        'Suche in der VIX-Tabelle nach Regimen mit vielen Trades, aber schwacher Sharpe '
        'oder hoher MaxDD. Diese Phasen sind Kandidaten, in denen du die Strategie '
        'abschalten oder den Einsatz reduzieren solltest.</p>'
        '</div>'
    )
    html.append('</div>')  # card

    html.append('</div>')  # grid-2

    # VIX-Zeitreihe & HMM Overlay
    html.append('<div class="grid grid-2" style="margin-top:16px">')

    # VIX Regime Verlauf
    html.append('<div class="card">')
    html.append(
        '<div class="card-header">'
        '<div class="card-title">VIX Regime Verlauf</div>'
        '<span>Regime-Zeitreihe</span>'
        '</div>'
    )
    html.append('<div class="card-body">')
    html.append(plot_exists(output_dir, "vix_regime_timeseries"))
    html.append(
        '<p class="plot-caption">'
        'Zeitlicher Verlauf der erkannten VIX-Regime (Stufenplot). '
        'Horizontale Abschnitte markieren stabile Volatilitätsphasen.'
        '</p>'
        '<div class="explanation">'
        '<h4>So liest du den Verlauf</h4>'
        '<p>Jeder waagerechte Abschnitt steht für ein VIX-Regime '
        '(z.B. Low_Volatility, Range, High_Volatility). '
        'Lege diesen Plot gedanklich über deine Equity-Kurve: '
        'Laufen Drawdowns überwiegend in High_Volatility-Abschnitten, '
        'ist die Strategie dort anfällig. Verdient sie Geld vor allem in '
        'Low_Volatility, ist sie ein „Ruhige-Phasen“-System.</p>'
        '</div>'
    )
    html.append('</div>')  # card-body
    html.append('</div>')  # card

    # HMM Regime Overlay
    html.append('<div class="card">')
    html.append(
        '<div class="card-header">'
        '<div class="card-title">HMM Regime Overlay</div>'
        '<span>Equity nach Marktzustand</span>'
        '</div>'
    )
    html.append('<div class="card-body">')
    html.append(plot_exists(output_dir, "hmm_regime_overlay"))
    html.append(
        '<p class="plot-caption">'
        'Equity-Kurve, eingefärbt nach HMM-Regime. Cluster mit gleicher Farbe haben ähnliche Performance-Eigenschaften.'
        '</p>'
        '<div class="explanation">'
        '<h4>So interpretierst du die Farben</h4>'
        '<p>Die Linie zeigt deine Equity-Kurve; die Punkte in verschiedenen Farben '
        'markieren HMM-Regime. Jede Farbe steht für ein Muster im Equity-Verlauf, '
        'z.B. Trendphasen, Seitwärtsphasen mit kleinen Gewinnen oder längere '
        'Verlustcluster.</p>'
        '<p>Interessant sind Abschnitte, in denen ein Farbbereich klar schlechter '
        'aussieht (lange Dellen, viele rote Trades). '
        'Diesen Regime-State kannst du in der HMM-Tabelle wiederfinden und dort '
        'an Sharpe/MaxDD erkennen. So lassen sich problematische Phasen systematisch '
        'identifizieren und ggf. als Filter ausschalten.</p>'
        '</div>'
    )
    html.append('</div>')  # card-body
    html.append('</div>')  # card

    html.append('</div>')  # grid-2

    # HMM Tabelle
    html.append('<div class="grid grid-2" style="margin-top:16px">')
    html.append('<div class="card">')
    html.append(
        '<div class="card-header"><div class="card-title">'
        'HMM Regime Tabelle</div><span>Verborgene Marktzustände</span></div>'
    )
    html.append(render_hmm_table(hmm_results))
    html.append('</div>')
    html.append('</div>')  # grid-2


    # ------------------------------------------------------------------
    # Decision Gate & Multi-Asset
    # ------------------------------------------------------------------
    html.append('<h2 class="section-title">Multi-Asset & Decision Gate</h2>')
    html.append('<div class="grid grid-2" style="gap:24px">')

    # Sharpe-Rangliste
    html.append('<div class="card">')
    html.append(
        '<div class="card-header">'
        '<div class="card-title">Sharpe-Rangliste</div>'
        '<span>Top-Assets nach Sharpe</span>'
        "</div>"
    )
    ma_details = multi_asset.get("details", [])
    if ma_details:
        df_ma = pd.DataFrame(ma_details)
        expected_cols = ["symbol", "sharpe", "profit", "profit_factor", "equity_dd_pct", "trades"]
        available_cols = [c for c in expected_cols if c in df_ma.columns]
        df_ma = df_ma[available_cols]
        df_ma = df_ma.sort_values("sharpe", ascending=False)
        html.append(df_to_table_html(df_ma.head(11)))
        html.append(
            '<div class="explanation"><h4>Hinweis</h4>'
            "<p>Nur Symbole mit gültigen Optimizer-Ergebnissen werden angezeigt. "
            "Die rote horizontale Linie im separaten Multi-Asset-Plot (falls vorhanden) markiert die Sharpe-Schwelle.</p>"
            "</div>"
        )
    else:
        html.append(
            '<div class="card-body"><p style="text-align:left;color:var(--text-muted);padding:20px">'
            "Keine Multi-Asset Optimizer-Daten gefunden oder XML konnte nicht geparst werden."
            "</p></div>"
        )
    html.append("</div>")  # card

    # Decision Gate mit Schwellen
    html.append('<div class="card">')
    html.append(
        '<div class="card-header">'
        '<div class="card-title">Decision Gate</div>'
        '<span>Live-Readiness</span>'
        "</div>"
    )
    html.append('<div class="card-body">')
    html.append(f"<p style='font-size:13px;margin-bottom:6px'><strong>Begründung:</strong> {reason}</p>")

    if violations:
        html.append("<ul style='font-size:12px;color:var(--text-muted);margin-left:18px;margin-bottom:12px'>")
        for v in violations:
            html.append(f"<li>{v}</li>")
        html.append("</ul>")

    # Schwellenwerte aus Config, falls in Summary abgelegt (optional)
    validation_cfg = summary.get("validation_config", {})
    min_oos_sharpe = validation_cfg.get("min_oos_sharpe", 1.8)
    max_oos_dd = validation_cfg.get("max_oos_drawdown", 0.12)
    min_mc_pos = validation_cfg.get("min_mc_positive_prob", 0.95)
    min_mc_p95 = validation_cfg.get("min_mc_p95_return", 0.20)
    max_cvar5 = validation_cfg.get("max_cvar5", -0.15)
    min_kelly = validation_cfg.get("min_kelly_oos_full", 0.15)
    min_ma_hit = validation_cfg.get("min_multi_asset_hitrate", 0.75)
    min_corr = validation_cfg.get("min_mt5_correlation", 0.90)
    mt5_corr = gate_result.get("mt5_correlation", 0.0)

    kelly_oos_full = kelly_info.get("kelly_full", 0.0)

    gate_rows = [
        ("OOS Sharpe", oos_sharpe, f">= {min_oos_sharpe:.2f}"),
        ("OOS Max Drawdown", metrics.get("oos_max_drawdown", 0.0), f"<= {max_oos_dd:.2f}"),
        ("MC positive Szenarien", mc_success, f">= {min_mc_pos:.2f}"),
        ("MC P95 Return", mc_p95, f">= {min_mc_p95:.2f}"),
        ("CVaR 5%", tail_risk, f">= {max_cvar5:.2f}"),
        ("Kelly OOS (full)", kelly_oos_full, f">= {min_kelly:.2f}"),
        ("Multi-Asset Hit-Rate", multi_asset_hitrate / 100.0, f">= {min_ma_hit:.2f}"),
        ("MT5 Korrelation", mt5_corr, f">= {min_corr:.2f}"),
    ]

    html.append('<div class="table-responsive" style="margin-top:8px"><table class="table">')
    html.append("<thead><tr><th>Kriterium</th><th>Ist</th><th>Schwelle</th></tr></thead><tbody>")

    for name, value, thresh in gate_rows:
        try:
            thr_val = float(thresh.split()[-1])
        except Exception:
            thr_val = 0.0
        good = False
        if ">=" in thresh:
            good = value >= thr_val
        elif "<=" in thresh:
            good = value <= thr_val
        color = "#22c55e" if good else "#f97316"
        html.append(
            f"<tr><td>{name}</td>"
            f"<td style='color:{color}'>{value:.2f}</td>"
            f"<td>{thresh}</td></tr>"
        )

    html.append("</tbody></table></div>")

    html.append(
        '<div class="explanation"><h4>Einfach erklärt</h4>'
        "<p>Das Decision Gate fasst alle wichtigen Kennzahlen zusammen und entscheidet, "
        "ob die Strategie für Live-Trading geeignet ist. Grün markierte Kriterien erfüllen "
        "die Mindestanforderungen, orange markierte sind kritisch und sollten vor einem Live-Einsatz "
        "analysiert oder verbessert werden.</p>"
        "</div>"
    )

    html.append("</div>")  # card-body
    html.append("</div>")  # card
    html.append("</div>")  # grid-2

    # ------------------------------------------------------------------
    # Live-Tauglichkeit: Risk-Radar (VOLLE ERWEITERUNG)
    # ------------------------------------------------------------------
    html.append('<h2 class="section-title">🔥 LIVE-TAUGLICHKEIT – Risk-Radar</h2>')
    html.append('<div class="grid grid-2">')

    # Card 1: Kernmetriken (erweitert mit OOS + Schwellen)
    html.append('<div class="card">')
    html.append(
        '<div class="card-header"><div class="card-title">'
        'Kernmetriken vs. ELITE-Schwellen</div>'
        '<span>Full Sample | OOS | Monte Carlo</span></div>'
    )

    rows = []

    # 1. Sharpe Ratios (Full vs OOS vs MC)
    metrics = summary.get("metrics", {})
    wf_results = summary.get("walk_forward", {})
    mc_results = summary.get("mc_results", {})
    
    rows.append({
        "Metrik": "Sharpe (Full Sample)",
        "Ist": f"{metrics.get('sharpe_ratio', 0.0):.2f}",
        "Schwelle": "≥ 1.8",
        "Status": "✅ ELITE" if metrics.get("sharpe_ratio", 0.0) >= 1.8 else "⚠️ OK" if metrics.get("sharpe_ratio", 0.0) >= 1.2 else "❌ FAIL"
    })
    rows.append({
        "Metrik": "Sharpe (OOS)",
        "Ist": f"{wf_results.get('oos_sharpe', 0.0):.2f}",
        "Schwelle": "≥ 1.0",
        "Status": "✅ OK" if wf_results.get("oos_sharpe", 0.0) >= 1.0 else "❌ FAIL"
    })

    # 2. Drawdowns
    rows.append({
        "Metrik": "MaxDD (Full Sample)",
        "Ist": f"{metrics.get('max_drawdown', 0.0)*100:.1f}%",
        "Schwelle": "≤ 15%",
        "Status": "✅ ELITE" if metrics.get('max_drawdown', 0.0) <= 0.15 else "⚠️ OK" if metrics.get('max_drawdown', 0.0) <= 0.25 else "❌ FAIL"
    })
    rows.append({
        "Metrik": "MaxDD (OOS)",
        "Ist": f"{wf_results.get('oos_max_dd', 0.0)*100:.1f}%",
        "Schwelle": "≤ 20%",
        "Status": "✅ OK" if wf_results.get('oos_max_dd', 0.0) <= 0.20 else "❌ FAIL"
    })

    # 3. Win Rate & Kelly
    kelly_info = summary.get("kelly_oos", {})
    rows.append({
        "Metrik": "Win Rate (Full)",
        "Ist": f"{metrics.get('win_rate', 0.0)*100:.1f}%",
        "Schwelle": "≥ 45%",
        "Status": "✅ OK" if metrics.get('win_rate', 0.0) >= 0.45 else "⚠️ WARN"
    })
    rows.append({
        "Metrik": "Kelly OOS (Full)",
        "Ist": f"{kelly_info.get('kelly_full', 0.0)*100:.1f}%",
        "Schwelle": "≤ 2.0%",
        "Status": "✅ OK" if kelly_info.get('kelly_full', 0.0) <= 0.02 else "⚠️ HOCH"
    })

    # 4. Monte Carlo Robustness
    rows.append({
        "Metrik": "MC Positive Prob",
        "Ist": f"{mc_results.get('mc_positive_prob', 0.0)*100:.1f}%",
        "Schwelle": "≥ 90%",
        "Status": "✅ ELITE" if mc_results.get('mc_positive_prob', 0.0) >= 0.90 else "⚠️ OK" if mc_results.get('mc_positive_prob', 0.0) >= 0.75 else "❌ FAIL"
    })
    rows.append({
        "Metrik": "MC P95 Return",
        "Ist": f"{mc_results.get('mc_p95_return', 0.0)*100:.1f}%",
        "Schwelle": "≥ 15%",
        "Status": "✅ OK" if mc_results.get('mc_p95_return', 0.0) >= 0.15 else "⚠️ WARN"
    })

    radar_df = pd.DataFrame(rows)
    html.append(df_to_table_html(radar_df))

    html.append('</div>')  # card

    # Card 2: Stochastische Modelle (Heston vs GBM vs Jump)
    html.append('<div class="card">')
    html.append(
        '<div class="card-header"><div class="card-title">'
        'Stochastische Robustheit</div><span>Heston | GBM | Jump-Diffusion</span></div>'
    )

    sim_results = summary.get("sim_results", {})
    stoch_rows = []
    if sim_results:
        for model, data in sim_results.items():
            status = "✅ OK"
            if model == "heston" and data.get("median_return", 0.0) < 0:
                status = "⚠️ VOL-RISIKO"
            stoch_rows.append({
                "Modell": model.upper(),
                "Median Return": f"{data.get('median_return', 0.0)*100:.1f}%",
                "P95 MaxDD": f"{data.get('p95_maxdd', 0.0)*100:.1f}%",
                "Status": status
            })
    html.append(df_to_table_html(pd.DataFrame(stoch_rows)))
    html.append('</div>')  # card

    html.append('</div>')  # grid-2

    # Grid 2: Regime-Analyse (VIX + HMM)
    html.append('<div class="grid grid-2" style="margin-top:16px">')

    # VIX Worst Regime
    vix_alignment = summary.get("vix_alignment", {})
    vix_profile = summary.get("risk_profiles", {}).get("vix")
    html.append('<div class="card">')
    html.append('<div class="card-header"><div class="card-title">VIX Regime</div><span>Schwächstes Regime</span></div>')
    
    if vix_profile:
        html.append(f'''
        <div class="card-body" style="padding:20px;text-align:center">
            <div style="font-size:24px;font-weight:bold;color:#ff6b35">{vix_profile.get('worst_regime', 'N/A')}</div>
            <div style="font-size:16px;color:var(--text-muted);">Weighted Sharpe: {vix_profile.get('weighted_sharpe', 0.0):.2f}</div>
            <div style="margin-top:10px;padding:10px;background:#333;border-radius:6px;font-size:14px;">
                {vix_profile.get("total_trades", 0)} Trades | 
                {"✅ OK" if vix_profile.get('weighted_sharpe', 0.0) >= 1.5 else "⚠️ Regime-Filter nötig"}
            </div>
        </div>
        ''')
    else:
        html.append('<div class="card-body"><p style="text-align:center;color:var(--text-muted);">Keine VIX-Daten</p></div>')
    html.append('</div>')

    # HMM Worst Regime
    hmm_results = summary.get("hmm_results", {})
    hmm_profile = summary.get("risk_profiles", {}).get("hmm")
    html.append('<div class="card">')
    html.append('<div class="card-header"><div class="card-title">HMM Regime</div><span>Schwächstes Regime</span></div>')
    
    if hmm_profile:
        html.append(f'''
        <div class="card-body" style="padding:20px;text-align:center">
            <div style="font-size:24px;font-weight:bold;color:#ff6b35">Regime {hmm_profile.get('worst_regime', 'N/A')}</div>
            <div style="font-size:16px;color:var(--text-muted);">Weighted Sharpe: {hmm_profile.get('weighted_sharpe', 0.0):.2f}</div>
            <div style="margin-top:10px;padding:10px;background:#333;border-radius:6px;font-size:14px;">
                {hmm_profile.get("total_trades", 0)} Trades | 
                {"✅ OK" if hmm_profile.get('weighted_sharpe', 0.0) >= 1.2 else "⚠️ HMM-Filter empfohlen"}
            </div>
        </div>
        ''')
    else:
        html.append('<div class="card-body"><p style="text-align:center;color:var(--text-muted);">Keine HMM-Daten</p></div>')
    html.append('</div>')

    html.append('</div>')  # grid-2 regimes

    # FINALER STATUS: Gesamt-Score
    html.append('<div class="full-width-card" style="margin-top:24px">')
    html.append('<div class="card-header"><div class="card-title">FINALER LIVE-STATUS</div></div>')
    
    gate_result = summary.get("gate_result", {})
    status = gate_result.get("status", "UNKNOWN")
    confidence = gate_result.get("confidence", 0.0)
    
    status_class = "elite" if status == "ELITE" else "live-ok" if status == "LIVE_ELIGIBLE" else "wait" if status == "CONDITIONAL_PASS" else "fail"
    html.append(f'''
    <div class="card-body" style="text-align:center;padding:40px;">
        <div style="font-size:32px;font-weight:bold;margin-bottom:12px;
                    color:var(--{status_class}-color);">
            {status.replace("_", " ").title()}
        </div>
        <div style="font-size:20px;color:var(--text-muted);margin-bottom:20px;">
            Confidence: {confidence*100:.0f}%
        </div>
        <div style="font-size:16px;padding:16px;background:rgba(255,255,255,0.1);
                    border-radius:8px;max-width:600px;margin:0 auto;">
            {gate_result.get("reason", "No reason available")}
        </div>
        { "<div style='margin-top:20px;font-size:14px;color:#ff9800;'>⚠️ ACHTUNG: Mehrere Warnungen → Paper Trading empfohlen</div>" if "WARN" in str(rows) or confidence < 0.7 else "" }
    </div>
    ''')
    html.append('</div>')

    

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    html.append(
        f'<div class="footer">Generiert am {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} · '
        "Quant Validation Dashboard v5.0</div>"
    )
    html.append("</div>")  # container
    html.append("</body></html>")

    output_path = output_dir / f"{strategy_name}_dashboard.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(html))

    print(f"Dashboard generiert: {output_path}")
    return output_path


def render_html_for_strategy_dir(strategy_dir: Path) -> Path:
    """Hauptfunktion für run_pipeline.py – generiert Dashboard aus summary.json."""
    summary_path = strategy_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json nicht gefunden: {summary_path}")
    strategy_name = strategy_dir.name
    return generate_html_report(summary_path, strategy_dir, strategy_name)


if __name__ == "__main__":
    # Optionaler Testaufruf
    test_dir = Path("reports/RangeBreakoutUSDJPYv4")
    if test_dir.exists():
        render_html_for_strategy_dir(test_dir)
    else:
        print("Test-Ordner nicht gefunden. Bitte Pipeline ausführen.")
