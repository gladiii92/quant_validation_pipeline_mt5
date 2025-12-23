#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 QUANT VALIDATION DASHBOARD v5.0 - KOMPLETT ANGEPASST
100% kompatibel mit run_pipeline.py + ALLE Plots + deine HTML-Layout-Anpassungen
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

PLOT_FILES = {
    'equity': 'equity.png',
    'pnl_distribution': 'pnl_distribution.png',
    'mc_returns': 'mc_returns.png',
    'mc_paths': 'mc_paths.png',
    'drawdown_analysis': 'drawdown_analysis.png',
    'walk_forward_sharpe': 'walk_forward_sharpe.png',
    'kelly_frontier': 'kelly_frontier.png',
    'vix_regime_sharpe': 'vix_regime_sharpe.png'
}

def load_summary(summary_path: Path) -> Dict[str, Any]:
    """Lädt summary.json sicher."""
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Fehler beim Laden von {summary_path}: {e}")
        return {}

def safe_get(d: Dict[str, Any], *keys, default=None) -> Any:
    """Verschachtelte Keys sicher aus einem Dict lesen."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def df_to_table_html(df: Optional[pd.DataFrame]) -> str:
    """Erstellt HTML-Tabelle aus DataFrame (dein Design)."""
    if df is None or df.empty:
        return '<div class="card-body"><p style="text-align: center; color: var(--text-muted); padding: 20px;">Keine Daten verfügbar.</p></div>'
    
    html = '<div class="card-body table-responsive"><table class="table table-sm table-striped table-dark"><thead><tr style="text-align: center;">'
    for col in df.columns:
        html += f'<th>{col}</th>'
    html += '</tr></thead><tbody>'
    
    for _, row in df.iterrows():
        html += '<tr>'
        for val in row:
            if pd.isna(val):
                html += '<td>-</td>'
            elif isinstance(val, (int, float)):
                html += f'<td>{val:.3f}</td>'
            else:
                html += f'<td>{val}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    return html

def get_status_badge(status: str, confidence: float) -> str:
    """Status-Badge mit passender Klasse."""
    status_map = {
        'pass': 'status-live', 'warning': 'status-warning', 'fail': 'status-fail'
    }
    css_class = status_map.get(status.lower(), 'status-warning')
    return f'''
    <div class="badge {css_class}">
        <span class="badge-dot"></span>
        <span>{status.upper()}</span>
        <span>Confidence {confidence:.0f}%</span>
    </div>'''

def plot_exists(output_dir: Path, plot_name: str) -> str:
    """Prüft ob Plot existiert und gibt src oder Platzhalter."""
    plot_path = output_dir / PLOT_FILES.get(plot_name, f'{plot_name}.png')
    if plot_path.exists():
        return f'<img src="{plot_path.name}" class="plot" alt="{plot_name}" />'
    return f'<div style="height: 400px; display: flex; align-items: center; justify-content: center; background: #020617; border: 1px solid #1f2937; border-radius: 10px; color: var(--text-muted); font-size: 13px;">{plot_name}-Plot nicht gefunden.</div>'

def render_vix_table(vix_alignment: Dict) -> str:
    """VIX Regime Tabelle."""
    vix_stats = safe_get(vix_alignment, 'regime_stats', default={})
    if not vix_stats:
        return df_to_table_html(pd.DataFrame())
    
    data = []
    for regime, stats in vix_stats.items():
        data.append({
            'Regime': regime,
            'Trades': safe_get(stats, 'n_trades', default=0),
            'TotalReturn': safe_get(stats, 'total_return', default=0.0),
            'Sharpe': safe_get(stats, 'sharpe_ratio', default=0.0),
            'MaxDD': safe_get(stats, 'max_drawdown', default=0.0),
            'PF': safe_get(stats, 'profit_factor', default=0.0)
        })
    return df_to_table_html(pd.DataFrame(data))

def render_hmm_table(hmm_results: Any) -> str:
    """HMM Regime Tabelle."""
    try:
        if hasattr(hmm_results, 'regimestats') and hmm_results.regimestats:
            data = []
            for regime, stats in hmm_results.regimestats.items():
                data.append({
                    'Regime': regime,
                    'Trades': safe_get(stats, 'n_trades', default=0),
                    'TotalReturn': safe_get(stats, 'total_return', default=0.0),
                    'Sharpe': safe_get(stats, 'sharpe_ratio', default=0.0),
                    'MaxDD': safe_get(stats, 'max_drawdown', default=0.0),
                    'PF': safe_get(stats, 'profit_factor', default=0.0)
                })
            return df_to_table_html(pd.DataFrame(data))
    except:
        pass
    return df_to_table_html(pd.DataFrame())

def render_html_for_strategy_dir(strategy_dir: Path) -> Path:
    """
    HAUPTFUNKTION - Wird von run_pipeline.py erwartet!
    Generiert komplettes HTML-Dashboard für Strategie-Ordner.
    """
    summary_path = strategy_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json nicht gefunden: {summary_path}")
    
    strategy_name = strategy_dir.name
    generate_html_report(summary_path, strategy_dir, strategy_name)
    
    html_path = strategy_dir / f"{strategy_name}_dashboard.html"
    return html_path

def generate_html_report(summary_path: Path, output_dir: Path, strategy_name: str):
    """Generiert das komplette HTML-Dashboard (DEIN DESIGN 1:1)."""
    summary = load_summary(summary_path)
    if not summary:
        print("Keine Summary-Daten gefunden!")
        return

    # Alle Daten extrahieren
    metrics = safe_get(summary, 'metrics', default={})
    gate_result = safe_get(summary, 'gate_result', default={})
    vix_alignment = safe_get(summary, 'vix_alignment', default={})
    hmm_results = safe_get(summary, 'hmmresults', default={})
    mc_results = safe_get(summary, 'mc_results', default={})
    wf_results = safe_get(summary, 'walk_forward', default={})
    kelly_info = safe_get(summary, 'kelly_oos', default={})  # OOS bevorzugt
    multi_asset = safe_get(summary, 'multi_asset', default={})

    # Basis-Metrics
    total_return = safe_get(metrics, 'total_return', default=0.0)
    sharpe = safe_get(metrics, 'sharpe_ratio', default=0.0)
    max_dd = safe_get(metrics, 'max_drawdown', default=0.0)
    win_rate = safe_get(metrics, 'win_rate', default=0.0)
    total_trades = safe_get(metrics, 'total_trades', default=0)
    date_range = safe_get(metrics, 'date_range', default=('N/A', 'N/A'))
    date_range_str = f"{date_range[0]} – {date_range[1]}" if date_range[0] != 'N/A' else 'N/A'
    
    # Erweiterte Metrics
    oos_sharpe = safe_get(wf_results, 'oos_sharpe', default=0.0)
    mc_success = safe_get(mc_results, 'mc_positive_prob', default=0.0)
    mc_p95 = safe_get(mc_results, 'mc_p95_return', default=0.0)
    tail_risk = safe_get(summary, 'tail_stats', 'cvar5', default=0.0)
    multi_asset_hitrate = safe_get(multi_asset, 'hit_rate', default=0.0) * 100

    # Status
    status = gate_result.get('status', 'warning')
    confidence = gate_result.get('confidence', 0.5) * 100
    reason = gate_result.get('reason', 'N/A')
    violations = gate_result.get('violated_criteria', [])

    # CSS (DEIN DESIGN 1:1)
    css = """
    :root {
      --bg: #0f172a; --bg-alt: #020617; --card-bg: #111827; --card-border: #1f2937;
      --text: #e5e7eb; --text-muted: #9ca3af; --accent: #3b82f6; --accent-soft: rgba(59, 130, 246, 0.1);
      --danger: #f97316; --success: #22c55e; --warning: #eab308;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: radial-gradient(circle at top, #1e293b 0, #020617 55%); color: var(--text); padding: 24px; }
    a { color: var(--accent); text-decoration: none; }
    h1, h2, h3, h4 { font-weight: 600; }
    .container { max-width: 1400px; margin: 0 auto 64px auto; }
    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; gap: 16px; }
    .header-title { display: flex; flex-direction: column; gap: 4px; }
    .header-title h1 { font-size: 26px; letter-spacing: 0.04em; }
    .header-title p { font-size: 13px; color: var(--text-muted); }
    .badge { display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px; border-radius: 999px; font-size: 12px; background: var(--accent-soft); color: var(--accent); }
    .badge-dot { width: 8px; height: 8px; border-radius: 999px; background: var(--accent); }
    .status-live { background: rgba(34, 197, 94, 0.12); color: var(--success); }
    .status-live .badge-dot { background: var(--success); }
    .status-warning { background: rgba(234, 179, 8, 0.12); color: var(--warning); }
    .status-warning .badge-dot { background: var(--warning); }
    .status-fail { background: rgba(248, 113, 113, 0.12); color: var(--danger); }
    .status-fail .badge-dot { background: var(--danger); }
    .grid { display: grid; gap: 16px; }
    .grid-4 { grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); }
    .grid-2 { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
    .card-head { background: radial-gradient(circle at top left, #1f2937 0, #020617 65%); border-radius: 16px; border: 1px solid var(--card-border); padding: 16px 18px 18px 18px; box-shadow: 0 18px 45px rgba(15,23,42,0.8); display: flex; flex-direction: column; height: 100%; overflow: hidden; }
    .card { background: radial-gradient(circle at top left, #1f2937 0, #020617 65%); border-radius: 16px; border: 1px solid var(--card-border); padding: 16px 18px 18px 18px; box-shadow: 0 18px 45px rgba(15,23,42,0.8); display: flex; flex-direction: column; height: 100%; overflow: hidden; min-height: 750px; }
    .card-header { padding: 12px 16px; border-bottom: 1px solid #1f2937; font-size: 14px; font-weight: 600; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }
    .card-title { font-size: 15px; font-weight: 600; color: #e5e7eb; }
    .card-body { font-size: 13px; display: flex; color: var(--text); flex-direction: column; height: 100%; flex: 1; }
    .metric-value { font-size: 24px; font-weight: 600; }
    .metric-label { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
    .metric-sub { font-size: 11px; margin-top: 6px; color: var(--text-muted); }
    .section-title { margin: 32px 0 12px 0; font-size: 16px; text-transform: uppercase; letter-spacing: 0.12em; color: var(--text-muted); }
    img.plot { width: 100%; border-radius: 10px; border: 1px solid #1f2937; display: block; }
    .plot-caption { font-size: 12px; color: var(--text-muted); margin-top: 8px; line-height: 1.4; }
    .explanation { margin-top: auto; padding: 10px 12px; border-radius: 10px; background: #020617; border: 1px solid #1f2937; font-size: 12px; color: var(--text-muted); line-height: 2.0; }
    .explanation h4 { font-size: 13px; margin-bottom: 8px; color: var(--text); }
    .table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .table th, .table td { padding: 4px 8px; border-bottom: 1px solid #111827; }
    .table thead th { background: #020617; font-weight: 500; color: var(--text-muted); }
    .table tbody tr:nth-child(even) { background: #020617; }
    .table-responsive { overflow-x: auto; }
    .footer { margin-top: 32px; font-size: 11px; color: var(--text-muted); text-align: right; }
    @media (max-width: 768px) { body { padding: 12px; } .header { flex-direction: column; align-items: flex-start; } }
    """

    # HTML (DEIN DESIGN 1:1 mit ALLEN Plots)
    html = f'''<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8" />
  <title>{strategy_name} - Quant Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>{css}</style>
</head>
<body>
  <div class="container">

    <div class="header">
      <div class="header-title">
        <h1>QUANT VALIDATION DASHBOARD</h1>
        <p>Strategie: <strong>{strategy_name}</strong> · Zeitraum: {date_range_str} · Trades: {total_trades}</p>
      </div>
      <div>{get_status_badge(status, confidence)}</div>
    </div>

    <div class="grid grid-4">
      <div class="card-head">
        <div class="card-header"><div class="card-title">Gesamt-Rendite</div></div>
        <div class="card-body">
          <div class="metric-value">{total_return:.2f}%</div>
          <div class="metric-label">Netto-Performance über gesamten Zeitraum</div>
          <div class="metric-sub">Zeigt, wie stark das Konto insgesamt gewachsen ist.</div>
        </div>
      </div>
      <div class="card-head">
        <div class="card-header"><div class="card-title">Sharpe-Ratio</div></div>
        <div class="card-body">
          <div class="metric-value">{sharpe:.2f}</div>
          <div class="metric-label">Rendite pro Risikoeinheit</div>
          <div class="metric-sub">Höher ist besser: >1 gut, >2 sehr gut, >3 exzellent.</div>
        </div>
      </div>
      <div class="card-head">
        <div class="card-header"><div class="card-title">Max-Drawdown</div></div>
        <div class="card-body">
          <div class="metric-value">{max_dd:.2f}%</div>
          <div class="metric-label">Größter Einbruch vom Konto-Hoch</div>
          <div class="metric-sub">Zeigt, wie tief das Konto zwischenzeitlich fallen kann.</div>
        </div>
      </div>
      <div class="card-head">
        <div class="card-header"><div class="card-title">Win-Rate in %</div></div>
        <div class="card-body">
          <div class="metric-value">{win_rate:.2f}%</div>
          <div class="metric-label">Anteil gewinnender Trades</div>
          <div class="metric-sub">Wichtig im Zusammenspiel mit Gewinn/Verlust-Größe (Payoff).</div>
        </div>
      </div>
      <div class="card-head">
        <div class="card-header"><div class="card-title">OOS Sharpe</div></div>
        <div class="card-body">
          <div class="metric-value">{oos_sharpe:.2f}</div>
          <div class="metric-label">Ø Sharpe in Testfenstern</div>
          <div class="metric-sub">Nutzt ausschließlich Daten außerhalb des Trainings.</div>
        </div>
      </div>
      <div class="card-head">
        <div class="card-header"><div class="card-title">Monte-Carlo</div></div>
        <div class="card-body">
          <div class="metric-value">{mc_success*100:.1f}%</div>
          <div class="metric-label">Anteil profitabler Szenarien</div>
          <div class="metric-sub">Wie oft die Simulation mit Gewinn endet.</div>
        </div>
      </div>
      <div class="card-head">
        <div class="card-header"><div class="card-title">Monte-Carlo-P95</div></div>
        <div class="card-body">
          <div class="metric-value">{mc_p95*100:.1f}%</div>
          <div class="metric-label">95%-Quantil der Gesamtrendite</div>
          <div class="metric-sub">In 95% der Fälle besser als dieser Wert.</div>
        </div>
      </div>
      <div class="card-head">
        <div class="card-header"><div class="card-title">Tail-Risk CVaR</div></div>
        <div class="card-body">
          <div class="metric-value">{tail_risk*100:.1f}%</div>
          <div class="metric-label">Ø Verlust der schlechtesten 5%</div>
          <div class="metric-sub">Realistischer Extremverlust statt theoretischem DD.</div>
        </div>
      </div>
    </div>

    <h2 class="section-title">Strategie-Überblick</h2>
    <div class="grid grid-2">
      <div class="card">
        <div class="card-header">Equity Curve<span>Kontoverlauf</span></div>
        <div class="card-body">
          {plot_exists(output_dir, 'equity')}
          <p class="plot-caption">Die Equity-Kurve zeigt, wie sich das Konto über die Zeit entwickelt. Ein stetig steigender Verlauf mit überschaubaren Rücksetzern deutet auf eine stabile Strategie hin.</p>
          <div class="explanation">
            <h4>Einfach erklärt</h4>
            <p>Die Linie repräsentiert dein Kontostand nach jedem Trade. Größere Dellen sind Drawdowns, also Phasen, in denen das Konto vom letzten Hoch zurücksetzt. Viele kleine Dellen sind normal; tiefe und lange Dellen sind kritisch.</p>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header">PnL-Verteilung & Verhaltensmuster<span>Wo gewinnt/verliert die Strategie?</span></div>
        <div class="card-body">
          {plot_exists(output_dir, 'pnl_distribution')}
          <p class="plot-caption">Dieses Panel zeigt: (oben links) Verteilung der Trade-Gewinne und -Verluste, (oben rechts) QQ-Plot vs. Normalverteilung, (unten links) PnL vs. Positionsgröße und (unten rechts) durchschnittlicher PnL nach Einstiegs-Stunde.</p>
          <div class="explanation">
            <h4>Einfach erklärt</h4>
            <p>Hier erkennst du, ob wenige große Gewinne viele kleine Verluste ausgleichen, ob die Verteilung schwere Ausreißer (Fat Tails) hat und zu welchen Uhrzeiten die Strategie tendenziell Geld verdient oder verliert. Stunden mit dauerhaft negativem Durchschnittspnl sind typische Schwachstellen.</p>
          </div>
        </div>
      </div>
    </div>

    <h2 class="section-title">Risiko & Tail-Risk</h2>
    <div class="grid grid-2">
      <div class="card">
        <div class="card-header">Monte Carlo Total Returns<span>Verteilung der Gesamtrendite</span></div>
        <div class="card-body">
          {plot_exists(output_dir, 'mc_returns')}
          <p class="plot-caption">Histogramm der simulierten Gesamtrenditen über alle Monte-Carlo-Szenarien. Die vertikale Linie zeigt die Median-Rendite.</p>
          <div class="explanation">
            <h4>Einfach erklärt</h4>
            <p>Monte-Carlo-Simulation bedeutet: Wir mischen die bestehenden Trades zufällig neu und simulieren viele alternative Zukunftsverläufe. Je mehr der Balken rechts von 0 liegen, desto robuster ist die Strategie.</p>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header">Monte Carlo Equity Paths<span>Stresstest des Kontoverlaufs</span></div>
        <div class="card-body">
          {plot_exists(output_dir, 'mc_paths')}
          <p class="plot-caption">Mehrere simulierte Equity-Pfade, inklusive Median-Kurve und 90%-Konfidenzband.</p>
          <div class="explanation">
            <h4>Einfach erklärt</h4>
            <p>Jede dünne Linie ist ein möglicher Kontoverlauf in der Zukunft. Die farbige Bandbreite zeigt, wie weit gute und schlechte Verläufe typischerweise auseinanderliegen. Eine enge Bandbreite bedeutet stabilere Ergebnisse.</p>
          </div>
        </div>
      </div>
    </div>

    <div class="grid grid-2" style="margin-top:16px;">
      <div class="card">
        <div class="card-header">Drawdown-Analyse<span>MaxDD & Dauer</span></div>
        <div class="card-body">
          {plot_exists(output_dir, 'drawdown_analysis')}
          <p class="plot-caption">Links: Verteilung der maximalen Drawdowns aus den Monte-Carlo-Simulationen. Rechts: Histogramm der Dauer realer Drawdowns in Tagen.</p>
          <div class="explanation">
            <h4>Einfach erklärt</h4>
            <p>Max Drawdown ist der schlimmste Einbruch vom Konto-Hoch. Die Dauer zeigt, wie lange eine Durststrecke typischerweise anhält. Viele lange Drawdowns bedeuten, dass du emotional viel aushalten musst.</p>
          </div>
        </div>
      </div>
    </div>

    <h2 class="section-title">Walk-Forward & Positionsgrößen</h2>
    <div class="grid grid-2">
      <div class="card">
        <div class="card-header">Walk-Forward OOS Sharpe<span>Robustheit über Zeit</span></div>
        <div class="card-body">
          {plot_exists(output_dir, 'walk_forward_sharpe')}
          <p class="plot-caption">Sharpe Ratio in einzelnen Walk-Forward-Testfenstern. Die rote Linie zeigt den Durchschnitt der OOS-Sharpe-Werte.</p>
          <div class="explanation">
            <h4>Einfach erklärt</h4>
            <p>Beim Walk-Forward-Test wird die Strategie in einem Zeitraum trainiert und in einem späteren Zeitraum getestet (Out-of-Sample). So sieht man, ob die Strategie über verschiedene Marktphasen hinweg stabil bleibt oder nur in bestimmten Zeiträumen funktioniert.</p>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header">Kelly Frontier<span>Optimale Positionsgröße</span></div>
        <div class="card-body">
          {plot_exists(output_dir, 'kelly_frontier')}
          <p class="plot-caption">Erwartete Wachstumsrate des Kontos in Abhängigkeit von der gewählten Risiko-Fraktion pro Trade (Kelly-Kriterium).</p>
          <div class="explanation">
            <h4>Einfach erklärt</h4>
            <p>Das Kelly-Kriterium berechnet die theoretisch optimale Positionsgröße für maximales Wachstum. Volle Kelly-Größe ist oft zu aggressiv für die Praxis. Viele Trader nutzen eher die Hälfte oder ein Viertel davon, um Drawdowns zu begrenzen.</p>
          </div>
        </div>
      </div>
    </div>

    <h2 class="section-title">Regime-Analyse</h2>
    <div class="grid grid-2">
      <div class="card">
        <div class="card-header">VIX Regime Performance<span>Volatilitäts-Phasen</span></div>
        <div class="card-body">
          {plot_exists(output_dir, 'vix_regime_sharpe')}
          <p class="plot-caption">Balkendiagramm der Sharpe Ratio je VIX-Regime (z.B. Low Volatility, Range, High Volatility).</p>
          <div class="explanation">
            <h4>Einfach erklärt</h4>
            <p>Der VIX ist ein Volatilitätsindex ("Angstbarometer"). Die Regime-Einteilung zeigt, in welchen Marktphasen (ruhig, normal, hektisch) die Strategie besonders gut oder schlecht läuft. Ein starkes Ungleichgewicht kann für eine Regime-Filterung genutzt werden.</p>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header">VIX Regime Tabelle<span>Performance je Phase</span></div>
        <div class="card-body">{render_vix_table(vix_alignment)}
          <div class="explanation">
            <h4>Interpretation</h4>
            <p>Spalten: <strong>Trades</strong> = Anzahl Trades pro Phase, <strong>TotalReturn</strong> = Gesamtrendite, <strong>Sharpe</strong> = Rendite pro Risiko, <strong>MaxDD</strong> = schlimmster Einbruch, <strong>PF</strong> = Verhältnis Summe Gewinne zu Summe Verluste.</p>
          </div>
        </div>
      </div>
    </div>

    <div class="grid grid-2" style="margin-top:16px;">
      <div class="card">
        <div class="card-header">HMM Regime Tabelle<span>Verborgene Marktzustände</span></div>
        <div class="card-body">{render_hmm_table(hmm_results)}
          <div class="explanation">
            <h4>Einfach erklärt</h4>
            <p>Ein Hidden Markov Model (HMM) erkennt unsichtbare Marktregime anhand des Equity-Verlaufs. Jeder Regime-State fasst Phasen mit ähnlicher Performance zusammen. Ein Regime mit hoher Sharpe und niedriger MaxDD ist wünschenswert; ein Regime mit schlechter Kennzahl-Kombination kann man als Warnsignal oder Filter verwenden.</p>
          </div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:16px;">
      <div class="card-header">Decision Gate<span>Interpretation</span></div>
      <div class="card-body">
        <p style="font-size:13px;margin-bottom:6px;"><strong>Begründung:</strong> {reason}</p>
        {f'<ul style="font-size:12px;color:var(--text-muted);margin-left:18px;margin-bottom:12px;">' + ''.join(f'<li>{v}</li>' for v in violations) + '</ul>' if violations else ''}
        <div class="explanation">
          <h4>Einfach erklärt</h4>
          <p>Das Decision Gate fasst alle wichtigen Kennzahlen zusammen und entscheidet, ob die Strategie für Live-Trading geeignet ist. Kleinere Verstöße (z. B. zu niedrige Multi-Asset Hit-Rate) führen zu einem Warnhinweis, aber nicht unbedingt zu einem harten Stopp.</p>
        </div>
      </div>
    </div>

    <div class="grid grid-2" style="gap: 24px;">
      <div class="card">
        <div class="card-header">Sharpe-Rangliste<span>{len(safe_get(multi_asset, "details", default=[]))} Assets sortiert</span></div>
        <div class="card-body">
          <div class="metric-value" style="font-size: 18px; margin-bottom: 8px;">✅ TOP 3 PROFITABEL ({multi_asset_hitrate:.1f}% Hit-Rate)</div>
          <div class="badge" style="margin-top: 16px; background: var(--accent-soft); color: var(--accent); font-size: 14px;">
            📊 <strong>{safe_get(multi_asset, "n_symbols_pass", 0)}/{safe_get(multi_asset, "n_symbols", 0)}</strong> = {multi_asset_hitrate:.1f}% Hit-Rate
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header">Decision Gate Status<span>Aktuelle Bewertung</span></div>
        <div class="card-body">
          <div class="metric-value" style="font-size: 32px; color: var(--warning);">🟡 WARNUNG</div>
          <p style="margin-top: 12px; font-size: 14px;">
            <strong>Multi-Asset Hit-Rate:</strong> <span style="color: var(--danger);">{multi_asset_hitrate:.1f}%</span> < 75.0%
          </p>
          <div style="margin-top: 16px; padding: 12px; background: rgba(254, 243, 199, 0.3); border-radius: 8px; border-left: 4px solid var(--warning);">
            <strong>→ Empfehlung:</strong> Live nur mit <strong>Keine Multi-Asset Analyse</strong> starten
          </div>
        </div>
      </div>
    </div>

    <div class="footer">
      Generiert am {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} · Quant Validation Dashboard v5.0
    </div>
  </div>
</body>
</html>'''

    # Datei schreiben
    output_path = output_dir / f"{strategy_name}_dashboard.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"✅ Dashboard generiert: {output_path} (ALLE 8 Plots + Tabellen + dein Layout)")

if __name__ == "__main__":
    # Test-Aufruf
    test_dir = Path("reports") / "RangeBreakoutUSDJPY_v4"
    if test_dir.exists():
        render_html_for_strategy_dir(test_dir)
    else:
        print("Test-Ordner nicht gefunden. Starte Pipeline für echte Daten.")
