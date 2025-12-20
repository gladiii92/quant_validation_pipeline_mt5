"""
🏆 SENIOR QUANT DASHBOARD v5.0
Modernes HTML-Dashboard mit allen Plots + Laien-Erklärungen
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import pandas as pd


# ============================================================
# Hilfsfunktionen
# ============================================================

def load_summary(summary_path: Path) -> Dict[str, Any]:
    """Lädt summary.json sicher."""
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: Dict[str, Any], *keys, default=None):
    """Verschachtelte Keys sicher aus einem Dict lesen."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def df_to_html(df: pd.DataFrame, index: bool = True, title: str = "") -> str:
    """Bootstrap-ähnliche Tabelle aus DataFrame erzeugen."""
    if df is None or df.empty:
        return f"""
        <div class="card">
          <div class="card-header">{title}</div>
          <div class="card-body">
            <p>Keine Daten verfügbar.</p>
          </div>
        </div>
        """

    table_html = df.to_html(
        classes="table table-sm table-striped table-dark",
        border=0,
        index=index,
        justify="center",
        escape=False,
    )
    return f"""
    <div class="card">
      <div class="card-header">{title}</div>
      <div class="card-body table-responsive">
        {table_html}
      </div>
    </div>
    """


# ============================================================
# HTML-Template
# ============================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8" />
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {{
      --bg: #0f172a;
      --bg-alt: #020617;
      --card-bg: #111827;
      --card-border: #1f2937;
      --text: #e5e7eb;
      --text-muted: #9ca3af;
      --accent: #3b82f6;
      --accent-soft: rgba(59, 130, 246, 0.1);
      --danger: #f97316;
      --success: #22c55e;
      --warning: #eab308;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #1e293b 0, #020617 55%);
      color: var(--text);
      padding: 24px;
    }}

    a {{
      color: var(--accent);
      text-decoration: none;
    }}

    h1, h2, h3, h4 {{
      font-weight: 600;
    }}

    .container {{
      max-width: 1400px;
      margin: 0 auto 64px auto;
    }}

    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      gap: 16px;
    }}

    .header-title {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}

    .header-title h1 {{
      font-size: 26px;
      letter-spacing: 0.04em;
    }}

    .header-title p {{
      font-size: 13px;
      color: var(--text-muted);
    }}

    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      background: var(--accent-soft);
      color: var(--accent);
    }}

    .badge-dot {{
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--accent);
    }}

    .status-live {{
      background: rgba(34, 197, 94, 0.12);
      color: var(--success);
    }}
    .status-live .badge-dot {{
      background: var(--success);
    }}

    .status-warning {{
      background: rgba(234, 179, 8, 0.12);
      color: var(--warning);
    }}
    .status-warning .badge-dot {{
      background: var(--warning);
    }}

    .status-fail {{
      background: rgba(248, 113, 113, 0.12);
      color: var(--danger);
    }}
    .status-fail .badge-dot {{
      background: var(--danger);
    }}

    .grid {{
      display: grid;
      gap: 16px;
    }}

    .grid-4 {{
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    }}

    .grid-2 {{
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }}

    .card {{
      background: radial-gradient(circle at top left, #1f2937 0, #020617 65%);
      border-radius: 16px;
      border: 1px solid var(--card-border);
      padding: 16px 18px 18px 18px;
      box-shadow: 0 18px 45px rgba(15,23,42,0.8);
    }}

    .card-header {{
      padding: 12px 16px;
      border-bottom: 1px solid #1f2937;
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .card-header span {{
      font-size: 11px;
      font-weight: 500;
    }}

    .card-kicker {{
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
        margin-bottom: 2px;
    }}

    .card-title {{
        font-size: 15px;
        font-weight: 600;
        color: #e5e7eb;
    }}
    

    .card-body {{
      font-size: 13px;
      color: var(--text);
    }}

    .metric-value {{
      font-size: 24px;
      font-weight: 600;
    }}

    .metric-label {{
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 2px;
    }}

    .metric-sub {{
      font-size: 11px;
      margin-top: 6px;
      color: var(--text-muted);
    }}

    .tag {{
      display: inline-flex;
      align-items: center;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      background: #020617;
      border: 1px solid #1f2937;
      color: var(--text-muted);
      margin-left: 4px;
    }}

    .section-title {{
      margin: 32px 0 12px 0;
      font-size: 16px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--text-muted);
    }}

    img.plot {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid #1f2937;
      display: block;
    }}

    .plot-caption {{
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 8px;
      line-height: 1.4;
    }}

    .explanation {{
      margin-top: 10px;
      padding: 10px 12px;
      border-radius: 10px;
      background: #020617;
      border: 1px solid #1f2937;
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.5;
    }}

    .explanation h4 {{
      font-size: 13px;
      margin-bottom: 4px;
      color: var(--text);
    }}

    .pill-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }}

    .pill {{
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 999px;
      border: 1px solid #1f2937;
      background: #020617;
      color: var(--text-muted);
    }}

    .table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}

    .table th,
    .table td {{
      padding: 4px 8px;
      border-bottom: 1px solid #111827;
    }}

    .table thead th {{
      background: #020617;
      font-weight: 500;
      color: var(--text-muted);
    }}

    .table tbody tr:nth-child(even) {{
      background: #020617;
    }}

    .footer {{
      margin-top: 32px;
      font-size: 11px;
      color: var(--text-muted);
      text-align: right;
    }}

    @media (max-width: 768px) {{
      body {{
        padding: 12px;
      }}
      .header {{
        flex-direction: column;
        align-items: flex-start;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
"""

# ============================================================
# Hauptfunktion: HTML für eine Strategy-Directory erzeugen
# ============================================================

def render_html_for_strategy_dir(strategy_dir: Path) -> Path:
    """
    Erzeugt das Quant-Dashboard HTML aus summary.json + PNG-Plots.
    """
    strategy_dir = Path(strategy_dir)
    summary_path = strategy_dir / "summary.json"
    summary = load_summary(summary_path)

    metrics = summary.get("metrics", {})
    cost_results = summary.get("cost_results", {})
    kelly_info = summary.get("kelly", {})
    kelly_oos = summary.get("kelly_oos", {})
    vix_alignment = summary.get("vix_alignment", {})
    mc_results = summary.get("mc_results", {})
    wf_results = summary.get("walkforward", {})
    multi_asset = summary.get("multi_asset", {})
    sim_results = summary.get("sim_results", {})
    hmm_results = summary.get("hmm_results", {})
    gate = summary.get("gate_result", {})

    strategy_name = metrics.get("strategy_name", strategy_dir.name)
    date_range = metrics.get("date_range", (None, None))

    total_return = metrics.get("total_return", 0.0)
    sharpe = metrics.get("sharpe_ratio", 0.0)
    max_dd = metrics.get("max_drawdown", 0.0)
    win_rate = metrics.get("win_rate", 0.0)
    total_trades = metrics.get("total_trades", 0)

    oos_sharpe = wf_results.get("oos_sharpe", 0.0)
    oos_pf = wf_results.get("oos_profit_factor", 0.0)
    oos_maxdd = wf_results.get("oos_maxdd", 0.0)

    mc_pos_prob = mc_results.get("mc_positive_prob", 0.0)
    mc_p95_ret = mc_results.get("mc_p95_return", 0.0)
    cvar5 = safe_get(summary, "mc_results", "cvar5", default=None) or safe_get(summary, "gateresult", "cvar5", default=0.0)

    gate_status = gate.get("status", "N/A")
    gate_confidence = gate.get("confidence", 0.0)
    gate_reason = gate.get("reason", "")
    violated = gate.get("violated_criteria", []) or gate.get("violatedcriteria", [])

    # Badge-Statusklasse
    violated_any = bool(gate_reason or violated)
    status_display = gate_status if not violated_any else 'WATCHLIST'
    statusclass = 'status-live' if status_display == 'LIVE_ELIGIBLE' else 'status-warning' if status_display in ['WATCHLIST', 'REVIEW'] else 'status-fail'

    # Header / Summary
    html_parts = []

    header_html = f"""
    <div class="header">
    <div class="header-title">
        <h1>QUANT VALIDATION DASHBOARD</h1>
        <p>Strategie: <strong>{strategy_name}</strong> · Zeitraum: 
        {date_range[0]} – {date_range[1]} · Trades: {total_trades}</p>
    </div>
    <div>
        <div class="badge status-{statusclass}">
        <span class="badge-dot"></span>
        <span>{status_display}</span>
        <span>Confidence {gate_confidence:.0f}%</span>
        </div>
    </div>
    </div>
    """

    summary_html = f"""
    <div class="grid grid-4">

    <div class="card">
        <div class="card-header">
        <div class="card-title">Gesamt-Rendite</div>
        </div>
        <div class="card-body">
        <div class="metric-value">{total_return*100:.2f}%</div>
        <div class="metric-label">Netto-Performance über gesamten Zeitraum</div>
        <div class="metric-sub">
            Zeigt, wie stark das Konto insgesamt gewachsen ist.
        </div>
        </div>
    </div>

    <div class="card">
        <div class="card-header">
        <div class="card-title">Sharpe-Ratio</div>
        </div>
        <div class="card-body">
        <div class="metric-value">{sharpe:.2f}</div>
        <div class="metric-label">Rendite pro Risikoeinheit</div>
        <div class="metric-sub">
            Höher ist besser: &gt;1 gut, &gt;2 sehr gut, &gt;3 exzellent.
        </div>
        </div>
    </div>

    <div class="card">
        <div class="card-header">
        <div class="card-title">Max-Drawdown</div>
        </div>
        <div class="card-body">
        <div class="metric-value">{max_dd*100:.2f}%</div>
        <div class="metric-label">Größter Einbruch vom Konto-Hoch</div>
        <div class="metric-sub">
            Zeigt, wie tief das Konto zwischenzeitlich fallen kann.
        </div>
        </div>
    </div>

    <div class="card">
        <div class="card-header">
        <div class="card-title">Win-Rate in %</div>
        </div>
        <div class="card-body">
        <div class="metric-value">{win_rate*100:.2f}%</div>
        <div class="metric-label">Anteil gewinnender Trades</div>
        <div class="metric-sub">
            Wichtig im Zusammenspiel mit Gewinn/Verlust-Größe (Payoff).
        </div>
        </div>
    </div>

    <div class="card">
        <div class="card-header">
        <div class="card-title">OOS Sharpe</div>
        </div>
        <div class="card-body">
        <div class="metric-value">{oos_sharpe:.2f}</div>
        <div class="metric-label">Ø Sharpe in Testfenstern</div>
        <div class="metric-sub">
            Nutzt ausschließlich Daten außerhalb des Trainings.
        </div>
        </div>
    </div>

    <div class="card">
        <div class="card-header">
        <div class="card-title">Monte-Carlo</div>
        </div>
        <div class="card-body">
        <div class="metric-value">{mc_pos_prob*100:.1f}%</div>
        <div class="metric-label">Anteil profitabler Szenarien</div>
        <div class="metric-sub">
            Wie oft die Simulation mit Gewinn endet.
        </div>
        </div>
    </div>

    <div class="card">
        <div class="card-header">
        <div class="card-title">Monte-Carlo-P95</div>
        </div>
        <div class="card-body">
        <div class="metric-value">{mc_p95_ret*100:.1f}%</div>
        <div class="metric-label">95%-Quantil der Gesamtrendite</div>
        <div class="metric-sub">
            In 95% der Fälle besser als dieser Wert.
        </div>
        </div>
    </div>

    <div class="card">
        <div class="card-header">
        <div class="card-title">Tail-Risk CVaR</div>
        </div>
        <div class="card-body">
        <div class="metric-value">{cvar5*100:.1f}%</div>
        <div class="metric-label">Ø Verlust der schlechtesten 5%</div>
        <div class="metric-sub">
            Realistischer Extremverlust statt theoretischem DD.
        </div>
        </div>
    </div>

    </div>
    """

    # Gate-Reason + Violations
    violations_html = ""
    if gate_reason or violated:
        items = ""
        for v in violated:
            items += f"<li>{v}</li>"
        violations_html = f"""
        <div class="card" style="margin-top:16px;">
          <div class="card-header">Decision Gate<span>Interpretation</span></div>
          <div class="card-body">
            <p style="font-size:13px;margin-bottom:6px;">
              <strong>Begründung:</strong> {gate_reason or "N/A"}
            </p>
            {"<ul style='font-size:12px;color:var(--text-muted);margin-left:18px;'>" + items + "</ul>" if items else ""}
            <div class="explanation">
              <h4>Einfach erklärt</h4>
              <p>
                Das Decision Gate fasst alle wichtigen Kennzahlen zusammen und
                entscheidet, ob die Strategie für Live-Trading geeignet ist.
                Kleinere Verstöße (z.&nbsp;B. zu niedrige Multi-Asset Hit-Rate)
                führen zu einem Warnhinweis, aber nicht unbedingt zu einem harten Stopp.
              </p>
            </div>
          </div>
        </div>
        """

    html_parts.append(HTML_TEMPLATE.format(title=f"{strategy_name} - Quant Dashboard"))
    html_parts.append(header_html)
    html_parts.append(summary_html)
    html_parts.append(violations_html)

    # ========================================================
    # Sektion: Strategie-Überblick (Equity + PnL)
    # ========================================================

    equity_png = strategy_dir / "equity.png"
    pnl_png = strategy_dir / "pnl_distribution.png"

    overview_html = f"""
    <h2 class="section-title">Strategie-Überblick</h2>
    <div class="grid grid-2">
      <div class="card">
        <div class="card-header">Equity Curve<span>Kontoverlauf</span></div>
        <div class="card-body">
          {"<img src='equity.png' class='plot' alt='Equity Curve' />" if equity_png.exists() else "<p>Equity-Plot nicht gefunden.</p>"}
          <p class="plot-caption">
            Die Equity-Kurve zeigt, wie sich das Konto über die Zeit entwickelt.
            Ein stetig steigender Verlauf mit überschaubaren Rücksetzern deutet
            auf eine stabile Strategie hin.
          </p>
          <div class="explanation">
            <h4>Einfach erklärt</h4>
            <p>
              Die Linie repräsentiert dein Kontostand nach jedem Trade.
              Größere Dellen sind Drawdowns, also Phasen, in denen das Konto
              vom letzten Hoch zurücksetzt. Viele kleine Dellen sind normal;
              tiefe und lange Dellen sind kritisch.
            </p>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">PnL-Verteilung & Verhaltensmuster<span>Wo gewinnt/verliert die Strategie?</span></div>
        <div class="card-body">
          {"<img src='pnl_distribution.png' class='plot' alt='PnL Distribution' />" if pnl_png.exists() else "<p>PnL-Plot nicht gefunden.</p>"}
          <p class="plot-caption">
            Dieses Panel zeigt: (oben links) Verteilung der Trade-Gewinne und -Verluste,
            (oben rechts) QQ-Plot vs. Normalverteilung, (unten links) PnL vs. Positionsgröße
            und (unten rechts) durchschnittlicher PnL nach Einstiegs-Stunde.
          </p>
          <div class="explanation">
            <h4>Einfach erklärt</h4>
            <p>
              Hier erkennst du, ob wenige große Gewinne viele kleine Verluste ausgleichen,
              ob die Verteilung schwere Ausreißer (Fat Tails) hat und zu welchen Uhrzeiten
              die Strategie tendenziell Geld verdient oder verliert.
              Stunden mit dauerhaft negativem Durchschnittspnl sind typische Schwachstellen.
            </p>
          </div>
        </div>
      </div>
    </div>
    """
    html_parts.append(overview_html)

    # ========================================================
    # Sektion: Risiko & Tail-Risk (Monte Carlo + Drawdown)
    # ========================================================

    mc_returns_png = strategy_dir / "mc_returns.png"
    mc_paths_png = strategy_dir / "mc_paths.png"
    dd_png = strategy_dir / "drawdown_analysis.png"

    risk_html = f"""
    <h2 class="section-title">Risiko & Tail-Risk</h2>
    <div class="grid grid-2">
      <div class="card">
        <div class="card-header">Monte Carlo Total Returns<span>Verteilung der Gesamtrendite</span></div>
        <div class="card-body">
          {"<img src='mc_returns.png' class='plot' alt='Monte Carlo Returns' />" if mc_returns_png.exists() else "<p>Monte-Carlo-Return-Plot nicht gefunden.</p>"}
          <p class="plot-caption">
            Histogramm der simulierten Gesamtrenditen über alle Monte-Carlo-Szenarien.
            Die vertikale Linie zeigt die Median-Rendite.
          </p>
          <div class="explanation">
            <h4>Einfach erklärt</h4>
            <p>
              Monte-Carlo-Simulation bedeutet: Wir mischen die bestehenden Trades
              zufällig neu und simulieren viele alternative Zukunftsverläufe.
              Je mehr der Balken rechts von 0 liegen, desto robuster ist die Strategie.
            </p>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">Monte Carlo Equity Paths<span>Stresstest des Kontoverlaufs</span></div>
        <div class="card-body">
          {"<img src='mc_paths.png' class='plot' alt='Monte Carlo Paths' />" if mc_paths_png.exists() else "<p>Monte-Carlo-Pfade-Plot nicht gefunden.</p>"}
          <p class="plot-caption">
            Mehrere simulierte Equity-Pfade, inklusive Median-Kurve und 90%-Konfidenzband.
          </p>
          <div class="explanation">
            <h4>Einfach erklärt</h4>
            <p>
              Jede dünne Linie ist ein möglicher Kontoverlauf in der Zukunft.
              Die farbige Bandbreite zeigt, wie weit gute und schlechte Verläufe
              typischerweise auseinanderliegen. Eine enge Bandbreite bedeutet
              stabilere Ergebnisse.
            </p>
          </div>
        </div>
      </div>
    </div>

    <div class="grid grid-2" style="margin-top:16px;">
      <div class="card">
        <div class="card-header">Drawdown-Analyse<span>MaxDD & Dauer</span></div>
        <div class="card-body">
          {"<img src='drawdown_analysis.png' class='plot' alt='Drawdown Analysis' />" if dd_png.exists() else "<p>Drawdown-Plot nicht gefunden.</p>"}
          <p class="plot-caption">
            Links: Verteilung der maximalen Drawdowns aus den Monte-Carlo-Simulationen.
            Rechts: Histogramm der Dauer realer Drawdowns in Tagen.
          </p>
          <div class="explanation">
            <h4>Einfach erklärt</h4>
            <p>
              Max Drawdown ist der schlimmste Einbruch vom Konto-Hoch.
              Die Dauer zeigt, wie lange eine Durststrecke typischerweise anhält.
              Viele lange Drawdowns bedeuten, dass du emotional viel aushalten musst.
            </p>
          </div>
        </div>
      </div>
    </div>
    """
    html_parts.append(risk_html)

    # ========================================================
    # Sektion: Walk-Forward & Kelly
    # ========================================================

    wf_png = strategy_dir / "walk_forward_sharpe.png"
    kelly_png = strategy_dir / "kelly_frontier.png"

    wf_html = f"""
    <h2 class="section-title">Walk-Forward & Positionsgrößen</h2>
    <div class="grid grid-2">
      <div class="card">
        <div class="card-header">Walk-Forward OOS Sharpe<span>Robustheit über Zeit</span></div>
        <div class="card-body">
          {"<img src='walk_forward_sharpe.png' class='plot' alt='Walk-Forward Sharpe' />" if wf_png.exists() else "<p>Walk-Forward-Plot nicht gefunden.</p>"}
          <p class="plot-caption">
            Sharpe Ratio in einzelnen Walk-Forward-Testfenstern.
            Die rote Linie zeigt den Durchschnitt der OOS-Sharpe-Werte.
          </p>
          <div class="explanation">
            <h4>Einfach erklärt</h4>
            <p>
              Beim Walk-Forward-Test wird die Strategie in einem Zeitraum trainiert
              und in einem späteren Zeitraum getestet (Out-of-Sample). So sieht man,
              ob die Strategie über verschiedene Marktphasen hinweg stabil bleibt
              oder nur in bestimmten Zeiträumen funktioniert.
            </p>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">Kelly Frontier<span>Optimale Positionsgröße</span></div>
        <div class="card-body">
          {"<img src='kelly_frontier.png' class='plot' alt='Kelly Frontier' />" if kelly_png.exists() else "<p>Kelly-Plot nicht gefunden.</p>"}
          <p class="plot-caption">
            Erwartete Wachstumsrate des Kontos in Abhängigkeit von der gewählten
            Risiko-Fraktion pro Trade (Kelly-Kriterium).
          </p>
          <div class="explanation">
            <h4>Einfach erklärt</h4>
            <p>
              Das Kelly-Kriterium berechnet die theoretisch optimale
              Positionsgröße für maximales Wachstum. Volle Kelly-Größe ist oft
              zu aggressiv für die Praxis. Viele Trader nutzen eher die Hälfte
              oder ein Viertel davon, um Drawdowns zu begrenzen.
            </p>
          </div>
        </div>
      </div>
    </div>
    """
    html_parts.append(wf_html)

    # ========================================================
    # Sektion: Regime-Analyse (VIX + HMM)
    # ========================================================

    vix_png = strategy_dir / "vix_regime_sharpe.png"
    vix_stats = (vix_alignment or {}).get("regime_stats", {})

    vix_rows = []
    for rname, m in (vix_stats or {}).items():
        vix_rows.append(
            dict(
                Regime=rname,
                Trades=m.get("n_trades", 0),
                TotalReturn=f"{m.get('total_return', 0):.2f}",
                Sharpe=f"{m.get('sharpe_ratio', 0):.2f}",
                MaxDD=f"{m.get('max_drawdown', 0):.2f}",
                PF=f"{m.get('profit_factor', 0):.2f}",
            )
        )
    vix_df = pd.DataFrame(vix_rows)

    hmm_regimestats = (hmm_results or {}).get("regimestats", {})
    hmm_rows = []
    for rname, m in (hmm_regimestats or {}).items():
        hmm_rows.append(
            dict(
                Regime=rname,
                Trades=m.get("n_trades", 0),
                TotalReturn=f"{m.get('total_return', 0):.2f}",
                Sharpe=f"{m.get('sharpe_ratio', 0):.2f}",
                MaxDD=f"{m.get('max_drawdown', 0):.2f}",
                PF=f"{m.get('profit_factor', 0):.2f}",
            )
        )
    hmm_df = pd.DataFrame(hmm_rows)

    regime_html = f"""
    <h2 class="section-title">Regime-Analyse</h2>
    <div class="grid grid-2">
      <div class="card">
        <div class="card-header">VIX Regime Performance<span>Volatilitäts-Phasen</span></div>
        <div class="card-body">
          {"<img src='vix_regime_sharpe.png' class='plot' alt='VIX Regime Sharpe' />" if vix_png.exists() else "<p>VIX-Plot nicht gefunden.</p>"}
          <p class="plot-caption">
            Balkendiagramm der Sharpe Ratio je VIX-Regime (z.B. Low Volatility, Range, High Volatility).
          </p>
          <div class="explanation">
            <h4>Einfach erklärt</h4>
            <p>
              Der VIX ist ein Volatilitätsindex ("Angstbarometer"). Die
              Regime-Einteilung zeigt, in welchen Marktphasen (ruhig, normal,
              hektisch) die Strategie besonders gut oder schlecht läuft.
              Ein starkes Ungleichgewicht kann für eine Regime-Filterung genutzt werden.
            </p>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">VIX Regime Tabelle<span>Performance je Phase</span></div>
        <div class="card-body">
          {df_to_html(vix_df, index=False, title="VIX Regime Performance")}
          <div class="explanation">
            <h4>Interpretation</h4>
            <p>
              Spalten:
              <strong>Trades</strong> = Anzahl Trades pro Phase,
              <strong>TotalReturn</strong> = Gesamtrendite,
              <strong>Sharpe</strong> = Rendite pro Risiko,
              <strong>MaxDD</strong> = schlimmster Einbruch,
              <strong>PF</strong> = Verhältnis Summe Gewinne zu Summe Verluste.
            </p>
          </div>
        </div>
      </div>
    </div>

    <div class="grid grid-2" style="margin-top:16px;">
      <div class="card">
        <div class="card-header">HMM Regime Tabelle<span>Verborgene Marktzustände</span></div>
        <div class="card-body">
          {df_to_html(hmm_df, index=False, title="HMM Regime Performance")}
          <div class="explanation">
            <h4>Einfach erklärt</h4>
            <p>
              Ein Hidden Markov Model (HMM) erkennt unsichtbare Marktregime anhand
              des Equity-Verlaufs. Jeder Regime-State fasst Phasen mit ähnlicher
              Performance zusammen. Ein Regime mit hoher Sharpe und niedriger
              MaxDD ist wünschenswert; ein Regime mit schlechter Kennzahl-Kombination
              kann man als Warnsignal oder Filter verwenden.
            </p>
          </div>
        </div>
      </div>
    </div>
    """
    html_parts.append(regime_html)

    # ========================================================
    # Sektion: Stochastische Modelle & Multi-Asset
    # ========================================================

    # Stochastische Modelle
    sim_rows = []
    for name, m in (sim_results or {}).items():
        sim_rows.append(
            dict(
                Model=name,
                MedianReturn=f"{m.get('median_return', 0):.2f}",
                P5=f"{m.get('p5_return', 0):.2f}",
                P95=f"{m.get('p95_return', 0):.2f}",
                MedianMaxDD=f"{m.get('median_maxdd', 0):.2f}",
                P95MaxDD=f"{m.get('p95_maxdd', 0):.2f}",
            )
        )
    sim_df = pd.DataFrame(sim_rows)

    multiasset = summary.get('multiasset', None) or {}
    multiasset_details = multiasset.get('details', [])
    hit_rate = multiasset.get('hit_rate', 0.0)

    # DataFrame nur wenn Details vorhanden
    madf = None
    if multiasset_details:
        madf = pd.DataFrame(multiasset_details)
        if not madf.empty and 'sharpe' in madf.columns:
            madf = madf.sort_values('sharpe', ascending=False)
            top_symbols = ', '.join(madf.head(3)['symbol'].astype(str).tolist())
        else:
            top_symbols = 'Keine gültigen Daten'
    else:
        top_symbols = 'Keine Multi-Asset Analyse'

    # Multi-Asset
    multiasset_html = f"""
    <div class="grid grid-2" style="gap: 24px;">
    <!-- Sharpe-Rangliste (DEIN DESIGN) -->
    <div class="card">
        <div class="card-header">Sharpe-Rangliste<span>{len(multiasset_details)} Assets sortiert</span></div>
        <div class="card-body">
        <div class="metric-value" style="font-size: 18px; margin-bottom: 8px;">✅ TOP 3 PROFITABEL ({hit_rate:.1f}% Hit-Rate)</div>
        
        <!-- TOP 3 DYNAMISCH -->
        <div style="margin-bottom: 20px;">
            {''.join([f'''
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
            <div style="width: 250px; height: 20px; background: linear-gradient(to right, #22c55e 85%, #4ade80 85%); border-radius: 10px; box-shadow: 0 2px 4px rgba(34,197,94,0.3);"></div>
            <div style="min-width: 120px;">
                <div><strong>{row['symbol']}</strong> <span style="color: #059669;">Sharpe {row['sharpe']:.2f}</span></div>
                <div style="font-size: 12px; color: var(--text-muted);">{row.get('profit', 0):.0f}€ | {row.get('trades', 0)} Trades</div>
            </div>
            </div>''' for row in (multiasset_details[:3] if multiasset_details else [])])}
        </div>

        <!-- Hit-Rate Badge -->
        <div class="badge" style="margin-top: 16px; background: var(--accent-soft); color: var(--accent); font-size: 14px;">
            📊 <strong>{len([d for d in multiasset_details if d.get('sharpe', 0) > 0])}/{len(multiasset_details)}</strong> = {hit_rate:.0f}% Hit-Rate
        </div>
        </div>
    </div>

    <!-- Decision Gate (DEIN DESIGN) -->
    <div class="card">
        <div class="card-header">Decision Gate Status<span>Aktuelle Bewertung</span></div>
        <div class="card-body">
        <div class="metric-value" style="font-size: 32px; color: var(--warning);">🟡 WARNUNG</div>
        <p style="margin-top: 12px; font-size: 14px;">
            <strong>Multi-Asset Hit-Rate:</strong> <span style="color: var(--danger);">{hit_rate:.1f}%</span> < 75.0%
        </p>
        <div style="margin-top: 16px; padding: 12px; background: rgba(254, 243, 199, 0.3); border-radius: 8px; border-left: 4px solid var(--warning);">
            <strong>→ Empfehlung:</strong> Live nur mit <strong>{top_symbols}</strong> starten
        </div>
        </div>
    </div>
    </div>
    """

    html_parts.append(multiasset_html)

    # ========================================================
    # Footer & Datei schreiben
    # ========================================================

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    footer_html = f"""
    <div class="footer">
      Generiert am {now_str} · Quant Validation Dashboard v5.0
    </div>
  </div> <!-- container -->
</body>
</html>
    """

    html_parts.append(footer_html)

    html_content = "\n".join(html_parts)

    out_path = strategy_dir / "quant_dashboard_PRO.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return out_path
