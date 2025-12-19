"""
Erzeugt einen ausführlichen HTML-Report für eine Strategie auf Basis von summary.json
und der im Strategie-Report-Ordner liegenden PNG-Plots.

Der Report ist für Laien verständlich gehalten (auf Deutsch) und erklärt alle
wichtigen Kennzahlen (Sharpe, Max Drawdown, Kelly, Monte-Carlo, Walk-Forward,
Multi-Asset, VIX-Regime, Decision Gate).

Die Datei wird im gleichen Ordner wie summary.json als report.html abgelegt.

Typische Nutzung innerhalb der Pipeline:
- reports/<strategy_stem>/
    - summary.json
    - equity.png
    - mc_returns.png
    - vix_regime_sharpe.png
    - walk_forward_sharpe.png
    - multi_asset_sharpe.png (optional)

Dieses Skript kann auch standalone aufgerufen werden:

(venv) python generate_html_report.py --strategy-dir reports/RangeBreakoutUSDJPY__v4_USDJPY_M15_20240101_20250925

"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def load_summary(summary_path: Path) -> Dict[str, Any]:
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataframes(summary: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    metrics = summary["metrics"]
    cost_results = summary["cost_results"]
    kelly = summary["kelly"]
    kelly_oos = summary.get("kelly_oos", None)
    vix_alignment = summary["vix_alignment"]
    mc_results = summary["mc_results"]
    wf_results = summary["walk_forward"]
    multi_asset = summary.get("multi_asset", None)
    gate_result = summary["gate_result"]

    # Basis-Metriken
    base_df = pd.DataFrame(
        {
            "Total Return": [f"{metrics['total_return']:.2%}"],
            "Sharpe": [f"{metrics['sharpe_ratio']:.2f}"],
            "MaxDD": [f"{metrics['max_drawdown']:.2%}"],
            "WinRate": [f"{metrics['win_rate']:.2%}"],
            "Trades": [metrics["total_trades"]],
        },
        index=["Gesamt"],
    )

    # Kosten-Szenarien
    cost_rows = []
    for name, m in cost_results.items():
        cost_rows.append(
            {
                "Szenario": name,
                "Total Return": f"{m['total_return']:.2%}",
                "Sharpe": f"{m['sharpe_ratio']:.2f}",
                "MaxDD": f"{m['max_drawdown']:.2%}",
                "Profit Factor": f"{m['profit_factor']:.2f}",
            }
        )
    cost_df = pd.DataFrame(cost_rows)

    # Kelly (Full Sample)
    kelly_df = pd.DataFrame(
        {
            "Trefferquote": [f"{kelly['win_rate']:.2%}"],
            "Payoff (Ø Gewinn / Ø Verlust)": [f"{kelly['payoff_ratio']:.2f}"],
            "Kelly (voll)": [f"{kelly['kelly_full']:.2%}"],
            "Kelly (halb)": [f"{kelly['kelly_half']:.2%}"],
            "Kelly (viertel)": [f"{kelly['kelly_quarter']:.2%}"],
        },
        index=["Gesamter Backtest"],
    )

    # Kelly (OOS, Walk-Forward)
    if kelly_oos:
        kelly_oos_df = pd.DataFrame(
            {
                "Trefferquote": [f"{kelly_oos['win_rate']:.2%}"],
                "Payoff (Ø Gewinn / Ø Verlust)": [f"{kelly_oos['payoff_ratio']:.2f}"],
                "Kelly (voll)": [f"{kelly_oos['kelly_full']:.2%}"],
                "Kelly (halb)": [f"{kelly_oos['kelly_half']:.2%}"],
                "Kelly (viertel)": [f"{kelly_oos['kelly_quarter']:.2%}"],
            },
            index=["Nur Walk-Forward-OOS"],
        )
    else:
        kelly_oos_df = pd.DataFrame()

    # VIX-Regime
    vix_rows = []
    for r_name, m in vix_alignment["regime_stats"].items():
        vix_rows.append(
            {
                "Regime": r_name,
                "Trades": m["n_trades"],
                "Total Return": f"{m['total_return']:.2%}",
                "Sharpe": f"{m['sharpe_ratio']:.2f}",
                "MaxDD": f"{m['max_drawdown']:.2%}",
                "Profit Factor": f"{m['profit_factor']:.2f}",
            }
        )
    vix_df = pd.DataFrame(vix_rows)

    # Monte Carlo (Summary)
    mc_df = pd.DataFrame(
        {
            "Wahrscheinlichkeit für positives Ergebnis": [
                f"{mc_results['mc_positive_prob']:.2%}"
            ],
            "Median-Gesamtrendite": [f"{mc_results['mc_median_return']:.2%}"],
            "5%-Quantil Rendite": [f"{mc_results['mc_p5_return']:.2%}"],
            "95%-Quantil Rendite": [f"{mc_results['mc_p95_return']:.2%}"],
            "Median MaxDD": [f"{mc_results['mc_median_max_dd']:.2%}"],
            "95%-Quantil MaxDD": [f"{mc_results['mc_p95_max_dd']:.2%}"],
        },
        index=["Monte Carlo"],
    )

    # Walk-Forward Fenster
    wf_rows = []
    for w in wf_results["window_metrics"]:
        wf_rows.append(
            {
                "Fenster": w["window_id"],
                "Train": f"{w['train_start']} → {w['train_end']}",
                "Test": f"{w['test_start']} → {w['test_end']}",
                "Trades": w["test_n_trades"],
                "Sharpe": f"{w['test_sharpe']:.2f}",
                "Profit Factor": f"{w['test_profit_factor']:.2f}",
                "MaxDD": f"{w['test_max_dd']:.2%}",
                "Return": f"{w['test_total_return']:.2%}",
                "Kelly (voll)": f"{w.get('test_kelly_full', 0.0):.2%}",
                "Kelly (halb)": f"{w.get('test_kelly_half', 0.0):.2%}",
                "Kelly (viertel)": f"{w.get('test_kelly_quarter', 0.0):.2%}",
            }
        )
    wf_df = pd.DataFrame(wf_rows)

    # Multi-Asset (Symbol-Stats)
    if multi_asset and multi_asset.get("details"):
        ma_df = pd.DataFrame(multi_asset["details"])
        ma_df = ma_df.sort_values("sharpe", ascending=False)
        ma_df = ma_df[
            ["symbol", "sharpe", "profit", "profit_factor", "equity_dd_pct", "trades"]
        ].rename(
            columns={
                "symbol": "Symbol",
                "sharpe": "Sharpe",
                "profit": "Profit",
                "profit_factor": "Profit Factor",
                "equity_dd_pct": "MaxDD %",
                "trades": "Trades",
            }
        )
        # Prozentformat
        ma_df["MaxDD %"] = ma_df["MaxDD %"].map(lambda x: f"{x:.2f}%")
    else:
        ma_df = pd.DataFrame()

    # Gate-Result
    gate_df = pd.DataFrame(
        {
            "Status": [gate_result["status"]],
            "Confidence": [f"{gate_result['confidence']:.2%}"],
            "Reason": [gate_result["reason"]],
        }
    )

    return {
        "base_df": base_df,
        "cost_df": cost_df,
        "kelly_df": kelly_df,
        "kelly_oos_df": kelly_oos_df,
        "vix_df": vix_df,
        "mc_df": mc_df,
        "wf_df": wf_df,
        "ma_df": ma_df,
        "gate_df": gate_df,
    }


def df_to_html(df: pd.DataFrame, index: bool = True) -> str:
    if df.empty:
        return "<p>(Keine Daten verfügbar)</p>"
    return df.to_html(
        classes="table table-sm table-striped table-bordered",
        border=0,
        index=index,
        escape=False,
    )


def render_html_for_strategy_dir(strategy_dir: Path) -> Path:
    """
    Erzeugt report.html im angegebenen Strategie-Report-Ordner.
    Erwartet:
      - summary.json
      - equity.png
      - mc_returns.png
      - vix_regime_sharpe.png
      - walk_forward_sharpe.png
      - multi_asset_sharpe.png (optional)
    """
    summary_path = strategy_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json nicht gefunden in {strategy_dir}")

    summary = load_summary(summary_path)
    dfs = build_dataframes(summary)

    metrics = summary["metrics"]
    gate = summary["gate_result"]
    multi_asset = summary.get("multi_asset", None)
    wf_results = summary["walk_forward"]

    strategy_name = metrics["strategy_name"]
    date_from, date_to = metrics["date_range"]

    # Basis-Tabellen
    base_html = df_to_html(dfs["base_df"])
    cost_html = df_to_html(dfs["cost_df"], index=False)
    kelly_html = df_to_html(dfs["kelly_df"])
    kelly_oos_html = df_to_html(dfs["kelly_oos_df"])
    vix_html = df_to_html(dfs["vix_df"], index=False)
    mc_html = df_to_html(dfs["mc_df"])
    wf_html = df_to_html(dfs["wf_df"], index=False)
    ma_html = df_to_html(dfs["ma_df"], index=False)
    gate_html = df_to_html(dfs["gate_df"], index=False)

    # Plot-Dateien (relativ zum HTML)
    equity_png = "equity.png"
    mc_png = "mc_returns.png"
    vix_png = "vix_regime_sharpe.png"
    wf_png = "walk_forward_sharpe.png"
    ma_png = "multi_asset_sharpe.png"

    # Basis-Interpretationen
    sharpe = metrics["sharpe_ratio"]
    maxdd = metrics["max_drawdown"]
    oos_sharpe = wf_results["oos_sharpe"]
    oos_maxdd = wf_results["oos_max_dd"]
    mc_positive = summary["mc_results"]["mc_positive_prob"]
    ma_hit = (
        multi_asset["hit_rate"] if multi_asset and "hit_rate" in multi_asset else None
    )

    # Einfache „Ampel“-Interpretationen
    def sharpe_kommentar(x: float) -> str:
        if x >= 3:
            return "außergewöhnlich stark (Seltenheit, oft nur mit Overfitting erreichbar)"
        if x >= 2:
            return "sehr gut (professioneller Bereich, meist live nutzbar)"
        if x >= 1.5:
            return "gut (solide Strategie, aber auf weitere Risiken achten)"
        if x >= 1:
            return "ok, aber eher mittelmäßig – Verbesserungspotenzial"
        return "schwach – für Live-Handel meist nicht geeignet"

    def dd_kommentar(x: float) -> str:
        if x <= 0.05:
            return "sehr geringe Drawdowns (psychologisch gut handelbar)"
        if x <= 0.1:
            return "moderate Drawdowns (für viele Trader noch akzeptabel)"
        if x <= 0.2:
            return "hohe Drawdowns – nur mit kleiner Positionsgröße sinnvoll"
        return "extrem hohe Drawdowns – Risiko für Kontocrash"

    def mc_kommentar(p: float) -> str:
        if p >= 0.99:
            return "nahezu jede simulierte Zukunft war profitabel – sehr robustes Profil"
        if p >= 0.9:
            return "hohe Wahrscheinlichkeit für Profit – robust, aber nicht unzerstörbar"
        if p >= 0.7:
            return "ok, aber es gibt signifikante Szenarien mit Verlust"
        return "viele Szenarien enden im Minus – vorsichtig sein"

    gate_status = gate["status"]
    if gate_status.upper().startswith("LIVE"):
        live_text = "Ja, alle definierten Validierungskriterien wurden erfüllt."
    else:
        live_text = "Nein, mindestens ein Validierungskriterium wurde verletzt."

    title = f"Quant Validation Report – {strategy_name}"

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>{title}</title>
<link
  rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
/>
<style>
body {{ padding: 20px; }}
h1, h2, h3 {{ margin-top: 1.5rem; }}
.table-sm td, .table-sm th {{ padding: .3rem; }}
img.plot {{ max-width: 100%; height: auto; border: 1px solid #ccc; margin-bottom: 10px; }}
.explainer {{ font-size: 0.95rem; }}
.badge-metric {{ font-size: 1rem; }}
</style>
</head>
<body>
<div class="container-fluid">
  <h1>{title}</h1>
  <p><strong>Zeitraum:</strong> {date_from} – {date_to}</p>

  <h2>Live-Tauglichkeit (Decision Gate)</h2>
  <div class="row">
    <div class="col-md-6">
      {gate_html}
    </div>
    <div class="col-md-6 explainer">
      <p><strong>Interpretation:</strong> Das Decision Gate fasst mehrere Kennzahlen
      zu einer einfachen Ja/Nein-Entscheidung für den Live-Handel zusammen
      (z.&nbsp;B. minimale Sharpe im Backtest, maximale zulässige Drawdowns,
      Monte-Carlo-Robustheit, Korrelation zu MT5-Backtest).</p>
      <p><strong>Live-Eignung:</strong> {live_text}</p>
    </div>
  </div>

  <h2>1. Gesamt-Performance und Kennzahlen</h2>
  <div class="row">
    <div class="col-md-6">
      <h3>Equity-Kurve</h3>
      <img src="{equity_png}" alt="Equity Curve" class="plot" />
    </div>
    <div class="col-md-6 explainer">
      {base_html}
      <p><strong>Sharpe-Ratio:</strong> {sharpe:.2f} – {sharpe_kommentar(sharpe)}</p>
      <p><strong>Maximaler Drawdown:</strong> {maxdd:.2%} – {dd_kommentar(maxdd)}</p>
      <p>Die Equity-Kurve zeigt den Verlauf des Kontostands über alle Trades. Eine
      stetig steigende Kurve mit kleinen Rücksetzern ist ideal. Große Einbrüche
      (Drawdowns) sind psychologisch schwer zu handeln und erhöhen das Risiko,
      dass man die Strategie in der schlechtesten Phase abschaltet.</p>
    </div>
  </div>

  <h2>2. Kosten- und Slippage-Robustheit</h2>
  <div class="row">
    <div class="col-md-7">
      {cost_html}
    </div>
    <div class="col-md-5 explainer">
      <p>Diese Tabelle zeigt, wie empfindlich die Strategie auf höhere Kosten
      (Spreads, Kommissionen, Slippage) reagiert.</p>
      <ul>
        <li><strong>base:</strong> aktuelle angenommene Kosten.</li>
        <li><strong>cost_plus_25:</strong> Kosten um 25&nbsp;% erhöht
            (z.&nbsp;B. schlechtere Ausführung).</li>
        <li><strong>cost_plus_50:</strong> Kosten um 50&nbsp;% erhöht.</li>
      </ul>
      <p>Wenn Sharpe und Total Return bei höheren Kosten kaum fallen, ist die
      Strategie robust gegenüber schlechteren Marktbedingungen und Brokern.</p>
    </div>
  </div>

  <h2>3. Kelly-Kriterium und Positionsgrößen</h2>
  <div class="row">
    <div class="col-md-6">
      <h3>Kelly auf gesamtem Backtest</h3>
      {kelly_html}
    </div>
    <div class="col-md-6">
      <h3>Kelly nur auf Out-of-Sample (Walk-Forward)</h3>
      {kelly_oos_html}
    </div>
  </div>
  <div class="row">
    <div class="col explainer">
      <p>Das Kelly-Kriterium berechnet die theoretisch optimale Risikofraktion
      vom Kontostand pro Trade, basierend auf Trefferquote und durchschnittlichem
      Gewinn-Verlust-Verhältnis.</p>
      <p><strong>Wichtige Regeln:</strong></p>
      <ul>
        <li>Voller Kelly-Wert ist in der Praxis meist zu aggressiv.</li>
        <li>Häufig wird mit <strong>Kelly / 2</strong> oder <strong>Kelly / 4</strong>
            gearbeitet, um Drawdowns zu begrenzen.</li>
        <li>Die OOS-Kelly-Werte (nur Walk-Forward-Testfenster) sind meist realistischer
            als die Werte aus dem gesamten Backtest.</li>
      </ul>
    </div>
  </div>

  <h2>4. Monte-Carlo-Simulation</h2>
  <div class="row">
    <div class="col-md-6">
      <img src="{mc_png}" alt="Monte Carlo Returns" class="plot" />
    </div>
    <div class="col-md-6">
      {mc_html}
      <p class="explainer">
        Die Monte-Carlo-Simulation mischt die Trade-Reihenfolge zufällig durch und
        erstellt viele mögliche Zukunftsverläufe. Dadurch wird sichtbar, wie
        empfindlich die Strategie gegenüber Pechsträhnen ist.
      </p>
      <p class="explainer">
        <strong>mc_positive_prob:</strong> Anteil der Simulationen, die mit Gewinn
        enden. {mc_kommentar(mc_positive)}
      </p>
    </div>
  </div>

  <h2>5. Walk-Forward-Analyse (Out-of-Sample Stabilität)</h2>
  <div class="row">
    <div class="col-md-6">
      <img src="{wf_png}" alt="Walk-Forward Sharpe" class="plot" />
    </div>
    <div class="col-md-6 explainer">
      <p><strong>Durchschnittliche OOS-Sharpe:</strong> {oos_sharpe:.2f} –
      {sharpe_kommentar(oos_sharpe)}</p>
      <p><strong>Durchschnittlicher OOS-MaxDD:</strong> {oos_maxdd:.2%} –
      {dd_kommentar(oos_maxdd)}</p>
      <p>Die Walk-Forward-Analyse teilt die Historie in mehrere Zeitfenster:
      In jedem Fenster wird auf einem Teil der Daten trainiert und auf einem
      nachfolgenden Teil getestet (Out-of-Sample). So sieht man, ob die
      Strategie auch in unbekannten Daten stabil funktioniert.</p>
    </div>
  </div>
  <div class="row">
    <div class="col">
      <h3>Details pro Walk-Forward-Fenster</h3>
      {wf_html}
    </div>
  </div>

  <h2>6. VIX-Regime & Marktumgebung</h2>
  <div class="row">
    <div class="col-md-6">
      <img src="{vix_png}" alt="VIX Regime Sharpe" class="plot" />
    </div>
    <div class="col-md-6 explainer">
      <p>Die VIX-Regime-Analyse zeigt, wie die Strategie in verschiedenen
      Marktphasen (niedrige Volatilität, hohe Volatilität, Crash) abschneidet.</p>
      <p>Eine Strategie, die nur in einem Regime funktioniert (z.&nbsp;B. nur
      in ruhigen Märkten), ist anfälliger für plötzliche Umschwünge als eine
      Strategie, die in mehreren Regimen solide performt.</p>
    </div>
  </div>
  <div class="row">
    <div class="col">
      {vix_html}
    </div>
  </div>

  <h2>7. Multi-Asset-Robustheit</h2>
  <div class="row">
    <div class="col-md-6">
      <img src="{ma_png}" alt="Multi-Asset Sharpe" class="plot" />
    </div>
    <div class="col-md-6 explainer">
      <p>Die Multi-Asset-Analyse zeigt, auf wie vielen anderen Märkten das
      gleiche Regelwerk (oder ähnliche Parameter) ebenfalls eine akzeptable
      Performance erzielt.</p>
      <p>
        Ein hoher Anteil von Märkten mit Sharpe &gt; 1,0 deutet auf ein
        robustes Grundprinzip hin, das nicht nur auf einen speziellen Markt
        „hingefittet“ wurde.
      </p>
      <p>
        In dieser Auswertung liegt die Multi-Asset-Hit-Rate bei
        {f"{ma_hit:.1%}" if ma_hit is not None else "n/a"}.
      </p>
    </div>
  </div>
  <div class="row">
    <div class="col">
      {ma_html}
    </div>
  </div>

  <h2>8. Zusammenfassung für Laien</h2>
  <div class="row">
    <div class="col explainer">
      <p>Dieser Report fasst zusammen, ob die Strategie aus Sicht eines
      professionellen Quant-Traders für den Live-Handel geeignet ist.</p>
      <ul>
        <li><strong>Sharpe-Ratio:</strong> misst das Verhältnis von Rendite zu
            Schwankung. Höher ist besser; Werte über 2 gelten als sehr gut.</li>
        <li><strong>Maximaler Drawdown:</strong> größter Rückgang vom Hoch zum
            Tief. Kleine Werte bedeuten weniger Stress und geringeres
            „Abschalt-Risiko“.</li>
        <li><strong>Kelly:</strong> liefert eine theoretische Empfehlung für die
            Positionsgröße. In der Praxis nutzt man oft nur einen Bruchteil davon.</li>
        <li><strong>Monte-Carlo:</strong> prüft, ob die Strategie auch bei
            zufälligen Trade-Reihenfolgen robust bleibt.</li>
        <li><strong>Walk-Forward:</strong> simuliert echte Live-Bedingungen, indem
            nur auf bisher unbekannten Daten getestet wird.</li>
        <li><strong>VIX-Regime:</strong> zeigt, in welchen Marktphasen die
            Strategie funktioniert oder Schwierigkeiten hat.</li>
        <li><strong>Multi-Asset:</strong> prüft, ob das Konzept auf anderen Märkten
            ebenfalls trägt.</li>
      </ul>
      <p>Erst wenn mehrere dieser Blöcke gleichzeitig positiv ausfallen und das
      Decision Gate „LIVE_ELIGIBLE“ meldet, sollte eine Strategie ernsthaft
      für den Live-Handel in Betracht gezogen werden.</p>
    </div>
  </div>

  <hr />
  <p class="text-muted">Report-Verzeichnis: {strategy_dir}</p>

</div>
</body>
</html>
"""

    output_path = strategy_dir / "report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Erzeuge einen ausführlichen HTML-Report für eine Strategie."
    )
    parser.add_argument(
        "--strategy-dir",
        type=str,
        required=True,
        help="Pfad zum Strategie-Report-Ordner (z.B. reports/RangeBreakoutUSDJPY_v4_...).",
    )
    args = parser.parse_args()
    strategy_dir = Path(args.strategy_dir)

    output = render_html_for_strategy_dir(strategy_dir)
    print(f"HTML-Report geschrieben nach: {output}")


if __name__ == "__main__":
    main()
