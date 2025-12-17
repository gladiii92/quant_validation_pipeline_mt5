"""
Erzeugt einen HTML-Report aus der Summary-JSON und den vorhandenen Plots.

Nutzung:
    python generate_html_report.py --summary reports/RangeBreakOut_USDJPY_trades_merged_summary.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_summary(summary_path: str):
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataframes(summary: dict):
    metrics = summary["metrics"]
    cost_results = summary["cost_results"]
    kelly = summary["kelly"]
    vix_alignment = summary["vix_alignment"]
    mc_results = summary["mc_results"]
    gate_result = summary["gate_result"]

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
            "win_rate": [f"{kelly['win_rate']:.2%}"],
            "payoff": [f"{kelly['payoff_ratio']:.2f}"],
            "full": [f"{kelly['kelly_full']:.2%}"],
            "half": [f"{kelly['kelly_half']:.2%}"],
            "quarter": [f"{kelly['kelly_quarter']:.2%}"],
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

    mc_df = pd.DataFrame(
        {
            "mc_positive_prob": [f"{mc_results['mc_positive_prob']:.2%}"],
            "mc_median_return": [f"{mc_results['mc_median_return']:.2%}"],
            "mc_p5_return": [f"{mc_results['mc_p5_return']:.2%}"],
            "mc_p95_return": [f"{mc_results['mc_p95_return']:.2%}"],
            "mc_median_max_dd": [f"{mc_results['mc_median_max_dd']:.2%}"],
            "mc_p95_max_dd": [f"{mc_results['mc_p95_max_dd']:.2%}"],
        },
        index=["Monte Carlo"],
    )

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
        "vix_df": vix_df,
        "mc_df": mc_df,
        "gate_df": gate_df,
    }


def render_html(
    summary_path: str,
    png_prefix: str,
    output_html: str,
):
    summary = load_summary(summary_path)
    dfs = build_dataframes(summary)

    base_html = dfs["base_df"].to_html(classes="table table-sm", border=0)
    cost_html = dfs["cost_df"].to_html(classes="table table-sm", index=False, border=0)
    kelly_html = dfs["kelly_df"].to_html(classes="table table-sm", border=0)
    vix_html = dfs["vix_df"].to_html(classes="table table-sm", index=False, border=0)
    mc_html = dfs["mc_df"].to_html(classes="table table-sm", border=0)
    gate_html = dfs["gate_df"].to_html(classes="table table-sm", index=False, border=0)

    # Plot-Dateien (relativ zum HTML)
    equity_png = f"{png_prefix}_equity.png"
    mc_png = f"{png_prefix}_mc_returns.png"
    vix_png = f"{png_prefix}_vix_regime_sharpe.png"

    title = f"Quant Validation Report – {summary['metrics']['strategy_name']}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 20px;
    background-color: #111;
    color: #eee;
}}
h1, h2 {{
    color: #ffd700;
}}
.table {{
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 20px;
    font-size: 13px;
}}
.table th, .table td {{
    border: 1px solid #444;
    padding: 4px 6px;
}}
.table tr:nth-child(even) {{
    background-color: #222;
}}
.section {{
    margin-bottom: 30px;
}}
img {{
    max-width: 100%;
    border: 1px solid #444;
    margin-bottom: 15px;
}}
</style>
</head>
<body>

<h1>{title}</h1>
<p>Date Range: {summary["metrics"]["date_range"][0]} – {summary["metrics"]["date_range"][1]}</p>

<div class="section">
  <h2>Core Metrics</h2>
  {base_html}
</div>

<div class="section">
  <h2>Cost / Slippage Scenarios</h2>
  {cost_html}
</div>

<div class="section">
  <h2>Kelly Sizing</h2>
  {kelly_html}
</div>

<div class="section">
  <h2>VIX Regime Performance</h2>
  {vix_html}
</div>

<div class="section">
  <h2>Monte Carlo Summary</h2>
  {mc_html}
</div>

<div class="section">
  <h2>Decision Gate</h2>
  {gate_html}
</div>

<div class="section">
  <h2>Plots</h2>
  <h3>Equity Curve</h3>
  <img src="{equity_png}" alt="Equity Curve">
  <h3>Monte Carlo Returns</h3>
  <img src="{mc_png}" alt="Monte Carlo Returns">
  <h3>Sharpe by VIX Regime</h3>
  <img src="{vix_png}" alt="Sharpe by VIX Regime">
</div>

</body>
</html>
"""

    out_path = Path(output_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"HTML report written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from summary JSON")
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to *_summary.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML path (optional)",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary JSON not found: {summary_path}")

    # Prefix aus Dateinamen ableiten: reports/RangeBreakOut_USDJPY_trades_merged_summary.json
    stem = summary_path.stem.replace("_summary", "")
    png_prefix = str(summary_path.parent / stem)

    if args.output is None:
        output_html = str(summary_path.parent / f"{stem}_report.html")
    else:
        output_html = args.output

    render_html(
        summary_path=str(summary_path),
        png_prefix=png_prefix,
        output_html=output_html,
    )


if __name__ == "__main__":
    main()
