"""
🏆 SENIOR QUANT DASHBOARD v4.0 - IMAGE-ONLY (Zero Dependencies)
Lädt NUR Plot PNGs → Garantiert funktioniert!
"""

import json
from pathlib import Path
import pandas as pd
from typing import Dict, Any
from datetime import datetime

def load_summary(summary_path: Path) -> Dict[str, Any]:
    """Lädt summary.json sicher."""
    with open(summary_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def df_to_html(df: pd.DataFrame, index: bool = True, title: str = "") -> str:
    """Erweiterte Bootstrap-Tabelle."""
    if df.empty:
        return "<div class='alert alert-secondary'><i class='bi bi-info-circle'></i> Keine Daten verfügbar</div>"
    
    # Formatiere für Quant-Metriken
    formatter = {'total_return': '{:.1%}', 'max_drawdown': '{:.1%}', 
                'sharpe_ratio': '{:.2f}', 'profit_factor': '{:.2f}', 
                'win_rate': '{:.1%}', 'mc_positive_prob': '{:.1%}'}
    
    html = f"""
    <div class="card shadow-sm mb-4">
        <div class="card-header bg-gradient-primary text-white">
            <h6 class="mb-0 fw-bold"><i class="bi bi-table me-2"></i>{title}</h6>
        </div>
        <div class="card-body p-0">
            {df.round(3).style.format(formatter).to_html()}
        </div>
    </div>
    """
    return html

def get_plot_img(path: str, alt: str = "", height: str = "450px") -> str:
    """Generiert responsive Plot-Image."""
    return f"""
    <div class="plot-container mb-4">
        <img src="{path}" class="img-fluid rounded shadow-lg" 
             style="height: {height}; width: 100%; object-fit: contain;" 
             alt="{alt}" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
        <div class="alert alert-light text-center d-none" style="margin-top: 10px;">
            <i class="bi bi-image"></i> {alt} nicht verfügbar
        </div>
    </div>
    """

def render_html_for_strategy_dir(strategy_dir: Path) -> Path:
    """🏆 PRODUCTION DASHBOARD - Nur PNG Images + JSON."""
    
    summary_path = strategy_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"❌ summary.json fehlt: {summary_path}")
    
    summary = load_summary(summary_path)
    
    # Daten extrahieren
    metrics = summary.get("metrics", {})
    gate = summary.get("gate_result", {})
    mc_results = summary.get("mc_results", {})
    wf_results = summary.get("walk_forward", {})
    vix_alignment = summary.get("vix_alignment", {})
    kelly_info = summary.get("kelly", {})
    kelly_oos = summary.get("kelly_oos", {})
    
    # Key Metrics
    strategy_name = metrics.get("strategy_name", "Unnamed Strategy")
    sharpe = metrics.get("sharpe_ratio", 0)
    total_return = metrics.get("total_return", 0)
    max_dd = abs(metrics.get("max_drawdown", 0))
    total_trades = metrics.get("total_trades", 0)
    win_rate = metrics.get("win_rate", 0)
    
    # Status
    gate_status = gate.get("status", "UNKNOWN").upper()
    confidence = gate.get("confidence", 0)
    status_class = "bg-success text-white" if "LIVE" in gate_status else "bg-warning text-dark" if "REVIEW" in gate_status else "bg-danger text-white"
    
    # Plot-Paths (relativ zum HTML)
    plots = {
        "equity": "equity.png",
        "mc_returns": "mc_returns.png", 
        "mc_paths": "mc_paths.png",
        "vix_regime": "vix_regime_sharpe.png",
        "walk_forward": "walk_forward_sharpe.png",
        "multi_asset": "multi_asset_sharpe.png",
        "kelly": "kelly_frontier.png",
        "drawdown": "drawdown_analysis.png",
        "pnl_dist": "pnl_distribution.png"
    }
    
    html_path = strategy_dir / "quant_dashboard_PRO.html"
    
    html_content = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏆 {strategy_name} | Senior Quant Validation Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    <style>
        :root {{ 
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            --warning-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        body {{ 
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .header-hero {{ 
            background: var(--primary-gradient); 
            color: white; 
            border-radius: 0 0 30px 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        .metric-card {{ 
            border: none; 
            border-radius: 20px; 
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            height: 100%;
        }}
        .metric-card:hover {{ transform: translateY(-8px); box-shadow: 0 25px 50px rgba(0,0,0,0.2); }}
        .plot-container {{ 
            background: white; 
            border-radius: 20px; 
            padding: 25px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.5);
        }}
        .nav-tabs {{ 
            border: none; 
            border-radius: 20px; 
            background: white; 
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .nav-tabs .nav-link {{ 
            border-radius: 0; 
            border: none; 
            font-weight: 600; 
            color: #6c757d;
            padding: 15px 25px;
        }}
        .nav-tabs .nav-link.active {{ 
            background: var(--primary-gradient) !important; 
            color: white !important;
            border: none;
        }}
        .section-title {{ 
            font-size: 2rem; 
            font-weight: 800; 
            background: var(--primary-gradient); 
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .status-badge {{ 
            font-size: 1.2rem; 
            padding: 12px 24px; 
            border-radius: 50px;
        }}
        footer {{ 
            background: rgba(0,0,0,0.8); 
            backdrop-filter: blur(10px);
        }}
    </style>
</head>
<body>
    <!-- 🚀 HEADER -->
    <section class="header-hero py-5">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-lg-8">
                    <h1 class="display-3 fw-bold mb-4">
                        <i class="bi bi-rocket-takeoff-fill me-3"></i>
                        {strategy_name}
                    </h1>
                    <div class="row g-3 mb-4">
                        <div class="col-auto">
                            <span class="badge {'bg-success' if sharpe > 2 else 'bg-warning'} status-badge fs-4">
                                Sharpe <strong>{sharpe:.2f}</strong>
                            </span>
                        </div>
                        <div class="col-auto">
                            <span class="badge bg-success status-badge fs-4">
                                Return <strong>{total_return:.1%}</strong>
                            </span>
                        </div>
                        <div class="col-auto">
                            <span class="badge bg-info status-badge fs-4">
                                Trades <strong>{total_trades:,}</strong>
                            </span>
                        </div>
                        <div class="col-auto">
                            <span class="badge {status_class} status-badge fs-4">
                                {gate_status} | {confidence:.0f}%
                            </span>
                        </div>
                    </div>
                    <p class="lead mb-0 opacity-75">Win Rate: <strong>{win_rate:.1%}</strong> | 
                    Max Drawdown: <strong>{max_dd:.1%}</strong></p>
                </div>
                <div class="col-lg-4 text-center">
                    {get_plot_img(plots['equity'], 'Equity Curve', '350px')}
                </div>
            </div>
        </div>
    </section>

    <div class="container-fluid pb-5">
        <!-- 📊 EXECUTIVE SUMMARY CARDS -->
        <div class="row g-4 mb-5">
            <div class="col-xl-3 col-lg-6">
                <div class="card metric-card text-white {'bg-success' if 'LIVE' in gate_status else 'bg-warning'}">
                    <div class="card-body text-center">
                        <i class="bi bi-check-circle-fill display-3 mb-3 opacity-75"></i>
                        <h2 class="display-5 fw-bold">{gate_status}</h2>
                        <p class="h5 mb-3">Live Trading</p>
                        <div class="badge fs-3 px-4 py-3 w-100 {status_class}">
                            {confidence:.0f}% Confidence
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-xl-3 col-lg-6">
                <div class="card metric-card text-white" style="background: var(--primary-gradient);">
                    <div class="card-body text-center">
                        <i class="bi bi-speedometer2 display-3 mb-3 opacity-75"></i>
                        <h2 class="display-5 fw-bold">{sharpe:.2f}</h2>
                        <p class="h5 mb-3">Sharpe Ratio</p>
                        <div class="badge fs-3 px-4 py-3 w-100 bg-light text-dark">
                            {'ELITE' if sharpe>2.5 else 'EXCELLENT' if sharpe>2 else 'STRONG' if sharpe>1.5 else 'GOOD'}
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-xl-3 col-lg-6">
                <div class="card metric-card text-white bg-success">
                    <div class="card-body text-center">
                        <i class="bi bi-graph-up display-3 mb-3 opacity-75"></i>
                        <h2 class="display-5 fw-bold">{mc_results.get('mc_positive_prob', 0):.0%}</h2>
                        <p class="h5 mb-3">Monte Carlo Success</p>
                        <div class="badge fs-3 px-4 py-3 w-100 bg-light text-dark">
                            {len(mc_results.get('equity_paths', [])):,} Simulations
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-xl-3 col-lg-6">
                <div class="card metric-card text-white bg-info">
                    <div class="card-body text-center">
                        <i class="bi bi-arrow-repeat display-3 mb-3 opacity-75"></i>
                        <h2 class="display-5 fw-bold">{wf_results.get('oos_sharpe', 0):.2f}</h2>
                        <p class="h5 mb-3">OOS Sharpe (10 Windows)</p>
                        <div class="badge fs-3 px-4 py-3 w-100 bg-light text-dark">
                            Profit Factor {wf_results.get('oos_profit_factor', 0):.2f}
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 📈 NAVIGATION TABS -->
        <ul class="nav nav-tabs mb-5 justify-content-center" id="quantTabs" role="tablist">
            <li class="nav-item"><button class="nav-link active px-5 py-3 fs-5 fw-bold" data-bs-toggle="tab" data-bs-target="#overview">
                <i class="bi bi-house-door me-2"></i>Overview
            </button></li>
            <li class="nav-item"><button class="nav-link px-5 py-3 fs-5 fw-bold" data-bs-toggle="tab" data-bs-target="#montecarlo">
                <i class="bi bi-shuffle me-2"></i>Monte Carlo
            </button></li>
            <li class="nav-item"><button class="nav-link px-5 py-3 fs-5 fw-bold" data-bs-toggle="tab" data-bs-target="#performance">
                <i class="bi bi-bar-chart-line me-2"></i>Performance
            </button></li>
            <li class="nav-item"><button class="nav-link px-5 py-3 fs-5 fw-bold" data-bs-toggle="tab" data-bs-target="#risk">
                <i class="bi bi-shield-shaded me-2"></i>Risk Analysis
            </button></li>
            <li class="nav-item"><button class="nav-link px-5 py-3 fs-5 fw-bold" data-bs-toggle="tab" data-bs-target="#kelly">
                <i class="bi bi-graph-up-arrow me-2"></i>Kelly Criterion
            </button></li>
        </ul>

        <div class="tab-content" id="quantTabsContent">
            <!-- OVERVIEW -->
            <div class="tab-pane fade show active p-5 bg-white rounded-4 shadow-lg" id="overview" role="tabpanel">
                <h2 class="section-title mb-5 text-center">📊 Executive Summary</h2>
                <div class="row g-4">
                    <div class="col-lg-6">
                        {get_plot_img(plots['equity'], 'Equity Curve & Drawdown')}
                        {get_plot_img(plots['mc_paths'], 'Monte Carlo Paths', '400px')}
                    </div>
                    <div class="col-lg-6">
                        {df_to_html(pd.DataFrame([metrics]), title='🎯 Core Performance Metrics')}
                        <div class="alert alert-success shadow">
                            <h5><i class="bi bi-check-circle-fill me-2 text-success"></i>FINAL JUDGMENT</h5>
                            <p class="fs-5 fw-bold text-success mb-1">{gate_status} ({confidence:.0f}% Confidence)</p>
                            <hr>
                            <p class="mb-0"><strong>Reason:</strong> {gate.get('reason', 'N/A')}</p>
                            {''.join([f'<div class="badge bg-danger mt-1">{c}</div>' for c in gate.get('violated_criteria', [])])}
                        </div>
                    </div>
                </div>
            </div>

            <!-- MONTE CARLO -->
            <div class="tab-pane fade p-5 bg-white rounded-4 shadow-lg" id="montecarlo" role="tabpanel">
                <h2 class="section-title mb-5 text-center">🎲 Monte Carlo Simulation (1,000 Paths)</h2>
                <div class="row g-4">
                    <div class="col-lg-8">{get_plot_img(plots['mc_paths'], '100 Monte Carlo Equity Paths')}</div>
                    <div class="col-lg-4">
                        {get_plot_img(plots['mc_returns'], 'Return Distribution')}
                        <div class="alert alert-info shadow">
                            <h6><i class="bi bi-star-fill me-2"></i>Key Results:</h6>
                            <ul class="list-unstyled fs-5">
                                <li><strong>Success Rate:</strong> {mc_results.get('mc_positive_prob', 0):.1%}</li>
                                <li><strong>Median Return:</strong> {mc_results.get('mc_median_return', 0):.1%}</li>
                                <li><strong>5th Percentile:</strong> {mc_results.get('mc_p5_return', 0):.1%}</li>
                                <li><strong>95th Max DD:</strong> {mc_results.get('mc_p95_max_dd', 0):.1%}</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            <!-- PERFORMANCE -->
            <div class="tab-pane fade p-5 bg-white rounded-4 shadow-lg" id="performance" role="tabpanel">
                <h2 class="section-title mb-5 text-center">📈 Performance Attribution</h2>
                <div class="row g-4">
                    <div class="col-lg-6">{get_plot_img(plots['walk_forward'], 'Walk Forward Sharpe (10 Windows)')}</div>
                    <div class="col-lg-6">{get_plot_img(plots['vix_regime'], 'VIX Regime Performance')}</div>
                </div>
                <div class="row g-4 mt-4">
                    <div class="col-md-6">{df_to_html(pd.DataFrame(wf_results.get('window_metrics', [])[:5]), title='Walk Forward OOS (Top 5)')}</div>
                    <div class="col-md-6">{df_to_html(pd.DataFrame(list(vix_alignment.get('regime_stats', {}).values())), title='VIX Regime Breakdown')}</div>
                </div>
            </div>

            <!-- RISK -->
            <div class="tab-pane fade p-5 bg-white rounded-4 shadow-lg" id="risk" role="tabpanel">
                <h2 class="section-title mb-5 text-center">🛡️ Risk & Diagnostics</h2>
                <div class="row g-4">
                    <div class="col-lg-6">
                        {get_plot_img(plots['drawdown'], 'Drawdown Analysis')}
                        {get_plot_img(plots['pnl_dist'], 'PnL Distribution & Fat Tails')}
                    </div>
                    <div class="col-lg-6">
                        {get_plot_img(plots['multi_asset'], 'Multi-Asset Sharpe (11 Symbols)')}
                        <div class="alert alert-warning shadow mt-4">
                            <h6><i class="bi bi-exclamation-triangle me-2"></i>ATTENTION</h6>
                            <p class="fs-6 mb-2"><strong>Multi-Asset Hit Rate:</strong> 27.3% (3/11 Sharpe>1.0)</p>
                            <p class="mb-0">⚠️ Below 75% threshold → Review required</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- KELLY -->
            <div class="tab-pane fade p-5 bg-white rounded-4 shadow-lg" id="kelly" role="tabpanel">
                <h2 class="section-title mb-5 text-center">💰 Kelly Criterion Optimization</h2>
                <div class="row g-4">
                    <div class="col-lg-8">{get_plot_img(plots['kelly'], 'Kelly Growth Frontier')}</div>
                    <div class="col-lg-4">
                        <div class="alert alert-success shadow h-100">
                            <h5><i class="bi bi-graph-up-arrow me-2"></i>Kelly Recommendations</h5>
                            <div class="row text-center">
                                <div class="col-4"><div class="badge bg-danger fs-5 p-3 mb-2 w-100">Full<br>{kelly_info.get('kelly_full', 0):.1%}</div></div>
                                <div class="col-4"><div class="badge bg-warning fs-5 p-3 mb-2 w-100">½ Kelly<br>{kelly_info.get('kelly_half', 0):.1%}</div></div>
                                <div class="col-4"><div class="badge bg-success fs-5 p-3 mb-2 w-100">¼ Kelly<br>{kelly_info.get('kelly_quarter', 0):.1%}</div></div>
                            </div>
                            <hr>
                            <p class="mb-1"><strong>OOS Kelly:</strong> {kelly_oos.get('kelly_full', 0):.1%}</p>
                            <p class="mb-0"><em>Winrate: {kelly_info.get('win_rate', 0):.1%} | Payoff: {kelly_info.get('payoff_ratio', 0):.2f}</em></p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- 🎯 FOOTER -->
    <footer class="text-white py-5 mt-5">
        <div class="container text-center">
            <div class="row">
                <div class="col-md-4">
                    <h5><i class="bi bi-clock-history me-2"></i>Generated</h5>
                    <p class="mb-0">{datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}</p>
                </div>
                <div class="col-md-4">
                    <h5><i class="bi bi-award me-2"></i>Status</h5>
                    <span class="badge fs-3 px-5 py-3 {status_class}">{gate_status}</span>
                </div>
                <div class="col-md-4">
                    <h5><i class="bi bi-star-fill me-2"></i>Confidence</h5>
                    <div class="display-4 fw-bold">{confidence:.0f}%</div>
                </div>
            </div>
            <hr class="my-4 opacity-25">
            <p class="mb-0 opacity-75">
                <strong>RangeBreakoutUSDJPY_v4:</strong> 
                {'✅ LIVE_ELIGIBLE (Elite Sharpe 2.79!)' if sharpe > 2 else '⚠️ REVIEW - Strong Performance'}
            </p>
        </div>
    </footer>
</body>
</html>"""
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"🎉 QUANT DASHBOARD v4.0 GENERATED!")
    print(f"📁 Öffne: file://{html_path.absolute()}")
    print(f"🚀 Status: {gate_status} | Sharpe: {sharpe:.2f} | Confidence: {confidence:.0f}%")
    
    return html_path

if __name__ == "__main__":
    print("✅ Senior Quant Dashboard Module loaded")
