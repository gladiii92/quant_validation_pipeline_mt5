# Quant Strategy Validation Pipeline

## Trade-based, regime-aware, OOS-first evaluation framework

**Dieses Repository enthält eine production-taugliche Validierungs-Pipeline für systematische Trading-Strategien.**

Es beantwortet **eine einzige Frage**:

> **Ist diese Strategie robust genug für echtes Kapital – oder nicht?**

Die Pipeline ist datenquellen-agnostisch. MT5 ist als Adapter implementiert – jede Strategie, die eine trade-level CSV produziert, kann evaluiert werden.

---

## Was das Projekt ist

- **Strategie-Audit & Entscheidungssystem**
- Fokus auf **Out-of-Sample-Robustheit**
- Aufbau um **Risiko, Tail-Events und Regime-Stabilität**
- Vollständig **automatisiert, reproduzierbar, erweiterbar**

## Was das Projekt NICHT ist

- Signal-Generator
- Bar-by-Bar-Backtester
- Curve-Fitting-Tool
- Strategie-Finder

---

## Kern-Design-Philosophie

- **Trades, nicht Bars** → realistisch & schnell  
- **OOS first** → Walk-Forward ist Pflicht
- **Risiko vor Rendite**
- **Regime-Bewusstsein** (VIX-basiert)
- **Distributionelles Denken** (Monte Carlo, CVaR)
- **Position Sizing als Konsequenz, nicht Ziel**
- **Automatisierung über Diskretion**

*Strategien, die hier bestehen, verdienen das Recht, live zu gehen.*

---

## End-to-End Pipeline
- Trade CSV

- Datenvalidierung

- Full-Sample Metriken

- Kosten/Slippage Tests

- VIX Regime Alignment

- Walk-Forward OOS

- Monte Carlo Simulation

- Tail-Risiko (CVaR)

- Kelly Sizing (OOS)

- Multi-Asset Check

- ELITE Decision Gate

- HTML Reports + Dashboard


**Keine manuelle Intervention. Kein Cherry-Picking.**

---

## Projektstruktur

![alt text](image.png)

---

## ELITE Decision Gate

**Das finale Output ist keine Metrik, sondern eine ENTSCHEIDUNG.**

| Kriterium            | Anforderung         |
|----------------------|---------------------|
| **OOS Sharpe**       | > 1.2               |
| **OOS Profit Factor**| > 1.5               |
| **OOS Max DD**       | < 20%               |
| **MC Survival**      | > 80%               |
| **CVaR (5%)**        | < -15%              |
| **Kelly (OOS)**      | > 0.05              |
| **Regime Compat.**   | Alle Regimes profitabel |
| **Multi-Asset**      | > 70% Hit-Rate      |

### Decision Output

{
"status": "PASS | WATCH | FAIL",
"confidence": 0.83,
"reason": "OOS Sharpe 1.4 > 1.2, CVaR marginal",
"violated_criteria": ["maxdd_oos"]
}


**Nur `PASS`-Strategien sind capital-ready.**

---

## Outputs pro Strategie

| Visualisierung            | Datei                   |
|---------------------------|-------------------------|
| Equity Curve              | `equity.png`            |
| MC Return Distribution    | `mcreturns.png`         |
| MC Equity Paths (100+)    | `mcpaths.png`           |
| Drawdown Analysis         | `drawdownanalysis.png`  |
| Walk-Forward Sharpe       | `walkforwardsharpe.png` |
| VIX Regime Sharpe         | `vixregimesharpe.png`   |
| Kelly Frontier            | `kellyfrontier.png`     |
| PnL Distribution          | `pnldistribution.png`   |
| **Summary**               | `summary.json`          |
| **Report**                | `report.txt`            |
| **Dashboard**             | `dashboard.html`        |


---

## Schnellstart

### 1. Automatischer Run (MT5 Raw Files)

python run_pipeline.py

*Organisiert raw Daten → konvertiert MT5 → validiert neueste Strategie*

### 2. Manueller Run (beliebige CSV)

python run_pipeline.py --trades-file path/to/trades.csv

**Minimale CSV-Struktur:**

entry_time,exit_time,pnl,volume
2023-01-01 10:00,2023-01-01 11:00,150.5,0.1
...


---

## Erweiterbarkeit

- **MT5 nur erster Adapter** – jede Trade-Log-Quelle integrierbar
- Skaliert zu **Multi-Strategie** und **Portfolio-Level**
- **Live-Monitoring** ready

---

## Status

|  **Fertig**                   |  **Geplant**                  |
|-------------------------------|-------------------------------|
| Production-ready Prototype    | Portfolio-Aggregation         |
| Personal Validation Framework | Korrelation Stress-Tests      |
| Alle Core-Features            | Bayesian Regimes              |
|                               | Live Feedback Loop            |


---

## Warum dieses Repo?

**Dies ist mein Ansatz für:**
- Quant Research
- Risk Management  
- Robustness Validation
- Decision Automation

**Disclaimer:** Für Research only. Trading = Risk. Past performance ≠ future results.

---

*⭐ Star dieses Repo wenn es deine Quant-Validierung verbessert!*
