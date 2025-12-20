"""
Stochastic scenario analysis for trading strategies.

Erzeugt synthetische Equity-Kurven mit
- Geometric Brownian Motion (GBM)
- Heston-Stochastic-Volatilitätsmodell
- Merton Jump-Diffusion

und berechnet für jede Modellfamilie robuste Kennzahlen
(Median-Return, Perzentile, MaxDD etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from backtest.metrics import calculate_metrics  # existiert bereits in deinem Projekt
from utils.logger import get_logger

logger = get_logger("STOCH_SIM")


# ---------------------------------------------------------------------------
# Hilfsfunktionen: Parameter-Schätzung und Drawdown
# ---------------------------------------------------------------------------

def estimate_drift_vol(returns: pd.Series, annualization: int = 252) -> Tuple[float, float]:
    """
    Schätzt Drift (mu) und Volatilität (sigma) aus logarithmischen Returns.
    """
    clean = returns.dropna()
    if clean.empty:
        raise ValueError("Cannot estimate parameters from empty return series.")
    mu = clean.mean() * annualization
    sigma = clean.std(ddof=1) * np.sqrt(annualization)
    return float(mu), float(sigma)


def compute_drawdown(equity: np.ndarray) -> float:
    """
    Maximaler Drawdown einer Equity-Kurve (als positive Zahl, z.B. 0.25 = -25 %).
    """
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(np.min(dd)) * -1.0


# ---------------------------------------------------------------------------
# GBM
# ---------------------------------------------------------------------------

@dataclass
class GeometricBrownianMotion:
    S0: float
    mu: float
    sigma: float

    def simulate(self, T: float, n_steps: int, n_paths: int, seed: int | None = None) -> np.ndarray:
        """
        Simuliert GBM-Pfade.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1)
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        S = np.zeros((n_paths, n_steps + 1), dtype=float)
        S[:, 0] = self.S0

        # Normalverteilte Zufallszahlen für alle Schritte
        dW = np.random.randn(n_paths, n_steps) * np.sqrt(dt)

        drift = (self.mu - 0.5 * self.sigma ** 2) * dt
        vol_term = self.sigma * dW

        for i in range(1, n_steps + 1):
            S[:, i] = S[:, i - 1] * np.exp(drift + vol_term[:, i - 1])

        return S


# ---------------------------------------------------------------------------
# Vereinfachtes Heston-Modell (Euler mit Full Truncation)
# ---------------------------------------------------------------------------

@dataclass
class HestonModel:
    S0: float
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    r: float = 0.0

    def simulate(self, T: float, n_steps: int, n_paths: int, seed: int | None = None) -> np.ndarray:
        """
        Simuliert Heston-Pfade (nur Preis; Vol-Pfad wird intern genutzt).
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        S = np.zeros((n_paths, n_steps + 1), dtype=float)
        v = np.zeros((n_paths, n_steps + 1), dtype=float)

        S[:, 0] = self.S0
        v[:, 0] = self.v0

        for i in range(1, n_steps + 1):
            # korrelierte Normalvariablen
            z1 = np.random.randn(n_paths)
            z2 = self.rho * z1 + np.sqrt(1.0 - self.rho ** 2) * np.random.randn(n_paths)

            v_prev = np.maximum(v[:, i - 1], 0.0)
            v[:, i] = (
                v[:, i - 1]
                + self.kappa * (self.theta - v_prev) * dt
                + self.sigma * np.sqrt(v_prev * dt) * z2
            )
            v_trunc = np.maximum(v[:, i], 0.0)

            S[:, i] = S[:, i - 1] * np.exp(
                (self.r - 0.5 * v_trunc) * dt + np.sqrt(v_trunc * dt) * z1
            )

        return S


# ---------------------------------------------------------------------------
# Merton Jump-Diffusion
# ---------------------------------------------------------------------------

@dataclass
class JumpDiffusionModel:
    S0: float
    mu: float
    sigma: float
    lambda_jump: float
    mu_jump: float
    sigma_jump: float

    def simulate(self, T: float, n_steps: int, n_paths: int, seed: int | None = None) -> np.ndarray:
        """
        Simuliert Merton Jump-Diffusion-Pfade.
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        S = np.zeros((n_paths, n_steps + 1), dtype=float)
        S[:, 0] = self.S0

        for i in range(1, n_steps + 1):
            # Brownian-Teil
            z = np.random.randn(n_paths)
            dW = np.sqrt(dt) * z

            # Poisson-Anzahl der Sprünge pro Schritt
            n_jumps = np.random.poisson(self.lambda_jump * dt, size=n_paths)
            # Summe der Sprunggrößen (lognormal approx)
            jump_sizes = np.where(
                n_jumps > 0,
                np.random.normal(self.mu_jump, self.sigma_jump, size=n_paths) * n_jumps,
                0.0,
            )

            drift = (self.mu - 0.5 * self.sigma ** 2) * dt
            diffusion = self.sigma * dW

            S[:, i] = S[:, i - 1] * np.exp(drift + diffusion + jump_sizes)

        return S


# ---------------------------------------------------------------------------
# Hauptfunktion: aus Trades -> Szenarien
# ---------------------------------------------------------------------------

def _equity_from_returns(
    initial_capital: float,
    ret_paths: np.ndarray,
) -> np.ndarray:
    """
    Baut Equity-Pfade aus Return-Pfaden.
    ret_paths: Shape (n_paths, n_steps); einfache (nicht log) Returns pro Step.
    """
    n_paths, n_steps = ret_paths.shape
    equity = np.zeros((n_paths, n_steps + 1), dtype=float)
    equity[:, 0] = initial_capital
    for i in range(1, n_steps + 1):
        equity[:, i] = equity[:, i - 1] * (1.0 + ret_paths[:, i - 1])
    return equity


def _metrics_from_equity_paths(
    equity_paths: np.ndarray,
) -> Dict[str, float]:
    """
    Aggregiert Kennzahlen über viele Equity-Pfade.
    """
    # Total-Return pro Pfad
    total_returns = equity_paths[:, -1] / equity_paths[:, 0] - 1.0
    max_dds = np.array([compute_drawdown(eq) for eq in equity_paths], dtype=float)

    metrics = {
        "median_return": float(np.median(total_returns)),
        "p5_return": float(np.percentile(total_returns, 5)),
        "p95_return": float(np.percentile(total_returns, 95)),
        "median_maxdd": float(np.median(max_dds)),
        "p95_maxdd": float(np.percentile(max_dds, 95)),
    }
    return metrics


def simulate_paths_from_trades(
    trades_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """
    Haupt-Einstiegspunkt für die Pipeline.

    Nimmt die historischen Trades, schätzt Parameter und erzeugt
    Szenario-Kennzahlen für GBM, Heston und Jump-Diffusion.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Erwartet mindestens Spalten ['entrytime', 'pnl'].
    config : Dict[str, Any]
        Unterstruktur in config.yaml, z.B.:

        simulation:
          T_years: 1.0
          num_steps: 252
          num_paths: 5000
          heston:
            kappa: 2.0
            theta: 0.04
            sigma: 0.3
            rho: -0.7
            v0: 0.04
          jumpdiffusion:
            lambda_jump: 0.1
            mu_jump: -0.05
            sigma_jump: 0.1

    Returns
    -------
    Dict[str, Dict[str, float]]
        Kennzahlen je Modell-Familie, z.B. simresults["gbm"]["median_return"].
    """
    if trades_df.empty:
        raise ValueError("simulate_paths_from_trades: trades_df is empty.")

    if "entrytime" not in trades_df.columns:
        raise ValueError("trades_df must contain 'entrytime' column.")
    if "pnl" not in trades_df.columns:
        raise ValueError("trades_df must contain 'pnl' column.")

    # ------------------------------------------------------------------
    # 1) Equity aus realen Trades -> Rückschluss auf Returns
    # ------------------------------------------------------------------
    trades_sorted = trades_df.sort_values("entrytime").copy()
    # Annahme: gleich große Positionsgröße -> Equity ~ kumulative PnL
    # Initialkapital später von außen übergeben (Pipeline).
    initial_capital = float(config.get("initial_capital", 10_000.0))
    equity = initial_capital + trades_sorted["pnl"].cumsum().values
    equity_series = pd.Series(equity, index=trades_sorted["entrytime"])

    # einfache Returns aus Equity
    eq_ret = equity_series.pct_change().dropna()
    if eq_ret.empty:
        raise ValueError("Not enough data to compute equity returns for simulation.")

    # ------------------------------------------------------------------
    # 2) Parameter-Schätzung
    # ------------------------------------------------------------------
    mu_ann, sigma_ann = estimate_drift_vol(np.log1p(eq_ret))
    logger.info(
        "Stochastic params from equity: mu_ann=%.4f, sigma_ann=%.4f",
        mu_ann,
        sigma_ann,
    )

    T = float(config.get("T_years", 1.0))
    n_steps = int(config.get("num_steps", 252))
    n_paths = int(config.get("num_paths", 5000))
    seed = int(config.get("random_state", 12345))

    results: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # 3) GBM
    # ------------------------------------------------------------------
    try:
        logger.info("Simulating GBM scenarios...")
        gbm = GeometricBrownianMotion(
            S0=equity_series.iloc[-1],
            mu=mu_ann,
            sigma=max(sigma_ann, 1e-8),
        )
        gbm_paths = gbm.simulate(T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
        gbm_metrics = _metrics_from_equity_paths(gbm_paths)
        results["gbm"] = gbm_metrics
        logger.info(
            "GBM scenarios done: median_return=%.2f, median_maxdd=%.2f",
            gbm_metrics["median_return"],
            gbm_metrics["median_maxdd"],
        )
    except Exception as exc:
        logger.exception("GBM simulation failed: %s", exc)

    # ------------------------------------------------------------------
    # 4) Heston
    # ------------------------------------------------------------------
    try:
        logger.info("Simulating Heston scenarios...")
        hcfg = config.get("heston", {})
        heston = HestonModel(
            S0=equity_series.iloc[-1],
            v0=float(hcfg.get("v0", sigma_ann ** 2)),
            kappa=float(hcfg.get("kappa", 2.0)),
            theta=float(hcfg.get("theta", sigma_ann ** 2)),
            sigma=float(hcfg.get("sigma", 0.3)),
            rho=float(hcfg.get("rho", -0.7)),
            r=float(hcfg.get("r", 0.0)),
        )
        heston_paths = heston.simulate(T=T, n_steps=n_steps, n_paths=n_paths, seed=seed + 1)
        heston_metrics = _metrics_from_equity_paths(heston_paths)
        results["heston"] = heston_metrics
        logger.info(
            "Heston scenarios done: median_return=%.2f, median_maxdd=%.2f",
            heston_metrics["median_return"],
            heston_metrics["median_maxdd"],
        )
    except Exception as exc:
        logger.exception("Heston simulation failed: %s", exc)

    # ------------------------------------------------------------------
    # 5) Jump-Diffusion
    # ------------------------------------------------------------------
    try:
        logger.info("Simulating Jump-Diffusion scenarios...")
        jcfg = config.get("jumpdiffusion", {})
        jd = JumpDiffusionModel(
            S0=equity_series.iloc[-1],
            mu=mu_ann,
            sigma=max(sigma_ann, 1e-8),
            lambda_jump=float(jcfg.get("lambda_jump", 0.1)),
            mu_jump=float(jcfg.get("mu_jump", -0.05)),
            sigma_jump=float(jcfg.get("sigma_jump", 0.1)),
        )
        jd_paths = jd.simulate(T=T, n_steps=n_steps, n_paths=n_paths, seed=seed + 2)
        jd_metrics = _metrics_from_equity_paths(jd_paths)
        results["jumpdiffusion"] = jd_metrics
        logger.info(
            "Jump-Diffusion scenarios done: median_return=%.2f, median_maxdd=%.2f",
            jd_metrics["median_return"],
            jd_metrics["median_maxdd"],
        )
    except Exception as exc:
        logger.exception("Jump-Diffusion simulation failed: %s", exc)

    return results
