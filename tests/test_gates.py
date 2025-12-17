"""Tests für Decision Gates."""

import pytest

from validation.gates import DecisionGate, GateStatus


@pytest.fixture
def gate():
    """Initialisiert ein DecisionGate für Tests."""
    return DecisionGate()


def test_gate_live_eligible(gate):
    """Test: Alle Kriterien erfüllt → LIVE_ELIGIBLE."""
    metrics = {
        "oos_sharpe": 1.5,
        "max_drawdown": 0.15,
        "mc_positive_prob": 0.90,
        "mt5_correlation": 0.95,
    }

    result = gate.evaluate(metrics)

    assert result.status == GateStatus.LIVE_ELIGIBLE
    assert len(result.violated_criteria) == 0
    assert result.confidence >= 0.85


def test_gate_conditional_pass(gate):
    """Test: Ein Kriterium verletzt → CONDITIONAL_PASS."""
    metrics = {
        "oos_sharpe": 0.5,   # Unter z.B. minimum 0.8
        "max_drawdown": 0.15,
        "mc_positive_prob": 0.90,
        "mt5_correlation": 0.95,
    }

    result = gate.evaluate(metrics)

    assert result.status == GateStatus.CONDITIONAL_PASS
    assert len(result.violated_criteria) >= 1


def test_gate_fail_fast(gate):
    """Test: Mehrere Kriterien verletzt → FAIL_FAST."""
    metrics = {
        "oos_sharpe": 0.3,
        "max_drawdown": 0.50,  # Über z.B. maximum 0.25
        "mc_positive_prob": 0.50,
        "mt5_correlation": 0.50,
    }

    result = gate.evaluate(metrics)

    assert result.status == GateStatus.FAIL_FAST
    assert len(result.violated_criteria) >= 2
