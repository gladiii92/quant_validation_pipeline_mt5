import numpy as np


def calculate_cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    if returns.size == 0:
        return np.nan
    threshold = np.percentile(returns, alpha * 100)
    tail = returns[returns <= threshold]
    if tail.size == 0:
        return threshold  # worst case: nur der Schwellenwert
    return float(tail.mean())


def calculate_time_to_recovery(equity_curve: np.ndarray) -> int:
    if equity_curve.size == 0:
        return 0

    peak = equity_curve[0]
    max_recovery = 0
    current = 0

    for v in equity_curve:
        if v >= peak:
            peak = v
            max_recovery = max(max_recovery, current)
            current = 0
        else:
            current += 1

    return int(max(max_recovery, current))
