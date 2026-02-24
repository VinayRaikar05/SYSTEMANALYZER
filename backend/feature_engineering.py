"""
Feature Engineering Module
--------------------------
Takes a window of raw metric rows and computes:
  • Latest raw values (so sustained high load is visible)
  • Rolling mean  (5-point window)
  • Rolling variance
  • Trend slope   (linear regression)
  • Spike magnitude (deviation from rolling mean)
  • Rate of change

Returns a flat numpy array ready for the Isolation Forest model.
"""

from __future__ import annotations

import numpy as np


SIGNALS = ["cpu", "memory", "disk_io", "response_time", "network"]
WINDOW = 5          # points for rolling stats


def _slope(values: np.ndarray) -> float:
    """Slope of a simple linear regression over index."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = values.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (values - y_mean)).sum() / denom)


def compute_features(rows: list[dict]) -> np.ndarray | None:
    """
    Accepts a list of metric dicts (most-recent last).
    Returns a 1-D feature array or None if there are not enough rows.

    Per signal (5 signals) we produce 6 features:
      latest_value, rolling_mean, rolling_var, trend_slope, spike_mag, rate_of_change
    Total = 5 * 6 = 30 features
    """
    if len(rows) < WINDOW:
        return None

    window = rows[-WINDOW:]
    features: list[float] = []

    for sig in SIGNALS:
        values = np.array([r[sig] for r in window], dtype=float)

        latest = float(values[-1])
        rolling_mean = float(values.mean())
        rolling_var = float(values.var())
        trend = _slope(values)
        spike = float(values[-1] - rolling_mean)
        roc = float(values[-1] - values[-2]) if len(values) >= 2 else 0.0

        features.extend([latest, rolling_mean, rolling_var, trend, spike, roc])

    return np.array(features).reshape(1, -1)
