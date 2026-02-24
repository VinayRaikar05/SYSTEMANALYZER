"""
Failure Probability Predictor
-----------------------------
Logistic Regression trained on engineered failure-risk features:
  • Recent anomaly frequency
  • CPU / Memory rolling trends
  • Health score decay rate

Outputs: probability of system failure (0–1).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "failure_model.pkl"

_model = None


def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            return None
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_failure_probability(
    recent_health_scores: list[int],
    recent_anomaly_flags: list[bool],
    recent_metrics: list[dict],
) -> float:
    """
    Predict failure probability from recent system state.

    Parameters
    ----------
    recent_health_scores : last N health scores (newest first)
    recent_anomaly_flags : last N anomaly flags (newest first)
    recent_metrics       : last N raw metric dicts (newest first)

    Returns
    -------
    probability : float 0–1
    """
    model = _load_model()
    if model is None or len(recent_health_scores) < 5:
        return 0.0

    scores = recent_health_scores[:10]
    flags = recent_anomaly_flags[:10]
    metrics = recent_metrics[:10]

    # Feature: anomaly frequency in last 10
    anomaly_freq = sum(flags) / len(flags)

    # Feature: health score decay (slope of last scores)
    hs = np.array(scores[::-1], dtype=float)  # oldest first
    if len(hs) >= 2:
        x = np.arange(len(hs))
        health_slope = float(np.polyfit(x, hs, 1)[0])
    else:
        health_slope = 0.0

    # Feature: mean health
    mean_health = float(np.mean(scores))

    # Feature: CPU trend
    cpus = np.array([m.get("cpu", 0) for m in reversed(metrics)], dtype=float)
    cpu_trend = float(np.polyfit(np.arange(len(cpus)), cpus, 1)[0]) if len(cpus) >= 2 else 0.0
    cpu_mean = float(cpus.mean()) if len(cpus) > 0 else 0.0

    # Feature: Memory trend
    mems = np.array([m.get("memory", 0) for m in reversed(metrics)], dtype=float)
    mem_trend = float(np.polyfit(np.arange(len(mems)), mems, 1)[0]) if len(mems) >= 2 else 0.0
    mem_mean = float(mems.mean()) if len(mems) > 0 else 0.0

    # Feature: min health (worst point)
    min_health = float(min(scores))

    features = np.array([[
        anomaly_freq, health_slope, mean_health, min_health,
        cpu_trend, cpu_mean, mem_trend, mem_mean,
    ]])

    prob = float(model.predict_proba(features)[0][1])
    return round(prob, 4)
