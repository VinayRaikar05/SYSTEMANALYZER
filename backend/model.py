"""
ML Model Loader, Predictor & Explainer — Enterprise Edition
-------------------------------------------------------------
• predict()         → anomaly score + flag + confidence (0-1)
• explain()         → SHAP feature contributions
• get_model_info()  → model metadata
• Graceful degradation if model is missing/broken
"""

from __future__ import annotations

import time
from pathlib import Path
from collections import deque
import numpy as np
import joblib

from backend.logging_config import get_logger

logger = get_logger("model")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "isolation_model.pkl"

SIGNALS = ["cpu", "memory", "disk_io", "response_time", "network"]
FEATURE_SUFFIXES = ["latest", "rolling_mean", "rolling_var", "trend_slope", "spike_mag", "rate_of_change"]
FEATURE_NAMES = [f"{sig}_{suf}" for sig in SIGNALS for suf in FEATURE_SUFFIXES]

_model = None
_explainer = None
_load_time: float = 0.0
_model_status: str = "not_loaded"

# Historical scores for confidence normalization
_score_history: deque[float] = deque(maxlen=500)


def _load_model():
    global _model, _load_time, _model_status
    if _model is None:
        if not MODEL_PATH.exists():
            logger.error(f"Model file not found: {MODEL_PATH}")
            _model_status = "fallback_mode"
            return None
        try:
            t0 = time.time()
            _model = joblib.load(MODEL_PATH)
            _load_time = time.time() - t0
            _model_status = "active"
            logger.info(f"Model loaded in {_load_time:.3f}s from {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            _model_status = "fallback_mode"
            return None
    return _model


def predict(features: np.ndarray) -> tuple[float, bool, float]:
    """
    Returns (anomaly_score, is_anomaly, confidence).
    confidence is normalized 0-1 (1 = very confident it's anomalous).
    Falls back gracefully if model unavailable.
    """
    model = _load_model()
    if model is None:
        # Fallback: return neutral values
        return 0.0, False, 0.0

    try:
        score = float(model.decision_function(features)[0])
        label = int(model.predict(features)[0])
        is_anomaly = label == -1

        # Track for confidence normalization
        _score_history.append(score)

        # Confidence: normalize using historical min/max
        confidence = _compute_confidence(score)

        return score, is_anomaly, confidence
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 0.0, False, 0.0


def _compute_confidence(score: float) -> float:
    """
    Normalize anomaly score to 0-1 confidence.
    0 = definitely normal, 1 = definitely anomalous.
    Uses min-max scaling from historical distribution.
    """
    if len(_score_history) < 10:
        # Not enough history — use fixed range
        return max(0.0, min(1.0, 0.5 - score * 2))

    hist = np.array(_score_history)
    s_min, s_max = float(hist.min()), float(hist.max())
    if s_max == s_min:
        return 0.5

    # Invert: lower score = higher confidence of anomaly
    normalized = (s_max - score) / (s_max - s_min)
    return round(max(0.0, min(1.0, normalized)), 4)


def explain(features: np.ndarray) -> list[dict]:
    """SHAP TreeExplainer feature contributions."""
    global _explainer
    model = _load_model()
    if model is None:
        return []

    try:
        import shap

        if _explainer is None:
            _explainer = shap.TreeExplainer(model)

        shap_values = _explainer.shap_values(features)
        contributions = shap_values[0]

        results = []
        for i, (name, contrib) in enumerate(zip(FEATURE_NAMES, contributions)):
            results.append({
                "name": name,
                "value": round(float(features[0][i]), 4),
                "contribution": round(float(contrib), 6),
            })

        results.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return results[:8]
    except Exception as e:
        logger.error(f"SHAP explain failed: {e}")
        return []


def get_model_info() -> dict:
    """Return model metadata."""
    _load_model()  # Ensure model status is current

    from backend.retraining import get_retraining_info
    retrain_info = get_retraining_info()

    return {
        "model_status": _model_status,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "model_version": retrain_info.get("model_version", "v1.0-initial"),
        "last_retrained_at": retrain_info.get("last_retrained_at"),
        "training_sample_count": retrain_info.get("training_sample_count", 0),
        "contamination_rate": 0.02,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "load_time_ms": round(_load_time * 1000, 2),
        "confidence_samples": len(_score_history),
    }
