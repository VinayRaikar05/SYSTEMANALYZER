"""
ML Model Loader, Predictor & Explainer
---------------------------------------
Loads the pre-trained Isolation Forest model and exposes:
  • predict(features)  → anomaly score + flag
  • explain(features)  → SHAP-based feature contributions (top 5)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "isolation_model.pkl"

# Feature names: 6 per signal × 5 signals = 30
SIGNALS = ["cpu", "memory", "disk_io", "response_time", "network"]
FEATURE_SUFFIXES = ["latest", "rolling_mean", "rolling_var", "trend_slope", "spike_mag", "rate_of_change"]
FEATURE_NAMES = [f"{sig}_{suf}" for sig in SIGNALS for suf in FEATURE_SUFFIXES]

_model = None
_explainer = None


def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train_model.py first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(features: np.ndarray) -> tuple[float, bool]:
    """
    Parameters
    ----------
    features : ndarray of shape (1, n_features)

    Returns
    -------
    anomaly_score : float   – Isolation Forest decision_function score
    is_anomaly    : bool    – True when the model flags the point as anomaly
    """
    model = _load_model()
    score = float(model.decision_function(features)[0])
    label = int(model.predict(features)[0])
    return score, label == -1


def explain(features: np.ndarray) -> list[dict]:
    """
    Use SHAP TreeExplainer to get feature contributions.
    Returns list of { name, value, contribution } sorted by |contribution|.
    """
    global _explainer
    try:
        import shap
        model = _load_model()

        if _explainer is None:
            _explainer = shap.TreeExplainer(model)

        shap_values = _explainer.shap_values(features)
        contributions = shap_values[0]  # first sample

        results = []
        for i, (name, contrib) in enumerate(zip(FEATURE_NAMES, contributions)):
            results.append({
                "name": name,
                "value": round(float(features[0][i]), 4),
                "contribution": round(float(contrib), 6),
            })

        # Sort by absolute contribution, return top 8
        results.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return results[:8]
    except Exception as e:
        print(f"[SHAP explain] error: {e}")
        return []
