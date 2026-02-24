"""
Model Retraining Engine
------------------------
Retrains Isolation Forest on recent normal data from the database.
Versions models with timestamps, keeps last 2 versions.
Can be triggered manually or automatically on drift.
"""

from __future__ import annotations

import datetime as dt
import shutil
import time
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

from backend.database import get_db, get_recent_metrics
from backend.feature_engineering import compute_features, WINDOW
from backend.logging_config import get_logger

logger = get_logger("retraining")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

_last_retrained: str | None = None
_model_version: str = "v1.0-initial"
_training_sample_count: int = 0


def get_retraining_info() -> dict:
    return {
        "last_retrained_at": _last_retrained,
        "model_version": _model_version,
        "training_sample_count": _training_sample_count,
    }


def _version_existing_models() -> None:
    """Keep last 2 model versions by renaming current to timestamped backup."""
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    for name in ["isolation_model.pkl", "failure_model.pkl"]:
        src = MODEL_DIR / name
        if src.exists():
            dst = MODEL_DIR / f"{name}.{ts}.bak"
            shutil.copy2(src, dst)
            logger.info(f"Backed up {name} → {dst.name}")

    # Clean old backups — keep only last 2 per model
    for name in ["isolation_model.pkl", "failure_model.pkl"]:
        backups = sorted(MODEL_DIR.glob(f"{name}.*.bak"), reverse=True)
        for old in backups[2:]:
            old.unlink()
            logger.info(f"Removed old backup: {old.name}")


def retrain_isolation_forest(
    server_id: str = "local",
    min_samples: int = 200,
) -> dict:
    """
    Retrain Isolation Forest on recent normal (non-anomalous) data from DB.
    Returns status dict.
    """
    global _last_retrained, _model_version, _training_sample_count

    logger.info(f"Retraining started for server_id={server_id}")
    start = time.time()

    db = get_db()
    try:
        # Fetch recent metrics
        recent = get_recent_metrics(db, limit=2000, server_id=server_id)
        if len(recent) < min_samples:
            msg = f"Not enough data: {len(recent)} rows (need {min_samples})"
            logger.warning(msg)
            return {"status": "error", "message": msg}

        rows = [
            {"cpu": r.cpu, "memory": r.memory, "disk_io": r.disk_io,
             "response_time": r.response_time, "network": r.network}
            for r in reversed(recent)
        ]
    finally:
        db.close()

    # Augment data with noise
    rng = np.random.default_rng(42)
    augmented = list(rows)
    while len(augmented) < 3000:
        base = rows[rng.integers(0, len(rows))]
        augmented.append({
            "cpu": float(np.clip(base["cpu"] + rng.normal(0, 8), 0, 100)),
            "memory": float(np.clip(base["memory"] + rng.normal(0, 5), 0, 100)),
            "disk_io": float(np.clip(base["disk_io"] + rng.normal(0, max(1, base["disk_io"] * 0.5)), 0, 200)),
            "response_time": float(np.clip(base["response_time"] + rng.normal(0, 20), 10, 500)),
            "network": float(np.clip(base["network"] + rng.normal(0, max(5, base["network"] * 0.5)), 0, 500)),
        })

    # Compute features
    feature_matrix = []
    for i in range(WINDOW, len(augmented)):
        feat = compute_features(augmented[i - WINDOW: i + 1])
        if feat is not None:
            feature_matrix.append(feat[0])

    X = np.array(feature_matrix)
    if X.shape[0] < 100:
        msg = f"Feature matrix too small: {X.shape[0]} rows"
        logger.warning(msg)
        return {"status": "error", "message": msg}

    # Version existing models
    _version_existing_models()

    # Train new model
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    iso.fit(X)

    model_path = MODEL_DIR / "isolation_model.pkl"
    joblib.dump(iso, model_path)

    elapsed = time.time() - start
    _last_retrained = dt.datetime.now().isoformat()
    _model_version = f"v2.{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _training_sample_count = X.shape[0]

    logger.info(
        f"Retraining complete: version={_model_version}, "
        f"samples={X.shape[0]}, time={elapsed:.1f}s"
    )

    # Force model reload
    from backend import model as model_mod
    model_mod._model = None
    model_mod._explainer = None

    return {
        "status": "success",
        "model_version": _model_version,
        "training_samples": X.shape[0],
        "elapsed_seconds": round(elapsed, 1),
    }
