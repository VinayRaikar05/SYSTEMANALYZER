"""
Train Model Script v2
---------------------
1. Collects real system metrics via psutil for ~60s
2. Augments to 3000 normal samples
3. Trains Isolation Forest → isolation_model.pkl
4. Generates failure-labeled data → trains LogisticRegression → failure_model.pkl
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import psutil
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.feature_engineering import compute_features, SIGNALS, WINDOW

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

COLLECT_SECONDS = 60
SAMPLE_INTERVAL = 0.5


def _collect_real_baseline() -> list[dict]:
    print(f"📡 Collecting real system metrics for {COLLECT_SECONDS}s ...")
    rows: list[dict] = []
    prev_disk = psutil.disk_io_counters()
    prev_net = psutil.net_io_counters()
    psutil.cpu_percent(interval=None)

    end_time = time.time() + COLLECT_SECONDS
    while time.time() < end_time:
        time.sleep(SAMPLE_INTERVAL)
        cpu = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory().percent

        disk_now = psutil.disk_io_counters()
        disk_delta = (disk_now.read_bytes + disk_now.write_bytes) - \
                     (prev_disk.read_bytes + prev_disk.write_bytes)
        disk_io = disk_delta / (1024 * 1024 * SAMPLE_INTERVAL)
        prev_disk = disk_now

        net_now = psutil.net_io_counters()
        net_delta = (net_now.bytes_sent + net_now.bytes_recv) - \
                    (prev_net.bytes_sent + prev_net.bytes_recv)
        network = net_delta / (1024 * SAMPLE_INTERVAL)
        prev_net = net_now

        response_time = 50 + cpu * 1.5 + disk_io * 5
        rows.append({
            "cpu": round(cpu, 2), "memory": round(memory, 2),
            "disk_io": round(disk_io, 2), "response_time": round(response_time, 2),
            "network": round(network, 2),
        })

    print(f"   Collected {len(rows)} real samples")
    return rows


def _augment(rows: list[dict], target: int = 3000) -> list[dict]:
    augmented = list(rows)
    n = len(rows)
    while len(augmented) < target:
        base = rows[RNG.integers(0, n)]
        augmented.append({
            "cpu":           float(np.clip(base["cpu"] + RNG.normal(0, 8), 0, 100)),
            "memory":        float(np.clip(base["memory"] + RNG.normal(0, 5), 0, 100)),
            "disk_io":       float(np.clip(base["disk_io"] + RNG.normal(0, base["disk_io"] * 0.5 + 1), 0, 200)),
            "response_time": float(np.clip(base["response_time"] + RNG.normal(0, 20), 10, 500)),
            "network":       float(np.clip(base["network"] + RNG.normal(0, base["network"] * 0.5 + 5), 0, 500)),
        })
    return augmented


def _make_features(rows: list[dict]) -> np.ndarray:
    feature_matrix = []
    for i in range(WINDOW, len(rows)):
        feat = compute_features(rows[i - WINDOW: i + 1])
        if feat is not None:
            feature_matrix.append(feat[0])
    return np.array(feature_matrix)


def _train_failure_model(baseline: list[dict]) -> None:
    """Train a LogisticRegression that predicts failure probability from health indicators."""
    print("🔮 Training Failure Probability model ...")

    # Build training data: features = [anomaly_freq, health_slope, mean_health,
    #                                   min_health, cpu_trend, cpu_mean, mem_trend, mem_mean]
    X_list, y_list = [], []

    # Normal scenarios (label=0)
    for _ in range(500):
        n_pts = RNG.integers(5, 11)
        # Normal: low anomaly freq, stable health, normal CPU/Memory
        anomaly_freq = float(RNG.uniform(0, 0.2))
        health_slope = float(RNG.uniform(-1, 1))
        mean_health = float(RNG.uniform(75, 100))
        min_health = float(RNG.uniform(65, 95))
        base = baseline[RNG.integers(0, len(baseline))]
        cpu_mean = float(base["cpu"] + RNG.normal(0, 5))
        mem_mean = float(base["memory"] + RNG.normal(0, 3))
        cpu_trend = float(RNG.uniform(-2, 2))
        mem_trend = float(RNG.uniform(-1, 1))
        X_list.append([anomaly_freq, health_slope, mean_health, min_health,
                       cpu_trend, cpu_mean, mem_trend, mem_mean])
        y_list.append(0)

    # Failure scenarios (label=1)
    for _ in range(500):
        anomaly_freq = float(RNG.uniform(0.3, 1.0))
        health_slope = float(RNG.uniform(-8, -1))
        mean_health = float(RNG.uniform(20, 60))
        min_health = float(RNG.uniform(0, 40))
        cpu_mean = float(RNG.uniform(70, 100))
        mem_mean = float(RNG.uniform(75, 100))
        cpu_trend = float(RNG.uniform(1, 10))
        mem_trend = float(RNG.uniform(0.5, 5))
        X_list.append([anomaly_freq, health_slope, mean_health, min_health,
                       cpu_trend, cpu_mean, mem_trend, mem_mean])
        y_list.append(1)

    X = np.array(X_list)
    y = np.array(y_list)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)

    path = MODEL_DIR / "failure_model.pkl"
    joblib.dump(model, path)
    print(f"   ✅ Failure model saved → {path}")


def main() -> None:
    baseline = _collect_real_baseline()
    rows = _augment(baseline, target=3000)
    RNG.shuffle(rows)

    X = _make_features(rows)
    print(f"   Training samples : {X.shape[0]}")
    print(f"   Feature dim      : {X.shape[1]}")

    print("🤖 Training Isolation Forest …")
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    iso.fit(X)
    iso_path = MODEL_DIR / "isolation_model.pkl"
    joblib.dump(iso, iso_path)
    print(f"   ✅ Isolation Forest saved → {iso_path}")

    _train_failure_model(baseline)
    print("\n🎉 All models trained successfully!")


if __name__ == "__main__":
    main()
