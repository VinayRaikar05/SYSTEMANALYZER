# ═══════════════════════════════════════════════════════════════════
# AI-BASED SYSTEM FAILURE EARLY WARNING ENGINE
# COMPLETE PROJECT GUIDE — Architecture, Logic & Code Walkthrough
# ═══════════════════════════════════════════════════════════════════

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Original Requirements](#2-original-requirements)
3. [Architecture & Data Flow](#3-architecture--data-flow)
4. [Project Structure](#4-project-structure)
5. [Dependencies](#5-dependencies)
6. [FILE 1: feature_engineering.py — Feature Engineering](#6-file-1-feature_engineeringpy)
7. [FILE 2: model.py — ML Model + SHAP Explainer](#7-file-2-modelpy)
8. [FILE 3: risk_engine.py — Health Score + Risk + Root Cause](#8-file-3-risk_enginepy)
9. [FILE 4: failure_predictor.py — Failure Probability](#9-file-4-failure_predictorpy)
10. [FILE 5: database.py — Database Layer](#10-file-5-databasepy)
11. [FILE 6: train_model.py — Model Training](#11-file-6-train_modelpy)
12. [FILE 7: main.py — FastAPI Server](#12-file-7-mainpy)
13. [FILE 8: dashboard.html — Frontend Dashboard](#13-file-8-dashboardhtml)
14. [FILE 9: script.js — Dashboard Logic](#14-file-9-scriptjs)
15. [The 7 Upgrades — Detailed Breakdown](#15-the-7-upgrades)
16. [How to Run](#16-how-to-run)
17. [Verification Results](#17-verification-results)

---

## 1. PROJECT OVERVIEW

This project is a **production-ready AI system** that:

- **Monitors** your computer's real CPU, Memory, Disk I/O, and Network stats in real-time
- **Detects Anomalies** using an Isolation Forest (unsupervised ML)
- **Predicts Failures** before they happen using Logistic Regression
- **Explains Decisions** with SHAP (SHapley Additive exPlanations)
- **Forecasts Degradation** using linear regression on health trends
- **Alerts** with root cause diagnosis when risk escalates
- Displays everything on a **live dark-themed dashboard** with Chart.js

**Key principle:** No external AI APIs. All ML runs locally using Scikit-learn.

---

## 2. ORIGINAL REQUIREMENTS

The system was designed around these core requirements:

### Core Requirements
- **Backend-driven** FastAPI architecture
- **ML model runs locally** using Scikit-learn (no external AI APIs)
- **Real-time metric ingestion** from the host machine via psutil
- **Time-series feature engineering** (rolling stats, trends, spikes)
- **Anomaly detection** using Isolation Forest
- **Health score** (0-100) with risk levels (GREEN/YELLOW/RED)
- **Live monitoring dashboard** with Chart.js visualizations

### Functional Flow
```
Metric Ingestion → Feature Engineering → Anomaly Detection →
Health Score → Risk Level → Alert Generation → Dashboard Display
```

### Tech Stack
- **Backend:** Python, FastAPI, SQLAlchemy, SQLite
- **ML:** Scikit-learn (Isolation Forest, Logistic Regression), SHAP
- **Data:** Pandas, NumPy, psutil
- **Frontend:** HTML, Bootstrap 5, Chart.js
- **Serialization:** Joblib (model persistence)

### 7 Upgrades Added
1. Failure Probability Predictor (Logistic Regression)
2. SHAP-Based Explainability
3. Adaptive Threshold Tuning Slider
4. Failure Pattern Simulation Mode
5. Health Trend Forecast
6. Multi-Server Support
7. Alert Root Cause Hints

---

## 3. ARCHITECTURE & DATA FLOW

### High-Level Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    EVERY 2 SECONDS                               │
│                                                                  │
│  ┌─────────┐    ┌─────────────┐    ┌──────────────┐            │
│  │ psutil   │───>│ Feature     │───>│ Isolation    │            │
│  │ (Real    │    │ Engineering │    │ Forest       │            │
│  │ Metrics) │    │ (30 feats)  │    │ (Anomaly?)   │            │
│  └─────────┘    └─────────────┘    └──────┬───────┘            │
│                                           │                     │
│        ┌──────────┐    ┌─────────────────┐│                     │
│        │ SHAP     │<───│ Feature Vector  ││                     │
│        │ Explainer│    └─────────────────┘│                     │
│        └────┬─────┘                       │                     │
│             │         ┌───────────────────▼──────────┐          │
│             │         │ Risk Engine                   │          │
│             │         │ • Health Score (0-100)        │          │
│             │         │ • Risk Level (GREEN/YELLOW/RED│          │
│             │         │ • Severity Penalty            │          │
│             │         │ • Root Cause Hint             │          │
│             │         └───────────┬──────────────────┘          │
│             │                     │                              │
│             │    ┌────────────────▼─────────────┐               │
│             │    │ Failure Predictor             │               │
│             │    │ (Logistic Regression)         │               │
│             │    │ → Failure Probability 0-100%  │               │
│             │    └────────────────┬─────────────┘               │
│             │                     │                              │
│             │    ┌────────────────▼─────────────┐               │
│             │    │ SQLite Database              │               │
│             │    │ • raw_metrics                │               │
│             │    │ • health_records             │               │
│             │    │ • alerts                     │               │
│             │    └────────────────┬─────────────┘               │
│             │                     │                              │
│     ┌───────▼─────────────────────▼────────────────────┐        │
│     │           FastAPI REST API                        │        │
│     │  /health  /explain  /forecast  /alerts  etc.      │        │
│     └───────────────────────┬──────────────────────────┘        │
│                             │                                    │
│     ┌───────────────────────▼──────────────────────────┐        │
│     │        Dashboard (HTML + Chart.js)                │        │
│     │  Polls API every 2s, renders gauges/charts        │        │
│     └──────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────────┘
```

### Data Pipeline Detail

```
Raw Metric (5 values) → Feature Window (last 5 readings) →
Feature Vector (30 values) → Isolation Forest Score →
Health Score (0-100) - Severity Penalty → Risk Level →
Failure Probability → Root Cause → Store + Alert if escalated
```

---

## 4. PROJECT STRUCTURE

```
d:\PROJECTS\SYSTEM\
│
├── backend\                          # All server-side Python code
│   ├── __init__.py                   # Makes backend a Python package
│   ├── main.py                       # FastAPI app, endpoints, pipeline
│   ├── database.py                   # SQLite ORM models and CRUD
│   ├── feature_engineering.py        # Raw metrics → 30 ML features
│   ├── model.py                      # Isolation Forest + SHAP
│   ├── risk_engine.py                # Health score + risk + root cause
│   ├── failure_predictor.py          # Logistic Regression for failure %
│   └── train_model.py               # Training script (60s real data)
│
├── frontend\                         # Client-side dashboard
│   ├── dashboard.html                # Dark-themed Bootstrap 5 layout
│   └── script.js                     # Chart.js + API polling logic
│
├── models\                           # Trained ML models (auto-generated)
│   ├── isolation_model.pkl           # Isolation Forest (Joblib pickle)
│   └── failure_model.pkl             # Logistic Regression (Joblib pickle)
│
├── data\                             # Database (auto-generated)
│   └── system.db                     # SQLite database
│
├── requirements.txt                  # Python dependencies
└── PROJECT_GUIDE.md                  # This file
```

---

## 5. DEPENDENCIES

**File: `requirements.txt`**

```
fastapi>=0.115.0          # Web framework for REST API
uvicorn[standard]>=0.30.0 # ASGI server to run FastAPI
scikit-learn>=1.5.0       # ML models (Isolation Forest, Logistic Regression)
pandas>=2.2.0             # Data manipulation
numpy>=1.26.0             # Numerical computing
joblib>=1.4.0             # Model serialization (save/load .pkl files)
aiosqlite>=0.20.0         # Async SQLite driver
sqlalchemy[asyncio]>=2.0.30  # ORM for database operations
```

Additional packages installed separately:
- `psutil` — Cross-platform system metrics (CPU, Memory, Disk, Network)
- `shap` — SHAP explainability for ML models

---

## 6. FILE 1: feature_engineering.py

**Purpose:** Converts a window of raw metric readings into a 30-dimensional feature vector suitable for the Isolation Forest model.

**Why 30 features?** Raw values alone aren't enough. The model needs to understand *patterns* — is CPU trending up? Is there sudden variance? We compute 6 statistical features for each of 5 signals = 30 total.

### Complete Code with Annotations:

```python
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

# The 5 system signals we monitor
SIGNALS = ["cpu", "memory", "disk_io", "response_time", "network"]
WINDOW = 5   # How many recent readings we use for rolling calculations
```

**Design Decision:** Window of 5 means we look at the last 10 seconds of data (5 readings × 2s interval). This is enough to detect trends without being too laggy.

### The Slope Function (Linear Regression):

```python
def _slope(values: np.ndarray) -> float:
    """Slope of a simple linear regression over index."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)     # [0, 1, 2, 3, 4]
    x_mean = x.mean()                 # 2.0
    y_mean = values.mean()            # average of the signal
    denom = ((x - x_mean) ** 2).sum() # sum of squared deviations
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (values - y_mean)).sum() / denom)
```

**Logic:** This implements the ordinary least squares formula: `slope = Σ((xi - x̄)(yi - ȳ)) / Σ((xi - x̄)²)`. A positive slope means the metric is trending upward (worsening for CPU/Memory), negative means trending downward (improving).

### The Main Feature Computation:

```python
def compute_features(rows: list[dict]) -> np.ndarray | None:
    """
    Per signal (5 signals) we produce 6 features:
      latest_value, rolling_mean, rolling_var, trend_slope, spike_mag, rate_of_change
    Total = 5 * 6 = 30 features
    """
    if len(rows) < WINDOW:
        return None          # Not enough data yet, skip

    window = rows[-WINDOW:]  # Take the 5 most recent readings
    features: list[float] = []

    for sig in SIGNALS:      # Loop over cpu, memory, disk_io, etc.
        values = np.array([r[sig] for r in window], dtype=float)
        # e.g., values = [14.2, 15.1, 13.8, 16.0, 14.5] for CPU

        latest = float(values[-1])          # Feature 1: Current raw value
        rolling_mean = float(values.mean()) # Feature 2: Average over window
        rolling_var = float(values.var())   # Feature 3: How stable is it?
        trend = _slope(values)              # Feature 4: Going up or down?
        spike = float(values[-1] - rolling_mean)  # Feature 5: Deviation from normal
        roc = float(values[-1] - values[-2])      # Feature 6: Jump since last reading

        features.extend([latest, rolling_mean, rolling_var, trend, spike, roc])

    return np.array(features).reshape(1, -1)  # Shape: (1, 30)
```

**Example output for healthy system:**
```
[14.5, 14.72, 0.56, 0.15, -0.22, -1.5,   # CPU features
 65.3, 65.28, 0.01, 0.02, 0.02, 0.1,      # Memory features
 2.4, 3.1, 1.2, -0.3, -0.7, -0.5,         # Disk I/O features
 83.1, 82.5, 4.2, 0.8, 0.6, 1.2,          # Response Time features
 18.5, 22.1, 45.0, -2.1, -3.6, -5.0]      # Network features
```

**Why `latest_value` was added (critical fix):** Without it, a server running at constant 95% CPU would show zero variance, zero slope, zero spike — looking statistically "normal" to the model. Adding the raw value lets the model see absolute danger levels.

---

## 7. FILE 2: model.py

**Purpose:** Loads the trained Isolation Forest model, runs predictions, and provides SHAP-based explanations.

### Complete Code with Annotations:

```python
"""
ML Model Loader, Predictor & Explainer
---------------------------------------
Loads the pre-trained Isolation Forest model and exposes:
  • predict(features)  → anomaly score + flag
  • explain(features)  → SHAP-based feature contributions (top 8)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "isolation_model.pkl"

# Human-readable names for all 30 features
SIGNALS = ["cpu", "memory", "disk_io", "response_time", "network"]
FEATURE_SUFFIXES = ["latest", "rolling_mean", "rolling_var",
                    "trend_slope", "spike_mag", "rate_of_change"]
FEATURE_NAMES = [f"{sig}_{suf}" for sig in SIGNALS for suf in FEATURE_SUFFIXES]
# Produces: ["cpu_latest", "cpu_rolling_mean", "cpu_rolling_var", ...]

_model = None      # Cached model (loaded once)
_explainer = None   # Cached SHAP explainer
```

**Design Decision:** Models are loaded lazily and cached globally. The first prediction takes ~50ms to load from disk, subsequent ones are instant.

### Prediction Function:

```python
def predict(features: np.ndarray) -> tuple[float, bool]:
    """
    Returns
    -------
    anomaly_score : float   – higher = more normal, negative = anomalous
    is_anomaly    : bool    – True when model flags as anomaly
    """
    model = _load_model()
    score = float(model.decision_function(features)[0])
    label = int(model.predict(features)[0])   # 1 = normal, -1 = anomaly
    return score, label == -1
```

**How Isolation Forest works:**
- It builds 200 random binary trees (n_estimators=200)
- For each tree, it randomly selects a feature and a split point
- **Anomalies are isolated quickly** (few splits needed) because they're outliers
- **Normal points take many splits** because they're surrounded by similar data
- `decision_function` returns: positive = deep in tree (normal), negative = shallow (anomaly)
- `contamination=0.02` means the model expects 2% of data to be anomalous, setting the threshold

### SHAP Explanation Function:

```python
def explain(features: np.ndarray) -> list[dict]:
    """
    Use SHAP TreeExplainer to get feature contributions.
    Returns list of { name, value, contribution } sorted by |contribution|.
    """
    import shap

    if _explainer is None:
        _explainer = shap.TreeExplainer(model)

    shap_values = _explainer.shap_values(features)
    contributions = shap_values[0]  # SHAP values for first sample

    results = []
    for i, (name, contrib) in enumerate(zip(FEATURE_NAMES, contributions)):
        results.append({
            "name": name,                                    # e.g., "cpu_latest"
            "value": round(float(features[0][i]), 4),        # e.g., 14.5
            "contribution": round(float(contrib), 6),        # e.g., -0.358
        })

    results.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return results[:8]  # Return top 8 most impactful features
```

**How SHAP works:**
- Based on game theory (Shapley values) — each feature gets credit for its contribution
- **Negative contribution** = this feature pushes the score toward anomaly
- **Positive contribution** = this feature pushes toward normal
- Example: `cpu_trend_slope: -0.358` means "CPU is trending upward, which is 0.358 anomaly-points suspicious"

---

## 8. FILE 3: risk_engine.py

**Purpose:** Converts raw ML output into human-understandable health scores, risk levels, and root cause hints.

### Severity Penalty System:

```python
def compute_metric_severity(metrics: dict) -> int:
    """Compute a severity penalty (0–40) based on extreme raw metric values."""
    penalty = 0
    cpu = metrics.get("cpu", 0)
    mem = metrics.get("memory", 0)
    disk = metrics.get("disk_io", 0)
    resp = metrics.get("response_time", 0)

    # CPU penalty tiers:
    if cpu > 90:    penalty += 12   # Critical CPU
    elif cpu > 80:  penalty += 7    # High CPU
    elif cpu > 75:  penalty += 3    # Elevated CPU

    # Memory penalty tiers:
    if mem > 88:    penalty += 10   # Critical Memory
    elif mem > 78:  penalty += 5    # High Memory
    elif mem > 72:  penalty += 2    # Elevated Memory

    # Disk I/O penalty (measured in MB/s):
    if disk > 80:   penalty += 8    # Extreme disk throughput
    elif disk > 50: penalty += 4    # Heavy disk activity

    # Response time penalty (measured in ms):
    if resp > 250:   penalty += 10  # Very slow responses
    elif resp > 180: penalty += 5   # Slow responses

    return min(40, penalty)  # Cap at 40 points maximum
```

**Why this exists:** The Isolation Forest gives mild anomaly scores for sustained high load (e.g., constant CPU=92% has zero variance). The severity penalty catches these cases by looking at raw absolute values.

### Health Score Mapping:

```python
def compute_health_score(anomaly_score, metrics=None, sensitivity=5):
    """Map Isolation Forest score to 0–100, apply severity penalty."""

    # Sensitivity adjustment: 1=loose (less reactive), 10=strict (more reactive)
    # Formula: multiplier ranges from 0.68 (loose) to 1.4 (strict)
    sens_mult = 0.6 + (sensitivity / 10) * 0.8
    adj_score = anomaly_score / sens_mult

    # Piecewise mapping from anomaly score to health:
    # adj_score >= 0.15  → health 92-100 (clearly normal)
    # adj_score >= 0.05  → health 85-92  (normal)
    # adj_score >= 0.0   → health 70-85  (borderline)
    # adj_score >= -0.1  → health 50-70  (mild anomaly)
    # adj_score >= -0.25 → health 25-50  (moderate anomaly)
    # adj_score < -0.25  → health 0-25   (severe anomaly)

    # Then subtract severity penalty for dangerously high raw values
    if metrics:
        base -= compute_metric_severity(metrics)

    return max(0, min(100, base))  # Clamp to 0-100
```

**Sensitivity logic:** At sensitivity=1, the multiplier is 0.68, so the score is divided by 0.68 (making it look more positive/normal). At sensitivity=10, divided by 1.4 (making it look more negative/anomalous). This gives real-time control over how reactive the system is.

### Risk Level Evaluation:

```python
def evaluate_risk(recent_anomaly_flags: list[bool]) -> str:
    window = recent_anomaly_flags[:10]  # Last 10 readings
    count = sum(window)                 # How many were anomalous?
    if count >= 3:   return "RED"       # ≥3 anomalies = critical
    elif count >= 2: return "YELLOW"    # ≥2 anomalies = warning
    return "GREEN"                       # 0-1 anomalies = okay
```

### Alert Escalation Logic:

```python
def should_alert(prev_risk: str, new_risk: str) -> str | None:
    levels = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    if levels.get(new_risk, 0) > levels.get(prev_risk, 0):
        return "CRITICAL" if new_risk == "RED" else "WARNING"
    return None  # No alert if risk stayed same or decreased
```

**Logic:** Alerts only fire on **escalation**, not on sustained risk. So GREEN→RED = CRITICAL alert, GREEN→YELLOW = WARNING alert, but RED→RED = no alert (already high).

### Root Cause Diagnosis:

```python
def diagnose_root_cause(metrics: dict) -> str:
    issues = []

    if cpu > 75:
        issues.append((cpu, "High CPU — Check compute-heavy processes"))
    if mem > 78:
        issues.append((mem, "High Memory — Possible memory leak or cache bloat"))
    if disk > 50:
        issues.append((disk, "Heavy Disk I/O — Large file operations or swap"))
    if resp > 200:
        issues.append((resp/5, "Slow Response — Backend congestion"))
    if net > 200:
        issues.append((net/10, "High Network — Heavy data transfer"))

    if not issues:
        return "System nominal"

    issues.sort(key=lambda x: x[0], reverse=True)
    return issues[0][1]  # Return the most severe issue
```

**Logic:** Checks each metric against thresholds, ranks by severity, returns the worst one as a human-readable string.

---

## 9. FILE 4: failure_predictor.py

**Purpose:** A separate ML model (Logistic Regression) that predicts the probability of system failure from recent health history and metric trends.

### Why a separate model?
The Isolation Forest tells you "this moment is anomalous." But failure prediction needs to look at **trends over time**: Is health declining? Are anomalies becoming frequent? CPU trending upward? This requires different features.

### The 8 Failure-Prediction Features:

```python
features = [
    anomaly_freq,    # What % of last 10 readings were anomalous (0.0-1.0)
    health_slope,    # Slope of health score over time (negative = declining)
    mean_health,     # Average health in recent window
    min_health,      # Worst health score recently
    cpu_trend,       # Is CPU usage trending upward? (positive slope)
    cpu_mean,        # Average CPU usage recently
    mem_trend,       # Is memory trending upward?
    mem_mean,        # Average memory usage recently
]
```

### Prediction Logic:

```python
def predict_failure_probability(recent_health_scores, recent_anomaly_flags, recent_metrics):
    # Need at least 5 data points to make a meaningful prediction
    if len(recent_health_scores) < 5:
        return 0.0

    # Compute all 8 features from the recent history...

    # Get probability from Logistic Regression
    prob = model.predict_proba(features)[0][1]  # Index [1] = probability of class 1 (failure)
    return round(prob, 4)
```

**How it behaves:**
- Normal system: `anomaly_freq=0.0, health_slope=0.2, mean_health=90` → probability ~0.01 (1%)
- Degrading system: `anomaly_freq=0.5, health_slope=-3.0, mean_health=45` → probability ~0.85 (85%)
- Injected failure: `anomaly_freq=1.0, health_slope=-8.0, mean_health=30` → probability ~1.0 (100%)

---

## 10. FILE 5: database.py

**Purpose:** Defines the SQLite database schema using SQLAlchemy ORM and provides CRUD helper functions.

### Three Tables:

```python
class RawMetric(Base):
    __tablename__ = "raw_metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    server_id = Column(String(50), default="local", index=True)  # Multi-server support
    timestamp = Column(DateTime, default=dt.datetime.now)
    cpu = Column(Float)
    memory = Column(Float)
    disk_io = Column(Float)
    response_time = Column(Float)
    network = Column(Float)

class HealthRecord(Base):
    __tablename__ = "health_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    server_id = Column(String(50), default="local", index=True)
    timestamp = Column(DateTime, default=dt.datetime.now)
    health_score = Column(Integer)          # 0-100
    anomaly_score = Column(Float)           # Isolation Forest raw score
    anomaly_flag = Column(Boolean)          # True if anomalous
    risk_level = Column(String(10))         # GREEN, YELLOW, RED
    failure_prob = Column(Float, default=0.0)            # 0.0-1.0
    root_cause = Column(String(200), default="System nominal")  # Human hint

class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    server_id = Column(String(50), default="local", index=True)
    timestamp = Column(DateTime, default=dt.datetime.now)
    severity = Column(String(20))           # WARNING or CRITICAL
    message = Column(String(500))           # Alert details + root cause
```

### CRUD Helpers:

```python
# All queries filter by server_id for multi-server support
def get_recent_metrics(db, limit=50, server_id="local"):
    return db.query(RawMetric).filter(RawMetric.server_id == server_id)
              .order_by(RawMetric.id.desc()).limit(limit).all()

def get_server_ids(db):
    """Returns all unique server IDs in the database."""
    rows = db.query(RawMetric.server_id).distinct().all()
    return [r[0] for r in rows]
```

---

## 11. FILE 6: train_model.py

**Purpose:** Training script that collects real data from your machine and trains both ML models.

### Training Pipeline:

```
Step 1: Collect 60 seconds of real psutil data (~118 samples)
   ↓
Step 2: Augment to 3000 samples with gaussian noise
   ↓
Step 3: Compute 30 features for each sample
   ↓
Step 4: Train Isolation Forest → isolation_model.pkl
   ↓
Step 5: Generate synthetic failure scenarios
   ↓
Step 6: Train Logistic Regression → failure_model.pkl
```

### Real Data Collection:

```python
def _collect_real_baseline():
    """Sample real psutil metrics for 60 seconds."""
    end_time = time.time() + 60
    while time.time() < end_time:
        time.sleep(0.5)  # Sample every 0.5 seconds

        cpu = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory().percent
        disk_io = disk_delta / (1024 * 1024 * 0.5)   # Convert bytes to MB/s
        network = net_delta / (1024 * 0.5)            # Convert bytes to KB/s
        response_time = 50 + cpu * 1.5 + disk_io * 5  # Estimated ms

        rows.append({...})

    # Returns ~118 real samples
```

**Why real data?** Early versions used simulated data with guessed ranges (e.g., disk_io 10-350). But real psutil output was completely different (disk_io 0.1-15 MB/s, network 0-300 KB/s). Training on guessed data caused everything to look "anomalous" — constant RED risk on a healthy system. Training on *actual* machine data fixes this.

### Data Augmentation:

```python
def _augment(rows, target=3000):
    """Add gaussian noise around real samples to build robust training set."""
    while len(augmented) < target:
        base = random_real_sample()
        augmented.append({
            "cpu": base["cpu"] + gaussian_noise(std=8),        # ±8% variation
            "memory": base["memory"] + gaussian_noise(std=5),  # ±5% variation
            "disk_io": base["disk_io"] + proportional_noise(), # ±50% variation
            ...
        })
```

**Why augment?** 118 real samples isn't enough for robust training. Adding noise creates realistic variations while keeping the distribution centered on real values.

### Failure Model Training:

```python
def _train_failure_model(baseline):
    # 500 "normal" scenarios (label=0):
    #   - Low anomaly frequency (0-20%)
    #   - Stable health slope (-1 to +1)
    #   - High mean health (75-100)
    #   - CPU/Memory from real baseline

    # 500 "failure" scenarios (label=1):
    #   - High anomaly frequency (30-100%)
    #   - Negative health slope (-8 to -1)
    #   - Low mean health (20-60)
    #   - High CPU (70-100%), high Memory (75-100%)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
```

---

## 12. FILE 7: main.py

**Purpose:** The heart of the system — FastAPI server with all 13 endpoints, the real-time pipeline, and metric collection.

### The Pipeline (runs every 2 seconds):

```python
def _run_pipeline(data):
    # 1. Store raw metric in SQLite
    insert_metric(db, data)

    # 2. Get last 10 readings for feature window
    recent = get_recent_metrics(db, limit=10)

    # 3. Compute 30-feature vector
    features = compute_features(recent_dicts)

    # 4. Isolation Forest prediction
    anomaly_score, is_anomaly = predict(features)

    # 5. Health score (with sensitivity + severity penalty)
    health_score = compute_health_score(anomaly_score, metrics=data,
                                        sensitivity=_sensitivity)

    # 6. SHAP explanation (stored for /explain endpoint)
    _latest_explain = explain(features)

    # 7. Risk level (from anomaly frequency in last 10)
    risk_level = evaluate_risk(recent_flags)

    # 8. Root cause diagnosis
    root_cause = diagnose_root_cause(data)

    # 9. Failure probability
    failure_prob = predict_failure_probability(recent_scores, recent_anom, recent_met)

    # 10. Store health record
    insert_health_record(db, health_data)

    # 11. Fire alert if risk escalated
    if should_alert(_prev_risk, risk_level):
        insert_alert(db, alert_data)
```

### Real Metric Collection (psutil):

```python
def _collect_real_metrics():
    cpu = psutil.cpu_percent(interval=None)           # CPU usage %
    memory = psutil.virtual_memory().percent          # RAM usage %

    # Disk: bytes delta since last call / (1MB * 2 seconds) = MB/s
    disk_io = disk_delta / (1024 * 1024 * 2)

    # Network: bytes delta since last call / (1KB * 2 seconds) = KB/s
    network = net_delta / (1024 * 2)

    # Response time: estimated composite metric
    response_time = 50 + cpu * 1.5 + disk_io * 5
```

### Failure Injection:

```python
# Global flags
_inject_mode = False
_inject_remaining = 0

def _generate_failure_metrics():
    """Simulated crisis values for demo."""
    return {
        "cpu": ~92%,                # Very high CPU
        "memory": ~90%,             # Very high memory
        "disk_io": ~70 MB/s,        # Heavy disk
        "response_time": ~350 ms,   # Slow response
        "network": ~200 KB/s,       # Heavy network
    }

# In the background loop:
async def _metric_loop():
    while True:
        if _inject_mode and _inject_remaining > 0:
            data = _generate_failure_metrics()  # Use fake crisis data
            _inject_remaining -= 1
        else:
            data = _collect_real_metrics()      # Use real psutil data
        _run_pipeline(data)
        await asyncio.sleep(2)
```

### All 13 API Endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Serve dashboard HTML |
| `GET` | `/health` | Latest health score, risk, failure prob, root cause |
| `GET` | `/metrics/recent` | Recent raw metrics (up to 60) |
| `GET` | `/health/history` | Health score time series |
| `GET` | `/health/forecast` | 60-point future projection |
| `GET` | `/alerts` | Alert log (up to 30) |
| `GET` | `/explain` | SHAP feature contributions (top 8) |
| `GET` | `/settings` | Current sensitivity value |
| `GET` | `/servers` | List of monitored server IDs |
| `POST` | `/ingest-metrics` | Manual metric submission |
| `POST` | `/settings/sensitivity` | Change sensitivity (1-10) |
| `POST` | `/simulate/failure` | Start failure injection |
| `POST` | `/simulate/stop` | Stop injection |

---

## 13. FILE 8: dashboard.html

**Purpose:** The dark-themed monitoring dashboard built with Bootstrap 5.

### Layout Structure:

```
┌─ Top Bar ──────────────────────────────────────────────────────┐
│  ⚡ Title    [Server ▼]  [Sensitivity ════]  [💥 Inject]  🟢  │
└────────────────────────────────────────────────────────────────┘
┌─ KPI Cards ───────────────────────────────────────────────────┐
│  CPU Usage │ Memory Usage │ Disk I/O │ Response Time │ Network│
│  14.1%     │ 65.3%        │ 2.4 MB/s │ 83 ms         │ 18 KB │
└────────────────────────────────────────────────────────────────┘
┌─ Gauges ──────────────────────────────────────────────────────┐
│ [Health Score  ] [Failure Prob. ] [Risk Badge  ] [Timeline    ]│
│ [Gauge: 95     ] [0.0%          ] [● GREEN     ] [●●●●●●●●●● ]│
│                  [──────●──────] [Anomaly Score] [            ]│
│                  [System nominal] [0.1879      ] [            ]│
└────────────────────────────────────────────────────────────────┘
┌─ SHAP + Forecast ─────────────────────────────────────────────┐
│ [SHAP Bar Chart ────────] [Health Score History + Forecast ───]│
│ [cpu_trend    ████       ] [═══════════════╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌ ]│
│ [cpu_latest   ███        ] [             forecast line -----→ ]│
└────────────────────────────────────────────────────────────────┘
┌─ Live Charts ─────────────────────────────────────────────────┐
│ [CPU & Memory Line Chart ─] [Disk I/O & Response Time Chart ─]│
└────────────────────────────────────────────────────────────────┘
┌─ Network + Alerts ────────────────────────────────────────────┐
│ [Network Chart ──────] [Alert Log Table ─────────────────────]│
│                       │ Time │ Severity │ Message             │
│                       │ 11:23│ CRITICAL │ Risk escalated...   │
└────────────────────────────────────────────────────────────────┘
```

### Key Design Elements:

- **Color scheme:** Dark navy (#0f1123) background, cyan/green/amber accents
- **Gauge:** HTML5 Canvas arc with gradient (red→amber→green)
- **Failure bar:** CSS gradient bar with white marker showing probability position
- **Timeline dots:** Green = normal, red = anomaly (with glow animation)
- **Risk badge:** Color-coded with CSS glow shadows

---

## 14. FILE 9: script.js

**Purpose:** Polls all API endpoints every 2 seconds and updates the dashboard UI.

### Polling Architecture:

```javascript
async function pollAll() {
    await Promise.all([
        pollHealth(),           // /health → gauges, risk badge, failure prob
        pollMetrics(),          // /metrics/recent → KPI cards + line charts
        pollHealthHistory(),    // /health/history + /health/forecast → health chart
        pollAlerts(),           // /alerts → alert table
        pollShap(),             // /explain → SHAP bar chart
    ]);
}

pollAll();                      // Initial load
setInterval(pollAll, 2000);     // Then every 2 seconds
setInterval(pollServers, 15000); // Refresh server list every 15s
```

### Chart Setup:

5 Chart.js charts are created:
1. **cpuMemChart** — Line chart: CPU% (cyan) + Memory% (purple)
2. **diskRespChart** — Line chart: Disk I/O (blue) + Response Time (amber)
3. **netChart** — Area chart: Network KB/s (green)
4. **healthChart** — Line chart: Health score (green solid) + Forecast (amber dashed)
5. **shapChart** — Horizontal bar chart: Feature contributions (red=anomaly, green=normal)

### Injection Toggle Button:

```javascript
let injecting = false;
async function injectFailure() {
    if (!injecting) {
        await fetch('/simulate/failure?count=15', { method: 'POST' });
        btn.textContent = '⏹ Stop Injection';   // Button changes
        injecting = true;
    } else {
        await fetch('/simulate/stop', { method: 'POST' });
        btn.textContent = '💥 Inject Failure';   // Button reverts
        injecting = false;
    }
}
```

---

## 15. THE 7 UPGRADES — DETAILED BREAKDOWN

### Upgrade 1: Failure Probability Predictor
| Aspect | Detail |
|--------|--------|
| **Model** | Logistic Regression (Scikit-learn) |
| **Features** | 8 (anomaly freq, health slope, mean health, min health, CPU/mem trends/means) |
| **Training Data** | 500 normal + 500 failure synthetic scenarios |
| **Output** | 0.0 (safe) to 1.0 (imminent failure) |
| **Endpoint** | Included in `/health` response |
| **UI** | Large percentage display + gradient bar with marker |

### Upgrade 2: SHAP Explainability
| Aspect | Detail |
|--------|--------|
| **Library** | SHAP (TreeExplainer for Isolation Forest) |
| **Output** | Top 8 feature contributions with name + value + contribution |
| **Interpretation** | Negative contribution = pushes toward anomaly |
| **Endpoint** | `GET /explain` |
| **UI** | Horizontal bar chart (red = toward anomaly, green = toward normal) |

### Upgrade 3: Sensitivity Slider
| Aspect | Detail |
|--------|--------|
| **Range** | 1 (loose) to 10 (strict) |
| **Math** | Multiplier = 0.6 + (sensitivity/10) × 0.8, divides anomaly score |
| **Effect** | Higher sensitivity = health score drops more for same anomaly |
| **Endpoint** | `POST /settings/sensitivity?value=N` |
| **UI** | Slider in top bar with live value display |

### Upgrade 4: Failure Pattern Injection
| Aspect | Detail |
|--------|--------|
| **Injected Values** | CPU ~92%, Memory ~90%, Disk ~70 MB/s, Response ~350 ms |
| **Duration** | Configurable count (default 15 = 30 seconds of fake data) |
| **Effect** | Health drops, failure prob rises, risk goes RED, alert fires |
| **Endpoint** | `POST /simulate/failure?count=N` |
| **UI** | Toggle button that changes text on click |

### Upgrade 5: Health Trend Forecast
| Aspect | Detail |
|--------|--------|
| **Method** | Linear regression (numpy polyfit degree 1) on last 20 health scores |
| **Output** | 60-point projection (2 minutes at 2s intervals) + slope |
| **Endpoint** | `GET /health/forecast` |
| **UI** | Dashed amber line extending from the health history chart |

### Upgrade 6: Multi-Server Support
| Aspect | Detail |
|--------|--------|
| **Implementation** | `server_id` column on all 3 database tables |
| **Default** | "local" for the current machine |
| **Filtering** | All endpoints accept `?server_id=X` query parameter |
| **Endpoint** | `GET /servers` returns list of known server IDs |
| **UI** | Dropdown selector in top bar |

### Upgrade 7: Root Cause Hints
| Aspect | Detail |
|--------|--------|
| **Logic** | Rule-based thresholds per metric, sorted by severity |
| **Examples** | "High CPU (93%) — Check compute-heavy processes" |
| **Storage** | Stored in health_records table, included in alerts |
| **Endpoint** | Included in `/health` response as `root_cause` field |
| **UI** | Amber text below failure probability display |

---

## 16. HOW TO RUN

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install psutil shap

# 2. Train models (collects 60s of real data from your PC)
python -m backend.train_model

# 3. Start server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 4. Open browser
# http://localhost:8000/
```

---

## 17. VERIFICATION RESULTS

### Timestamp Sync Test
```
API Timestamp: 2026-02-24T11:22:45
PC Time:       2026-02-24T11:22:45
Result: ✅ EXACT MATCH
```

### Normal Operation
```
Health Score: 84-95
Risk Level: GREEN
Failure Probability: 0%
Anomaly Flag: False
Root Cause: "System nominal" or "High Memory (65%)"
```

### Failure Injection Test
```
Before:  Health=84, Risk=GREEN, Failure=0%
After:   Health=39, Risk=RED,   Failure=100%
Alert:   CRITICAL — "Risk escalated to RED — High CPU (93%)"
```

### All 7 Upgrades Verified
| # | Feature | Status |
|---|---------|--------|
| 1 | Failure Probability | ✅ 0% normal, 100% on injection |
| 2 | SHAP Explainability | ✅ 8 features returned |
| 3 | Sensitivity Slider | ✅ Changed 5→8 successfully |
| 4 | Failure Injection | ✅ Health dropped, CRITICAL alert fired |
| 5 | Health Forecast | ✅ 60-point projection, slope -0.129 |
| 6 | Multi-Server | ✅ Returns ["local"] |
| 7 | Root Cause Hints | ✅ Correct diagnosis on injection |
