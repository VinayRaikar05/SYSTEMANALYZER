# ═══════════════════════════════════════════════════════════════════
# AI-BASED SYSTEM FAILURE EARLY WARNING ENGINE
# COMPLETE PROJECT GUIDE — Architecture, Logic & Code Walkthrough
# Enterprise Edition v3.0
# ═══════════════════════════════════════════════════════════════════

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Architecture & Data Flow](#2-architecture--data-flow)
3. [Project Structure](#3-project-structure)
4. [Dependencies](#4-dependencies)
5. [feature_engineering.py — 30 ML Features](#5-feature_engineeringpy)
6. [model.py — Anomaly Detection + Confidence + SHAP](#6-modelpy)
7. [risk_engine.py — Health Score + Root Cause + Fallback](#7-risk_enginepy)
8. [failure_predictor.py — Failure Probability](#8-failure_predictorpy)
9. [database.py — ORM + ServerConfig](#9-databasepy)
10. [train_model.py — Real Data Training](#10-train_modelpy)
11. [drift_detector.py — Concept Drift](#11-drift_detectorpy)
12. [retraining.py — Model Lifecycle](#12-retrainingpy)
13. [auth.py — API Authentication](#13-authpy)
14. [logging_config.py — Structured Logging](#14-logging_configpy)
15. [performance_monitor.py — Observability](#15-performance_monitorpy)
16. [remediation.py — Automated Response](#16-remediationpy)
17. [main.py — FastAPI Server (18 Endpoints)](#17-mainpy)
18. [Frontend Dashboard](#18-frontend-dashboard)
19. [The 17 Features Summary](#19-all-features)
20. [How to Run](#20-how-to-run)
21. [Verification Results](#21-verification-results)

---

## 1. PROJECT OVERVIEW

This is an **enterprise-grade AI observability platform** that:

- **Monitors** real system metrics (CPU, Memory, Disk, Network) via psutil every 2 seconds
- **Detects anomalies** using Isolation Forest (unsupervised ML, 200 trees)
- **Predicts failures** using Logistic Regression (0–100% probability)
- **Explains decisions** with SHAP (which features caused the anomaly?)
- **Scores confidence** (0–1 normalized from historical score distribution)
- **Detects concept drift** (is the model becoming stale?)
- **Self-retrains** on recent data with model versioning
- **Authenticates** API access with X-API-KEY headers
- **Logs everything** to rotating structured log files
- **Monitors its own performance** (inference latency, memory usage)
- **Falls back gracefully** to rule-based detection if ML model fails
- **Triggers remediation** hooks when risk is critical

**All ML runs locally using Scikit-learn — no external AI APIs.**

---

## 2. ARCHITECTURE & DATA FLOW

```
┌──────────────────────────────────────────────────────────────────┐
│                    EVERY 2 SECONDS                               │
│                                                                  │
│  psutil → Feature Engineering (30 features)                     │
│         → Isolation Forest (anomaly score + confidence)         │
│         → SHAP Explainer (top 8 feature contributions)          │
│         → Risk Engine (health 0-100 + severity penalty)         │
│         → Failure Predictor (probability 0-100%)                │
│         → Drift Detector (rolling anomaly rate tracking)        │
│         → Root Cause Diagnosis (human-readable hint)            │
│         → SQLite Storage (4 tables, server_id indexed)          │
│         → Alert Generation (if risk escalated)                  │
│         → Structured Logging (logs/system.log)                  │
│         → Performance Tracking (latency, req/min)               │
│                                                                  │
│  FastAPI (18 endpoints + auth middleware)                        │
│         → Dashboard (Chart.js + Bootstrap 5)                    │
│         → Enterprise Status Bar (drift, latency, retrain btn)  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. PROJECT STRUCTURE

```
d:\PROJECTS\SYSTEM\
├── backend/                          # 13 Python modules
│   ├── __init__.py                   # Package marker
│   ├── main.py                       # FastAPI app, 18 endpoints, pipeline
│   ├── database.py                   # SQLite ORM (4 tables + CRUD)
│   ├── feature_engineering.py        # Raw metrics → 30 features
│   ├── model.py                      # Isolation Forest + SHAP + confidence
│   ├── risk_engine.py                # Health score + risk + root cause + fallback
│   ├── failure_predictor.py          # Logistic Regression failure probability
│   ├── train_model.py               # Training script (60s real data)
│   ├── drift_detector.py            # Concept drift detection
│   ├── retraining.py                # Model retraining + versioning
│   ├── auth.py                      # API key authentication
│   ├── logging_config.py            # Structured logging config
│   ├── performance_monitor.py       # Latency + request tracking
│   └── remediation.py               # Automated remediation hooks
├── frontend/
│   ├── dashboard.html               # Enterprise monitoring UI
│   └── script.js                    # Chart.js + enterprise polling
├── models/                          # Trained models + versioned backups
├── logs/                            # Structured logs (auto-created)
├── data/                            # SQLite database (auto-created)
└── requirements.txt
```

---

## 4. DEPENDENCIES

```
fastapi>=0.115.0       # Web framework
uvicorn[standard]      # ASGI server
scikit-learn>=1.5.0    # ML: Isolation Forest, Logistic Regression
pandas>=2.2.0          # Data manipulation
numpy>=1.26.0          # Numerical computing
joblib>=1.4.0          # Model serialization
aiosqlite>=0.20.0      # Async SQLite
sqlalchemy[asyncio]    # ORM
psutil                 # System metrics (CPU, Memory, Disk, Network)
shap                   # SHAP explainability
```

---

## 5. feature_engineering.py

**Purpose:** Converts a window of 5 raw metric readings into a 30-dimensional feature vector.

**5 signals × 6 features each = 30 total:**

| Feature | What It Measures | Why It Matters |
|---------|-----------------|----------------|
| `latest_value` | Current raw value | Catches sustained high load |
| `rolling_mean` | Average of last 5 | Baseline context |
| `rolling_var` | Variance over window | Stability measure |
| `trend_slope` | Linear regression slope | Is it getting worse? |
| `spike_mag` | Deviation from mean | Sudden jumps |
| `rate_of_change` | Delta from previous | Acceleration |

**Key function:**
```python
def compute_features(rows: list[dict]) -> np.ndarray | None:
    # Returns shape (1, 30) or None if < 5 rows
```

---

## 6. model.py

**Purpose:** ML prediction with confidence scoring and graceful degradation.

**Key changes in Enterprise Edition:**

```python
def predict(features) -> tuple[float, bool, float]:
    # Returns: (anomaly_score, is_anomaly, confidence)
    # confidence = normalized 0-1 using historical min/max scaling
    # If model missing/broken → returns (0.0, False, 0.0) instead of crashing

def explain(features) -> list[dict]:
    # SHAP TreeExplainer → top 8 feature contributions

def get_model_info() -> dict:
    # Returns: model_status, version, feature_count, contamination,
    #          load_time_ms, confidence_samples
```

**Confidence normalization logic:**
```python
# Invert: lower score = higher confidence of anomaly
# Uses min-max scaling from last 500 predictions
normalized = (s_max - score) / (s_max - s_min)  # 0=normal, 1=anomaly
```

**Graceful degradation:** If `isolation_model.pkl` is missing or `predict()` throws, returns neutral values and sets `model_status = "fallback_mode"`. The pipeline then switches to `rule_based_anomaly_check()`.

---

## 7. risk_engine.py

**Purpose:** Maps ML output to human-understandable health scores, risk levels, and root cause hints.

**Health Score Mapping (sensitivity-adjusted):**
```
anomaly_score >= 0.15  → health 92-100 (clearly normal)
anomaly_score >= 0.05  → health 85-92
anomaly_score >= 0.0   → health 70-85  (borderline)
anomaly_score >= -0.1  → health 50-70  (mild anomaly)
anomaly_score >= -0.25 → health 25-50  (moderate)
anomaly_score < -0.25  → health 0-25   (severe)
- severity penalty subtracted for extreme raw values (up to -40)
```

**Rule-based fallback (Upgrade 9):**
```python
def rule_based_anomaly_check(metrics) -> tuple[score, is_anomaly, health]:
    # Used when ML model is unavailable
    # Checks: CPU>90 → penalty 12, Memory>88 → penalty 10, etc.
    # Returns pseudo-score, anomaly flag, and rule-based health score
```

---

## 8. failure_predictor.py

**8 features for failure prediction:**

| Feature | Source | Meaning |
|---------|--------|---------|
| `anomaly_freq` | Recent flags | % of anomalous readings |
| `health_slope` | Health scores | Negative = declining |
| `mean_health` | Health scores | Average level |
| `min_health` | Health scores | Worst case |
| `cpu_trend` | Raw metrics | CPU going up? |
| `cpu_mean` | Raw metrics | Average CPU |
| `mem_trend` | Raw metrics | Memory going up? |
| `mem_mean` | Raw metrics | Average memory |

**Output:** `predict_proba()[0][1]` → probability of class 1 (failure), range 0.0–1.0

---

## 9. database.py

**4 tables (Enterprise Edition adds `ServerConfig`):**

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `raw_metrics` | Store psutil readings | cpu, memory, disk_io, response_time, network |
| `health_records` | Pipeline output | health_score, anomaly_score, failure_prob, **confidence**, **model_status** |
| `alerts` | Risk escalation events | severity (CRITICAL/WARNING), message |
| `server_configs` | **Per-server settings** | server_id, sensitivity |

All tables indexed by `server_id` for multi-server filtering.

---

## 10. train_model.py

**Training pipeline:**
1. Collects 60 seconds of real psutil data (~118 samples)
2. Augments with gaussian noise to 3000 samples
3. Computes 30 features per sample
4. Trains Isolation Forest (n_estimators=200, contamination=0.02)
5. Generates 500 normal + 500 failure synthetic scenarios
6. Trains Logistic Regression for failure probability

---

## 11. drift_detector.py

**Purpose:** Detects concept drift — when the model's learned "normal" no longer matches reality.

**How it works:**
```python
WINDOW_SIZE = 200        # Track last 200 predictions
DRIFT_THRESHOLD = 0.06   # contamination (0.02) × 3
SUSTAINED_SECONDS = 120  # Must persist for 2 minutes

# Every prediction: drift_detector.record(is_anomaly, health_score)
# If anomaly_rate > 6% for 2+ continuous minutes → DRIFT CONFIRMED
```

**Endpoint:** `GET /model/drift-status` returns:
```json
{
  "drift_detected": false,
  "anomaly_rate": 0.03,
  "expected_rate": 0.02,
  "drift_threshold": 0.06,
  "samples_tracked": 200
}
```

---

## 12. retraining.py

**Purpose:** Retrain the Isolation Forest on recent normal data from the database.

**Process:**
1. Fetch last 2000 raw metrics from DB
2. Augment with noise to 3000 samples
3. Compute features and train new Isolation Forest
4. Back up current model with timestamp (keep last 2 backups)
5. Save new model, force reload in memory
6. Log retraining event

**Endpoint:** `POST /model/retrain`

---

## 13. auth.py

**Purpose:** Lightweight API key authentication.

```python
# Keys from environment variable (comma-separated)
# API_KEYS=key1,key2,key3

# If no keys configured → auth is DISABLED (dev mode)
# If keys configured → X-API-KEY header required on protected endpoints
# Returns 401 for unauthorized access
```

**Protected endpoints:** All POST + `/alerts`, `/model/info`, `/system/performance`

---

## 14. logging_config.py

**Purpose:** Replace all `print()` with structured Python logging.

```python
# Format: 2026-02-24 23:00:03 | INFO | model | Model loaded in 0.028s
# File: logs/system.log (rotating, 5MB max, 3 backups)
# Levels: DEBUG (file only), INFO (file + console), WARNING, ERROR
```

**What gets logged:**
- Model loading and prediction errors
- Risk escalations and alert triggers
- Retraining events
- Drift detection warnings
- Sensitivity changes
- Auth failures
- Remediation actions

---

## 15. performance_monitor.py

**Tracks:**
| Metric | How | Window |
|--------|-----|--------|
| Inference latency | `time.time()` around pipeline | Last 100 calls |
| Requests per minute | Timestamp deque | Last 60 seconds |
| App memory | `psutil.Process().memory_info()` | Current |

**Endpoint:** `GET /system/performance` returns:
```json
{
  "avg_inference_latency_ms": 36.2,
  "max_inference_latency_ms": 54.1,
  "min_inference_latency_ms": 23.5,
  "requests_per_minute": 4,
  "app_memory_mb": 276.0
}
```

---

## 16. remediation.py

**Purpose:** Simulate automated remediation when risk is RED.

**3 simulated actions:**
1. **Service restart** — Graceful restart of application service
2. **Resource scale-up** — Request +2 CPU cores and +4GB RAM
3. **Cleanup** — Clear temp files, rotate logs, flush caches

All actions logged to `logs/system.log`. Returns JSON with action details.

---

## 17. main.py

**The heart of the system — 18 endpoints + pipeline + metric collection.**

**Pipeline flow (every 2 seconds):**
```
collect_real_metrics() → insert_metric() → compute_features()
→ predict() [with confidence] → compute_health_score() [with sensitivity]
→ explain() → drift_detector.record() → evaluate_risk()
→ diagnose_root_cause() → predict_failure_probability()
→ insert_health_record() → fire alert if escalated
→ record_inference_latency()
```

**18 Endpoints:**

| # | Method | Path | Auth | Purpose |
|---|--------|------|------|---------|
| 1 | GET | `/` | No | Dashboard HTML |
| 2 | GET | `/health` | No | Latest health + confidence + model_status |
| 3 | GET | `/metrics/recent` | No | Raw metric history |
| 4 | GET | `/health/history` | No | Health score time series |
| 5 | GET | `/health/forecast` | No | 60-point projection |
| 6 | GET | `/explain` | No | SHAP contributions |
| 7 | GET | `/servers` | No | Server IDs |
| 8 | GET | `/settings` | No | Per-server sensitivity |
| 9 | GET | `/model/drift-status` | No | Drift detection |
| 10 | POST | `/ingest-metrics` | Yes | Manual metric input |
| 11 | POST | `/settings/sensitivity` | Yes | Change sensitivity |
| 12 | POST | `/simulate/failure` | Yes | Inject failure spikes |
| 13 | POST | `/simulate/stop` | Yes | Stop injection |
| 14 | POST | `/model/retrain` | Yes | Trigger retraining |
| 15 | POST | `/remediation/trigger` | Yes | Run remediation |
| 16 | GET | `/model/info` | Yes | Model metadata |
| 17 | GET | `/system/performance` | Yes | Performance metrics |
| 18 | GET | `/alerts` | Yes | Alert log |

---

## 18. FRONTEND DASHBOARD

**Enterprise dashboard elements:**
- **Top bar:** Server dropdown, sensitivity slider, inject button, ML status pill, drift warning badge
- **KPI cards:** CPU%, Memory%, Disk I/O, Response Time, Network
- **Gauges:** Health score (canvas arc), Failure probability (gradient bar), Confidence score
- **Charts:** SHAP bar chart, Health history + forecast, CPU/Memory, Disk/Response, Network
- **Alert log:** Timestamped severity + message table
- **Enterprise bar:** Model status, version, drift status, anomaly rate, avg latency, req/min, app memory, retrain button

---

## 19. ALL FEATURES

| # | Feature | Type | Module |
|---|---------|------|--------|
| 1 | Real-time psutil monitoring | Core | main.py |
| 2 | 30-feature engineering | Core | feature_engineering.py |
| 3 | Isolation Forest anomaly detection | Core | model.py |
| 4 | SHAP explainability | Upgrade 2 | model.py |
| 5 | Health score (0-100) | Core | risk_engine.py |
| 6 | Risk levels (GREEN/YELLOW/RED) | Core | risk_engine.py |
| 7 | Failure probability (0-100%) | Upgrade 1 | failure_predictor.py |
| 8 | Sensitivity slider (1-10) | Upgrade 3 | risk_engine.py |
| 9 | Failure injection mode | Upgrade 4 | main.py |
| 10 | Health trend forecast | Upgrade 5 | main.py |
| 11 | Multi-server support | Upgrade 6 | database.py |
| 12 | Root cause hints | Upgrade 7 | risk_engine.py |
| 13 | Confidence scoring (0-1) | Enterprise | model.py |
| 14 | Concept drift detection | Enterprise | drift_detector.py |
| 15 | Model retraining + versioning | Enterprise | retraining.py |
| 16 | API authentication | Enterprise | auth.py |
| 17 | Structured logging | Enterprise | logging_config.py |
| 18 | Performance monitoring | Enterprise | performance_monitor.py |
| 19 | Model metadata endpoint | Enterprise | model.py |
| 20 | Graceful degradation | Enterprise | risk_engine.py |
| 21 | Per-server sensitivity | Enterprise | database.py |
| 22 | Automated remediation | Enterprise | remediation.py |

---

## 20. HOW TO RUN

```bash
# Install
pip install -r requirements.txt
pip install psutil shap

# Train (60 seconds of real data collection)
python -m backend.train_model

# Start
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Open http://localhost:8000/

# Optional: Enable API auth
set API_KEYS=my-secret-key
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## 21. VERIFICATION RESULTS

### Normal Operation
```
Health: 90, Risk: GREEN, Confidence: 0.05, Failure: 0%
Model: active, Drift: false, Anomaly rate: 3%
Performance: 36ms avg latency, 276MB memory
```

### Under Failure Injection
```
Health: 42, Risk: RED, Confidence: 0.85, Failure: 100%
Drift: anomaly rate jumped to 10%
Alert: CRITICAL — Risk escalated to RED
```

### Structured Logging
```
2026-02-24 22:59:55 | INFO  | main  | Engine started
2026-02-24 23:00:03 | INFO  | model | Model loaded in 0.028s
2026-02-24 23:11:18 | WARN  | main  | ALERT [CRITICAL]: Risk escalated to RED
2026-02-24 23:15:11 | INFO  | main  | Sensitivity changed to 8 for server local
```
