# ═══════════════════════════════════════════════════════════════════
# AI-BASED SYSTEM FAILURE EARLY WARNING ENGINE
# COMPLETE PROJECT GUIDE — Architecture, Logic & Code Walkthrough
# v4.0
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
9. [database.py — ORM](#9-databasepy)
10. [train_model.py — Real Data Training](#10-train_modelpy)
11. [drift_detector.py — Concept Drift](#11-drift_detectorpy)
12. [retraining.py — Model Lifecycle](#12-retrainingpy)
13. [auth.py — API Authentication](#13-authpy)
14. [logging_config.py — Structured Logging](#14-logging_configpy)
15. [captcha.py — CAPTCHA Verification](#15-captchapy)
16. [main.py — FastAPI Server (13 Endpoints)](#16-mainpy)
17. [Frontend Dashboard](#17-frontend-dashboard)
18. [All Features Summary](#18-all-features)
19. [How to Run](#19-how-to-run)
20. [Verification Results](#20-verification-results)

---

## 1. PROJECT OVERVIEW

This is an **AI-powered system monitoring platform** that:

- **Monitors** real system metrics (CPU, Memory, Disk I/O, Process Count, Network) via psutil every 1 second
- **Detects anomalies** using Isolation Forest (unsupervised ML, 200 trees)
- **Predicts failures** using Logistic Regression (0–100% probability)
- **Explains decisions** with SHAP (which features caused the anomaly?)
- **Scores confidence** (0–1 normalized from historical score distribution)
- **Detects concept drift** (is the model becoming stale?)
- **Self-retrains** on recent data with model versioning
- **Authenticates** API access with X-API-KEY headers
- **Logs everything** to rotating structured log files
- **Falls back gracefully** to rule-based detection if ML model fails

**All ML runs locally using Scikit-learn — no external AI APIs.**

---

## 2. ARCHITECTURE & DATA FLOW

```
┌──────────────────────────────────────────────────────────────────┐
│                    EVERY 1 SECOND                                │
│                                                                  │
│  psutil → Feature Engineering (30 features)                     │
│         → Isolation Forest (anomaly score + confidence)         │
│         → SHAP Explainer (top 8 feature contributions)          │
│         → Risk Engine (health 0-100 + severity penalty)         │
│         → Failure Predictor (probability 0-100%)                │
│         → Drift Detector (rolling anomaly rate tracking)        │
│         → Root Cause Diagnosis (human-readable hint)            │
│         → SQLite Storage (3 tables)                             │
│         → Alert Generation (if risk escalated)                  │
│         → Structured Logging (logs/system.log)                  │
│                                                                  │
│  FastAPI (10 endpoints + auth)                                   │
│         → Dashboard (Chart.js + Bootstrap 5, 1s polling)        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. PROJECT STRUCTURE

```
d:\PROJECTS\SYSTEM\
├── backend/                          # 12 Python modules
│   ├── __init__.py                   # Package marker
│   ├── main.py                       # FastAPI app, 13 endpoints, pipeline
│   ├── database.py                   # SQLite ORM (3 tables + CRUD)
│   ├── feature_engineering.py        # Raw metrics → 30 features
│   ├── model.py                      # Isolation Forest + SHAP + confidence
│   ├── risk_engine.py                # Health score + risk + root cause + fallback
│   ├── failure_predictor.py          # Logistic Regression failure probability
│   ├── train_model.py               # Training script (60s real data)
│   ├── drift_detector.py            # Concept drift detection
│   ├── retraining.py                # Model retraining + versioning
│   ├── captcha.py                   # Math CAPTCHA generation + verification
│   ├── auth.py                      # API key authentication
│   └── logging_config.py            # Structured logging config
├── frontend/
│   ├── dashboard.html               # Real-time monitoring UI
│   ├── captcha.html                 # CAPTCHA verification page
│   └── script.js                    # Chart.js + 1s live polling
├── models/                          # Trained models + versioned backups
├── logs/                            # Structured logs (auto-created)
├── data/                            # SQLite database (auto-created)
├── Dockerfile                       # Docker container configuration
├── .dockerignore                    # Docker build exclusions
├── .gitignore                       # Git-tracked file exclusions
├── .gitattributes                   # Git line-ending configuration
└── requirements.txt                 # Python dependencies
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
psutil                 # System metrics (CPU, Memory, Disk, Process Count, Network)
shap                   # SHAP explainability
```

---

## 5. feature_engineering.py

**Purpose:** Converts a window of 5 raw metric readings into a 30-dimensional feature vector.

**5 signals × 6 features each = 30 total:**

| Signal | What It Measures |
|--------|-----------------|
| `cpu` | CPU usage percentage |
| `memory` | RAM usage percentage |
| `disk_io` | Disk read+write rate (MB/s) |
| `process_count` | Number of running processes |
| `network` | Network throughput (KB/s) |

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

**Health Score Mapping:**
```
anomaly_score >= 0.15  → health 92-100 (clearly normal)
anomaly_score >= 0.05  → health 85-92
anomaly_score >= 0.0   → health 70-85  (borderline)
anomaly_score >= -0.1  → health 50-70  (mild anomaly)
anomaly_score >= -0.25 → health 25-50  (moderate)
anomaly_score < -0.25  → health 0-25   (severe)
- severity penalty subtracted for extreme raw values (up to -40)
```

**Severity penalties:**
- CPU > 90% → penalty 12, > 80% → 7, > 75% → 3
- Memory > 88% → penalty 10, > 78% → 5, > 72% → 2
- Disk I/O > 80 MB/s → penalty 8, > 50 → 4
- Process Count > 400 → penalty 10, > 300 → 5, > 250 → 2

**Rule-based fallback:**
```python
def rule_based_anomaly_check(metrics) -> tuple[score, is_anomaly, health]:
    # Used when ML model is unavailable
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

**3 tables:**

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `raw_metrics` | Store psutil readings | cpu, memory, disk_io, process_count, network |
| `health_records` | Pipeline output | health_score, anomaly_score, failure_prob, confidence, model_status |
| `alerts` | Risk escalation events | severity (CRITICAL/WARNING), message |

---

## 10. train_model.py

**Training pipeline:**
1. Collects 60 seconds of real psutil data (~120 samples at 0.5s intervals)
2. Uses `time.monotonic()` for accurate disk I/O (MB/s) and network (KB/s) rate calculations
3. Captures real process count via `psutil.pids()`
4. Augments with gaussian noise to 3000 samples
5. Computes 30 features per sample
6. Trains Isolation Forest (n_estimators=200, contamination=0.02)
7. Generates 500 normal + 500 failure synthetic scenarios
8. Trains Logistic Regression for failure probability

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

**Protected endpoints:** `/ingest-metrics`, `/model/retrain`, `/model/info`

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
- Auth failures
- CAPTCHA generation and verification

---

## 15. captcha.py

**Purpose:** Self-contained math CAPTCHA system — no external APIs needed.

**How it works:**
1. `generate_captcha()` creates a random math problem (addition, subtraction, or multiplication)
2. Returns a `token` and `question` string (e.g., `"11 + 7 = ?"`)
3. User submits `token` + `answer` to `verify_captcha()`
4. On correct answer, a session token is issued (stored in cookie for 24h)
5. Dashboard route checks this cookie — if invalid/missing, redirects to `/captcha`

**Configuration:**
```python
_CAPTCHA_TTL = 300     # Unsolved CAPTCHAs expire after 5 minutes
_VERIFIED_TTL = 86400  # Session lasts 24 hours
```

**Operations:** Addition, subtraction (never negative), multiplication. Numbers range 2–20.

---

## 16. main.py

**The heart of the system — 13 endpoints + pipeline + metric collection.**

**Pipeline flow (every 1 second):**
```
collect_real_metrics() → insert_metric() → compute_features()
→ predict() [with confidence] → compute_health_score()
→ explain() → drift_detector.record() → evaluate_risk()
→ diagnose_root_cause() → predict_failure_probability()
→ insert_health_record() → fire alert if escalated
```

**Metric collection accuracy:**
- Uses `time.monotonic()` to calculate elapsed time between readings
- Disk I/O rate = bytes delta / (1024² × elapsed seconds) → accurate MB/s
- Network rate = bytes delta / (1024 × elapsed seconds) → accurate KB/s
- Process count = `len(psutil.pids())` → real OS process count

**13 Endpoints:**

| # | Method | Path | Auth | Purpose |
|---|--------|------|------|---------|
| 1 | GET | `/` | CAPTCHA | Dashboard HTML |
| 2 | GET | `/captcha` | No | CAPTCHA verification page |
| 3 | GET | `/captcha/generate` | No | Generate math CAPTCHA |
| 4 | POST | `/captcha/verify` | No | Verify answer, return session |
| 5 | GET | `/health` | No | Latest health + confidence + model_status |
| 6 | GET | `/metrics/recent` | No | Raw metric history |
| 7 | GET | `/health/history` | No | Health score time series |
| 8 | GET | `/health/forecast` | No | 60-point projection |
| 9 | GET | `/explain` | No | SHAP contributions |
| 10 | GET | `/alerts` | No | Alert log |
| 11 | GET | `/model/drift-status` | No | Drift detection |
| 12 | POST | `/ingest-metrics` | API Key | Manual metric input |
| 13 | POST | `/model/retrain` | API Key | Trigger retraining |

---

## 17. FRONTEND DASHBOARD

**CAPTCHA page:**
- Glassmorphism card with animated background orbs
- Random math challenge (addition, subtraction, multiplication)
- Correct answer sets session cookie and redirects to dashboard

**Dashboard elements:**
- **Top bar:** ML status pill, drift warning badge, live indicator (1s polling)
- **KPI cards:** CPU%, Memory%, Disk I/O (MB/s), Process Count, Network (KB/s)
- **Gauges:** Health score (canvas arc), Failure probability (gradient bar), Confidence score
- **Charts:** SHAP bar chart, Health history + forecast, CPU/Memory, Disk I/O/Processes, Network
- **Alert log:** Timestamped severity + message table

---

## 18. ALL FEATURES

| # | Feature | Module |
|---|---------|--------|
| 1 | Real-time psutil monitoring (1s interval) | main.py |
| 2 | Accurate rate calculations (disk I/O MB/s, network KB/s) | main.py, train_model.py |
| 3 | Real process count monitoring | main.py |
| 4 | 30-feature engineering | feature_engineering.py |
| 5 | Isolation Forest anomaly detection | model.py |
| 6 | SHAP explainability | model.py |
| 7 | Health score (0-100) | risk_engine.py |
| 8 | Risk levels (GREEN/YELLOW/RED) | risk_engine.py |
| 9 | Failure probability (0-100%) | failure_predictor.py |
| 10 | Health trend forecast | main.py |
| 11 | Root cause hints | risk_engine.py |
| 12 | Confidence scoring (0-1) | model.py |
| 13 | Concept drift detection | drift_detector.py |
| 14 | Model retraining + versioning | retraining.py |
| 15 | CAPTCHA dashboard gate | captcha.py |
| 16 | API authentication | auth.py |
| 17 | Structured logging | logging_config.py |
| 18 | Graceful degradation | risk_engine.py |

---

## 19. HOW TO RUN

```bash
# Install
pip install -r requirements.txt

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

## 20. VERIFICATION RESULTS

### Normal Operation
```
Health: 90, Risk: GREEN, Confidence: 0.05, Failure: 0%
Model: active, Drift: false, Anomaly rate: 3%
Metrics: CPU 20%, Memory 80%, Disk 0.4 MB/s, Processes 341, Network 1.3 KB/s
```

### Under Stress
```
Health: 30, Risk: RED, Confidence: 0.79, Failure: 100%
Root cause: High Memory (88%) — Possible memory leak or cache bloat
Alert: CRITICAL — Risk escalated to RED
```

### Structured Logging
```
2026-03-07 20:59:03 | INFO  | main  | Engine started
2026-03-07 20:59:04 | INFO  | model | Model loaded in 0.028s
2026-03-07 20:59:42 | WARN  | main  | ALERT [CRITICAL]: Risk escalated to RED
```
