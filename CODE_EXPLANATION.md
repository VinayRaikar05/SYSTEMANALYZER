# Code Files Explanation
## AI-Based System Failure Early Warning Engine

---

## Backend (Python)

### `main.py` — Application Core
The FastAPI server and central orchestrator. It runs a **1-second metric collection loop** using `psutil`, pipes data through the full ML pipeline (feature engineering → anomaly detection → risk assessment → failure prediction), stores results in SQLite, and serves 13 REST API endpoints. Also handles CAPTCHA-gated dashboard serving and static file routing.

**Key functions:**
- `_collect_real_metrics()` — Reads CPU, memory, disk I/O, process count, and network using `psutil` with elapsed-time-based rate calculations
- `_run_pipeline(data)` — Runs the complete ML inference pipeline: feature computation → Isolation Forest prediction → SHAP explanation → health scoring → risk evaluation → failure prediction → alert generation
- `_metric_loop()` — Async background task that calls `_collect_real_metrics()` every 1 second

---

### `database.py` — Data Layer
SQLAlchemy ORM for SQLite with 3 tables:
- **`raw_metrics`** — Stores every psutil reading (cpu, memory, disk_io, process_count, network)
- **`health_records`** — Pipeline output per tick (health_score, anomaly_score, risk_level, failure_prob, confidence, root_cause)
- **`alerts`** — Risk escalation events (CRITICAL/WARNING with timestamped messages)

Provides CRUD helpers: `insert_metric()`, `get_recent_metrics()`, `insert_health_record()`, `get_recent_health()`, `insert_alert()`, `get_recent_alerts()`.

---

### `feature_engineering.py` — ML Feature Extraction
Transforms a sliding window of 5 raw metric readings into a **30-dimensional feature vector** (5 signals × 6 features each). For each signal (cpu, memory, disk_io, process_count, network), it computes:
- Latest value, rolling mean, rolling variance, linear trend slope, spike magnitude, rate of change

This rich feature set lets the Isolation Forest detect both sudden spikes and gradual degradation patterns.

---

### `model.py` — ML Prediction & Explainability
Loads the trained Isolation Forest model and provides:
- **`predict(features)`** — Returns anomaly score, is_anomaly flag, and normalized confidence (0-1)
- **`explain(features)`** — SHAP TreeExplainer showing which features contributed most to the anomaly score (top 8)
- **`get_model_info()`** — Model metadata (version, status, feature count, load time)

Includes **graceful degradation** — if the model file is missing or broken, returns neutral values and sets `model_status = "fallback_mode"` so the pipeline switches to rule-based detection.

---

### `risk_engine.py` — Health Scoring & Root Cause
Converts raw ML output into human-understandable assessments:
- **`compute_health_score()`** — Maps anomaly score to 0-100 health, with severity penalties for extreme metric values (e.g., CPU>90% adds penalty of 12)
- **`evaluate_risk()`** — Counts recent anomalies to determine GREEN/YELLOW/RED risk level
- **`diagnose_root_cause()`** — Returns a human-readable hint like "High CPU (90%) — Check compute-heavy processes"
- **`rule_based_anomaly_check()`** — Fallback when ML model is unavailable; uses threshold-based detection

---

### `failure_predictor.py` — Failure Probability
A Logistic Regression model that predicts the probability (0-100%) of imminent system failure. Uses 8 features: anomaly frequency, health score slope/mean/min, and CPU/memory trends. Returns `predict_proba()` for class 1 (failure).

---

### `train_model.py` — Model Training Script
Run once before starting the server. Collects **60 seconds of real system metrics** from your machine, augments to 3000 samples with gaussian noise, computes 30 features, and trains:
1. **Isolation Forest** (200 trees, contamination=0.02) for anomaly detection
2. **Logistic Regression** for failure probability prediction

Uses `time.monotonic()` for accurate elapsed-time rate calculations.

---

### `drift_detector.py` — Concept Drift Detection
Monitors whether the model's definition of "normal" still matches reality. Tracks a rolling window of 200 predictions — if the anomaly rate exceeds 6% (3× the expected 2% contamination) for 2+ continuous minutes, drift is confirmed and retraining is recommended.

---

### `retraining.py` — Model Lifecycle Management
Retrains the Isolation Forest using recent normal data from the database. Backs up the current model with a timestamp, trains a new one, and force-reloads it into memory. Keeps the last 2 model versions for rollback safety.

---

### `captcha.py` — CAPTCHA Verification
Self-contained math CAPTCHA system (no external APIs). Generates random math problems (addition, subtraction, multiplication), issues one-time verification tokens, and manages 24-hour session cookies. The dashboard route checks for a valid session cookie before serving the page.

---

### `auth.py` — API Key Authentication
Lightweight API key authentication via `X-API-KEY` header. Keys are loaded from the `API_KEYS` environment variable (comma-separated). If no keys are configured, auth is disabled (dev mode). Protects POST endpoints like `/ingest-metrics` and `/model/retrain`.

---

### `logging_config.py` — Structured Logging
Configures Python's logging module with a rotating file handler. All events go to `logs/system.log` (5MB max, 3 backups) in the format: `2026-03-08 11:15:16 | INFO | main | Engine started`. Console output shows INFO and above; file captures DEBUG level too.

---

## Frontend (HTML/JS)

### `captcha.html` — CAPTCHA Verification Page
Glassmorphism-styled gate page with animated background orbs. Displays a random math challenge, validates the answer via the `/captcha/verify` API, sets a session cookie on success, and redirects to the dashboard.

---

### `dashboard.html` — Monitoring Dashboard
Bootstrap 5 dark-themed dashboard with:
- **KPI cards** — CPU%, Memory%, Disk I/O (MB/s), Process Count, Network (KB/s)
- **Health gauge** — Canvas-drawn arc gauge (0-100)
- **Failure probability** — Gradient bar with marker + root cause text
- **Risk badge** — Color-coded GREEN/YELLOW/RED with glow effects
- **Anomaly timeline** — 60-dot grid showing normal (green) vs anomaly (red) readings
- **Charts** — CPU/Memory, Disk I/O/Processes, Network, SHAP contributions, Health history + forecast

---

### `script.js` — Dashboard Logic
Handles all data fetching and UI updates with 1-second polling:
- `pollHealth()` — Updates health gauge, risk badge, failure probability, confidence, model status
- `pollMetrics()` — Updates KPI cards and metric charts
- `pollHealthHistory()` — Updates health trend chart and anomaly timeline
- `pollShap()` — Updates SHAP feature contribution bar chart
- `pollAlerts()` — Updates the alert log table
- `pollDrift()` — Shows/hides the drift warning badge
