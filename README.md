# ⚡ AI-Based System Failure Early Warning Engine

An enterprise-grade AI observability platform that monitors real-time system metrics, detects anomalies using Isolation Forest, predicts failures, explains decisions with SHAP, and self-adapts through drift detection and automated retraining.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green?logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 What It Does

### Core AI Engine
| Capability | Description |
|---|---|
| **Real-Time Monitoring** | Collects CPU, Memory, Disk I/O, Network via `psutil` every 2 seconds |
| **Anomaly Detection** | Isolation Forest (200 trees) trained on your machine's real data |
| **Failure Prediction** | Logistic Regression predicts crash probability (0–100%) |
| **SHAP Explainability** | Shows *which features* caused the anomaly score |
| **Confidence Scoring** | Normalized 0–1 confidence from historical score distribution |
| **Health Forecasting** | Linear regression projects health 2 minutes ahead |
| **Root Cause Hints** | Auto-suggests "Check compute-heavy process", "Possible memory leak", etc. |

### Enterprise Features
| Capability | Description |
|---|---|
| **Model Retraining** | Retrain from recent DB data with versioning, keeps last 2 models |
| **Drift Detection** | Rolling anomaly rate tracker flags concept drift |
| **API Authentication** | X-API-KEY header protection on sensitive endpoints |
| **Structured Logging** | Rotating file handler → `logs/system.log` |
| **Performance Metrics** | Inference latency, req/min, memory usage tracking |
| **Graceful Degradation** | Rule-based fallback if ML model is unavailable |
| **Per-Server Sensitivity** | Stored per server_id in database |
| **Automated Remediation** | Simulated service restart, scaling, cleanup hooks |
| **Multi-Server Ready** | `server_id` field across all tables and endpoints |

---

## 🏗 Architecture

```
psutil (Real Metrics) → Feature Engineering (30 features)
    → Isolation Forest (Anomaly Score + Confidence)
    → Logistic Regression (Failure Probability)
    → SHAP (Explainability)
    → Risk Engine (Health Score + Root Cause)
    → Drift Detector (Concept Drift Monitoring)
    → SQLite (Storage) → FastAPI (REST API + Auth) → Dashboard (Chart.js)
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/VinayRaikar05/SYSTEMANALYZER.git
cd SYSTEMANALYZER

# 2. Install dependencies
pip install -r requirements.txt
pip install psutil shap

# 3. Train models (collects 60s of real data from YOUR machine)
python -m backend.train_model

# 4. Start server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 5. Open http://localhost:8000/
```

### Optional: Enable API Authentication
```bash
set API_KEYS=your-secret-key-here
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
# Now POST endpoints require X-API-KEY header
```

---

## 📁 Project Structure

```
├── backend/
│   ├── main.py                 # FastAPI server + 18 endpoints
│   ├── database.py             # SQLite ORM (4 tables incl. ServerConfig)
│   ├── feature_engineering.py  # Raw metrics → 30 ML features
│   ├── model.py                # Isolation Forest + SHAP + confidence
│   ├── risk_engine.py          # Health score + risk + root cause + fallback
│   ├── failure_predictor.py    # Failure probability (Logistic Regression)
│   ├── train_model.py          # Train on real psutil data
│   ├── drift_detector.py       # Concept drift detection (200-window)
│   ├── retraining.py           # Model retraining with versioning
│   ├── auth.py                 # API key authentication
│   ├── logging_config.py       # Structured logging (rotating file)
│   ├── performance_monitor.py  # Inference latency + request tracking
│   └── remediation.py          # Automated remediation hooks
├── frontend/
│   ├── dashboard.html          # Enterprise monitoring dashboard
│   └── script.js               # Chart.js + live polling + enterprise UI
├── models/                     # Trained ML models (.pkl) + backups
├── logs/                       # Structured log files (auto-created)
├── requirements.txt
└── PROJECT_GUIDE.md            # Full code walkthrough (1000+ lines)
```

---

## 🔌 API Endpoints (18 Total)

### Public Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard |
| `GET` | `/health` | Health, risk, failure prob, confidence, model status |
| `GET` | `/metrics/recent` | Recent raw metrics |
| `GET` | `/health/history` | Health score time series |
| `GET` | `/health/forecast` | 60-point future projection |
| `GET` | `/explain` | SHAP feature contributions |
| `GET` | `/servers` | Monitored server IDs |
| `GET` | `/settings` | Per-server sensitivity |
| `GET` | `/model/drift-status` | Drift detection status |

### Protected Endpoints (require X-API-KEY when auth enabled)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest-metrics` | Manual metric submission |
| `POST` | `/settings/sensitivity` | Adjust per-server sensitivity (1–10) |
| `POST` | `/simulate/failure` | Inject failure spikes for demo |
| `POST` | `/simulate/stop` | Stop injection |
| `POST` | `/model/retrain` | Trigger model retraining |
| `POST` | `/remediation/trigger` | Trigger remediation actions |
| `GET` | `/model/info` | Model metadata + version |
| `GET` | `/system/performance` | Inference latency, memory, req/min |
| `GET` | `/alerts` | Alert log |

---

## 🧠 ML Models

### Isolation Forest (Anomaly Detection)
- **Type:** Unsupervised, 200 trees, contamination=0.02
- **Features:** 30 (6 per signal: latest, rolling mean, variance, trend slope, spike, rate of change)
- **Output:** Anomaly score + flag + normalized confidence (0–1)

### Logistic Regression (Failure Prediction)
- **Features:** 8 (anomaly freq, health slope, CPU/memory trends, min/mean health)
- **Output:** Failure probability 0.0–1.0

### SHAP (Explainability)
- **Method:** TreeExplainer on Isolation Forest
- **Output:** Per-feature contribution scores (top 8)

---

## 🛡 Enterprise Features Detail

### Concept Drift Detection
Tracks rolling anomaly rate over 200 predictions. If rate exceeds `contamination × 3` (6%) for 2+ minutes, drift is flagged and model retraining is recommended.

### Graceful Degradation
If the model file is missing or prediction fails, the system falls back to rule-based detection using metric thresholds (CPU>90, Memory>88, etc.). Returns `model_status: "fallback_mode"`.

### API Authentication
Set `API_KEYS` environment variable (comma-separated). All POST and sensitive GET endpoints require `X-API-KEY` header. No keys configured = dev mode (auth disabled).

### Structured Logging
All events logged to `logs/system.log` with rotating file handler (5MB max, 3 backups). Levels: INFO, WARNING, ERROR.

---

## 📖 Documentation

See **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** for a comprehensive 1000+ line walkthrough covering every file, function, algorithm, and design decision.

---

## 📝 License

MIT License — free to use, modify, and distribute.
