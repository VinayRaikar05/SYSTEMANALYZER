# ⚡ AI-Based System Failure Early Warning Engine

A production-ready AI system that monitors real-time system metrics, detects anomalies using Isolation Forest, predicts failure probability, and displays live insights through a dark-themed dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green?logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 What It Does

| Capability | Description |
|---|---|
| **Real-Time Monitoring** | Collects CPU, Memory, Disk I/O, Network via `psutil` every 2 seconds |
| **Anomaly Detection** | Isolation Forest (200 trees) trained on your machine's real data |
| **Failure Prediction** | Logistic Regression predicts crash probability (0–100%) |
| **SHAP Explainability** | Shows *which features* caused the anomaly — enterprise-grade transparency |
| **Health Forecasting** | Linear regression projects health score 2 minutes into the future |
| **Root Cause Hints** | Auto-suggests "Check compute-heavy process", "Possible memory leak", etc. |
| **Dynamic Sensitivity** | Adjust anomaly threshold via real-time slider (1–10) |
| **Failure Simulation** | Inject fake spikes for dramatic demos: Normal → RED → Alert fires |
| **Multi-Server Ready** | `server_id` field supports monitoring multiple systems |

---

## 🏗 Architecture

```
psutil (Real Metrics) → Feature Engineering (30 features)
    → Isolation Forest (Anomaly Score)
    → Logistic Regression (Failure Probability)
    → SHAP (Explainability)
    → Risk Engine (Health Score + Root Cause)
    → SQLite (Storage) → FastAPI (REST API) → Dashboard (Chart.js)
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

---

## 📁 Project Structure

```
├── backend/
│   ├── main.py                 # FastAPI server + 13 endpoints
│   ├── database.py             # SQLite ORM (3 tables)
│   ├── feature_engineering.py  # Raw metrics → 30 ML features
│   ├── model.py                # Isolation Forest + SHAP explainer
│   ├── risk_engine.py          # Health score + risk + root cause
│   ├── failure_predictor.py    # Failure probability (Logistic Regression)
│   └── train_model.py          # Train on real psutil data
├── frontend/
│   ├── dashboard.html          # Dark-themed Bootstrap 5 UI
│   └── script.js               # Chart.js + live API polling
├── models/                     # Trained ML models (.pkl)
├── requirements.txt
└── PROJECT_GUIDE.md            # Detailed code walkthrough (700+ lines)
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard |
| `GET` | `/health` | Health score, risk level, failure prob, root cause |
| `GET` | `/metrics/recent` | Recent raw metrics |
| `GET` | `/health/history` | Health score time series |
| `GET` | `/health/forecast` | 60-point future projection |
| `GET` | `/explain` | SHAP feature contributions |
| `GET` | `/alerts` | Alert log |
| `GET` | `/servers` | Monitored server IDs |
| `GET` | `/settings` | Current sensitivity |
| `POST` | `/settings/sensitivity?value=N` | Adjust sensitivity (1–10) |
| `POST` | `/simulate/failure` | Inject failure spikes |
| `POST` | `/simulate/stop` | Stop injection |
| `POST` | `/ingest-metrics` | Manual metric submission |

---

## 🧠 ML Models

### Isolation Forest (Anomaly Detection)
- **Type:** Unsupervised — learns "normal" behavior without labels
- **Training:** 60s of real psutil data → augmented to 3000 samples
- **Features:** 30 (6 per signal: latest value, rolling mean, variance, trend slope, spike magnitude, rate of change)
- **Output:** Anomaly score (positive = normal, negative = anomaly)

### Logistic Regression (Failure Prediction)
- **Type:** Supervised — binary classification (normal vs failure)
- **Features:** 8 (anomaly frequency, health slope, CPU/memory trends, min/mean health)
- **Output:** Probability 0.0–1.0

### SHAP (Explainability)
- **Method:** TreeExplainer on Isolation Forest
- **Output:** Per-feature contribution scores — shows *why* the model flagged an anomaly

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, Uvicorn |
| ML | Scikit-learn, SHAP, NumPy |
| Database | SQLite + SQLAlchemy |
| Metrics | psutil |
| Frontend | HTML, Bootstrap 5, Chart.js |
| Serialization | Joblib |

---

## 📖 Documentation

See **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** for a comprehensive 700+ line walkthrough covering:
- Complete architecture with diagrams
- Every file explained with annotated code
- ML algorithm details
- All 7 upgrades with implementation notes
- Verification results

---

## 📝 License

MIT License — free to use, modify, and distribute.
