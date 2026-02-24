"""
FastAPI Application – System Failure Early Warning Engine v2
=============================================================
All 7 Upgrades integrated:
 1. Failure probability prediction
 2. SHAP explainability
 3. Sensitivity slider
 4. Failure injection mode
 5. Health trend forecast
 6. Multi-server support
 7. Root cause hints
"""

from __future__ import annotations

import asyncio
import datetime as dt
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import psutil
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.database import (
    init_db, get_db,
    insert_metric, get_recent_metrics,
    insert_health_record, get_recent_health,
    insert_alert, get_recent_alerts, get_server_ids,
)
from backend.feature_engineering import compute_features
from backend.model import predict, explain
from backend.risk_engine import (
    compute_health_score, evaluate_risk, should_alert, diagnose_root_cause,
)
from backend.failure_predictor import predict_failure_probability

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class MetricIn(BaseModel):
    cpu: float
    memory: float
    disk_io: float
    response_time: float
    network: float
    server_id: str = "local"


# ── Globals ───────────────────────────────────────────────────────────────────

RNG = np.random.default_rng()
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

_prev_risk: str = "GREEN"
_bg_task: asyncio.Task | None = None
_sensitivity: int = 5          # 1-10
_inject_mode: bool = False     # failure injection flag
_inject_remaining: int = 0     # how many injections left
_latest_explain: list[dict] = []
_latest_data: dict = {}

_prev_disk = psutil.disk_io_counters()
_prev_net = psutil.net_io_counters()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _run_pipeline(data: dict) -> dict | None:
    global _prev_risk, _latest_explain, _latest_data

    server_id = data.pop("server_id", "local")
    data_with_sid = {**data, "server_id": server_id}

    db = get_db()
    try:
        insert_metric(db, data_with_sid)

        recent = get_recent_metrics(db, limit=10, server_id=server_id)
        recent_dicts = [
            {"cpu": r.cpu, "memory": r.memory, "disk_io": r.disk_io,
             "response_time": r.response_time, "network": r.network}
            for r in reversed(recent)
        ]

        features = compute_features(recent_dicts)
        if features is None:
            return None

        anomaly_score, is_anomaly = predict(features)
        health_score = compute_health_score(anomaly_score, metrics=data, sensitivity=_sensitivity)

        # SHAP explanation
        _latest_explain = explain(features)
        _latest_data = data

        # Risk evaluation
        health_rows = get_recent_health(db, limit=10, server_id=server_id)
        recent_flags = [h.anomaly_flag for h in health_rows]
        recent_flags.insert(0, is_anomaly)
        risk_level = evaluate_risk(recent_flags)

        # Root cause hint
        root_cause = diagnose_root_cause(data)

        # Failure probability
        recent_scores = [h.health_score for h in health_rows]
        recent_scores.insert(0, health_score)
        recent_anom = [h.anomaly_flag for h in health_rows]
        recent_anom.insert(0, is_anomaly)
        recent_met = recent_dicts[-10:] if len(recent_dicts) >= 5 else recent_dicts
        failure_prob = predict_failure_probability(recent_scores, recent_anom, recent_met)

        now = dt.datetime.now()
        health_data = {
            "server_id": server_id,
            "timestamp": now,
            "health_score": health_score,
            "anomaly_score": anomaly_score,
            "anomaly_flag": is_anomaly,
            "risk_level": risk_level,
            "failure_prob": failure_prob,
            "root_cause": root_cause,
        }
        insert_health_record(db, health_data)

        # Alert on escalation
        alert_severity = should_alert(_prev_risk, risk_level)
        if alert_severity:
            insert_alert(db, {
                "server_id": server_id,
                "timestamp": now,
                "severity": alert_severity,
                "message": f"Risk escalated to {risk_level} — {root_cause} "
                           f"(health={health_score}, failure_prob={failure_prob:.1%})",
            })
        _prev_risk = risk_level

        return {
            "timestamp": now.isoformat(),
            "health_score": health_score,
            "anomaly_score": round(anomaly_score, 4),
            "anomaly_flag": is_anomaly,
            "risk_level": risk_level,
            "failure_prob": failure_prob,
            "root_cause": root_cause,
        }
    finally:
        db.close()


# ── Real system metric collector ──────────────────────────────────────────────

def _collect_real_metrics() -> dict:
    global _prev_disk, _prev_net

    cpu = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory().percent

    disk_now = psutil.disk_io_counters()
    disk_delta = (disk_now.read_bytes + disk_now.write_bytes) - \
                 (_prev_disk.read_bytes + _prev_disk.write_bytes)
    disk_io = disk_delta / (1024 * 1024 * 2)
    _prev_disk = disk_now

    net_now = psutil.net_io_counters()
    net_delta = (net_now.bytes_sent + net_now.bytes_recv) - \
                (_prev_net.bytes_sent + _prev_net.bytes_recv)
    network = net_delta / (1024 * 2)
    _prev_net = net_now

    response_time = 50 + cpu * 1.5 + disk_io * 5

    return {
        "cpu": round(cpu, 2), "memory": round(memory, 2),
        "disk_io": round(disk_io, 2), "response_time": round(response_time, 2),
        "network": round(network, 2), "server_id": "local",
    }


def _generate_failure_metrics() -> dict:
    """Simulated failure spike for demo injection."""
    return {
        "cpu": round(float(np.clip(RNG.normal(92, 5), 80, 100)), 2),
        "memory": round(float(np.clip(RNG.normal(90, 4), 82, 99)), 2),
        "disk_io": round(float(np.clip(RNG.normal(70, 15), 40, 120)), 2),
        "response_time": round(float(np.clip(RNG.normal(350, 60), 200, 600)), 2),
        "network": round(float(np.clip(RNG.normal(200, 50), 100, 400)), 2),
        "server_id": "local",
    }


async def _metric_loop() -> None:
    global _inject_mode, _inject_remaining
    while True:
        try:
            if _inject_mode and _inject_remaining > 0:
                data = _generate_failure_metrics()
                _inject_remaining -= 1
                if _inject_remaining <= 0:
                    _inject_mode = False
            else:
                data = _collect_real_metrics()
            _run_pipeline(data)
        except Exception as exc:
            print(f"[metric_loop] error: {exc}")
        await asyncio.sleep(2)


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bg_task
    init_db()
    _bg_task = asyncio.create_task(_metric_loop())
    yield
    _bg_task.cancel()


app = FastAPI(title="System Failure Early Warning Engine", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.post("/ingest-metrics")
def ingest_metrics(payload: MetricIn):
    data = payload.model_dump()
    result = _run_pipeline(data)
    if result is None:
        return {"status": "buffering", "message": "Need more data points"}
    return {"status": "ok", "health": result}


@app.get("/health")
def health(server_id: str = "local"):
    db = get_db()
    try:
        rows = get_recent_health(db, limit=1, server_id=server_id)
        if not rows:
            return {"health_score": 100, "risk_level": "GREEN", "anomaly_score": 0.0,
                    "anomaly_flag": False, "failure_prob": 0.0, "root_cause": "System nominal"}
        r = rows[0]
        return {
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "health_score": r.health_score,
            "anomaly_score": round(r.anomaly_score, 4),
            "anomaly_flag": r.anomaly_flag,
            "risk_level": r.risk_level,
            "failure_prob": r.failure_prob or 0.0,
            "root_cause": r.root_cause or "System nominal",
        }
    finally:
        db.close()


@app.get("/metrics/recent")
def recent_metrics(limit: int = 50, server_id: str = "local"):
    db = get_db()
    try:
        rows = get_recent_metrics(db, limit=limit, server_id=server_id)
        return [
            {"id": r.id, "timestamp": r.timestamp.isoformat() if r.timestamp else None,
             "cpu": round(r.cpu, 2), "memory": round(r.memory, 2),
             "disk_io": round(r.disk_io, 2), "response_time": round(r.response_time, 2),
             "network": round(r.network, 2)}
            for r in rows
        ]
    finally:
        db.close()


@app.get("/health/history")
def health_history(limit: int = 50, server_id: str = "local"):
    db = get_db()
    try:
        rows = get_recent_health(db, limit=limit, server_id=server_id)
        return [
            {"timestamp": r.timestamp.isoformat() if r.timestamp else None,
             "health_score": r.health_score, "anomaly_score": round(r.anomaly_score, 4),
             "anomaly_flag": r.anomaly_flag, "risk_level": r.risk_level,
             "failure_prob": r.failure_prob or 0.0, "root_cause": r.root_cause or ""}
            for r in rows
        ]
    finally:
        db.close()


@app.get("/alerts")
def alerts(limit: int = 20, server_id: str = "local"):
    db = get_db()
    try:
        rows = get_recent_alerts(db, limit=limit, server_id=server_id)
        return [
            {"id": r.id, "timestamp": r.timestamp.isoformat() if r.timestamp else None,
             "severity": r.severity, "message": r.message}
            for r in rows
        ]
    finally:
        db.close()


# ── Upgrade 2: SHAP Explain ──────────────────────────────────────────────────

@app.get("/explain")
def get_explanation():
    return {"contributions": _latest_explain}


# ── Upgrade 3: Sensitivity ───────────────────────────────────────────────────

@app.get("/settings")
def get_settings():
    return {"sensitivity": _sensitivity}


@app.post("/settings/sensitivity")
def set_sensitivity(value: int = Query(ge=1, le=10)):
    global _sensitivity
    _sensitivity = value
    return {"sensitivity": _sensitivity}


# ── Upgrade 4: Failure Injection ─────────────────────────────────────────────

@app.post("/simulate/failure")
def simulate_failure(count: int = 10):
    global _inject_mode, _inject_remaining
    _inject_mode = True
    _inject_remaining = count
    return {"status": "injecting", "count": count}


@app.post("/simulate/stop")
def simulate_stop():
    global _inject_mode, _inject_remaining
    _inject_mode = False
    _inject_remaining = 0
    return {"status": "stopped"}


# ── Upgrade 5: Health Forecast ───────────────────────────────────────────────

@app.get("/health/forecast")
def health_forecast(server_id: str = "local"):
    db = get_db()
    try:
        rows = get_recent_health(db, limit=20, server_id=server_id)
        if len(rows) < 5:
            return {"forecast": [], "slope": 0}

        scores = [r.health_score for r in reversed(rows)]  # oldest first
        x = np.arange(len(scores))
        coeffs = np.polyfit(x, scores, 1)
        slope = coeffs[0]

        # Project 60 points into the future (2 minutes at 2s intervals)
        forecast = []
        for i in range(1, 61):
            projected = coeffs[0] * (len(scores) + i) + coeffs[1]
            forecast.append(max(0, min(100, round(float(projected)))))

        return {"forecast": forecast, "slope": round(float(slope), 3)}
    finally:
        db.close()


# ── Upgrade 6: Server list ──────────────────────────────────────────────────

@app.get("/servers")
def list_servers():
    db = get_db()
    try:
        return {"servers": get_server_ids(db)}
    finally:
        db.close()


# ── Serve frontend ────────────────────────────────────────────────────────────

@app.get("/")
def serve_dashboard():
    return FileResponse(FRONTEND_DIR / "dashboard.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
