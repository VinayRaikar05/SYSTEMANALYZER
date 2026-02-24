"""
FastAPI Application – Enterprise AI Observability Platform
============================================================
All 7 original upgrades + 10 enterprise features:
 1-7.  Original: failure prob, SHAP, sensitivity, injection, forecast, multi-server, root cause
 8.    Model retraining with versioning
 9.    Concept drift detection
 10.   Per-server sensitivity (DB-backed)
 11.   API key authentication
 12.   Structured logging
 13.   Performance monitoring metrics
 14.   Model metadata endpoint
 15.   Confidence scoring
 16.   Graceful degradation mode
 17.   Automated remediation hook
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import psutil
from fastapi import FastAPI, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Setup logging FIRST ──────────────────────────────────────────────────────
from backend.logging_config import setup_logging, get_logger
setup_logging()
logger = get_logger("main")

from backend.database import (
    init_db, get_db,
    insert_metric, get_recent_metrics,
    insert_health_record, get_recent_health,
    insert_alert, get_recent_alerts, get_server_ids,
    get_server_sensitivity, set_server_sensitivity,
)
from backend.feature_engineering import compute_features
from backend.model import predict, explain, get_model_info
from backend.risk_engine import (
    compute_health_score, evaluate_risk, should_alert,
    diagnose_root_cause, rule_based_anomaly_check,
)
from backend.failure_predictor import predict_failure_probability
from backend.drift_detector import drift_detector
from backend.performance_monitor import record_inference_latency, record_request, get_performance_stats
from backend.retraining import retrain_isolation_forest, get_retraining_info
from backend.remediation import trigger_remediation
from backend.auth import require_api_key


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
_inject_mode: bool = False
_inject_remaining: int = 0
_latest_explain: list[dict] = []

_prev_disk = psutil.disk_io_counters()
_prev_net = psutil.net_io_counters()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _run_pipeline(data: dict) -> dict | None:
    global _prev_risk, _latest_explain

    t0 = time.time()
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

        # Get per-server sensitivity
        sensitivity = get_server_sensitivity(db, server_id)

        # ML prediction with confidence (graceful degradation built into predict)
        anomaly_score, is_anomaly, confidence = predict(features)

        # Check model status
        model_info = get_model_info()
        model_status = model_info.get("model_status", "active")

        # If in fallback mode, use rule-based detection
        if model_status == "fallback_mode":
            anomaly_score, is_anomaly, health_score = rule_based_anomaly_check(data)
            confidence = 0.5
            logger.warning("Using rule-based fallback for anomaly detection")
        else:
            health_score = compute_health_score(anomaly_score, metrics=data, sensitivity=sensitivity)

        # SHAP explanation (skip in fallback mode)
        if model_status != "fallback_mode":
            _latest_explain = explain(features)

        # Drift tracking
        drift_detector.record(is_anomaly, health_score)

        # Risk evaluation
        health_rows = get_recent_health(db, limit=10, server_id=server_id)
        recent_flags = [h.anomaly_flag for h in health_rows]
        recent_flags.insert(0, is_anomaly)
        risk_level = evaluate_risk(recent_flags)

        # Root cause
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
            "server_id": server_id, "timestamp": now,
            "health_score": health_score, "anomaly_score": anomaly_score,
            "anomaly_flag": is_anomaly, "risk_level": risk_level,
            "failure_prob": failure_prob, "root_cause": root_cause,
            "confidence": confidence, "model_status": model_status,
        }
        insert_health_record(db, health_data)

        # Alert on escalation
        alert_severity = should_alert(_prev_risk, risk_level)
        if alert_severity:
            alert_msg = (
                f"Risk escalated to {risk_level} — {root_cause} "
                f"(health={health_score}, failure_prob={failure_prob:.1%}, "
                f"confidence={confidence:.2f})"
            )
            insert_alert(db, {
                "server_id": server_id, "timestamp": now,
                "severity": alert_severity, "message": alert_msg,
            })
            logger.warning(f"ALERT [{alert_severity}]: {alert_msg}")

        _prev_risk = risk_level

        # Record performance
        record_inference_latency(time.time() - t0)

        return {
            "timestamp": now.isoformat(),
            "health_score": health_score,
            "anomaly_score": round(anomaly_score, 4),
            "anomaly_flag": is_anomaly,
            "risk_level": risk_level,
            "failure_prob": failure_prob,
            "root_cause": root_cause,
            "confidence": round(confidence, 4),
            "model_status": model_status,
        }
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return None
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
            logger.error(f"Metric loop error: {exc}", exc_info=True)
        await asyncio.sleep(2)


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bg_task
    init_db()
    logger.info("System Failure Early Warning Engine started")
    _bg_task = asyncio.create_task(_metric_loop())
    yield
    _bg_task.cancel()
    logger.info("Engine shut down")


app = FastAPI(title="System Failure Early Warning Engine", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Middleware: request counter ───────────────────────────────────────────────

@app.middleware("http")
async def perf_middleware(request, call_next):
    record_request()
    response = await call_next(request)
    return response


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENDPOINTS (no auth required)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health(server_id: str = "local"):
    db = get_db()
    try:
        rows = get_recent_health(db, limit=1, server_id=server_id)
        if not rows:
            return {"health_score": 100, "risk_level": "GREEN", "anomaly_score": 0.0,
                    "anomaly_flag": False, "failure_prob": 0.0, "root_cause": "System nominal",
                    "confidence": 0.0, "model_status": "active"}
        r = rows[0]
        return {
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "health_score": r.health_score,
            "anomaly_score": round(r.anomaly_score, 4),
            "anomaly_flag": r.anomaly_flag,
            "risk_level": r.risk_level,
            "failure_prob": r.failure_prob or 0.0,
            "root_cause": r.root_cause or "System nominal",
            "confidence": round(r.confidence or 0.0, 4),
            "model_status": r.model_status or "active",
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
             "failure_prob": r.failure_prob or 0.0, "root_cause": r.root_cause or "",
             "confidence": round(r.confidence or 0.0, 4), "model_status": r.model_status or "active"}
            for r in rows
        ]
    finally:
        db.close()


@app.get("/health/forecast")
def health_forecast(server_id: str = "local"):
    db = get_db()
    try:
        rows = get_recent_health(db, limit=20, server_id=server_id)
        if len(rows) < 5:
            return {"forecast": [], "slope": 0}
        scores = [r.health_score for r in reversed(rows)]
        x = np.arange(len(scores))
        coeffs = np.polyfit(x, scores, 1)
        forecast = [max(0, min(100, round(float(coeffs[0] * (len(scores) + i) + coeffs[1]))))
                    for i in range(1, 61)]
        return {"forecast": forecast, "slope": round(float(coeffs[0]), 3)}
    finally:
        db.close()


@app.get("/explain")
def get_explanation():
    return {"contributions": _latest_explain}


@app.get("/servers")
def list_servers():
    db = get_db()
    try:
        return {"servers": get_server_ids(db)}
    finally:
        db.close()


@app.get("/settings")
def get_settings(server_id: str = "local"):
    db = get_db()
    try:
        return {"sensitivity": get_server_sensitivity(db, server_id), "server_id": server_id}
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════════════════════
# PROTECTED ENDPOINTS (require API key when auth is enabled)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/ingest-metrics", dependencies=[Depends(require_api_key)])
def ingest_metrics(payload: MetricIn):
    data = payload.model_dump()
    result = _run_pipeline(data)
    if result is None:
        return {"status": "buffering", "message": "Need more data points"}
    return {"status": "ok", "health": result}


@app.post("/settings/sensitivity", dependencies=[Depends(require_api_key)])
def set_sensitivity(value: int = Query(ge=1, le=10), server_id: str = "local"):
    db = get_db()
    try:
        set_server_sensitivity(db, server_id, value)
        logger.info(f"Sensitivity changed to {value} for server {server_id}")
        return {"sensitivity": value, "server_id": server_id}
    finally:
        db.close()


@app.post("/simulate/failure", dependencies=[Depends(require_api_key)])
def simulate_failure(count: int = 10):
    global _inject_mode, _inject_remaining
    _inject_mode = True
    _inject_remaining = count
    logger.warning(f"Failure injection started: {count} spikes")
    return {"status": "injecting", "count": count}


@app.post("/simulate/stop", dependencies=[Depends(require_api_key)])
def simulate_stop():
    global _inject_mode, _inject_remaining
    _inject_mode = False
    _inject_remaining = 0
    logger.info("Failure injection stopped")
    return {"status": "stopped"}


# ── Enterprise: Model management ─────────────────────────────────────────────

@app.get("/model/info", dependencies=[Depends(require_api_key)])
def model_info():
    return get_model_info()


@app.get("/model/drift-status")
def drift_status():
    return drift_detector.status()


@app.post("/model/retrain", dependencies=[Depends(require_api_key)])
def retrain_model(server_id: str = "local"):
    logger.info(f"Manual retrain triggered for {server_id}")
    result = retrain_isolation_forest(server_id=server_id)
    return result


# ── Enterprise: System performance ───────────────────────────────────────────

@app.get("/system/performance", dependencies=[Depends(require_api_key)])
def system_performance():
    return get_performance_stats()


# ── Enterprise: Alerts (protected) ───────────────────────────────────────────

@app.get("/alerts", dependencies=[Depends(require_api_key)])
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


# ── Enterprise: Remediation ──────────────────────────────────────────────────

@app.post("/remediation/trigger", dependencies=[Depends(require_api_key)])
def remediation_trigger(server_id: str = "local"):
    db = get_db()
    try:
        rows = get_recent_health(db, limit=1, server_id=server_id)
        risk = rows[0].risk_level if rows else "GREEN"
    finally:
        db.close()
    return trigger_remediation(server_id=server_id, risk_level=risk)


# ── Serve frontend ────────────────────────────────────────────────────────────

@app.get("/")
def serve_dashboard():
    return FileResponse(FRONTEND_DIR / "dashboard.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
