"""
Database layer – Enterprise Edition
-------------------------------------
Tables: raw_metrics, health_records, alerts, server_configs
All tables include server_id for multi-server support.
ServerConfig stores per-server sensitivity and settings.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Boolean,
    create_engine,
)
from sqlalchemy.orm import declarative_base, Session, sessionmaker

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "system.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()


# ── ORM Models ────────────────────────────────────────────────────────────────

class RawMetric(Base):
    __tablename__ = "raw_metrics"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    server_id = Column(String(50), default="local", nullable=False, index=True)
    timestamp = Column(DateTime, default=dt.datetime.now)
    cpu = Column(Float, nullable=False)
    memory = Column(Float, nullable=False)
    disk_io = Column(Float, nullable=False)
    response_time = Column(Float, nullable=False)
    network = Column(Float, nullable=False)


class HealthRecord(Base):
    __tablename__ = "health_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    server_id = Column(String(50), default="local", nullable=False, index=True)
    timestamp = Column(DateTime, default=dt.datetime.now)
    health_score = Column(Integer, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    anomaly_flag = Column(Boolean, nullable=False)
    risk_level = Column(String(10), nullable=False)
    failure_prob = Column(Float, default=0.0)
    root_cause = Column(String(200), default="System nominal")
    confidence = Column(Float, default=0.0)
    model_status = Column(String(20), default="active")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    server_id = Column(String(50), default="local", nullable=False, index=True)
    timestamp = Column(DateTime, default=dt.datetime.now)
    severity = Column(String(20), nullable=False)
    message = Column(String(500), nullable=False)


class ServerConfig(Base):
    __tablename__ = "server_configs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    server_id = Column(String(50), unique=True, nullable=False, index=True)
    sensitivity = Column(Integer, default=5)
    updated_at = Column(DateTime, default=dt.datetime.now, onupdate=dt.datetime.now)


# ── Create tables ─────────────────────────────────────────────────────────────

def init_db() -> None:
    Base.metadata.create_all(bind=engine)


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def get_db() -> Session:
    return SessionLocal()


def insert_metric(db: Session, data: dict) -> RawMetric:
    metric = RawMetric(**data)
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


def get_recent_metrics(db: Session, limit: int = 50, server_id: str = "local") -> list[RawMetric]:
    return (
        db.query(RawMetric)
        .filter(RawMetric.server_id == server_id)
        .order_by(RawMetric.id.desc())
        .limit(limit)
        .all()
    )


def insert_health_record(db: Session, data: dict) -> HealthRecord:
    record = HealthRecord(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_recent_health(db: Session, limit: int = 50, server_id: str = "local") -> list[HealthRecord]:
    return (
        db.query(HealthRecord)
        .filter(HealthRecord.server_id == server_id)
        .order_by(HealthRecord.id.desc())
        .limit(limit)
        .all()
    )


def insert_alert(db: Session, data: dict) -> Alert:
    alert = Alert(**data)
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def get_recent_alerts(db: Session, limit: int = 20, server_id: str = "local") -> list[Alert]:
    return (
        db.query(Alert)
        .filter(Alert.server_id == server_id)
        .order_by(Alert.id.desc())
        .limit(limit)
        .all()
    )


def get_server_ids(db: Session) -> list[str]:
    rows = db.query(RawMetric.server_id).distinct().all()
    return [r[0] for r in rows] if rows else ["local"]


# ── Server Config CRUD ───────────────────────────────────────────────────────

def get_server_sensitivity(db: Session, server_id: str = "local") -> int:
    cfg = db.query(ServerConfig).filter(ServerConfig.server_id == server_id).first()
    return cfg.sensitivity if cfg else 5


def set_server_sensitivity(db: Session, server_id: str, value: int) -> int:
    cfg = db.query(ServerConfig).filter(ServerConfig.server_id == server_id).first()
    if cfg:
        cfg.sensitivity = value
        cfg.updated_at = dt.datetime.now()
    else:
        cfg = ServerConfig(server_id=server_id, sensitivity=value)
        db.add(cfg)
    db.commit()
    return value
