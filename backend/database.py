"""
Database layer
---------------
Tables: raw_metrics, health_records, alerts
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
    timestamp = Column(DateTime, default=dt.datetime.now)
    cpu = Column(Float, nullable=False)
    memory = Column(Float, nullable=False)
    disk_io = Column(Float, nullable=False)
    process_count = Column(Float, nullable=False)
    network = Column(Float, nullable=False)


class HealthRecord(Base):
    __tablename__ = "health_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
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
    timestamp = Column(DateTime, default=dt.datetime.now)
    severity = Column(String(20), nullable=False)
    message = Column(String(500), nullable=False)


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


def get_recent_metrics(db: Session, limit: int = 50) -> list[RawMetric]:
    return (
        db.query(RawMetric)
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


def get_recent_health(db: Session, limit: int = 50) -> list[HealthRecord]:
    return (
        db.query(HealthRecord)
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


def get_recent_alerts(db: Session, limit: int = 20) -> list[Alert]:
    return (
        db.query(Alert)
        .order_by(Alert.id.desc())
        .limit(limit)
        .all()
    )
