"""
Risk Engine — Enterprise Edition
----------------------------------
• compute_health_score    – anomaly score → health (0–100) with severity penalty + sensitivity
• compute_metric_severity – raw metric penalty for sustained high load
• evaluate_risk           – anomaly frequency → GREEN/YELLOW/RED
• should_alert            – detect risk escalation
• diagnose_root_cause     – identify primary issue
• rule_based_anomaly_check – fallback when ML model unavailable
"""

from __future__ import annotations

from backend.logging_config import get_logger

logger = get_logger("risk_engine")


def compute_metric_severity(metrics: dict) -> int:
    """Compute a severity penalty (0–40) based on extreme raw metric values."""
    penalty = 0
    cpu = metrics.get("cpu", 0)
    mem = metrics.get("memory", 0)
    disk = metrics.get("disk_io", 0)
    resp = metrics.get("response_time", 0)

    if cpu > 90:    penalty += 12
    elif cpu > 80:  penalty += 7
    elif cpu > 75:  penalty += 3

    if mem > 88:    penalty += 10
    elif mem > 78:  penalty += 5
    elif mem > 72:  penalty += 2

    if disk > 80:   penalty += 8
    elif disk > 50: penalty += 4

    if resp > 250:   penalty += 10
    elif resp > 180: penalty += 5

    return min(40, penalty)


def compute_health_score(
    anomaly_score: float,
    metrics: dict | None = None,
    sensitivity: int = 5,
) -> int:
    """Map Isolation Forest score to 0–100, apply severity penalty."""
    sens_mult = 0.6 + (sensitivity / 10) * 0.8
    adj_score = anomaly_score / sens_mult

    if adj_score >= 0.15:
        base = min(100, int(92 + adj_score * 20))
    elif adj_score >= 0.05:
        base = int(85 + (adj_score - 0.05) * 70)
    elif adj_score >= 0.0:
        base = int(70 + adj_score * 300)
    elif adj_score >= -0.1:
        base = int(70 + adj_score * 200)
    elif adj_score >= -0.25:
        base = int(50 + (adj_score + 0.1) * 167)
    else:
        base = max(0, int(25 + (adj_score + 0.25) * 100))

    if metrics:
        base -= compute_metric_severity(metrics)

    return max(0, min(100, base))


def evaluate_risk(recent_anomaly_flags: list[bool]) -> str:
    window = recent_anomaly_flags[:10]
    count = sum(window)
    if count >= 3:   return "RED"
    elif count >= 2: return "YELLOW"
    return "GREEN"


def should_alert(prev_risk: str, new_risk: str) -> str | None:
    levels = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    if levels.get(new_risk, 0) > levels.get(prev_risk, 0):
        return "CRITICAL" if new_risk == "RED" else "WARNING"
    return None


def diagnose_root_cause(metrics: dict) -> str:
    """Return a human-readable root cause hint for the dominant issue."""
    issues: list[tuple[float, str]] = []

    cpu = metrics.get("cpu", 0)
    mem = metrics.get("memory", 0)
    disk = metrics.get("disk_io", 0)
    resp = metrics.get("response_time", 0)
    net = metrics.get("network", 0)

    if cpu > 75:
        issues.append((cpu, f"High CPU ({cpu:.0f}%) — Check compute-heavy processes"))
    if mem > 78:
        issues.append((mem, f"High Memory ({mem:.0f}%) — Possible memory leak or cache bloat"))
    if disk > 50:
        issues.append((disk, f"Heavy Disk I/O ({disk:.1f} MB/s) — Large file operations or swap activity"))
    if resp > 200:
        issues.append((resp / 5, f"Slow Response ({resp:.0f}ms) — Backend congestion"))
    if net > 200:
        issues.append((net / 10, f"High Network ({net:.0f} KB/s) — Heavy data transfer"))

    if not issues:
        return "System nominal"

    issues.sort(key=lambda x: x[0], reverse=True)
    return issues[0][1]


def rule_based_anomaly_check(metrics: dict) -> tuple[float, bool, int]:
    """
    Fallback anomaly detection when ML model is unavailable.
    Returns (pseudo_score, is_anomaly, health_score).
    """
    penalty = compute_metric_severity(metrics)

    if penalty >= 20:
        score, anomaly, health = -0.3, True, max(0, 40 - penalty)
    elif penalty >= 10:
        score, anomaly, health = -0.1, True, max(30, 65 - penalty)
    elif penalty >= 5:
        score, anomaly, health = 0.0, False, 80 - penalty
    else:
        score, anomaly, health = 0.15, False, 92

    logger.info(f"Rule-based fallback: penalty={penalty}, health={health}, anomaly={anomaly}")
    return score, anomaly, max(0, min(100, health))
