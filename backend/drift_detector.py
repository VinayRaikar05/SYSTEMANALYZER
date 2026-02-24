"""
Concept Drift Detector
-----------------------
Tracks rolling anomaly rate and health score trends.
Flags drift when anomaly rate exceeds contamination × 3 for a sustained window.
"""

from __future__ import annotations

import time
from collections import deque
from backend.logging_config import get_logger

logger = get_logger("drift")

CONTAMINATION = 0.02
DRIFT_THRESHOLD = CONTAMINATION * 3   # 6%
WINDOW_SIZE = 200
SUSTAINED_SECONDS = 120               # Must persist for 2 minutes


class DriftDetector:
    """Singleton-style drift tracker."""

    def __init__(self) -> None:
        self._anomaly_flags: deque[bool] = deque(maxlen=WINDOW_SIZE)
        self._health_scores: deque[int] = deque(maxlen=WINDOW_SIZE)
        self._drift_start: float | None = None
        self._drift_detected: bool = False

    def record(self, is_anomaly: bool, health_score: int) -> None:
        """Record a new prediction result."""
        self._anomaly_flags.append(is_anomaly)
        self._health_scores.append(health_score)
        self._evaluate()

    def _evaluate(self) -> None:
        if len(self._anomaly_flags) < 50:
            return  # Not enough data yet

        rate = sum(self._anomaly_flags) / len(self._anomaly_flags)

        if rate > DRIFT_THRESHOLD:
            if self._drift_start is None:
                self._drift_start = time.time()
                logger.warning(
                    f"Drift suspected: anomaly rate {rate:.2%} > threshold {DRIFT_THRESHOLD:.2%}"
                )
            elif time.time() - self._drift_start >= SUSTAINED_SECONDS:
                if not self._drift_detected:
                    self._drift_detected = True
                    logger.error(
                        f"DRIFT CONFIRMED: anomaly rate {rate:.2%} sustained for "
                        f"{SUSTAINED_SECONDS}s — model retraining recommended"
                    )
        else:
            if self._drift_detected:
                logger.info("Drift resolved: anomaly rate returned to normal")
            self._drift_start = None
            self._drift_detected = False

    @property
    def anomaly_rate(self) -> float:
        if not self._anomaly_flags:
            return 0.0
        return sum(self._anomaly_flags) / len(self._anomaly_flags)

    @property
    def mean_health(self) -> float:
        if not self._health_scores:
            return 100.0
        return sum(self._health_scores) / len(self._health_scores)

    def status(self) -> dict:
        return {
            "drift_detected": self._drift_detected,
            "anomaly_rate": round(self.anomaly_rate, 4),
            "expected_rate": CONTAMINATION,
            "drift_threshold": DRIFT_THRESHOLD,
            "mean_health_score": round(self.mean_health, 2),
            "samples_tracked": len(self._anomaly_flags),
            "sustained_since": (
                round(time.time() - self._drift_start, 1) if self._drift_start else None
            ),
        }


# Module-level singleton
drift_detector = DriftDetector()
