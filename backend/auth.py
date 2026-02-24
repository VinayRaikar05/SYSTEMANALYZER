"""
API Authentication
-------------------
Lightweight API-key authentication via X-API-KEY header.
Keys sourced from API_KEYS environment variable (comma-separated).
If no keys are configured, auth is disabled (dev mode).
"""

from __future__ import annotations

import os
from fastapi import Request, HTTPException, Depends
from backend.logging_config import get_logger

logger = get_logger("auth")

_allowed_keys: set[str] | None = None


def _load_keys() -> set[str]:
    global _allowed_keys
    if _allowed_keys is None:
        raw = os.environ.get("API_KEYS", "")
        _allowed_keys = {k.strip() for k in raw.split(",") if k.strip()}
        if _allowed_keys:
            logger.info(f"API auth enabled with {len(_allowed_keys)} key(s)")
        else:
            logger.info("API auth DISABLED (no API_KEYS env var set)")
    return _allowed_keys


def require_api_key(request: Request) -> str | None:
    """
    FastAPI dependency — checks X-API-KEY header.
    If no keys configured (dev mode), allows all requests.
    """
    keys = _load_keys()
    if not keys:
        return None  # Auth disabled

    api_key = request.headers.get("X-API-KEY")
    if not api_key or api_key not in keys:
        logger.warning(f"Unauthorized access attempt from {request.client.host}")
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key
