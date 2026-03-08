"""
CAPTCHA Module
---------------
Generates and verifies simple math CAPTCHAs.
Uses server-side token store — no external APIs needed.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from backend.logging_config import get_logger

logger = get_logger("captcha")

# In-memory store: token → (answer, expiry_timestamp)
_captcha_store: dict[str, tuple[str, float]] = {}
_CAPTCHA_TTL = 300  # 5 minutes
_VERIFIED_TOKENS: dict[str, float] = {}
_VERIFIED_TTL = 86400  # 24 hours


def _cleanup() -> None:
    """Remove expired entries."""
    now = time.time()
    expired = [k for k, (_, exp) in _captcha_store.items() if now > exp]
    for k in expired:
        del _captcha_store[k]
    expired_v = [k for k, exp in _VERIFIED_TOKENS.items() if now > exp]
    for k in expired_v:
        del _VERIFIED_TOKENS[k]


def generate_captcha() -> dict:
    """Generate a math CAPTCHA and return token + question."""
    _cleanup()

    import random
    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("×", lambda a, b: a * b),
    ]
    op_symbol, op_func = random.choice(ops)
    a = random.randint(2, 20)
    b = random.randint(2, 15)

    # Ensure subtraction doesn't go negative
    if op_symbol == "-" and a < b:
        a, b = b, a

    answer = str(op_func(a, b))
    question = f"{a} {op_symbol} {b} = ?"

    token = secrets.token_urlsafe(32)
    _captcha_store[token] = (answer, time.time() + _CAPTCHA_TTL)

    logger.info(f"CAPTCHA generated: {question}")
    return {"token": token, "question": question}


def verify_captcha(token: str, user_answer: str) -> str | bool:
    """Verify a CAPTCHA answer. Returns True if correct."""
    _cleanup()

    entry = _captcha_store.get(token)
    if not entry:
        logger.warning("CAPTCHA verification failed: invalid/expired token")
        return False

    correct_answer, expiry = entry
    del _captcha_store[token]  # One-time use

    if time.time() > expiry:
        logger.warning("CAPTCHA verification failed: expired")
        return False

    if user_answer.strip() != correct_answer:
        logger.warning("CAPTCHA verification failed: wrong answer")
        return False

    # Issue a verified session token
    session_token = secrets.token_urlsafe(32)
    _VERIFIED_TOKENS[session_token] = time.time() + _VERIFIED_TTL
    logger.info("CAPTCHA verified successfully")
    return session_token


def is_verified(session_token: str | None) -> bool:
    """Check if a session token is valid."""
    if not session_token:
        return False
    _cleanup()
    return session_token in _VERIFIED_TOKENS
