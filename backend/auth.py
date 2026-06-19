"""
Supabase JWT authentication for the FastAPI backend.

Verification strategy: forward the caller's bearer token to Supabase's
`GET /auth/v1/user` endpoint, which validates the signature + expiry and
returns the user. This works regardless of the project's token signing
scheme (HS256 shared-secret vs. asymmetric ES256/JWKS).

Enforcement is gated by the REQUIRE_AUTH env flag so the backend can be
deployed *before* the frontend starts sending tokens (staged rollout):

    REQUIRE_AUTH unset/false → tokens are still verified when present, but
                               anonymous requests are allowed (permissive).
    REQUIRE_AUTH=true        → missing/invalid tokens are rejected with 401.

Required env (add to the HF Space secrets before flipping REQUIRE_AUTH on):
    SUPABASE_URL          (or NEXT_PUBLIC_SUPABASE_URL)
    SUPABASE_ANON_KEY     (or NEXT_PUBLIC_SUPABASE_ANON_KEY)
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

import requests
from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)


def _auth_enforced() -> bool:
    return os.getenv("REQUIRE_AUTH", "false").strip().lower() in ("1", "true", "yes", "on")


_SUPABASE_URL = (
    os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or ""
).rstrip("/")
_SUPABASE_ANON_KEY = (
    os.getenv("SUPABASE_ANON_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY") or ""
)

# Small in-process cache so repeated calls (e.g. SSE reconnects) don't hit
# Supabase on every request. token → (user, expires_at_monotonic).
_TOKEN_CACHE: dict[str, tuple[dict[str, Any], float]] = {}
_TOKEN_CACHE_TTL = 60.0  # seconds


def _verify_sync(token: str) -> Optional[dict[str, Any]]:
    if not _SUPABASE_URL or not _SUPABASE_ANON_KEY:
        logger.warning(
            "Supabase auth not configured (SUPABASE_URL / SUPABASE_ANON_KEY missing) — "
            "cannot verify tokens."
        )
        return None

    cached = _TOKEN_CACHE.get(token)
    if cached and cached[1] > time.monotonic():
        return cached[0]

    try:
        resp = requests.get(
            f"{_SUPABASE_URL}/auth/v1/user",
            headers={
                "apikey": _SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {token}",
            },
            timeout=10,
        )
    except Exception as exc:
        logger.warning("Token verification request failed: %s", exc)
        return None

    if resp.status_code != 200:
        return None

    try:
        user = resp.json()
    except Exception:
        return None

    if not user.get("id"):
        return None

    _TOKEN_CACHE[token] = (user, time.monotonic() + _TOKEN_CACHE_TTL)
    return user


async def _verify_token(token: str) -> Optional[dict[str, Any]]:
    return await asyncio.to_thread(_verify_sync, token)


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if authorization and authorization.lower().startswith("bearer "):
        tok = authorization[7:].strip()
        return tok or None
    return None


async def get_current_user(
    authorization: Optional[str] = Header(None),
) -> Optional[dict[str, Any]]:
    """
    FastAPI dependency. Returns the verified Supabase user dict, or None.

    When REQUIRE_AUTH is on, a missing/invalid token raises 401. When off,
    anonymous callers get None (so endpoints stay reachable during rollout).
    """
    token = _extract_bearer(authorization)
    if not token:
        if _auth_enforced():
            raise HTTPException(401, "Authentication required.")
        return None

    user = await _verify_token(token)
    if user is None and _auth_enforced():
        raise HTTPException(401, "Invalid or expired session. Please sign in again.")
    return user
