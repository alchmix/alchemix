"""
Talks to the Visual Engine service (a separate process on this machine, default http://127.0.0.1:5001).

Every call is signed for the AlchemyMix user it is made for: a JWT that lives 2 minutes, signed with
VE_SHARED_SECRET (the same value is in the engine's .env). The engine checks the signature and re-checks the user
(and any admin claim) against the shared `users` collection, so it never trusts this backend blindly either.

    .env
        VE_ENGINE_URL=http://127.0.0.1:5001
        VE_SHARED_SECRET=<same long random value as the engine>
        VE_CALLBACK_SECRET=<optional; defaults to VE_SHARED_SECRET>
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any

import jwt
import requests

log = logging.getLogger("visual_engine_client")

CONNECT_TIMEOUT = 5
READ_TIMEOUT = 30


class EngineError(Exception):
    """The engine answered with an error, or could not be reached (status 502/503/504)."""

    def __init__(self, status: int, code: str, message: str, body: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.body = status, code, message, body or {}

    def to_response(self) -> tuple[dict, int]:
        err = (self.body.get("error") if isinstance(self.body, dict) else None) or {"code": self.code,
                                                                                    "message": self.message}
        return {"error": err}, self.status


def engine_url() -> str:
    return os.getenv("VE_ENGINE_URL", "http://127.0.0.1:5001").rstrip("/")


def _secret() -> str:
    s = os.getenv("VE_SHARED_SECRET", "")
    if len(s) < 32:
        raise EngineError(503, "not_configured", "VE_SHARED_SECRET is missing or shorter than 32 characters")
    return s


def service_token(user_email: str, role: str) -> str:
    now = int(time.time())
    claims = {"iss": os.getenv("VE_TOKEN_ISSUER", "alchemymix"), "aud": os.getenv("VE_TOKEN_AUDIENCE",
                                                                                 "visual-engine"),
              "sub": user_email, "role": "admin" if role == "admin" else "user", "iat": now, "exp": now + 120,
              "jti": uuid.uuid4().hex}
    return jwt.encode(claims, _secret(), algorithm="HS256")


class EngineClient:
    def __init__(self, base_url: str | None = None, session: Any = None):
        self.base = (base_url or engine_url()).rstrip("/")
        self.http = session or requests.Session()

    def call(self, method: str, path: str, *, user: str, role: str = "user", json: Any = None,
             params: dict | None = None, headers: dict | None = None, stream: bool = False,
             timeout: tuple[float, float] | float = (CONNECT_TIMEOUT, READ_TIMEOUT)) -> Any:
        h = {"Authorization": f"Bearer {service_token(user, role)}", **(headers or {})}
        url = f"{self.base}{path}"
        tries = 2 if method == "GET" else 1  # only reads are retried automatically
        last: Exception | None = None
        for i in range(tries):
            try:
                r = self.http.request(method, url, json=json, params=params, headers=h, timeout=timeout,
                                      stream=stream)
                break
            except requests.Timeout as e:
                last = e
                if i + 1 == tries:
                    raise EngineError(504, "engine_timeout", "The video engine did not answer in time.") from e
            except requests.RequestException as e:
                last = e
                if i + 1 == tries:
                    log.warning("visual engine unreachable at %s: %s", url, e)
                    raise EngineError(502, "engine_unavailable", "The video engine is not running.") from e
            time.sleep(0.5)
        else:  # pragma: no cover
            raise EngineError(502, "engine_unavailable", str(last))
        if stream:
            if r.status_code >= 400:
                raise EngineError(r.status_code, "engine_error", "file not available", _json(r))
            return r
        data = _json(r)
        if r.status_code >= 400:
            err = (data or {}).get("error") or {}
            raise EngineError(r.status_code, err.get("code", "engine_error"), err.get("message", r.text[:200]),
                              data)
        return data

    # convenience
    def create_video(self, user: str, role: str, body: dict, idempotency_key: str | None = None) -> dict:
        h = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return self.call("POST", "/v1/videos", user=user, role=role, json=body, headers=h)

    def catalog(self, user: str, role: str) -> dict:
        return self.call("GET", "/v1/catalog", user=user, role=role)


def _json(r) -> dict:
    try:
        return r.json()
    except Exception:
        return {}


_client: EngineClient | None = None


def get_client() -> EngineClient:
    global _client
    if _client is None:
        _client = EngineClient()
    return _client


def set_client(c: EngineClient | None) -> None:  # tests
    global _client
    _client = c
