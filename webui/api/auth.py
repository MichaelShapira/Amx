"""Amazon Cognito authentication and authorization for the API (the cloud deployment).

The local app binds 127.0.0.1 and runs with the operator's own AWS credentials; nothing else can
reach it, so it has no login. The cloud deployment (AmxBench-WebApp) is reachable from the
internet, so every API call must prove who is calling. Configuration comes from the environment
the stack sets:

    AMXBENCH_COGNITO_USER_POOL_ID      us-east-1_XXXXXXXXX
    AMXBENCH_COGNITO_REGION            us-east-1
    AMXBENCH_COGNITO_DOMAIN            https://<prefix>.auth.us-east-1.amazoncognito.com  (hosted login)
    AMXBENCH_COGNITO_CLIENT_ID         the app client id, or
    AMXBENCH_COGNITO_CLIENT_ID_PARAM   the SSM parameter holding it (the client's callback URL is the
                                       service URL, which exists only after the service, so the id
                                       cannot be an environment value of the service itself)

With no pool id set, `enabled()` is False and the gate is off; `__main__` refuses to bind anything
but loopback in that state (fail closed).

The gate (CognitoAuthMiddleware), for every request under /api/ except PUBLIC_PATHS:
  * `Authorization: Bearer <token>`, an ACCESS token issued by the pool to the app client: RS256
    signature against the pool's JWKS, `iss`, `token_use == "access"`, `client_id`, `exp` (30 s
    leeway). Anything else is 401 with a WWW-Authenticate header naming the gate.
  * Authorization: any user of the pool may read (GET). Changing anything (POST, PUT, PATCH,
    DELETE: starting, stopping or deleting runs, calling Bedrock) needs the `operators` group,
    else 403. The deploy script puts the initial user in that group; the pool admits nobody on
    its own (self sign-up is disabled in the stack).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import jwt
from jwt import PyJWKClient
from starlette.requests import Request
from starlette.responses import JSONResponse

PUBLIC_PATHS = frozenset({"/api/health", "/api/auth/config"})
OPERATOR_GROUP = "operators"
MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
REALM = "amx-gpu-bench"
LEEWAY_S = 30

_lock = threading.Lock()
_state: dict[str, Any] = {"client_id": None, "client_id_checked": 0.0, "jwks": None}


def settings() -> dict[str, str] | None:
    pool = os.environ.get("AMXBENCH_COGNITO_USER_POOL_ID", "").strip()
    if not pool:
        return None
    region = os.environ.get("AMXBENCH_COGNITO_REGION", "").strip() or pool.split("_")[0]
    return {"user_pool_id": pool, "region": region,
            "issuer": f"https://cognito-idp.{region}.amazonaws.com/{pool}",
            "domain": os.environ.get("AMXBENCH_COGNITO_DOMAIN", "").strip().rstrip("/"),
            "client_id_env": os.environ.get("AMXBENCH_COGNITO_CLIENT_ID", "").strip(),
            "client_id_param": os.environ.get("AMXBENCH_COGNITO_CLIENT_ID_PARAM", "").strip()}


def enabled() -> bool:
    return settings() is not None


def client_id() -> str | None:
    """The app client id: from the environment, else the SSM parameter (read once it exists; a
    freshly created service may start a minute before the parameter is written)."""
    s = settings()
    if s is None:
        return None
    if s["client_id_env"]:
        return s["client_id_env"]
    with _lock:
        if _state["client_id"]:
            return _state["client_id"]
        if time.time() - _state["client_id_checked"] < 10:
            return None
        _state["client_id_checked"] = time.time()
        if not s["client_id_param"]:
            return None
        try:
            from webui.api import aws

            value = aws.client("ssm").get_parameter(Name=s["client_id_param"])["Parameter"]["Value"].strip()
        except Exception:  # noqa: BLE001 - not there yet, or no permission: the gate stays closed
            return None
        _state["client_id"] = value or None
        return _state["client_id"]


def public_config() -> dict:
    """What the browser needs to sign in; nothing here is secret (the client is public, PKCE)."""
    s = settings()
    if s is None:
        return {"enabled": False}
    cid = client_id()
    return {"enabled": True, "ready": bool(cid), "region": s["region"], "user_pool_id": s["user_pool_id"],
            "issuer": s["issuer"], "client_id": cid, "domain": s["domain"], "operator_group": OPERATOR_GROUP}


def _jwks() -> PyJWKClient:
    s = settings()
    with _lock:
        if _state["jwks"] is None:
            _state["jwks"] = PyJWKClient(f"{s['issuer']}/.well-known/jwks.json", cache_keys=True, lifespan=3600)
        return _state["jwks"]


class AuthError(Exception):
    def __init__(self, description: str, status: int = 401, error: str = "invalid_token"):
        super().__init__(description)
        self.status, self.error, self.description = status, error, description


def verify(token: str) -> dict:
    """Claims of a valid access token for this pool and client, else AuthError."""
    s = settings()
    if s is None:
        raise AuthError("authentication is not configured", 503, "temporarily_unavailable")
    cid = client_id()
    if not cid:
        raise AuthError("the app client is not registered yet; try again in a minute", 503, "temporarily_unavailable")
    try:
        key = _jwks().get_signing_key_from_jwt(token).key
        claims = jwt.decode(token, key, algorithms=["RS256"], issuer=s["issuer"], leeway=LEEWAY_S,
                            options={"require": ["exp", "iat", "sub", "token_use", "client_id"],
                                     "verify_aud": False})
    except jwt.ExpiredSignatureError as e:
        raise AuthError("the token has expired") from e
    except (jwt.PyJWTError, ValueError) as e:
        raise AuthError(f"the token is not valid: {type(e).__name__}") from e
    if claims.get("token_use") != "access":
        raise AuthError("an access token is required")
    if claims.get("client_id") != cid:
        raise AuthError("the token was not issued to this application")
    return claims


def principal(claims: dict) -> dict:
    return {"sub": claims.get("sub"), "username": claims.get("username"),
            "groups": list(claims.get("cognito:groups") or [])}


def _unauthenticated(err: AuthError) -> JSONResponse:
    www = f'Bearer realm="{REALM}", error="{err.error}", error_description="{err.description}"'
    return JSONResponse(status_code=err.status, content={"detail": err.description, "code": "unauthenticated"},
                        headers={"WWW-Authenticate": www, "Cache-Control": "no-store"})


class CognitoAuthMiddleware:
    """Pure ASGI middleware: gates /api/* when Cognito is configured, passes everything else."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not enabled():
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        if not path.startswith("/api/") or path in PUBLIC_PATHS:
            return await self.app(scope, receive, send)
        request = Request(scope)
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        try:
            if scheme.lower() != "bearer" or not token.strip():
                raise AuthError("sign in to use this application", 401, "invalid_request")
            claims = verify(token.strip())
        except AuthError as e:
            return await _unauthenticated(e)(scope, receive, send)
        user = principal(claims)
        if scope.get("method", "GET").upper() in MUTATING_METHODS and OPERATOR_GROUP not in user["groups"]:
            resp = JSONResponse(status_code=403, content={
                "detail": f"Your account can view results but not start, stop or delete runs: that needs the "
                          f"'{OPERATOR_GROUP}' group in the user pool.", "code": "forbidden"},
                headers={"Cache-Control": "no-store"})
            return await resp(scope, receive, send)
        scope.setdefault("state", {})["user"] = user
        return await self.app(scope, receive, send)


def reset_for_tests() -> None:
    with _lock:
        _state.update(client_id=None, client_id_checked=0.0, jwks=None)
