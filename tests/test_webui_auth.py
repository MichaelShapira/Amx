"""The Cognito gate on the API (webui/api/auth.py), with real RS256 tokens against a test key
standing in for the pool's JWKS, and the security headers."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webui.api import auth  # noqa: E402
from webui.api.main import app  # noqa: E402

POOL, REGION, CLIENT = "us-east-1_TestPool1", "us-east-1", "7abcdefghijklmnopqrstuvwxy"
ISSUER = f"https://cognito-idp.{REGION}.amazonaws.com/{POOL}"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PEM = KEY.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
OTHER_PEM = OTHER_KEY.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption())


class _Key:
    def __init__(self, key):
        self.key = key


class _Jwks:
    def get_signing_key_from_jwt(self, token):
        return _Key(KEY.public_key())


def token(**over) -> str:
    now = int(time.time())
    claims = {"sub": "11111111-2222-3333-4444-555555555555", "iss": ISSUER, "client_id": CLIENT, "token_use": "access",
              "scope": "openid email profile", "username": "someone", "iat": now, "exp": now + 3600,
              "cognito:groups": ["operators"]}
    claims.update(over)
    key = claims.pop("_key", PEM)
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "k1"})


@pytest.fixture()
def gated(monkeypatch):
    monkeypatch.setenv("AMXBENCH_COGNITO_USER_POOL_ID", POOL)
    monkeypatch.setenv("AMXBENCH_COGNITO_REGION", REGION)
    monkeypatch.setenv("AMXBENCH_COGNITO_CLIENT_ID", CLIENT)
    monkeypatch.setenv("AMXBENCH_COGNITO_DOMAIN", "https://amx-gpu-bench-123.auth.us-east-1.amazoncognito.com")
    auth.reset_for_tests()
    monkeypatch.setattr(auth, "_jwks", lambda: _Jwks())
    yield TestClient(app, raise_server_exceptions=False)
    auth.reset_for_tests()


def test_without_a_pool_the_local_app_is_open_and_says_so(monkeypatch):
    monkeypatch.delenv("AMXBENCH_COGNITO_USER_POOL_ID", raising=False)
    auth.reset_for_tests()
    c = TestClient(app)
    assert c.get("/api/auth/config").json() == {"enabled": False}
    assert c.get("/api/catalog").status_code == 200           # no token needed on loopback
    assert c.get("/api/health").json() == {"ok": True}


def test_anonymous_calls_are_refused_with_an_identifiable_gate(gated):
    c = gated
    r = c.get("/api/runs")
    assert r.status_code == 401 and r.json()["code"] == "unauthenticated"
    assert r.headers["WWW-Authenticate"].startswith('Bearer realm="amx-gpu-bench", error="invalid_request"')
    assert r.headers["Cache-Control"] == "no-store"
    for path in ("/api/catalog", "/api/runs", "/api/aws/status", "/api/data/standard"):
        assert c.get(path).status_code == 401, path
    assert c.post("/api/plan", json={}).status_code == 401
    assert c.delete("/api/runs/x").status_code == 401
    # The two public routes stay public: liveness, and what the browser needs to start signing in.
    assert c.get("/api/health").status_code == 200
    cfg = c.get("/api/auth/config").json()
    assert cfg["enabled"] and cfg["ready"] and cfg["client_id"] == CLIENT and cfg["issuer"] == ISSUER
    assert cfg["user_pool_id"] == POOL and cfg["domain"].startswith("https://") and cfg["operator_group"] == "operators"
    # The app shell itself (index.html, assets) is not data and is served; the SPA then signs in.
    assert c.get("/").status_code == 200


def test_bad_tokens_are_refused(gated):
    c = gated
    bad = {
        "garbage": "Bearer not.a.token",
        "wrong key": "Bearer " + token(_key=OTHER_PEM),
        "expired": "Bearer " + token(exp=int(time.time()) - 120),
        "other issuer": "Bearer " + token(iss="https://cognito-idp.us-east-1.amazonaws.com/us-east-1_Other"),
        "other client": "Bearer " + token(client_id="zzzzzzzzzzzzzzzzzzzzzzzzzz"),
        "id token": "Bearer " + token(token_use="id"),
        "basic scheme": "Basic dXNlcjpwYXNz",
        "none alg": "Bearer " + jwt.encode({"sub": "x", "iss": ISSUER, "client_id": CLIENT, "token_use": "access",
                                            "exp": int(time.time()) + 60, "iat": int(time.time())}, "", algorithm="none"),
    }
    for why, header in bad.items():
        r = c.get("/api/catalog", headers={"Authorization": header})
        assert r.status_code == 401, (why, r.status_code, r.text)
        assert "Bearer" in r.headers["WWW-Authenticate"], why


def test_a_valid_access_token_reads_and_the_operator_group_writes(gated, monkeypatch):
    c = gated
    reader = {"Authorization": "Bearer " + token(**{"cognito:groups": []})}
    operator = {"Authorization": "Bearer " + token()}
    assert c.get("/api/catalog", headers=reader).status_code == 200
    assert c.get("/api/catalog", headers=operator).status_code == 200
    # A reader may not change anything: 403 with the reason, before the route runs.
    r = c.delete("/api/runs/does-not-matter", headers=reader)
    assert r.status_code == 403 and "operators" in r.json()["detail"]
    r = c.post("/api/plan", json={"use_case": "standard", "models": ["m1-qwen2.5-3b"]}, headers=reader)
    assert r.status_code == 403
    # An operator reaches the route (the plan runs and answers normally).
    r = c.post("/api/plan", json={"use_case": "standard", "models": ["m1-qwen2.5-3b"],
                                  "gpu": {"hardware": "t1-g6", "instance_type": "g6.xlarge", "precisions": ["bf16"]},
                                  "cpus": []}, headers=operator)
    assert r.status_code == 200 and "series" in r.json()


def test_client_id_comes_from_ssm_when_not_in_the_environment(gated, monkeypatch):
    monkeypatch.delenv("AMXBENCH_COGNITO_CLIENT_ID")
    monkeypatch.setenv("AMXBENCH_COGNITO_CLIENT_ID_PARAM", "/amx-gpu-bench/webapp/cognito-client-id")
    auth.reset_for_tests()
    monkeypatch.setattr(auth, "_jwks", lambda: _Jwks())
    calls = []

    class Ssm:
        def get_parameter(self, Name):
            calls.append(Name)
            if len(calls) == 1:
                raise RuntimeError("ParameterNotFound")       # the service started before the client existed
            return {"Parameter": {"Value": CLIENT}}

    monkeypatch.setattr(auth.aws if hasattr(auth, "aws") else __import__("webui.api.aws", fromlist=["aws"]),
                        "client", lambda name: Ssm())
    c = gated
    # Not registered yet: the gate stays closed (503, not 500, not open) and says to retry.
    r = c.get("/api/catalog", headers={"Authorization": "Bearer " + token()})
    assert r.status_code == 503 and "try again" in r.json()["detail"]
    assert c.get("/api/auth/config").json()["ready"] is False
    auth._state["client_id_checked"] = 0.0                      # past the 10 s back-off
    r = c.get("/api/catalog", headers={"Authorization": "Bearer " + token()})
    assert r.status_code == 200 and calls == ["/amx-gpu-bench/webapp/cognito-client-id"] * 2
    assert c.get("/api/auth/config").json()["client_id"] == CLIENT


def test_security_headers_on_every_response(gated):
    c = gated
    for path, status in (("/api/health", 200), ("/api/runs", 401), ("/", 200)):
        r = c.get(path)
        assert r.status_code == status, path
        csp = r.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp and "object-src 'none'" in csp
        assert "https://amx-gpu-bench-123.auth.us-east-1.amazoncognito.com" in csp      # token endpoint
        assert "https://cognito-idp.us-east-1.amazonaws.com" in csp                       # discovery + JWKS
        assert r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["X-Frame-Options"] == "DENY"
        assert r.headers["Strict-Transport-Security"].startswith("max-age=31536000")
    assert c.get("/api/health").headers["Cache-Control"] == "no-store"


def test_public_bind_is_refused_without_a_pool(monkeypatch):
    from webui.api import main

    monkeypatch.delenv("AMXBENCH_COGNITO_USER_POOL_ID", raising=False)
    monkeypatch.setenv("AMXBENCH_BIND", "0.0.0.0")                   # noqa: S104 - the refused case
    auth.reset_for_tests()
    with pytest.raises(SystemExit, match="refusing to bind 0.0.0.0"):
        main.bind_address()
    monkeypatch.setenv("AMXBENCH_BIND", "127.0.0.1")
    assert main.bind_address() == ("127.0.0.1", 8601)
    monkeypatch.setenv("AMXBENCH_BIND", "0.0.0.0")                   # noqa: S104
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("AMXBENCH_COGNITO_USER_POOL_ID", POOL)
    auth.reset_for_tests()
    assert main.bind_address() == ("0.0.0.0", 8080)                  # noqa: S104 - with the pool, the container bind
    auth.reset_for_tests()


def test_spa_route_never_serves_outside_dist():
    """Holmes CSR finding 2026-11-16: the catch-all route joined the URL path onto dist/ and served
    any file it landed on. Paths that resolve outside dist/ now get the app shell, like any
    unknown route; real assets still serve."""
    from webui.api import main

    if not main.DIST.exists():
        pytest.skip("webui/dist is not built")
    c = TestClient(main.app)
    index = (main.DIST / "index.html").read_bytes()
    for p in ("/../pyproject.toml", "/..%2F..%2Fpyproject.toml", "/assets/../../pyproject.toml", "/%2e%2e/%2e%2e/README.md",
              "//etc/passwd", "/runs/../../../.env"):
        r = c.get(p)
        assert r.status_code == 200 and r.content == index, p
    asset = next((main.DIST / "assets").glob("*.js"))
    r = c.get(f"/assets/{asset.name}")
    assert r.status_code == 200 and r.content == asset.read_bytes()
    assert c.get("/runs").content == index                      # client-side route
