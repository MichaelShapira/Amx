"""Browser security headers for every response (local and cloud).

The app is a single-page React application served by the API: scripts and styles come from
/assets, fonts from Google Fonts (index.html), data from /api, and in the cloud deployment the
browser also talks to Cognito (token endpoint on the hosted-login domain, JWKS/discovery on the
issuer). Everything else is refused by the Content-Security-Policy. The page can never be framed,
and transport security is pinned for a year once the app is served over HTTPS (the cloud
deployment; the local app is plain http on loopback, where HSTS would be wrong).
"""

from __future__ import annotations

from starlette.datastructures import MutableHeaders

FONT_CSS = "https://fonts.googleapis.com"
FONT_FILES = "https://fonts.gstatic.com"


def csp(extra_connect: list[str]) -> str:
    connect = " ".join(["'self'", *extra_connect])
    return (f"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' {FONT_CSS}; "
            f"font-src 'self' {FONT_FILES}; img-src 'self' data:; connect-src {connect}; "
            f"form-action 'self' {' '.join(extra_connect)}; frame-ancestors 'none'; base-uri 'self'; "
            "object-src 'none'")


class SecurityHeadersMiddleware:
    """`policy()` returns (csp, hsts) for the current configuration; evaluated per response so the
    policy follows the Cognito settings wherever they are set (environment at start, tests)."""

    def __init__(self, app, *, policy):
        self.app = app
        self.policy = policy

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        content_security_policy, hsts = self.policy()

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                h = MutableHeaders(scope=message)
                h.setdefault("Content-Security-Policy", content_security_policy)
                h.setdefault("X-Content-Type-Options", "nosniff")
                h.setdefault("X-Frame-Options", "DENY")
                h.setdefault("Referrer-Policy", "same-origin")
                h.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
                if hsts:
                    h.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
                if scope.get("path", "").startswith("/api/"):
                    h.setdefault("Cache-Control", "no-store")
            await send(message)

        return await self.app(scope, receive, send_with_headers)
