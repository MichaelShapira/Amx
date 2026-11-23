"""The Hugging Face token: one place that knows what it looks like, where it lives and how to check it.

The token is needed only for GATED models (the Gemma family, EmbeddingGemma: licenses accepted on huggingface.co,
downloads need a token of that account). Non-gated models (Qwen3, RoBERTa, the Strands Deciders' adapters) run
without one. It lives in the Core stack's Secrets Manager secret `amx-gpu-bench/hf-token`, which the stack creates
with the placeholder `unset`; three paths write the real value through this module: `infra/deploy.sh` at deploy
time (HF_TOKEN in the environment or --hf-token-stdin), `bench hf-token` from a shell, and the web app's Setup page.
The hosts read the secret through their instance role when they stage a gated model; preflight checks that the token
reaches every gated model's pinned revision and refuses the run with a plain sentence otherwise. The value is never
printed, logged, put in CloudFormation or committed.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from bench.deployment import HF_TOKEN_UNSET

HF_TOKEN_RE = re.compile(r"^hf_[A-Za-z0-9]{20,}$")
WHOAMI_URL = "https://huggingface.co/api/whoami-v2"


def looks_like_token(value: str | None) -> bool:
    """A real token, as opposed to the stack's placeholder or an empty slot."""
    v = (value or "").strip()
    return bool(v) and v != HF_TOKEN_UNSET and bool(HF_TOKEN_RE.match(v))


def status(sm, secret_id: str) -> dict:
    """Whether a token is stored, and when the secret last changed; never the value."""
    d = sm.describe_secret(SecretId=secret_id)
    try:
        value = sm.get_secret_value(SecretId=secret_id).get("SecretString")
    except Exception:  # noqa: BLE001 - a slot with no version yet
        value = None
    changed = d.get("LastChangedDate")
    return {"configured": looks_like_token(value), "name": d.get("Name"),
            "last_changed": changed.strftime("%Y-%m-%d %H:%M UTC") if changed else None}


def store(sm, secret_id: str, token: str) -> dict:
    """Validate the shape and write the token; returns status()."""
    token = (token or "").strip()
    if not HF_TOKEN_RE.match(token):
        raise ValueError("That does not look like a Hugging Face token: it starts with hf_ followed by letters and digits "
                         "(Hugging Face -> Settings -> Access Tokens; a read token is enough).")
    sm.put_secret_value(SecretId=secret_id, SecretString=token)
    return status(sm, secret_id)


def verify(token: str, timeout: float = 20) -> dict:
    """Ask Hugging Face whose token this is: {ok, name, error}. Never returns or logs the token."""
    req = urllib.request.Request(WHOAMI_URL, headers={"Authorization": f"Bearer {token.strip()}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https URL  # nosec B310  # nosemgrep (urllib: constant URL)
            body = json.loads(r.read().decode())
        return {"ok": True, "name": body.get("name") or body.get("fullname"), "error": None}
    except urllib.error.HTTPError as e:
        return {"ok": False, "name": None, "error": f"Hugging Face answered HTTP {e.code}" + (" (invalid or revoked token)" if e.code == 401 else "")}
    except Exception as e:  # noqa: BLE001 - network: say so, do not guess
        return {"ok": False, "name": None, "error": f"{type(e).__name__}"}
