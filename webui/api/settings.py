"""Setup: what an operator configures at run time instead of at deploy time.

* The Hugging Face token: written to the deployment's Secrets Manager secret (the hosts and the
  pre-check already read it from there). The API only ever writes it and reports when it was last
  set; the value never comes back out through the web app.
* The reasoning model: the Amazon Bedrock model that writes hardware proposals and executive
  summaries. Chosen from the models the account can call through the Bedrock Converse API
  (bedrock-runtime), limited to Anthropic Claude and OpenAI models, which is also the extent of
  the task role's bedrock:InvokeModel permission. Default: the newest Claude Opus.
* Where it lives: one DynamoDB item (pk SETTINGS, sk app) in the run table, so the Lambdas and a
  restarted web app see the same values; no new store.

The model catalog itself (which Hugging Face models the benchmark offers) is bench/catalog_store.py.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

from webui.api import aws

SETTINGS_KEY = {"pk": "SETTINGS", "sk": "app"}
HF_TOKEN_RE = re.compile(r"^hf_[A-Za-z0-9]{20,}$")
ALLOWED_PROVIDERS = ("anthropic", "openai")       # what the task role may invoke (infra/stacks/webapp.py)
_cache: dict[str, Any] = {}          # the settings item (30 s)
_models: dict[str, Any] = {}         # the Bedrock model list (1 h)


def hf_secret_arn() -> str | None:
    return os.environ.get("AMXBENCH_HF_SECRET_ARN", "").strip() or None


# ------------------------------------------------------------------------- the settings item

def get() -> dict:
    """The stored settings (reasoning model and when it was set); empty when never saved."""
    if _cache.get("at", 0) > time.time() - 30:
        return _cache["item"]
    from lambdas.common import plain

    item = aws.table().get_item(Key=SETTINGS_KEY).get("Item")
    out = {k: v for k, v in plain(item).items() if k not in SETTINGS_KEY} if item else {}
    _cache.update(item=out, at=time.time())
    return out


def put(fields: dict, who: str | None) -> dict:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    expr, names, values = [], {}, {}
    for i, (k, v) in enumerate(fields.items()):
        expr.append(f"#k{i} = :v{i}")
        names[f"#k{i}"], values[f":v{i}"] = k, v
    expr += ["#u = :u", "#w = :w"]
    names.update({"#u": "updated_at", "#w": "updated_by"})
    values.update({":u": now, ":w": who or "unknown"})
    aws.table().update_item(Key=SETTINGS_KEY, UpdateExpression="SET " + ", ".join(expr),
                            ExpressionAttributeNames=names, ExpressionAttributeValues=values)
    _cache.clear()
    return get()


# ------------------------------------------------------------------------- Hugging Face token

def hf_token_status() -> dict:
    """Whether a token is stored and when it last changed; never the token."""
    arn = hf_secret_arn()
    if not arn:
        return {"configured": False, "available": False, "reason": "no secret is wired to this deployment"}
    try:
        d = aws.client("secretsmanager").describe_secret(SecretId=arn)
    except Exception as e:  # noqa: BLE001 - say what happened, with the error code only
        return {"configured": False, "available": False, "reason": f"the secret cannot be read: {type(e).__name__}"}
    changed = d.get("LastChangedDate")
    # The stack creates the secret with a placeholder value, so "a value exists" is not "a token is stored".
    return {"configured": hf_token_for_onboarding() is not None, "available": True, "name": d.get("Name"),
            "last_changed": changed.strftime("%Y-%m-%d %H:%M UTC") if changed else None}


def set_hf_token(token: str, who: str | None) -> dict:
    token = (token or "").strip()
    if not HF_TOKEN_RE.match(token):
        raise ValueError("That does not look like a Hugging Face token: it starts with hf_ followed by letters and digits "
                         "(Hugging Face → Settings → Access Tokens).")
    arn = hf_secret_arn()
    if not arn:
        raise RuntimeError("no secret is wired to this deployment")
    aws.client("secretsmanager").put_secret_value(SecretId=arn, SecretString=token)
    put({"hf_token_set_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, who)
    return hf_token_status()


def hf_token_for_onboarding() -> str | None:
    """The token, for the onboarding job's own download of a gated model's tokenizer. Never
    returned to a client; callers must not log it."""
    arn = hf_secret_arn()
    if not arn:
        return None
    try:
        v = aws.client("secretsmanager").get_secret_value(SecretId=arn).get("SecretString") or ""
    except Exception:  # noqa: BLE001 - not set yet
        return None
    v = v.strip()
    return v if HF_TOKEN_RE.match(v) else None          # the placeholder the stack seeds is not a token


# ------------------------------------------------------------------------- reasoning model

def list_reasoning_models(bedrock=None) -> list[dict]:
    """Bedrock text models from Anthropic and OpenAI the account can call: on-demand ones by
    model id, profile-only ones through their us. cross-Region inference profile. Newest first."""
    if _models.get("at", 0) > time.time() - 3600:
        return _models["models"]
    bedrock = bedrock or aws.client("bedrock")
    profiles = {p["inferenceProfileId"] for p in bedrock.list_inference_profiles()["inferenceProfileSummaries"]}
    out = []
    for provider in ALLOWED_PROVIDERS:
        for m in bedrock.list_foundation_models(byProvider=provider, byOutputModality="TEXT")["modelSummaries"]:
            if m.get("modelLifecycle", {}).get("status") != "ACTIVE":
                continue
            mid = m["modelId"]
            if "embed" in mid or "rerank" in mid:
                continue
            types = set(m.get("inferenceTypesSupported") or [])
            call_id = mid if "ON_DEMAND" in types else (f"us.{mid}" if f"us.{mid}" in profiles else None)
            if not call_id:
                continue
            out.append({"model_id": mid, "call_id": call_id, "name": m.get("modelName", mid), "provider": provider,
                        "family": _family(mid), "sort": _sort_key(mid)})
    out.sort(key=lambda x: (x["provider"] != "anthropic", x["sort"]), reverse=False)
    out.sort(key=lambda x: x["sort"], reverse=True)
    for x in out:
        x.pop("sort", None)
    _models.update(models=out, at=time.time())
    return out


def _family(mid: str) -> str:
    for f in ("opus", "sonnet", "haiku", "gpt-oss", "gpt"):
        if f in mid:
            return f
    return mid.split(".")[-1].split("-")[0]


def _sort_key(mid: str) -> tuple:
    nums = [int(x) for x in re.findall(r"(?<![a-z])(\d+)(?![a-z])", mid.split(".")[-1])[:3]]
    rank = {"opus": 3, "sonnet": 2, "haiku": 1}.get(_family(mid), 0)
    return (rank, *nums, 0, 0, 0)[:4]


def default_reasoning_model(models: list[dict] | None = None) -> dict | None:
    """The newest Claude Opus the account can call, else the newest listed model."""
    models = models if models is not None else list_reasoning_models()
    opus = [m for m in models if m["family"] == "opus"]
    return (opus or models or [None])[0]


def reasoning_model(bedrock=None) -> dict:
    """The configured reasoning model if it is still callable, else the default (and say so)."""
    models = list_reasoning_models(bedrock)
    chosen = get().get("reasoning_model")
    if chosen:
        hit = next((m for m in models if m["call_id"] == chosen.get("call_id")), None)
        if hit:
            return {**hit, "source": "setup"}
    d = default_reasoning_model(models)
    if d is None:
        raise RuntimeError("no Anthropic or OpenAI text model is callable in Bedrock for this account and region")
    return {**d, "source": "default"}


def set_reasoning_model(call_id: str, who: str | None, bedrock=None) -> dict:
    hit = next((m for m in list_reasoning_models(bedrock) if m["call_id"] == call_id), None)
    if not hit:
        raise ValueError("Pick a model from the list; that id is not one this deployment can call.")
    put({"reasoning_model": {"call_id": hit["call_id"], "model_id": hit["model_id"], "name": hit["name"]}}, who)
    return reasoning_model(bedrock)


def reset_for_tests() -> None:
    _cache.clear()
    _models.clear()
