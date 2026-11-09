"""Load-generator HTTP client (design section 9.3).

Durations use the monotonic clock; one wall-clock anchor per call places requests on a
shared timeline. Usage (token counts) always comes from the server; a missing usage chunk
is an error, the harness never counts tokens itself.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import httpx


def _now() -> tuple[int, int]:
    return time.time_ns(), time.perf_counter_ns()


def _base_record(wall0: int, mono0: int) -> dict[str, Any]:
    return {"wall_send_ns": wall0, "send_to_first_ns": None, "send_to_last_ns": None,
            "send_to_end_ns": None, "prompt_tokens": None, "completion_tokens": None,
            "n_events": 0, "http_status": None, "error": None, "timeout": False}


def parse_sse_line(line: str) -> dict | str | None:
    """Return the decoded JSON event, the string '[DONE]', or None for non-data lines."""
    if not line or not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return "[DONE]"
    return json.loads(payload)


def event_text(event: dict) -> str:
    choices = event.get("choices") or []
    if not choices:
        return ""
    c = choices[0]
    if "text" in c and c["text"] is not None:
        return c["text"]
    delta = c.get("delta") or {}
    return delta.get("content") or ""


def stream_completion(client: httpx.Client, url: str, payload: dict, *, timeout_s: float,
                      keep_text: bool = False) -> dict:
    """POST a streaming completion and time first token, last token and end."""
    body = dict(payload)
    body["stream"] = True
    body.setdefault("stream_options", {"include_usage": True})
    wall0, mono0 = _now()
    rec = _base_record(wall0, mono0)
    pieces: list[str] = []
    try:
        with client.stream("POST", url, json=body, timeout=timeout_s) as resp:
            rec["http_status"] = resp.status_code
            if resp.status_code != 200:
                resp.read()
                rec["error"] = f"http_{resp.status_code}"
                rec["timeout"] = resp.status_code in (408, 504)
                rec["send_to_end_ns"] = time.perf_counter_ns() - mono0
                return rec
            for line in resp.iter_lines():
                ev = parse_sse_line(line)
                if ev is None:
                    continue
                if ev == "[DONE]":
                    break
                rec["n_events"] += 1
                text = event_text(ev)
                if text:
                    t = time.perf_counter_ns() - mono0
                    if rec["send_to_first_ns"] is None:
                        rec["send_to_first_ns"] = t
                    rec["send_to_last_ns"] = t
                    pieces.append(text)
                usage = ev.get("usage")
                if usage:
                    rec["prompt_tokens"] = usage.get("prompt_tokens")
                    rec["completion_tokens"] = usage.get("completion_tokens")
        rec["send_to_end_ns"] = time.perf_counter_ns() - mono0
    except httpx.TimeoutException:
        rec.update(error="timeout", timeout=True, send_to_end_ns=time.perf_counter_ns() - mono0)
        return rec
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        rec.update(error=f"client_{type(e).__name__}", send_to_end_ns=time.perf_counter_ns() - mono0)
        return rec
    if rec["completion_tokens"] is None:
        rec["error"] = "no_usage"
    text = "".join(pieces)
    rec["output_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    if keep_text:
        rec["output_text"] = text
    return rec


def post_json(client: httpx.Client, url: str, payload: dict, *, timeout_s: float) -> tuple[dict, dict | None]:
    """Non-streaming POST (classifiers, detectors). Returns (record, parsed body)."""
    wall0, mono0 = _now()
    rec = _base_record(wall0, mono0)
    try:
        resp = client.post(url, json=payload, timeout=timeout_s)
        rec["send_to_end_ns"] = time.perf_counter_ns() - mono0
        rec["http_status"] = resp.status_code
        if resp.status_code != 200:
            rec["error"] = f"http_{resp.status_code}"
            rec["timeout"] = resp.status_code in (408, 504)
            return rec, None
        body = resp.json()
    except httpx.TimeoutException:
        rec.update(error="timeout", timeout=True, send_to_end_ns=time.perf_counter_ns() - mono0)
        return rec, None
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        rec.update(error=f"client_{type(e).__name__}", send_to_end_ns=time.perf_counter_ns() - mono0)
        return rec, None
    usage = body.get("usage") or {}
    rec["prompt_tokens"] = usage.get("prompt_tokens")
    return rec, body


def completion_payload(model: str, prompt: str, *, max_tokens: int, temperature: float,
                       seed: int, ignore_eos: bool, stop: list[str] | None = None,
                       choices: list[str] | None = None) -> dict:
    p: dict[str, Any] = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
                         "temperature": temperature, "seed": seed}
    if ignore_eos:
        p["ignore_eos"] = True
    if stop:
        p["stop"] = stop
    if choices:
        p["structured_outputs"] = {"choice": choices}
    return p


def classify_payload(model: str, inputs: list[str] | list[list[str]]) -> dict:
    return {"model": model, "input": inputs}
