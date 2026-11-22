"""Load-generator worker (design 9.2): one closed-loop client per Lambda environment.

Event (one Distributed Map item, built by plan_wave):
  mode            "measure" or "warm" (warm: sleep 3 s so C environments exist at once)
  run_id, cell_key, level_key, conc, repeat, wave, worker
  endpoint        http://<private-ip>:8000
  api_path        /v1/completions, /v3/completions, /classify, /detect, /v1/embeddings or /v1/systemone
  api_task        generate | classify | detect | embed | decide
  payload         request template (model, max_tokens, temperature, seed, ignore_eos, stop, choices)
  pool            {"bucket", "key", "size"}: request pool (JSONL, one prompt or input per line)
  quota           requests for this worker (perf levels), or
  ranges          [[start, stop], ...] pool indices to process exactly once (accuracy levels)
  duration_s      alternative to quota
  batch           inputs per classify request (default 1)
  start_at_ns     wall-clock barrier
  request_timeout_s, keep_output, health_path
Requests are never retried. Records go to raw/<cell>/<level>/w<wave>/worker-<k>.jsonl.gz.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from bench import client as bc
from lambdas import common

COLD = True
_POOLS: dict[tuple[str, str], list[dict]] = {}
BUDGET_MARGIN_S = 30


def load_pool(bucket: str, key: str) -> list[dict]:
    k = (bucket, key)
    if k not in _POOLS:
        _POOLS[k] = common.read_jsonl(key, bucket_name=bucket)
    return _POOLS[k]


def _inputs(line: dict) -> list[str]:
    x = line.get("input")
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [x]
    raise ValueError(f"pool line {line.get('id')} has no input")


def send(cl: httpx.Client, ev: dict, lines: list[dict]) -> dict:
    url = ev["endpoint"] + ev["api_path"]
    timeout = float(ev.get("request_timeout_s", 600))
    keep = bool(ev.get("keep_output"))
    expected = sum(int(x.get("prompt_tokens_expected") or x.get("tokens") or 0) for x in lines) or None
    if ev["api_task"] == "generate":
        line = lines[0]
        body = {**ev["payload"], "prompt": line["prompt"]}
        rec = bc.stream_completion(cl, url, body, timeout_s=timeout, keep_text=keep)
    elif ev["api_task"] == "decide":
        # Strands Decider: one state per request, `questions` of its typed questions riding on it. The pool line
        # carries the state and the questions in the server's own shape (bench/decision.request_body).
        from bench import decision
        line = lines[0]
        body = decision.request_body(line, int(ev["payload"].get("questions", 1)))
        rec, resp = bc.post_json(cl, url, body, timeout_s=timeout)
        if resp:
            rec["prompt_tokens"] = (resp.get("usage") or {}).get("input_tokens")
            if keep:
                rec["output"] = {"answers": resp.get("answers"), "latency_ms": resp.get("latency_ms")}
        rec["n_inputs"] = len(body["questions"])
    else:
        inputs = [s for line in lines for s in _inputs(line)]
        rec, resp = bc.post_json(cl, url, {"model": ev["payload"]["model"], "input": inputs}, timeout_s=timeout)
        if keep and resp:
            rec["output"] = [{k: d.get(k) for k in ("label", "probs", "spans", "category_max", "embedding") if k in d}
                             for d in resp.get("data", [])]
        rec["n_inputs"] = len(inputs)
    rec["prompt_tokens_expected"] = expected
    rec["prompt_id"] = lines[0].get("id") if len(lines) == 1 else [x.get("id") for x in lines]
    item_ids = [x["item_id"] for x in lines if x.get("item_id")]
    if item_ids:
        rec["item_ids"] = item_ids
    rec["batch_size"] = len(lines) if ev["api_task"] not in ("generate", "decide") else 1
    if ev["api_task"] == "decide":
        rec["batch_size"] = rec["n_inputs"]          # items per request = questions answered
    rec["content_bytes"] = sum(int(x.get("bytes") or 0) for x in lines) or None
    rec["content_chars"] = sum(int(x.get("chars") or 0) for x in lines) or None    # translation: source characters
    return rec


def _indices(ev: dict, pool_len: int):
    """Yield pool index lists: deterministic striding for perf levels, exact ranges for accuracy."""
    b = int(ev.get("batch", 1))
    if ev.get("ranges"):
        for start, stop in ev["ranges"]:
            for i in range(start, stop, b):
                yield list(range(i, min(i + b, stop)))
        return
    c, k = int(ev["conc"]), int(ev["worker"])
    i = 0
    while True:
        base = (k + i * c) * b
        yield [(base + j) % pool_len for j in range(b)]
        i += 1


def handler(event: dict, context: Any = None) -> dict:
    global COLD
    if event.get("mode") == "warm":
        time.sleep(3)
        return {"warm": True}
    cold = COLD
    pool = load_pool(event["pool"]["bucket"], event["pool"]["key"])
    quota = event.get("quota")
    duration_ns = int(event["duration_s"] * 1e9) if event.get("duration_s") else None
    timeout_s = float(event.get("request_timeout_s", 600))
    records: list[dict] = []
    unprocessed: list[list[int]] = []
    with httpx.Client(http2=False, limits=httpx.Limits(max_connections=1, max_keepalive_connections=1)) as cl:
        try:   # open the connection before the barrier
            cl.get(event["endpoint"] + event.get("health_path", "/health"), timeout=10)
        except httpx.HTTPError:
            pass
        start_at = int(event["start_at_ns"])
        delay = (start_at - time.time_ns()) / 1e9
        if delay > 0:
            time.sleep(delay)
        late_ms = (time.time_ns() - start_at) / 1e6
        for n, idx in enumerate(_indices(event, len(pool))):
            if quota is not None and not event.get("ranges") and n >= int(quota):
                break
            if duration_ns is not None and time.time_ns() >= start_at + duration_ns:
                break
            if context is not None and context.get_remaining_time_in_millis() < (timeout_s + BUDGET_MARGIN_S) * 1000:
                if event.get("ranges"):
                    unprocessed = _rest(event, idx)
                break
            rec = send(cl, event, [pool[i] for i in idx])
            rec.update(run_id=event["run_id"], cell=event["cell_key"], level=event["level_key"],
                       conc=int(event["conc"]), repeat=int(event.get("repeat", 0)), wave=int(event["wave"]),
                       worker=int(event["worker"]), req=n, start_at_ns=start_at, late_start_ms=round(late_ms, 3),
                       cold_env=cold and n == 0)
            records.append(rec)
            if endpoint_down(records):
                break
    errors = sum(1 for r in records if r.get("error"))
    key = (f"{common.level_raw_prefix(event['run_id'], event['cell_key'], event['level_key'])}"
           f"/w{int(event['wave']):03d}/worker-{int(event['worker']):04d}.jsonl.gz")
    common.put_jsonl_gz(key, records)
    COLD = False
    if endpoint_down(records):
        # Do not count these toward the quota: the level is re-run once the target is back
        # (Spot interruption) and must not end as "done" with a 100% error rate. The flag on
        # the level item is how MarkLevelFailed learns the cause (the wave Map hides it).
        common.table().update_item(Key={"pk": f"RUN#{event['run_id']}", "sk": f"LEVEL#{event['level_key']}"},
                                   UpdateExpression="SET endpoint_down = :t", ExpressionAttributeValues={":t": True})
        raise EndpointUnreachable(f"{DOWN_STREAK} consecutive connection failures on {event['endpoint']}: "
                                  f"{records[-1].get('error')}")
    _count(event, len(records), errors, unprocessed)
    return {"worker": event["worker"], "n": len(records), "errors": errors, "late_ms": round(late_ms, 1),
            "unprocessed": sum(b - a for a, b in unprocessed)}


DOWN_STREAK = 5
# Transport-level failures only. A request timeout is a slow engine, which is a measurement.
CONNECTION_ERRORS = ("client_ConnectError", "client_ConnectTimeout", "client_RemoteProtocolError",
                     "client_ReadError")


class EndpointUnreachable(RuntimeError):
    """The target stopped answering at the transport level (interrupted or dead instance)."""


def endpoint_down(records: list[dict]) -> bool:
    """True when the last DOWN_STREAK records all failed at the connection level. HTTP errors
    (an overloaded engine answering 5xx) do not count: those are measured, not recovered from."""
    if len(records) < DOWN_STREAK:
        return False
    return all((r.get("error") or "").startswith(CONNECTION_ERRORS) for r in records[-DOWN_STREAK:])


def _rest(event: dict, current: list[int]) -> list[list[int]]:
    """Ranges not yet processed, from the current batch onwards."""
    first = current[0]
    out = []
    for start, stop in event["ranges"]:
        if stop <= first:
            continue
        out.append([max(start, first), stop])
    return out


def _count(event: dict, n: int, errors: int, unprocessed: list[list[int]]) -> None:
    expr = "ADD completed :n, errors :e SET last_worker_at = :t"
    vals: dict[str, Any] = {":n": n, ":e": errors, ":t": int(time.time())}
    if unprocessed:
        expr += ", retry_ranges = list_append(if_not_exists(retry_ranges, :empty), :r)"
        vals[":r"] = unprocessed
        vals[":empty"] = []
    common.table().update_item(Key={"pk": f"RUN#{event['run_id']}", "sk": f"LEVEL#{event['level_key']}"},
                               UpdateExpression=expr, ExpressionAttributeValues=vals)
    if event.get("target_id"):
        common.touch_target(event["target_id"], event["run_id"])
