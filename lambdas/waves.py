"""Level state machine steps around the workers (design 9.1): PlanWave and WaveCheck.

Level item in DynamoDB (RUN#<id> / LEVEL#<level_key>): completed, errors (atomic counters
from the workers), waves, and for accuracy levels next_index and retry_ranges.

Perf levels plan per-worker quotas that sum to the remaining requests. Accuracy levels
(every pool item exactly once, design 24.6) hand each worker a contiguous index range;
a worker that runs out of time returns its unprocessed range, which the next wave plans
first.
"""

from __future__ import annotations

import math
import time
from typing import Any

from lambdas import common

BARRIER_S = 20
PASS_THROUGH = ("run_id", "cell_key", "level_key", "conc", "repeat", "endpoint", "api_path", "api_task",
                "payload", "pool", "batch", "request_timeout_s", "keep_output", "health_path", "target_id",
                "duration_s")


def _level(run_id: str, level_key: str) -> dict:
    item = common.table().get_item(Key={"pk": f"RUN#{run_id}", "sk": f"LEVEL#{level_key}"}).get("Item") or {}
    return common.plain(item)


def _split(total: int, parts: int) -> list[int]:
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def plan_wave(event: dict, context: Any = None) -> dict:
    lvl = event["level"]
    run_id, key = lvl["run_id"], lvl["level_key"]
    wave = int(event.get("wave", 0))
    c = int(lvl["conc"])
    state = _level(run_id, key)
    start_at_ns = int((time.time() + BARRIER_S) * 1e9)
    base = {k: lvl[k] for k in PASS_THROUGH if k in lvl}
    base.update(mode="measure", wave=wave, start_at_ns=start_at_ns)
    items: list[dict] = []
    if lvl.get("accuracy"):
        size = int(lvl["pool"]["size"])
        nxt = int(state.get("next_index", 0))
        todo = [list(map(int, r)) for r in state.get("retry_ranges", [])] + ([[nxt, size]] if nxt < size else [])
        remaining = sum(b - a for a, b in todo)
        per_worker = max(1, min(math.ceil(remaining / c), int(lvl.get("max_items_per_worker", remaining or 1))))
        assigned: list[list[list[int]]] = [[] for _ in range(c)]
        leftover: list[list[int]] = []
        w = filled = 0
        for a, b in todo:
            while a < b:
                if w >= c:
                    leftover.append([a, b])
                    break
                take = min(per_worker - filled, b - a)
                assigned[w].append([a, a + take])
                a += take
                filled += take
                if filled >= per_worker:
                    w, filled = w + 1, 0
        retry_left = [r for r in leftover if r[0] < nxt]
        main_left = [r for r in leftover if r[0] >= nxt]
        new_next = main_left[0][0] if main_left else max(nxt, size)
        items = [{**base, "worker": k, "ranges": rs} for k, rs in enumerate(assigned) if rs]
        common.table().update_item(
            Key={"pk": f"RUN#{run_id}", "sk": f"LEVEL#{key}"},
            UpdateExpression="SET next_index = :n, retry_ranges = :r, waves = :w",
            ExpressionAttributeValues={":n": new_next, ":r": retry_left, ":w": wave + 1})
        return {"items": items, "wave": wave, "remaining": remaining, "start_at_ns": start_at_ns}
    if lvl.get("duration_s"):
        items = [{**base, "worker": k, "quota": None} for k in range(c)]
        remaining = None
    else:
        remaining = max(0, int(lvl["quota"]) - int(state.get("completed", 0)))
        for k, q in enumerate(_split(remaining, c)):
            if q:
                items.append({**base, "worker": k, "quota": q})
    common.table().update_item(Key={"pk": f"RUN#{run_id}", "sk": f"LEVEL#{key}"},
                               UpdateExpression="SET waves = :w", ExpressionAttributeValues={":w": wave + 1})
    return {"items": items, "wave": wave, "remaining": remaining, "start_at_ns": start_at_ns}


def wave_check(event: dict, context: Any = None) -> dict:
    lvl = event["level"]
    wave = int(event.get("wave", 0))
    state = _level(lvl["run_id"], lvl["level_key"])
    max_waves = int(lvl.get("max_waves", 20))
    if lvl.get("duration_s"):
        more = False
    elif lvl.get("accuracy"):
        more = bool(state.get("retry_ranges")) or int(state.get("next_index", 0)) < int(lvl["pool"]["size"])
    else:
        more = int(state.get("completed", 0)) < int(lvl["quota"])
    if more and wave + 1 >= max_waves:
        common.set_status(lvl["run_id"], f"LEVEL#{lvl['level_key']}", "incomplete", reason="max_waves")
        more = False
    return {"more": more, "wave": wave + 1, "completed": int(state.get("completed", 0)),
            "errors": int(state.get("errors", 0))}


def init_level(event: dict, context: Any = None) -> dict:
    """Reset the level item before the first wave (re-runs of a level start from zero)."""
    lvl = event["level"]
    common.table().put_item(Item={"pk": f"RUN#{lvl['run_id']}", "sk": f"LEVEL#{lvl['level_key']}",
                                  "status": "running", "completed": 0, "errors": 0, "waves": 0,
                                  "next_index": 0, "retry_ranges": [], "cell_key": lvl["cell_key"],
                                  "started_at": common.now_iso()})
    return {"wave": 0}
