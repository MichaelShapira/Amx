"""Level and cell side tasks: warm-up, quality, perf counters, AMX and memory probes
(design 9.1, 13, 14, 15). Each function is its own Lambda handler."""

from __future__ import annotations

import gzip
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from bench import amx
from lambdas import common
from lambdas.worker import load_pool, send

# ---------------------------------------------------------------------------- warm-up

LAMBDA_BUDGET_S = 15 * 60
SAFETY = 0.6


def _drive(lvl: dict, lines: list[list[dict]], conc: int, keep: bool = False) -> list[dict]:
    ev = {**lvl, "keep_output": keep}

    def one(batch: list[dict]) -> dict:
        with httpx.Client(limits=httpx.Limits(max_connections=1, max_keepalive_connections=1)) as cl:
            return send(cl, ev, batch)

    with ThreadPoolExecutor(max_workers=conc) as ex:
        return list(ex.map(one, lines))


def warmup(event: dict, context: Any = None) -> dict:
    """Send warmup_requests (>= 20) at concurrency min(C, 8); results are excluded from metrics.
    Also estimates how many items one worker can finish inside a Lambda invocation, which
    caps accuracy-level chunks."""
    lvl = event["level"]
    pool = load_pool(lvl["pool"]["bucket"], lvl["pool"]["key"])
    n = int(lvl.get("warmup_requests", 20))
    b = int(lvl.get("batch", 1))
    batches = [[pool[(i * b + j) % len(pool)] for j in range(b)] for i in range(n)]
    recs = _drive(lvl, batches, min(int(lvl["conc"]), 8))
    common.put_jsonl_gz(f"runs/{lvl['run_id']}/warmup/{lvl['cell_key']}/{lvl['level_key']}.jsonl.gz", recs)
    good = [r["send_to_end_ns"] / 1e9 for r in recs if not r.get("error")]
    errors = len(recs) - len(good)
    if errors == len(recs):
        raise RuntimeError(f"all {len(recs)} warm-up requests failed: {recs[0].get('error')}")
    p50 = statistics.median(good)
    # Under load a request takes longer than at warm-up concurrency; budget with a margin.
    slow = p50 * max(1.0, int(lvl["conc"]) / max(1, min(int(lvl["conc"]), 8)))
    per_worker = max(1, int((LAMBDA_BUDGET_S - float(lvl.get("request_timeout_s", 600)) - 60) * SAFETY / max(slow, 1e-3)))
    return {"n": len(recs), "errors": errors, "p50_s": round(p50, 4), "max_items_per_worker": per_worker}


# ---------------------------------------------------------------------------- quality

def quality(event: dict, context: Any = None) -> dict:
    """Run the quality set once per cell (design 15) and store outputs for aggregate_run."""
    lvl = event["quality"]
    pool = load_pool(lvl["pool"]["bucket"], lvl["pool"]["key"])
    recs = _drive(lvl, [[x] for x in pool], int(lvl.get("conc", 8)), keep=True)
    rows = []
    for line, r in zip(pool, recs, strict=True):
        row = {"id": line["id"], "error": r.get("error"), "completion_tokens": r.get("completion_tokens")}
        if "output_text" in r:
            row["text"] = r["output_text"]
        if "output" in r:
            row["label"] = r["output"][0].get("label") if r["output"] else None
            row["probs"] = r["output"][0].get("probs") if r["output"] else None
        rows.append(row)
    key = f"runs/{lvl['run_id']}/quality/{lvl['cell_key']}/outputs.jsonl.gz"
    common.put_jsonl_gz(key, rows)
    return {"n": len(rows), "errors": sum(1 for x in rows if x["error"]), "key": key}


# ---------------------------------------------------------------------------- perf counters

def perf_counters(event: dict, context: Any = None) -> dict:
    lvl = event["level"]
    action = event["action"]
    cmd = f"{common.bench_home()}/perf_ctl.sh {action} {common.bucket()} {lvl['run_id']} {lvl['level_key']}"
    res = common.ssm_run(lvl["instance_id"], [cmd], timeout_s=120, comment=f"perf {action}")
    if res["status"] != "Success":
        # A missing counter is not fatal: the verdict falls back to the engine trace.
        return {"ok": False, "status": res["status"], "stderr": res["stderr"][-500:]}
    return {"ok": True, "counts": amx.parse_perf_csv(res["stdout"]) if action == "stop" else None}


# ---------------------------------------------------------------------------- AMX and memory probes

def amx_probe(event: dict, context: Any = None) -> dict:
    run_id, cell = event["run_id"], event["cell"]
    res = common.ssm_run(event["instance_id"], [f"{common.bench_home()}/probe_amx.sh {cell['spec_uri']}"],
                         timeout_s=3600, comment=f"amx {cell['cell_key']}"[:100])
    if res["status"] != "Success":
        raise RuntimeError(f"AMX probe failed for {cell['cell_id']}: {res['stderr'][-500:]}")
    prefix = f"runs/{run_id}/amx/{cell['cell_key']}"
    ev: dict[str, Any] = {"flags": sorted(amx.parse_cpu_flags(common.get_bytes(f"{prefix}/flags.txt").decode()))}
    if cell["engine"] in ("vllm", "thin-torch"):
        raw = common.get_bytes(f"{prefix}/trace.log.gz")
        text = gzip.decompress(raw).decode(errors="replace") if raw else ""
        ev["trace"] = amx.parse_onednn_verbose(text)
        ev["dispatch"] = amx.parse_vllm_dispatch(text)
    else:
        ev["trace"] = amx.parse_ov_probe(common.get_bytes(f"{prefix}/probe_ov.json").decode())
    common.put_json(f"{prefix}/evidence.json", ev)
    return {"ok": True, "gemm_amx": ev["trace"].get("gemm_amx"), "gemm_total": ev["trace"].get("gemm_total")}


def membw_probe(event: dict, context: Any = None) -> dict:
    res = common.ssm_run(event["instance_id"],
                         [f"{common.bench_home()}/probe_membw.sh {common.bucket()} {event['run_id']}"],
                         timeout_s=1800, comment="membw")
    return {"ok": res["status"] == "Success", "stdout": res["stdout"][-1000:]}
