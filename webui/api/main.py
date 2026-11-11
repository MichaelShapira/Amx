"""Local API for the web UI. Binds 127.0.0.1 only; every AWS call uses the operator's own
credentials. Serves the built React app from webui/dist when present.

    .venv/bin/python -m webui.api.main            # http://127.0.0.1:8601
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import pairing, report  # noqa: E402
from bench.config import RunConfig, load_models, load_targets, resolve_cells, run_instance_types  # noqa: E402
from bench.estimate_run import estimate  # noqa: E402
from bench.plan import levels_for_cell  # noqa: E402
from bench.pools import local_pool_sizes  # noqa: E402
from webui.api import catalog as C  # noqa: E402
from webui.api import compare, propose  # noqa: E402

REGION, ACCOUNT = "us-east-1", "346399954218"
app = FastAPI(title="GPU vs AMX CPU benchmark", docs_url=None, redoc_url=None)
_CACHE: dict[str, Any] = {}


# ----------------------------------------------------------------------------- catalog and data

@app.get("/api/catalog")
def get_catalog():
    if "catalog" not in _CACHE:
        _CACHE["catalog"] = C.catalog()
    return _CACHE["catalog"]


@app.get("/api/data/{use_case}")
def get_data(use_case: str):
    return C.dataset_info(use_case)


@app.get("/api/data/{use_case}/sample")
def get_sample(use_case: str, prompt_set: str | None = None, model: str | None = None, workload: str | None = None,
               label: str | None = None, hard_case: str | None = None, n: int = 5, seed: int = 0):
    return C.sample(use_case, prompt_set=prompt_set, model=model, workload=workload, label=label, hard_case=hard_case,
                    n=min(n, 20), seed=seed)


# ----------------------------------------------------------------------------- wizard -> run config

class Wizard(BaseModel):
    """What the wizard collects. Everything else in the run config is derived here."""
    use_case: str = "standard"
    prompt_set: str = "short"
    answer_tokens: int = 128
    workloads: list[str] = ["W1", "W3"]
    test_items: int = 300
    preset: str = "inline_standard"
    depth: str = "quick"                       # quick | full | custom
    requests: int | None = None
    passes: int | None = None
    models: list[str] = []
    precision: str = "bf16"
    gpu: dict | None = None                    # {"hardware": "t1-g6", "instance_type": "g6.xlarge", "precisions": [...]}
    cpus: list[dict] = []                      # [{"hardware": "t2-m8i-4xl", "instance_type": "m8i.4xlarge", "precisions": [...]}]
    parallel_targets: bool = True
    request_mode: str = "concurrent"           # single | concurrent | batch
    concurrency: list[int] = [1, 4, 16]
    batch_sizes: list[int] = [8]
    sla: dict = {}
    purchase: str = "spot"
    spot_fallback: bool = True
    name: str = ""
    allow_unfair: bool = False


def _run_dict(w: Wizard) -> dict:
    depth = C.DEPTHS.get(w.depth)
    requests = w.requests if w.depth == "custom" else (depth[2] if depth else 50)
    passes = w.passes if w.depth == "custom" else (depth[3] if depth else 1)
    conc = [1] if w.request_mode == "single" else sorted(set(int(c) for c in w.concurrency)) or [1]
    sides = ([w.gpu] if w.gpu else []) + list(w.cpus)
    cells, itypes, alts = [], {}, {}
    for side in sides:
        for p in side.get("precisions") or [w.precision]:
            for mid in w.models:
                cells.append({"target": side["hardware"], "precision": p, "model": mid})
        if side.get("instance_type"):
            itypes[side["hardware"]] = side["instance_type"]
        if side.get("alternates") is not None:
            alts[side["hardware"]] = list(side["alternates"])
    run: dict = {
        "name": w.name or f"{w.use_case}-{time.strftime('%m%d-%H%M')}", "suite": w.use_case,
        "mode": "full" if (requests >= 200 and passes >= 3) else "smoke", "account": ACCOUNT, "region": REGION,
        "az_id": "use1-az1", "models": w.models, "cells": cells,
        "prompt_set": w.prompt_set if w.use_case == "standard" else "dspm",
        "sampling": {"temperature": 0.0, "max_new_tokens": int(w.answer_tokens) if w.use_case == "standard" else 4,
                     "ignore_eos": w.use_case == "standard", "seed": 0},
        "concurrency": conc, "per_level": {"requests": int(requests)}, "repeats": int(passes), "warmup_requests": 20,
        "quality": {"enabled": w.use_case == "standard"},
        "sla": {k: v for k, v in (w.sla or {}).items() if v} | {"max_error_pct": float((w.sla or {}).get("max_error_pct") or 1.0)},
        "purchase": {"type": w.purchase, "spot_fallback_on_demand": bool(w.spot_fallback)},
        "execution": {"parallel_targets": bool(w.parallel_targets)},
    }
    if w.request_mode == "batch":
        run["batch_sizes"] = sorted(set(int(b) for b in w.batch_sizes)) or [8]
    elif w.request_mode in ("single", "concurrent"):
        run["batch_sizes"] = [1]
    if itypes:
        run["instance_types"] = itypes
    if alts:
        run["alternates"] = alts
    if w.use_case == "dspm":
        run["dspm"] = {"workloads": w.workloads, "sla_preset": w.preset, "test_items": int(w.test_items), "prefix_caching": True}
        run["sla"] = {"p99_e2e_ms": C.PRESETS[w.preset][2], "max_error_pct": 1.0}
        run["request_timeout_s"] = 120
    return run


def _dedupe_cells(run: dict) -> dict:
    seen, out = set(), []
    for c in run["cells"]:
        k = (c["target"], c["precision"], c["model"])
        if k not in seen:
            seen.add(k)
            out.append(c)
    return {**run, "cells": out}


@app.post("/api/plan")
def post_plan(w: Wizard):
    """Validate the wizard state and return the plan: series, estimate, fairness, warnings, YAML."""
    import yaml

    run = _dedupe_cells(_run_dict(w))
    models, targets = load_models(ROOT / "config/models.yaml"), load_targets(ROOT / "config/targets.yaml")
    tmap = targets.resolved()
    problems: list[str] = []
    if not w.models:
        problems.append("Pick at least one model.")
    if not w.gpu and not w.cpus:
        problems.append("Pick at least one hardware option.")
    if w.use_case == "dspm" and not w.workloads:
        problems.append("Pick at least one detection scenario.")
    try:
        rc = RunConfig.model_validate(run)
        cells = resolve_cells(rc, models, targets)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "problems": problems + [str(e)], "run": run}
    sizes = local_pool_sizes()
    rtypes = run_instance_types(rc, cells, tmap)
    series, missing = [], []
    for c in cells:
        try:
            n = len(levels_for_cell(rc, c, models.get(c.model), run_id="ui", bucket="ui", pool_sizes=sizes))
        except ValueError as e:
            n, missing = 0, missing + [f"{c.cell_id}: {e}"]
        if n == 0 and not c.quality_only and not missing:
            problems.append(f"{C.model_name(c.model)} has no measurements for this choice (for example a 512-token model "
                            f"with long prompts); drop it or change the input length.")
        series.append({"model": C.model_name(c.model), "model_id": c.model, "hardware": c.target,
                       "instance_type": rtypes[c.target], "accelerator": tmap[c.target].accelerator,
                       "precision": C.PRECISIONS.get(c.precision, (c.precision,))[0], "engine": C.ENGINES[c.engine][0],
                       "measurements": n})
    acc = {t: tmap[t].accelerator for t in rtypes}
    pairs = pairing.judge_run(rtypes, acc)
    unfair = [p for p in pairs if p["fair"] is False]
    cat = pairing.load_catalog()["instances"]
    prices = {t: cat[t]["usd_per_hour"] for t in set(rtypes.values()) if t in cat}
    est = None
    try:
        est = estimate(rc, cells, models, tmap, prices, sizes, instance_types=rtypes)
    except Exception as e:  # noqa: BLE001
        problems.append(f"No estimate: {e}")
    warnings = []
    if rc.is_smoke:
        warnings.append("Quick check: every number will be labelled SMOKE. Use it to see the shape of the results, not to decide.")
    if not any(a == "gpu" for a in acc.values()) or not any(a == "cpu" for a in acc.values()):
        warnings.append("Only one side (GPU or CPU) is in this run; you will get measurements but no GPU-versus-CPU verdict.")
    for m in w.models:
        ms = models.get(m)
        if ms.license == "qwen-research":
            warnings.append(f"{C.model_name(m)} has a research-only license; its results are for evaluation, not a commercial decision.")
    if unfair and not w.allow_unfair:
        problems.append("Unfair pairing: " + " ".join(p["reason"] for p in unfair) + " Pick comparable sizes or allow it explicitly.")
    return {"ok": not problems and not missing, "problems": problems + missing, "warnings": warnings, "series": series,
            "pairing": pairs, "estimate": est, "grade": "quick" if rc.is_smoke else "full",
            "yaml": yaml.safe_dump({"run": run}, sort_keys=False), "run": run,
            "request_mode": compare.request_mode(run)}


class ProposeIn(BaseModel):
    models: list[str] = []
    precision: str = "bf16"
    gpu_type: str | None = None
    cpu_type: str | None = None


@app.post("/api/propose")
def post_propose(p: ProposeIn):
    cat = get_catalog()
    mods = [m for m in cat["models"] if m["id"] in p.models] or cat["models"][:1]
    return propose.propose(mods, p.precision, p.gpu_type, p.cpu_type, _capacity_notes())


def _capacity_notes() -> dict[str, int]:
    """Instance types refused for capacity in recent runs (from runs/*/capacity/*.json), cached 10 min."""
    now = time.time()
    if _CACHE.get("cap_at", 0) > now - 600:
        return _CACHE["cap"]
    notes: dict[str, int] = {}
    try:
        import boto3

        s3 = boto3.client("s3", region_name=REGION)
        bucket = _outputs()["BucketName"]
        keys = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="runs/"):
            keys += [o["Key"] for o in page.get("Contents", []) if "/capacity/" in o["Key"]]
        for k in sorted(keys)[-40:]:
            for e in json.loads(s3.get_object(Bucket=bucket, Key=k)["Body"].read()):
                t = e.get("instance_type")
                if t and t != "template":
                    notes[t] = notes.get(t, 0) + 1
    except Exception as e:  # noqa: BLE001 - offline is fine
        notes["_error"] = 0 if not notes else notes.get("_error", 0)
        _CACHE["cap_error"] = str(e)[:200]
    notes.pop("_error", None)
    _CACHE["cap"], _CACHE["cap_at"] = notes, now
    return notes


# ----------------------------------------------------------------------------- AWS

def _outputs() -> dict:
    if "outputs" in _CACHE and _CACHE.get("outputs_at", 0) > time.time() - 300:
        return _CACHE["outputs"]
    import boto3

    stk = boto3.client("cloudformation", region_name=REGION).describe_stacks(StackName="AmxBench-Core")["Stacks"][0]
    out = {o["OutputKey"]: o["OutputValue"] for o in stk.get("Outputs", [])}
    _CACHE["outputs"], _CACHE["outputs_at"] = out, time.time()
    return out


@app.get("/api/aws/status")
def aws_status():
    try:
        import boto3

        ident = boto3.client("sts", region_name=REGION).get_caller_identity()
        out = _outputs()
        return {"ok": True, "account": ident["Account"], "identity": ident["Arn"].split("/")[-1], "bucket": out.get("BucketName")}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:300]}


@app.post("/api/runs")
def start_run(w: Wizard):
    import boto3

    plan = post_plan(w)
    if not plan["ok"]:
        raise HTTPException(400, "; ".join(plan["problems"]))
    out = _outputs()
    run_id = f"{plan['run']['name']}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    boto3.client("s3", region_name=REGION).put_object(Bucket=out["BucketName"], Key=f"runs/{run_id}/run.yaml",
                                                      Body=plan["yaml"].encode())
    boto3.client("stepfunctions", region_name=REGION).start_execution(
        stateMachineArn=out["RunStateMachineArn"], name=run_id, input=json.dumps({"run_id": run_id}))
    return {"run_id": run_id}


def _table():
    import boto3

    return boto3.resource("dynamodb", region_name=REGION).Table("amx-gpu-bench")


RUN_STATUS = {"running": "Running", "complete": "Finished", "refused": "Refused by the pre-checks", "planned": "Planned"}
CELL_STATUS = {"planned": "Planned", "configuring": "Starting the model server", "ready": "Serving",
               "bringup_failed": "Failed to start", "interrupted": "Instance lost, replacing it", "failed": "Failed"}
LEVEL_STATUS = {"running": "Running", "done": "Done", "failed": "Failed", "incomplete": "Stopped early",
                "pending": "Waiting to be re-run"}


@app.get("/api/runs")
def list_runs():
    from lambdas.common import plain

    tbl = _table()
    items, kw = [], {"FilterExpression": "sk = :m", "ExpressionAttributeValues": {":m": "META"}}
    while True:
        r = tbl.scan(**kw)
        items += [plain(x) for x in r["Items"]]
        if "LastEvaluatedKey" not in r:
            break
        kw["ExclusiveStartKey"] = r["LastEvaluatedKey"]
    items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return [{"run_id": i["pk"][4:], "status": i.get("status"), "status_label": RUN_STATUS.get(i.get("status"), i.get("status")),
             "updated_at": i.get("updated_at"), "grade": "quick" if i.get("smoke") else "full", "purchase": i.get("purchase"),
             "n_cells": i.get("n_cells"), "errors": i.get("errors")} for i in items]


def _parse_level(level_key: str, cell_key: str) -> dict:
    rest = level_key[len(cell_key) + 2:].split("__")
    name = rest[0]
    conc = next((int(x[1:]) for x in rest[1:] if x.startswith("c") and x[1:].isdigit()), None)
    batch = next((int(x[1:]) for x in rest[1:] if x.startswith("b") and x[1:].isdigit()), 1)
    rep = next((int(x[1:]) for x in rest[1:] if x.startswith("r") and x[1:].isdigit()), None)
    base = name[:-9] if name.endswith("-accuracy") else name
    wl = base.split("-")[0]
    if wl in C.WORKLOADS:
        what = C.WORKLOADS[wl][0] + (f" ({base.split('-', 1)[1]} documents)" if "-" in base else "")
    elif name in C.PROMPT_SETS:
        what = C.PROMPT_SETS[name][0]
    else:
        what = name
    if name.endswith("-accuracy"):
        what = "Accuracy pass: " + what
    return {"what": what, "conc": conc, "batch": batch, "pass": None if rep is None else rep + 1}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    from boto3.dynamodb.conditions import Key

    from lambdas.common import plain

    items = [plain(x) for x in _table().query(KeyConditionExpression=Key("pk").eq(f"RUN#{run_id}"))["Items"]]
    if not items:
        raise HTTPException(404, "unknown run")
    meta = next((i for i in items if i["sk"] == "META"), {})
    hw = {h["id"]: h for h in get_catalog()["hardware"]}
    cells, levels = [], []
    for i in items:
        if i["sk"].startswith("CELL#"):
            mid, tid, prec = (i["sk"][5:].split("|") + [None, None, None])[:3]
            cells.append({"model": C.model_name(mid), "model_id": mid, "hardware": tid, "hardware_label": hw.get(tid, {}).get("label", tid),
                          "precision": C.PRECISIONS.get(prec, (prec,))[0], "engine": C.ENGINES.get(i.get("engine"), (i.get("engine"),))[0],
                          "status": i.get("status"), "status_label": CELL_STATUS.get(i.get("status"), i.get("status")),
                          "measurements": i.get("n_levels"), "instance_id": i.get("instance_id"),
                          "error": (i.get("error") or i.get("reason") or i.get("stderr") or "")[:300]})
        elif i["sk"].startswith("LEVEL#"):
            key = i["sk"][6:]
            ck = i.get("cell_key") or "__".join(key.split("__")[:3])
            mid, tid = (ck.split("__") + [None, None])[:2]
            levels.append({"model": C.model_name(mid), "hardware": tid, "hardware_label": hw.get(tid, {}).get("label", tid),
                           **_parse_level(key, ck), "status": i.get("status"),
                           "status_label": LEVEL_STATUS.get(i.get("status"), i.get("status")),
                           "completed": i.get("completed"), "errors": i.get("errors"), "detail": (i.get("error") or i.get("reason") or "")[:200]})
    done = sum(1 for lv in levels if lv["status"] == "done")
    return {"run_id": run_id, "status": meta.get("status"), "status_label": RUN_STATUS.get(meta.get("status"), meta.get("status")),
            "grade": "quick" if meta.get("smoke") else "full", "purchase": meta.get("purchase"), "updated_at": meta.get("updated_at"),
            "refusal": meta.get("errors") if meta.get("status") == "refused" else None,
            "progress": {"done": done, "total": len(levels)}, "cells": cells,
            "levels": sorted(levels, key=lambda x: (x["hardware"], x["model"], x["what"], x["conc"] or 0, x["pass"] or 0)),
            "has_report": meta.get("status") == "complete"}


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str):
    import boto3

    arn = _outputs()["RunStateMachineArn"].replace(":stateMachine:", ":execution:")
    boto3.client("stepfunctions", region_name=REGION).stop_execution(executionArn=f"{arn}:{run_id}", cause="stopped from the UI")
    return {"stopped": run_id}


def _summary(run_id: str) -> dict:
    key = f"summary:{run_id}"
    if key in _CACHE:
        return _CACHE[key]
    import boto3

    try:
        body = boto3.client("s3", region_name=REGION).get_object(Bucket=_outputs()["BucketName"],
                                                                 Key=f"runs/{run_id}/report/summary.json")["Body"].read()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(404, f"no report for {run_id}: {str(e)[:120]}") from e
    s = json.loads(body)
    _CACHE[key] = s
    return s


@app.get("/api/runs/{run_id}/summary")
def get_summary(run_id: str):
    s = _summary(run_id)
    return {"run_id": run_id, "shape": compare.shape(s), "default_sla": compare.default_sla([s]), "run": s.get("run"),
            "smoke": s.get("smoke"), "warnings": s.get("warnings"), "dspm_hashes": s.get("dspm_hashes"),
            "models": sorted({r["model"] for r in s["rows"]}), "hardware": sorted({r["target"] for r in s["rows"]}),
            "instance_types": sorted({r["instance_type"] for r in s["rows"]}),
            "provenance": s.get("provenance")}


class CompareIn(BaseModel):
    run_ids: list[str]
    sla: dict | None = None
    limit: int = 4
    models: list[str] | None = None            # restrict to these model ids
    series: list[str] | None = None            # restrict to these series names (hardware / precision)


@app.post("/api/compare")
def post_compare(c: CompareIn):
    if not c.run_ids:
        raise HTTPException(400, "pick at least one run")
    sums = [_summary(r) for r in c.run_ids]
    comp = compare.comparability(sums)
    sla = c.sla or compare.default_sla(sums)
    if c.models:
        sums = [{**s, "rows": [r for r in s["rows"] if r["model"] in c.models]} for s in sums]
    analysis = compare.analyze(sums, {k: v for k, v in sla.items() if k != "preset"}, c.limit) if comp["ok"] else None
    if analysis and c.series:
        for g in analysis["groups"]:
            g["points"] = [p for p in g["points"] if p["series"] in c.series]
    return {"comparability": comp, "sla": sla, "analysis": analysis,
            "runs": [{"run_id": s["run_id"], "shape": compare.shape(s), "smoke": s.get("smoke")} for s in sums]}


@app.get("/api/runs/{run_id}/report.md")
def get_report_md(run_id: str):
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(report.render(_summary(run_id), with_charts=False)["summary.md"].decode())


# ----------------------------------------------------------------------------- static app

DIST = ROOT / "webui/dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if f.is_file() else DIST / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8601, log_level="info")
