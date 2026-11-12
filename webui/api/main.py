"""Local API for the web UI. Binds 127.0.0.1 only; every AWS call uses the operator's own
credentials. Serves the built React app from webui/dist when present.

    .venv/bin/python -m webui.api.main            # http://127.0.0.1:8601
"""

from __future__ import annotations

import json
import re
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
    spot_wait_minutes: int = 10                # try Spot this long before falling back to On-Demand
    launch_attempts: int = 30                  # launch attempts (about one a minute) before giving up on a hardware option
    name: str = ""
    allow_unfair: bool = False


def _available(mid: str, precision: str, hardware: str, cat: dict) -> bool:
    """Does this number format of this model run on this hardware option's serving software?
    (Qwen in BF16 runs on vLLM/PyTorch, not on the OpenVINO option; there it is INT8/INT4.)"""
    m = next((x for x in cat["models"] if x["id"] == mid), None)
    h = next((x for x in cat["hardware"] if x["id"] == hardware), None)
    if not m or not h:
        return True                     # let the config validator report unknown ids
    spec = m["precisions"].get(precision)
    return bool(spec) and not spec.get("quality_only") and any(e in h["engines"] for e in spec["engines"])


def _run_dict(w: Wizard, notes: list[str] | None = None) -> dict:
    """The run config for the wizard state. A number format the hardware option cannot serve for
    the model is dropped with a note (never a hard failure): ticking the OpenVINO option with a
    chat model whose BF16 runs only on vLLM must not kill the plan."""
    cat = get_catalog()
    depth = C.DEPTHS.get(w.depth)
    requests = w.requests if w.depth == "custom" else (depth[2] if depth else 50)
    passes = w.passes if w.depth == "custom" else (depth[3] if depth else 1)
    conc = [1] if w.request_mode == "single" else sorted(set(int(c) for c in w.concurrency)) or [1]
    sides = ([w.gpu] if w.gpu else []) + list(w.cpus)
    hw = {h["id"]: h for h in cat["hardware"]}
    cells, itypes, alts = [], {}, {}
    for side in sides:
        before, dropped = len(cells), []
        for p in side.get("precisions") or [w.precision]:
            for mid in w.models:
                if _available(mid, p, side["hardware"], cat):
                    cells.append({"target": side["hardware"], "precision": p, "model": mid})
                else:
                    dropped.append(f"{hw.get(side['hardware'], {}).get('label', side['hardware'])}: {C.model_name(mid)} is not available "
                                   f"in {C.PRECISIONS.get(p, (p,))[0]} on this serving software; that measurement is left out.")
        if len(cells) > before and notes is not None:
            notes.extend(dropped)
        if len(cells) == before:
            # The option runs nothing: leave it out of instance_types/alternates (the config
            # validator rejects a target without cells) and say so.
            if notes is not None and w.models:
                m = next((x for x in cat["models"] if x["id"] == w.models[0]), None)
                ok = [C.PRECISIONS.get(p, (p,))[0] for p, s in (m or {}).get("precisions", {}).items()
                      if not s.get("quality_only") and any(e in hw.get(side["hardware"], {}).get("engines", []) for e in s["engines"])]
                notes.append(f"{hw.get(side['hardware'], {}).get('label', side['hardware'])} runs nothing in this plan: "
                             + (f"pick one of its number formats for {C.model_name(w.models[0])}: {', '.join(ok)}." if ok
                                else f"it cannot serve {C.model_name(w.models[0])} at all; untick it."))
            continue
        if side.get("instance_type"):
            itypes[side["hardware"]] = side["instance_type"]
        if side.get("alternates") is not None:
            alts[side["hardware"]] = list(side["alternates"])
    run: dict = {
        "name": run_slug(w.name) or f"{w.use_case}-{time.strftime('%m%d-%H%M')}", "suite": w.use_case,
        "mode": "full" if (requests >= 200 and passes >= 3) else "smoke", "account": ACCOUNT, "region": REGION,
        "az_id": "use1-az1", "models": w.models, "cells": cells,
        "prompt_set": w.prompt_set if w.use_case == "standard" else "dspm",
        "sampling": {"temperature": 0.0, "max_new_tokens": int(w.answer_tokens) if w.use_case == "standard" else 4,
                     "ignore_eos": w.use_case == "standard", "seed": 0},
        "concurrency": conc, "per_level": {"requests": int(requests)}, "repeats": int(passes), "warmup_requests": 20,
        "quality": {"enabled": w.use_case == "standard"},
        "sla": {k: v for k, v in (w.sla or {}).items() if v} | {"max_error_pct": float((w.sla or {}).get("max_error_pct") or 1.0)},
        "purchase": {"type": w.purchase, "spot_fallback_on_demand": bool(w.spot_fallback),
                     "spot_capacity_window_s": max(1, int(w.spot_wait_minutes)) * 60},
        "execution": {"parallel_targets": bool(w.parallel_targets)},
    }
    dep = deployed_backend()
    if dep["supports"].get("max_launch_attempts", True):
        run["purchase"]["max_launch_attempts"] = max(1, int(w.launch_attempts))
    elif notes is not None:
        notes.append(f"The deployed backend (AmxBench-Core, built from commit {str(dep['commit'])[:7]}) predates the "
                     f"'give up after N failed launch attempts' setting, so this run keeps the fixed 30-minute capacity "
                     f"window{' instead of your ' + str(w.launch_attempts) + ' attempts' if w.launch_attempts != 30 else ''}. "
                     "Deploy the Core stack (cd infra && cdk deploy AmxBench-Core) to use it.")
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

    notes: list[str] = []
    run = _dedupe_cells(_run_dict(w, notes))
    models, targets = load_models(ROOT / "config/models.yaml"), load_targets(ROOT / "config/targets.yaml")
    tmap = targets.resolved()
    problems: list[str] = []
    if not w.models:
        problems.append("Pick at least one model.")
    if not w.gpu and not w.cpus:
        problems.append("Pick at least one hardware option.")
    if w.use_case == "dspm" and not w.workloads:
        problems.append("Pick at least one detection scenario.")
    if not run["cells"] and w.models and (w.gpu or w.cpus):
        problems.append("No hardware option can serve the chosen model in the chosen number formats.")
    failed = {"ok": False, "problems": problems, "warnings": notes, "series": [], "pairing": [], "estimate": None,
              "grade": "quick" if run["mode"] == "smoke" else "full", "request_mode": w.request_mode, "yaml": "", "run": run}
    if problems:
        return failed
    try:
        rc = RunConfig.model_validate(run)
        cells = resolve_cells(rc, models, targets)
    except Exception as e:  # noqa: BLE001
        return {**failed, "problems": problems + [str(e)]}
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
    warnings = list(notes)
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


class SuggestIn(BaseModel):
    models: list[str] = []
    precision: str = "bf16"
    gpu_type: str


class InitialIn(BaseModel):
    model: str
    precision: str = "bf16"


@app.post("/api/suggest/initial")
def post_suggest_initial(p: InitialIn):
    """What the hardware step pre-selects for the chosen model: GPU best fit, then the CPU best
    match against it, both with reasons. A classifier and a chat model get different pairs."""
    cat = get_catalog()
    mods = [m for m in cat["models"] if m["id"] == p.model]
    if not mods:
        raise HTTPException(400, f"unknown model {p.model}")
    return propose.initial_selection(mods, p.precision, _capacity_notes())


@app.post("/api/suggest")
def post_suggest(p: SuggestIn):
    """Auto-suggest: CPU sizes for the chosen GPU and models — one best match, up to two
    alternatives, the rest folded away."""
    cat = get_catalog()
    mods = [m for m in cat["models"] if m["id"] in p.models]
    if not mods:
        raise HTTPException(400, "pick at least one model first")
    if p.gpu_type not in cat["instances"]:
        raise HTTPException(400, f"{p.gpu_type} is not in the instance catalog")
    return propose.suggest_cpu(mods, p.precision, p.gpu_type, _capacity_notes())


def _ai_context(w: Wizard) -> dict:
    """What the AI reviewer is told: the wizard state plus the tool's own analysis, in plain facts."""
    from webui.api import ai_propose

    cat = get_catalog()
    if not w.gpu or not w.cpus:
        raise HTTPException(400, "choose the GPU option with its size and at least one CPU option first")
    mods = [m for m in cat["models"] if m["id"] in w.models]
    if not mods:
        raise HTTPException(400, "pick at least one model first")
    specs = {m.id: m for m in load_models(ROOT / "config/models.yaml").models}
    hw = {h["id"]: h for h in cat["hardware"]}
    gpu_spec, cpu_specs = cat["instances"].get(w.gpu["instance_type"]), {c["hardware"]: cat["instances"].get(c["instance_type"]) for c in w.cpus}
    if not gpu_spec or any(s is None for s in cpu_specs.values()):
        raise HTTPException(400, "an instance size is not in the catalog")
    depth = C.DEPTHS.get(w.depth)
    requests = w.requests if w.depth == "custom" else (depth[2] if depth else 50)
    passes = w.passes if w.depth == "custom" else (depth[3] if depth else 1)
    test = {"use case": C.USE_CASES[w.use_case]["name"] if w.use_case in C.USE_CASES else w.use_case,
            "request shape": C.REQUEST_MODES.get(w.request_mode, (w.request_mode,))[0],
            "requests at a time (each measured separately)": ", ".join(map(str, sorted(set(int(c) for c in w.concurrency)))) if w.request_mode != "single" else "1",
            "requests per measurement": requests, "repeated": f"{passes} time(s)", "number format": w.precision}
    if w.request_mode == "batch":
        test["inputs per request"] = ", ".join(map(str, w.batch_sizes))
    if w.use_case == "standard":
        test["prompts"] = f"{C.PROMPT_SETS.get(w.prompt_set, (w.prompt_set,))[0]}, answers of {w.answer_tokens} tokens"
        test["targets"] = ", ".join(f"{k} <= {v}" for k, v in (w.sla or {}).items() if v) or "none"
    else:
        test["scenarios"] = "; ".join(C.WORKLOADS.get(x, (x,))[0] for x in w.workloads)
        test["decision deadline"] = C.PRESETS.get(w.preset, (w.preset, "", None))[0]
    models_ctx = []
    for m in mods:
        s = specs.get(m["id"])
        models_ctx.append({"name": m["name"], "hf_repo": s.hf_repo if s else "?", "kind": m["kind"], "type": m["type"],
                           "params_b": m["params_b"], "license": m["license"], "max_input_tokens": s.max_input_tokens if s else "?",
                           "precisions": {p: v["engines"] for p, v in m["precisions"].items()}})
    sug = propose.suggest_cpu(mods, w.precision, w.gpu["instance_type"], _capacity_notes())
    sug_small = {"basis": sug["basis"], "best": {"instance_type": sug["best"]["instance_type"]} if sug["best"] else None,
                 "alternatives": [{"instance_type": a["instance_type"], "role_label": a["role_label"]} for a in sug["alternatives"]]}
    return {"region": REGION, "az": "use1-az1", "price_snapshot": cat.get("price_snapshot"), "test": test, "models": models_ctx,
            "gpu": {"instance_type": w.gpu["instance_type"], "spec": gpu_spec, "software": hw.get(w.gpu["hardware"], {}).get("software", [])},
            "cpus": [{"instance_type": c["instance_type"], "spec": cpu_specs[c["hardware"]],
                      "software": hw.get(c["hardware"], {}).get("software", [])} for c in w.cpus],
            "pairing": [pairing.judge(c["instance_type"], w.gpu["instance_type"]) for c in w.cpus],
            "suggest": sug_small, "capacity_notes": {k: v for k, v in _capacity_notes().items() if k in {w.gpu["instance_type"], *[c["instance_type"] for c in w.cpus]}},
            "allowed": {"gpu": pairing.types_for("gpu"), "cpu": pairing.types_for("cpu")},
            "_module": ai_propose}


@app.post("/api/propose/ai")
def post_propose_ai(w: Wizard):
    """Start the AI review as a background job; poll /api/propose/ai/{id}."""
    ctx = _ai_context(w)
    mod = ctx.pop("_module")
    return {"job_id": mod.start(ctx), "prompt_preview": mod.build_prompt(ctx)}


@app.get("/api/propose/ai/{job_id}")
def get_propose_ai(job_id: str):
    from webui.api import ai_propose

    s = ai_propose.status(job_id)
    if s is None:
        raise HTTPException(404, "unknown or expired job")
    return s


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
        dep = deployed_backend()
        return {"ok": True, "account": ident["Account"], "identity": ident["Arn"].split("/")[-1], "bucket": out.get("BucketName"),
                "backend": {"commit": dep["commit"], "local_commit": dep["local_commit"], "behind": dep["behind"], "missing": dep["missing"]}}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:300]}


# ----------------------------------------------------------------------------- deployed backend

# Settings the Lambdas only understand once the Core stack built from (or after) this commit is
# deployed. RunConfig forbids unknown fields, so sending one to an older preflight fails the run
# before anything starts (seen 2026-11-12: purchase.max_launch_attempts -> "Extra inputs are not
# permitted", no META row, 'unknown run' in the UI). The plan leaves such a field out and warns.
FEATURE_COMMITS = {"max_launch_attempts": "2be19638163d809fa83e21a73271a184ff068b4c"}
PREFLIGHT_FUNCTION = "amx-gpu-bench-preflight"


def _git_contains(deployed: str, feature: str) -> bool | None:
    """Is the feature commit an ancestor of the deployed commit? None when git cannot tell."""
    import subprocess

    try:
        r = subprocess.run(["git", "merge-base", "--is-ancestor", feature, deployed], cwd=ROOT, capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001 - no git, no repo
        return None
    return True if r.returncode == 0 else (False if r.returncode == 1 else None)


def deployed_backend() -> dict:
    """The commit the deployed Lambdas were built from and which newer settings they accept.
    Cached 5 minutes; unknown means 'assume supported' (the failure is then reported on the run)."""
    if _CACHE.get("deployed_at", 0) > time.time() - 300:
        return _CACHE["deployed"]
    commit, error = None, None
    try:
        import boto3

        env = boto3.client("lambda", region_name=REGION).get_function_configuration(FunctionName=PREFLIGHT_FUNCTION)
        commit = (env.get("Environment") or {}).get("Variables", {}).get("HARNESS_COMMIT")
    except Exception as e:  # noqa: BLE001 - offline or no permission
        error = str(e)[:160]
    supports = {}
    for feature, fc in FEATURE_COMMITS.items():
        has = _git_contains(commit, fc) if commit and commit != "unknown" else None
        supports[feature] = True if has is None else has
    head = None
    try:
        import subprocess

        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        head = None
    behind = bool(commit and head and commit != head and _git_contains(head, commit) is True and _git_contains(commit, head) is False)
    out = {"commit": commit, "local_commit": head, "supports": supports, "behind": behind, "error": error,
           "missing": sorted(k for k, v in supports.items() if not v)}
    _CACHE["deployed"], _CACHE["deployed_at"] = out, time.time()
    return out


def run_slug(name: str) -> str:
    """A run name as it appears in the run id, S3 prefix and Step Functions execution name:
    lowercase letters, digits and dashes, at most 40 characters."""
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return s[:40].rstrip("-")


@app.get("/api/runs/slug")
def preview_slug(name: str = "", use_case: str = "standard"):
    slug = run_slug(name) or f"{use_case}-{time.strftime('%m%d-%H%M')}"
    return {"slug": slug, "run_id": f"{slug}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"}


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
    # The row first, so the run is on the list the moment it starts; preflight overwrites the status.
    _table().put_item(Item={"pk": f"RUN#{run_id}", "sk": "META", "status": "planned", "name": w.name.strip() or plan["run"]["name"],
                            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "purchase": plan["run"]["purchase"]["type"], "smoke": plan["grade"] == "quick",
                            "n_cells": len(plan["series"]), "model": (w.models or [None])[0]})
    boto3.client("stepfunctions", region_name=REGION).start_execution(
        stateMachineArn=out["RunStateMachineArn"], name=run_id, input=json.dumps({"run_id": run_id}))
    return {"run_id": run_id}


def _table():
    import boto3

    return boto3.resource("dynamodb", region_name=REGION).Table("amx-gpu-bench")


RUN_STATUS = {"running": "Running", "complete": "Finished", "refused": "Refused by the pre-checks", "planned": "Starting",
              "stopped": "Stopped by the operator", "failed": "Failed", "timed_out": "Timed out"}
SFN_STATUS = {"ABORTED": "stopped", "FAILED": "failed", "TIMED_OUT": "timed_out"}
CELL_STATUS = {"planned": "Waiting", "configuring": "Loading the model", "ready": "Measuring",
               "bringup_failed": "Failed to start", "interrupted": "Instance lost, replacing it", "failed": "Failed",
               "done": "Done", "not_run": "Not run — no capacity"}
INSTANCE_STATE = {"pending": "Starting", "running": "Running", "stopping": "Stopping", "stopped": "Stopped",
                  "shutting-down": "Terminating", "terminated": "Terminated"}


def _s3_json(key: str):
    import boto3

    try:
        return json.loads(boto3.client("s3", region_name=REGION).get_object(Bucket=_outputs()["BucketName"], Key=key)["Body"].read())
    except Exception:  # noqa: BLE001 - absent until the first event
        return None


def _instance_states(ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    import boto3

    out = {}
    try:
        r = boto3.client("ec2", region_name=REGION).describe_instances(InstanceIds=ids)
        for res in r["Reservations"]:
            for i in res["Instances"]:
                out[i["InstanceId"]] = {"state": i["State"]["Name"], "type": i["InstanceType"],
                                        "lifecycle": "spot" if i.get("InstanceLifecycle") == "spot" else "on-demand",
                                        "launched_at": i["LaunchTime"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                                        "reason": (i.get("StateReason") or {}).get("Message")}
    except Exception as e:  # noqa: BLE001 - an id may have aged out of describe-instances
        out["_error"] = {"state": None, "error": str(e)[:120]}
    out.pop("_error", None)
    return out


def _purchase_cfg(run_id: str) -> dict:
    """The run's purchase settings (Spot window, attempt budget) from runs/<id>/resolved.json."""
    key = f"purchase:{run_id}"
    if key not in _CACHE:
        res = _s3_json(f"runs/{run_id}/resolved.json") or {}
        _CACHE[key] = dict(res.get("purchase") or {})
    return _CACHE[key]


def _tried(caps: list[dict]) -> tuple[str, str]:
    """'g6.2xlarge, g5.xlarge' and 'Spot and On-Demand' from a capacity log."""
    types = sorted({str(c.get("instance_type")) for c in caps if c.get("instance_type")})
    mk = {"spot": "Spot", "on-demand": "On-Demand"}
    markets = sorted({mk.get(str(c.get("market")), str(c.get("market"))) for c in caps if c.get("market")})
    return ", ".join(types) or "the requested size", " and ".join(markets) or "the requested market"


def _minutes_between(caps: list[dict]) -> int | None:
    times = sorted(str(c.get("time")) for c in caps if c.get("time"))
    if len(times) < 2:
        return None
    try:
        t0, t1 = (time.mktime(time.strptime(t[:19], "%Y-%m-%dT%H:%M:%S")) for t in (times[0], times[-1]))
    except ValueError:
        return None
    return max(1, round((t1 - t0) / 60))


def not_run_reason(caps: list[dict], purchase: dict, gave_up: dict | None) -> str:
    """Why a hardware option never got an instance, for a reader who is not an engineer."""
    if gave_up and gave_up.get("reason"):
        return str(gave_up["reason"])
    sizes, markets = _tried(caps)
    span = _minutes_between(caps)
    budget = purchase.get("max_launch_attempts")
    if caps:
        return (f"AWS had no capacity: it refused {len(caps)} launch request(s)"
                f"{f' over {span} minutes' if span else ''} for {sizes} ({markets}). The run gave up"
                f"{f' after its limit of {budget} attempts' if budget else ''} and continued with the other hardware options.")
    return "No instance was ever launched for this hardware option, and no capacity refusal was logged."


def target_status(run_id: str, target_ids: list[str], run_finished: bool, hw: dict) -> list[dict]:
    """Per hardware option: the instance that serves it, its live state, market, whether On-Demand
    came from a Spot fallback, capacity refusals and interruptions, in one sentence. A hardware
    option that never got an instance is reported as *did not run*, with the reason — never as a
    bare 'Not run' or as still waiting once the run is over."""
    out = []
    purchase = _purchase_cfg(run_id)
    budget = purchase.get("max_launch_attempts")
    spot_minutes = int(purchase.get("spot_capacity_window_s") or 600) // 60
    for tid in target_ids:
        events = _s3_json(f"runs/{run_id}/targets/{tid}.json") or []
        caps = _s3_json(f"runs/{run_id}/capacity/{tid}.json") or []
        launches = [e for e in events if e.get("event") in ("launched", "started", "reused")]
        last = launches[-1] if launches else None
        live = _instance_states([last["instance_id"]]).get(last["instance_id"]) if last else None
        interruptions = [e for e in events if e.get("event") == "interrupted"]
        gave_up = next((e for e in reversed(events) if e.get("event") == "gave_up"), None)
        requested = next((e.get("requested") for e in reversed(events) if e.get("requested")), None) or purchase.get("type")
        lifecycle = (live or {}).get("lifecycle") or (last or {}).get("lifecycle")
        fallback = bool(requested == "spot" and lifecycle == "on-demand")
        state = (live or {}).get("state")
        did_not_run, reason = False, None
        market = None
        if lifecycle:
            market = "Spot" if lifecycle == "spot" else (f"On-Demand (fell back from Spot: no Spot capacity for {spot_minutes} minutes)"
                                                        if fallback else "On-Demand")
        if last is None:
            if gave_up or run_finished:
                did_not_run = True
                reason = not_run_reason(caps, purchase, gave_up)
                phrase = f"Did not run. {reason}"
                state = "not_run"
            elif caps:
                sizes, markets = _tried(caps)
                phrase = (f"Waiting for capacity: AWS refused {len(caps)} launch request(s) so far for {sizes} ({markets})"
                          + (f" — the run gives up after {budget} attempts" if budget else ""))
            else:
                phrase = "Requesting an instance"
        elif live is None:
            # EC2 forgets terminated instances after about an hour: the run's own log is the record.
            ended = "Terminated after the run" if lifecycle == "spot" else "Stopped after the run"
            phrase = f"{ended if run_finished else 'No longer present'} · {market}" if market else ended
            state = "gone"
        else:
            label = INSTANCE_STATE.get(state or "", state or "unknown")
            phrase = f"{label} · {market}" if market else label
            if state == "running" and (live or {}).get("launched_at"):
                phrase += f" · since {live['launched_at'][11:16]} UTC"
            if interruptions:
                phrase += f" · {len(interruptions)} Spot interruption(s), replaced"
            if caps and not interruptions:
                phrase += f" · after {len(caps)} capacity refusal(s)"
        out.append({"hardware": tid, "hardware_label": hw.get(tid, {}).get("label", tid),
                    "instance_id": (last or {}).get("instance_id"), "instance_type": (live or last or {}).get("type") or (last or {}).get("instance_type"),
                    "state": state, "state_label": "Did not run" if did_not_run else INSTANCE_STATE.get(state or "", state),
                    "lifecycle": lifecycle, "requested": requested,
                    "fallback_from_spot": fallback, "capacity_refusals": len(caps), "interruptions": len(interruptions),
                    "attempts_max": budget, "did_not_run": did_not_run, "not_run_reason": reason,
                    "launched_at": (live or {}).get("launched_at") or (last or {}).get("time"), "status": phrase,
                    "events": [{k: e.get(k) for k in ("time", "event", "instance_id", "instance_type", "lifecycle", "state_reason")}
                               for e in events][-8:]})
    return out


def derive_cell_status(cell: dict, siblings: list[dict], target: dict | None, run_status: str) -> str:
    """'planned' is ambiguous: say what the cell is waiting for — or that it never ran, and why."""
    s = cell.get("status")
    if s != "planned":
        return CELL_STATUS.get(s, s)
    if target and target.get("did_not_run"):
        return "Not run — no capacity"
    if run_status in ("complete", "stopped", "failed", "timed_out"):
        return "Not run"
    if target and target.get("state") != "running":
        return "Waiting for the instance" if target.get("instance_id") else "Waiting for an instance (capacity)"
    busy = [c for c in siblings if c is not cell and c.get("status") in ("configuring", "ready")]
    if busy:
        return f"Queued behind {busy[0]['model']}"
    return "About to start"


def run_notices(targets: list[dict], cells: list[dict], run_status: str) -> list[dict]:
    """The things a reader must not miss, in full sentences: a side that did not run and what
    that means for the results."""
    notes = []
    for t in targets:
        if not t.get("did_not_run"):
            continue
        side = "GPU" if t["hardware_label"].startswith("GPU") else "CPU"
        others = [x for x in targets if x is not t and not x.get("did_not_run")]
        other_side = "CPU" if side == "GPU" else "GPU"
        consequence = (f" The {other_side} results below stand alone: there is no {side} measurement to compare them with."
                       if others else " Nothing was measured in this run.")
        notes.append({"kind": "bad", "title": f"The {side} side did not run",
                      "text": f"{t['hardware_label']}: {t['not_run_reason']}{consequence} "
                              "Start the run again later, choose another instance size, or raise the number of attempts "
                              "before giving up."})
    waiting = [t for t in targets if not t.get("did_not_run") and t.get("instance_id") is None and t.get("capacity_refusals")]
    if run_status == "running" and waiting:
        for t in waiting:
            side = "GPU" if t["hardware_label"].startswith("GPU") else "CPU"
            notes.append({"kind": "warn", "title": f"The {side} side is waiting for capacity",
                          "text": f"{t['hardware_label']}: {t['status']}."})
    return notes
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
    _reconcile(items)
    items += _orphan_executions({i["pk"][4:] for i in items})
    items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    hw = {h["id"]: h for h in get_catalog()["hardware"]}

    def label(i: dict) -> str:
        base = RUN_STATUS.get(i.get("status"), i.get("status"))
        nr = i.get("not_run") or []          # written by target_control when a hardware option is given up
        if nr and i.get("status") == "complete":
            sides = sorted({"GPU" if hw.get(x.get("target_id"), {}).get("label", "").startswith("GPU") else "CPU" for x in nr})
            return f"{base} — {' and '.join(sides)} side did not run"
        return base

    return [{"run_id": i["pk"][4:], "name": i.get("name"), "model": C.model_name(i["model"]) if i.get("model") else None,
             "status": i.get("status"), "status_label": label(i),
             "updated_at": i.get("updated_at"), "grade": "quick" if i.get("smoke") else "full", "purchase": i.get("purchase"),
             "n_cells": i.get("n_cells"), "errors": i.get("errors"),
             "not_run": [{"hardware": x.get("target_id"), "hardware_label": hw.get(x.get("target_id"), {}).get("label", x.get("target_id")),
                          "reason": x.get("reason")} for x in (i.get("not_run") or [])]} for i in items]


def _reconcile(items: list[dict]) -> None:
    """A run whose Step Functions execution ended without the aggregation step (stopped, failed,
    timed out) keeps its last status in the table; ask Step Functions and record what happened."""
    import boto3

    sfn = boto3.client("stepfunctions", region_name=REGION)
    base = _outputs()["RunStateMachineArn"].replace(":stateMachine:", ":execution:")
    for i in items:
        if i.get("status") not in ("running", "planned"):
            continue
        try:
            ex = sfn.describe_execution(executionArn=f"{base}:{i['pk'][4:]}")
        except Exception as e:  # noqa: BLE001 - executions older than 90 days are gone; leave the row as is
            i["reconcile_error"] = str(e)[:80]
            ex = None
        new = SFN_STATUS.get(ex["status"]) if ex else None
        if new:
            i["status"] = new
            errors = i.get("errors") or []
            if ex.get("error") or ex.get("cause"):
                errors = [failure_sentence(ex.get("error"), ex.get("cause"))]
                i["errors"] = errors
            stop = ex.get("stopDate")
            i["updated_at"] = stop.strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(stop, "strftime") else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _table().update_item(Key={"pk": i["pk"], "sk": "META"}, UpdateExpression="SET #s = :s, #e = :e, #u = :u",
                                 ExpressionAttributeNames={"#s": "status", "#e": "errors", "#u": "updated_at"},
                                 ExpressionAttributeValues={":s": new, ":e": errors, ":u": i["updated_at"]})


def _execution_row(ex: dict, sfn) -> dict:
    """A run known only to Step Functions (its execution died before preflight wrote the META row),
    shaped like a table row so the list and the run page can show it with its cause."""
    status = {"RUNNING": "running", "SUCCEEDED": "complete", **SFN_STATUS}.get(ex["status"], ex["status"].lower())
    errors: list[str] = []
    if ex["status"] in ("FAILED", "TIMED_OUT", "ABORTED"):
        try:
            d = sfn.describe_execution(executionArn=ex["executionArn"])
            errors = [failure_sentence(d.get("error"), d.get("cause"))]
        except Exception as e:  # noqa: BLE001
            errors = [f"Failed before any measurement ({str(e)[:80]})"]
    when = ex.get("stopDate") or ex.get("startDate")
    return {"pk": f"RUN#{ex['name']}", "sk": "META", "status": status, "errors": errors, "orphan": True,
            "updated_at": when.strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(when, "strftime") else None,
            "smoke": None, "purchase": None, "n_cells": 0}


def _run_yaml_exists(run_id: str) -> bool:
    import boto3

    if not RUN_ID.match(run_id):
        return False
    try:
        boto3.client("s3", region_name=REGION).head_object(Bucket=_outputs()["BucketName"], Key=f"runs/{run_id}/run.yaml")
        return True
    except Exception:  # noqa: BLE001 - 404 or offline
        return False


def _orphan_executions(known: set[str]) -> list[dict]:
    """Recent executions with no table row: before 2026-11-12 a run that failed in preflight (for
    example on a config field the deployed Lambda did not know) left no trace in the UI at all."""
    if _CACHE.get("orphans_at", 0) > time.time() - 30 and _CACHE.get("orphans_known") == known:
        return _CACHE["orphans"]
    import boto3

    out: list[dict] = []
    try:
        sfn = boto3.client("stepfunctions", region_name=REGION)
        exs = sfn.list_executions(stateMachineArn=_outputs()["RunStateMachineArn"], maxResults=60)["executions"]
        # A run the operator deleted also has no row but has no run.yaml either; only runs whose
        # definition is still in the bucket are shown.
        out = [_execution_row(ex, sfn) for ex in exs if ex["name"] not in known and _run_yaml_exists(ex["name"])]
    except Exception as e:  # noqa: BLE001 - offline: the table is the whole truth
        _CACHE["orphans_error"] = str(e)[:160]
    _CACHE["orphans"], _CACHE["orphans_at"], _CACHE["orphans_known"] = out, time.time(), known
    return out


def failure_sentence(error: str | None, cause: str | None) -> str:
    """A Step Functions failure for the reader: the Lambda's own message when the cause is its JSON
    envelope, plus what to do when it is the backend-is-older-than-the-app case."""
    msg = cause or ""
    try:
        env = json.loads(cause or "")
        msg = env.get("errorMessage") or env.get("Cause") or msg
    except (TypeError, ValueError):
        pass
    msg = " ".join(str(msg).split())[:400]
    text = f"Failed before any measurement: {error or 'error'}: {msg}" if msg else f"Failed before any measurement: {error or 'unknown error'}"
    if "Extra inputs are not permitted" in msg or "extra_forbidden" in msg:
        text += (" — the deployed backend (AmxBench-Core) is older than this app and does not know a setting this run used. "
                 "Deploy the Core stack, then start the run again.")
    return text


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
        # Known only to Step Functions? Then show what it says instead of 'unknown run'.
        import boto3

        if RUN_ID.match(run_id) and _run_yaml_exists(run_id):
            try:
                sfn = boto3.client("stepfunctions", region_name=REGION)
                arn = _outputs()["RunStateMachineArn"].replace(":stateMachine:", ":execution:") + f":{run_id}"
                d = sfn.describe_execution(executionArn=arn)
                items = [_execution_row({**d, "executionArn": arn}, sfn)]
            except Exception:  # noqa: BLE001 - truly unknown
                items = []
    if not items:
        raise HTTPException(404, "No record of this run: it was deleted, or it was never started.")
    meta = next((i for i in items if i["sk"] == "META"), {})
    if meta.get("status") in ("running", "planned"):
        _reconcile([meta])          # a failed or stopped execution shows as such, with its cause
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
    finished = meta.get("status") in ("complete", "refused")
    targets = target_status(run_id, sorted({c["hardware"] for c in cells}), finished, hw) if meta.get("status") != "refused" else []
    tmap = {t["hardware"]: t for t in targets}
    for c in cells:
        sibs = [x for x in cells if x["hardware"] == c["hardware"]]
        t = tmap.get(c["hardware"])
        c["status_label"] = derive_cell_status(c, sibs, t, meta.get("status"))
        if c["status"] in ("planned", "not_run") and t and t.get("did_not_run"):
            c["status"] = "not_run"
            c["error"] = c["error"] or t.get("not_run_reason") or ""
        if c["status"] == "ready" and all(lv["status"] in ("done", "failed") for lv in levels
                                          if lv["hardware"] == c["hardware"] and lv["model"] == c["model"]) and levels:
            c["status_label"] = "Done"
    notices = run_notices(targets, cells, meta.get("status"))
    status_label = RUN_STATUS.get(meta.get("status"), meta.get("status"))
    if meta.get("status") == "complete" and any(t.get("did_not_run") for t in targets):
        sides = sorted({"GPU" if t["hardware_label"].startswith("GPU") else "CPU" for t in targets if t.get("did_not_run")})
        status_label = f"Finished — {' and '.join(sides)} side did not run"
    if meta.get("status") in ("failed", "timed_out") and not cells:
        status_label = "Failed before the pre-checks" if meta.get("status") == "failed" else status_label
    return {"run_id": run_id, "name": meta.get("name"), "status": meta.get("status"), "status_label": status_label,
            "grade": "quick" if meta.get("smoke") else "full", "purchase": meta.get("purchase"), "updated_at": meta.get("updated_at"),
            "refusal": meta.get("errors") if meta.get("status") == "refused" else None, "notices": notices,
            "failure": (meta.get("errors") or None) if meta.get("status") in ("failed", "timed_out", "stopped") else None,
            "progress": {"done": done, "total": len(levels)}, "cells": cells, "targets": targets,
            "levels": sorted(levels, key=lambda x: (x["hardware"], x["model"], x["what"], x["conc"] or 0, x["pass"] or 0)),
            "has_report": meta.get("status") == "complete"}


RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")


def _run_items(run_id: str) -> list[dict]:
    from boto3.dynamodb.conditions import Key

    from lambdas.common import plain

    tbl, items, kw = _table(), [], {"KeyConditionExpression": Key("pk").eq(f"RUN#{run_id}")}
    while True:
        r = tbl.query(**kw)
        items += [plain(x) for x in r["Items"]]
        if "LastEvaluatedKey" not in r:
            break
        kw["ExclusiveStartKey"] = r["LastEvaluatedKey"]
    return items


def delete_run_data(run_id: str) -> dict:
    """Everything the run left behind: every S3 object under runs/<id>/ (plan, raw records,
    quality outputs, capacity and target logs, the report) and every row of RUN#<id> in the table.
    Instances, the shared model cache (models/) and other runs are untouched. Irreversible."""
    import boto3

    if not RUN_ID.match(run_id):
        raise HTTPException(400, "not a run id")
    items = _run_items(run_id)
    if not items and not _run_yaml_exists(run_id):
        raise HTTPException(404, "unknown run")
    meta = next((i for i in items if i["sk"] == "META"), {})
    if meta:
        _reconcile([meta])          # a run whose execution ended without the final step is not "running"
    if meta.get("status") in ("running", "planned"):
        raise HTTPException(409, "the run is still going: stop it first, then delete it")
    if not items:
        # Known only to Step Functions (failed in preflight): make sure it is not still running.
        import boto3

        try:
            arn = _outputs()["RunStateMachineArn"].replace(":stateMachine:", ":execution:") + f":{run_id}"
            if boto3.client("stepfunctions", region_name=REGION).describe_execution(executionArn=arn)["status"] == "RUNNING":
                raise HTTPException(409, "the run is still going: stop it first, then delete it")
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001 - no execution: nothing to protect
            _CACHE["delete_note"] = str(e)[:120]
    s3 = boto3.client("s3", region_name=REGION)
    bucket = _outputs()["BucketName"]
    keys: list[str] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"runs/{run_id}/"):
        keys += [o["Key"] for o in page.get("Contents", [])]
    for i in range(0, len(keys), 1000):
        s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]], "Quiet": True})
    with _table().batch_writer() as bw:
        for it in items:
            bw.delete_item(Key={"pk": it["pk"], "sk": it["sk"]})
    for k in (f"summary:{run_id}", f"purchase:{run_id}", "cap_at", "orphans_at"):
        _CACHE.pop(k, None)
    return {"deleted": run_id, "objects": len(keys), "rows": len(items), "status_was": meta.get("status") or "failed"}


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    return delete_run_data(run_id)


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str):
    import boto3

    arn = _outputs()["RunStateMachineArn"].replace(":stateMachine:", ":execution:")
    boto3.client("stepfunctions", region_name=REGION).stop_execution(executionArn=f"{arn}:{run_id}", cause="stopped from the UI")
    _table().update_item(Key={"pk": f"RUN#{run_id}", "sk": "META"}, UpdateExpression="SET #s = :s",
                         ExpressionAttributeNames={"#s": "status"}, ExpressionAttributeValues={":s": "stopped"})
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


class SummaryIn(CompareIn):
    force: bool = False


@app.get("/api/executive-summary/model")
def exec_model():
    import boto3

    from webui.api import exec_summary

    try:
        return exec_summary.newest_opus(boto3.client("bedrock", region_name=REGION))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"Bedrock is not reachable: {str(e)[:200]}") from e


@app.post("/api/executive-summary")
def exec_summary_post(c: SummaryIn):
    """Executive summary of the current comparison, written by the newest Opus on Bedrock.
    Cached per (runs, targets, limit, models) so re-rendering the page does not re-bill."""
    from webui.api import exec_summary

    out = post_compare(CompareIn(run_ids=c.run_ids, sla=c.sla, limit=c.limit, models=c.models))
    if not out["analysis"]:
        raise HTTPException(400, "the selected runs are not comparable; fix that first")
    key = "exec:" + json.dumps({"r": sorted(c.run_ids), "s": out["sla"], "l": c.limit, "m": c.models}, sort_keys=True, default=str)
    if key in _CACHE and not c.force:
        return _CACHE[key]
    try:
        res = exec_summary.write(out["analysis"], out["comparability"], out["sla"], out["runs"])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Bedrock call failed: {str(e)[:300]}") from e
    _CACHE[key] = res
    return res


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
