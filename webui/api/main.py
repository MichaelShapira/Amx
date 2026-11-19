"""Local API for the web UI. Binds 127.0.0.1 only; every AWS call uses the operator's own
credentials. Serves the built React app from webui/dist when present.

    .venv/bin/python -m webui.api.main            # http://127.0.0.1:8601
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import catalog_store, pairing, report, utilization  # noqa: E402
from bench.config import RunConfig, load_targets, resolve_cells, run_instance_types  # noqa: E402
from bench.estimate_run import estimate  # noqa: E402
from bench.plan import GENERATIVE, levels_for_cell  # noqa: E402
from bench.pools import local_pool_sizes  # noqa: E402
from webui.api import auth, aws, compare, headers  # noqa: E402
from webui.api import catalog as C  # noqa: E402

REGION, ACCOUNT = "us-east-1", "346399954218"
app = FastAPI(title="GPU vs AMX CPU benchmark", docs_url=None, redoc_url=None)
_CACHE: dict[str, Any] = {}
catalog_store.s3_factory = lambda: aws.client("s3")       # the live catalog is read with the app's (file-refreshed) credentials

# Cloud deployment (AmxBench-WebApp): every /api call needs a Cognito access token, see auth.py.
# Locally (no pool configured, loopback bind) the gate is off. Security headers go on every
# response; HSTS only where the app is served over HTTPS.
def _header_policy() -> tuple[str, bool]:
    s = auth.settings()
    if not s:
        return headers.csp([]), False
    return headers.csp([s["domain"], f"https://cognito-idp.{s['region']}.amazonaws.com"]), True


app.add_middleware(auth.CognitoAuthMiddleware)
app.add_middleware(headers.SecurityHeadersMiddleware, policy=_header_policy)


@app.get("/api/health")
def health():
    """Liveness for the platform's health check; says nothing about data or identity."""
    return {"ok": True}


@app.get("/api/auth/config")
def auth_config():
    """Public: what the browser needs to start the sign-in (pool, client id, hosted-login
    domain). `enabled: false` means the local, loopback-only mode."""
    return auth.public_config()


@app.exception_handler(ClientError)
@app.exception_handler(NoCredentialsError)
@app.exception_handler(PartialCredentialsError)
async def _aws_failure(request, exc):
    """An AWS call failed inside a route. Until 2026-11-15 this surfaced as a bare 500 "Internal
    Server Error" on every page once the operator's one-hour session had expired. Now: 401 with
    the one sentence that says what to do when the session is gone (the UI shows it and keeps
    polling /api/aws/status), 502 with the service's own message for anything else."""
    if aws.is_expired(exc):
        return JSONResponse(status_code=401, content={"detail": aws.expired_sentence(), "code": "aws_session_expired"})
    err = getattr(exc, "response", {}).get("Error", {}) if isinstance(exc, ClientError) else {}
    op = getattr(exc, "operation_name", None) or "an AWS call"
    msg = f"{op} failed: {err.get('Code', type(exc).__name__)}: {str(err.get('Message') or exc)[:300]}"
    return JSONResponse(status_code=502, content={"detail": msg, "code": "aws_error"})


# ----------------------------------------------------------------------------- catalog and data

@app.get("/api/catalog")
def get_catalog():
    """The wizard's choices. Cached for a minute: the model list is live (Setup adds and removes
    models) and a change must show on the next page load, not the next restart."""
    if "catalog" not in _CACHE or _CACHE.get("catalog_at", 0) < time.time() - 60:
        _CACHE["catalog"], _CACHE["catalog_at"] = C.catalog(), time.time()
    return _CACHE["catalog"]


@app.get("/api/data/{use_case}")
def get_data(use_case: str):
    return C.dataset_info(use_case)


@app.get("/api/data/{use_case}/sample")
def get_sample(use_case: str, prompt_set: str | None = None, model: str | None = None, workload: str | None = None,
               label: str | None = None, hard_case: str | None = None, n: int = 5, seed: int = 0,
               direction: str | None = None):
    return C.sample(use_case, prompt_set=prompt_set, model=model, workload=workload, label=label, hard_case=hard_case,
                    n=min(n, 20), seed=seed, direction=direction)


# ----------------------------------------------------------------------------- wizard -> run config

class Wizard(BaseModel):
    """What the wizard collects. Everything else in the run config is derived here."""
    use_case: str = "standard"
    prompt_set: str = "short"
    answer_tokens: int = 128
    workloads: list[str] = ["W1", "W3"]
    test_items: int = 300
    preset: str = "inline_standard"
    directions: list[str] = ["en-de", "de-en", "en-es", "es-en"]     # translation: which language pairs
    text_size: str = "sentence"                                       # translation and embedding: sentence | paragraph
    translation_items: int = 100                                      # translation: items per direction in the quality pass
    embedding_items: int = 300                                        # embedding: texts in the quality pass
    intel_io: str = "128/128"                                         # intel: input/output tokens
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
    sized_for: str | None = None               # "<model>|<precision>" the hardware step last sized for
    copied_from: str | None = None             # run id this draft came from ("Copy to new run")
    recommendation: str | None = None          # what "Re-run with recommended sizes" changed, for the banner


def _unavailable_reason(mid: str, precision: str, hardware: str, cat: dict) -> str | None:
    """Why this number format of this model cannot run on this hardware option, or None when it
    can. Two reasons exist: the option's serving software does not serve the model in that format
    (Qwen in BF16 runs on vLLM/PyTorch, not on the OpenVINO option), or the format needs a CPU
    feature the option's instances do not expose (FP16 needs AMX-FP16, which no AWS Intel
    instance has). The wizard greys the format out from the same two facts, so a run that
    reaches here with it is a stale copy, and it is dropped with this sentence."""
    m = next((x for x in cat["models"] if x["id"] == mid), None)
    h = next((x for x in cat["hardware"] if x["id"] == hardware), None)
    if not m or not h:
        return None                     # let the config validator report unknown ids
    spec = m["precisions"].get(precision)
    if not spec or spec.get("quality_only") or not any(e in h["engines"] for e in spec["engines"]):
        return "is not available in {fmt} on this serving software; that measurement is left out."
    flag = spec.get("requires_cpu_flag")
    if flag and h["accelerator"] == "cpu" and flag not in (h.get("cpu_features") or []):
        return "in {fmt} " + C.cpu_flag_note(flag) + " That measurement is left out."
    return None


def _sla_for_models(w: Wizard, cat: dict) -> dict:
    """The wizard's targets, minus the first-word and per-word limits when no chosen model generates
    tokens (a classifier is judged on the whole decision only; the limits would read 'not measured')."""
    sla = {k: v for k, v in (w.sla or {}).items() if v and k != "max_error_pct"}
    types = {m["id"]: m["type"] for m in cat["models"]}
    if not any(types.get(m) in GENERATIVE for m in w.models):
        sla = {k: v for k, v in sla.items() if k not in report.GENERATION_ONLY_TARGETS}
    return sla


def _run_dict(w: Wizard, notes: list[str] | None = None) -> dict:
    """The run config for the wizard state. A number format the hardware option cannot serve for
    the model is dropped with a note (never a hard failure): ticking the OpenVINO option with a
    chat model whose BF16 runs only on vLLM must not kill the plan."""
    cat = get_catalog()
    depth = C.DEPTHS.get(w.depth)
    requests = (w.requests or 50) if w.depth == "custom" else (depth[2] if depth else 50)
    passes = (w.passes or 1) if w.depth == "custom" else (depth[3] if depth else 1)
    conc = [1] if w.request_mode == "single" else sorted(set(int(c) for c in w.concurrency)) or [1]
    sides = ([w.gpu] if w.gpu else []) + list(w.cpus)
    hw = {h["id"]: h for h in cat["hardware"]}
    cells, itypes, alts = [], {}, {}
    for side in sides:
        before, dropped = len(cells), []
        for p in side.get("precisions") or [w.precision]:
            for mid in w.models:
                why = _unavailable_reason(mid, p, side["hardware"], cat)
                if why is None:
                    cells.append({"target": side["hardware"], "precision": p, "model": mid})
                else:
                    dropped.append(f"{hw.get(side['hardware'], {}).get('label', side['hardware'])}: {C.model_name(mid)} "
                                   + why.format(fmt=C.PRECISIONS.get(p, (p,))[0]))
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
        "prompt_set": (w.prompt_set if w.use_case == "standard" else w.text_size if w.use_case in ("translation", "embedding")
                       else "intel" if w.use_case == "intel" else "dspm"),
        "sampling": {"temperature": 0.0,
                     "max_new_tokens": (int(w.answer_tokens) if w.use_case == "standard"
                                        else C.TEXT_SIZES.get(w.text_size, C.TEXT_SIZES["sentence"])[2] if w.use_case == "translation"
                                        else _intel_io(w)[1] if w.use_case == "intel"
                                        else 4),
                     "ignore_eos": w.use_case in ("standard", "intel"), "seed": 0},
        "concurrency": conc, "per_level": {"requests": int(requests)}, "repeats": int(passes), "warmup_requests": 20,
        "quality": {"enabled": w.use_case == "standard"},
        "sla": _sla_for_models(w, cat) | {"max_error_pct": float((w.sla or {}).get("max_error_pct") or 1.0)},
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
    if w.use_case == "translation":
        run["translation"] = {"directions": [d for d in w.directions if d in C.DIRECTIONS] or list(C.DIRECTIONS),
                              "test_items": int(w.translation_items) if w.translation_items else None}
    if w.use_case == "embedding":
        run["embedding"] = {"test_items": int(w.embedding_items) if w.embedding_items else None}
        run["quality"] = {"enabled": False}
    if w.use_case == "intel":
        i, o = _intel_io(w)
        run["intel"] = {"input_tokens": i, "output_tokens": o}
        run["quality"] = {"enabled": False}
        # Intel's pass criteria, unless the wizard set its own: 100 ms per generated token, 3 s to the first token.
        run["sla"] = {"p50_tpot_ms": 100.0, "p99_ttft_ms": 3000.0, "max_error_pct": 1.0} | {k: v for k, v in run["sla"].items() if v}
        run["request_timeout_s"] = 600
    return run


def _intel_io(w: Wizard) -> tuple[int, int]:
    io = w.intel_io if w.intel_io in C.INTEL_IO else "128/128"
    i, o = io.split("/")
    return int(i), int(o)


def _dedupe_cells(run: dict) -> dict:
    seen, out = set(), []
    for c in run["cells"]:
        k = (c["target"], c["precision"], c["model"])
        if k not in seen:
            seen.add(k)
            out.append(c)
    return {**run, "cells": out}


def refusal_sentences(errors: list[str]) -> list[str]:
    """The pre-check refusals (lambdas/preflight.py) for a reader who is not an engineer: every
    known message becomes what went wrong and what to do, with model and hardware names instead
    of ids. Unknown messages are passed through."""
    cat = get_catalog()
    hw = {h["id"]: h.get("label", h["id"]) for h in cat["hardware"]}
    out = []
    for e in errors or []:
        e = str(e)
        # The config validator's form: "<model>/<precision> needs CPU flag <flag>, which <target> lacks".
        m_flag = re.match(r"^([\w.-]+)/([\w.-]+) needs CPU flag (\w+), which ([\w.-]+) lacks$", e)
        if m_flag:
            mid, prec, flag, tid = m_flag.groups()
            out.append(f"{C.model_name(mid)} on {hw.get(tid, tid)} in {C.PRECISIONS.get(prec, (prec,))[0]}: this format "
                       f"{C.cpu_flag_note(flag)} Untick it for this option.")
            continue
        m_cell = re.match(r"^([\w.-]+)\|([\w.-]+)\|([\w.-]+): (.*)$", e)
        if m_cell:
            mid, tid, prec, rest = m_cell.groups()
            where = f"{C.model_name(mid)} on {hw.get(tid, tid)} in {C.PRECISIONS.get(prec, (prec,))[0]}"
            if "has no amx_fp16" in rest or "lack amx_fp16" in rest:
                is_gpu = next((h for h in cat["hardware"] if h["id"] == tid), {}).get("accelerator") == "gpu"
                if is_gpu:
                    out.append(f"{where}: refused by mistake. FP16 on a GPU never depended on a CPU feature; the pre-check "
                               "that said so was fixed on 2026-11-12. Start the run again.")
                else:
                    itype = rest.split(" has no")[0] if " has no" in rest else None
                    out.append(f"{where}: the FP16 (half precision) format {C.cpu_flag_note('amx_fp16')}"
                               + (f" This run asked for it on {itype}." if itype else "")
                               + " Since 2026-11-15 the wizard greys FP16 out for the CPU options, so this cannot happen "
                               "to a new run.")
            elif "none of" in rest and "available on" in rest:
                out.append(f"{where}: this serving software cannot run the model in that number format. Untick the format "
                           "for this option.")
            else:
                out.append(f"{where}: {rest}")
            continue
        m = re.match(r"^worker reserved concurrency (\d+) < highest level (\d+)", e)
        if m:
            out.append(f"The load generator in this deployment can send at most {m.group(1)} requests at a time, and this run asks "
                       f"for {m.group(2)}. Lower the highest 'requests at a time' value, or redeploy the Core stack with a higher "
                       "worker concurrency.")
            continue
        m = re.match(r"^instance types not offered in ([\w-]+): \[(.*)\]", e)
        if m:
            types = m.group(2).replace("'", "")
            out.append(f"AWS does not offer these instance sizes in this zone ({m.group(1)}): {types}. Pick another size on the "
                       "Hardware step.")
            continue
        m = re.match(r"^target ([\w-]+) is not deployed", e)
        if m:
            out.append(f"The hardware option {hw.get(m.group(1), m.group(1))} is not deployed in this account "
                       f"(stack AmxBench-Target-{m.group(1)}). Deploy it, or untick the option.")
            continue
        m = re.match(r"^Hugging Face access to gated ([\w./-]+)@(\w+) returned HTTP (\d+)", e)
        if m:
            out.append(f"The model {m.group(1)} is gated on Hugging Face and the stored token was refused (HTTP {m.group(3)}). "
                       "Accept the model's license with the account that owns the token, or store a token that has access, "
                       "then start the run again.")
            continue
        if e.startswith("price unavailable"):
            out.append("The On-Demand price of an instance size could not be read from the AWS Price List, and without it the "
                       f"cost columns cannot be computed. Try again in a minute. Details: {e[len('price unavailable: '):][:160]}")
            continue
        if e.startswith("DSPM suite needs the dataset"):
            out.append("The sensitive-data test data has not been uploaded to this account's bucket. From the repository run "
                       "`bench upload-artifacts`, then start the run again.")
            continue
        if e.startswith("DSPM request pools were built from a different dataset"):
            out.append("The sensitive-data request pools in the bucket were built from a different dataset version than the "
                       "one uploaded, so the answers could not be scored. Rebuild them (`tools.dspm.build_requests`) and run "
                       "`bench upload-artifacts`.")
            continue
        if e.startswith("Translation suite needs the parallel corpus"):
            out.append("The translation corpus has not been uploaded to this account's bucket. From the repository run "
                       "`python -m tools.translation.build_dataset`, `python -m tools.translation.build_requests` and "
                       "`bench upload-artifacts`, then start the run again.")
            continue
        if e.startswith("translation request pools were rendered from a different corpus"):
            out.append("The translation request pools in the bucket were rendered from a different corpus version than the "
                       "one uploaded, so the translations could not be scored. Rebuild them (`tools.translation.build_requests`) "
                       "and run `bench upload-artifacts`.")
            continue
        out.append(e)
    return out


@app.post("/api/plan")
def post_plan(w: Wizard):
    """Validate the wizard state and return the plan: series, estimate, fairness, warnings, YAML."""
    import yaml

    notes: list[str] = []
    run = _dedupe_cells(_run_dict(w, notes))
    models, targets = catalog_store.load_catalog(), load_targets(ROOT / "config/targets.yaml")
    tmap = targets.resolved()
    problems: list[str] = []
    if not w.models:
        problems.append("Pick at least one model.")
    if not w.gpu and not w.cpus:
        problems.append("Pick at least one hardware option.")
    if w.use_case == "dspm" and not w.workloads:
        problems.append("Pick at least one detection scenario.")
    if w.use_case == "translation" and not [d for d in w.directions if d in C.DIRECTIONS]:
        problems.append("Pick at least one translation direction.")
    if w.use_case == "intel" and w.request_mode == "batch":
        problems.append("Intel's benchmark sends one prompt per request; pick single or concurrent requests.")
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
        if c.engine == "vllm" and tmap[c.target].accelerator == "cpu":
            # The host must hold the model with its KV cache (bench/cellspec.py); the pre-check would refuse it too.
            from bench import cellspec
            eng = tmap[c.target].engines["vllm"]
            try:
                cellspec.cpu_kv_cache_gib(int(eng.args.get("kv_cache_gib", 16)), tmap[c.target], rtypes[c.target],
                                          models.get(c.model), c.precision)
            except ValueError as e:
                problems.append(str(e))
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
    if any((w.sla or {}).get(k) for k in report.GENERATION_ONLY_TARGETS) and not any(k in run["sla"] for k in report.GENERATION_ONLY_TARGETS):
        warnings.append("A classifier generates no words, so the first-word and per-word targets do not apply and were dropped; "
                        "only the whole-decision limit is checked.")
    for m in w.models:
        ms = models.get(m)
        if ms.license == "qwen-research":
            warnings.append(f"{C.model_name(m)} has a research-only license; its results are for evaluation, not a commercial decision.")
    if unfair and not w.allow_unfair:
        # Two CPU options on the same instance type share one reason: say it once.
        reasons = list(dict.fromkeys(p["reason"].rstrip(".") for p in unfair))
        problems.append("Unfair pairing: " + ". ".join(reasons) + ". Pick comparable sizes, or allow the pairing explicitly below.")
    return {"ok": not problems and not missing, "problems": problems + missing, "warnings": warnings, "series": series,
            "pairing": pairs, "estimate": est, "grade": "quick" if rc.is_smoke else "full",
            "yaml": yaml.safe_dump({"run": run}, sort_keys=False), "run": run,
            "request_mode": compare.request_mode(run)}


def _ai_context(w: Wizard) -> dict:
    """What the reasoning model is told: the chosen model(s), the test, every allowed instance size
    with its catalog facts, the fairness rule, recent capacity refusals and, when the page already
    has a selection, that selection with the tool's own price check. No sizing rule of ours: the
    model catalog is open, so the proposal is reasoned per model (ai_propose.py)."""
    from webui.api import ai_propose

    cat = get_catalog()
    mods = [m for m in cat["models"] if m["id"] in w.models]
    if not mods:
        raise HTTPException(400, "pick at least one model first")
    specs = {m.id: m for m in catalog_store.load_catalog().models}
    hw = {h["id"]: h for h in cat["hardware"]}
    instances = cat["instances"]
    depth = C.DEPTHS.get(w.depth)
    requests = (w.requests or 50) if w.depth == "custom" else (depth[2] if depth else 50)
    passes = (w.passes or 1) if w.depth == "custom" else (depth[3] if depth else 1)
    test = {"use case": C.USE_CASES[w.use_case]["name"] if w.use_case in C.USE_CASES else w.use_case,
            "request shape": C.REQUEST_MODES.get(w.request_mode, (w.request_mode,))[0],
            "requests at a time (each measured separately)": ", ".join(map(str, sorted(set(int(c) for c in w.concurrency)))) if w.request_mode != "single" else "1",
            "requests per measurement": requests, "repeated": f"{passes} time(s)", "number format": w.precision}
    if w.request_mode == "batch":
        test["inputs per request"] = ", ".join(map(str, w.batch_sizes))
    if w.use_case == "standard":
        test["prompts"] = f"{C.PROMPT_SETS.get(w.prompt_set, (w.prompt_set,))[0]}, answers of {w.answer_tokens} tokens"
        test["targets"] = ", ".join(f"{k} <= {v}" for k, v in (w.sla or {}).items() if v) or "none"
    elif w.use_case == "translation":
        test["text"] = f"{C.TEXT_SIZES.get(w.text_size, (w.text_size,))[0]}, synthetic EN/DE/ES parallel corpus"
        test["directions"] = ", ".join(C.DIRECTIONS.get(d, (d,))[0] for d in w.directions)
        test["targets"] = ", ".join(f"{k} <= {v}" for k, v in (w.sla or {}).items() if v) or "none"
    elif w.use_case == "embedding":
        test["text"] = f"{C.TEXT_SIZES.get(w.text_size, (w.text_size,))[0]} in English, German and Spanish, embedded"
        test["targets"] = ", ".join(f"{k} <= {v}" for k, v in (w.sla or {}).items() if v) or "none"
    elif w.use_case == "intel":
        test["shape"] = f"Intel's benchmark, {C.INTEL_IO.get(w.intel_io, (w.intel_io,))[0]} tokens, random prompts, fixed answer length"
        test["targets"] = "Intel's: 100 ms per generated token (median), first token within 3 s"
    else:
        test["scenarios"] = "; ".join(C.WORKLOADS.get(x, (x,))[0] for x in w.workloads)
        test["decision deadline"] = C.PRESETS.get(w.preset, (w.preset, "", None))[0]
    proven = _proven_engines() if any(specs.get(m["id"]) and specs[m["id"]].user_added for m in mods) else {}
    models_ctx = []
    for m in mods:
        s = specs.get(m["id"])
        models_ctx.append({"name": m["name"], "hf_repo": s.hf_repo if s else "?", "kind": m["kind"], "type": m["type"],
                           "params_b": m["params_b"], "license": m["license"], "max_input_tokens": s.max_input_tokens if s else "?",
                           "precisions": {p: v["engines"] for p, v in m["precisions"].items()},
                           "user_added": bool(s and s.user_added), "notes": s.notes if s else None,
                           "proven_engines": sorted(proven.get(m["id"], ()))})
    allowed = {side: [(t, instances[t]) for t in pairing.types_for(side) if t in instances] for side in ("gpu", "cpu")}
    current = None
    if w.gpu and w.cpus and w.gpu["instance_type"] in instances and all(c["instance_type"] in instances for c in w.cpus):
        current = {"gpu": {"instance_type": w.gpu["instance_type"], "software": hw.get(w.gpu["hardware"], {}).get("software", [])},
                   "cpus": [{"instance_type": c["instance_type"], "software": hw.get(c["hardware"], {}).get("software", [])} for c in w.cpus],
                   "pairing": [pairing.judge(c["instance_type"], w.gpu["instance_type"]) for c in w.cpus]}
    watched = {t for t, _ in allowed["gpu"]} | {t for t, _ in allowed["cpu"]}
    return {"region": REGION, "az": "use1-az1", "price_snapshot": cat.get("price_snapshot"), "test": test, "models": models_ctx,
            "allowed": allowed, "fair_ratio": list(pairing.FAIR_RATIO), "current": current,
            "capacity_notes": {k: v for k, v in _capacity_notes().items() if k in watched}, "_module": ai_propose}


@app.post("/api/propose/ai")
def post_propose_ai(w: Wizard):
    """Start the hardware proposal as a background job; poll /api/propose/ai/{id}."""
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
        s3 = aws.client("s3")
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
    """The Core stack's outputs (bucket, state machine). The cloud deployment gets them from the
    environment (AMXBENCH_OUTPUTS, JSON, set by the WebApp stack from the Core stack's own
    values) so its role needs no CloudFormation permission; locally they are read from the stack."""
    if "outputs" in _CACHE and _CACHE.get("outputs_at", 0) > time.time() - 300:
        return _CACHE["outputs"]
    env_json = os.environ.get("AMXBENCH_OUTPUTS", "").strip()
    if env_json:
        out = json.loads(env_json)
    else:
        stk = aws.client("cloudformation").describe_stacks(StackName="AmxBench-Core")["Stacks"][0]
        out = {o["OutputKey"]: o["OutputValue"] for o in stk.get("Outputs", [])}
    _CACHE["outputs"], _CACHE["outputs_at"] = out, time.time()
    return out


@app.get("/api/aws/status")
def aws_status():
    """Never raises: the sidebar pill polls this, and when the session is gone the answer is the
    same sentence the failing pages show, with `expired: true` so the UI can say "AWS session
    expired" instead of "AWS not reachable"."""
    try:
        ident = aws.client("sts").get_caller_identity()
        out = _outputs()
        dep = deployed_backend()
        return {"ok": True, "account": ident["Account"], "identity": ident["Arn"].split("/")[-1], "bucket": out.get("BucketName"),
                "credentials": aws.source(),
                "backend": {"commit": dep["commit"], "local_commit": dep["local_commit"], "behind": dep["behind"], "missing": dep["missing"]}}
    except Exception as e:  # noqa: BLE001
        if aws.is_expired(e):
            return {"ok": False, "expired": True, "error": aws.expired_sentence(), "credentials": aws.source()}
        return {"ok": False, "expired": False, "error": str(e)[:300], "credentials": aws.source()}


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
        env = aws.client("lambda").get_function_configuration(FunctionName=PREFLIGHT_FUNCTION)
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
    plan = post_plan(w)
    if not plan["ok"]:
        raise HTTPException(400, "; ".join(plan["problems"]))
    out = _outputs()
    run_id = f"{plan['run']['name']}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    aws.client("s3").put_object(Bucket=out["BucketName"], Key=f"runs/{run_id}/run.yaml",
                                                      Body=plan["yaml"].encode())
    # The row first, so the run is on the list the moment it starts; preflight overwrites the status.
    _table().put_item(Item={"pk": f"RUN#{run_id}", "sk": "META", "status": "planned", "name": w.name.strip() or plan["run"]["name"],
                            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "purchase": plan["run"]["purchase"]["type"], "smoke": plan["grade"] == "quick",
                            "n_cells": len(plan["series"]), "model": (w.models or [None])[0]})
    aws.client("stepfunctions").start_execution(
        stateMachineArn=out["RunStateMachineArn"], name=run_id, input=json.dumps({"run_id": run_id}))
    return {"run_id": run_id}


def _table():
    return aws.table()


RUN_STATUS = {"running": "Running", "complete": "Finished", "refused": "Refused by the pre-checks", "planned": "Starting",
              "stopped": "Stopped by the operator", "failed": "Failed", "timed_out": "Timed out"}
SFN_STATUS = {"ABORTED": "stopped", "FAILED": "failed", "TIMED_OUT": "timed_out"}
CELL_STATUS = {"planned": "Waiting", "configuring": "Loading the model", "ready": "Measuring",
               "bringup_failed": "Failed to start", "interrupted": "Instance lost, replacing it", "failed": "Failed",
               "done": "Done", "not_run": "Not run — no capacity"}
INSTANCE_STATE = {"pending": "Starting", "running": "Running", "stopping": "Stopping", "stopped": "Stopped",
                  "shutting-down": "Terminating", "terminated": "Terminated"}


def _s3_json(key: str):
    try:
        return json.loads(aws.client("s3").get_object(Bucket=_outputs()["BucketName"], Key=key)["Body"].read())
    except Exception:  # noqa: BLE001 - absent until the first event
        return None


def _instance_states(ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}

    out = {}
    try:
        r = aws.client("ec2").describe_instances(InstanceIds=ids)
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


def _unwrap_error(raw: str) -> str:
    """Older rows hold the Python repr of the Step Functions error object; newer ones the message.
    Either way, return the Lambda's own sentence."""
    s = str(raw).strip()
    if s.startswith("{") and "'Cause'" in s or s.startswith("{") and '"Cause"' in s:
        import ast

        try:
            obj = ast.literal_eval(s) if "'Cause'" in s else json.loads(s)
            cause = obj.get("Cause") or ""
            try:
                env = json.loads(cause)
                return str(env.get("errorMessage") or cause)
            except (TypeError, ValueError):
                return str(cause or obj.get("Error") or s)
        except (ValueError, SyntaxError):
            return s
    return s


def cell_error_text(raw: str, stderr: str = "") -> str:
    """A cell's failure for the reader: the known signatures get a sentence, then the root cause
    (the last exception line that is not a wrapper, from the stored stderr when the short error
    lost it), then the raw tail."""
    from lambdas.target_control import root_cause

    s = _unwrap_error(raw)
    if not s:
        return ""
    tail = s.split("failed (Failed):", 1)[1].strip() if "failed (Failed):" in s else s
    cause = root_cause(stderr) or root_cause(s)
    if "No such container" in s and "exited before becoming healthy" in s:
        return ("Another run took over this instance while the model was loading (before 2026-11-13 runs shared one "
                "instance per hardware option; every run now gets its own). Start this benchmark again. Details: " + tail[-200:])
    if "export did not report a hash" in s or "Failed to export llm model" in s or "optimum-cli export" in s \
            or "OpenVINO export failed" in s:
        if cause and "does not allow for the export of the model" in cause:
            why = ("the converter release refused this model because its version table stops at an older Transformers "
                   "library than the one pinned (Gemma 2: cap 5.0, pinned 5.5.4). Upstream lifted the cap; fixed here "
                   "2026-11-15, start this benchmark again")
        elif "Failed to export llm model" in s:
            why = ("the export tool was not told the model's task, which it cannot infer for a local copy of the model (fixed "
                   "2026-11-12; start this benchmark again)")
        else:
            why = "see the details"
        return f"Converting the model to OpenVINO's format failed: {why}. " + (f"Cause: {cause}" if cause else f"Details: {tail[-240:]}")
    if "exited before becoming healthy" in s:
        if cause:
            return f"The serving software stopped while loading the model. Cause: {cause}"
        return ("The serving software stopped while loading the model; the run kept only the last lines of its log, which do "
                "not name the cause (runs since 2026-11-15 keep the whole log). Details: " + tail[-240:])
    if "first request failed on every API path" in s:
        answers = re.findall(r"first request to (\S+) -> HTTP (\S+): (.*?)(?=\s+\d{4}-\d{2}-\d{2}T|\s+ERROR:|$)", s, re.S)
        if answers:
            said = "; ".join(f"{path} answered HTTP {code}: {body.strip()[:200]}" for path, code, body in answers[:3])
            return ("The serving software started and reported healthy, but rejected the first real request. " + said +
                    (f" Cause: {cause}" if cause else ". The whole server log is kept under bringup/ in the run's folder."))
        return ("The serving software started and reported healthy, but rejected the first real request, and this run (before "
                "2026-11-18) kept neither the server's answer nor its log, so the cause is not recorded. Runs since then keep both. "
                "Details: " + tail[-240:])
    if "bring-up of" in s and not tail:
        return ("The instance was taken away while the model was loading, and the run gave up after its replacements were "
                "taken away too (before 2026-11-13 runs shared one instance per hardware option; every run now gets its own). "
                "Start this benchmark again.")
    if "bring-up of" in s:
        return "The model could not be served on this instance. Details: " + tail[-300:]
    return s[:300]


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
             "n_cells": i.get("n_cells"), "errors": refusal_sentences(i.get("errors")) if i.get("status") == "refused" else i.get("errors"),
             "not_run": [{"hardware": x.get("target_id"), "hardware_label": hw.get(x.get("target_id"), {}).get("label", x.get("target_id")),
                          "reason": x.get("reason")} for x in (i.get("not_run") or [])]} for i in items]


def _reconcile(items: list[dict]) -> None:
    """A run whose Step Functions execution ended without the aggregation step (stopped, failed,
    timed out) keeps its last status in the table; ask Step Functions and record what happened."""

    sfn = aws.client("stepfunctions")
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
    if not RUN_ID.match(run_id):
        return False
    try:
        aws.client("s3").head_object(Bucket=_outputs()["BucketName"], Key=f"runs/{run_id}/run.yaml")
        return True
    except Exception:  # noqa: BLE001 - 404 or offline
        return False


def _orphan_executions(known: set[str]) -> list[dict]:
    """Recent executions with no table row: before 2026-11-12 a run that failed in preflight (for
    example on a config field the deployed Lambda did not know) left no trace in the UI at all."""
    if _CACHE.get("orphans_at", 0) > time.time() - 30 and _CACHE.get("orphans_known") == known:
        return _CACHE["orphans"]

    out: list[dict] = []
    try:
        sfn = aws.client("stepfunctions")
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

        if RUN_ID.match(run_id) and _run_yaml_exists(run_id):
            try:
                sfn = aws.client("stepfunctions")
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
                          "error": cell_error_text(i.get("error") or i.get("reason") or i.get("stderr") or "", i.get("stderr") or "")})
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
            "refusal": refusal_sentences(meta.get("errors")) if meta.get("status") == "refused" else None, "notices": notices,
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

        try:
            arn = _outputs()["RunStateMachineArn"].replace(":stateMachine:", ":execution:") + f":{run_id}"
            if aws.client("stepfunctions").describe_execution(executionArn=arn)["status"] == "RUNNING":
                raise HTTPException(409, "the run is still going: stop it first, then delete it")
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001 - no execution: nothing to protect
            _CACHE["delete_note"] = str(e)[:120]
    terminated = terminate_run_instances(run_id)      # a leftover instance of a finished run goes with it
    s3 = aws.client("s3")
    bucket = _outputs()["BucketName"]
    keys: list[str] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"runs/{run_id}/"):
        keys += [o["Key"] for o in page.get("Contents", [])]
    for i in range(0, len(keys), 1000):
        s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]], "Quiet": True})
    tbl = _table()
    for it in items:                               # one DeleteItem per row: the role has no BatchWriteItem
        tbl.delete_item(Key={"pk": it["pk"], "sk": it["sk"]})
    for k in (f"summary:{run_id}", f"purchase:{run_id}", "cap_at", "orphans_at"):
        _CACHE.pop(k, None)
    return {"deleted": run_id, "objects": len(keys), "rows": len(items), "status_was": meta.get("status") or "failed",
            "terminated_instances": terminated}


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    return delete_run_data(run_id)


def terminate_run_instances(run_id: str) -> list[str]:
    """Every instance this run launched (tags bench:managed + bench:run), terminated now. Used when
    a run is stopped from the UI (the state machine's own end-of-run step does not execute for an
    aborted execution; without this the reaper would only catch them after 30 idle minutes) and
    when a run is deleted."""

    if not RUN_ID.match(run_id):
        return []
    ec2 = aws.client("ec2")
    r = ec2.describe_instances(Filters=[{"Name": "tag:bench:managed", "Values": ["true"]},
                                        {"Name": "tag:bench:run", "Values": [run_id]},
                                        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}])
    ids = [i["InstanceId"] for res in r["Reservations"] for i in res["Instances"]]
    if ids:
        ec2.terminate_instances(InstanceIds=ids)
    return ids


def wizard_from_run(run: dict, name: str | None = None, source_run_id: str | None = None) -> dict:
    """The wizard draft that reproduces a stored run definition (run.yaml / resolved.json["run"]):
    'Copy to new run'. Nothing is started; the user lands on the Review step and can change any
    step. Sizes and formats are carried over as they ran, so the hardware step does not re-size
    (sized_for is set)."""
    cat = get_catalog()
    hw = {h["id"]: h for h in cat["hardware"]}
    suite = run.get("suite", "standard")
    # run.yaml is written without defaults: a run with one repeat has no `repeats` key (and an older run
    # may lack per_level). Fill the RunConfig defaults, else the draft is 'custom' with passes=None and
    # /api/plan dies on int(None) (the Estimate panel then spun forever: 2026-11-19).
    per = (run.get("per_level") or {}).get("requests") or 50
    reps = run.get("repeats") or 1
    depth = next((k for k, d in C.DEPTHS.items() if d[2] == per and d[3] == reps), "custom")
    conc = sorted({int(c) for c in run.get("concurrency") or [1]})
    batches = run.get("batch_sizes") or [1]
    request_mode = "batch" if any(int(b) > 1 for b in batches) else ("single" if conc == [1] else "concurrent")
    sides: dict[str, dict] = {}
    for c in run.get("cells") or []:
        side = sides.setdefault(c["target"], {"hardware": c["target"], "precisions": [],
                                             "instance_type": (run.get("instance_types") or {}).get(c["target"])
                                             or hw.get(c["target"], {}).get("default_instance_type"),
                                             **({"alternates": (run.get("alternates") or {})[c["target"]]} if c["target"] in (run.get("alternates") or {}) else {})})
        if c["precision"] not in side["precisions"]:
            side["precisions"].append(c["precision"])
    gpu = next((v for k, v in sides.items() if hw.get(k, {}).get("accelerator") == "gpu"), None)
    cpus = [v for k, v in sides.items() if hw.get(k, {}).get("accelerator") != "gpu"]
    sla = run.get("sla") or {}
    purchase = run.get("purchase") or {}
    dspm = run.get("dspm") or {}
    trans = run.get("translation") or {}
    models = list(run.get("models") or [])
    base_name = (name or run.get("name") or "run").strip()
    w = {"use_case": suite, "prompt_set": run.get("prompt_set") if suite == "standard" else "short",
         "answer_tokens": int((run.get("sampling") or {}).get("max_new_tokens") or 128) if suite == "standard" else 128,
         "workloads": list(dspm.get("workloads") or ["W1", "W3"]), "test_items": int(dspm.get("test_items") or 300),
         "preset": dspm.get("sla_preset") or "inline_standard",
         "directions": list(trans.get("directions") or list(C.DIRECTIONS)),
         "text_size": run.get("prompt_set") if suite in ("translation", "embedding") and run.get("prompt_set") in C.TEXT_SIZES else "sentence",
         "translation_items": int(trans.get("test_items") or 100),
         "embedding_items": int((run.get("embedding") or {}).get("test_items") or 300),
         "intel_io": f"{(run.get('intel') or {}).get('input_tokens', 128)}/{(run.get('intel') or {}).get('output_tokens', 128)}",
         "depth": depth, "requests": per if depth == "custom" else None, "passes": reps if depth == "custom" else None,
         "models": models, "precision": "bf16", "gpu": gpu, "cpus": cpus,
         "parallel_targets": bool((run.get("execution") or {}).get("parallel_targets", True)),
         "request_mode": request_mode, "concurrency": conc, "batch_sizes": [int(b) for b in batches] if request_mode == "batch" else [8],
         "sla": {k: sla[k] for k in ("p99_ttft_ms", "p50_tpot_ms", "p99_e2e_ms") if sla.get(k) is not None} if suite in ("standard", "translation", "embedding", "intel") else {},
         "purchase": purchase.get("type", "spot"), "spot_fallback": bool(purchase.get("spot_fallback_on_demand", True)),
         "spot_wait_minutes": max(1, int(purchase.get("spot_capacity_window_s", 600)) // 60),
         "launch_attempts": int(purchase.get("max_launch_attempts", 30)),
         "name": f"{base_name} copy" if not base_name.endswith("copy") else base_name, "allow_unfair": False,
         "sized_for": f"{models[0]}|bf16" if models else None, "copied_from": source_run_id}
    return w


@app.get("/api/runs/{run_id}/wizard")
def run_as_wizard(run_id: str):
    """A new wizard draft prefilled from this run ('Copy to new run'); nothing is started."""
    import yaml

    if not RUN_ID.match(run_id):
        raise HTTPException(400, "not a run id")
    text = None

    try:
        text = aws.client("s3").get_object(Bucket=_outputs()["BucketName"], Key=f"runs/{run_id}/run.yaml")["Body"].read().decode()
    except Exception:  # noqa: BLE001 - older run without run.yaml: fall back to the resolved definition
        text = None
    run = (yaml.safe_load(text) or {}).get("run") if text else (_s3_json(f"runs/{run_id}/resolved.json") or {}).get("run")
    if not run:
        raise HTTPException(404, "this run's definition is no longer stored")
    meta = next((i for i in _run_items(run_id) if i["sk"] == "META"), {})
    return wizard_from_run(run, name=meta.get("name"), source_run_id=run_id)


@app.get("/api/runs/{run_id}/wizard/recommended")
def run_as_wizard_recommended(run_id: str):
    """The run's settings with every hardware option moved to the size its utilization verdict
    suggests (bench/utilization.suggest): the biggest change wins when one option served several
    number formats (too small beats keep, keep beats smaller), 'keep' leaves the size. Nothing
    starts; the banner on the New benchmark page lists what changed."""
    w = dict(run_as_wizard(run_id))
    s = _summary(run_id)
    analysis = compare.analyze([s], compare.default_sla([s]))
    cat = get_catalog()
    engines_of = {h["id"]: set(h["engines"]) for h in cat["hardware"]}
    order = {"bigger": 3, "stronger": 3, "smaller": 2, "none": 1, "keep": 1, "growth": 1}
    changes, notes = [], []

    def pick(side: dict | None, accelerator: str) -> dict | None:
        """The suggestion for one wizard side entry: among the sizing verdicts of its instance type and
        software, the highest-ranked change (a dearer size wins a tie)."""
        if not side:
            return None
        best = None
        for g in analysis["groups"]:
            for z in g["sizing"]:
                sug = z.get("suggestion")
                if z["side"] != accelerator or z.get("instance_type") != side.get("instance_type") or not sug:
                    continue
                if z.get("engine") and side.get("hardware") in engines_of and z["engine"] not in engines_of[side["hardware"]]:
                    continue
                rank = (order.get(sug["kind"], 0), sug.get("usd_per_hour") or 0)
                if best is None or rank > (order.get(best["kind"], 0), best.get("usd_per_hour") or 0):
                    best = {**sug, "series": z["series"], "verdict": z["verdict"]}
        return best

    moves = ("smaller", "bigger", "stronger")
    if w.get("gpu"):
        sug = pick(w["gpu"], "gpu")
        if sug and sug.get("instance_type") and sug["kind"] in moves:
            changes.append(f"GPU: {w['gpu']['instance_type']} -> {sug['instance_type']} ({sug['why'][:160]})")
            w["gpu"] = {**w["gpu"], "instance_type": sug["instance_type"], "alternates": []}
        elif sug:
            notes.append(f"GPU {w['gpu']['instance_type']} kept: {sug['why'][:120]}")
    new_cpus = []
    for c in w.get("cpus") or []:
        sug = pick(c, "cpu")
        if sug and sug.get("instance_type") and sug["kind"] in moves:
            changes.append(f"CPU ({C.hardware_label(c['hardware'])}): {c['instance_type']} -> {sug['instance_type']} ({sug['why'][:160]})")
            new_cpus.append({**c, "instance_type": sug["instance_type"], "alternates": []})
        else:
            if sug:
                notes.append(f"CPU ({C.hardware_label(c['hardware'])}) {c['instance_type']} kept: {sug['why'][:120]}")
            new_cpus.append(c)
    w["cpus"] = new_cpus
    w["name"] = ""
    w["copied_from"] = run_id
    w["sized_for"] = f"{w['models'][0]}|{w.get('precision', 'bf16')}" if w.get("models") else None
    w["recommendation"] = (("Sizes set from this run's utilization verdicts. " + " ".join(changes)) if changes else
                           "Every hardware option of this run was already the right size for its load; the sizes are unchanged.")
    if notes:
        w["recommendation"] += " " + " ".join(notes)
    # Each side is sized for its own load, so the pair can leave the fair price band (a mostly idle CPU
    # shrinks to m8i.xlarge while the GPU keeps its smallest size). Say so here; the plan will refuse it
    # until the pairing is allowed explicitly or one side is changed.
    if changes and w.get("gpu") and new_cpus:
        unfair = [pairing.judge(c["instance_type"], w["gpu"]["instance_type"]) for c in new_cpus if c.get("instance_type")]
        unfair = list(dict.fromkeys(p["reason"].rstrip(".") for p in unfair if p.get("fair") is False))
        if unfair:
            w["recommendation"] += (" Note: the recommended sizes are not a fair pair (" + "; ".join(unfair) +
                                    "). The plan will ask you to confirm, or pick a GPU size closer in price.")
    w["changes"] = changes
    return w


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str):
    if not RUN_ID.match(run_id):
        raise HTTPException(400, "not a run id")
    arn = _outputs()["RunStateMachineArn"].replace(":stateMachine:", ":execution:")
    try:
        aws.client("stepfunctions").stop_execution(executionArn=f"{arn}:{run_id}", cause="stopped from the UI")
    except Exception as e:  # noqa: BLE001 - already ended: still make sure nothing is left running
        _CACHE["stop_note"] = str(e)[:120]
    terminated = terminate_run_instances(run_id)
    _table().update_item(Key={"pk": f"RUN#{run_id}", "sk": "META"}, UpdateExpression="SET #s = :s, #u = :u",
                         ExpressionAttributeNames={"#s": "status", "#u": "updated_at"},
                         ExpressionAttributeValues={":s": "stopped", ":u": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return {"stopped": run_id, "terminated_instances": terminated}


def _summary(run_id: str) -> dict:
    key = f"summary:{run_id}"
    if key in _CACHE:
        return _CACHE[key]

    try:
        body = aws.client("s3").get_object(Bucket=_outputs()["BucketName"],
                                                                 Key=f"runs/{run_id}/report/summary.json")["Body"].read()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(404, f"no report for {run_id}: {str(e)[:120]}") from e
    s = json.loads(body)
    _backfill_util(run_id, s)
    _CACHE[key] = s
    return s


def _run_metrics(run_id: str) -> list[dict]:
    """Every measurement's metrics file (runs/<id>/metrics/<cell>/<level>.json), cached per run."""
    key = f"metrics:{run_id}"
    if key in _CACHE:
        return _CACHE[key]
    s3, bucket, out, skipped = aws.client("s3"), _outputs()["BucketName"], [], []
    kw = {"Bucket": bucket, "Prefix": f"runs/{run_id}/metrics/"}
    while True:
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents") or []:
            if o["Key"].endswith(".json"):
                try:
                    out.append(json.loads(s3.get_object(Bucket=bucket, Key=o["Key"])["Body"].read()))
                except Exception as e:  # noqa: BLE001 - one unreadable file loses one measurement's utilization, not the page
                    skipped.append(f"{o['Key'].rsplit('/', 1)[-1]}: {type(e).__name__}")
        if not r.get("IsTruncated"):
            break
        kw["ContinuationToken"] = r["NextContinuationToken"]
    if skipped:
        _CACHE[f"metrics_skipped:{run_id}"] = skipped
    _CACHE[key] = out
    return out


def _backfill_util(run_id: str, summary: dict) -> None:
    """Reports written before utilization was carried per row get it from the metrics files: the
    resources of every repeat of a pooled row, merged (bench/utilization.merge_stats)."""
    rows = summary.get("rows") or []
    if not rows or all("util" in r for r in rows):
        return
    try:
        by: dict[tuple[str, str], list[dict]] = {}
        for m in _run_metrics(run_id):
            if m.get("accuracy") or not m.get("level_key"):
                continue
            res = m.get("resources") or {}
            if "_source" not in res and m.get("window_wall_ns") and m.get("target_id"):
                # Recorded by the per-target rollup, which mixed in any other host under the same hardware
                # option at the time: re-read this level's window from CloudWatch for the run's own hosts.
                res = _level_util_from_cloudwatch(run_id, m) or res
            by.setdefault((m["cell_key"], report.group_key(m["level_key"])), []).append(utilization.legacy(res))
        for r in rows:
            if "util" not in r:
                parts = by.get((r.get("cell_key"), r.get("level_group")), [])
                r["util"] = utilization.merge_stats(parts) or None
    except Exception as e:  # noqa: BLE001 - the report still renders without utilization
        summary.setdefault("warnings", []).append(f"Utilization could not be read for this run: {type(e).__name__}")
        for r in rows:
            r.setdefault("util", None)


def _level_util_from_cloudwatch(run_id: str, m: dict) -> dict | None:
    """One legacy measurement's utilization from CloudWatch, scoped to the run's hosts: 1-minute
    points while CloudWatch keeps them (15 days), 5-minute after that; None when the hosts are not
    known or nothing came back. Windows under 90 s are marked rough."""
    hosts = _run_hosts(run_id, m["target_id"])
    if not hosts:
        return None
    key = f"lvlutil:{run_id}:{m['level_key']}"
    if key in _CACHE:
        return _CACHE[key]
    start, end = m["window_wall_ns"][0] / 1e9, m["window_wall_ns"][1] / 1e9
    age_days = (time.time() - end) / 86400
    period = 60 if age_days < 15 else 300
    out = None
    try:
        resp = aws.client("cloudwatch").get_metric_data(
            MetricDataQueries=utilization.queries(m["target_id"], period, hosts=hosts), ScanBy="TimestampAscending",
            StartTime=datetime.fromtimestamp(start - start % period, UTC), EndTime=datetime.fromtimestamp(end + period, UTC))
        out = utilization.summarize(utilization.merge(resp.get("MetricDataResults", []))) or None
        if out:
            out["_source"] = f"cloudwatch-{period}s-host"
            out["_window_s"] = round(end - start, 1)
            if end - start < 90:
                out["_rough"] = True
    except Exception:  # noqa: BLE001 - fall back to the stored (possibly diluted) figures
        out = None
    _CACHE[key] = out
    return out


def _run_hosts(run_id: str, target_id: str) -> list[str]:
    """Hostnames (the CloudWatch `host` dimension) of the instances this run launched for one
    hardware option: instance ids from runs/<id>/targets/<tid>.json, hostnames from the facts each
    host uploaded at boot (hosts/<tid>/<iid>-<boot>.json, `uname`). Empty when either is missing."""
    key = f"hosts:{run_id}:{target_id}"
    if key in _CACHE:
        return _CACHE[key]
    s3, bucket, out = aws.client("s3"), _outputs()["BucketName"], []
    try:
        events = json.loads(s3.get_object(Bucket=bucket, Key=f"runs/{run_id}/targets/{target_id}.json")["Body"].read())
        iids = sorted({e["instance_id"] for e in events if isinstance(e, dict) and e.get("instance_id")})
        for iid in iids:
            keys = sorted(o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=f"hosts/{target_id}/{iid}-").get("Contents") or [])
            if not keys:
                continue
            facts = json.loads(s3.get_object(Bucket=bucket, Key=keys[-1])["Body"].read())
            parts = str(facts.get("uname") or "").split()
            if len(parts) > 1 and parts[1].startswith("ip-"):
                out.append(parts[1])
    except Exception:  # noqa: BLE001 - no record: fall back to every host that served this hardware option
        out = []
    _CACHE[key] = out
    return out


@app.get("/api/runs/{run_id}/utilization")
def get_utilization(run_id: str):
    """CPU, memory and GPU utilization over the whole run, per hardware option, from CloudWatch
    (1-minute points for 15 days, 5-minute ones for 63 days), with every measurement's window so
    the chart can show which load setting was running when."""
    if not RUN_ID.match(run_id):
        raise HTTPException(400, "not a run id")
    key = f"util:{run_id}"
    if _CACHE.get(key, (0, None))[0] > time.time() - 600:
        return _CACHE[key][1]
    s = _summary(run_id)
    mets = [m for m in _run_metrics(run_id) if m.get("window_wall_ns")]
    if not mets:
        out = {"run_id": run_id, "targets": [], "reason": "no measurement windows recorded for this run"}
        _CACHE[key] = (time.time(), out)
        return out
    start = min(m["window_wall_ns"][0] for m in mets) / 1e9 - 120
    end = max(m["window_wall_ns"][1] for m in mets) / 1e9 + 120
    age_days = (time.time() - end) / 86400
    period = 60 if age_days < 15 else 300 if age_days < 63 else 3600
    cat = pairing.load_catalog()["instances"]
    itypes = (s.get("run") or {}).get("instance_types") or {}
    cw = aws.client("cloudwatch")
    targets = []
    for tid in sorted({m["target_id"] for m in mets if m.get("target_id")}):
        hosts = _run_hosts(run_id, tid)                 # the run's own instances, by hostname (empty: not known)
        try:
            resp = cw.get_metric_data(MetricDataQueries=utilization.queries(tid, period, hosts=hosts or None), ScanBy="TimestampAscending",
                                      StartTime=datetime.fromtimestamp(start, UTC), EndTime=datetime.fromtimestamp(end, UTC))
            series = utilization.merge(resp.get("MetricDataResults", []))
            error = None
        except Exception as e:  # noqa: BLE001 - say so per hardware option; the page keeps the per-measurement numbers
            series, error = {}, (aws.expired_sentence() if aws.is_expired(e) else f"{type(e).__name__}: {str(e)[:120]}")
        itype = itypes.get(tid)
        spec = cat.get(itype) if itype else None
        windows = [{"level": report.group_key(m["level_key"]), "name": m.get("name"), "conc": m.get("conc"), "batch": m.get("batch") or 1,
                    "start": m["window_wall_ns"][0] / 1e9, "end": m["window_wall_ns"][1] / 1e9}
                   for m in mets if m.get("target_id") == tid]
        targets.append({"target": tid, "instance_type": itype, "accelerator": (spec or {}).get("accelerator") or compare._accelerator(tid),
                        "spec": spec, "series": utilization.compact(series, limit=600), "hosts": max([v["hosts"] for v in series.values()] or [0]),
                        "scoped_to_run": bool(hosts), "instances": len(hosts),
                        "gpu_mem_total_mib": max(series["gpu_mem_total"]["v"]) if series.get("gpu_mem_total", {}).get("v") else None,
                        "windows": sorted(windows, key=lambda w: w["start"]), "error": error})
    out = {"run_id": run_id, "start": start, "end": end, "period_s": period, "targets": targets,
           "retention_note": ("1-minute averages; CloudWatch keeps them for 15 days after the run." if period == 60 else
                              "5-minute averages: CloudWatch has aged out the 1-minute points (kept 15 days)." if period == 300 else
                              "Hourly averages: the finer points have aged out of CloudWatch.")}
    _CACHE[key] = (time.time(), out)
    return out


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

    try:
        from webui.api import settings

        return settings.reasoning_model()
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
    # Keyed on the findings themselves (not only the selection), so a re-judged analysis (new targets, utilization
    # read anew, a redeployed rule) never serves the summary of an earlier analysis.
    findings = hashlib.sha256(json.dumps(exec_summary._compact(out["analysis"], out["comparability"], out["sla"], out["runs"]),
                                         sort_keys=True, default=str).encode()).hexdigest()[:16]
    key = "exec:" + json.dumps({"r": sorted(c.run_ids), "s": out["sla"], "l": c.limit, "m": c.models, "f": findings}, sort_keys=True, default=str)
    if key in _CACHE and not c.force:
        return _CACHE[key]
    try:
        res = exec_summary.write(out["analysis"], out["comparability"], out["sla"], out["runs"])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Bedrock call failed: {str(e)[:300]}") from e
    if not res.get("truncated"):            # a cut summary is shown once, never served again from the cache
        _CACHE[key] = res
    return res


@app.get("/api/runs/{run_id}/report.md")
def get_report_md(run_id: str):
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(report.render(_summary(run_id), with_charts=False)["summary.md"].decode())


# ----------------------------------------------------------------------------- setup

class HfTokenIn(BaseModel):
    token: str


class ReasoningModelIn(BaseModel):
    call_id: str


class AddModelIn(BaseModel):
    repo: str


def _who(request: Request) -> str | None:
    u = getattr(request.state, "user", None) if hasattr(request, "state") else None
    return (u or {}).get("username") if isinstance(u, dict) else None


@app.get("/api/setup")
def get_setup():
    """Everything the Setup page shows. Reads only; the token is reported as set/unset and when."""
    from webui.api import onboarding, settings

    cat = catalog_store.load_catalog(fresh=True)
    _CACHE.pop("proven_error", None)
    proven = _proven_engines()
    models = []
    for m in cat.models:
        engines = sorted({e for p in m.precisions.values() if p.enabled for e in p.engines})
        models.append({"id": m.id, "name": C.model_name(m.id), "repo": m.hf_repo, "revision": m.revision[:12], "type": m.type,
                       "kind": C.MODEL_KIND.get(m.type, m.type), "params_b": m.nominal_size_b, "license": m.license,
                       "gated": m.gated, "suites": list(m.suites), "user_added": m.user_added, "enabled": m.enabled,
                       "engines": [{"id": e, "label": C.ENGINES.get(e, (e,))[0], "proven": e in proven.get(m.id, set())}
                                   for e in engines],
                       "notes": m.notes})
    try:
        reasoning = {"models": settings.list_reasoning_models(), "current": settings.reasoning_model()}
    except Exception as e:  # noqa: BLE001
        reasoning = {"models": [], "current": None, "error": aws.expired_sentence() if aws.is_expired(e) else str(e)[:200]}
    return {"hf_token": settings.hf_token_status(), "models": models, "catalog_source": catalog_store.source(),
            "proven_error": _CACHE.get("proven_error"),      # badges default to "not yet run" when the run table is unreadable
            "reasoning": reasoning, "running_job": next((onboarding.status(j) for j, v in onboarding._jobs.items()
                                                         if v["status"] == "running"), None)}


def _proven_engines() -> dict[str, set[str]]:
    """Which serving software has served which model to a finished measurement here: from the
    run table's cells (status ready). Nothing else counts as proof."""
    out: dict[str, set[str]] = {}
    try:
        kw = {"FilterExpression": "begins_with(sk, :c) AND #s = :r", "ExpressionAttributeNames": {"#s": "status"},
              "ExpressionAttributeValues": {":c": "CELL#", ":r": "ready"}, "ProjectionExpression": "sk, engine"}
        while True:
            r = _table().scan(**kw)
            for it in r["Items"]:
                mid = it["sk"][5:].split("|")[0]
                if it.get("engine"):
                    out.setdefault(mid, set()).add(it["engine"])
            if "LastEvaluatedKey" not in r:
                break
            kw["ExclusiveStartKey"] = r["LastEvaluatedKey"]
    except Exception as e:  # noqa: BLE001 - the page still renders, without badges
        _CACHE["proven_error"] = aws.expired_sentence() if aws.is_expired(e) else f"{type(e).__name__}: {str(e)[:120]}"
    return out


@app.post("/api/setup/hf-token")
def post_hf_token(body: HfTokenIn, request: Request):
    from webui.api import settings

    try:
        return settings.set_hf_token(body.token, _who(request))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/setup/reasoning-model")
def post_reasoning_model(body: ReasoningModelIn, request: Request):
    from webui.api import settings

    try:
        return settings.set_reasoning_model(body.call_id, _who(request))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/setup/models")
def post_add_model(body: AddModelIn, request: Request):
    from webui.api import onboarding

    try:
        job_id = onboarding.start(body.repo, _who(request))
    except onboarding.OnboardingError as e:
        raise HTTPException(409, str(e)) from e
    return {"job_id": job_id}


@app.get("/api/setup/models/jobs/{job_id}")
def get_add_model_job(job_id: str):
    from webui.api import onboarding

    s = onboarding.status(job_id)
    if not s:
        raise HTTPException(404, "unknown job")
    if s["status"] == "done":
        _CACHE.pop("catalog", None)
    return s


@app.delete("/api/setup/models/{model_id}")
def delete_model(model_id: str, request: Request):
    from webui.api import onboarding

    running = [i for i in _running_runs_using(model_id)]
    if running:
        raise HTTPException(409, f"{C.model_name(model_id)} is used by a run that is still going ({running[0]}); "
                                 "stop or finish it first.")
    try:
        out = onboarding.remove(model_id, _who(request))
    except onboarding.OnboardingError as e:
        raise HTTPException(404 if "no such" in str(e) else 409, str(e)) from e
    _CACHE.pop("catalog", None)
    return out


def _running_runs_using(model_id: str) -> list[str]:
    out = []
    try:
        for r in list_runs():
            if r.get("status") in ("running", "planned") and model_id in (r.get("models") or []):
                out.append(r["run_id"])
    except Exception as e:  # noqa: BLE001 - an unreadable run table must not block removal; the page shows the reason
        _CACHE["runs_error"] = str(e)[:120]
    return out


# ----------------------------------------------------------------------------- technical details

TECH_FILES = {
    "servers/thin/Dockerfile.torch": "Thin PyTorch server image (CPU and CUDA builds differ only in BASE_IMAGE)",
    "servers/thin/Dockerfile.openvino": "Thin OpenVINO server image",
    "servers/export/Dockerfile": "Export image: Hugging Face checkpoint -> OpenVINO IR",
    "webui/Dockerfile": "Web application image (ECS Fargate)",
    "host/stage_model.sh": "Stage the pinned model snapshot on a host (S3 cache, else Hugging Face)",
    "host/serve.sh": "Run the engine container for one cell, wait for health, make the first request",
    "servers/export/export.py": "Export driver: OVMS export_model.py for chat models, optimum-cli for the rest; content-hashed IR cache",
    "servers/export/optimum_export.py": "optimum-cli with upstream-lifted Transformers version caps",
}


@app.get("/api/technical")
def technical():
    """What the Technical details page shows: hardware options with their engine images, the images
    this project builds, the Dockerfiles and host scripts verbatim, and how each number format is
    prepared for its engine. Read from the deployed tree, so it is what this build runs."""
    import yaml

    from bench import cellspec

    key = "technical"
    if key in _CACHE:
        return _CACHE[key]
    targets = load_targets(ROOT / "config/targets.yaml").resolved()
    images = yaml.safe_load((ROOT / "config/images.yaml").read_text())["images"]
    hardware = []
    for tid, t in targets.items():
        if not t.enabled:
            continue
        hardware.append({"id": tid, "label": t.label, "accelerator": t.accelerator, "instance_type": t.instance_type,
                         "alternates": list(t.alternates), "cpu_features": list(t.cpu_features),
                         "engines": [{"engine": name, "label": C.ENGINES.get(name, (name,))[0], "help": C.ENGINES.get(name, ("", ""))[1],
                                      "image": e.image, "built": e.image.startswith("built:"),
                                      "dockerfile": images.get(e.image[6:], {}).get("dockerfile") if e.image.startswith("built:") else None,
                                      "port": e.port, "env": dict(e.env), "args": dict(e.args)}
                                     for name, e in t.engines.items()]})
    files = {}
    for rel, title in TECH_FILES.items():
        path = ROOT / rel
        files[rel] = {"title": title, "text": path.read_text() if path.exists() else None, "present": path.exists()}
    ov = {k: v for k, v in yaml.safe_load((ROOT / "config/models.yaml").read_text()).items() if k.startswith("x-ov-")}
    # Models that need serve flags of their own on vLLM (config/models.yaml vllm_args), with the reason.
    model_flags = [{"model": m.id, "name": C.model_name(m.id), "hf_repo": m.hf_repo, "engine": "vllm",
                    "args": list(m.vllm_args), "notes": m.notes}
                   for m in catalog_store.load_catalog().models if m.enabled and m.vllm_args]
    out = {
        "hardware": hardware,
        "model_flags": model_flags,
        "images": [{"name": n, **spec} for n, spec in images.items()],
        "files": files,
        "thin_tasks": dict(cellspec.THIN_TASK),
        "export_presets": ov,
        "precisions": {k: {"label": v[0], "help": v[1]} for k, v in C.PRECISIONS.items()},
        "diagram": {"svg": "/api/technical/architecture.svg", "drawio": "/api/technical/architecture.drawio",
                    "present": (ROOT / "docs/architecture.svg").exists()},
        "harness_commit": os.environ.get("HARNESS_COMMIT"),
    }
    _CACHE[key] = out
    return out


@app.get("/api/technical/architecture.{ext}")
def technical_diagram(ext: str):
    if ext not in ("svg", "drawio", "png"):
        raise HTTPException(404, "no such file")
    path = ROOT / "docs" / f"architecture.{ext}"
    if not path.exists():
        raise HTTPException(404, "the diagram is not part of this build")
    media = {"svg": "image/svg+xml", "drawio": "application/xml", "png": "image/png"}[ext]
    return FileResponse(path, media_type=media, headers={"Cache-Control": "public, max-age=3600"},
                        filename=f"architecture.{ext}" if ext == "drawio" else None)


# ----------------------------------------------------------------------------- static app

DIST = ROOT / "webui/dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    _DIST_ROOT = DIST.resolve()

    @app.get("/{path:path}")
    def spa(path: str):
        """The app shell. A real file under dist/ is served as is; every other path, including
        anything that resolves outside dist/ (".." segments, absolute paths), gets index.html so
        the client-side router can answer. Nothing outside dist/ is ever read."""
        f = (_DIST_ROOT / path).resolve() if path else _DIST_ROOT
        if f.is_relative_to(_DIST_ROOT) and f.is_file():
            return FileResponse(f)
        return FileResponse(_DIST_ROOT / "index.html")


def bind_address() -> tuple[str, int]:
    """Where to listen. Default: loopback, port 8601 (the local app). AMXBENCH_BIND=0.0.0.0 is
    for the container, where the platform publishes the port and every request must carry a
    Cognito token; without a user pool configured a non-loopback bind is refused."""
    host = os.environ.get("AMXBENCH_BIND", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get("PORT", "8601"))
    if host not in ("127.0.0.1", "localhost", "::1") and not auth.enabled():
        raise SystemExit(f"refusing to bind {host}: no Cognito user pool is configured (AMXBENCH_COGNITO_USER_POOL_ID), "
                         "so the API would be reachable without authentication. Bind 127.0.0.1 or configure the pool.")
    return host, port


if __name__ == "__main__":
    import uvicorn

    _host, _port = bind_address()
    uvicorn.run(app, host=_host, port=_port, log_level="info", proxy_headers=_host != "127.0.0.1")
