"""Level plan per cell (design 9.1, 11, 24.5-24.6): which request pool, which payload,
which concurrency levels, how many requests, how many repeats.

A level is the unit the level state machine runs: one cell, one concurrency, one repeat,
and for classifiers one batch size, for DSPM/DLP one workload and length bucket. DSPM
cells also get one accuracy level per workload (every test item exactly once at C=16).
"""

from __future__ import annotations

from bench.cellspec import cell_key
from bench.config import ModelSpec, ResolvedCell, RunConfig

GENERATIVE = ("decoder", "encoder-decoder")
CLASSIFIER_SETS = {"short": "short", "medium": "cls512", "long": "cls512", "xlong": "cls512"}

# DSPM/DLP workloads (design 24.1): models and concurrency levels per workload.
WORKLOAD_SLOTS = {"W1": {"M1", "M2", "M4", "M5", "M3", "D1"}, "W2": {"M1", "M2", "M4", "M5"},
                  "W3": {"M1", "M2", "M4", "M5", "M3", "D1", "D2"}, "W4": {"M1", "M2", "M4"},
                  "W5": {"M1", "M2", "M4"}}
WORKLOAD_LEVELS = {"W1": (8, 16, 32, 64), "W2": (16, 32), "W3": (1, 4, 8, 16), "W4": (8, 32), "W5": (4, 16)}
WORKLOAD_BUCKETS = {"W1": ("short", "long"), "W2": ("short",), "W3": (None,), "W4": (None,), "W5": (None,)}
SHORT_ONLY_SLOTS = {"M5", "M3"}       # 512-token models skip the ~2,048-token W1 chunks
ACCURACY_CONC = 16
DEV_THRESHOLD_SLOTS = {"M3"}
CATEGORY_WORKLOADS = ("W2", "W4")
LABELS = {"W1": ["SENSITIVE", "NOT_SENSITIVE"], "W3": ["BLOCK", "ALLOW"],
          "W2": ["GOV_ID", "PERSONAL", "FINANCIAL", "HEALTH", "CREDENTIALS", "CONFIDENTIAL", "NONE"],
          "W4": ["GOV_ID", "PERSONAL", "FINANCIAL", "HEALTH", "CREDENTIALS", "CONFIDENTIAL", "NONE"]}


def api_path(model: ModelSpec, engine: str) -> str:
    if model.type in GENERATIVE:
        return "/v3/completions" if engine == "ovms" else "/v1/completions"
    return "/detect" if model.type == "token-classifier" else "/classify"


def health_path(engine: str) -> str:
    return "/v2/health/ready" if engine == "ovms" else "/health"


def standard_pool(model: ModelSpec, run: RunConfig) -> str | None:
    """Prompt set name for the standard suite, or None when the model skips this set."""
    if model.type in GENERATIVE:
        if model.type == "encoder-decoder" and run.prompt_set in ("long", "xlong"):
            return None
        return run.prompt_set
    return CLASSIFIER_SETS.get(run.prompt_set, "short")


def dspm_pool(model: ModelSpec, workload: str, bucket: str | None) -> str:
    """Request pool built by tools/dspm/build_requests.py. The constrained variant sends the
    same prompts with a different payload, so pools do not depend on the variant."""
    name = workload if bucket is None else f"{workload}-{bucket}"
    return f"artifacts/dspm/v1/requests/{model.id}/{name}.jsonl"


def _gen_payload(model: ModelSpec, run: RunConfig) -> dict:
    s = run.sampling
    return {"model": model.id, "max_tokens": s.max_new_tokens, "temperature": s.temperature, "seed": s.seed,
            "ignore_eos": s.ignore_eos}


def dspm_payload(model: ModelSpec, workload: str, variant: str) -> dict:
    """Narrow-label generation (design 24.4): 4 tokens for binary labels, 8 for categories,
    stop at newline, greedy; the constrained variant adds vLLM's choice decoding."""
    p = {"model": model.id, "max_tokens": 8 if workload in CATEGORY_WORKLOADS else 4, "temperature": 0.0,
         "seed": 0, "ignore_eos": False, "stop": ["\n"]}
    if workload == "W5":
        p.update(max_tokens=128, stop=[])
    if variant == "constrained" and workload in LABELS:
        p["structured_outputs"] = {"choice": LABELS[workload]}
    return p


def levels_for_cell(run: RunConfig, cell: ResolvedCell, model: ModelSpec, *, run_id: str, bucket: str,
                    pool_sizes: dict[str, int]) -> list[dict]:
    """pool_sizes: S3 key -> number of lines (from the prompt and request manifests)."""
    ck = cell_key(cell.cell_id)
    base = {"run_id": run_id, "cell_key": ck, "api_path": api_path(model, cell.engine),
            "api_task": "generate" if model.type in GENERATIVE else
            ("detect" if model.type == "token-classifier" else "classify"),
            "health_path": health_path(cell.engine), "request_timeout_s": run.request_timeout_s,
            "max_waves": run.max_waves, "warmup_requests": run.warmup_requests}
    per = run.per_level
    out: list[dict] = []

    def add(name: str, pool_key: str, payload: dict, conc: int, rep: int, *, batch: int = 1, accuracy: bool = False,
            workload: str | None = None, extra: dict | None = None) -> None:
        if pool_key not in pool_sizes:
            raise ValueError(f"request pool {pool_key} is missing from the manifest")
        key = f"{ck}__{name}__c{conc}" + (f"__b{batch}" if batch > 1 else "") + ("" if accuracy else f"__r{rep}")
        size = pool_sizes[pool_key]
        if accuracy and run.dspm and run.dspm.test_items:
            size = min(size, run.dspm.test_items)    # request pools are shuffled, so a prefix is a fair sample
        lvl = {**base, "level_key": key, "name": name, "conc": conc, "repeat": rep, "batch": batch, "payload": payload,
               "pool": {"bucket": bucket, "key": pool_key, "size": size},
               "accuracy": accuracy, "keep_output": accuracy, "workload": workload, **(extra or {})}
        if not accuracy:
            if per.duration_s:
                lvl["duration_s"] = per.duration_s
            else:
                lvl["quota"] = per.requests
        out.append(lvl)

    if run.suite == "standard":
        name = standard_pool(model, run)
        if name is None:
            return []
        key = f"artifacts/prompts/{model.id}/{name}.jsonl"
        payload = _gen_payload(model, run) if model.type in GENERATIVE else {"model": model.id}
        if model.type in GENERATIVE or not model.classifier:
            batches = [1]
        else:
            batches = run.batch_sizes or model.classifier.batch_sizes
        for conc in run.concurrency:
            for b in batches:
                for rep in range(run.repeats):
                    add(name, key, payload, conc, rep, batch=b)
        return out

    # DSPM/DLP suite
    assert run.dspm is not None
    variant = cell.variant
    for wl in run.dspm.workloads:
        if model.dspm_profile not in WORKLOAD_SLOTS[wl]:
            continue
        if wl in LABELS and model.type == "token-classifier" and wl not in ("W1", "W3"):
            continue
        for bkt in WORKLOAD_BUCKETS[wl]:
            if bkt == "long" and (model.dspm_profile in SHORT_ONLY_SLOTS or model.max_input_tokens < 2560):
                continue
            key = dspm_pool(model, wl, bkt)
            payload = dspm_payload(model, wl, variant) if model.type in GENERATIVE else {"model": model.id}
            name = wl if bkt is None else f"{wl}-{bkt}"
            add(f"{name}-accuracy", key, payload, ACCURACY_CONC, 0, accuracy=True, workload=wl)
            if model.dspm_profile in DEV_THRESHOLD_SLOTS:
                # M3's decision threshold is chosen on dev items only (design 24.4, Requirement 23).
                dev_key = dspm_pool(model, f"dev-{wl}", bkt)
                add(f"dev-{name}-accuracy", dev_key, payload, ACCURACY_CONC, 0, accuracy=True, workload=wl,
                    extra={"split": "dev"})
            # The run's concurrency list applies to every scenario; WORKLOAD_LEVELS are the
            # suggested defaults per scenario, not a filter (a chosen level must never be dropped).
            for conc in run.concurrency:
                for rep in range(run.repeats):
                    add(name, key, payload, conc, rep, workload=wl)
    return out


def quality_level(run: RunConfig, cell: ResolvedCell, model: ModelSpec, *, run_id: str, bucket: str,
                  pool_sizes: dict[str, int]) -> dict | None:
    """The quality pass of design 15: quality-50 (generation) or quality-50-nli (M3)."""
    if not run.quality.enabled or run.suite != "standard":
        return None
    if model.type in GENERATIVE:
        name = run.quality.prompt_set
        payload = {"model": model.id, "max_tokens": run.quality.max_new_tokens, "temperature": 0.0, "seed": 0,
                   "ignore_eos": False}
    elif model.type == "encoder-classifier" and model.classifier and not model.classifier.multi_label:
        name = run.quality.prompt_set + "-nli"
        payload = {"model": model.id}
    else:
        return None
    key = f"artifacts/prompts/{model.id}/{name}.jsonl"
    if key not in pool_sizes:
        raise ValueError(f"quality pool {key} is missing from the manifest")
    return {"run_id": run_id, "cell_key": cell_key(cell.cell_id), "api_path": api_path(model, cell.engine),
            "api_task": "generate" if model.type in GENERATIVE else "classify", "payload": payload,
            "pool": {"bucket": bucket, "key": key, "size": pool_sizes[key]}, "conc": 4, "batch": 1,
            "request_timeout_s": run.request_timeout_s}
