"""Preflight (design 9.1 step 1, Requirement 1.4, 4.4, 5.8, 6.2, 10.2).

Input: {"run_id"}; reads runs/<id>/run.yaml (uploaded by the CLI or UI). Refuses the run
on any failed check, with every reason listed. On success writes resolved.json, one cell
spec and one level plan per cell, the run and cell items, and returns the target/cell
outline the run state machine maps over (levels stay in S3, out of the state payload).
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from bench import cellspec, cost, plan
from bench.amx import parse_cpu_flags
from bench.config import RunConfig, config_hash, load_models, load_targets, resolve_cells
from lambdas import common

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
WORKER_HEADROOM = 8


class PreflightError(RuntimeError):
    pass


def _param(name: str) -> str | None:
    try:
        return common.client("ssm").get_parameter(Name=name)["Parameter"]["Value"]
    except common.client("ssm").exceptions.ParameterNotFound:
        return None


def _hf_access(repo: str, revision: str, token: str) -> int:
    req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}/revision/{revision}",
                                 headers={"Authorization": f"Bearer {token}"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:  # noqa: S310 - fixed https host
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def _latest_host_facts(target_id: str) -> dict | None:
    keys = common.list_keys(f"hosts/{target_id}/")
    return common.get_json(keys[-1]) if keys else None


def _pool_sizes() -> dict[str, int]:
    sizes: dict[str, int] = {}
    for key in ("artifacts/prompts/manifest.json", "artifacts/dspm/v1/requests/manifest.json"):
        if common.exists(key):
            sizes.update(common.get_json(key).get("pool_sizes", {}))
    return sizes


def check_run(run: RunConfig, cells, models, tmap, *, reserved_worker_concurrency: int | None,
              offerings: set[str] | None, host_flags: dict[str, set[str] | None]) -> list[str]:
    """Pure checks (unit-tested); AWS lookups happen in the handler."""
    errors = []
    max_c = max(run.concurrency)
    if run.suite == "dspm":
        max_c = max(max_c, plan.ACCURACY_CONC)
    if reserved_worker_concurrency is not None and reserved_worker_concurrency < max_c:
        errors.append(f"worker reserved concurrency {reserved_worker_concurrency} < highest level {max_c}")
    types = {tmap[c.target].instance_type for c in cells}
    if offerings is not None:
        missing = sorted(types - offerings)
        if missing:
            errors.append(f"instance types not offered in {run.az_id}: {missing}")
    for c in cells:
        p = models.get(c.model).precisions[c.precision]
        if p.requires_cpu_flag:
            flags = host_flags.get(c.target)
            if flags is not None and p.requires_cpu_flag not in flags:
                errors.append(f"{c.cell_id}: host facts of {c.target} lack {p.requires_cpu_flag}")
    return errors


def handler(event: dict, context: Any = None) -> dict:
    run_id = event["run_id"]
    prefix = common.run_prefix(run_id)
    raw = common.get_bytes(f"{prefix}/run.yaml").decode()
    data = yaml.safe_load(raw)
    run = RunConfig.model_validate(data["run"] if "run" in data else data)
    models, targets = load_models(CONFIG_DIR / "models.yaml"), load_targets(CONFIG_DIR / "targets.yaml")
    tmap = targets.resolved()
    cells = resolve_cells(run, models, targets)
    errors: list[str] = []

    # Lambda concurrency (worker reserved concurrency must cover the highest level).
    worker = common.env("WORKER_FUNCTION_NAME", "")
    reserved = None
    if worker:
        reserved = common.client("lambda").get_function_concurrency(FunctionName=worker).get(
            "ReservedConcurrentExecutions")
    # Instance offerings in the AZ.
    types = sorted({tmap[c.target].instance_type for c in cells})
    offered = common.client("ec2").describe_instance_type_offerings(
        LocationType="availability-zone-id",
        Filters=[{"Name": "location", "Values": [run.az_id]}, {"Name": "instance-type", "Values": types}])
    offerings = {o["InstanceType"] for o in offered["InstanceTypeOfferings"]}
    # Host facts (FP16 flag) where the target has booted before.
    host_flags: dict[str, set[str] | None] = {}
    for tid in {c.target for c in cells}:
        facts = _latest_host_facts(tid)
        host_flags[tid] = parse_cpu_flags(facts["lscpu"]) if facts else None
    errors += check_run(run, cells, models, tmap, reserved_worker_concurrency=reserved, offerings=offerings,
                        host_flags=host_flags)

    # Deployed targets: instance id and AMI from the target stacks.
    instances: dict[str, str] = {}
    amis: dict[str, str | None] = {}
    for tid in sorted({c.target for c in cells}):
        iid = _param(f"/amx-gpu-bench/targets/{tid}")
        if not iid:
            errors.append(f"target {tid} is not deployed (no SSM parameter /amx-gpu-bench/targets/{tid})")
        else:
            instances[tid] = iid
        amis[tid] = _param(f"/amx-gpu-bench/targets/{tid}/ami")

    # Built images.
    built = json.loads(_param("/amx-gpu-bench/images") or "{}")
    ecr_uri = common.env("ECR_REPOSITORY_URI", "")

    # Gated models: the stored token must reach the pinned revision.
    secret = common.env("HF_TOKEN_SECRET_ARN")
    gated = [models.get(m) for m in run.models if models.get(m).gated]
    if gated:
        token = common.client("secretsmanager").get_secret_value(SecretId=secret)["SecretString"]
        for m in gated:
            code = _hf_access(m.hf_repo, m.revision, token)
            if code != 200:
                errors.append(f"Hugging Face access to gated {m.hf_repo}@{m.revision[:8]} returned HTTP {code}")
        del token

    # Prices (no fallback: a missing price stops the run).
    try:
        prices = cost.snapshot(common.client_in("pricing", "us-east-1"), types, run.region,
                               time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    except cost.PriceUnavailable as e:
        errors.append(f"price unavailable: {e}")
        prices = {}

    pools = _pool_sizes()
    specs, outline = {}, {}
    for c in cells:
        m, t = models.get(c.model), tmap[c.target]
        try:
            spec = cellspec.build(c, m, t, run, run_id=run_id, bucket=common.bucket(), hf_token_secret=secret,
                                  ecr_repo_uri=ecr_uri or None, built_tags=built)
            levels = plan.levels_for_cell(run, c, m, run_id=run_id, bucket=common.bucket(), pool_sizes=pools)
            qual = None if c.quality_only else plan.quality_level(run, c, m, run_id=run_id, bucket=common.bucket(),
                                                                  pool_sizes=pools)
            if c.quality_only:      # FP32 anchor cells run the quality set only (design 15)
                qual = plan.quality_level(run, c, m, run_id=run_id, bucket=common.bucket(), pool_sizes=pools)
                levels = []
        except ValueError as e:
            errors.append(f"{c.cell_id}: {e}")
            continue
        specs[c.cell_id] = (spec, levels, qual)
    if errors:
        common.put_json(f"{prefix}/preflight_errors.json", {"errors": errors})
        common.set_status(run_id, "META", "refused", errors=errors[:50])
        raise PreflightError("; ".join(errors[:10]))

    membw_done: set[str] = set()
    for c in cells:
        spec, levels, qual = specs[c.cell_id]
        t = tmap[c.target]
        common.put_json(f"{prefix}/cells/{spec['cell_key']}.json", spec)
        common.put_json(f"{prefix}/plan/{spec['cell_key']}.json", {"levels": levels, "quality": qual})
        cpu = t.accelerator == "cpu"
        membw = cpu and t.instance_type not in membw_done
        if membw:
            membw_done.add(t.instance_type)
        outline.setdefault(c.target, {"target_id": c.target, "instance_id": instances[c.target], "cpu": cpu,
                                      "cells": []})["cells"].append({
            "cell_id": c.cell_id, "cell_key": spec["cell_key"], "engine": c.engine, "cpu": cpu,
            "spec_uri": f"s3://{common.bucket()}/{prefix}/cells/{spec['cell_key']}.json",
            "amx": cpu, "membw": membw, "quality": qual is not None, "n_levels": len(levels)})
        common.set_status(run_id, f"CELL#{c.cell_id}", "planned", engine=c.engine, n_levels=len(levels))
    resolved = {"run_id": run_id, "run": run.model_dump(mode="json"), "run_hash": config_hash(run),
                "models_hash": config_hash(models), "targets_hash": config_hash(targets),
                "cells": [c.model_dump() for c in cells], "instances": instances, "amis": amis,
                "prices": prices, "built_images": built, "ecr_repository_uri": ecr_uri,
                "harness_commit": common.env("HARNESS_COMMIT", "unknown"), "smoke": run.is_smoke,
                "created_at": common.now_iso()}
    common.put_json(f"{prefix}/resolved.json", resolved)
    common.set_status(run_id, "META", "running", smoke=run.is_smoke, n_cells=len(cells))
    for tid in instances:
        common.touch_target(tid, run_id)
    return {"run_id": run_id, "targets": list(outline.values()),
            "stop_targets_after_run": run.stop_targets_after_run}
