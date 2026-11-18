"""Preflight (design 9.1 step 1, Requirement 1.4, 4.4, 5.8, 6.2, 10.2).

Input: {"run_id"}; reads runs/<id>/run.yaml (uploaded by the CLI or UI). Refuses the run
on any failed check, with every reason listed. On success writes resolved.json, one cell
spec and one level plan per cell, the run and cell items, and returns the target/cell
outline the run state machine maps over (levels stay in S3, out of the state payload).
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from bench import catalog_store, cellspec, cost, pairing, plan
from bench.amx import parse_cpu_flags
from bench.config import (
    RunConfig,
    config_hash,
    load_targets,
    resolve_cells,
    run_alternates,
    run_instance_types,
)
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


def _latest_host_facts(target_id: str, instance_type: str | None = None) -> dict | None:
    """The newest host facts recorded for this target, and for THIS instance type when given:
    instance sizes vary per run, so facts from a c8i must not answer for an m7i."""
    keys = common.list_keys(f"hosts/{target_id}/")
    for k in reversed(keys):
        facts = common.get_json(k)
        if instance_type is None or facts.get("instance_type") == instance_type:
            return facts
    return None


# There is no per-family feature table here any more. The one that existed said "8i has
# amx_fp16" because Granite Rapids does; the EC2 hosts do not expose it (config/targets.yaml has
# the measurement), and the table disagreed with the host facts, so the wizard accepted FP16 on a
# c8i.4xlarge and the pre-check refused it with advice to pick a c8i (2026-11-15). The target's
# cpu_features in targets.yaml is the single statement of what a CPU option's instances have;
# host facts, when a host of the run's type has booted, are the check against it.


DSPM_REQUESTS_MANIFEST = "artifacts/dspm/v1/requests/manifest.json"
DSPM_DATASET_MANIFEST = "datasets/dspm/v1/manifest.json"
TRANSLATION_DATASET_MANIFEST = "datasets/translation/v1/manifest.json"
PROMPTS_MANIFEST = "artifacts/prompts/manifest.json"


def _pool_sizes() -> dict[str, int]:
    sizes: dict[str, int] = {}
    for key in ("artifacts/prompts/manifest.json", DSPM_REQUESTS_MANIFEST):
        if common.exists(key):
            sizes.update(common.get_json(key).get("pool_sizes", {}))
    return sizes


def dspm_hashes(requests_manifest: dict | None, dataset_manifest_bytes: bytes | None) -> tuple[dict, list[str]]:
    """Dataset and policy hashes for the report, plus errors when the uploaded request pools
    were not built from the uploaded dataset (scoring would use the wrong labels)."""
    if requests_manifest is None or dataset_manifest_bytes is None:
        return {}, ["DSPM suite needs the dataset and request pools in S3; run `bench upload-artifacts`"]
    ds = json.loads(dataset_manifest_bytes)
    out = {"dataset_version": ds.get("version"), "dataset_sha256": ds.get("dataset_sha256"),
           "dataset_manifest_sha256": hashlib.sha256(dataset_manifest_bytes).hexdigest(),
           "span_label_map_sha256": ds.get("span_label_map_sha256"), "dataset_seed": ds.get("seed"),
           **(requests_manifest.get("policy") or {})}
    errors = []
    if requests_manifest.get("dataset_manifest_sha256") != out["dataset_manifest_sha256"]:
        errors.append("DSPM request pools were built from a different dataset than datasets/dspm/v1/manifest.json; "
                      "rerun tools.dspm.build_requests and `bench upload-artifacts`")
    return out, errors


def translation_hashes(prompts_manifest: dict | None, dataset_manifest_bytes: bytes | None) -> tuple[dict, list[str]]:
    """The translation corpus fingerprint for the report, plus an error when the uploaded request
    pools were rendered from a different corpus than the one the report would score against."""
    if dataset_manifest_bytes is None:
        return {}, ["Translation suite needs the parallel corpus in S3 (datasets/translation/v1); run `bench upload-artifacts`"]
    ds = json.loads(dataset_manifest_bytes)
    out = {"dataset_version": ds.get("version"), "dataset_sha256": ds.get("dataset_sha256"), "dataset_seed": ds.get("seed"),
           "items": ds.get("items")}
    rendered = ((prompts_manifest or {}).get("translation") or {}).get("dataset_sha256")
    errors = []
    if rendered and rendered != out["dataset_sha256"]:
        errors.append("translation request pools were rendered from a different corpus than datasets/translation/v1/manifest.json; "
                      "rerun tools.translation.build_requests and `bench upload-artifacts`")
    return out, errors


def check_run(run: RunConfig, cells, models, tmap, *, reserved_worker_concurrency: int | None,
              offerings: set[str] | None, host_flags: dict[str, set[str] | None]) -> list[str]:
    """Pure checks (unit-tested); AWS lookups happen in the handler."""
    errors = []
    max_c = max(run.concurrency)
    if run.suite == "dspm":
        max_c = max(max_c, plan.ACCURACY_CONC)
    if run.suite == "translation":
        max_c = max(max_c, plan.TRANSLATION_ACCURACY_CONC)
    if reserved_worker_concurrency is not None and reserved_worker_concurrency < max_c:
        errors.append(f"worker reserved concurrency {reserved_worker_concurrency} < highest level {max_c}")
    types = set(run_instance_types(run, cells, tmap).values())
    if offerings is not None:
        missing = sorted(types - offerings)
        if missing:
            errors.append(f"instance types not offered in {run.az_id}: {missing}")
    itypes = run_instance_types(run, cells, tmap)
    for c in cells:
        p = models.get(c.model).precisions[c.precision]
        # A CPU feature requirement (AMX-FP16) is about the CPU option's processor; a GPU cell
        # runs FP16 on the GPU and is never subject to it (2026-11-12: a RoBERTa FP16 cell on
        # g6.xlarge was refused with "host facts of t1-g6 lack amx_fp16").
        if p.requires_cpu_flag and tmap[c.target].accelerator == "cpu":
            flags = host_flags.get(c.target)
            if flags is None:
                flags = set(tmap[c.target].cpu_features)
            if p.requires_cpu_flag not in flags:
                errors.append(f"{c.cell_id}: {itypes.get(c.target)} has no {p.requires_cpu_flag} "
                              f"(the {c.precision} format needs it)")
    return errors


def handler(event: dict, context: Any = None) -> dict:
    run_id = event["run_id"]
    prefix = common.run_prefix(run_id)
    raw = common.get_bytes(f"{prefix}/run.yaml").decode()
    data = yaml.safe_load(raw)
    run = RunConfig.model_validate(data["run"] if "run" in data else data)
    models, targets = catalog_store.load_catalog(), load_targets(CONFIG_DIR / "targets.yaml")
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
    itypes = run_instance_types(run, cells, tmap)
    alts = {tid: run_alternates(run, tid, tmap) for tid in itypes}
    types = sorted(set(itypes.values()))
    offered = common.client("ec2").describe_instance_type_offerings(
        LocationType="availability-zone-id",
        Filters=[{"Name": "location", "Values": [run.az_id]}, {"Name": "instance-type", "Values": types}])
    offerings = {o["InstanceType"] for o in offered["InstanceTypeOfferings"]}
    # Host facts (FP16 flag) where a host of this run's instance type has booted before; else the
    # family rule in check_run decides.
    host_flags: dict[str, set[str] | None] = {}
    for tid in {c.target for c in cells}:
        facts = _latest_host_facts(tid, itypes.get(tid))
        host_flags[tid] = parse_cpu_flags(facts["lscpu"]) if facts else None
    errors += check_run(run, cells, models, tmap, reserved_worker_concurrency=reserved, offerings=offerings,
                        host_flags=host_flags)

    # Deployed targets: launch template and AMI from the target stacks. The instance itself is
    # launched per run by target_control (On-Demand or Spot, run.purchase).
    templates: dict[str, str] = {}
    amis: dict[str, str | None] = {}
    for tid in sorted({c.target for c in cells}):
        lt = _param(f"/amx-gpu-bench/targets/{tid}/launch-template")
        if not lt:
            errors.append(f"target {tid} is not deployed (no SSM parameter /amx-gpu-bench/targets/{tid}/launch-template)")
        else:
            templates[tid] = lt
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

    # Prices (no fallback: a missing price stops the run). Alternates are priced too, so a
    # capacity fallback (g6 -> g5) still gets cost columns.
    priced = sorted(set(types) | {a for al in alts.values() for a in al})
    try:
        prices = cost.snapshot(common.client_in("pricing", "us-east-1"), priced, run.region,
                               time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    except cost.PriceUnavailable as e:
        errors.append(f"price unavailable: {e}")
        prices = {}

    hashes: dict = {}
    if run.suite == "dspm":
        req = common.get_json(DSPM_REQUESTS_MANIFEST) if common.exists(DSPM_REQUESTS_MANIFEST) else None
        dsb = common.get_bytes(DSPM_DATASET_MANIFEST) if common.exists(DSPM_DATASET_MANIFEST) else None
        hashes, errs = dspm_hashes(req, dsb)
        errors += errs
    if run.suite == "translation":
        pm = common.get_json(PROMPTS_MANIFEST) if common.exists(PROMPTS_MANIFEST) else None
        tb = common.get_bytes(TRANSLATION_DATASET_MANIFEST) if common.exists(TRANSLATION_DATASET_MANIFEST) else None
        hashes, errs = translation_hashes(pm, tb)
        errors += errs

    pools = _pool_sizes()
    specs, outline = {}, {}
    for c in cells:
        m, t = models.get(c.model), tmap[c.target]
        try:
            spec = cellspec.build(c, m, t, run, run_id=run_id, bucket=common.bucket(), hf_token_secret=secret,
                                  ecr_repo_uri=ecr_uri or None, built_tags=built, instance_type=itypes[c.target])
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
        membw = cpu and itypes[c.target] not in membw_done
        if membw:
            membw_done.add(itypes[c.target])
        outline.setdefault(c.target, {"target_id": c.target, "cpu": cpu, "instance_type": itypes[c.target],
                                      "alternates": alts[c.target], "cells": []})["cells"].append({
            "cell_id": c.cell_id, "cell_key": spec["cell_key"], "engine": c.engine, "cpu": cpu,
            "spec_uri": f"s3://{common.bucket()}/{prefix}/cells/{spec['cell_key']}.json",
            "amx": cpu, "membw": membw, "quality": qual is not None, "n_levels": len(levels)})
        common.set_status(run_id, f"CELL#{c.cell_id}", "planned", engine=c.engine, n_levels=len(levels))
    resolved = {"run_id": run_id, "run": run.model_dump(mode="json"), "run_hash": config_hash(run),
                "models_hash": config_hash(models), "targets_hash": config_hash(targets),
                "cells": [c.model_dump() for c in cells], "launch_templates": templates, "amis": amis,
                "purchase": run.purchase.model_dump(), "prices": prices, "built_images": built,
                "ecr_repository_uri": ecr_uri, "harness_commit": common.env("HARNESS_COMMIT", "unknown"),
                "smoke": run.is_smoke, "dspm_hashes": hashes, "instance_types_requested": itypes,
                "alternates": alts, "pairing": pairing.judge_run(itypes, {tid: tmap[tid].accelerator for tid in itypes}),
                "created_at": common.now_iso()}
    common.put_json(f"{prefix}/resolved.json", resolved)
    common.set_status(run_id, "META", "running", smoke=run.is_smoke, n_cells=len(cells),
                      purchase=run.purchase.type)
    for tid in templates:
        common.touch_target(tid, run_id)
    return {"run_id": run_id, "targets": list(outline.values()), "purchase": run.purchase.model_dump(),
            "stop_targets_after_run": run.stop_targets_after_run,
            # Step Functions Map MaxConcurrency: 0 = unlimited (all hardware options at once), 1 = one at a time.
            "max_target_concurrency": 0 if run.execution.parallel_targets else 1}
