"""Duration and cost estimate before a run starts (design section 20, Requirement 15.2).

Speeds are priors (assumptions, not measurements) until a summary from an earlier run of
the same model and target is passed in; the output says which source each cell used.
"""

from __future__ import annotations

import math

from bench.config import ModelCatalog, ResolvedCell, RunConfig, TargetSpec
from bench.plan import levels_for_cell

BARRIER_AND_MAP_S = 40          # per wave: barrier lead plus Map start
CELL_OVERHEAD_S = {"fetch_and_load": 6 * 60, "amx_probe": 3 * 60}
QUALITY_S = 4 * 60
NAT_USD_PER_HOUR = 0.045
LAMBDA_USD_PER_GB_S = 0.0000133334
WORKER_GB = 1769 / 1024
# Priors: prefill tok/s and decode tok/s at C=1, and how decode slows with concurrency.
PRIORS = {
    ("gpu", "decoder"): {"prefill": 8000.0, "decode": 80.0, "conc_knee": 32},
    ("cpu", "decoder"): {"prefill": 1000.0, "decode": 12.0, "conc_knee": 8},
    ("gpu", "encoder-decoder"): {"prefill": 6000.0, "decode": 60.0, "conc_knee": 16},
    ("cpu", "encoder-decoder"): {"prefill": 800.0, "decode": 10.0, "conc_knee": 8},
    ("gpu", "classifier"): {"prefill": 20000.0, "decode": 0.0, "conc_knee": 32},
    ("cpu", "classifier"): {"prefill": 3000.0, "decode": 0.0, "conc_knee": 8},
}


def _prior(target: TargetSpec, model_type: str) -> dict:
    kind = "classifier" if model_type in ("encoder-classifier", "token-classifier") else model_type
    return PRIORS[(target.accelerator, kind)]


def request_seconds(lvl: dict, model_type: str, target: TargetSpec, prompt_tokens: int,
                    measured: dict | None = None) -> tuple[float, str]:
    if measured and measured.get("e2e_p50"):
        return measured["e2e_p50"] / 1000.0, "measured"
    p = _prior(target, model_type)
    slow = 1 + lvl["conc"] / p["conc_knee"]
    prefill = prompt_tokens * lvl.get("batch", 1) / p["prefill"] * slow
    out = lvl["payload"].get("max_tokens", 0) if lvl["api_task"] == "generate" else 0
    decode = (out / p["decode"] * slow) if p["decode"] and out else 0.0
    return prefill + decode, "prior"


def estimate(run: RunConfig, cells: list[ResolvedCell], models: ModelCatalog, targets: dict[str, TargetSpec],
             prices: dict[str, float], pool_sizes: dict[str, int], *, prompt_tokens: dict[str, int] | None = None,
             measured: dict[tuple, dict] | None = None, instance_types: dict[str, str] | None = None) -> dict:
    """instance_types: the type per target this run launches (run.instance_types override)."""
    prompt_tokens = prompt_tokens or {}
    itype = {t: (instance_types or {}).get(t) or targets[t].instance_type for t in targets}
    per_cell, per_target_s = [], {}
    lambda_gb_s = 0.0
    for c in cells:
        m, t = models.get(c.model), targets[c.target]
        levels = levels_for_cell(run, c, m, run_id="estimate", bucket="estimate", pool_sizes=pool_sizes)
        secs, sources = 0.0, set()
        for lvl in levels:
            n = lvl["pool"]["size"] if lvl.get("accuracy") else (lvl.get("quota") or 0)
            ptok = prompt_tokens.get(lvl["pool"]["key"], 512)
            rs, src = request_seconds(lvl, m.type, t, ptok, (measured or {}).get((c.cell_id, lvl["conc"])))
            sources.add(src)
            if lvl.get("duration_s"):
                level_s = lvl["duration_s"] + BARRIER_AND_MAP_S
            else:
                per_worker = math.ceil(n / lvl["conc"]) * rs
                waves = max(1, math.ceil(per_worker / (12 * 60)))
                level_s = per_worker + waves * BARRIER_AND_MAP_S + lvl.get("warmup_requests", 20) * rs / min(8, lvl["conc"])
            secs += level_s
            lambda_gb_s += lvl["conc"] * level_s * WORKER_GB
        secs += sum(CELL_OVERHEAD_S.values()) if t.accelerator == "cpu" else CELL_OVERHEAD_S["fetch_and_load"]
        if run.quality.enabled and run.suite == "standard":
            secs += QUALITY_S
        per_cell.append({"cell_id": c.cell_id, "levels": len(levels), "hours": round(secs / 3600, 2),
                         "speed_source": "measured" if sources == {"measured"} else "prior"})
        per_target_s[c.target] = per_target_s.get(c.target, 0.0) + secs
    run_hours = max(per_target_s.values(), default=0.0) / 3600       # targets run in parallel
    instance_cost = sum(s / 3600 * prices[itype[t]] for t, s in per_target_s.items())
    nat = run_hours * NAT_USD_PER_HOUR
    lam = lambda_gb_s * LAMBDA_USD_PER_GB_S
    return {"cells": per_cell, "target_hours": {t: round(s / 3600, 2) for t, s in per_target_s.items()},
            "instance_types": {t: itype[t] for t in per_target_s},
            "wall_clock_hours": round(run_hours, 2), "usd": {"instances": round(instance_cost, 2),
                                                             "nat_gateway": round(nat, 2), "lambda": round(lam, 2),
                                                             "total": round(instance_cost + nat + lam, 2)},
            "note": "Speeds are priors unless marked measured; NAT data processing ($0.045/GB) is not included."}
