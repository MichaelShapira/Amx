"""Migration effort calculator (design section 18, Requirements 16 and 17).

estimate() returns low / likely / high engineer-days with a per-task breakdown that names
the rule behind every line, the risks, and a capacity-driven Go / Conditional / No-go
recommendation per model, read from a benchmark summary (bench/report.py). Cost is shown
next to every verdict and never changes it.
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from bench import report

ROOT = Path(__file__).resolve().parents[1]
Stack = Literal["pytorch_transformers", "sagemaker_pytorch", "vllm_gpu", "tgi", "triton_trtllm"]
CudaDep = Literal["flash_attention", "bitsandbytes", "custom_kernels", "tensorrt_engines"]
Precision = Literal["bf16", "int8", "int4"]
ADDON_FOR_DEP = {"flash_attention": "flash_attention_to_sdpa", "bitsandbytes": "bitsandbytes_to_ov_awq_gptq",
                 "custom_kernels": "custom_cuda_or_triton_kernels", "tensorrt_engines": "tensorrt_engines_outside_triton"}
# Benchmark precisions that satisfy a requested precision, from most to least precise.
PRECISION_RANK = {"fp32": 0, "fp16": 1, "bf16": 1, "w8a8": 2, "int8-ov": 2, "awq-int4": 3, "int4-ov": 3}
REQUEST_RANK = {"bf16": 1, "int8": 2, "int4": 3}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelInput(Strict):
    name: str
    model_type: Literal["decoder", "encoder-decoder", "encoder-classifier", "token-classifier"]
    params_b: float
    stack: Stack
    cuda_deps: list[CudaDep] = Field(default_factory=list)
    precision: Precision = "bf16"
    required_concurrency: int | None = None
    license: str | None = None
    gated: bool = False
    workload: str | None = None        # DSPM workload to read results for (W1, W3, ...)


class SlaInput(Strict):
    p99_ttft_ms: float | None = None
    p50_tpot_ms: float | None = None
    p99_e2e_ms: float | None = None
    max_error_pct: float = 1.0


class ProjectInput(Strict):
    models: list[ModelInput]
    infrastructure: Literal["eks", "ecs", "ec2", "sagemaker"] = "ec2"
    iac_and_cicd_exist: bool = True
    familiarity: Literal["low", "medium", "high"] = "medium"
    sla: SlaInput = Field(default_factory=SlaInput)


class JournalEntry(Strict):
    rule_id: str
    actual_days: float
    predicted_mid_days: float
    model: str | None = None
    target: str | None = None
    precision: str | None = None


def load_config(path: Path = ROOT / "config/migration_effort.yaml") -> dict:
    return yaml.safe_load(path.read_text())


def _mid(r: list[float]) -> float:
    return (r[0] + r[1]) / 2


def calibration_factors(journal: list[JournalEntry], clamp: list[float]) -> dict[str, tuple[float, int]]:
    by: dict[str, list[float]] = {}
    for e in journal:
        if e.predicted_mid_days > 0:
            by.setdefault(e.rule_id, []).append(e.actual_days / e.predicted_mid_days)
    return {k: (min(max(statistics.median(v), clamp[0]), clamp[1]), len(v)) for k, v in by.items()}


def _half(x: float) -> float:
    return math.floor(x * 2 + 0.5) / 2


def base_rule(m: ModelInput, cfg: dict) -> dict | None:
    matches = []
    for r in cfg["base_rules"]:
        w = r["when"]
        if ("stack" in w and m.stack in w["stack"]) or ("model_type" in w and m.model_type in w["model_type"]):
            matches.append(r)
    if not matches:
        return None
    return max(matches, key=lambda r: (r["days"][1] - r["days"][0], r["days"][1]))


def risks(p: ProjectInput, rec: dict[str, dict], benchmarked_max_b: float) -> list[str]:
    out = []
    for m in p.models:
        if "custom_kernels" in m.cuda_deps:
            out.append(f"{m.name}: custom CUDA/Triton kernels have no CPU equivalent and must be rewritten or replaced")
        if "bitsandbytes" in m.cuda_deps:
            out.append(f"{m.name}: bitsandbytes weights change format (AWQ/GPTQ/OpenVINO); accuracy must be re-validated")
        if m.precision == "int4":
            out.append(f"{m.name}: INT4 carries the largest accuracy risk")
        if m.model_type == "encoder-decoder" and m.stack in ("vllm_gpu", "tgi", "triton_trtllm"):
            out.append(f"{m.name}: encoder-decoder on {m.stack} needs an engine change (no vLLM CPU path for T5)")
        if m.stack == "triton_trtllm":
            out.append(f"{m.name}: Triton + TensorRT-LLM means both engine and model format change")
        if m.gated:
            out.append(f"{m.name}: gated model; license acceptance and token handling are needed on CPU too")
        if m.license and m.license in ("qwen-research", "cc-by-nc-4.0", "cc-by-nc-nd-4.0"):
            out.append(f"{m.name}: non-commercial license ({m.license})")
        if m.params_b > benchmarked_max_b:
            out.append(f"{m.name}: {m.params_b}B is above the 4B benchmark cap; results are extrapolated")
        r = rec.get(m.name, {})
        if r.get("verdict") in ("No-go", "Conditional"):
            out.append(f"{m.name}: {r['reason']}")
        if r.get("amx_unverified"):
            out.append(f"{m.name}: AMX was not verified for some proxy results")
        eq = r.get("capacity_equivalence")
        if eq and eq.get("cpu_instances_per_gpu", 0) > 4:
            out.append(f"{m.name}: {eq['cpu_instances_per_gpu']} CPU instances per GPU instance "
                       f"({eq['standard_vcpus_needed']} Standard vCPUs of On-Demand quota)")
    if p.familiarity == "low":
        out.append("team has low CPU-tuning familiarity (thread binding, NUMA, oneDNN)")
    return out


def nearest_benchmarked(m: ModelInput, catalog) -> tuple[object | None, str | None]:
    same = [x for x in catalog.models if x.enabled and x.type == m.model_type]
    if not same:
        return None, f"no benchmarked model of type {m.model_type}"
    best = min(same, key=lambda x: abs(math.log(max(m.params_b, 1e-3) / (x.params / 1e9))))
    return best, None


def recommend(m: ModelInput, summary: dict | None, sla: SlaInput, *, catalog, limit: int,
              far_factor: float) -> dict:
    if summary is None:
        return {"verdict": "Insufficient data", "reason": "no benchmark results loaded"}
    proxy, err = nearest_benchmarked(m, catalog)
    if proxy is None:
        return {"verdict": "Insufficient data", "reason": err}
    ratio = max(m.params_b, proxy.params / 1e9) / min(m.params_b, proxy.params / 1e9)
    if ratio > far_factor:
        return {"verdict": "Insufficient data", "proxy": proxy.id,
                "reason": f"{m.params_b}B is far from the nearest benchmarked {proxy.id} ({proxy.params / 1e9:.2f}B)"}
    rows = [r for r in summary["rows"] if r["model"] == proxy.id and (m.workload is None or r["workload"] == m.workload)]
    if not rows:
        return {"verdict": "Insufficient data", "proxy": proxy.id, "reason": f"no result rows for {proxy.id}"}
    s = sla.model_dump()
    for r in rows:
        r["user_meets"], r["user_reason"] = report.meets_sla(r, s)
    gpu_ok = [r for r in rows if r["accelerator"] == "gpu" and r["user_meets"]]
    cpu_ok = [r for r in rows if r["accelerator"] == "cpu" and r["user_meets"]]
    best_gpu = max(gpu_ok, key=report._throughput) if gpu_ok else None
    amx_unverified = any(r["accelerator"] == "cpu" and r.get("amx") != "VERIFIED" for r in rows)
    if not cpu_ok:
        return {"verdict": "No-go", "proxy": proxy.id, "reason": f"no AMX-verified, quality-passing CPU row of "
                f"{proxy.id} meets the SLA", "rows_used": [], "amx_unverified": amx_unverified,
                "best_gpu": report._row_ref(best_gpu)}
    want = REQUEST_RANK[m.precision]

    def eq_for(r):
        if best_gpu is None or not r.get("usd_per_hour") or not best_gpu.get("usd_per_hour"):
            return None
        e = report.cost.capacity_equivalence(
            gpu_throughput=report._throughput(best_gpu), cpu_throughput=report._throughput(r),
            cpu_vcpus=report.INSTANCE_VCPUS.get(r["instance_type"], 0),
            gpu_vcpus=report.INSTANCE_VCPUS.get(best_gpu["instance_type"], 0),
            cpu_price=r["usd_per_hour"], gpu_price=best_gpu["usd_per_hour"])
        return None if e is None else e.__dict__

    def score(r):
        eq = eq_for(r)
        return (PRECISION_RANK.get(r["precision"], 9) > want,                     # quantized below request
                m.required_concurrency is not None and r["conc"] < m.required_concurrency,
                eq is not None and eq["cpu_instances_per_gpu"] > limit,
                eq["cpu_instances_per_gpu"] if eq else 0, -report._throughput(r))

    best = min(cpu_ok, key=score)
    eq = eq_for(best)
    below, low_conc, too_many = score(best)[:3]
    reasons = []
    if below:
        reasons.append(f"meets the SLA only at {best['precision']}, below the requested {m.precision}")
    if low_conc:
        reasons.append(f"meets the SLA only up to concurrency {max(r['conc'] for r in cpu_ok)} "
                       f"(required {m.required_concurrency})")
    if too_many:
        reasons.append(f"needs {eq['cpu_instances_per_gpu']} CPU instances per GPU instance (limit {limit})")
    verdict = "Conditional" if reasons else "Go"
    return {"verdict": verdict, "proxy": proxy.id,
            "reason": "; ".join(reasons) or f"{best['instance_type']} {best['engine']} {best['precision']} at C="
                                            f"{best['conc']} meets the SLA",
            "rows_used": [report._row_ref(best)] + ([report._row_ref(best_gpu)] if best_gpu else []),
            "capacity_equivalence": eq, "hourly_cost_ratio": eq["hourly_cost_ratio"] if eq else None,
            "amx_unverified": amx_unverified, "best_gpu": report._row_ref(best_gpu)}


def estimate(p: ProjectInput, *, cfg: dict | None = None, journal: list[JournalEntry] | None = None,
             summary: dict | None = None, catalog=None) -> dict:
    from bench.config import load_models

    cfg = cfg or load_config()
    catalog = catalog or load_models(ROOT / "config/models.yaml")
    factors = calibration_factors(journal or [], cfg["calibration"]["clamp"])
    lines: list[dict] = []

    def add(task: str, rule: str, days: list[float], model: str | None = None) -> None:
        f, n = factors.get(rule, (1.0, 0))
        lines.append({"model": model, "task": task, "rule": rule, "days": [days[0] * f, days[1] * f],
                      "calibration": f"calibrated from {n} entries (x{f:.2f})" if n else "uncalibrated"})

    for m in p.models:
        r = base_rule(m, cfg)
        if r is None:
            raise ValueError(f"{m.name}: no base rule matches stack {m.stack}")
        add(f"port {m.name} from {m.stack}", r["id"], r["days"], m.name)
        for dep in m.cuda_deps:
            add(f"{m.name}: replace {dep}", ADDON_FOR_DEP[dep], cfg["addons_per_model"][ADDON_FOR_DEP[dep]], m.name)
        add(f"{m.name}: accuracy validation at {m.precision}", "accuracy_validation",
            cfg["addons_per_model"]["accuracy_validation"][m.precision], m.name)
    add("load and latency testing", "load_latency_testing", cfg["addons_per_project"]["load_latency_testing"])
    infra = cfg["addons_per_project"]["infra_ci_changes"]
    add(f"infrastructure and CI changes ({p.infrastructure})", "infra_ci_changes",
        infra["iac_and_cicd_exist"] if p.iac_and_cicd_exist else infra["otherwise"])
    mult = cfg["familiarity_multiplier"][p.familiarity]
    low = sum(x["days"][0] for x in lines) * mult
    high = sum(x["days"][1] for x in lines) * mult
    likely = sum(_mid(x["days"]) for x in lines) * mult
    rec_cfg = cfg.get("recommendation", {})
    recs = {m.name: recommend(m, summary, p.sla, catalog=catalog, limit=rec_cfg.get("max_cpu_instances_per_gpu", 4),
                              far_factor=rec_cfg.get("far_outside_size_factor", 3.0)) for m in p.models}
    order = ["No-go", "Insufficient data", "Conditional", "Go"]
    overall = min((r["verdict"] for r in recs.values()), key=order.index) if recs else "Insufficient data"
    bench_max = max(x.params for x in catalog.models if x.enabled) / 1e9
    return {"low": _half(low), "likely": _half(likely), "high": _half(high), "units": cfg["units"],
            "familiarity_multiplier": mult, "breakdown": [{**x, "days": [_half(x["days"][0] * mult),
                                                                          _half(x["days"][1] * mult)]} for x in lines],
            "risks": risks(p, recs, bench_max), "recommendations": recs, "overall": overall,
            "note": "Capacity decides the verdict; cost ratios are shown for information and never change it."}
