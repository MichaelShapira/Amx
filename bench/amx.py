"""AMX verification (design section 14, Requirement 12).

A CPU result counts only when the CPU flags are present AND either the hardware counter
(exe.amx_busy) or an engine trace shows AMX kernels running. Otherwise it is INVALID.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

REQUIRED_FLAGS = {"amx_bf16", "amx_int8", "amx_tile"}
COUNTER_RATIO_MIN = 1e-4
GEMM_KINDS = {"matmul", "inner_product", "convolution"}
OV_GEMM_TYPES = {"FullyConnected", "MatMul", "Convolution", "FullyConnectedCompressed"}


def parse_cpu_flags(text: str) -> set[str]:
    """Flags from `lscpu` ("Flags:") or /proc/cpuinfo ("flags :")."""
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key.strip().lower() == "flags":
            return set(rest.split())
    return set()


def parse_perf_csv(text: str) -> dict:
    """`perf stat -x,` output -> {event: count}; unsupported events map to None."""
    counts: dict[str, int | None] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        value, event = parts[0].strip(), parts[2].strip()
        if value.startswith("<"):
            counts[event] = None
        else:
            try:
                counts[event] = int(float(value))
            except ValueError:
                counts[event] = None
    return counts


_VERBOSE_EXEC = re.compile(r"onednn_verbose,(?:[^,]*,)*?exec,cpu,([a-z_]+),([^,]+),")


def parse_onednn_verbose(text: str) -> dict:
    """Count executed GEMM-like primitives by implementation name."""
    impls: Counter[str] = Counter()
    fp16_amx = 0
    int8_amx = 0
    for line in text.splitlines():
        if "onednn_verbose" not in line or ",exec," not in line:
            continue
        m = _VERBOSE_EXEC.search(line)
        if not m:
            continue
        kind, impl = m.group(1), m.group(2)
        if kind not in GEMM_KINDS:
            continue
        impls[impl] += 1
        if "amx" in impl:
            if "amx_fp16" in impl or re.search(r"src:f16", line):
                fp16_amx += 1
            if re.search(r"src:[su]8", line):
                int8_amx += 1
    amx = sum(v for k, v in impls.items() if "amx" in k)
    return {"gemm_total": sum(impls.values()), "gemm_amx": amx, "fp16_amx": fp16_amx,
            "int8_amx": int8_amx, "impls": dict(impls.most_common(10))}


_DISPATCH = re.compile(r"CPU unquantized GEMM dispatch: using (.+)")


def parse_vllm_dispatch(text: str) -> dict:
    out: Counter[str] = Counter()
    int8_kernel = 0
    for line in text.splitlines():
        m = _DISPATCH.search(line)
        if m:
            what = m.group(1)
            key = ("sgl" if "sgl" in what else "onednn" if "oneDNN" in what or "onednn" in what
                   else "zentorch" if "zentorch" in what else "torch_fallback")
            out[key] += 1
        if re.search(r"int8_scaled_mm|w8a8|Int8", line) and re.search(r"sgl|amx|oneDNN", line, re.I):
            int8_kernel += 1
    return {"dispatch": dict(out), "int8_amx_lines": int8_kernel}


def parse_ov_probe(text: str) -> dict:
    """Output of host/probe_ov.py: {"inference_precision": .., "nodes": [{"type", "exec_type"}]}."""
    data = json.loads(text)
    total = amx = 0
    types: Counter[str] = Counter()
    for n in data.get("nodes", []):
        if n.get("type") not in OV_GEMM_TYPES:
            continue
        total += 1
        et = n.get("exec_type", "")
        types[et] += 1
        if "amx" in et.lower():
            amx += 1
    return {"gemm_total": total, "gemm_amx": amx,
            "inference_precision": data.get("inference_precision"),
            "exec_types": dict(types.most_common(10))}


@dataclass
class Evidence:
    cpu_flags: set[str]
    counter: dict | None = None          # {"exe.amx_busy": int|None, "cycles": int|None}
    trace: dict | None = None            # parse_onednn_verbose or parse_ov_probe result
    dispatch: dict | None = None         # parse_vllm_dispatch result
    notes: list[str] = field(default_factory=list)


def verdict(ev: Evidence, *, precision: str, engine: str) -> dict:
    """Return {"verdict": VERIFIED|INVALID, "basis", "reason", "effective_precision", ...}."""
    result = {"verdict": "INVALID", "basis": None, "reason": "", "effective_precision": precision,
              "precision_flag": None}
    missing = REQUIRED_FLAGS - ev.cpu_flags
    if precision == "fp16":
        missing |= {"amx_fp16"} - ev.cpu_flags
    if missing:
        result["reason"] = f"missing CPU flags: {sorted(missing)}"
        return result

    busy = cycles = None
    if ev.counter:
        busy = ev.counter.get("exe.amx_busy")
        cycles = ev.counter.get("cycles") or ev.counter.get("cpu-cycles")
    counter_available = busy is not None and bool(cycles)
    trace_amx = bool(ev.trace and ev.trace.get("gemm_total") and
                     ev.trace.get("gemm_amx", 0) >= 0.5 * ev.trace["gemm_total"])

    if counter_available:
        ratio = busy / cycles
        result["counter_ratio"] = ratio
        if busy > 0 and ratio >= COUNTER_RATIO_MIN:
            result.update(verdict="VERIFIED", basis="hardware_counter",
                          reason=f"exe.amx_busy/cycles = {ratio:.2e}")
        else:
            result["reason"] = f"exe.amx_busy/cycles = {ratio:.2e} below {COUNTER_RATIO_MIN:g}"
            return result
    elif trace_amx:
        result.update(verdict="VERIFIED", basis="engine_trace",
                      reason=f"{ev.trace['gemm_amx']}/{ev.trace['gemm_total']} GEMMs ran AMX kernels")
    else:
        result["reason"] = "hardware counter unavailable and engine trace shows no AMX GEMMs"
        return result

    # Precision-specific checks: an FP16 cell must really run FP16 on AMX.
    if precision == "fp16" and ev.trace is not None:
        ran_fp16 = (ev.trace.get("fp16_amx", 0) > 0 or
                    (engine == "thin-openvino" and ev.trace.get("inference_precision") == "f16"))
        if not ran_fp16:
            ip = ev.trace.get("inference_precision") or "bf16"
            result["effective_precision"] = ip
            result["precision_flag"] = f"labeled fp16 but ran {ip}"
    if precision == "w8a8" and ev.dispatch is not None and not ev.dispatch.get("int8_amx_lines"):
        if not (ev.trace and ev.trace.get("int8_amx")):
            result["precision_flag"] = "no INT8 AMX kernel evidence for W8A8 (counter still shows AMX)"
    return result
