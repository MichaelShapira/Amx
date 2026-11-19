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
GEMM_KINDS = {"matmul", "inner_product", "convolution", "brgemm"}
# oneDNN's brgemm micro-kernel (`onednn_verbose,v1,ukernel,exec,cpu,brgemm,,undef,src:u8 ...`) is how
# vLLM's SGL kernels (csrc/cpu/sgl-kernels, through at::native::cpublas::brgemm) run their GEMMs: the
# W8A8 path and the small BF16 projections never appear as oneDNN primitives, so a W8A8 cell showed
# `gemm_total: 0` and was INVALID on every run through 2026-11-18. The ukernel line carries no ISA
# name; PyTorch's gate (aten/native/CPUBlas.cpp device_check) admits u8 sources only when the
# effective ISA is avx512_core_amx, so a u8/s8 brgemm ukernel IS the AMX INT8 path. A bf16 ukernel
# needs only avx512_core there and is counted as AMX when the host has amx_bf16 (oneDNN picks the
# highest ISA the CPU offers; the primitive lines of the same trace say brg:avx512_core_amx).
UKERNEL_IMPL = "ukernel:brgemm"
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


def _exec_fields(line: str) -> tuple[str, str] | None:
    """(primitive kind, implementation) of an `onednn_verbose,...,exec,cpu,<kind>,<impl>,...` line.
    Plain field splitting: the line is comma-separated and `exec,cpu` fixes the position."""
    parts = line.split(",")
    for i, tok in enumerate(parts[:-3]):
        if tok == "exec" and parts[i + 1] == "cpu":
            kind, impl = parts[i + 2], parts[i + 3]
            if kind == "brgemm" and not impl and "ukernel" in parts[:i]:
                return kind, UKERNEL_IMPL
            return (kind, impl) if kind and impl else None
    return None


def parse_onednn_verbose(text: str, cpu_flags: set[str] | None = None) -> dict:
    """Count executed GEMM-like primitives and brgemm micro-kernels by implementation name.
    cpu_flags: the host's flags, which decide whether bf16 micro-kernel calls count as AMX."""
    impls: Counter[str] = Counter()
    fp16_amx = 0
    int8_amx = 0
    ukernel_u8s8 = ukernel_bf16 = ukernel_other = 0
    for line in text.splitlines():
        if "onednn_verbose" not in line or ",exec," not in line:
            continue
        fields = _exec_fields(line)
        if not fields:
            continue
        kind, impl = fields
        if kind not in GEMM_KINDS:
            continue
        impls[impl] += 1
        if impl == UKERNEL_IMPL:
            if re.search(r"src:[su]8", line):
                ukernel_u8s8 += 1
            elif re.search(r"src:bf16", line):
                ukernel_bf16 += 1
            else:
                ukernel_other += 1
            continue
        if "amx" in impl:
            if "amx_fp16" in impl or re.search(r"src:f16", line):
                fp16_amx += 1
            if re.search(r"src:[su]8", line):
                int8_amx += 1
    amx = sum(v for k, v in impls.items() if "amx" in k)
    amx += ukernel_u8s8                                   # u8 sources pass PyTorch's gate only on avx512_core_amx
    int8_amx += ukernel_u8s8
    bf16_ukernel_is_amx = bool(cpu_flags is None or "amx_bf16" in cpu_flags)
    if bf16_ukernel_is_amx:
        amx += ukernel_bf16
    return {"gemm_total": sum(impls.values()), "gemm_amx": amx, "fp16_amx": fp16_amx,
            "int8_amx": int8_amx, "impls": dict(impls.most_common(10)),
            "ukernel": {"u8s8": ukernel_u8s8, "bf16": ukernel_bf16, "other": ukernel_other,
                        "bf16_counted_as_amx": bf16_ukernel_is_amx}}


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
    out = {"gemm_total": total, "gemm_amx": amx,
           "inference_precision": data.get("inference_precision"),
           "exec_types": dict(types.most_common(10))}
    inf = data.get("inference") or {}
    if total and not any(types.keys() - {""}):
        # Every GEMM has an empty kernel name: the probe saw the compiled graph but no inference
        # ran (OpenVINO fills execType only after a request), so AMX cannot be judged.
        out["note"] = ("kernel names unavailable: the probe did not run an inference"
                       + (f" ({inf.get('error')})" if inf.get("error") else ""))
    return out


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
        uk = (ev.trace.get("ukernel") or {}).get("u8s8", 0)
        via = (f" ({uk} oneDNN brgemm micro-kernel calls with u8/s8 inputs, the AMX INT8 path)" if uk else "")
        result.update(verdict="VERIFIED", basis="engine_trace",
                      reason=f"{ev.trace['gemm_amx']}/{ev.trace['gemm_total']} GEMMs ran AMX kernels{via}")
    else:
        note = (ev.trace or {}).get("note")
        result["reason"] = ("hardware counter unavailable and " + (note if note else "engine trace shows no AMX GEMMs"))
        return result

    # Precision-specific checks: an FP16 cell must really run FP16 on AMX.
    if precision == "fp16" and ev.trace is not None:
        ran_fp16 = (ev.trace.get("fp16_amx", 0) > 0 or
                    (engine == "thin-openvino" and ev.trace.get("inference_precision") == "f16"))
        if not ran_fp16:
            ip = ev.trace.get("inference_precision") or "bf16"
            result["effective_precision"] = ip
            result["precision_flag"] = f"labeled fp16 but ran {ip}"
    # An OpenVINO BF16 cell is an FP16 IR that the CPU plugin runs in BF16 by default; the probe
    # records the precision it actually chose, and anything else (f16, f32) relabels the cell.
    if precision == "bf16-ov" and ev.trace is not None:
        ip = ev.trace.get("inference_precision")
        if ip and ip != "bf16":
            result["effective_precision"] = ip
            result["precision_flag"] = f"labeled bf16-ov but ran {ip}"
    if precision == "w8a8" and ev.dispatch is not None and not ev.dispatch.get("int8_amx_lines"):
        if not (ev.trace and ev.trace.get("int8_amx")):
            result["precision_flag"] = "no INT8 AMX kernel evidence for W8A8 (counter still shows AMX)"
    return result
