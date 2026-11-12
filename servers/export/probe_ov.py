"""Compile an OpenVINO IR with a cell's properties, run ONE inference on synthetic inputs and
report which kernels the CPU plugin executed (design 7.4 and 14.1).

    python probe_ov.py --xml /models/ov/<hash>/model/openvino_model.xml --hint f16 > probe.json

Output: {"inference_precision", "properties", "inference": {"ran", "error", "profiled_nodes"},
"nodes": [{"name", "type", "exec_type", "runtime_precision"}]}. `exec_type` names the kernel
(for example brgemm_avx512_amx_bf16 or brgemm_avx10_1_512_amx); bench/amx.py turns this into
the AMX verdict and catches a cell labeled FP16 that runs in BF16.

Why an inference is needed: OpenVINO 2026.4 leaves `execType` EMPTY in the runtime model's
rt_info until a request has run; only `InferRequest.profiling_info` (with PERF_COUNT) carries
the kernel names. The first version of this probe read rt_info only, so every OpenVINO cell
came out "AMX INVALID: engine trace shows no AMX GEMMs" and was excluded from the rankings
(seen 2026-11-12 on RoBERTa BF16/INT8). The synthetic inputs are a short token sequence:
token ids, an all-ones attention mask, positions 0..n-1, beam index 0.
"""

from __future__ import annotations

import argparse
import json
import sys

PROPS = ("INFERENCE_PRECISION_HINT", "EXECUTION_MODE_HINT", "PERFORMANCE_HINT", "NUM_STREAMS",
         "INFERENCE_NUM_THREADS", "DYNAMIC_QUANTIZATION_GROUP_SIZE", "KV_CACHE_PRECISION",
         "ENABLE_HYPER_THREADING", "ENABLE_CPU_PINNING")
SEQ = 8     # tokens in the synthetic request


def dummy_feeds(inputs: list[tuple[str, list[int | None], str]], seq: int = SEQ) -> dict:
    """Synthetic inputs for one inference: (name, dims with None for dynamic, numpy dtype name).
    Dynamic dims become 1 (first, the batch) or `seq` (others). Names decide the content: an
    attention mask is all ones, position ids count 0..n-1, a beam index is 0, token-type ids are
    0, anything else (token ids) is a small constant id."""
    import numpy as np

    feeds = {}
    for name, dims, dtype in inputs:
        shape = [d if d is not None else (1 if i == 0 else seq) for i, d in enumerate(dims)] or [1]
        low = name.lower()
        if "attention_mask" in low:
            arr = np.ones(shape, dtype=dtype)
        elif "position_ids" in low:
            arr = np.broadcast_to(np.arange(shape[-1], dtype=dtype), shape).copy()
        elif "beam_idx" in low or "token_type" in low:
            arr = np.zeros(shape, dtype=dtype)
        else:
            arr = np.full(shape, 10, dtype=dtype)
        feeds[name] = arr
    return feeds


def probe(xml: str, hint: str | None, perf_hint: str | None) -> dict:
    import openvino as ov

    core = ov.Core()
    cfg = {"PERF_COUNT": "YES"}
    if hint:
        cfg["INFERENCE_PRECISION_HINT"] = hint
    if perf_hint:
        cfg["PERFORMANCE_HINT"] = perf_hint
    compiled = core.compile_model(core.read_model(xml), "CPU", cfg)
    props = {}
    for p in PROPS:
        try:
            v = compiled.get_property(p)
        except Exception:  # noqa: BLE001, S112 - not every property exists on every build
            continue
        props[p] = v.to_string() if hasattr(v, "to_string") else str(v)   # ov.Type -> "bf16", "f16"
    # One inference so the plugin reports the kernel it chose for every node.
    profiled: dict[str, tuple[str, str]] = {}
    inference = {"ran": False, "error": None, "profiled_nodes": 0}
    try:
        specs = []
        for inp in compiled.inputs:
            ps = inp.get_partial_shape()
            dims = [d.get_length() if d.is_static else None for d in ps] if not ps.rank.is_dynamic else [None, None]
            specs.append((inp.get_any_name(), dims, inp.get_element_type().to_dtype().name))
        req = compiled.create_infer_request()
        req.infer(dummy_feeds(specs))
        for p in req.profiling_info:
            profiled[p.node_name] = (p.node_type, p.exec_type)
        inference.update(ran=True, profiled_nodes=len(profiled))
    except Exception as e:  # noqa: BLE001 - report, keep the compile-time view
        inference["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    nodes = []
    for op in compiled.get_runtime_model().get_ordered_ops():
        rt = op.get_rt_info()

        def val(key: str, rt=rt) -> str:
            return str(rt[key].astype(str)) if key in rt else ""

        name = op.get_friendly_name()
        prof = profiled.get(name)
        nodes.append({"name": name, "type": val("layerType") or (prof[0] if prof else ""),
                      "exec_type": val("execType") or (prof[1] if prof else ""),
                      "runtime_precision": val("runtimePrecision")})
    seen = {n["name"] for n in nodes}
    for name, (ntype, etype) in profiled.items():          # nodes only the profiler knows about
        if name not in seen:
            nodes.append({"name": name, "type": ntype, "exec_type": etype, "runtime_precision": ""})
    ip = {"bfloat16": "bf16", "float16": "f16", "float32": "f32"}.get(
        props.get("INFERENCE_PRECISION_HINT", ""), props.get("INFERENCE_PRECISION_HINT"))
    return {"inference_precision": ip or None, "requested_hint": hint, "properties": props,
            "openvino": ov.get_version(), "cpu": core.get_property("CPU", "FULL_DEVICE_NAME"),
            "inference": inference, "nodes": nodes}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--hint", choices=["bf16", "f16", "f32"], default=None)
    ap.add_argument("--performance-hint", choices=["LATENCY", "THROUGHPUT"], default=None)
    a = ap.parse_args(argv)
    json.dump(probe(a.xml, a.hint, a.performance_hint), sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
