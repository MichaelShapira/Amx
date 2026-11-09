"""Compile an OpenVINO IR with a cell's properties and report what the CPU plugin will
actually execute (design 7.4 and 14.1).

    python probe_ov.py --xml /models/ov/<hash>/model/openvino_model.xml --hint f16 > probe.json

Output: {"inference_precision", "properties", "nodes": [{"name", "type", "exec_type",
"runtime_precision"}]}. The execution graph is fixed at compile time, so no inference is
needed: execType names the kernel (for example brgemm_avx512_amx_bf16) and
runtimePrecision the data type the node runs in. bench/amx.py turns this into the AMX
verdict and catches a cell labeled FP16 that runs in BF16.
"""

from __future__ import annotations

import argparse
import json
import sys

PROPS = ("INFERENCE_PRECISION_HINT", "EXECUTION_MODE_HINT", "PERFORMANCE_HINT", "NUM_STREAMS",
         "INFERENCE_NUM_THREADS", "DYNAMIC_QUANTIZATION_GROUP_SIZE", "KV_CACHE_PRECISION",
         "ENABLE_HYPER_THREADING", "ENABLE_CPU_PINNING")


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
    nodes = []
    for op in compiled.get_runtime_model().get_ordered_ops():
        rt = op.get_rt_info()

        def val(key: str, rt=rt) -> str:
            return str(rt[key].astype(str)) if key in rt else ""

        nodes.append({"name": op.get_friendly_name(), "type": val("layerType"), "exec_type": val("execType"),
                      "runtime_precision": val("runtimePrecision")})
    ip = {"bfloat16": "bf16", "float16": "f16", "float32": "f32"}.get(
        props.get("INFERENCE_PRECISION_HINT", ""), props.get("INFERENCE_PRECISION_HINT"))
    return {"inference_precision": ip or None, "requested_hint": hint, "properties": props,
            "openvino": ov.get_version(), "cpu": core.get_property("CPU", "FULL_DEVICE_NAME"), "nodes": nodes}


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
