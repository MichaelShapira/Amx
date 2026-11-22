import json

from bench import amx

LSCPU = """Architecture: x86_64
Model name: Intel(R) Xeon(R) 6975P-C
Flags: fpu vme avx512f avx512_bf16 amx_bf16 avx512_fp16 amx_tile amx_int8 amx_fp16
"""
LSCPU_SPR = "Flags: fpu avx512f amx_bf16 amx_tile amx_int8\n"

VERBOSE_AMX = """onednn_verbose,v1,info,oneDNN v3.13.0
onednn_verbose,v1,primitive,exec,cpu,matmul,brg_matmul:avx10_1_512_amx,undef,src:bf16::blocked:ab::f0 wei:bf16::blocked:ab::f0 dst:bf16::blocked:ab::f0,,,32x2048:2048x2048,0.21
onednn_verbose,v1,primitive,exec,cpu,matmul,brg_matmul:avx10_1_512_amx,undef,src:bf16::blocked:ab::f0,,,32x2048:2048x8192,0.5
onednn_verbose,v1,primitive,exec,cpu,reorder,jit:uni,undef,src:f32 dst:bf16,,,2048x2048,0.1
"""
VERBOSE_AVX = """onednn_verbose,v1,primitive,exec,cpu,matmul,brg_matmul:avx512_core_bf16,undef,src:bf16,,,1x2048:2048x2048,0.9
"""
VERBOSE_FP16 = """onednn_verbose,v1,primitive,exec,cpu,matmul,brg_matmul:avx10_1_512_amx_fp16,undef,src:f16::blocked:ab::f0,,,8x1024:1024x1024,0.1
"""
PERF_OK = "123456789,,exe.amx_busy,1000,100.00,,\n9876543210,,cycles,1000,100.00,,\n"
PERF_ZERO = "0,,exe.amx_busy,1000,100.00,,\n9876543210,,cycles,1000,100.00,,\n"
PERF_UNSUPPORTED = "<not supported>,,exe.amx_busy,0,0.00,,\n9876543210,,cycles,1000,100.00,,\n"


def ev(flags=LSCPU, perf=None, verbose=None, ov=None, dispatch=None):
    return amx.Evidence(
        cpu_flags=amx.parse_cpu_flags(flags),
        counter=amx.parse_perf_csv(perf) if perf else None,
        trace=(amx.parse_onednn_verbose(verbose) if verbose else
               amx.parse_ov_probe(ov) if ov else None),
        dispatch=amx.parse_vllm_dispatch(dispatch) if dispatch else None)


def test_parsers():
    assert {"amx_bf16", "amx_fp16"} <= amx.parse_cpu_flags(LSCPU)
    assert amx.parse_perf_csv(PERF_OK)["exe.amx_busy"] == 123456789
    assert amx.parse_perf_csv(PERF_UNSUPPORTED)["exe.amx_busy"] is None
    t = amx.parse_onednn_verbose(VERBOSE_AMX)
    assert t["gemm_total"] == 2 and t["gemm_amx"] == 2  # reorder excluded
    d = amx.parse_vllm_dispatch("INFO CPU unquantized GEMM dispatch: using oneDNN onednn_mm\n"
                                "DEBUG CPU unquantized GEMM dispatch: using sgl-kernel weight_packed_linear")
    assert d["dispatch"] == {"onednn": 1, "sgl": 1}


def test_missing_flags_invalid():
    v = amx.verdict(ev(flags="Flags: avx512f\n", perf=PERF_OK), precision="bf16", engine="vllm")
    assert v["verdict"] == "INVALID" and "missing CPU flags" in v["reason"]


def test_counter_verifies():
    v = amx.verdict(ev(perf=PERF_OK), precision="bf16", engine="vllm")
    assert v["verdict"] == "VERIFIED" and v["basis"] == "hardware_counter"


def test_counter_zero_invalid_even_with_trace():
    v = amx.verdict(ev(perf=PERF_ZERO, verbose=VERBOSE_AMX), precision="bf16", engine="vllm")
    assert v["verdict"] == "INVALID"


def test_trace_fallback_when_counter_unsupported():
    v = amx.verdict(ev(perf=PERF_UNSUPPORTED, verbose=VERBOSE_AMX), precision="bf16", engine="vllm")
    assert v["verdict"] == "VERIFIED" and v["basis"] == "engine_trace"
    v2 = amx.verdict(ev(perf=PERF_UNSUPPORTED, verbose=VERBOSE_AVX), precision="bf16", engine="vllm")
    assert v2["verdict"] == "INVALID"


def test_fp16_needs_amx_fp16_flag_and_execution():
    v = amx.verdict(ev(flags=LSCPU_SPR, perf=PERF_OK), precision="fp16", engine="thin-torch")
    assert v["verdict"] == "INVALID"
    v2 = amx.verdict(ev(perf=PERF_OK, verbose=VERBOSE_AMX), precision="fp16", engine="thin-torch")
    assert v2["verdict"] == "VERIFIED" and v2["precision_flag"] and v2["effective_precision"] == "bf16"
    v3 = amx.verdict(ev(perf=PERF_OK, verbose=VERBOSE_FP16), precision="fp16", engine="thin-torch")
    assert v3["precision_flag"] is None


def test_openvino_probe():
    probe = json.dumps({"inference_precision": "f16", "nodes": [
        {"type": "FullyConnected", "exec_type": "brgemm_avx512_amx_f16"},
        {"type": "FullyConnected", "exec_type": "brgemm_avx512_amx_f16"},
        {"type": "Softmax", "exec_type": "jit_avx512"}]})
    v = amx.verdict(ev(perf=PERF_UNSUPPORTED, ov=probe), precision="fp16", engine="thin-openvino")
    assert v["verdict"] == "VERIFIED" and v["precision_flag"] is None


def test_ov_probe_without_inference_is_named_not_mislabeled():
    """2026-11-12: every OpenVINO cell came out INVALID because the probe read execType from
    rt_info, which OpenVINO leaves empty until an inference runs. The parser now says so."""
    import json

    from bench import amx

    empty = json.dumps({"inference_precision": "bf16", "inference": {"ran": False, "error": "RuntimeError: x"},
                        "nodes": [{"type": "FullyConnected", "exec_type": ""}] * 3 + [{"type": "Eltwise", "exec_type": ""}]})
    t = amx.parse_ov_probe(empty)
    assert t["gemm_total"] == 3 and t["gemm_amx"] == 0 and "did not run an inference" in t["note"] and "RuntimeError" in t["note"]
    v = amx.verdict(amx.Evidence(cpu_flags=amx.REQUIRED_FLAGS | {"amx_fp16"}, trace=t), precision="bf16", engine="thin-openvino")
    assert v["verdict"] == "INVALID" and "did not run an inference" in v["reason"]
    good = json.dumps({"inference_precision": "bf16", "inference": {"ran": True, "profiled_nodes": 4},
                       "nodes": [{"type": "FullyConnected", "exec_type": "brgemm_avx10_1_512_amx"}] * 3 + [{"type": "Eltwise", "exec_type": "jit_avx512"}]})
    t2 = amx.parse_ov_probe(good)
    assert t2["gemm_amx"] == 3 and "note" not in t2
    assert amx.verdict(amx.Evidence(cpu_flags=amx.REQUIRED_FLAGS, trace=t2), precision="bf16", engine="thin-openvino")["verdict"] == "VERIFIED"


def test_probe_dummy_feeds_follow_input_names_and_shapes():
    import importlib.util
    from pathlib import Path

    import numpy as np

    spec = importlib.util.spec_from_file_location("probe_ov", Path(__file__).resolve().parents[1] / "servers/export/probe_ov.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    feeds = mod.dummy_feeds([("input_ids", [None, None], "int64"), ("attention_mask", [None, None], "int64"),
                             ("position_ids", [None, None], "int64"), ("beam_idx", [None], "int32"),
                             ("token_type_ids", [1, 8], "int64")], seq=8)
    assert feeds["input_ids"].shape == (1, 8) and feeds["input_ids"].dtype == np.int64 and int(feeds["input_ids"][0, 0]) == 10
    assert feeds["attention_mask"].tolist() == [[1] * 8]
    assert feeds["position_ids"].tolist() == [list(range(8))]
    assert feeds["beam_idx"].shape == (1,) and feeds["beam_idx"].dtype == np.int32 and int(feeds["beam_idx"][0]) == 0
    assert feeds["token_type_ids"].tolist() == [[0] * 8]


def test_openvino_bf16_cell_is_relabeled_when_the_plugin_did_not_run_bf16():
    """bf16-ov is an FP16 IR that the CPU plugin runs in BF16 by default; the probe records the
    precision it chose, and a different one relabels the cell instead of passing as BF16."""
    def trace(ip):
        return amx.parse_ov_probe(json.dumps({"inference_precision": ip, "inference": {"ran": True, "profiled_nodes": 3},
                                              "nodes": [{"type": "FullyConnected", "exec_type": "brgemm_avx512_amx"}] * 3}))
    flags = amx.REQUIRED_FLAGS | {"amx_fp16"}
    ok = amx.verdict(amx.Evidence(cpu_flags=flags, trace=trace("bf16")), precision="bf16-ov", engine="ovms")
    assert ok["verdict"] == "VERIFIED" and ok["precision_flag"] is None and ok["effective_precision"] == "bf16-ov"
    f16 = amx.verdict(amx.Evidence(cpu_flags=flags, trace=trace("f16")), precision="bf16-ov", engine="ovms")
    assert f16["verdict"] == "VERIFIED" and f16["effective_precision"] == "f16" and "ran f16" in f16["precision_flag"]
    # The plain vLLM bf16 cell is untouched by this check (its trace carries no inference_precision).
    assert amx.verdict(amx.Evidence(cpu_flags=flags, trace=trace("f16")), precision="bf16", engine="vllm")["precision_flag"] is None


UKERNEL_INT8 = """onednn_verbose,v1,info,oneDNN v3.13.0
onednn_verbose,v1,ukernel,exec,cpu,brgemm,,undef,src:u8::blocked:ab::f0 wei:s8::blocked:ab::f0 dst:s32::blocked:ab::f0,,bs:1 beta:0,32x2048:2048x32,0.00219727
onednn_verbose,v1,ukernel,exec,cpu,brgemm,,undef,src:u8::blocked:ab::f0 wei:s8::blocked:ab::f0 dst:s32::blocked:ab::f0,,bs:1 beta:0,32x2048:2048x32,0.000976562
onednn_verbose,v1,ukernel,exec,cpu,brgemm,,undef,src:bf16::blocked:ab::f0 wei:bf16::blocked:ab::f0 dst:f32::blocked:ab::f0,,bs:1 beta:0,32x256:256x32,0.001
"""


def test_w8a8_through_the_brgemm_micro_kernel_is_amx_verified():
    """The vLLM SGL INT8 path runs oneDNN's brgemm micro-kernel: no primitive lines, so the cell read
    `gemm_total: 0` and was INVALID on every run through 2026-11-18 (run qwen-2-5-20261115T102048Z)."""
    flags = {"amx_bf16", "amx_int8", "amx_tile", "avx512_bf16"}
    tr = amx.parse_onednn_verbose(UKERNEL_INT8, flags)
    assert tr["gemm_total"] == 3 and tr["gemm_amx"] == 3 and tr["int8_amx"] == 2
    assert tr["impls"] == {"ukernel:brgemm": 3} and tr["ukernel"] == {"u8s8": 2, "bf16": 1, "other": 0, "bf16_counted_as_amx": True}
    ev = amx.Evidence(cpu_flags=flags, counter={"exe.amx_busy": None, "cycles": None}, trace=tr,
                      dispatch={"dispatch": {}, "int8_amx_lines": 0})
    v = amx.verdict(ev, precision="w8a8", engine="vllm")
    assert v["verdict"] == "VERIFIED" and v["basis"] == "engine_trace" and v["precision_flag"] is None
    assert "u8/s8 inputs" in v["reason"]
    # Without amx_bf16 on the host a bf16 micro-kernel call is not counted as AMX (u8 still is).
    tr2 = amx.parse_onednn_verbose(UKERNEL_INT8, {"avx512_bf16", "avx512_vnni"})
    assert tr2["gemm_amx"] == 2 and tr2["ukernel"]["bf16_counted_as_amx"] is False
    # The old parse of the same lines saw nothing: the regression this guards against.
    assert amx.parse_onednn_verbose(UKERNEL_INT8.replace(",ukernel,", ",primitive,"), flags)["gemm_total"] == 0


def test_probe_is_scheduled_only_for_engines_the_host_script_handles():
    """Every engine in PROBED_ENGINES has a case in host/probe_amx.sh, and the decision model's server has
    none: scheduling the probe for it failed the whole CPU cell on 2026-11-22."""
    import re
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "host" / "probe_amx.sh"
    cases = set()
    for line in script.read_text().splitlines():
        m = re.match(r"\s*([a-z0-9|_-]+)\)\s*$", line)
        if m and m.group(1) != "*":
            cases.update(m.group(1).split("|"))
    assert amx.PROBED_ENGINES <= cases
    assert not amx.has_software_probe("decider")
    assert "decider" in cases  # by hand the script is a no-op, not a failure
    assert amx.has_software_probe("vllm")
