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
