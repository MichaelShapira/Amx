"""Migration calculator: rules, add-ons, calibration, risks and verdicts (task 14)."""

from __future__ import annotations

import pytest

from bench import estimator as E


def M(**kw) -> E.ModelInput:
    base = {"name": "chat", "model_type": "decoder", "params_b": 3.0, "stack": "vllm_gpu"}
    return E.ModelInput(**{**base, **kw})


def P(models, **kw) -> E.ProjectInput:
    return E.ProjectInput(models=models, **kw)


@pytest.mark.parametrize("stack,mtype,rule,days", [
    ("pytorch_transformers", "decoder", "pytorch_or_sagemaker", [1, 3]),
    ("sagemaker_pytorch", "decoder", "pytorch_or_sagemaker", [1, 3]),
    ("vllm_gpu", "decoder", "vllm_gpu_to_vllm_cpu", [3, 7]),
    ("tgi", "decoder", "tgi_to_optimum_ovms", [5, 10]),
    ("triton_trtllm", "decoder", "triton_trtllm", [10, 20]),
    ("pytorch_transformers", "encoder-decoder", "flan_t5_to_openvino", [3, 7]),   # widest wins over pytorch
    ("vllm_gpu", "encoder-decoder", "vllm_gpu_to_vllm_cpu", [3, 7]),             # tie: same width
])
def test_base_rules(stack, mtype, rule, days):
    r = E.base_rule(M(stack=stack, model_type=mtype), E.load_config())
    assert r["id"] == rule and r["days"] == days


def test_totals_addons_and_familiarity():
    p = P([M(cuda_deps=["flash_attention", "bitsandbytes"], precision="int4")], iac_and_cicd_exist=False,
          familiarity="low")
    out = E.estimate(p)
    # base 3-7, flash 1-1, bnb 2-5, int4 validation 4-5, load test 3-5, infra (no IaC) 3-5
    assert out["low"] == E._half(16 * 1.3) and out["high"] == E._half(28 * 1.3)
    assert out["likely"] == E._half(22 * 1.3)
    rules = [x["rule"] for x in out["breakdown"]]
    assert rules == ["vllm_gpu_to_vllm_cpu", "flash_attention_to_sdpa", "bitsandbytes_to_ov_awq_gptq",
                     "accuracy_validation", "load_latency_testing", "infra_ci_changes"]
    assert all(x["calibration"] == "uncalibrated" for x in out["breakdown"])
    assert any("bitsandbytes" in r for r in out["risks"]) and any("INT4" in r for r in out["risks"])
    assert any("low CPU-tuning" in r for r in out["risks"])
    assert out["overall"] == "Insufficient data"            # no benchmark summary given


def test_calibration_median_and_clamp():
    j = [E.JournalEntry(rule_id="vllm_gpu_to_vllm_cpu", actual_days=a, predicted_mid_days=5) for a in (10, 12, 100)]
    f = E.calibration_factors(j, [0.5, 2.0])
    assert f["vllm_gpu_to_vllm_cpu"] == (2.0, 3)            # median 2.4, clamped to 2.0
    j2 = [E.JournalEntry(rule_id="load_latency_testing", actual_days=1, predicted_mid_days=4)]
    assert E.calibration_factors(j2, [0.5, 2.0])["load_latency_testing"] == (0.5, 1)
    out = E.estimate(P([M()]), journal=j)
    line = out["breakdown"][0]
    assert line["days"] == [6.0, 14.0] and "calibrated from 3 entries" in line["calibration"]


def _row(acc, precision, conc, tp, *, meets=True, amx="VERIFIED", itype=None, price=None, e2e=500.0):
    return {"model": "m1-qwen2.5-3b", "workload": None, "bucket": None, "kind": "generation", "batch": 1,
            "accelerator": acc, "target": "t1-g6" if acc == "gpu" else "t2-m8i-4xl", "precision": precision,
            "variant": "default", "conc": conc, "engine": "vllm", "agg_out_tok_s": tp, "cls_s": None, "docs_s": None,
            "instance_type": itype or ("g6.xlarge" if acc == "gpu" else "m8i.4xlarge"),
            "usd_per_hour": price or (0.8048 if acc == "gpu" else 0.8467), "amx": amx if acc == "cpu" else "n/a",
            "ttft_p99": 300.0, "tpot_p50": 40.0 if meets else 400.0, "e2e_p99": e2e, "error_pct": 0.0,
            "quality_flag": None}


SLA = E.SlaInput(p99_ttft_ms=2000, p50_tpot_ms=150)


def test_go_conditional_nogo():
    gpu = _row("gpu", "bf16", 16, 1000.0)
    go = {"rows": [gpu, _row("cpu", "bf16", 16, 400.0)]}
    r = E.estimate(P([M()], sla=SLA), summary=go)["recommendations"]["chat"]
    assert r["verdict"] == "Go" and r["proxy"] == "m1-qwen2.5-3b"
    assert r["capacity_equivalence"]["cpu_instances_per_gpu"] == 3
    # only INT8 meets -> below the requested BF16
    quant = {"rows": [gpu, _row("cpu", "bf16", 16, 400.0, meets=False), _row("cpu", "w8a8", 16, 500.0)]}
    r = E.estimate(P([M()], sla=SLA), summary=quant)["recommendations"]["chat"]
    assert r["verdict"] == "Conditional" and "below the requested bf16" in r["reason"]
    # too many instances per GPU
    slow = {"rows": [gpu, _row("cpu", "bf16", 16, 100.0)]}
    r = E.estimate(P([M()], sla=SLA), summary=slow)["recommendations"]["chat"]
    assert r["verdict"] == "Conditional" and "10 CPU instances" in r["reason"]
    # only at low concurrency
    low = {"rows": [gpu, _row("cpu", "bf16", 4, 400.0), _row("cpu", "bf16", 16, 400.0, meets=False)]}
    r = E.estimate(P([M(required_concurrency=16)], sla=SLA), summary=low)["recommendations"]["chat"]
    assert r["verdict"] == "Conditional" and "concurrency" in r["reason"]
    # nothing valid: AMX not verified
    bad = {"rows": [gpu, _row("cpu", "bf16", 16, 400.0, amx="INVALID")]}
    out = E.estimate(P([M()], sla=SLA), summary=bad)
    assert out["recommendations"]["chat"]["verdict"] == "No-go" and out["overall"] == "No-go"


def test_cost_never_changes_the_verdict():
    gpu = _row("gpu", "bf16", 16, 1000.0)
    expensive = {"rows": [gpu, _row("cpu", "bf16", 16, 400.0, price=50.0)]}
    r = E.estimate(P([M()], sla=SLA), summary=expensive)["recommendations"]["chat"]
    assert r["verdict"] == "Go" and r["hourly_cost_ratio"] > 100


def test_insufficient_data_for_far_sizes_and_unknown_types():
    s = {"rows": [_row("gpu", "bf16", 16, 1000.0)]}
    r = E.estimate(P([M(params_b=70)], sla=SLA), summary=s)["recommendations"]["chat"]
    assert r["verdict"] == "Insufficient data" and "far from" in r["reason"]
    out = E.estimate(P([M(params_b=70)], sla=SLA), summary=s)
    assert any("above the 4B benchmark cap" in x for x in out["risks"])
