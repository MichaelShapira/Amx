"""bench/demand: the cheapest option that meets the targets at a given load, under a pricing basis."""
from __future__ import annotations

import pytest

from bench import demand

CATALOG = {"price_snapshot": "t", "instances": {
    "g6.xlarge": {"accelerator": "gpu", "family": "g6", "vcpus": 4, "memory_gib": 16, "gpus": 1, "gpu_model": "L4", "usd_per_hour": 0.80,
                  "savings_plan_usd_per_hour": {"compute_1y": 0.65, "compute_3y": 0.46, "ec2_instance_1y": 0.52, "ec2_instance_3y": 0.37}},
    "m8i.4xlarge": {"accelerator": "cpu", "family": "m8i", "vcpus": 16, "memory_gib": 64, "gpus": 0, "usd_per_hour": 0.85,
                    "savings_plan_usd_per_hour": {"compute_1y": 0.62, "compute_3y": 0.43, "ec2_instance_1y": 0.56, "ec2_instance_3y": 0.38}},
    "m8i.2xlarge": {"accelerator": "cpu", "family": "m8i", "vcpus": 8, "memory_gib": 32, "gpus": 0, "usd_per_hour": 0.42,
                    "savings_plan_usd_per_hour": {"compute_1y": 0.31}},
    "m8i.xlarge": {"accelerator": "cpu", "family": "m8i", "vcpus": 4, "memory_gib": 16, "gpus": 0, "usd_per_hour": 0.21,
                   "savings_plan_usd_per_hour": {"compute_1y": 0.16}},
    "m8i.large": {"accelerator": "cpu", "family": "m8i", "vcpus": 2, "memory_gib": 8, "gpus": 0, "usd_per_hour": 0.11,
                  "savings_plan_usd_per_hour": {}},
}}


def row(acc, itype, conc, req_s, meets=True, engine="vllm", busy=10.0):
    return {"accelerator": acc, "instance_type": itype, "engine": engine, "precision": "bf16", "conc": conc, "batch": 1,
            "req_s": req_s, "meets_user_sla": meets, "e2e_p99": 100.0, "run_id": "r", "model": "m",
            "cpu_util_mean": busy if acc == "cpu" else None, "gpu_util_mean": busy if acc == "gpu" else None}


ROWS = [row("gpu", "g6.xlarge", 1, 20), row("gpu", "g6.xlarge", 32, 500), row("gpu", "g6.xlarge", 64, 600, meets=False),
        row("cpu", "m8i.4xlarge", 1, 10), row("cpu", "m8i.4xlarge", 32, 120), row("cpu", "m8i.4xlarge", 64, 130, meets=False)]
NAME = lambda r: f"{r['accelerator'].upper()} · {r['instance_type']} · vLLM · bf16"  # noqa: E731


def test_price_per_hour_follows_the_basis_and_the_override():
    od = demand.price_per_hour("m8i.4xlarge", "cpu", None, CATALOG)
    assert od["usd_per_hour"] == 0.85 and od["basis"] == "on_demand" and od["label"] == "On-Demand"
    sp = demand.price_per_hour("m8i.4xlarge", "cpu", {"basis": "compute_3y"}, CATALOG)
    assert sp["usd_per_hour"] == 0.43 and "3 years" in sp["label"]
    missing = demand.price_per_hour("m8i.large", "cpu", {"basis": "compute_3y"}, CATALOG)
    assert missing["usd_per_hour"] == 0.11 and "On-Demand used" in missing["label"]
    own = demand.price_per_hour("m8i.4xlarge", "cpu", {"basis": "compute_3y", "cpu_usd_per_hour": 0.0}, CATALOG)
    assert own["usd_per_hour"] == 0.0 and own["override"] and own["basis"] == "override"
    # an override for the CPU side leaves the GPU side on the basis
    g = demand.price_per_hour("g6.xlarge", "gpu", {"basis": "compute_1y", "cpu_usd_per_hour": 0.0}, CATALOG)
    assert g["usd_per_hour"] == 0.65
    assert "unknown" and demand.price_per_hour("g6.xlarge", "gpu", {"basis": "nonsense"}, CATALOG)["basis"] == "on_demand"


def test_savings_options_list_every_commitment_with_its_saving():
    opts = {o["basis"]: o for o in demand.savings_options("g6.xlarge", CATALOG)}
    assert opts["on_demand"]["usd_per_hour"] == 0.80 and opts["on_demand"]["saving_pct"] == 0
    assert opts["compute_3y"]["usd_per_hour"] == 0.46 and opts["compute_3y"]["saving_pct"] == 43
    assert "compute_1y" not in {o["basis"] for o in demand.savings_options("m8i.large", CATALOG)}


def test_capacity_is_the_highest_rate_that_still_met_the_targets():
    cap = demand.capacity([r for r in ROWS if r["accelerator"] == "cpu"])
    assert cap["req_s"] == 120 and cap["conc"] == 32
    assert demand.capacity([row("cpu", "m8i.4xlarge", 1, 10, meets=False)]) is None


def test_plan_sizes_both_sides_and_finds_the_crossover():
    # 60 requests/minute = 1/s: one small CPU estimate wins; the GPU costs 0.80 whatever the load.
    p = demand.plan(ROWS, 1.0, None, CATALOG, NAME, params=300_000_000)
    assert p["demand_rpm"] == 60
    cheapest = p["cheapest"]
    assert cheapest["side"] == "cpu" and cheapest["estimated"] and cheapest["instance_type"] == "m8i.large" and cheapest["instances"] == 1
    assert cheapest["usd_per_hour"] == pytest.approx(0.11) and cheapest["usd_per_month"] == pytest.approx(0.11 * 730)
    assert p["cheapest_measured"]["instance_type"] == "g6.xlarge" or p["cheapest_measured"]["instance_type"] == "m8i.4xlarge"
    assert p["cheapest_gpu"]["instances"] == 1 and p["cheapest_gpu"]["usd_per_hour"] == pytest.approx(0.80)
    # the measured m8i.4xlarge carries 120/s; its 2-core estimate carries 15/s (120 x 2/8 cores)
    small = next(o for o in p["options"] if o["instance_type"] == "m8i.large")
    assert small["capacity_req_s"] == pytest.approx(15.0) and "estimated" in small["capacity_basis"]
    assert p["crossover"]["kind"] == "crossover" and p["crossover"]["req_s"] > 1.0
    assert "requests/minute" in p["text"] and "estimated size; run it to confirm" in p["text"]
    # heavy load: the GPU wins outright
    heavy = demand.plan(ROWS, 450.0, None, CATALOG, NAME, params=300_000_000)
    assert heavy["cheapest"]["side"] == "gpu" and heavy["cheapest_cpu"]["instances"] >= 4
    # curve covers the GPU's capacity and beyond, with both sides' costs
    assert len(p["curve"]) == 41 and p["curve"][0]["rpm"] == 0 and p["curve"][-1]["rpm"] > 500 * 60 and all("gpu_usd_per_hour" in c for c in p["curve"])


def test_plan_without_estimates_or_demand():
    assert demand.plan(ROWS, None, None, CATALOG, NAME) is None
    p = demand.plan(ROWS, 1.0, None, CATALOG, NAME, include_estimates=False)
    assert all(not o["estimated"] for o in p["options"]) and p["cheapest"]["instance_type"] == "g6.xlarge"
    # memory check: a 20B-parameter model (40 GB of weights) rules every smaller m8i size out
    big = demand.plan(ROWS, 1.0, None, CATALOG, NAME, params=20_000_000_000)
    assert {o["instance_type"] for o in big["options"] if o["side"] == "cpu"} == {"m8i.4xlarge"}


def test_plan_applies_the_pricing_basis_to_both_sides():
    p = demand.plan(ROWS, 450.0, {"basis": "compute_3y"}, CATALOG, NAME, include_estimates=False)
    assert p["cheapest_gpu"]["usd_per_hour"] == pytest.approx(0.46)
    assert p["cheapest_cpu"]["usd_per_hour"] == pytest.approx(4 * 0.43)
    free = demand.plan(ROWS, 450.0, {"cpu_usd_per_hour": 0.0}, CATALOG, NAME, include_estimates=False)
    assert free["cheapest"]["side"] == "cpu" and free["cheapest"]["usd_per_hour"] == 0.0
    assert free["crossover"]["kind"] == "cpu_cheaper_everywhere"


def test_curve_is_json_safe_when_a_side_has_no_measured_option():
    import json
    gpu_only = [r for r in ROWS if r["accelerator"] == "gpu"]
    p = demand.plan(gpu_only, 1.0, None, CATALOG, NAME)
    json.dumps(p)                                   # no NaN anywhere (the API serializer rejects it)
    assert all(c["cpu_measured_usd_per_hour"] is None and c["cpu_usd_per_hour"] is None for c in p["curve"])
    assert p["crossover"] is None and p["cheapest"]["side"] == "gpu"
