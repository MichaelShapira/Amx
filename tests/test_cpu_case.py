"""bench/cpu_case: the computed statement of where the CPU is the better choice, and where it is not."""
from __future__ import annotations

from bench import cpu_case
from tests.test_demand import CATALOG, NAME, ROWS, row


def _summary(caps: dict, requested: dict, rows: list[dict]) -> dict:
    rows = [{**r, "target": r.get("target") or ("t1-g6" if r["accelerator"] == "gpu" else "t2-m8i-4xl")} for r in rows]
    return {"run_id": "r", "run": {"instance_types": requested}, "rows": rows, "capacity_errors": caps}


def test_cost_wins_names_the_range_and_worked_examples():
    group = {"label": "Q", "key": "q"}
    c = cpu_case.cost_wins(group, ROWS, None, CATALOG, NAME, 300_000_000)
    assert c["kind"] == "cpu_below_crossover" and c["crossover_rpm"] and c["crossover_rpm"] > 60
    assert "cheaper below about" in c["text"] and "estimated size" in c["text"] and "GPU's throughput per dollar wins" in c["text"]
    labels = [e["label"] for e in c["examples"]]
    assert labels and labels[0].startswith("Pilot") and all(e["ratio"] > 1 for e in c["examples"])
    # a group where no CPU row met the targets
    gpu_only = [r for r in ROWS if r["accelerator"] == "gpu"] + [row("cpu", "m8i.4xlarge", 1, 10, meets=False)]
    c2 = cpu_case.cost_wins(group, gpu_only, None, CATALOG, NAME, None)
    assert c2["kind"] == "gpu_only" and "only option" in c2["text"]
    # a free CPU (capacity already paid for) is cheaper everywhere
    c3 = cpu_case.cost_wins(group, ROWS, {"cpu_usd_per_hour": 0.0}, CATALOG, NAME, None)
    assert c3["kind"] == "cpu_everywhere"


def test_availability_follows_the_capacity_log_by_market():
    rows = [row("gpu", "g6.xlarge", 1, 20), row("cpu", "m8i.4xlarge", 1, 10)]
    rows[0]["target"], rows[1]["target"] = "t1-g6", "t2-m8i-4xl"
    spot = {"code": "InsufficientInstanceCapacity", "market": "spot"}
    od = {"code": "InsufficientInstanceCapacity", "market": "on-demand"}
    # GPU refused On-Demand too, CPU only on Spot, GPU landed on a fallback type: a real benefit
    a = cpu_case.availability([_summary({"t1-g6": [spot] * 9 + [od] * 9, "t2-m8i-4xl": [spot] * 8},
                                        {"t1-g6": "g6.xlarge", "t2-m8i-4xl": "m8i.4xlarge"},
                                        [{**rows[0], "instance_type": "g5.xlarge"}, rows[1]])])
    assert a["kind"] == "gpu_scarcer" and a["gpu_refusals"] == 18 and a["cpu_refusals"] == 8
    assert "9 On-Demand" in a["text"] and "g6.xlarge -> g5.xlarge" in a["text"] and "all on Spot" in a["text"]
    # Spot-only refusals on both sides: a mild edge at most
    b = cpu_case.availability([_summary({"t1-g6": [spot] * 3, "t2-m8i-4xl": [spot]}, {}, rows)])
    assert b["kind"] == "gpu_scarcer_spot" and "mild" in b["text"]
    # nothing refused: no evidence either way
    c = cpu_case.availability([_summary({}, {}, rows)])
    assert c["kind"] == "none" and "no evidence" in c["text"]
    # CPU refused more: no advantage
    d = cpu_case.availability([_summary({"t2-m8i-4xl": [od] * 2}, {}, rows)])
    assert d["kind"] == "no_advantage"


def test_build_assembles_headline_wins_losses_and_not_won():
    groups = [{"label": "Q1", "key": "q1", "verdict": {"instances_per_gpu": 5, "hourly_cost_ratio": 5.3}},
              {"label": "Q2", "key": "q2", "verdict": {"instances_per_gpu": 8, "hourly_cost_ratio": 8.5}}]
    gpu_only = [r for r in ROWS if r["accelerator"] == "gpu"] + [row("cpu", "m8i.4xlarge", 1, 10, meets=False)]
    summaries = [_summary({}, {}, ROWS)]
    w = cpu_case.build(groups, {"q1": ROWS, "q2": gpu_only}, summaries, None, CATALOG, NAME, {"m": 300_000_000})
    assert w["any_benefit"] and w["headline"].startswith("The CPU is the better choice for 1 of 2 questions")
    assert [x["key"] for x in w["wins"]] == ["q1"] and [x["key"] for x in w["losses"]] == ["q2"]
    assert w["granularity"] and w["granularity"]["cpu"] == "m8i.large" and w["granularity"]["gpu"] == "g6.xlarge" and w["granularity"]["ratio"] > 7
    assert len(w["not_won"]["peak_throughput"]) == 2 and "GPU wins every question" in w["not_won"]["text"]
    assert {x["title"] for x in w["structural"]} == {"Capacity you already pay for", "Same fleet, same tooling"}
    # no benefit anywhere: say so
    w2 = cpu_case.build(groups[1:], {"q2": gpu_only}, summaries, None, CATALOG, NAME, {})
    assert not w2["any_benefit"] and "no benefit" in w2["headline"]
