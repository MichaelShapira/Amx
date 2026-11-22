"""The decision suite: generated agent decisions, the Strands Decider payload, scoring, plan, cell spec."""
from __future__ import annotations

import collections
import json
from pathlib import Path

from bench import cellspec, decision, plan
from bench.config import load_models, load_run, load_targets, resolve_cells

ROOT = Path(__file__).resolve().parents[1]
MODELS = load_models(ROOT / "config/models.yaml")
TARGETS = load_targets(ROOT / "config/targets.yaml")
TMAP = TARGETS.resolved()


def test_generated_items_are_balanced_deterministic_and_first_person():
    short = decision.generate(300, "short")
    long_ = decision.generate(120, "long")
    assert len(short) == 300 and len(long_) == 120
    assert collections.Counter(x["gold"]["route"] for x in short) == {t: 60 for t in decision.TEAMS}
    assert collections.Counter(x["gold"]["urgent"] for x in short) == {True: 150, False: 150}
    assert set(x["gold"]["severity"] for x in short) == set(decision.SEVERITY)
    words_s = [len(x["state"].split()) for x in short]
    words_l = [len(x["state"].split()) for x in long_]
    assert 20 <= min(words_s) and max(words_s) <= 60 and 180 <= min(words_l) and max(words_l) <= 300
    assert decision.dataset_sha256(short) == decision.dataset_sha256(decision.generate(300, "short"))
    assert decision.dataset_sha256(short) != decision.dataset_sha256(decision.generate(300, "short", seed=1))
    assert not any(" I says " in x["state"] or " I wants " in x["state"] or " I asks " in x["state"] for x in short)
    assert all(set(x["questions"]) == {"route", "urgent", "severity"} for x in short)


def test_request_body_is_the_servers_shape():
    it = decision.generate(5, "short")[0]
    b1 = decision.request_body(it, 1)
    assert b1["state"] == it["state"] and list(b1["questions"]) == ["route"]
    assert b1["questions"]["route"]["type"] == "choice" and set(b1["questions"]["route"]["criteria"]) == set(decision.TEAMS)
    b3 = decision.request_body(it, 3)
    assert list(b3["questions"]) == ["route", "urgent", "severity"]
    assert b3["questions"]["urgent"]["type"] == "noul" and "criteria" not in b3["questions"]["urgent"]
    assert b3["questions"]["severity"] == {"type": "score", "instructions": "How severe is the problem described?", "criteria": list(decision.SEVERITY)}
    assert decision.request_body(it, 9) == b3 and list(decision.request_body(it, 0)["questions"]) == ["route"]
    json.dumps(b3)


def _resp(route, urgent_p, sev_level, conf=0.9):
    probs = {t: 0.02 for t in decision.TEAMS}; probs[route] = 0.92
    return {"answers": {"route": {"type": "choice", "choice": route, "confidence": conf, "probabilities": probs},
                        "urgent": {"type": "noul", "noul": urgent_p},
                        "severity": {"type": "score", "score": float(decision.SEVERITY.index(sev_level)), "confidence": conf,
                                     "probabilities": {s: (0.85 if s == sev_level else 0.05) for s in decision.SEVERITY}}},
            "usage": {"input_tokens": 120, "output_tokens": 1}, "latency_ms": 140.0}


def test_scoring_against_gold_and_against_the_reference_cell():
    items = decision.generate(20, "short")
    by_id = {x["id"]: x for x in items}
    # the GPU gets 18 of 20 routes right, every urgency, severity off by one on two items
    gpu = []
    for i, it in enumerate(items):
        g = it["gold"]
        route = g["route"] if i < 18 else next(t for t in decision.TEAMS if t != g["route"])
        sev = g["severity"] if i >= 2 else decision.SEVERITY[(decision.SEVERITY.index(g["severity"]) + 1) % 4]
        gpu.append({"item_id": it["id"], "output": _resp(route, 0.9 if g["urgent"] else 0.1, sev)})
    ref = decision.score_cell(gpu, by_id)
    assert ref["n"] == 20 and ref["accuracy"]["route"] == 0.9 and ref["accuracy"]["urgent"] == 1.0 and ref["accuracy"]["severity"] == 0.9
    assert abs(ref["accuracy_overall"] - (18 + 20 + 18) / 60) < 1e-9 and ref["confidence_mean"] == 0.9
    # the CPU answers the same except one route and a slightly different urgency probability
    cpu = [dict(r) for r in gpu]
    cpu[0] = {"item_id": items[0]["id"], "output": _resp("sales" if items[0]["gold"]["route"] != "sales" else "billing",
                                                        0.86 if items[0]["gold"]["urgent"] else 0.14,
                                                        decision.SEVERITY[(decision.SEVERITY.index(items[0]["gold"]["severity"]) + 1) % 4])}
    sc = decision.score_cell(cpu, by_id, ref["answers"])
    vs = sc["vs_reference_cell"]
    assert vs["n"] == 60 and abs(vs["agreement"] - 59 / 60) < 1e-9 and vs["prob_gap_max"] >= 0.9 - 0.02
    assert decision.flag(sc, precision="bf16") is None
    sc["vs_reference_cell"]["agreement"] = 0.9
    assert "agreement with the GPU's decisions 0.900" in decision.flag(sc, precision="bf16")
    # records with errors or no answers are skipped, not counted
    assert decision.score_cell([{"item_id": items[0]["id"], "output": None}], by_id)["n"] == 0


def test_decision_run_plans_and_builds_cells_on_both_sides():
    run = load_run(ROOT / "config/runs/decision-smoke.yaml")
    cells = resolve_cells(run, MODELS, TARGETS)
    assert {(c.target, c.engine, c.precision) for c in cells} == {("t1-g6", "decider", "bf16"), ("t2-m8i-4xl", "decider", "bf16")}
    m = MODELS.get("d1-strands-decider-e4b")
    assert m.type == "decision" and m.decision.base_repo == "google/gemma-4-E4B-it" and plan.api_path(m, "decider") == "/v1/systemone"
    assert plan.api_task(m) == "decide"
    sizes = {plan.decision_pool(m, "short"): 600}
    levels = plan.levels_for_cell(run, cells[0], m, run_id="r", bucket="b", pool_sizes=sizes)
    acc = [lv for lv in levels if lv["accuracy"]]
    perf = [lv for lv in levels if not lv["accuracy"]]
    assert len(acc) == 1 and acc[0]["payload"]["questions"] == 3 and acc[0]["pool"]["size"] == 300 and acc[0]["api_task"] == "decide"
    assert len(perf) == len(run.concurrency) * len(run.decision.questions) * run.repeats
    assert {lv["payload"]["questions"] for lv in perf} == {1, 3} and {lv["batch"] for lv in perf} == {1, 3}
    for c in cells:
        spec = cellspec.build(c, m, TMAP[c.target], run, run_id="r", bucket="b", hf_token_secret="s",
                              ecr_repo_uri="1.dkr.ecr.us-east-1.amazonaws.com/x", built_tags={"decider-cpu": "c", "decider-cuda": "g"},
                              instance_type=TMAP[c.target].instance_type)
        opts = spec["container"]["run_opts"]
        assert spec["api_task"] == "decide" and spec["health"]["probe"]["path"] == "/v1/systemone" and spec["health"]["path"] == "/health"
        assert spec["base_weights"] == {"repo": "google/gemma-4-E4B-it", "revision": m.decision.base_revision, "gated": False}
        assert "DECIDER_CHECKPOINT={MODEL_DIR}" in opts and "DECIDER_BASE_DIR={BASE_DIR}" in opts
        assert ("DECIDER_DEVICE=cuda" in opts) == (c.target == "t1-g6")
        assert spec["container"]["image"].endswith(":g" if c.target == "t1-g6" else ":c")
        if c.target != "t1-g6":
            assert "OMP_NUM_THREADS=8" in opts
    # the server's health probe body is a real systemone request
    body = cellspec.health(cells[0], m)["probe"]["body"]
    assert body["questions"]["urgent"]["type"] == "noul" and body["state"]


def test_decision_suite_refuses_other_models_and_wrong_prompt_set():
    import pytest
    from bench.config import RunConfig
    base = dict(name="t", account="1", suite="decision", models=["m1-qwen3-1.7b"], prompt_set="short",
                cells=[{"target": "t1-g6", "precision": "bf16"}], concurrency=[1], per_level={"requests": 50}, repeats=1)
    with pytest.raises(ValueError, match="not part of the decision suite"):
        resolve_cells(RunConfig.model_validate(base), MODELS, TARGETS)
    with pytest.raises(ValueError, match="short .* or long"):
        RunConfig.model_validate({**base, "models": ["d1-strands-decider-e4b"], "prompt_set": "medium"})
