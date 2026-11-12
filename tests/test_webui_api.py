"""The local API behind the React UI: catalog, data previews, proposals with reasons, wizard ->
run config, and the comparability rules. Offline (no AWS)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from webui.api import compare  # noqa: E402
from webui.api.main import app  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
c = TestClient(app)
GPU = {"hardware": "t1-g6", "instance_type": "g6.xlarge", "precisions": ["bf16"]}
CPU = {"hardware": "t2-m8i-4xl", "instance_type": "m8i.4xlarge", "precisions": ["bf16"]}


def test_catalog_speaks_plain_language():
    cat = c.get("/api/catalog").json()
    assert {m["id"] for m in cat["models"]} >= {"m1-qwen2.5-3b", "d2-roblox-pii-v2"}
    m = next(x for x in cat["models"] if x["id"] == "m1-qwen2.5-3b")
    assert m["name"] == "Qwen2.5 3B Instruct" and m["commercial_ok"] is False and m["batch_capable"] is False
    d2 = next(x for x in cat["models"] if x["id"] == "d2-roblox-pii-v2")
    assert d2["batch_capable"] is True and d2["use_cases"] == ["dspm"]
    hw = {h["id"]: h for h in cat["hardware"]}
    assert hw["t1-g6"]["accelerator"] == "gpu" and hw["t2-m8i-4xl"]["software"] == ["vLLM"]
    assert "t2-c8i-4xl" not in hw          # same software as t2-m8i-4xl: one option, the size is chosen separately
    assert cat["instances"]["g6.xlarge"]["gpu_model"] == "L4" and cat["instances"]["m8i.4xlarge"]["amx"] is True
    for k in ("use_cases", "prompt_sets", "workloads", "presets", "depths", "request_modes", "purchase"):
        for v in cat[k].values():
            assert any(key in v for key in ("help", "long"))


@pytest.mark.skipif(not (ROOT / "data/dspm/v1/test.jsonl.gz").exists(), reason="dataset not built")
def test_data_previews_show_documents_with_expected_answers():
    d = c.get("/api/data/dspm").json()
    assert d["available"] and d["documents"] == 3000
    assert d["workloads"]["W1"]["documents"] == 1400 and d["workloads"]["W2"]["documents"] > 0
    s = c.get("/api/data/dspm/sample", params={"workload": "W1", "label": "SENSITIVE", "n": 3}).json()
    assert len(s) == 3 and all(x["label"] == "SENSITIVE" for x in s) and all(x["values"] for x in s)
    s2 = c.get("/api/data/dspm/sample", params={"workload": "W2", "n": 2}).json()
    assert all(x["label"] in ("GOV_ID", "PERSONAL", "FINANCIAL", "HEALTH", "CREDENTIALS", "CONFIDENTIAL", "NONE") for x in s2)
    std = c.get("/api/data/standard").json()
    if std["available"]:
        p = c.get("/api/data/standard/sample", params={"prompt_set": "short", "model": "m1-qwen2.5-3b", "n": 2}).json()
        assert len(p) == 2 and p[0]["tokens"] > 100


def test_proposals_carry_reasons_and_fairness():
    p = c.post("/api/propose", json={"models": ["m2-qwen3-4b-2507"], "cpu_type": "m8i.4xlarge"}).json()
    assert p["gpu"][0]["instance_type"] == "g6.xlarge" and p["gpu"][0]["fit"] == "recommended"
    kinds = {r["kind"] for r in p["gpu"][0]["reasons"]}
    assert {"memory", "gpus", "price", "generation"} <= kinds
    assert any("7.5 GiB" in r["text"] for r in p["gpu"][0]["reasons"])
    big = c.post("/api/propose", json={"models": ["m2-qwen3-4b-2507"], "gpu_type": "g6.48xlarge"}).json()
    # against an 8-GPU box the fair CPU counterparts are the 48xlarge CPUs: price band, not size, decides
    assert all(x["spec"]["vcpus"] >= 128 for x in big["cpu"][:3])
    assert big["pairing"] is None
    both = c.post("/api/propose", json={"models": ["m1-qwen2.5-3b"], "gpu_type": "g6.xlarge", "cpu_type": "m8i.4xlarge"}).json()
    assert both["pairing"]["fair"] is True


def test_plan_builds_a_valid_run_and_refuses_unfair_pairs():
    w = {"use_case": "standard", "models": ["m1-qwen2.5-3b"], "gpu": GPU, "cpus": [CPU], "request_mode": "concurrent",
         "concurrency": [1, 10], "depth": "full", "sla": {"p99_ttft_ms": 2000, "p50_tpot_ms": 150}}
    pl = c.post("/api/plan", json=w).json()
    assert pl["ok"] and pl["grade"] == "full" and pl["request_mode"] == "concurrent"
    assert pl["run"]["concurrency"] == [1, 10] and pl["run"]["per_level"] == {"requests": 200} and pl["run"]["repeats"] == 3
    assert pl["run"]["purchase"]["type"] == "spot" and pl["run"]["execution"]["parallel_targets"] is True
    assert pl["estimate"]["usd"]["total"] > 0 and len(pl["series"]) == 2
    single = c.post("/api/plan", json={**w, "request_mode": "single"}).json()
    assert single["run"]["concurrency"] == [1] and single["request_mode"] == "single"
    bad = c.post("/api/plan", json={**w, "gpu": {**GPU, "instance_type": "g6.48xlarge"}}).json()
    assert not bad["ok"] and bad["problems"][0].startswith("Unfair pairing")
    allowed = c.post("/api/plan", json={**w, "gpu": {**GPU, "instance_type": "g6.48xlarge"}, "allow_unfair": True}).json()
    assert allowed["ok"] and allowed["pairing"][0]["fair"] is False
    seq = c.post("/api/plan", json={**w, "parallel_targets": False}).json()
    assert seq["run"]["execution"]["parallel_targets"] is False


def test_batch_mode_only_for_classifiers():
    w = {"use_case": "dspm", "models": ["d2-roblox-pii-v2"], "workloads": ["W3"], "gpu": GPU, "cpus": [CPU],
         "request_mode": "batch", "batch_sizes": [8, 32], "concurrency": [4]}
    pl = c.post("/api/plan", json=w).json()
    assert pl["ok"] and pl["request_mode"] == "batch" and pl["run"]["batch_sizes"] == [8, 32]
    gen = c.post("/api/plan", json={**w, "models": ["m1-qwen2.5-3b"]}).json()
    assert gen["request_mode"] == "batch"          # the config carries it; the planner gives chat models one prompt per request


def _summary(run: dict, rows: list[dict], **extra) -> dict:
    return {"run_id": run["name"], "run": run, "rows": rows, "smoke": run.get("mode") == "smoke", **extra}


def test_comparability_rules():
    base = {"name": "a", "suite": "standard", "mode": "smoke", "concurrency": [1, 4, 16], "prompt_set": "short",
            "sampling": {"max_new_tokens": 128, "temperature": 0.0}, "per_level": {"requests": 50}, "repeats": 1}
    a = _summary(base, [])
    b = _summary({**base, "name": "b"}, [])
    assert compare.comparability([a, b])["ok"]
    assert compare.request_mode({**base, "concurrency": [1]}) == "single"
    assert compare.request_mode({**base, "batch_sizes": [8]}) == "batch"
    r = compare.comparability([a, _summary({**base, "name": "c", "concurrency": [1]}, [])])
    assert not r["ok"] and "request shapes" in r["blockers"][0]
    r = compare.comparability([a, _summary({**base, "name": "d", "prompt_set": "long"}, [])])
    assert not r["ok"] and "input lengths" in r["blockers"][0]
    r = compare.comparability([a, _summary({**base, "name": "e", "sampling": {"max_new_tokens": 256}}, [])])
    assert not r["ok"] and "answer lengths" in r["blockers"][0]
    r = compare.comparability([a, _summary({**base, "name": "f", "mode": "full", "repeats": 3, "per_level": {"requests": 200}}, [])])
    assert r["ok"] and any("quick check" in w for w in r["warnings"])
    r = compare.comparability([a, _summary({**base, "name": "g", "suite": "dspm"}, [])])
    assert not r["ok"] and "use cases" in r["blockers"][0]
    d1 = _summary({**base, "name": "h", "suite": "dspm", "dspm": {"workloads": ["W1", "W3"]}}, [], dspm_hashes={"dataset_sha256": "x" * 40, "policy_sha256": "p" * 40})
    d2 = _summary({**base, "name": "i", "suite": "dspm", "dspm": {"workloads": ["W1"]}}, [], dspm_hashes={"dataset_sha256": "y" * 40, "policy_sha256": "p" * 40})
    r = compare.comparability([d1, d2])
    assert not r["ok"] and "dataset versions" in r["blockers"][0]
    d3 = _summary({**base, "name": "j", "suite": "dspm", "dspm": {"workloads": ["W1"]}}, [], dspm_hashes={"dataset_sha256": "x" * 40, "policy_sha256": "p" * 40})
    r = compare.comparability([d1, d3])
    assert r["ok"] and any("scenarios all runs share" in w for w in r["warnings"])


def test_analysis_verdict_and_equivalence():
    def row(acc, itype, conc, thr, ttft, price):
        return {"model": "m1-qwen2.5-3b", "target": "t", "accelerator": acc, "instance_type": itype, "engine": "vllm", "precision": "bf16",
                "kind": "generation", "workload": None, "bucket": None, "batch": 1, "conc": conc, "agg_out_tok_s": thr,
                "ttft_p99": ttft, "tpot_p50": 30.0, "e2e_p99": 3000, "e2e_p50": 2900, "usd_per_hour": price,
                "usd_per_1m_tokens": price / (thr * 3.6), "error_pct": 0.0, "amx": "VERIFIED" if acc == "cpu" else None,
                "cell_key": "ck", "quality_delta": 0.2 if acc == "cpu" else None, "req_s": 1.0}
    rows = [row("gpu", "g6.xlarge", 16, 500.0, 300.0, 0.80), row("cpu", "m8i.4xlarge", 16, 170.0, 1500.0, 0.85),
            row("cpu", "m8i.4xlarge", 4, 50.0, 400.0, 0.85)]
    s = _summary({"name": "r", "suite": "standard", "mode": "smoke", "concurrency": [4, 16], "sla": {"p99_ttft_ms": 2000}}, rows)
    out = compare.analyze([s], {"p99_ttft_ms": 2000, "max_error_pct": 1.0}, limit=4)
    g = out["groups"][0]
    assert g["best_gpu"]["instance_type"] == "g6.xlarge" and g["best_cpu"]["conc"] == 16
    assert g["verdict"]["instances_per_gpu"] == 3 and g["verdict"]["within_limit"] is True
    assert "To match one g6.xlarge you need 3 × m8i.4xlarge" in g["verdict"]["text"]
    assert [e["instances_per_gpu"] for e in g["equivalence"]] == [3, 10]
    strict = compare.analyze([s], {"p99_ttft_ms": 1000, "max_error_pct": 1.0}, limit=4)["groups"][0]
    assert strict["best_cpu"]["conc"] == 4 and strict["verdict"]["instances_per_gpu"] == 10 and strict["verdict"]["within_limit"] is False
    assert out["pairing"][0]["fair"] is True


CAPS = [{"time": "2026-11-11T09:38:00Z", "code": "InsufficientInstanceCapacity", "market": "spot", "instance_type": "g6.2xlarge"},
        {"time": "2026-11-11T09:52:00Z", "code": "InsufficientInstanceCapacity", "market": "on-demand", "instance_type": "g5.xlarge"},
        {"time": "2026-11-11T10:09:00Z", "code": "InsufficientInstanceCapacity", "market": "spot", "instance_type": "g5.4xlarge"}]


def test_a_side_that_never_got_an_instance_says_did_not_run_and_why(monkeypatch):
    """The 2026-11-11 run: GPU refused 25 times, run finished, UI said 'Not run' with no reason
    and the instance row still said 'Waiting for capacity'."""
    from webui.api import main

    s3 = {"runs/r1/capacity/t1-g6.json": CAPS, "runs/r1/targets/t2-m8i-4xl.json": [
        {"time": "2026-11-11T09:38:02Z", "event": "launched", "instance_id": "i-1", "instance_type": "m8i.8xlarge",
         "lifecycle": "spot", "requested": "spot"}],
          "runs/r1/resolved.json": {"purchase": {"type": "spot", "max_launch_attempts": 30, "spot_capacity_window_s": 600}}}
    monkeypatch.setattr(main, "_s3_json", lambda key: s3.get(key))
    monkeypatch.setattr(main, "_instance_states", lambda ids: {})
    main._CACHE.pop("purchase:r1", None)
    hw = {h["id"]: h for h in main.get_catalog()["hardware"]}
    # Finished run: the GPU side did not run, in words.
    gpu, cpu = main.target_status("r1", ["t1-g6", "t2-m8i-4xl"], True, hw)
    assert gpu["did_not_run"] is True and gpu["state_label"] == "Did not run" and gpu["attempts_max"] == 30
    assert gpu["status"].startswith("Did not run. AWS had no capacity: it refused 3 launch request(s) over 31 minutes for "
                                    "g5.4xlarge, g5.xlarge, g6.2xlarge (On-Demand and Spot). The run gave up after its limit of 30 attempts")
    assert "Waiting" not in gpu["status"]
    assert cpu["did_not_run"] is False and cpu["state"] == "gone"
    cells = [{"model": "RoBERTa", "hardware": "t1-g6", "status": "planned", "error": ""},
             {"model": "RoBERTa", "hardware": "t2-m8i-4xl", "status": "ready", "error": ""}]
    assert main.derive_cell_status(cells[0], cells, gpu, "complete") == "Not run — no capacity"
    notes = main.run_notices([gpu, cpu], cells, "complete")
    assert len(notes) == 1 and notes[0]["kind"] == "bad" and notes[0]["title"] == "The GPU side did not run"
    assert "no GPU measurement to compare them with" in notes[0]["text"] and "raise the number of attempts" in notes[0]["text"]
    # Still running: waiting, with the attempt budget, and a warning notice rather than a failure.
    gpu_running = main.target_status("r1", ["t1-g6"], False, hw)[0]
    assert gpu_running["did_not_run"] is False
    assert gpu_running["status"] == ("Waiting for capacity: AWS refused 3 launch request(s) so far for g5.4xlarge, g5.xlarge, g6.2xlarge "
                                     "(On-Demand and Spot) — the run gives up after 30 attempts")
    assert main.derive_cell_status(cells[0], cells, gpu_running, "running") == "Waiting for an instance (capacity)"
    assert main.run_notices([gpu_running], cells, "running")[0]["kind"] == "warn"
    # A gave_up event written by the Lambda wins over the derived sentence, even mid-run.
    s3["runs/r1/targets/t1-g6.json"] = [{"time": "2026-11-11T10:10:00Z", "event": "gave_up", "attempts": 30, "reason": "No capacity: AWS refused 25 launch request(s)."}]
    g = main.target_status("r1", ["t1-g6"], False, hw)[0]
    assert g["did_not_run"] is True and g["status"] == "Did not run. No capacity: AWS refused 25 launch request(s)."
    main._CACHE.pop("purchase:r1", None)


def test_comparison_marks_a_missing_side_instead_of_pretending_to_compare():
    def row(acc, itype, conc, thr):
        return {"model": "d1", "target": "t2-m8i-4xl", "accelerator": acc, "instance_type": itype, "engine": "vllm", "precision": "bf16",
                "kind": "classifier", "workload": None, "bucket": None, "batch": 1, "conc": conc, "cls_s": thr, "req_s": thr,
                "ttft_p99": None, "tpot_p50": None, "e2e_p99": 300.0, "e2e_p50": 200.0, "usd_per_hour": 1.7,
                "usd_per_1m_items": 1.7 / (thr * 3.6), "error_pct": 0.0, "amx": "VERIFIED", "cell_key": "ck", "quality_delta": None}
    run = {"name": "r", "suite": "standard", "mode": "smoke", "concurrency": [1, 10], "sla": {"p99_e2e_ms": 1000},
           "cells": [{"target": "t1-g6"}, {"target": "t2-m8i-4xl"}], "instance_types": {"t1-g6": "g6.2xlarge"},
           "purchase": {"type": "spot", "max_launch_attempts": 30}}
    s = _summary(run, [row("cpu", "m8i.8xlarge", 1, 20.0), row("cpu", "m8i.8xlarge", 10, 40.0)])
    s["capacity_errors"] = {"t1-g6": CAPS}
    s.pop("not_run", None)                                   # an older report: derived from cells + capacity log
    nr = compare.not_run(s)
    assert len(nr) == 1 and nr[0]["side"] == "GPU" and nr[0]["instance_type"] == "g6.2xlarge" and nr[0]["refusals"] == 3
    assert nr[0]["reason"].startswith("No capacity: AWS refused 3 launch request(s) for g5.4xlarge, g5.xlarge, g6.2xlarge (On-Demand and Spot)")
    out = compare.analyze([s], {"p99_e2e_ms": 1000, "max_error_pct": 1.0}, limit=4)
    g = out["groups"][0]
    assert out["not_run"] == nr and g["missing"]["gpu"] == nr[0]["reason"] and g["missing"]["cpu"] is None
    assert g["best_gpu"] is None and g["verdict"]["missing_side"] == "gpu"
    assert g["verdict"]["text"].startswith("The GPU side did not run (no capacity), so there is nothing to compare")
    # A newer report carries not_run itself; it is taken as is.
    s2 = {**s, "not_run": [{"target": "t1-g6", "side": "GPU", "instance_type": "g6.2xlarge", "refusals": 25, "reason": "written by the report"}]}
    assert compare.not_run(s2)[0]["reason"] == "written by the report" and compare.not_run(s2)[0]["run_id"] == s["run_id"]
    # Nothing missing when every planned side measured something.
    full = _summary(run, [row("cpu", "m8i.8xlarge", 1, 20.0), {**row("gpu", "g6.2xlarge", 1, 80.0), "target": "t1-g6", "amx": None}])
    full.pop("not_run", None)
    assert compare.not_run(full) == [] and compare.analyze([full], {"p99_e2e_ms": 1000, "max_error_pct": 1.0})["groups"][0]["missing"] == {"gpu": None, "cpu": None}


def test_plan_survives_a_hardware_option_that_cannot_serve_the_format():
    """The Review page crashed ('reading map') when the OpenVINO option was ticked with a chat
    model in BF16: resolve_cells raised and the error reply lacked series/pairing/warnings. Now
    the format is dropped with a note, the option is left out of instance_types, and every reply
    carries every key."""
    OV = {"hardware": "t3-m8i-4xl-ov", "instance_type": "m8i.4xlarge", "precisions": ["bf16"]}
    w = {"use_case": "standard", "models": ["m1-qwen2.5-3b"], "gpu": GPU, "cpus": [CPU, OV], "request_mode": "concurrent", "concurrency": [1, 10]}
    pl = c.post("/api/plan", json=w).json()
    assert pl["ok"] is True and {"series", "pairing", "warnings", "problems", "estimate"} <= set(pl)
    assert {s["hardware"] for s in pl["series"]} == {"t1-g6", "t2-m8i-4xl"}
    assert any("runs nothing in this plan" in x and "INT8 compressed (OpenVINO)" in x for x in pl["warnings"])
    assert "t3-m8i-4xl-ov" not in pl["run"]["instance_types"]
    # The OpenVINO option with a format it can serve takes part.
    pl2 = c.post("/api/plan", json={**w, "cpus": [CPU, {**OV, "precisions": ["bf16", "int8-ov"]}]}).json()
    assert pl2["ok"] is True and {s["hardware"] for s in pl2["series"]} == {"t1-g6", "t2-m8i-4xl", "t3-m8i-4xl-ov"}
    assert any("not available in BF16" in x for x in pl2["warnings"])
    # Nothing can serve it at all -> a problem, still with every key.
    pl3 = c.post("/api/plan", json={**w, "models": ["m5-flan-t5-xl"], "gpu": None, "cpus": [CPU]}).json()
    assert pl3["ok"] is False and pl3["series"] == [] and pl3["pairing"] == [] and any("No hardware option can serve" in p for p in pl3["problems"])


def _backend(supports: bool, commit="b75d598bd40e1fe19fc155d47d2b0f34f4ca09f0"):
    return {"commit": commit, "local_commit": "head", "supports": {"max_launch_attempts": supports}, "behind": not supports,
            "error": None, "missing": [] if supports else ["max_launch_attempts"]}


def test_plan_carries_the_retry_budget_and_spot_wait(monkeypatch):
    from webui.api import main

    monkeypatch.setattr(main, "deployed_backend", lambda: _backend(True))
    w = {"use_case": "standard", "models": ["m1-qwen2.5-3b"], "gpu": GPU, "cpus": [CPU], "purchase": "spot",
         "spot_fallback": True, "spot_wait_minutes": 25, "launch_attempts": 60}
    pl = c.post("/api/plan", json=w).json()
    assert pl["run"]["purchase"] == {"type": "spot", "spot_fallback_on_demand": True, "spot_capacity_window_s": 1500, "max_launch_attempts": 60}
    assert c.post("/api/plan", json={**w, "launch_attempts": 0}).json()["run"]["purchase"]["max_launch_attempts"] == 1
    d = c.post("/api/plan", json={"use_case": "standard", "models": ["m1-qwen2.5-3b"], "gpu": GPU, "cpus": [CPU]}).json()
    assert d["run"]["purchase"]["max_launch_attempts"] == 30 and d["run"]["purchase"]["spot_capacity_window_s"] == 600


def test_plan_leaves_out_settings_the_deployed_backend_does_not_know(monkeypatch):
    """2026-11-12: the deployed preflight (commit b75d598) forbade purchase.max_launch_attempts;
    every run failed before the pre-checks and vanished from the UI. The plan now omits the
    field for such a backend and says so."""
    from webui.api import main

    monkeypatch.setattr(main, "deployed_backend", lambda: _backend(False))
    w = {"use_case": "standard", "models": ["m1-qwen2.5-3b"], "gpu": GPU, "cpus": [CPU], "launch_attempts": 60}
    pl = c.post("/api/plan", json=w).json()
    assert pl["ok"] is True and "max_launch_attempts" not in pl["run"]["purchase"]
    assert any("predates the 'give up after N failed launch attempts' setting" in x and "instead of your 60 attempts" in x
               and "cdk deploy AmxBench-Core" in x for x in pl["warnings"])
    # The default value is silently equivalent (30 attempts ~ 30 minutes): no 'instead of' clause.
    pl2 = c.post("/api/plan", json={**w, "launch_attempts": 30}).json()
    assert any("predates" in x and "instead of" not in x for x in pl2["warnings"])
    # The failure sentence a reader sees when it does happen.
    s = main.failure_sentence("ValidationError", json.dumps({"errorMessage": "1 validation error for RunConfig\npurchase.max_launch_attempts\n  Extra inputs are not permitted [type=extra_forbidden]"}))
    assert s.startswith("Failed before any measurement: ValidationError: 1 validation error for RunConfig purchase.max_launch_attempts")
    assert "deployed backend (AmxBench-Core) is older than this app" in s and "Deploy the Core stack" in s
    assert main.failure_sentence("States.Timeout", None) == "Failed before any measurement: States.Timeout"


def test_run_names_become_safe_slugs():
    from webui.api import main

    assert main.run_slug("My First Test!! (v2)") == "my-first-test-v2"
    assert main.run_slug("   ") == "" and len(main.run_slug("x" * 80)) == 40
    r = c.get("/api/runs/slug", params={"name": "Gemma vs L4", "use_case": "standard"}).json()
    assert r["slug"] == "gemma-vs-l4" and r["run_id"].startswith("gemma-vs-l4-20")
    assert c.get("/api/runs/slug", params={"name": "", "use_case": "dspm"}).json()["slug"].startswith("dspm-")
    pl = c.post("/api/plan", json={"use_case": "standard", "models": ["m1-qwen2.5-3b"], "gpu": GPU, "cpus": [CPU], "name": "Gemma vs L4"}).json()
    assert pl["run"]["name"] == "gemma-vs-l4"


def test_cell_failures_read_as_sentences():
    """The three failures of 2026-11-12, as stored (Step Functions error object repr with the
    Lambda's JSON envelope inside), rendered for the Why column."""
    from webui.api.main import cell_error_text

    def sfn(msg):
        return str({"Error": "RuntimeError", "Cause": json.dumps({"errorMessage": msg, "errorType": "RuntimeError",
                                                                  "stackTrace": ["  File x, line 1"]})})
    taken = sfn("bring-up of m3|t1-g6|bf16 failed (Failed): 2026-11-12T12:51:18Z serve: m3 (vllm)\nError response from daemon: "
                "No such container: bench-engine\n2026-11-12T12:57:28Z ERROR: container exited before becoming healthy")
    assert cell_error_text(taken).startswith("Another run took over this instance while the model was loading")
    assert "stackTrace" not in cell_error_text(taken) and "RuntimeError" not in cell_error_text(taken)
    empty = sfn("bring-up of m4-gemma-2-2b|t2-m8i-4xl|bf16 failed (Failed): ")
    assert cell_error_text(empty).startswith("The instance was taken away while the model was loading")
    export = sfn("bring-up of m4|t3-m8i-4xl-ov|int8-ov failed (Failed): ValueError: ('Failed to export llm model', '/models/hf/google/gemma')"
                 "\njq: parse error\n2026-11-12T13:00:02Z ERROR: export did not report a hash: Running command: optimum-cli export openvino ...")
    assert cell_error_text(export).startswith("Converting the model to OpenVINO's format failed: the export tool was not told the model's task")
    assert cell_error_text("plain text reason") == "plain text reason" and cell_error_text("") == ""


def test_delete_run_removes_its_objects_and_rows_only(monkeypatch):
    moto = pytest.importorskip("moto")
    import boto3

    from webui.api import main

    bucket = "amx-gpu-bench-123456789012-us-east-1"
    for k, v in (("AWS_DEFAULT_REGION", "us-east-1"), ("AWS_ACCESS_KEY_ID", "testing"), ("AWS_SECRET_ACCESS_KEY", "testing")):
        monkeypatch.setenv(k, v)
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)
        boto3.client("dynamodb", region_name="us-east-1").create_table(
            TableName="amx-gpu-bench", BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}, {"AttributeName": "sk", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}])
        tbl = boto3.resource("dynamodb", region_name="us-east-1").Table("amx-gpu-bench")
        for rid, status in (("r-done", "complete"), ("r-keep", "complete"), ("r-live", "running")):
            tbl.put_item(Item={"pk": f"RUN#{rid}", "sk": "META", "status": status})
            tbl.put_item(Item={"pk": f"RUN#{rid}", "sk": "CELL#m|t|bf16", "status": "ready"})
            tbl.put_item(Item={"pk": f"RUN#{rid}", "sk": "LEVEL#ck__short__c1__r0", "status": "done"})
            for key in ("resolved.json", "report/summary.json", "raw/ck__short__c1__r0.jsonl.gz", "capacity/t1-g6.json"):
                s3.put_object(Bucket=bucket, Key=f"runs/{rid}/{key}", Body=b"{}")
        s3.put_object(Bucket=bucket, Key="runs/r-done-2/resolved.json", Body=b"{}")      # a prefix that merely starts the same
        s3.put_object(Bucket=bucket, Key="models/shared/weights.bin", Body=b"x")
        monkeypatch.setattr(main, "_outputs", lambda: {"BucketName": bucket, "RunStateMachineArn": "arn:aws:states:us-east-1:1:stateMachine:x"})
        monkeypatch.setattr(main, "_reconcile", lambda items: None)
        main._CACHE["summary:r-done"] = {"x": 1}
        # A leftover instance of the finished run is terminated with it; another run's instance is not.
        ec2 = boto3.client("ec2", region_name="us-east-1")
        ami = ec2.describe_images(Owners=["amazon"])["Images"][0]["ImageId"]

        def launch(run):
            return ec2.run_instances(ImageId=ami, InstanceType="m8i.4xlarge", MinCount=1, MaxCount=1, TagSpecifications=[
                {"ResourceType": "instance", "Tags": [{"Key": "bench:managed", "Value": "true"}, {"Key": "bench:run", "Value": run}]}])["Instances"][0]["InstanceId"]
        mine, theirs = launch("r-done"), launch("r-keep")

        r = c.delete("/api/runs/r-done")
        assert r.status_code == 200 and r.json() == {"deleted": "r-done", "objects": 4, "rows": 3, "status_was": "complete",
                                                     "terminated_instances": [mine]}
        states = {i["InstanceId"]: i["State"]["Name"] for res in ec2.describe_instances()["Reservations"] for i in res["Instances"]}
        assert states[mine] in ("shutting-down", "terminated") and states[theirs] == "running"
        # Stopping a run from the UI terminates its instances at once, even when the execution is already gone.
        going = launch("r-go")
        tbl.put_item(Item={"pk": "RUN#r-go", "sk": "META", "status": "running"})
        r = c.post("/api/runs/r-go/stop")
        assert r.status_code == 200 and r.json()["terminated_instances"] == [going]
        states = {i["InstanceId"]: i["State"]["Name"] for res in ec2.describe_instances()["Reservations"] for i in res["Instances"]}
        assert states[going] in ("shutting-down", "terminated") and states[theirs] == "running"
        assert tbl.get_item(Key={"pk": "RUN#r-go", "sk": "META"})["Item"]["status"] == "stopped"
        left = sorted(o["Key"] for o in s3.list_objects_v2(Bucket=bucket)["Contents"])
        assert not any(k.startswith("runs/r-done/") for k in left)
        assert "runs/r-done-2/resolved.json" in left and "models/shared/weights.bin" in left and len([k for k in left if k.startswith("runs/r-keep/")]) == 4
        assert tbl.query(KeyConditionExpression=boto3.dynamodb.conditions.Key("pk").eq("RUN#r-done"))["Count"] == 0
        assert tbl.query(KeyConditionExpression=boto3.dynamodb.conditions.Key("pk").eq("RUN#r-keep"))["Count"] == 3
        assert "summary:r-done" not in main._CACHE
        # Guard rails: a running run must be stopped first; unknown and malformed ids are refused.
        assert c.delete("/api/runs/r-live").status_code == 409 and "stop it first" in c.delete("/api/runs/r-live").json()["detail"]
        assert c.delete("/api/runs/r-done").status_code == 404
        assert c.delete("/api/runs/..%2Fmodels").status_code in (400, 404, 405)
        with pytest.raises(main.HTTPException) as ei:
            main.delete_run_data("../models")
        assert ei.value.status_code == 400
        assert main.RUN_ID.match("runs/../x") is None and main.RUN_ID.match("standard-1111-1137-20261111T093750Z")


def test_spa_is_served_when_built():
    r = c.get("/api/aws/status")
    assert r.status_code == 200 and "ok" in r.json()
    if (ROOT / "webui/dist/index.html").exists():
        assert c.get("/").status_code == 200 and c.get("/results").status_code == 200
        assert "root" in c.get("/results").text


def test_wizard_yaml_round_trips_through_the_cli_schema():
    from bench.config import RunConfig

    w = {"use_case": "dspm", "models": ["m1-qwen2.5-3b"], "workloads": ["W1", "W3"], "gpu": GPU, "cpus": [CPU],
         "request_mode": "concurrent", "concurrency": [1, 8, 16], "preset": "bulk", "test_items": 1000}
    pl = c.post("/api/plan", json=w).json()
    assert pl["ok"]
    import yaml

    rc = RunConfig.model_validate(yaml.safe_load(pl["yaml"])["run"])
    assert rc.dspm.sla_preset == "bulk" and rc.dspm.test_items == 1000 and rc.sla.p99_e2e_ms == 30000
    assert json.dumps(pl["run"], sort_keys=True)  # serializable
