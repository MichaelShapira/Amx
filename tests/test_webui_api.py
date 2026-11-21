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
    assert {m["id"] for m in cat["models"]} >= {"m1-qwen3-1.7b", "m3-roberta-large-mnli", "e1-embeddinggemma-300m"}
    m = next(x for x in cat["models"] if x["id"] == "m1-qwen3-1.7b")
    assert m["name"] == "Qwen3 1.7B" and m["commercial_ok"] is True and m["batch_capable"] is False
    e1 = next(x for x in cat["models"] if x["id"] == "e1-embeddinggemma-300m")
    assert e1["batch_capable"] is True and e1["use_cases"] == ["embedding"] and e1["batch_sizes_default"] == [1, 8, 32, 64]
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
        p = c.get("/api/data/standard/sample", params={"prompt_set": "short", "model": "m1-qwen3-1.7b", "n": 2}).json()
        assert len(p) == 2 and p[0]["tokens"] > 100


def test_plan_builds_a_valid_run_and_refuses_unfair_pairs():
    w = {"use_case": "standard", "models": ["m1-qwen3-1.7b"], "gpu": GPU, "cpus": [CPU], "request_mode": "concurrent",
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
    w = {"use_case": "dspm", "models": ["m3-roberta-large-mnli"], "workloads": ["W3"], "gpu": GPU, "cpus": [CPU],
         "request_mode": "batch", "batch_sizes": [8, 32], "concurrency": [4]}
    pl = c.post("/api/plan", json=w).json()
    assert pl["ok"] and pl["request_mode"] == "batch" and pl["run"]["batch_sizes"] == [8, 32]
    gen = c.post("/api/plan", json={**w, "models": ["m1-qwen3-1.7b"]}).json()
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
        return {"model": "m1-qwen3-1.7b", "target": "t", "accelerator": acc, "instance_type": itype, "engine": "vllm", "precision": "bf16",
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
    assert "To match one g6.xlarge's peak throughput you need 3 × m8i.4xlarge" in g["verdict"]["text"]
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
    """The Review page crashed ('reading map') when an option was ticked with a format it cannot serve: resolve_cells
    raised and the error reply lacked series/pairing/warnings. Now the format is dropped with a note, the option is
    left out of instance_types, and every reply carries every key. Since 2026-11-21 the retired OpenVINO option is
    disabled, so a draft that still names it (an old browser tab) is handled the same way."""
    OV = {"hardware": "t3-m8i-4xl-ov", "instance_type": "m8i.4xlarge", "precisions": ["bf16"]}
    w = {"use_case": "standard", "models": ["m1-qwen3-1.7b"], "gpu": GPU, "cpus": [CPU, OV], "request_mode": "concurrent", "concurrency": [1, 10]}
    pl = c.post("/api/plan", json=w).json()
    assert {"series", "pairing", "warnings", "problems", "estimate"} <= set(pl)
    assert pl["ok"] is True, pl["problems"]
    assert {s["hardware"] for s in pl["series"]} == {"t1-g6", "t2-m8i-4xl"}
    assert "t3-m8i-4xl-ov" not in pl["run"]["instance_types"]
    assert any("t3-m8i-4xl-ov" in x or "OpenVINO" in x for x in pl["warnings"]), pl["warnings"]
    # a format the GPU option cannot serve for this model (fp16 is RoBERTa-only) is dropped with a note too
    pl2 = c.post("/api/plan", json={**w, "cpus": [CPU], "gpu": {**GPU, "precisions": ["fp16"]}}).json()
    assert {"series", "pairing", "warnings", "problems", "estimate"} <= set(pl2)
    assert all(s_["hardware"] != "t1-g6" for s_ in pl2["series"]) and any("runs nothing in this plan" in x for x in pl2["warnings"])

def _backend(supports: bool, commit="b75d598bd40e1fe19fc155d47d2b0f34f4ca09f0"):
    return {"commit": commit, "local_commit": "head", "supports": {"max_launch_attempts": supports}, "behind": not supports,
            "error": None, "missing": [] if supports else ["max_launch_attempts"]}


def test_plan_carries_the_retry_budget_and_spot_wait(monkeypatch):
    from webui.api import main

    monkeypatch.setattr(main, "deployed_backend", lambda: _backend(True))
    w = {"use_case": "standard", "models": ["m1-qwen3-1.7b"], "gpu": GPU, "cpus": [CPU], "purchase": "spot",
         "spot_fallback": True, "spot_wait_minutes": 25, "launch_attempts": 60}
    pl = c.post("/api/plan", json=w).json()
    assert pl["run"]["purchase"] == {"type": "spot", "spot_fallback_on_demand": True, "spot_capacity_window_s": 1500, "max_launch_attempts": 60}
    assert c.post("/api/plan", json={**w, "launch_attempts": 0}).json()["run"]["purchase"]["max_launch_attempts"] == 1
    d = c.post("/api/plan", json={"use_case": "standard", "models": ["m1-qwen3-1.7b"], "gpu": GPU, "cpus": [CPU]}).json()
    assert d["run"]["purchase"]["max_launch_attempts"] == 30 and d["run"]["purchase"]["spot_capacity_window_s"] == 600


def test_plan_leaves_out_settings_the_deployed_backend_does_not_know(monkeypatch):
    """2026-11-12: the deployed preflight (commit b75d598) forbade purchase.max_launch_attempts;
    every run failed before the pre-checks and vanished from the UI. The plan now omits the
    field for such a backend and says so."""
    from webui.api import main

    monkeypatch.setattr(main, "deployed_backend", lambda: _backend(False))
    w = {"use_case": "standard", "models": ["m1-qwen3-1.7b"], "gpu": GPU, "cpus": [CPU], "launch_attempts": 60}
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
    pl = c.post("/api/plan", json={"use_case": "standard", "models": ["m1-qwen3-1.7b"], "gpu": GPU, "cpus": [CPU], "name": "Gemma vs L4"}).json()
    assert pl["run"]["name"] == "gemma-vs-l4"


def test_copy_to_new_run_round_trips_the_definition(monkeypatch):
    """'Copy to new run': the wizard rebuilt from a stored run definition plans the same run (same
    cells, sizes, formats, load, targets, purchase), carries a 'copy' name and does not re-size."""
    from webui.api import main

    monkeypatch.setattr(main, "deployed_backend", lambda: _backend(True))
    w = {"use_case": "standard", "models": ["m1-qwen3-1.7b"], "prompt_set": "medium", "answer_tokens": 256, "depth": "custom",
         "requests": 120, "passes": 2, "gpu": {**GPU, "instance_type": "g6.2xlarge", "precisions": ["bf16"], "alternates": ["g5.2xlarge"]},
         "cpus": [{**CPU, "precisions": ["bf16"]}],
         "request_mode": "concurrent", "concurrency": [2, 8], "sla": {"p99_ttft_ms": 1500, "p50_tpot_ms": 120},
         "purchase": "on-demand", "spot_fallback": False, "spot_wait_minutes": 20, "launch_attempts": 12, "parallel_targets": False,
         "name": "My Qwen test"}
    pl = c.post("/api/plan", json=w).json()
    assert pl["ok"], pl["problems"]
    w2 = main.wizard_from_run(pl["run"], name="My Qwen test", source_run_id="my-qwen-test-20261112T000000Z")
    assert w2["name"] == "My Qwen test copy" and w2["copied_from"] == "my-qwen-test-20261112T000000Z"
    assert w2["sized_for"] == "m1-qwen3-1.7b|bf16" and w2["depth"] == "custom" and (w2["requests"], w2["passes"]) == (120, 2)
    assert w2["gpu"]["instance_type"] == "g6.2xlarge" and w2["gpu"]["alternates"] == ["g5.2xlarge"]
    assert {s_["hardware"]: s_["precisions"] for s_ in w2["cpus"]} == {"t2-m8i-4xl": ["bf16"]}
    assert w2["purchase"] == "on-demand" and w2["spot_fallback"] is False and w2["spot_wait_minutes"] == 20 and w2["launch_attempts"] == 12
    assert w2["parallel_targets"] is False and w2["concurrency"] == [2, 8] and w2["request_mode"] == "concurrent"
    assert w2["sla"] == {"p99_ttft_ms": 1500, "p50_tpot_ms": 120} and w2["prompt_set"] == "medium" and w2["answer_tokens"] == 256
    pl2 = c.post("/api/plan", json=w2).json()
    assert pl2["ok"], pl2["problems"]
    same = {k: v for k, v in pl["run"].items() if k != "name"}
    assert {k: v for k, v in pl2["run"].items() if k != "name"} == same
    # Quick-check depth and batch mode map back to their wizard choices.
    d = {"use_case": "dspm", "models": ["m3-roberta-large-mnli"], "workloads": ["W1", "W3"], "preset": "bulk", "test_items": 500,
         "gpu": GPU, "cpus": [CPU], "request_mode": "batch", "batch_sizes": [8, 32], "concurrency": [1, 4]}
    pd = c.post("/api/plan", json=d).json()
    assert pd["ok"], pd["problems"]
    w3 = main.wizard_from_run(pd["run"])
    assert w3["use_case"] == "dspm" and w3["depth"] == "quick" and w3["request_mode"] == "batch" and w3["batch_sizes"] == [8, 32]
    assert w3["workloads"] == ["W1", "W3"] and w3["preset"] == "bulk" and w3["test_items"] == 500 and w3["name"].endswith(" copy")
    assert c.post("/api/plan", json=w3).json()["ok"]


def test_refusals_read_as_sentences_with_names():
    """The 2026-11-12 refusal 'm3-roberta-large-mnli|t1-g6|fp16: host facts of t1-g6 lack amx_fp16'
    and the other pre-check messages, as the reader sees them."""
    from webui.api.main import refusal_sentences

    old = refusal_sentences(["m3-roberta-large-mnli|t2-m8i-4xl|fp16: host facts of t2-m8i-4xl lack amx_fp16"])[0]
    assert old.startswith("RoBERTa-large (classifier) on CPU · Intel Xeon with AMX · vLLM in FP16 (half precision): the FP16")
    # 2026-11-15: the sentence told the reader to "pick an m8i, c8i or r8i size" for a run that was
    # already on c8i.4xlarge. No size has AMX-FP16 on EC2; the sentence must say so and never
    # recommend a size.
    assert "AMX-FP16" in old and "do not expose" in old and "No instance size changes this" in old
    assert "m8i, c8i or r8i size" not in old and "Pick an" not in old and "greys FP16 out" in old
    gpu = refusal_sentences(["m3-roberta-large-mnli|t1-g6|fp16: host facts of t1-g6 lack amx_fp16"])[0]
    assert gpu.startswith("RoBERTa-large (classifier) on GPU · NVIDIA · vLLM in FP16 (half precision): refused by mistake.")
    assert "Start the run again" in gpu
    new = refusal_sentences(["m3-roberta-large-mnli|t2-m8i-4xl|fp16: c8i.4xlarge has no amx_fp16 (the fp16 format needs it)"])[0]
    assert new.startswith("RoBERTa-large (classifier) on CPU · Intel Xeon with AMX · vLLM in FP16 (half precision): the FP16")
    assert "c8i.4xlarge" in new and "No instance size changes this" in new and "Pick an" not in new
    # The config validator's own refusal (what the pre-checks now say first) reads the same way.
    cfg = refusal_sentences(["m3-roberta-large-mnli/fp16 needs CPU flag amx_fp16, which t2-m8i-4xl lacks"])[0]
    assert cfg.startswith("RoBERTa-large (classifier) on CPU · Intel Xeon with AMX · vLLM in FP16 (half precision): this format needs the AMX-FP16")
    assert "Untick it for this option" in cfg
    s = refusal_sentences(["worker reserved concurrency 32 < highest level 64",
                           "instance types not offered in use1-az1: ['g6e.48xlarge', 'm8i.96xlarge']",
                           "target t5-m7i-4xl is not deployed (no SSM parameter /amx-gpu-bench/targets/t5-m7i-4xl/launch-template)",
                           "Hugging Face access to gated google/gemma-2-2b-it@299a8560 returned HTTP 403",
                           "price unavailable: no price for g6e.xlarge in us-east-1",
                           "DSPM suite needs the dataset and request pools in S3; run `bench upload-artifacts`",
                           "m1-qwen3-1.7b|t2-m8i-4xl|bf16: none of ['ovms'] is available on t2-m8i-4xl",
                           "something nobody mapped"])
    assert s[0].startswith("The load generator in this deployment can send at most 32 requests at a time, and this run asks for 64.")
    assert s[1] == "AWS does not offer these instance sizes in this zone (use1-az1): g6e.48xlarge, m8i.96xlarge. Pick another size on the Hardware step."
    assert s[2].startswith("The hardware option t5-m7i-4xl is not deployed") or "is not deployed in this account" in s[2]
    assert "gated on Hugging Face" in s[3] and "HTTP 403" in s[3] and "Accept the model's license" in s[3]
    assert s[4].startswith("The On-Demand price of an instance size could not be read") and "g6e.xlarge" in s[4]
    assert "bench upload-artifacts" in s[5]
    assert s[6].startswith("Qwen3 1.7B on CPU · Intel Xeon with AMX · vLLM in BF16 (standard): this serving software cannot run")
    assert s[7] == "something nobody mapped"
    assert refusal_sentences(None) == []


def test_fp16_is_blocked_for_cpu_options_at_selection_and_in_the_plan(monkeypatch):
    """2026-11-15: a run with RoBERTa FP16 on the OpenVINO option at c8i.4xlarge passed the wizard
    and the plan, then the pre-checks refused it with advice to pick a c8i. The wizard believed a
    family table (Granite Rapids has AMX-FP16); the pre-checks believed the booted host, which does
    not expose it. One fact now: targets.yaml cpu_features, served in the catalog so the wizard
    greys FP16 out for every CPU option, and applied by the plan, which drops a stale copy with
    the reason instead of letting it reach the pre-checks."""
    from webui.api import main
    monkeypatch.setattr(main, "deployed_backend", lambda: _backend(True))
    cat = c.get("/api/catalog").json()
    for h in cat["hardware"]:
        if h["accelerator"] == "cpu":
            assert "amx_fp16" not in h["cpu_features"] and {"amx_bf16", "amx_int8", "amx_tile"} <= set(h["cpu_features"])
    m3 = next(m for m in cat["models"] if m["id"] == "m3-roberta-large-mnli")
    assert m3["precisions"]["fp16"]["requires_cpu_flag"] == "amx_fp16" and m3["precisions"]["bf16"]["requires_cpu_flag"] is None
    note_fp16 = cat["cpu_flag_notes"]["amx_fp16"]
    assert "do not expose" in note_fp16 and "36 so far" in note_fp16 and "r8i" not in note_fp16
    assert "greyed out for the CPU options" in cat["precisions"]["fp16"]["help"]
    w = {"use_case": "standard", "models": ["m3-roberta-large-mnli"], "gpu": {**GPU, "precisions": ["fp16"]},
         "cpus": [{"hardware": "t2-m8i-4xl", "instance_type": "c8i.4xlarge", "precisions": ["bf16", "fp16"]}]}
    pl = c.post("/api/plan", json=w).json()
    assert pl["ok"] is True, pl["problems"]
    series = {(s_["hardware"], s_["precision"]) for s_ in pl["series"]}
    assert ("t1-g6", "FP16 (half precision)") in series                  # FP16 on the GPU is fine
    assert ("t2-m8i-4xl", "BF16 (standard)") in series and ("t2-m8i-4xl", "FP16 (half precision)") not in series
    note = next(x for x in pl["warnings"] if "FP16" in x)
    assert note.startswith("CPU · Intel Xeon with AMX · vLLM: RoBERTa-large (classifier) in FP16 (half precision) needs the AMX-FP16")
    assert "c8i" in note_fp16 and "Pick an" not in note
    assert "fp16" not in {cell["precision"] for cell in pl["run"]["cells"] if cell["target"] != "t1-g6"}
    # The same for any size: the feature is per family on EC2, not per size.
    for itype in ("m8i.4xlarge", "m8i.8xlarge", "c8i.4xlarge", "m7i.4xlarge"):
        pl2 = c.post("/api/plan", json={**w, "cpus": [{"hardware": "t2-m8i-4xl", "instance_type": itype, "precisions": ["fp16"]}]}).json()
        assert ("t2-m8i-4xl", "FP16 (half precision)") not in {(s_["hardware"], s_["precision"]) for s_ in pl2["series"]}
        assert any("runs nothing in this plan" in x for x in pl2["warnings"]), (itype, pl2["warnings"])


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

    w = {"use_case": "dspm", "models": ["m1-qwen3-1.7b"], "workloads": ["W1", "W3"], "gpu": GPU, "cpus": [CPU],
         "request_mode": "concurrent", "concurrency": [1, 8, 16], "preset": "bulk", "test_items": 1000}
    pl = c.post("/api/plan", json=w).json()
    assert pl["ok"]
    import yaml

    rc = RunConfig.model_validate(yaml.safe_load(pl["yaml"])["run"])
    assert rc.dspm.sla_preset == "bulk" and rc.dspm.test_items == 1000 and rc.sla.p99_e2e_ms == 30000
    assert json.dumps(pl["run"], sort_keys=True)  # serializable


def test_gemma_failures_of_2026_11_15_show_their_cause():
    """Run gemma-20261115T102422Z, as stored. The OpenVINO cells showed only the CalledProcessError
    wrapper; the stored stderr has the converter's ValueError, which is what the reader needs. The
    W8A8 cell kept only the last 50 lines of the vLLM log, which name no cause: say so instead of
    pretending the wrapper is the reason."""
    from webui.api.main import cell_error_text

    cells = json.loads((Path(__file__).parent / "fixtures/gemma-20261115-cells.json").read_text())
    ov = cells["m4-gemma-2-2b|t3-m8i-4xl-ov|int8-ov"]
    text = cell_error_text(ov["error"], ov["stderr"])
    assert text.startswith("Converting the model to OpenVINO's format failed: the converter release refused this model")
    assert "Cause: ValueError: The current version of Transformers does not allow for the export of the model." in text
    assert "Maximum required is 5.0, got: 5.5.4" in text and "start this benchmark again" in text
    assert "CalledProcessError" not in text and "returned non-zero exit status" not in text
    w8 = cells["m4-gemma-2-2b|t2-m8i-4xl|w8a8"]
    text = cell_error_text(w8["error"], w8["stderr"])
    assert text.startswith("The serving software stopped while loading the model; the run kept only the last lines")
    assert "runs since 2026-11-15 keep the whole log" in text
    # With the root cause present (what a run since 2026-11-15 stores), it is named.
    later = w8["stderr"] + "\n--- exception lines from the container log\n(EngineCore_DP0 pid=212) NotImplementedError: no kernel"
    assert cell_error_text(w8["error"], later) == ("The serving software stopped while loading the model. "
                                                   "Cause: NotImplementedError: no kernel")


def test_sample_never_reads_outside_the_prompt_directory():
    """Holmes CSR finding 2026-11-16: /api/data/standard/sample built a file path from the
    caller's `model` and `prompt_set`. Only catalog ids and known set names name a file now."""
    from webui.api import catalog as C

    assert C.sample("standard", model="../../pyproject", prompt_set="short") == []
    assert C.sample("standard", model="m1-qwen3-1.7b", prompt_set="../../../pyproject") == []
    assert C.sample("standard", model="m1-qwen3-1.7b", prompt_set="short/../../.env") == []
    assert C.sample("standard", model="/etc/passwd", prompt_set="short") == []
    for bad in ("..%2F..%2Fpyproject", "../../README", "m1-qwen3-1.7b/../../pyproject"):
        r = c.get("/api/data/standard/sample", params={"model": bad, "prompt_set": "short", "n": 1})
        assert r.status_code == 200 and r.json() == [], bad
    # The real catalog still answers (when the prompt sets are built locally).
    ok = C.sample("standard", model="m1-qwen3-1.7b", prompt_set="short", n=1)
    assert isinstance(ok, list)


def test_classifier_runs_are_judged_on_the_whole_decision_only():
    """RoBERTa generates no tokens: first-word and per-word targets are dropped from its run config
    with a warning, and a classifier row never fails on 'p99_ttft_ms not measured'."""
    from bench import report
    w = {"use_case": "standard", "models": ["m3-roberta-large-mnli"], "gpu": GPU, "cpus": [CPU], "request_mode": "concurrent",
         "concurrency": [1, 4], "depth": "quick", "sla": {"p99_ttft_ms": 2000, "p50_tpot_ms": 150}}
    pl = c.post("/api/plan", json=w).json()
    assert pl["ok"] and pl["run"]["sla"] == {"max_error_pct": 1.0}
    assert any("do not apply and were dropped" in x for x in pl["warnings"])
    w2 = {**w, "sla": {"p99_e2e_ms": 1000}}
    assert c.post("/api/plan", json=w2).json()["run"]["sla"] == {"p99_e2e_ms": 1000.0, "max_error_pct": 1.0}
    chat = {**w, "models": ["m1-qwen3-1.7b"]}
    assert c.post("/api/plan", json=chat).json()["run"]["sla"] == {"p99_ttft_ms": 2000.0, "p50_tpot_ms": 150.0, "max_error_pct": 1.0}
    cls_row = {"kind": "classifier", "accelerator": "gpu", "ttft_p99": None, "tpot_p50": None, "e2e_p99": 40.0, "error_pct": 0}
    assert report.meets_sla(cls_row, {"p99_ttft_ms": 2000, "p50_tpot_ms": 150, "max_error_pct": 1.0}) == (True, "meets SLA")
    assert report.meets_sla(cls_row, {"p99_e2e_ms": 30, "max_error_pct": 1.0}) == (False, "p99_e2e_ms 40 > 30")
    gen_row = {**cls_row, "kind": "generation"}
    assert report.meets_sla(gen_row, {"p99_ttft_ms": 2000, "max_error_pct": 1.0}) == (False, "p99_ttft_ms not measured")


def test_quality_flag_is_a_warning_not_an_exclusion():
    """Owner decision 2026-11-19: a flagged row keeps its SLA verdict and stays in the rankings; the flag travels
    with it on the compare card/point and in the report notes."""
    from bench import report
    row = {"kind": "generation", "accelerator": "cpu", "amx": "VERIFIED", "ttft_p99": 500.0, "tpot_p50": 50.0,
           "e2e_p99": 3000.0, "error_pct": 0, "quality_flag": "rouge_l_mean 0.503 < CPU BF16 0.731 - 0.1"}
    assert report.meets_sla(row, {"p99_ttft_ms": 2000, "p50_tpot_ms": 150, "max_error_pct": 1.0}) == (True, "meets SLA")
    src = open("webui/api/compare.py").read()
    assert src.count('"quality_flag": r.get("quality_flag")') == 2      # card and point
    assert 'warning only; the row stays in the rankings' in open("bench/report.py").read()


def test_a_rejected_first_request_names_the_servers_answer_or_says_it_was_not_kept():
    from webui.api.main import cell_error_text as failure_sentence
    old = ("bring-up of m5-flan-t5-xl|t1-g6|bf16 failed (Failed): ...credentials-store 2026-11-18T07:28:13Z stage: syncing google/flan-t5-xl "
           "from the S3 cache 2026-11-18T07:31:30Z ERROR: first request failed on every API path failed to run commands: exit status 1")
    t = failure_sentence(old, None)
    assert t.startswith("The serving software started and reported healthy, but rejected the first real request") and "was not kept" in t
    new = ("bring-up of m5 failed (Failed): 2026-11-19T07:31:20Z first request to /v1/completions -> HTTP 500: "
           '{"error":{"message":"CUDA error: device-side assert triggered","type":"server_error","code":500}} '
           "2026-11-19T07:31:30Z ERROR: first request failed on every API path (the server's answers are above; its log is in bringup/)")
    t = failure_sentence(new, None)
    assert "/v1/completions answered HTTP 500" in t and "device-side assert" in t and "not recorded" not in t
    assert failure_sentence(new, "RuntimeError: CUDA error").endswith("Cause: RuntimeError: CUDA error")


def test_technical_details_describe_this_build():
    t = c.get("/api/technical").json()
    ids = {h["id"] for h in t["hardware"]}
    assert {"t1-g6", "t2-m8i-4xl"} <= ids and not ({"t3-m8i-4xl-ov", "t4-m8i-4xl-torch"} & ids)   # PyTorch (11-20) and OpenVINO (11-21) retired
    g6 = next(h for h in t["hardware"] if h["id"] == "t1-g6")
    assert g6["accelerator"] == "gpu" and {e["engine"] for e in g6["engines"]} == {"vllm", "thin-torch"}
    torch = next(e for e in g6["engines"] if e["engine"] == "thin-torch")
    assert torch["built"] and torch["dockerfile"] == "servers/thin/Dockerfile.torch" and torch["image"] == "built:thin-torch-cuda"
    vllm = next(e for e in g6["engines"] if e["engine"] == "vllm")
    assert not vllm["built"] and "@sha256:" in vllm["image"]
    assert {im["name"] for im in t["images"]} >= {"thin-torch-cuda"} and not ({"thin-torch-cpu", "thin-openvino", "export"} & {im["name"] for im in t["images"]})
    assert t["files"]["host/serve.sh"]["present"] and "first_request" in t["files"]["host/serve.sh"]["text"]
    assert t["files"]["servers/export/Dockerfile"]["present"] and t["thin_tasks"]["encoder-decoder"] == "seq2seq"
    assert t["export_presets"]["x-ov-int4"]["group_size"] == 128 and t["diagram"]["svg"] == "/api/technical/architecture.svg"
    assert c.get("/api/technical/architecture.svg").status_code == 200 and c.get("/api/technical/architecture.exe").status_code == 404


# ----------------------------------------------------------------------------- translation use case
def test_catalog_offers_the_translation_use_case_and_gemma_4():
    cat = c.get("/api/catalog").json()
    assert set(cat["use_cases"]) == {"standard", "dspm", "translation", "embedding", "intel"}
    assert set(cat["directions"]) == {"en-de", "de-en", "en-es", "es-en"} and set(cat["text_sizes"]) == {"sentence", "paragraph"}
    assert cat["text_sizes"]["sentence"]["answer_tokens"] == 192 and cat["text_sizes"]["paragraph"]["answer_tokens"] == 448
    trans = {m["id"] for m in cat["models"] if "translation" in m["use_cases"]}
    assert trans == {"m1-qwen3-1.7b", "m7-gemma-4-e2b", "m6-gemma-4-e4b"}        # Flan-T5 retired 2026-11-21
    g = next(m for m in cat["models"] if m["id"] == "m6-gemma-4-e4b")
    assert g["name"] == "Gemma 4 E4B Instruct" and g["params_b"] == 8.0 and g["commercial_ok"] and g["license"] == "apache-2.0"
    assert set(g["precisions"]) == {"bf16"} and set(g["precisions"]["bf16"]["engines"]) == {"vllm"}


@pytest.mark.skipif(not (ROOT / "data/translation/v1/items.jsonl.gz").exists(), reason="corpus not built")
def test_translation_data_preview_shows_source_and_reference():
    d = c.get("/api/data/translation").json()
    assert d["available"] and d["kind"] == "parallel" and d["items"] == 400 and set(d["sizes"]) == {"sentence", "paragraph"}
    assert d["sizes"]["sentence"]["items"] == 200 and "Translate the following text" in d["instruction"]
    s = c.get("/api/data/translation/sample", params={"prompt_set": "paragraph", "direction": "es-en", "n": 2, "seed": 1}).json()
    assert len(s) == 2 and all(x["direction"] == "es-en" and x["size"] == "paragraph" for x in s)
    assert all("from Spanish to English" in x["text"] and x["source"] in x["text"] and x["reference"] for x in s)
    assert all(x["chars"] == len(x["source"]) for x in s)
    # unknown direction / size fall back instead of erroring
    s2 = c.get("/api/data/translation/sample", params={"prompt_set": "../etc", "direction": "xx-yy", "n": 1}).json()
    assert len(s2) == 1 and s2[0]["direction"] == "en-de" and s2[0]["size"] == "sentence"


def test_translation_plan_builds_a_translation_run_and_round_trips():
    w = {"use_case": "translation", "models": ["m6-gemma-4-e4b"], "gpu": GPU, "cpus": [CPU], "request_mode": "concurrent",
         "concurrency": [1, 4], "depth": "quick", "directions": ["en-de", "es-en"], "text_size": "paragraph",
         "translation_items": 50, "sla": {"p99_e2e_ms": 10000, "p99_ttft_ms": 2000}}
    pl = c.post("/api/plan", json=w).json()
    assert pl["ok"], pl["problems"]
    run = pl["run"]
    assert run["suite"] == "translation" and run["prompt_set"] == "paragraph"
    assert run["translation"] == {"directions": ["en-de", "es-en"], "test_items": 50}
    assert run["sampling"] == {"temperature": 0.0, "max_new_tokens": 448, "ignore_eos": False, "seed": 0}
    assert run["quality"] == {"enabled": False} and run["sla"]["p99_e2e_ms"] == 10000 and run["sla"]["p99_ttft_ms"] == 2000
    assert {(x["target"], x["precision"]) for x in run["cells"]} == {("t1-g6", "bf16"), ("t2-m8i-4xl", "bf16")}
    from bench import catalog_store
    from bench.config import RunConfig, load_targets, resolve_cells
    rc = RunConfig.model_validate(run)
    cells = resolve_cells(rc, catalog_store.load_catalog(), load_targets(ROOT / "config/targets.yaml"))
    assert {cl.engine for cl in cells} == {"vllm"}
    from webui.api import main
    w2 = main.wizard_from_run(run)
    assert w2["use_case"] == "translation" and w2["text_size"] == "paragraph" and w2["directions"] == ["en-de", "es-en"]
    assert w2["translation_items"] == 50 and w2["sla"] == {"p99_e2e_ms": 10000, "p99_ttft_ms": 2000}
    assert c.post("/api/plan", json=w2).json()["ok"]
    none = c.post("/api/plan", json={**w, "directions": []}).json()
    assert not none["ok"] and any("direction" in p for p in none["problems"])
    review = c.post("/api/plan", json=w).json()
    assert review["ok"]


def test_translation_runs_compare_on_text_size_and_directions():
    from webui.api import compare
    base = {"suite": "translation", "prompt_set": "sentence", "concurrency": [1, 4], "batch_sizes": [1],
            "per_level": {"requests": 50}, "repeats": 1, "sampling": {"max_new_tokens": 192},
            "translation": {"directions": ["en-de", "de-en"], "test_items": 100}}
    a = {"run_id": "a", "run": base, "smoke": True, "dspm_hashes": {"dataset_sha256": "00e1953d272fabc"}}
    b = {"run_id": "b", "run": {**base, "translation": {"directions": ["en-de"], "test_items": 100}}, "smoke": True,
         "dspm_hashes": {"dataset_sha256": "00e1953d272fabc"}}
    out = compare.comparability([a, b])
    assert out["ok"] and any("Only the directions all runs share" in w for w in out["warnings"])
    assert out["shapes"][0]["directions"] == ["de-en", "en-de"] and out["shapes"][0]["dataset"] == "00e1953d272f"
    c2 = {"run_id": "c", "run": {**base, "prompt_set": "paragraph"}, "smoke": True, "dspm_hashes": {"dataset_sha256": "00e1953d272fabc"}}
    out2 = compare.comparability([a, c2])
    assert not out2["ok"] and any("text sizes" in bl for bl in out2["blockers"])
    d = {"run_id": "d", "run": base, "smoke": True, "dspm_hashes": {"dataset_sha256": "ffff"}}
    assert any("corpus versions" in bl for bl in compare.comparability([a, d])["blockers"])
    row = {"model": "m6-gemma-4-e4b", "direction": "en-de", "text_size": "sentence", "workload": None, "kind": "generation", "batch": 1}
    key, label = compare._group_key(row)
    assert key == "m6-gemma-4-e4b|en-de|sentence" and "English → German" in label and "Short messages" in label
    u = compare._unit(row)
    assert u["thr"] == "chars_s" and u["cost"] == "usd_per_1m_chars" and u["lat_sla"] == "p99_e2e_ms"


def test_technical_details_list_gemma_4_serving_flags():
    t = c.get("/api/technical").json()
    flags = {m["model"]: m for m in t["model_flags"]}
    assert set(flags) == {"m6-gemma-4-e4b", "m7-gemma-4-e2b"}
    g = flags["m6-gemma-4-e4b"]
    assert g["engine"] == "vllm" and "--reasoning-parser" in g["args"] and "gemma4" in g["args"] and "--chat-template" not in g["args"]
    assert g["hf_repo"] == "google/gemma-4-E4B-it" and g["name"] == "Gemma 4 E4B Instruct"


def test_a_stored_run_without_repeats_still_plans(monkeypatch):
    """run.yaml omits defaults, so a one-repeat run has no `repeats` key. The draft 'Re-run with recommended
    sizes' builds from it must plan (2026-11-19: depth 'custom' with passes=None made /api/plan raise on
    int(None) and the Estimate panel spun forever)."""
    from webui.api import main
    run = {"name": "embedding-smoke-e1", "suite": "embedding", "mode": "smoke", "models": ["e1-embeddinggemma-300m"],
           "cells": [{"target": "t1-g6", "precision": "bf16"}, {"target": "t2-m8i-4xl", "precision": "bf16"}],
           "prompt_set": "sentence", "embedding": {"test_items": 300}, "concurrency": [1, 8, 32], "batch_sizes": [1, 8, 32],
           "per_level": {"requests": 50}, "quality": {"enabled": False}, "sla": {"p99_e2e_ms": 1000, "max_error_pct": 1.0},
           "purchase": {"type": "spot", "spot_fallback_on_demand": True},
           "instance_types": {"t1-g6": "g6.xlarge", "t2-m8i-4xl": "m8i.4xlarge"}}
    w = main.wizard_from_run(run)
    assert w["depth"] == "quick" and w["requests"] is None and w["passes"] is None
    pl = c.post("/api/plan", json={**w, "name": "again"}).json()
    assert pl["ok"], pl.get("problems")
    assert pl["run"]["repeats"] == 1 and pl["run"]["per_level"] == {"requests": 50}
    # And a hand-made custom draft with the counts missing falls back to the defaults instead of failing.
    pl2 = c.post("/api/plan", json={**w, "name": "again", "depth": "custom", "requests": None, "passes": None}).json()
    assert pl2["ok"], pl2.get("problems")
    assert pl2["run"]["repeats"] == 1 and pl2["run"]["per_level"] == {"requests": 50}
