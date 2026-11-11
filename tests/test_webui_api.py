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
