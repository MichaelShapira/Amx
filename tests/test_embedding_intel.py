"""Embedding and Intel-benchmark use cases: scoring, plan, cell spec, report and the wizard API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench import embedding, mechanics, metrics, plan, report
from bench.cellspec import build, cell_key
from bench.config import RunConfig, load_models, load_run, load_targets, resolve_cells
from tools.translation import generator

ROOT = Path(__file__).resolve().parents[1]
MODELS = load_models(ROOT / "config/models.yaml")
TARGETS = load_targets(ROOT / "config/targets.yaml")
TMAP = TARGETS.resolved()


# ---------------------------------------------------------------------------- embedding scoring
def test_cosine_agreement_and_cross_lingual_retrieval():
    en = {"t-s-0000:en": [1, 0, 0], "t-s-0001:en": [0, 1, 0], "t-s-0002:en": [0, 0, 1]}
    de = {"t-s-0000:de": [0.9, 0.1, 0], "t-s-0001:de": [0.1, 0.9, 0], "t-s-0002:de": [0, 0.8, 0.6]}   # the last one misses
    outs = {**en, **de}
    rt = embedding.retrieval(outs)
    assert rt["de"] == {"recall_at_1": pytest.approx(2 / 3), "n": 3} and "es" not in rt
    ag = embedding.agreement(outs, {k: [v[0] * 2, v[1] * 2, v[2] * 2] for k, v in outs.items()})      # same direction, other norm
    assert ag["n"] == 6 and ag["cosine_mean"] == pytest.approx(1.0) and ag["cosine_min"] == pytest.approx(1.0)
    sc = embedding.score(outs, en)
    assert sc["n"] == 6 and sc["dim"] == 3 and sc["vs_reference_cell"]["n"] == 3
    assert embedding.flag({"vs_reference_cell": {"cosine_mean": 0.97}}, {"vs_reference_cell": {"cosine_mean": 0.999}}, precision="int8-ov")
    assert embedding.flag({"vs_reference_cell": {"cosine_mean": 0.97}}, {"vs_reference_cell": {"cosine_mean": 0.999}}, precision="bf16") is None


# ---------------------------------------------------------------------------- config, plan, cell spec
def test_embedding_run_plan_and_cells():
    run = load_run(ROOT / "config/runs/embedding-smoke.yaml")
    cells = resolve_cells(run, MODELS, TARGETS)
    assert {(c.target, c.engine) for c in cells} == {("t1-g6", "vllm"), ("t2-m8i-4xl", "vllm"), ("t4-m8i-4xl-torch", "thin-torch")}
    m = MODELS.get("e1-embeddinggemma-300m")
    sizes = {plan.embedding_pool(m, "sentence"): 600}
    levels = plan.levels_for_cell(run, cells[0], m, run_id="r", bucket="b", pool_sizes=sizes)
    acc = [lv for lv in levels if lv["accuracy"]]
    perf = [lv for lv in levels if not lv["accuracy"]]
    assert len(acc) == 1 and acc[0]["pool"]["size"] == 300 and acc[0]["api_task"] == "embed" and acc[0]["api_path"] == "/v1/embeddings"
    assert len(perf) == 3 * 3 and {lv["batch"] for lv in perf} == {1, 8, 32} and all(lv["text_size"] == "sentence" for lv in perf)
    spec = build(cells[1], m, TMAP["t2-m8i-4xl"], run, run_id="r", bucket="b", hf_token_secret="s", instance_type="m8i.4xlarge")
    cmd = spec["container"]["cmd"]
    assert "--runner" in cmd and cmd[cmd.index("--runner") + 1] == "pooling" and cmd[cmd.index("--max-model-len") + 1] == "512"
    assert spec["health"]["probe"]["path"] == "/v1/embeddings" and spec["api_task"] == "embed"
    thin = build(cells[2], m, TMAP["t4-m8i-4xl-torch"], run, run_id="r", bucket="b", hf_token_secret="s",
                 ecr_repo_uri="1.dkr.ecr.us-east-1.amazonaws.com/x", built_tags={"thin-torch-cpu": "t"})
    opts = thin["container"]["run_opts"]
    assert "THIN_TASK=embed" in opts and "THIN_MAX_INPUT_TOKENS=512" in opts
    d = run.model_dump(mode="json")
    d["prompt_set"] = "short"
    with pytest.raises(ValueError, match="text size"):
        RunConfig.model_validate(d)


def test_intel_run_plan_and_rules():
    run = load_run(ROOT / "config/runs/intel-smoke.yaml")
    cells = resolve_cells(run, MODELS, TARGETS)
    m = MODELS.get("m1-qwen3-1.7b")
    sizes = {plan.intel_pool(m, 128): 512}
    levels = plan.levels_for_cell(run, cells[0], m, run_id="r", bucket="b", pool_sizes=sizes)
    assert len(levels) == 3 and all(not lv["accuracy"] for lv in levels)
    assert levels[0]["payload"] == {"model": m.id, "max_tokens": 128, "temperature": 0.0, "seed": 0, "ignore_eos": True}
    assert levels[0]["pool"]["key"].endswith("/intel-random-128.jsonl") and [lv["conc"] for lv in levels] == [16, 64, 128]
    assert plan.quality_level(run, cells[0], m, run_id="r", bucket="b", pool_sizes=sizes) is None
    d = run.model_dump(mode="json")
    d["sampling"]["max_new_tokens"] = 64
    with pytest.raises(ValueError, match="exactly intel.output_tokens"):
        RunConfig.model_validate(d)
    d = run.model_dump(mode="json")
    d["prompt_set"] = "short"
    with pytest.raises(ValueError, match="'intel'"):
        RunConfig.model_validate(d)
    # the recipe's max-num-seqs lets the sweep reach 256 in flight
    assert TMAP["t2-m8i-4xl"].engines["vllm"].args["max_num_seqs"] == 256 and TMAP["t1-g6"].engines["vllm"].args["max_num_seqs"] == 256


def test_qwen3_template_kwargs_turn_thinking_off():
    m = MODELS.get("m1-qwen3-1.7b")
    assert m.prompt_format.template_kwargs == {"enable_thinking": False} and m.suites == ["standard", "dspm", "translation", "intel"]


def test_intel_reference_rows_come_from_the_catalog_file():
    run = load_run(ROOT / "config/runs/intel-smoke.yaml")
    rows = report.intel_reference(run, MODELS)
    assert rows and all(r["model_id"] == "Qwen/Qwen3-1.7B" and r["io"] == "128/128" for r in rows)
    assert all(r["tok_s_per_core"] and r["total_cores"] >= 96 for r in rows)
    data = json.loads((ROOT / "config/intel_catalog_benchmarks.json").read_text())
    assert {r["model_id"] for r in data["rows"]} >= {"Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"} and len(data["rows"]) >= 70
    assert report.intel_reference(load_run(ROOT / "config/runs/smoke-m1.yaml"), MODELS) == []


# ---------------------------------------------------------------------------- report: embedding quality
MS = 1_000_000


def _cls_records(conc: int, n: int, e2e_ms: float, batch: int, tokens: int = 40, start: int = 10**18) -> list[dict]:
    out = []
    for w in range(conc):
        t = start
        for _ in range(n):
            out.append({"wall_send_ns": t, "send_to_first_ns": None, "send_to_last_ns": None, "send_to_end_ns": int(e2e_ms * MS),
                        "prompt_tokens": tokens * batch, "prompt_tokens_expected": tokens * batch, "completion_tokens": 0,
                        "error": None, "timeout": False, "worker": w, "wave": 0, "start_at_ns": start, "batch_size": batch})
            t += int((e2e_ms + 1) * MS)
    return out


def test_report_scores_embeddings_and_labels_rows():
    run = load_run(ROOT / "config/runs/embedding-smoke.yaml")
    d = run.model_dump(mode="json")
    cells = [{"cell_id": "e1-embeddinggemma-300m|t1-g6|bf16", "model": "e1-embeddinggemma-300m", "target": "t1-g6", "precision": "bf16",
              "engine": "vllm", "variant": "default", "quality_only": False},
             {"cell_id": "e1-embeddinggemma-300m|t2-m8i-4xl|bf16", "model": "e1-embeddinggemma-300m", "target": "t2-m8i-4xl", "precision": "bf16",
              "engine": "vllm", "variant": "default", "quality_only": False}]
    gpu, cpu = cell_key(cells[0]["cell_id"]), cell_key(cells[1]["cell_id"])
    prices = {"g6.xlarge": {"usd_per_hour": 0.8048, "sku": "G6", "effective_date": "2026-11-01", "price_list_arn": "arn", "publication_date": "2026-11-01"},
              "m8i.4xlarge": {"usd_per_hour": 0.8467, "sku": "M8I", "effective_date": "2026-11-01", "price_list_arn": "arn", "publication_date": "2026-11-01"}}
    resolved = {"run_id": "r-emb", "run": d, "cells": cells, "prices": prices, "smoke": True, "run_hash": "h", "models_hash": "m",
                "targets_hash": "t", "harness_commit": "abc"}
    ms = []
    for ck, e2e in ((gpu, 12.0), (cpu, 40.0)):
        for conc in (1, 8):
            for b in (1, 8):
                m = metrics.summarize_level(_cls_records(conc, 6, e2e * (1 + b / 8), b), concurrency=conc, kind="classifier")
                m.update(cell_key=ck, level_key=f"{ck}__embed-sentence__c{conc}" + (f"__b{b}" if b > 1 else "") + "__r0", name="embed-sentence",
                         batch=b, repeat=0, workload=None, text_size="sentence", accuracy=False)
                ms.append(m)
    ev = {cpu: {"flags": ["amx_bf16", "amx_int8", "amx_tile"], "trace": {"gemm_total": 10, "gemm_amx": 10}, "dispatch": {}}}
    items = generator.build_items(4, 0)
    sent = [it for it in items if it["size"] == "sentence"]
    import numpy as np
    rng = np.random.default_rng(0)
    base = {it["id"]: rng.normal(size=8) for it in sent}
    preds = {}
    for ck, noise in ((gpu, 0.0), (cpu, 0.01)):
        rows = []
        for it in sent:
            for lang in ("en", "de", "es"):
                vec = base[it["id"]] + (0.05 if lang != "en" else 0) * rng.normal(size=8) + noise * rng.normal(size=8)
                rows.append({"item_id": embedding.item_key(it["id"], lang), "output": [{"embedding": vec.tolist()}]})
        preds[f"{ck}/embed-sentence-accuracy.jsonl.gz"] = rows
    s = report.build_summary(report.RunInputs(resolved=resolved, metrics=ms, evidence=ev, predictions=preds, dataset_items=items))
    eq = s["embedding_quality"]
    assert eq[cpu]["vs_reference_cell"]["cosine_mean"] > 0.99 and eq[cpu]["retrieval"]["de"]["recall_at_1"] == 1.0 and eq[cpu]["flag"] is None
    assert "vs_reference_cell" not in eq[gpu] and eq[gpu]["retrieval"]["es"]["n"] == 4
    rows = s["rows"]
    assert all(r["model_type"] == "embedding" and r["text_size"] == "sentence" for r in rows)
    assert set(s["per_model"]) == {"e1-embeddinggemma-300m sentence b1", "e1-embeddinggemma-300m sentence b8"}
    cpu_rows = [r for r in rows if r["accelerator"] == "cpu"]
    assert all(r["quality_delta"] is not None and r["quality_delta"] < 0.01 for r in cpu_rows)
    md = report.render(s, with_charts=False)["summary.md"].decode()
    assert "## Embedding quality" in md and "recall@1" in md


# ---------------------------------------------------------------------------- mechanics
def test_mechanics_explains_reading_versus_writing():
    params = {"m": 4_000_000_000}
    instances = {"g6.xlarge": {"gpu_model": "L4", "gpus": 1, "vcpus": 4}, "m8i.4xlarge": {"vcpus": 16}}
    hosts = {"membw-m8i.4xlarge": {"triad_mb_s": 107_000}}
    rows = [{"model": "m", "accelerator": "gpu", "instance_type": "g6.xlarge", "engine": "vllm", "precision": "bf16", "kind": "generation",
             "conc": 1, "batch": 1, "req_s": 0.3, "agg_in_tok_s": 18, "agg_out_tok_s": 25, "tpot_p50": 37.0, "e2e_p50": 3000, "workload": None},
            {"model": "m", "accelerator": "cpu", "instance_type": "m8i.4xlarge", "engine": "vllm", "precision": "bf16", "kind": "generation",
             "conc": 1, "batch": 1, "req_s": 0.1, "agg_in_tok_s": 6, "agg_out_tok_s": 8, "tpot_p50": 115.0, "e2e_p50": 9500, "workload": None},
            {"model": "m", "accelerator": "cpu", "instance_type": "m8i.4xlarge", "engine": "vllm", "precision": "bf16", "kind": "generation",
             "conc": 16, "batch": 1, "req_s": 1.0, "agg_in_tok_s": 60, "agg_out_tok_s": 85, "tpot_p50": 150.0, "e2e_p50": 12000, "workload": None}]
    m = mechanics.explain_group(rows, params, hosts, instances)
    assert m["shape"]["generation"] and 60 <= m["shape"]["input_tokens"] <= 61 and 80 <= m["shape"]["output_tokens"] <= 84
    assert 0.9 < m["shape"]["writing_share"] <= 1.0
    cpu, gpu = m["sides"]["cpu"], m["sides"]["gpu"]
    assert cpu["weights_gb"] == 8.0 and cpu["bandwidth_gb_s"] == 107.0 and 69 < cpu["implied_bandwidth_gb_s"] < 70
    assert gpu["bandwidth_gb_s"] == 300 and 215 < gpu["implied_bandwidth_gb_s"] < 217
    assert cpu["top_conc"] == 16 and cpu["achieved_tflops"] == pytest.approx(16 * 8e9 / 0.150 / 1e12)
    assert "memory is 2.8x faster" in m["why"] and "mostly writing" in m["why"]
    # With per-request medians present the shape comes from them, not from agg_tok_s / req_s, which overstates the
    # per-request token count at high concurrency (the live 128/128 Intel smoke run read as 186/185, 100% writing).
    live = [{"model": "m", "accelerator": "gpu", "instance_type": "g6.xlarge", "engine": "vllm", "precision": "bf16", "kind": "generation",
             "conc": 16, "batch": 1, "req_s": 7.07, "agg_in_tok_s": 909.9, "agg_out_tok_s": 904.7, "ttft_p50": 141.96, "tpot_p50": 16.688,
             "e2e_p50": 2260.85, "prefill_tok_s_p50": 905.5, "workload": None}]
    m3 = mechanics.explain_group(live, params, hosts, instances)["shape"]
    assert 127 <= m3["input_tokens"] <= 129 and 127 <= m3["output_tokens"] <= 129 and 0.93 < m3["writing_share"] < 0.95
    emb = [{"model": "m", "accelerator": "cpu", "instance_type": "m8i.4xlarge", "engine": "vllm", "precision": "bf16", "kind": "classifier",
            "conc": 8, "batch": 8, "req_s": 10, "cls_s": 80, "agg_in_tok_s": 3200, "workload": None, "e2e_p50": 100}]
    m2 = mechanics.explain_group(emb, params, hosts, instances)
    assert not m2["shape"]["generation"] and m2["shape"]["writing_share"] == 0.0 and "All reading" in m2["why"]
