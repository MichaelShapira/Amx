"""Reports on a synthetic run (task 13): SLA highlighting, capacity equivalence, AMX and
quality handling, DSPM accuracy, CSV/Markdown/chart rendering."""

from __future__ import annotations

import csv
import io

import pytest

from bench import metrics, report
from bench.cellspec import cell_key

MS = 1_000_000
PRICES = {"g6.xlarge": {"usd_per_hour": 0.8048, "sku": "G6", "effective_date": "2026-11-01", "price_list_arn": "arn",
                        "publication_date": "2026-11-01"},
          "m8i.4xlarge": {"usd_per_hour": 0.8467, "sku": "M8I", "effective_date": "2026-11-01", "price_list_arn": "arn",
                          "publication_date": "2026-11-01"}}


def records(conc: int, n: int, ttft_ms: float, tpot_ms: float, tokens: int = 128, prompt: int = 128,
            start: int = 10**18) -> list[dict]:
    out = []
    e2e = ttft_ms + tpot_ms * (tokens - 1)
    for w in range(conc):
        t = start
        for i in range(n):
            out.append({"wall_send_ns": t, "send_to_first_ns": int(ttft_ms * MS),
                        "send_to_last_ns": int((ttft_ms + tpot_ms * (tokens - 1)) * MS),
                        "send_to_end_ns": int((e2e + 1) * MS), "prompt_tokens": prompt, "prompt_tokens_expected": prompt,
                        "completion_tokens": tokens, "error": None, "timeout": False, "worker": w, "wave": 0,
                        "start_at_ns": start, "batch_size": 1})
            t += int((e2e + 1) * MS)
    return out


def level_metric(ck: str, conc: int, ttft: float, tpot: float, **kw) -> dict:
    m = metrics.summarize_level(records(conc, 6, ttft, tpot), concurrency=conc, kind="generation",
                                ignore_eos=True, max_new_tokens=128)
    m.update(cell_key=ck, level_key=f"{ck}__short__c{conc}__r0", name="short", batch=1, repeat=0, workload=None,
             accuracy=False, **kw)
    return m


CELLS = [{"cell_id": "m1-qwen2.5-3b|t1-g6|bf16", "model": "m1-qwen2.5-3b", "target": "t1-g6", "precision": "bf16",
          "engine": "vllm", "variant": "default", "quality_only": False},
         {"cell_id": "m1-qwen2.5-3b|t2-m8i-4xl|bf16", "model": "m1-qwen2.5-3b", "target": "t2-m8i-4xl",
          "precision": "bf16", "engine": "vllm", "variant": "default", "quality_only": False}]
GPU, CPU = cell_key(CELLS[0]["cell_id"]), cell_key(CELLS[1]["cell_id"])


def resolved(suite: str = "standard", **run_overrides) -> dict:
    from bench.config import load_run

    run = load_run("config/runs/smoke-m1.yaml" if suite == "standard" else "config/runs/dspm-smoke.yaml")
    d = run.model_dump(mode="json")
    d.update(run_overrides)
    return {"run_id": "r-test", "run": d, "cells": CELLS, "prices": PRICES, "smoke": True,
            "run_hash": "h", "models_hash": "m", "targets_hash": "t", "harness_commit": "abc"}


def inputs(amx_ok: bool = True) -> report.RunInputs:
    ms = [level_metric(GPU, c, 40, 8) for c in (1, 4, 16)]
    ms += [level_metric(CPU, c, 120 * (1 + c / 8), 30 + c * 8, amx_counter={"exe.amx_busy": 5_000, "cycles": 1_000_000})
           for c in (1, 4, 16)]
    ev = {CPU: {"flags": ["amx_bf16", "amx_int8", "amx_tile", "amx_fp16"],
                "trace": {"gemm_total": 10, "gemm_amx": 10 if amx_ok else 0}, "dispatch": {}}}
    if not amx_ok:
        for m in ms:
            m.pop("amx_counter", None)
    q = {GPU: [{"id": f"q{i}", "text": f"answer {i} alpha beta", "error": None} for i in range(5)],
         CPU: [{"id": f"q{i}", "text": f"answer {i} alpha beta" if i else "different", "error": None} for i in range(5)]}
    return report.RunInputs(resolved=resolved(), metrics=ms, evidence=ev, quality=q,
                            capacity={"t1-g6": [{"time": "2026-11-09T10:00:00Z", "code": "InsufficientInstanceCapacity",
                                                 "instance_type": "g6.xlarge", "availability_zone": "us-east-1a"}]})


def test_rows_sla_and_capacity_equivalence():
    s = report.build_summary(inputs())
    rows = {(r["target"], r["conc"]): r for r in s["rows"]}
    assert len(rows) == 6
    gpu16, cpu1, cpu16 = rows[("t1-g6", 16)], rows[("t2-m8i-4xl", 1)], rows[("t2-m8i-4xl", 16)]
    assert gpu16["meets_sla"] and gpu16["amx"] == "n/a"
    assert cpu1["amx"] == "VERIFIED" and cpu1["amx_basis"] == "hardware_counter"
    assert cpu1["meets_sla"]                                   # TTFT 135 ms, TPOT 38 ms
    assert not cpu16["meets_sla"] and "p50_tpot_ms" in cpu16["sla_reason"]   # TPOT 158 ms > 150 ms
    pm = s["per_model"]["m1-qwen2.5-3b"]
    assert pm["best_gpu"]["conc"] == 16
    eq = {e["cpu_row"]["conc"]: e for e in pm["capacity_equivalence"]}
    gpu_tp, cpu_tp = gpu16["agg_out_tok_s"], rows[("t2-m8i-4xl", 4)]["agg_out_tok_s"]
    assert eq[4]["cpu_instances_per_gpu"] == pytest.approx(-(-gpu_tp // cpu_tp))
    assert eq[4]["standard_vcpus_needed"] == eq[4]["cpu_instances_per_gpu"] * 16
    assert eq[4]["hourly_cost_ratio"] == pytest.approx(eq[4]["cpu_instances_per_gpu"] * 0.8467 / 0.8048)
    assert gpu16["usd_per_1m_tokens"] == pytest.approx(0.8048 / (gpu16["agg_out_tok_s"] * 3600) * 1e6)


def test_invalid_amx_excludes_cpu_rows_and_warns():
    s = report.build_summary(inputs(amx_ok=False))
    cpu = [r for r in s["rows"] if r["accelerator"] == "cpu"]
    assert all(r["amx"] == "INVALID" and not r["meets_sla"] for r in cpu)
    assert any("AMX INVALID" in w for w in s["warnings"])
    assert any("non-commercial" in w for w in s["warnings"]) and any("SMOKE" in w for w in s["warnings"])


def test_quality_scored_against_gpu_reference():
    s = report.build_summary(inputs())
    q = s["quality"][CPU]
    assert q["reference_cell"] == GPU and q["n"] == 5 and q["exact_match_rate"] == pytest.approx(0.8)
    assert q["quality_delta"] == pytest.approx(1 - q["rouge_l_mean"])


def test_render_csv_markdown_and_charts():
    s = report.build_summary(inputs())
    files = report.render(s)
    rows = list(csv.DictReader(io.StringIO(files["summary.csv"].decode())))
    assert len(rows) == 6 and set(report.CSV_COLUMNS) == set(rows[0])
    md = files["summary.md"].decode()
    for part in ("SMOKE RUN", "## m1-qwen2.5-3b", "**best GPU**", "Capacity equivalence", "## Price source",
                 "InsufficientInstanceCapacity", "## Warnings", "TTFT P1/P50/P99"):
        assert part in md, part
    assert any(k.startswith("charts/") and k.endswith(".png") for k in files)
    assert files["charts/m1-qwen2.5-3b-e2e_p99.png"][:4] == b"\x89PNG"


def test_group_key_strips_only_the_repeat():
    assert report.group_key("m1__t2__bf16__short__c4__r2") == "m1__t2__bf16__short__c4"
    assert report.group_key("m1__t2__bf16__W1-short-accuracy__c16") == "m1__t2__bf16__W1-short-accuracy__c16"


def test_dspm_accuracy_and_cpu_gpu_agreement():
    items = [{"id": f"i{k}", "label": "BLOCK" if k % 2 else "ALLOW", "categories": ["FINANCIAL"] if k % 2 else [],
              "hard_case": None if k % 2 else "masked", "category_top": "FINANCIAL" if k % 2 else "NONE"}
             for k in range(40)]

    def preds(flip: set[int]):
        rows = []
        for k in range(40):
            truth = "BLOCK" if k % 2 else "ALLOW"
            ans = ("ALLOW" if truth == "BLOCK" else "BLOCK") if k in flip else truth
            rows.append({"item_id": f"i{k}", "output": [{"text": ans + "\n"}]})
        return rows

    inp = report.RunInputs(resolved=resolved("dspm"), metrics=[], dataset_items=items,
                           predictions={f"{GPU}/W3-accuracy.jsonl.gz": preds({1}),
                                        f"{CPU}/W3-accuracy.jsonl.gz": preds({1, 3, 5})})
    s = report.build_summary(inp)
    g, c = s["dspm_accuracy"][GPU]["W3"], s["dspm_accuracy"][CPU]["W3"]
    assert g["recall"]["value"] == pytest.approx(19 / 20) and c["recall"]["value"] == pytest.approx(17 / 20)
    vs = c["vs_gpu"]
    assert vs["agreement"]["value"] == pytest.approx(38 / 40) and not vs["qualifies"]
    md = report.render(s, with_charts=False)["summary.md"].decode()
    assert "DSPM/DLP accuracy" in md and "CC BY 4.0" in md and "not like-for-like" in md
    assert "Speed results transfer" in md
