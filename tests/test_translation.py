"""Translation use case: the synthetic parallel corpus, chrF scoring, the run plan, the report's
translation quality and the Gemma 4 E4B catalog entry (vLLM flags, host-memory KV cap)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench import metrics, plan, report, translation
from bench.cellspec import build, cell_key, cpu_kv_cache_gib
from bench.config import ModelCatalog, RunConfig, load_models, load_run, load_targets, resolve_cells
from tools.translation import build_dataset, generator

ROOT = Path(__file__).resolve().parents[1]
MODELS = load_models(ROOT / "config/models.yaml")
TARGETS = load_targets(ROOT / "config/targets.yaml")
TMAP = TARGETS.resolved()


# ---------------------------------------------------------------------------- corpus
def test_corpus_is_deterministic_and_parallel():
    a, b = generator.build_items(6, 7), generator.build_items(6, 7)
    assert a == b and len(a) == 12
    for it in a:
        assert set(it["text"]) == {"en", "de", "es"}
        for lang, text in it["text"].items():
            assert "{" not in text and "}" not in text, f"unfilled slot in {it['id']} {lang}: {text}"
            assert it["chars"][lang] == len(text) and it["words"][lang] == len(text.split())
        assert it["sentences"] == (1 if it["size"] == "sentence" and it["text"]["en"].count(". ") == 0 else it["sentences"])
    sizes = {it["size"] for it in a}
    assert sizes == {"sentence", "paragraph"}
    para = next(it for it in a if it["size"] == "paragraph")
    assert 5 <= para["sentences"] <= 7
    assert generator.build_items(6, 8) != a                      # the seed matters


def test_slot_values_are_localized_per_language():
    rng = __import__("random").Random(3)
    slots = generator.make_slots(rng)
    en, de, es = (slots.render(lang) for lang in ("en", "de", "es"))
    assert en["amount"].startswith("€") and de["amount"].endswith(" €") and es["amount"].endswith(" €")
    assert generator._money(1250, 0, "en") == "€1,250.00" and generator._money(1250, 0, "de") == "1.250,00 €"
    assert generator._money(1250, 0, "es") == "1.250,00 €" and generator._money(349, 90, "en") == "€349.90"
    assert en["weekday"] != de["weekday"] != es["weekday"]
    assert en["date"].split()[0] in [m[0] for m in generator.MONTHS]      # "March 14, 2026"
    assert de["date"].split()[1] in [m[1] for m in generator.MONTHS]      # "14. März 2026"
    assert es["date"].split()[1] == "de" and es["date"].split()[2] in [m[2] for m in generator.MONTHS]
    assert en["percent"].endswith("%") and de["percent"].endswith(" %")


def test_dataset_build_writes_manifest_with_fingerprint(tmp_path):
    man = build_dataset.build(tmp_path, 4, 1)
    assert man["items"] == 8 and man["n_per_size"] == 4 and len(man["dataset_sha256"]) == 64
    assert (tmp_path / "items.jsonl.gz").exists() and (tmp_path / "review_sample.md").exists()
    items = build_dataset.read_items(tmp_path / "items.jsonl.gz")
    assert [it["id"] for it in items] == [it["id"] for it in generator.build_items(4, 1)]
    again = build_dataset.build(tmp_path / "b", 4, 1)
    assert again["dataset_sha256"] == man["dataset_sha256"]           # byte-identical rebuild


# ---------------------------------------------------------------------------- chrF
def test_chrf_identity_similarity_and_zero():
    ref = "Der Lkw hat das Verteilzentrum in München um 09:00 Uhr verlassen."
    assert translation.chrf(ref, ref) == pytest.approx(100.0)
    close = translation.chrf("Der Lastwagen hat das Verteilzentrum in München um 09:00 Uhr verlassen.", ref)
    far = translation.chrf("The parcel was delivered yesterday evening.", ref)
    assert 70 < close < 100 and far < 25
    assert translation.chrf("", ref) == 0.0 and translation.chrf(ref, "") == 0.0


def test_score_and_agreement_and_flag():
    refs = {"a": "Hallo Welt.", "b": "Vielen Dank für Ihre Geduld."}
    outs = {"a": "Übersetzung: Hallo Welt.", "b": "\u201eVielen Dank für Ihre Geduld.\u201c", "c": "extra"}
    sc = translation.score(outs, refs, max_tokens=8, completion_tokens={"a": 8, "b": 3})
    assert sc["n"] == 2 and sc["chrf"] == pytest.approx(100.0) and sc["truncated_pct"] == 50.0 and sc["empty_pct"] == 0.0
    ag = translation.agreement(outs, {"a": "Hallo Welt.", "b": "Danke für Ihre Geduld."})
    assert ag["n"] == 2 and ag["exact_match_rate"] == 0.5 and 50 < ag["chrf"] < 100
    floor = {"chrf": 80.0}
    assert translation.flag({"chrf": 76.5}, floor, precision="w8a8") is not None
    assert translation.flag({"chrf": 78.0}, floor, precision="w8a8") is None
    assert translation.flag({"chrf": 10.0}, floor, precision="bf16") is None       # BF16 is never flagged against itself


# ---------------------------------------------------------------------------- config and plan
def test_translation_run_config_rules():
    run = load_run(ROOT / "config/runs/translation-smoke.yaml")
    assert run.suite == "translation" and run.translation.directions == ["en-de", "de-en", "en-es", "es-en"]
    d = run.model_dump(mode="json")
    d["sampling"]["ignore_eos"] = True
    with pytest.raises(ValueError, match="ignore_eos"):
        RunConfig.model_validate(d)
    d = run.model_dump(mode="json")
    d["prompt_set"] = "short"
    with pytest.raises(ValueError, match="text size"):
        RunConfig.model_validate(d)
    d = run.model_dump(mode="json")
    d["translation"] = {"directions": []}
    with pytest.raises(ValueError, match="at least one direction"):
        RunConfig.model_validate(d)
    d = run.model_dump(mode="json")
    d.pop("translation")
    assert RunConfig.model_validate(d).translation.directions == ["en-de", "de-en", "en-es", "es-en"]   # default: all four


def test_vllm_args_may_not_take_over_harness_flags():
    data = {m.id: m.model_dump(mode="json") for m in MODELS.models}
    g = data["m6-gemma-4-e4b"]
    g["vllm_args"] = ["--port", "1234"]
    with pytest.raises(ValueError, match="harness owns it"):
        ModelCatalog.model_validate({"size_cap_b": 8.0, "models": [g]})


def test_translation_levels_one_pool_per_direction_plus_accuracy_pass():
    run = load_run(ROOT / "config/runs/translation-smoke.yaml")
    cells = resolve_cells(run, MODELS, TARGETS)
    m = MODELS.get("m6-gemma-4-e4b")
    sizes = {plan.translation_pool(m, d, "sentence"): 200 for d in plan.TRANSLATION_DIRECTIONS}
    levels = plan.levels_for_cell(run, cells[0], m, run_id="r", bucket="b", pool_sizes=sizes)
    acc = [lv for lv in levels if lv["accuracy"]]
    perf = [lv for lv in levels if not lv["accuracy"]]
    assert len(acc) == 4 and len(perf) == 4 * 3                       # 4 directions x 3 concurrency levels x 1 repeat
    assert {lv["direction"] for lv in acc} == set(plan.TRANSLATION_DIRECTIONS)
    a = next(lv for lv in acc if lv["direction"] == "en-de")
    assert a["name"] == "en-de-sentence-accuracy" and a["keep_output"] and a["conc"] == plan.TRANSLATION_ACCURACY_CONC
    assert a["pool"]["size"] == 100 and a["pool"]["key"].endswith("/translate-en-de-sentence.jsonl")   # test_items caps it
    p = perf[0]
    assert p["payload"]["ignore_eos"] is False and p["payload"]["max_tokens"] == 192 and p["quota"] == 50
    assert p["api_path"] == "/v1/completions" and p["direction"] in plan.TRANSLATION_DIRECTIONS
    assert plan.quality_level(run, cells[0], m, run_id="r", bucket="b", pool_sizes=sizes) is None
    with pytest.raises(ValueError, match="missing from the manifest"):
        plan.levels_for_cell(run, cells[0], m, run_id="r", bucket="b", pool_sizes={})


# ---------------------------------------------------------------------------- Gemma 4 cell spec
def test_gemma4_vllm_command_carries_the_recipe_flags_and_fits_the_host():
    run = load_run(ROOT / "config/runs/translation-smoke.yaml")
    m = MODELS.get("m6-gemma-4-e4b")
    assert m.params == 7996156490 and m.license == "apache-2.0" and not m.gated
    cells = {c.target: c for c in resolve_cells(run, MODELS, TARGETS)}
    cpu = build(cells["t2-m8i-4xl"], m, TMAP["t2-m8i-4xl"], run, run_id="r", bucket="b", hf_token_secret="s",
                instance_type="m8i.4xlarge")
    cmd = cpu["container"]["cmd"]
    for flag in ("--enable-auto-tool-choice", "--tool-call-parser", "--reasoning-parser", "--limit-mm-per-prompt"):
        assert flag in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "gemma4" and cmd[cmd.index("--reasoning-parser") + 1] == "gemma4"
    # The recipe's example template file exists only in the CPU image's working directory (the GPU
    # DLC refused it on 2026-11-19); the model's own chat template carries the tool format.
    assert "--chat-template" not in cmd
    assert json.loads(cmd[cmd.index("--limit-mm-per-prompt") + 1]) == {"image": 0, "audio": 0}
    assert cmd.index("--limit-mm-per-prompt") > cmd.index("--block-size")      # appended after the harness's own flags
    # 64 GiB host: the engine holds 2.2 x 14.9 GiB (measured 2.15x, tools/gemma4_cpu_probe.py) plus 6 GiB
    # of headroom, which leaves 25 GiB for the KV cache, not the target's 32 (the budget that hung the
    # first live host). A 32 GiB host cannot hold the model at all and is refused with the size it needs.
    assert cmd[cmd.index("--kv-cache-memory-bytes") + 1] == str(25 * 2**30)
    with pytest.raises(ValueError, match="needs about 43 GiB.*c8i.4xlarge has 32 GiB"):
        build(cells["t2-m8i-4xl"], m, TMAP["t2-m8i-4xl"], run, run_id="r", bucket="b", hf_token_secret="s",
              instance_type="c8i.4xlarge")
    # A 4B model keeps the target's budget on the 64 GiB host and gets a smaller one on a 32 GiB host.
    q = MODELS.get("m2-qwen3-4b-2507")
    assert cpu_kv_cache_gib(32, TMAP["t2-m8i-4xl"], "m8i.4xlarge", q, "bf16") == 32
    assert cpu_kv_cache_gib(32, TMAP["t2-m8i-4xl"], "c8i.4xlarge", q, "bf16") == 9
    gpu = build(cells["t1-g6"], m, TMAP["t1-g6"], run, run_id="r", bucket="b", hf_token_secret="s")["container"]["cmd"]
    assert "--limit-mm-per-prompt" in gpu and "--gpu-memory-utilization" in gpu
    # Other models' commands carry none of these.
    other = build(cells["t2-m8i-4xl"], q, TMAP["t2-m8i-4xl"], run, run_id="r", bucket="b", hf_token_secret="s")["container"]["cmd"]
    assert "--reasoning-parser" not in other


# ---------------------------------------------------------------------------- metrics and report
MS = 1_000_000


def _records(conc: int, n: int, ttft: float, tpot: float, tokens: int, chars: int, start: int = 10**18) -> list[dict]:
    out = []
    e2e = ttft + tpot * (tokens - 1)
    for w in range(conc):
        t = start
        for _ in range(n):
            out.append({"wall_send_ns": t, "send_to_first_ns": int(ttft * MS), "send_to_last_ns": int((ttft + tpot * (tokens - 1)) * MS),
                        "send_to_end_ns": int((e2e + 1) * MS), "prompt_tokens": 60, "prompt_tokens_expected": 60,
                        "completion_tokens": tokens, "error": None, "timeout": False, "worker": w, "wave": 0,
                        "start_at_ns": start, "batch_size": 1, "content_chars": chars})
            t += int((e2e + 1) * MS)
    return out


def test_chars_per_second_flows_from_pool_lines_to_level_metrics():
    recs = _records(4, 5, 100, 20, 40, 120)
    m = metrics.summarize_level(recs, concurrency=4, kind="generation")
    assert m["chars_per_s"] is not None and m["chars_per_s"] > 0
    assert m["chars_per_s"] == pytest.approx(m["requests_per_s"] * 120, rel=1e-6)
    combined = metrics.combine_repeats([m, m], recs + recs, concurrency=4)
    assert combined["chars_per_s"] == pytest.approx(m["chars_per_s"])


def _translation_inputs(cpu_quality: str = "same") -> report.RunInputs:
    run = load_run(ROOT / "config/runs/translation-smoke.yaml")
    d = run.model_dump(mode="json")
    cells = [{"cell_id": "m6-gemma-4-e4b|t1-g6|bf16", "model": "m6-gemma-4-e4b", "target": "t1-g6", "precision": "bf16",
              "engine": "vllm", "variant": "default", "quality_only": False},
             {"cell_id": "m6-gemma-4-e4b|t2-m8i-4xl|bf16", "model": "m6-gemma-4-e4b", "target": "t2-m8i-4xl", "precision": "bf16",
              "engine": "vllm", "variant": "default", "quality_only": False}]
    gpu, cpu = cell_key(cells[0]["cell_id"]), cell_key(cells[1]["cell_id"])
    prices = {"g6.xlarge": {"usd_per_hour": 0.8048, "sku": "G6", "effective_date": "2026-11-01", "price_list_arn": "arn", "publication_date": "2026-11-01"},
              "m8i.4xlarge": {"usd_per_hour": 0.8467, "sku": "M8I", "effective_date": "2026-11-01", "price_list_arn": "arn", "publication_date": "2026-11-01"}}
    resolved = {"run_id": "r-tr", "run": d, "cells": cells, "prices": prices, "smoke": True, "run_hash": "h", "models_hash": "m",
                "targets_hash": "t", "harness_commit": "abc", "dspm_hashes": {"dataset_sha256": "00e1953d272f"}}
    items = generator.build_items(3, 0)
    sent = [it for it in items if it["size"] == "sentence"]
    ms = []
    for ck, ttft, tpot in ((gpu, 40, 8), (cpu, 120, 30)):
        for conc in (1, 4):
            for direction in ("en-de", "de-en"):
                m = metrics.summarize_level(_records(conc, 6, ttft, tpot, 40, 110), concurrency=conc, kind="generation")
                name = f"{direction}-sentence"
                m.update(cell_key=ck, level_key=f"{ck}__{name}__c{conc}__r0", name=name, batch=1, repeat=0, workload=None,
                         direction=direction, accuracy=False)
                ms.append(m)
    ev = {cpu: {"flags": ["amx_bf16", "amx_int8", "amx_tile"], "trace": {"gemm_total": 10, "gemm_amx": 10}, "dispatch": {}}}
    for m in ms:
        if m["cell_key"] == cpu:
            m["amx_counter"] = {"exe.amx_busy": 5_000, "cycles": 1_000_000}
    preds = {}
    for ck in (gpu, cpu):
        for direction in ("en-de", "de-en"):
            tgt = direction.split("-")[1]
            rows = []
            for it in sent:
                text = it["text"][tgt]
                if ck == cpu and cpu_quality == "worse":
                    text = "Etwas ganz anderes." if tgt == "de" else "Something else entirely."
                rows.append({"item_id": it["id"], "output": [{"text": text, "completion_tokens": 20}]})
            preds[f"{ck}/{direction}-sentence-accuracy.jsonl.gz"] = rows
    return report.RunInputs(resolved=resolved, metrics=ms, evidence=ev, predictions=preds, dataset_items=items)


def test_report_scores_translation_quality_and_groups_rows_by_direction():
    s = report.build_summary(_translation_inputs())
    tq = s["translation_quality"]
    assert len(tq) == 2
    for ck in tq:
        for sc in tq[ck].values():
            assert sc["n"] == 3 and sc["chrf"] == pytest.approx(100.0) and sc["truncated_pct"] == 0.0 and sc["flag"] is None
    cpu_sc = next(v for ck, v in tq.items() if "t2-m8i" in ck)["en-de-sentence"]
    assert cpu_sc["vs_reference_cell"]["exact_match_rate"] == 1.0 and cpu_sc["quality_delta"] == pytest.approx(0.0)
    rows = s["rows"]
    assert {r["direction"] for r in rows} == {"en-de", "de-en"} and all(r["text_size"] == "sentence" for r in rows)
    assert all(r["chars_s"] and r["usd_per_1m_chars"] for r in rows)
    assert set(s["per_model"]) == {"m6-gemma-4-e4b en-de-sentence", "m6-gemma-4-e4b de-en-sentence"}
    pm = s["per_model"]["m6-gemma-4-e4b en-de-sentence"]
    assert pm["unit"] == "chars_s" and pm["best_gpu"] and pm["cheapest_cpu"] and pm["capacity_equivalence"]
    cpu_rows = [r for r in rows if r["accelerator"] == "cpu"]
    assert all(r["meets_sla"] for r in cpu_rows), [r["sla_reason"] for r in cpu_rows]
    files = report.render(s, with_charts=False)
    md = files["summary.md"].decode()
    assert "## Translation quality" in md and "chrF vs reference" in md and "$/1M chars" in md
    assert "direction" in files["summary.csv"].decode().splitlines()[0]


def test_report_flags_nothing_for_bf16_but_records_a_cpu_that_translates_differently():
    s = report.build_summary(_translation_inputs("worse"))
    cpu_sc = next(v for ck, v in s["translation_quality"].items() if "t2-m8i" in ck)["en-de-sentence"]
    assert cpu_sc["chrf"] < 30 and cpu_sc["vs_reference_cell"]["chrf"] < 30 and cpu_sc["quality_delta"] > 0.5
    assert cpu_sc["flag"] is None          # BF16 vs BF16: reported, not flagged (only compressed formats are judged against the floor)
    cpu_rows = [r for r in s["rows"] if r["accelerator"] == "cpu"]
    assert all(r["quality_delta"] > 0.5 for r in cpu_rows)
