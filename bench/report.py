"""Run summary and reports (design 11-16 and 24.7, Requirements 10, 13, 14, 21, 23).

build_summary() turns everything a run wrote (level metrics, AMX evidence, quality
outputs, DSPM predictions, prices, capacity errors) into one summary dict; render() turns
that into summary.json, summary.csv, summary.md and PNG charts. Both are pure functions so
the same code runs in the aggregate_run Lambda, the CLI and the UI.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bench import (
    amx,
    cost,
    decision,
    dspm,
    dspm_scoring,
    embedding,
    metrics,
    pairing,
    quality,
    translation,
    utilization,
)
from bench.cellspec import cell_key
from bench.config import ModelSpec, ResolvedCell, RunConfig, load_targets

ROOT = Path(__file__).resolve().parents[1]
def _vcpus(instance_type: str) -> int:
    """From the instance catalog; a literal table here missed g6.2xlarge on 2026-11-09 and the
    capacity-equivalence column showed 0 G vCPUs."""
    s = pairing.spec(instance_type)
    return int(s["vcpus"]) if s else 0


SLA_PRESETS = {"inline_strict": {"p99_e2e_ms": 250.0}, "inline_standard": {"p99_e2e_ms": 1000.0},
               "bulk": {"p99_e2e_ms": 30000.0}}
TRANSFER = ("Speed results transfer to fine-tuned models of the same architecture, size and precision; "
            "accuracy results do not. The public checkpoints here are zero-shot or policy-prompted and will "
            "score lower than a production model fine-tuned on the customer's data.")
MANAGED_PRICES = ("Context, not like-for-like: Amazon Macie sensitive data discovery lists $1.00/GB for the first "
                  "50,000 GB per month (then $0.50 and $0.25); Google Sensitive Data Protection storage inspection "
                  "lists $1.00/GiB up to 50 TiB. Both include connectors, file parsing and their own detectors; the "
                  "figures here cover classification of already-extracted text only.")
MANAGED_MT_PRICES = ("Context, not like-for-like: managed machine translation is priced per character; Amazon Translate "
                     "lists $15 per million characters and Google Cloud Translation $20 per million characters (list "
                     "prices, 2026). The $ per 1M characters here is the instance's hourly price spread over the "
                     "characters it translates at that load, with no request routing or operations included.")
ATTRIBUTION = ("Dataset sources: NVIDIA Nemotron-PII (CC BY 4.0), Gretel synthetic_pii_finance_multilingual "
               "(Apache-2.0), and a seeded generator in this repository. All values are synthetic.")


@dataclass
class RunInputs:
    resolved: dict
    metrics: list[dict]
    evidence: dict[str, dict] = field(default_factory=dict)
    bringup: dict[str, dict] = field(default_factory=dict)
    quality: dict[str, list[dict]] = field(default_factory=dict)
    predictions: dict[str, list[dict]] = field(default_factory=dict)
    capacity: dict[str, Any] = field(default_factory=dict)
    hosts: dict[str, dict] = field(default_factory=dict)
    dataset_items: list[dict] | None = None
    dev_items: list[dict] | None = None
    raw_loader: Callable[[str, str], list[dict]] | None = None


def group_key(level_key: str) -> str:
    """Level key without its repeat suffix (__r<n>), so repeats pool together."""
    head, sep, last = level_key.rpartition("__")
    return head if sep and last.startswith("r") and last[1:].isdigit() else level_key


def _g(d: dict | None, *path, default=None):
    for p in path:
        if not isinstance(d, dict):
            return default
        d = d.get(p)
    return default if d is None else d


def _fmt(v, nd=1) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.{nd}f}"
    return str(v)


# ---------------------------------------------------------------------------- AMX and quality

def amx_verdicts(inp: RunInputs, cells: dict[str, ResolvedCell], targets: dict) -> dict[str, dict]:
    out = {}
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for m in inp.metrics:
        by_cell[m["cell_key"]].append(m)
    for ck, c in cells.items():
        if targets[c.target].accelerator != "cpu":
            continue
        if c.engine == "decider":
            # The decision model's server runs the torso in fp32 on CPU (the project's own setting), so AMX, which
            # has no fp32 path, cannot be used: not a configuration fault but the model's design. The row stays in
            # the rankings; the report and the summary say what the CPU side actually used.
            out[ck] = {"verdict": "NOT_APPLICABLE", "basis": None, "effective_precision": "fp32",
                       "reason": "the decision model's server runs the torso in fp32 on CPU; AMX has no fp32 path, so the CPU "
                                 "side uses AVX-512 (the project's own setting, same on every CPU)"}
            continue
        ev = inp.evidence.get(ck)
        if ev is None:
            out[ck] = {"verdict": "INVALID", "basis": None, "reason": "no AMX evidence collected",
                       "effective_precision": c.precision}
            continue
        busy = sum(_g(m, "amx_counter", "exe.amx_busy", default=0) for m in by_cell[ck])
        cycles = sum(_g(m, "amx_counter", "cycles", default=0) for m in by_cell[ck])
        have_counter = any(_g(m, "amx_counter", "exe.amx_busy") is not None for m in by_cell[ck])
        counter = {"exe.amx_busy": busy, "cycles": cycles} if have_counter else None
        e = amx.Evidence(cpu_flags=set(ev.get("flags", [])), counter=counter, trace=ev.get("trace"),
                         dispatch=ev.get("dispatch"))
        out[ck] = amx.verdict(e, precision=c.precision, engine=c.engine)
    return out


def quality_scores(inp: RunInputs, cells: dict[str, ResolvedCell], models) -> dict[str, dict]:
    """Scores against the model's reference cell (and the FP32 anchor for M3), plus flags."""
    out: dict[str, dict] = {}
    by_model_ref: dict[str, str] = {}
    for ck, c in cells.items():
        ref = models.get(c.model).quality_reference
        if ref and c.target == ref.target and c.precision == ref.precision:
            by_model_ref[c.model] = ck
    fp32 = {c.model: ck for ck, c in cells.items() if c.precision == "fp32"}
    cpu_bf16: dict[str, dict] = {}
    for ck, c in cells.items():
        rows = inp.quality.get(ck)
        ref_ck = by_model_ref.get(c.model)
        if not rows or not ref_ck or ref_ck == ck or ref_ck not in inp.quality:
            continue
        ref_rows = {r["id"]: r for r in inp.quality[ref_ck] if not r.get("error")}
        mine = {r["id"]: r for r in rows if not r.get("error")}
        if "text" in next(iter(mine.values()), {}):
            s = quality.score_generation({k: v["text"] for k, v in mine.items()},
                                         {k: v["text"] for k, v in ref_rows.items()})
        else:
            s = quality.score_classifier(mine, ref_rows)
            if c.model in fp32 and fp32[c.model] != ck and fp32[c.model] in inp.quality:
                anchor = {r["id"]: r for r in inp.quality[fp32[c.model]] if not r.get("error")}
                s["vs_fp32"] = quality.score_classifier(mine, anchor)
        s["reference_cell"] = ref_ck
        out[ck] = s
        # The CPU BF16 floor for the compressed formats: the vLLM/PyTorch BF16 cell when the run
        # has one, else the OpenVINO BF16 cell (bf16-ov).
        if quality.is_bf16(c.precision) and "rouge_l_mean" in s and (c.precision == "bf16" or c.model not in cpu_bf16):
            cpu_bf16[c.model] = s
    for ck, s in out.items():
        c = cells[ck]
        flag = (quality.generation_flag(s, cpu_bf16.get(c.model), precision=c.precision) if "rouge_l_mean" in s
                else quality.classifier_flag(s.get("vs_fp32"), precision=c.precision))
        s["flag"] = flag
        s["quality_delta"] = quality.quality_delta(s)
    return out


def decision_quality(inp: RunInputs, cells: dict[str, ResolvedCell], models, run: RunConfig) -> dict[str, dict]:
    """Per cell (decision suite): accuracy against the pool's gold labels per question type, mean confidence, and
    agreement with the model's GPU reference cell (predictions/<cell>/decide-<size>-accuracy)."""
    if run.suite != "decision" or not inp.dataset_items:
        return {}
    items_by_id = {it["id"]: it for it in inp.dataset_items if it.get("size") == run.prompt_set}
    records: dict[str, list[dict]] = {}
    for key, rows in inp.predictions.items():
        ck, name = key.split("/", 1)
        if not name.startswith("decide-") or ck not in cells:
            continue
        for r in rows:
            if r.get("error") or not r.get("output"):
                continue
            ids = r.get("item_ids") or ([r.get("item_id")] if r.get("item_id") else [])
            if ids:
                records.setdefault(ck, []).append({"item_id": ids[0], "output": r["output"]})
    by_model_ref: dict[str, str] = {}
    for ck, c in cells.items():
        ref = models.get(c.model).quality_reference
        if ref and c.target == ref.target and c.precision == ref.precision:
            by_model_ref[c.model] = ck
    out: dict[str, dict] = {}
    ref_answers: dict[str, dict] = {}
    for ck in sorted(records, key=lambda k: (k not in by_model_ref.values(), k)):   # reference cells first
        ref_ck = by_model_ref.get(cells[ck].model)
        reference = ref_answers.get(ref_ck) if ref_ck and ref_ck != ck else None
        sc = decision.score_cell(records[ck], items_by_id, reference)
        if ref_ck == ck:
            ref_answers[ck] = sc["answers"]
        sc.pop("answers", None)
        if ref_ck and ref_ck != ck:
            sc["reference_cell"] = ref_ck
        sc["size"] = run.prompt_set
        sc["flag"] = decision.flag(sc, precision=cells[ck].precision)
        sc["quality_delta"] = (1 - sc["vs_reference_cell"]["agreement"]) if sc.get("vs_reference_cell", {}).get("agreement") is not None else None
        out[ck] = sc
    return out


# ---------------------------------------------------------------------------- rows

def translation_quality(inp: RunInputs, cells: dict[str, ResolvedCell], models, run: RunConfig) -> dict[str, dict]:
    """Per cell and direction: chrF against the synthetic reference, agreement with the model's GPU
    BF16 reference cell, and a flag for a compressed format that falls behind the CPU BF16 cell.
    Predictions come from the accuracy levels (predictions/<cell>/<direction>-<size>-accuracy)."""
    if run.suite != "translation" or not inp.dataset_items:
        return {}
    size = run.prompt_set
    refs: dict[str, dict[str, str]] = {}                         # direction -> item_id -> reference text
    for it in inp.dataset_items:
        if it.get("size") != size:
            continue
        for d in translation.DIRECTIONS:
            _, tgt = translation.split_direction(d)
            refs.setdefault(d, {})[it["id"]] = it["text"][tgt]
    max_tokens = run.sampling.max_new_tokens
    outputs: dict[tuple[str, str], dict[str, str]] = {}
    tokens: dict[tuple[str, str], dict[str, int]] = {}
    for key, rows in inp.predictions.items():
        ck, name = key.split("/", 1)
        name = name.removesuffix(".jsonl.gz")
        if not name.endswith("-accuracy"):
            continue
        d = name[: -len(f"-{size}-accuracy")]
        if d not in refs or ck not in cells:
            continue
        for r in rows:
            if r.get("error") or not r.get("output"):
                continue
            first = r["output"][0] or {}
            outputs.setdefault((ck, d), {})[r["item_id"]] = first.get("text") or ""
            if first.get("completion_tokens") is not None:
                tokens.setdefault((ck, d), {})[r["item_id"]] = int(first["completion_tokens"])
    by_model_ref: dict[str, str] = {}
    for ck, c in cells.items():
        ref = models.get(c.model).quality_reference
        if ref and c.target == ref.target and c.precision == ref.precision:
            by_model_ref[c.model] = ck
    out: dict[str, dict] = {}
    for (ck, d), outs in sorted(outputs.items()):
        sc = translation.score(outs, refs[d], max_tokens=max_tokens, completion_tokens=tokens.get((ck, d)))
        sc["direction"] = d
        sc["size"] = size
        ref_ck = by_model_ref.get(cells[ck].model)
        if ref_ck and ref_ck != ck and (ref_ck, d) in outputs:
            sc["vs_reference_cell"] = translation.agreement(outs, outputs[(ref_ck, d)]) | {"cell": ref_ck}
        out.setdefault(ck, {})[f"{d}-{size}"] = sc
    # CPU BF16 floor per model and direction (the vLLM/PyTorch BF16 cell, else the OpenVINO one).
    floors: dict[tuple[str, str], dict] = {}
    for ck, per in out.items():
        c = cells[ck]
        if quality.is_bf16(c.precision):
            for name, sc in per.items():
                k = (c.model, name)
                if c.precision == "bf16" or k not in floors:
                    floors[k] = sc
    for ck, per in out.items():
        c = cells[ck]
        for name, sc in per.items():
            sc["flag"] = translation.flag(sc, floors.get((c.model, name)), precision=c.precision)
            vs = sc.get("vs_reference_cell") or {}
            sc["quality_delta"] = (1.0 - vs["chrf"] / 100.0) if vs.get("chrf") is not None else None
    return out


def embedding_quality(inp: RunInputs, cells: dict[str, ResolvedCell], models, run: RunConfig) -> dict[str, dict]:
    """Per cell: cosine agreement with the model's GPU BF16 reference cell and cross-lingual retrieval
    recall@1 over the pool (predictions/<cell>/embed-<size>-accuracy)."""
    if run.suite != "embedding":
        return {}
    outputs: dict[str, dict[str, list[float]]] = {}
    for key, rows in inp.predictions.items():
        ck, name = key.split("/", 1)
        if not name.startswith("embed-") or ck not in cells:
            continue
        for r in rows:
            if r.get("error") or not r.get("output"):
                continue
            vec = (r["output"][0] or {}).get("embedding")
            if vec:
                outputs.setdefault(ck, {})[r["item_id"]] = vec
    by_model_ref: dict[str, str] = {}
    for ck, c in cells.items():
        ref = models.get(c.model).quality_reference
        if ref and c.target == ref.target and c.precision == ref.precision:
            by_model_ref[c.model] = ck
    out: dict[str, dict] = {}
    for ck, outs in sorted(outputs.items()):
        ref_ck = by_model_ref.get(cells[ck].model)
        ref = outputs.get(ref_ck) if ref_ck and ref_ck != ck else None
        sc = embedding.score(outs, ref)
        if ref_ck and ref_ck != ck:
            sc["reference_cell"] = ref_ck
        sc["size"] = run.prompt_set
        out[ck] = sc
    floors: dict[str, dict] = {}
    for ck, sc in out.items():
        c = cells[ck]
        if quality.is_bf16(c.precision) and (c.precision == "bf16" or c.model not in floors):
            floors[c.model] = sc
    for ck, sc in out.items():
        c = cells[ck]
        sc["flag"] = embedding.flag(sc, floors.get(c.model), precision=c.precision)
        vs = sc.get("vs_reference_cell") or {}
        sc["quality_delta"] = (1.0 - vs["cosine_mean"]) if vs.get("cosine_mean") is not None else None
    return out


def intel_reference(run: RunConfig, models, config_dir: Path = ROOT / "config") -> list[dict]:
    """Intel's published rows for this run's models and token shape (config/intel_catalog_benchmarks.json),
    with per-core figures, for the intel suite's report."""
    if run.suite != "intel":
        return []
    path = config_dir / "intel_catalog_benchmarks.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    repos = {models.get(m).hf_repo for m in run.models}
    io = f"{run.intel.input_tokens}/{run.intel.output_tokens}" if run.intel else "128/128"
    rows = []
    for r in data["rows"]:
        if r["model_id"] in repos and r["io"] == io:
            per_core = (r["throughput_tok_s"] / r["total_cores"]) if r.get("total_cores") else None
            rows.append({**{k: r.get(k) for k in ("model_id", "cpu_model", "sockets", "total_cores", "memory_gb", "ddr5_mt_s", "io",
                                                   "max_concurrency", "throughput_tok_s", "tpot_ms", "ttft_ms", "vllm_version", "tested")},
                         "tok_s_per_core": per_core})
    return rows


def _throughput(row: dict) -> float | None:
    if row.get("direction"):
        return row["chars_s"]
    if row["workload"]:
        return row["docs_s"]
    return row["agg_out_tok_s"] if row["kind"] == "generation" else row["cls_s"]


def sla_for(row: dict, run: RunConfig) -> tuple[dict, str]:
    if run.suite == "dspm" and run.dspm:
        if row["workload"] == "W3":
            preset = run.dspm.sla_preset if run.dspm.sla_preset.startswith("inline") else "inline_standard"
        else:
            preset = "bulk"
        return {**SLA_PRESETS[preset], "max_error_pct": run.sla.max_error_pct}, preset
    s = run.sla
    return {"p99_ttft_ms": s.p99_ttft_ms, "p50_tpot_ms": s.p50_tpot_ms, "p99_e2e_ms": s.p99_e2e_ms,
            "max_error_pct": s.max_error_pct}, "run"


GENERATION_ONLY_TARGETS = ("p99_ttft_ms", "p50_tpot_ms")   # a classifier generates no tokens: first-word and per-word limits do not apply


def meets_sla(row: dict, sla: dict) -> tuple[bool, str]:
    checks = [("p99_ttft_ms", row.get("ttft_p99")), ("p50_tpot_ms", row.get("tpot_p50")),
              ("p99_e2e_ms", row.get("e2e_p99"))]
    if row.get("kind") != "generation":
        checks = [(k, v) for k, v in checks if k not in GENERATION_ONLY_TARGETS]
    for key, val in checks:
        lim = sla.get(key)
        if lim is None:
            continue
        if val is None:
            return False, f"{key} not measured"
        if val > lim:
            return False, f"{key} {val:.0f} > {lim:.0f}"
    if (row.get("error_pct") or 0) > sla.get("max_error_pct", 1.0):
        return False, f"error {row['error_pct']:.2f}% > {sla['max_error_pct']}%"
    # A quality flag (`quality_flag`) is a warning, not an exclusion (owner decision 2026-11-19): the generation
    # flag measures divergence from the GPU's text (two BF16 cells of the same weights agree only 0.73-0.76),
    # not a loss, so the row stays in the rankings and the flag travels with it.
    if row["accelerator"] == "cpu" and row.get("amx") not in ("VERIFIED", "NOT_APPLICABLE"):
        return False, "AMX not verified"
    return True, "meets SLA"


def make_row(m: dict, c: ResolvedCell, model: ModelSpec, target, price: dict | None, amx_v: dict | None,
             qual: dict | None, smoke: bool, tq: dict | None = None) -> dict:
    """tq: this cell's translation quality by level name (translation_quality), for translation rows."""
    usd = price["usd_per_hour"] if price else None
    gen = m["kind"] == "generation"
    workload = m.get("workload")
    direction = m.get("direction")
    name = m.get("name") or m["level_key"][len(m["cell_key"]) + 2:].split("__")[0]
    bucket = name.split("-", 1)[1] if workload and "-" in name else None
    if direction:
        qual = (tq or {}).get(name)        # chrF agreement with the GPU reference cell on this direction
    if model.type == "embedding" and tq is not None and "retrieval" in tq:
        qual = tq                          # embedding_quality: one entry per cell
    if model.type == "decision" and tq is not None and "accuracy_overall" in tq:
        qual = tq                          # decision_quality: one entry per cell
    row = {
        "model": model.id, "slot": model.slot, "license": model.license, "target": target.id,
        "target_label": target.label, "instance_type": target.instance_type, "accelerator": target.accelerator,
        "engine": c.engine, "precision": c.precision, "variant": c.variant, "workload": workload, "bucket": bucket,
        "direction": direction, "model_type": model.type,
        "text_size": (name[len(direction) + 1:] if direction else
                      m.get("text_size") or (name.split("-", 1)[1] if model.type == "embedding" and name.startswith("embed-") else None)),
        "chars_s": m.get("chars_per_s"), "chrf": (qual or {}).get("chrf") if direction else None,
        "conc": m["concurrency"], "batch": m.get("batch") or 1, "repeats": m.get("repeats", 1), "kind": m["kind"],
        "n_ok": m["n_ok"], "n_requests": m["n_requests"],
        "e2e_p1": _g(m, "e2e_ms", "p1"), "e2e_p50": _g(m, "e2e_ms", "p50"), "e2e_p99": _g(m, "e2e_ms", "p99"),
        "e2e_mean": _g(m, "e2e_ms", "mean"),
        "ttft_p1": _g(m, "ttft_ms", "p1"), "ttft_p50": _g(m, "ttft_ms", "p50"), "ttft_p99": _g(m, "ttft_ms", "p99"),
        "tpot_p50": _g(m, "tpot_ms", "p50"), "prefill_tok_s_p50": _g(m, "prefill_tok_s_per_request", "p50"),
        "decode_tok_s_p50": _g(m, "decode_tok_s_per_request", "p50"),
        "agg_out_tok_s": m.get("agg_out_tok_s"), "agg_in_tok_s": m.get("agg_in_tok_s"),
        "req_s": m.get("requests_per_s"), "cls_s": m.get("classifications_per_s"),
        "docs_s": m.get("documents_per_s") if gen else m.get("classifications_per_s"),
        "bytes_s": m.get("bytes_per_s"), "error_pct": m.get("error_pct"), "timeout_pct": m.get("timeout_pct"),
        "effective_conc": m.get("effective_concurrency"),
        "usd_per_hour": usd,
        "cpu_util_mean": _g(m, "resources", "cpu", "mean"), "gpu_util_mean": _g(m, "resources", "gpu", "mean"),
        "util": m.get("resources") or None,                    # host-level mean/p95/max per metric (bench/utilization.py)
        "amx": (amx_v or {}).get("verdict") if target.accelerator == "cpu" else "n/a",
        "amx_basis": (amx_v or {}).get("basis"), "effective_precision": (amx_v or {}).get("effective_precision",
                                                                                        c.precision),
        "quality_delta": (qual or {}).get("quality_delta"), "quality_flag": (qual or {}).get("flag"),
        "cv_e2e_p99": _g(m, "variance", "e2e_ms.p99", "cv"), "flags": ";".join(m.get("flags", [])),
        "smoke": smoke, "level_group": group_key(m["level_key"]), "cell_key": m["cell_key"],
    }
    if workload and gen and m.get("documents_per_s") is None:
        row["docs_s"] = m.get("requests_per_s")
    row["usd_per_1m_tokens"] = cost.usd_per_million_tokens(usd, row["agg_out_tok_s"]) if usd and gen else None
    row["usd_per_1k_requests"] = cost.usd_per_thousand_requests(usd, row["req_s"]) if usd else None
    row["usd_per_1m_items"] = cost.usd_per_million_items(usd, row["cls_s"]) if usd and not gen else None
    row["usd_per_1m_docs"] = cost.usd_per_million_items(usd, row["docs_s"]) if usd and workload else None
    row["usd_per_gb"] = cost.usd_per_gb(usd, row["bytes_s"]) if usd and workload else None
    row["gb_per_hour"] = cost.gb_per_hour(row["bytes_s"]) if workload else None
    row["usd_per_1m_chars"] = cost.usd_per_million_items(usd, row["chars_s"]) if usd and direction else None
    return row


def pooled_levels(inp: RunInputs) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m in inp.metrics:
        if m.get("accuracy"):
            continue
        groups[(m["cell_key"], group_key(m["level_key"]))].append(m)
    out = []
    for (ck, _), per in sorted(groups.items()):
        per = sorted(per, key=lambda x: x.get("repeat") or 0)
        if len(per) == 1 or inp.raw_loader is None:
            pooled = dict(per[0])
            pooled["repeats"] = len(per)
        else:
            recs = [r for m in per for r in inp.raw_loader(ck, m["level_key"])]
            pooled = metrics.combine_repeats(per, recs, concurrency=per[0]["concurrency"], kind=per[0]["kind"])
            # text_size was missing here, so a repeated (full-grade) embedding run lost it and its groups read
            # "embeddings, None" while the one-repeat smoke run kept it (2026-11-20).
            for k in ("cell_key", "level_key", "batch", "workload", "direction", "name", "text_size", "accuracy", "target_id"):
                pooled[k] = per[0].get(k)
            pooled["resources"] = utilization.merge_stats([m.get("resources") or {} for m in per]) or None
        out.append(pooled)
    return out


# ---------------------------------------------------------------------------- DSPM accuracy

def dspm_accuracy(inp: RunInputs, cells: dict[str, ResolvedCell], models, targets, run: RunConfig) -> dict:
    if not inp.dataset_items:
        return {}
    test = {x["id"]: x for x in inp.dataset_items}
    dev = {x["id"]: x for x in (inp.dev_items or [])}
    results: dict[str, dict] = {}
    preds_by: dict[tuple[str, str], dict[str, str | None]] = {}
    thresholds: dict[tuple[str, str], dict] = {}
    for key, rows in inp.predictions.items():
        ck, fname = key.split("/", 1)
        name = fname.removesuffix(".jsonl.gz").removesuffix("-accuracy")
        if name.startswith("dev-"):
            c = cells[ck]
            wl = name[4:].split("-")[0]
            scores = {r["item_id"]: dspm.mnli_score(r.get("output") or []) for r in rows if not r.get("error")}
            scores = {k: v for k, v in scores.items() if v is not None}
            thresholds[(ck, name[4:])] = dspm_scoring.choose_threshold(scores, [dev[i] for i in scores if i in dev],
                                                                       workload=wl)
    for key, rows in inp.predictions.items():
        ck, fname = key.split("/", 1)
        name = fname.removesuffix(".jsonl.gz").removesuffix("-accuracy")
        if name.startswith("dev-"):
            continue
        c = cells[ck]
        model = models.get(c.model)
        wl = name.split("-")[0]
        th = thresholds.get((ck, name), {}).get("threshold")
        preds = {}
        for r in rows:
            preds[r["item_id"]] = None if r.get("error") else dspm.decide(model.dspm_profile, wl, r.get("output"),
                                                                           threshold=th if th is not None else 0.5)
        items = [test[i] for i in preds if i in test]
        if wl == "W2":
            items = [{**x, "label": x["category_top"]} for x in items]
        s = dspm_scoring.score(items, preds, workload=wl)
        s["fitness"] = dspm_scoring.fitness(s, recall_min=run.dspm.fitness_recall_min, fpr_max=run.dspm.fitness_fpr_max)
        if model.dspm_profile == "M3":
            s["threshold"] = thresholds.get((ck, name))
        results.setdefault(ck, {})[name] = s
        preds_by[(ck, name)] = preds
    # CPU vs GPU agreement for the same model, precision and variant (Requirement 23.2-23.3).
    for ck, per in results.items():
        c = cells[ck]
        if targets[c.target].accelerator != "cpu":
            continue
        gpu = next((g for g, gc in cells.items() if gc.model == c.model and gc.variant == c.variant
                    and targets[gc.target].accelerator == "gpu" and gc.precision in (c.precision, "bf16")), None)
        for name, s in per.items():
            if gpu is None or (gpu, name) not in preds_by:
                continue
            agree = dspm_scoring.agreement(preds_by[(ck, name)], preds_by[(gpu, name)])
            ok, why = dspm_scoring.capacity_qualifies(s, results[gpu][name], agree,
                                                      agreement_min=run.dspm.agreement_min,
                                                      recall_drop_max_pp=run.dspm.recall_drop_max_pp)
            s["vs_gpu"] = {"gpu_cell": gpu, "agreement": agree, "qualifies": ok, "reason": why}
    return results


# ---------------------------------------------------------------------------- summary

def build_summary(inp: RunInputs, config_dir: Path = ROOT / "config") -> dict:
    res = inp.resolved
    run = RunConfig.model_validate(res["run"])
    from bench import catalog_store

    models = catalog_store.load_catalog()
    targets = load_targets(config_dir / "targets.yaml").resolved()
    cells = {cell_key(c["cell_id"]): ResolvedCell.model_validate(c) for c in res["cells"]}
    prices = res.get("prices", {})
    amx_v = amx_verdicts(inp, cells, targets)
    qual = quality_scores(inp, cells, models)
    tq = translation_quality(inp, cells, models, run)
    eq = embedding_quality(inp, cells, models, run)
    dq = decision_quality(inp, cells, models, run)
    smoke = bool(res.get("smoke", run.is_smoke))
    rows = []
    for m in pooled_levels(inp):
        c = cells[m["cell_key"]]
        t = targets[c.target]
        actual = (res.get("instance_types") or {}).get(c.target)       # capacity fallback, e.g. g6 -> g5
        if actual and actual != t.instance_type:
            t = t.model_copy(update={"instance_type": actual})
        row = make_row(m, c, models.get(c.model), t, prices.get(t.instance_type), amx_v.get(m["cell_key"]),
                       qual.get(m["cell_key"]), smoke, tq.get(m["cell_key"]) or eq.get(m["cell_key"]) or dq.get(m["cell_key"]))
        sla, preset = sla_for(row, run)
        row["meets_sla"], row["sla_reason"] = meets_sla(row, sla)
        row["sla_preset"] = preset
        rows.append(row)
    acc = dspm_accuracy(inp, cells, models, targets, run) if run.suite == "dspm" else {}
    # DSPM: a CPU row also needs accuracy parity with the GPU cell (Requirement 23.3).
    for row in rows:
        if row["workload"] and row["accelerator"] == "cpu" and row["meets_sla"]:
            name = row["workload"] + (f"-{row['bucket']}" if row["bucket"] else "")
            vs = _g(acc, row["cell_key"], name, "vs_gpu")
            if vs is not None and not vs["qualifies"]:
                row["meets_sla"], row["sla_reason"] = False, vs["reason"]
    per_model = analyze(rows, run)
    warnings = collect_warnings(rows, cells, models, amx_v, qual, smoke)
    for ck, sc in sorted(eq.items()):
        if sc.get("flag"):
            warnings.append(f"{cells[ck].cell_id}: embedding quality flag: {sc['flag']}")
    for ck, sc in sorted(dq.items()):
        if sc.get("flag"):
            warnings.append(f"{cells[ck].cell_id}: decision quality flag: {sc['flag']} (warning only; the row stays in the rankings)")
    for c in cells.values():
        if models.get(c.model).type == "decision" and targets[c.target].accelerator == "cpu":
            warnings.append(f"{c.cell_id}: the decision model's server runs the torso in fp32 on CPU (the project's own setting), "
                            "so this cell uses AVX-512, not AMX; its AMX verdict is not applicable.")
    for ck, per in sorted(tq.items()):
        for name, sc in sorted(per.items()):
            if sc.get("flag"):
                warnings.append(f"{cells[ck].cell_id} {name}: translation quality flag: {sc['flag']}")
            if (sc.get("truncated_pct") or 0) > 5:
                warnings.append(f"{cells[ck].cell_id} {name}: {sc['truncated_pct']:.0f}% of translations hit the "
                                f"{run.sampling.max_new_tokens}-token answer limit; raise it for this text size.")
    not_run = targets_not_run(cells, targets, rows, inp.capacity, run.purchase.max_launch_attempts)
    for nr in not_run:
        warnings.append(f"{nr['target']} DID NOT RUN ({nr['side']} side): {nr['reason']}")
    for tid, events in sorted((res.get("interruptions") or {}).items()):
        if events:
            warnings.append(f"{tid}: {len(events)} instance interruption(s); the affected levels were re-run on a "
                            f"replacement instance ({(res.get('lifecycle') or {}).get(tid)}).")
    # Pairing on the types that actually ran (a capacity fallback can differ from the request).
    ran = {**(res.get("instance_types_requested") or {}), **{k: v for k, v in (res.get("instance_types") or {}).items() if v}}
    pairs = pairing.judge_run(ran, {tid: targets[tid].accelerator for tid in ran if tid in targets}) if ran else []
    for j in pairs:
        if j["fair"] is False:
            warnings.append(f"UNFAIR PAIRING {j['cpu_target']} vs {j['gpu_target']}: {j['reason']}")
    return {"run_id": res["run_id"], "run": res["run"], "smoke": smoke, "rows": rows, "per_model": per_model,
            "amx": amx_v, "quality": qual, "dspm_accuracy": acc, "translation_quality": tq, "embedding_quality": eq, "decision_quality": dq,
            "intel_reference": intel_reference(run, models, config_dir), "prices": prices,
            "capacity_errors": inp.capacity, "bringup": inp.bringup, "hosts": inp.hosts, "warnings": warnings,
            "not_run": not_run, "dspm_hashes": res.get("dspm_hashes") or {}, "pairing": pairs,
            "provenance": {k: res.get(k) for k in ("run_hash", "models_hash", "targets_hash", "harness_commit",
                                                   "instances", "lifecycle", "interruptions", "purchase", "amis",
                                                   "built_images", "created_at")}}


def analyze(rows: list[dict], run: RunConfig) -> dict:
    """Per model (and workload): best GPU row, qualifying CPU rows, cheapest, capacity equivalence."""
    out: dict[str, dict] = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[analysis_key(r)].append(r)
    limit = run.capacity.max_cpu_instances_per_gpu
    for key, rs in sorted(groups.items()):
        gpu = [r for r in rs if r["accelerator"] == "gpu" and r["meets_sla"] and _throughput(r)]
        cpu = [r for r in rs if r["accelerator"] == "cpu" and r["meets_sla"] and _throughput(r)]
        best_gpu = max(gpu, key=_throughput) if gpu else None
        unit = ("chars_s" if rs[0].get("direction") else "docs_s" if rs[0]["workload"]
                else ("agg_out_tok_s" if rs[0]["kind"] == "generation" else "cls_s"))
        cheapest = None
        price_key = ("usd_per_1m_chars" if rs[0].get("direction") else "usd_per_1m_docs" if rs[0]["workload"] else
                     "usd_per_1m_tokens" if rs[0]["kind"] == "generation" else "usd_per_1m_items")
        priced = [r for r in cpu if r.get(price_key) is not None]
        if priced:
            cheapest = min(priced, key=lambda r: r[price_key])
        equivalence = []
        if best_gpu:
            for r in cpu:
                if r["usd_per_hour"] is None or best_gpu["usd_per_hour"] is None:
                    continue
                eq = cost.capacity_equivalence(
                    gpu_throughput=_throughput(best_gpu), cpu_throughput=_throughput(r),
                    cpu_vcpus=_vcpus(r["instance_type"]), gpu_vcpus=_vcpus(best_gpu["instance_type"]),
                    cpu_price=r["usd_per_hour"], gpu_price=best_gpu["usd_per_hour"])
                if eq is None:
                    continue
                d = asdict(eq)
                d.update(cpu_row=_row_ref(r), within_limit=d["cpu_instances_per_gpu"] <= limit)
                equivalence.append(d)
        out[key] = {"unit": unit, "best_gpu": _row_ref(best_gpu) if best_gpu else None,
                    "cpu_meeting_sla": [_row_ref(r) for r in cpu], "cheapest_cpu": _row_ref(cheapest) if cheapest else None,
                    "capacity_equivalence": equivalence,
                    "note": None if best_gpu else "no GPU row meets the SLA; capacity equivalence not computed"}
    return out


def analysis_key(r: dict) -> str:
    """The question a row answers: model, plus the DSPM workload and bucket, the translation
    direction and text size, or the classifier batch size."""
    key = r["model"]
    if r.get("direction"):
        return f"{key} {r['direction']}-{r['text_size']}"
    if r.get("model_type") == "embedding":
        return f"{key} {r.get('text_size') or 'embed'} b{r['batch']}"
    if r["workload"]:
        key += f" {r['workload']}" + (f"-{r['bucket']}" if r["bucket"] else "")
    elif r["kind"] == "classifier":
        key += f" b{r['batch']}"
    return key


def shown_precision(r: dict) -> str:
    """What the cell actually ran, when the AMX evidence relabelled it (an fp16 cell that ran bf16, an OpenVINO bf16
    cell that ran f32, the decision model's fp32 torso on CPU); otherwise the configured precision."""
    return r.get("effective_precision") or r["precision"]
def _row_ref(r: dict | None) -> dict | None:
    if r is None:
        return None
    return {k: r.get(k) for k in ("model", "target", "instance_type", "engine", "precision", "variant", "workload",
                                  "bucket", "direction", "conc", "batch")} | {"throughput": _throughput(r)}


def targets_not_run(cells, targets, rows, capacity: dict, max_attempts: int) -> list[dict]:
    """Planned targets that produced no measurement at all — the GPU side that never got an
    instance, for example. The reason comes from the capacity log so the report and the UI say
    *why* instead of leaving the side silently absent (seen 2026-11-11: 25 refusals, 'Not run')."""
    ran = {r["target"] for r in rows}
    out = []
    for tid in sorted({c.target for c in cells.values()} - ran):
        caps = capacity.get(tid) or []
        side = "GPU" if targets[tid].accelerator == "gpu" else "CPU"
        if caps:
            types = sorted({str(c.get("instance_type")) for c in caps if c.get("instance_type")})
            markets = sorted({str(c.get("market")) for c in caps if c.get("market")})
            mk = {"spot": "Spot", "on-demand": "On-Demand"}
            reason = (f"No capacity: AWS refused {len(caps)} launch request(s) for "
                      f"{', '.join(types) or 'the requested size'} ({' and '.join(mk.get(m, m) for m in markets) or 'requested market'}); "
                      f"the run gave up after at most {max_attempts} attempts. There are no {side} numbers in this report.")
        else:
            reason = f"No instance served it (no capacity refusal was logged). There are no {side} numbers in this report."
        out.append({"target": tid, "side": side, "instance_type": targets[tid].instance_type, "refusals": len(caps),
                    "reason": reason})
    return out


def collect_warnings(rows, cells, models, amx_v, qual, smoke) -> list[str]:
    w = []
    if smoke:
        w.append("SMOKE RUN: below the methodology defaults (200 requests x 3 repeats per level); "
                 "numbers are indicative only.")
    for mid in sorted({c.model for c in cells.values()}):
        m = models.get(mid)
        if m.non_commercial:
            w.append(f"{mid}: license {m.license} is non-commercial; results are for evaluation only.")
    for ck, v in sorted(amx_v.items()):
        if v["verdict"] == "NOT_APPLICABLE":
            continue                        # said once, in the decision warning below
        if v["verdict"] != "VERIFIED":
            w.append(f"{cells[ck].cell_id}: AMX {v['verdict']} ({v.get('reason')}); excluded from SLA and cost rankings.")
        if v.get("precision_flag"):
            w.append(f"{cells[ck].cell_id}: {v['precision_flag']}")
    for ck, q in sorted(qual.items()):
        if q.get("flag"):
            w.append(f"{cells[ck].cell_id}: quality flag: {q['flag']} (warning only; the row stays in the rankings)")
    for r in rows:
        for f in (r["flags"] or "").split(";"):
            if f.startswith(("prompt_token_mismatch", "cv_", "output_length_mismatch", "effective_concurrency", "short_window")):
                w.append(f"{r['model']} {r['target']} {r['precision']} c{r['conc']}: {f}")
    return w


# ---------------------------------------------------------------------------- rendering

CSV_COLUMNS = ("model", "slot", "model_type", "target", "target_label", "instance_type", "engine", "precision", "variant",
               "workload", "bucket", "direction", "text_size", "chars_s", "usd_per_1m_chars", "chrf",
               "conc", "batch", "repeats", "n_ok", "e2e_p1", "e2e_p50", "e2e_p99", "ttft_p1",
               "ttft_p50", "ttft_p99", "tpot_p50", "prefill_tok_s_p50", "decode_tok_s_p50", "agg_out_tok_s",
               "agg_in_tok_s", "req_s", "cls_s", "docs_s", "bytes_s", "error_pct", "timeout_pct", "effective_conc",
               "usd_per_hour", "usd_per_1m_tokens", "usd_per_1k_requests", "usd_per_1m_items", "usd_per_1m_docs",
               "usd_per_gb", "gb_per_hour", "amx", "amx_basis", "effective_precision", "quality_delta",
               "quality_flag", "meets_sla", "sla_reason", "sla_preset", "cpu_util_mean", "gpu_util_mean",
               "cv_e2e_p99", "flags", "smoke", "license")


def to_csv(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CSV_COLUMNS})
    return buf.getvalue().encode()


def _md_table(headers: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def to_markdown(s: dict, chart_names: list[str]) -> str:
    run = s["run"]
    out = [f"# Benchmark report: {run['name']} ({s['run_id']})", ""]
    if s["smoke"]:
        out += ["> **SMOKE RUN.** Fewer requests and repeats than the methodology defaults. Use these numbers "
                "to check the harness, not to decide.", ""]
    for nr in s.get("not_run") or []:
        out += [f"> **{nr['side']} SIDE DID NOT RUN** ({nr['target']}, {nr['instance_type']}): {nr['reason']} "
                "Nothing in this report compares the two sides.", ""]
    out += [f"Suite **{run['suite']}**, region {run['region']} ({run['az_id']}). Prices are On-Demand from the AWS "
            "Price List API; $/unit figures assume the instance is held at that load for a whole hour. Capacity, "
            "not cost, decides the verdict.", ""]
    for key, pm in s["per_model"].items():
        rows = [r for r in s["rows"] if analysis_key(r) == key]
        if not rows:
            continue
        out += [f"## {key}", ""]
        gen = rows[0]["kind"] == "generation"
        if rows[0].get("direction"):
            hdr = ["Target", "Instance", "Engine", "Precision", "C", "E2E P1/P50/P99 ms", "TTFT P50 ms", "TPOT P50 ms",
                   "chars/s", "Agg tok/s", "req/s", "err %", "$/hr", "$/1M chars", "AMX", "chrF", "Q delta", "SLA"]
        elif gen and not rows[0]["workload"]:
            hdr = ["Target", "Instance", "Engine", "Precision", "C", "E2E P1/P50/P99 ms", "TTFT P1/P50/P99 ms",
                   "TPOT P50 ms", "Prefill tok/s P50", "Agg tok/s", "req/s", "err %", "$/hr", "$/1M tok", "AMX",
                   "Q delta", "SLA"]
        else:
            item = ("docs" if rows[0]["workload"] else
                    {"decision": "decisions", "embedding": "embeddings"}.get(rows[0].get("model_type") or "", "cls"))
            hdr = ["Target", "Instance", "Engine", "Precision", "C", "E2E P1/P50/P99 ms", f"{item}/s", "err %", "$/hr",
                   f"$/1M {item}", "$/GB", "GB/h", "AMX", "Q delta", "SLA"]
        cheapest = pm.get("cheapest_cpu")
        body = []
        for r in sorted(rows, key=lambda r: (r["accelerator"] != "gpu", r["target"], r["precision"], r["conc"])):
            mark = "meets" if r["meets_sla"] else r["sla_reason"]
            if cheapest and _row_ref(r) == cheapest:
                mark = "**cheapest CPU meeting SLA**"
            elif pm.get("best_gpu") and _row_ref(r) == pm["best_gpu"]:
                mark = "**best GPU**"
            e2e = f"{_fmt(r['e2e_p1'], 0)}/{_fmt(r['e2e_p50'], 0)}/{_fmt(r['e2e_p99'], 0)}"
            if r.get("direction"):
                body.append([r["target"], r["instance_type"], r["engine"], shown_precision(r), str(r["conc"]), e2e,
                             _fmt(r["ttft_p50"], 0), _fmt(r["tpot_p50"]), _fmt(r["chars_s"], 0), _fmt(r["agg_out_tok_s"], 0),
                             _fmt(r["req_s"], 2), _fmt(r["error_pct"], 2), _fmt(r["usd_per_hour"], 3),
                             _fmt(r["usd_per_1m_chars"], 2), r["amx"] or "-", _fmt(r.get("chrf")), _fmt(r["quality_delta"], 3), mark])
            elif gen and not r["workload"]:
                body.append([r["target"], r["instance_type"], r["engine"], shown_precision(r), str(r["conc"]), e2e,
                             f"{_fmt(r['ttft_p1'], 0)}/{_fmt(r['ttft_p50'], 0)}/{_fmt(r['ttft_p99'], 0)}",
                             _fmt(r["tpot_p50"]), _fmt(r["prefill_tok_s_p50"], 0), _fmt(r["agg_out_tok_s"], 0),
                             _fmt(r["req_s"], 2), _fmt(r["error_pct"], 2), _fmt(r["usd_per_hour"], 3),
                             _fmt(r["usd_per_1m_tokens"], 2), r["amx"] or "-", _fmt(r["quality_delta"], 3), mark])
            else:
                body.append([r["target"], r["instance_type"], r["engine"], shown_precision(r), str(r["conc"]), e2e,
                             _fmt(r["docs_s"] if r["workload"] else r["cls_s"], 1), _fmt(r["error_pct"], 2),
                             _fmt(r["usd_per_hour"], 3),
                             _fmt(r["usd_per_1m_docs"] if r["workload"] else r["usd_per_1m_items"], 2),
                             _fmt(r["usd_per_gb"], 4), _fmt(r["gb_per_hour"], 3), r["amx"] or "-",
                             _fmt(r["quality_delta"], 3), mark])
        out += [_md_table(hdr, body), ""]
        if pm["capacity_equivalence"]:
            out += ["Capacity equivalence (CPU instances needed to match the best GPU row's SLA-meeting throughput):",
                    ""]
            eq_rows = [[e["cpu_row"]["instance_type"], e["cpu_row"]["engine"], e["cpu_row"]["precision"],
                        str(e["cpu_row"]["conc"]), str(e.get("cpu_instances_per_gpu")), str(e.get("standard_vcpus_needed")),
                        str(e.get("gpu_vcpus_replaced")), _fmt(e.get("hourly_cost_ratio"), 2),
                        "yes" if e["within_limit"] else "no (Conditional)"] for e in pm["capacity_equivalence"]]
            out += [_md_table(["CPU instance", "Engine", "Precision", "C", "Instances per GPU", "Standard vCPUs",
                               "G vCPUs (GPU)", "Hourly cost ratio", "Within limit"], eq_rows), ""]
        elif pm.get("note"):
            out += [f"_{pm['note']}_", ""]
    if s.get("dspm_accuracy"):
        out += dspm_markdown(s)
    if s.get("translation_quality"):
        out += translation_markdown(s)
    if s.get("embedding_quality"):
        out += embedding_markdown(s)
    if s.get("intel_reference"):
        out += intel_markdown(s)
    if chart_names:
        out += ["## Charts", ""] + [f"![{n}](charts/{n})" for n in chart_names] + [""]
    out += ["## Price source", ""]
    for t, p in sorted(s["prices"].items()):
        out.append(f"- {t}: ${p['usd_per_hour']}/h, SKU {p['sku']}, effective {p['effective_date']}, "
                   f"price list {p.get('price_list_arn')} (published {p.get('publication_date')})")
    out.append("")
    if s["capacity_errors"]:
        out += ["## Capacity errors seen while starting instances", ""]
        for t, events in s["capacity_errors"].items():
            for e in events:
                out.append(f"- {t}: {e['time']} {e['code']} ({e.get('instance_type')}, {e.get('availability_zone')})")
        out.append("")
    if s["warnings"]:
        out += ["## Warnings", ""] + [f"- {w}" for w in s["warnings"]] + [""]
    prov = s["provenance"]
    out += ["## Provenance", "", f"- run hash {prov.get('run_hash')}, models {prov.get('models_hash')}, "
            f"targets {prov.get('targets_hash')}", f"- harness commit {prov.get('harness_commit')}",
            f"- AMIs {json.dumps(prov.get('amis'))}",
            f"- instances {json.dumps(prov.get('instances'))}, market {json.dumps(prov.get('lifecycle'))} "
            f"(requested {((prov.get('purchase') or {}).get('type'))}); cost columns use On-Demand prices"]
    for j in s.get("pairing") or []:
        out.append(f"- pairing {j['cpu_target']} ({j['cpu']}) vs {j['gpu_target']} ({j['gpu']}): "
                   f"{'fair' if j['fair'] else 'UNFAIR'}, GPU/CPU hourly price ratio {j['ratio']}")
    if s.get("dspm_hashes"):
        out.append(f"- DSPM dataset and policy {json.dumps(s['dspm_hashes'], sort_keys=True)}")
    out.append("")
    return "\n".join(out)


def dspm_markdown(s: dict) -> list[str]:
    out = ["## DSPM/DLP accuracy", "", TRANSFER, ""]
    body, hard_rows, cov_rows = [], [], []
    for ck, per in sorted(s["dspm_accuracy"].items()):
        for name, a in sorted(per.items()):
            r, f, p = a["recall"], a["fpr"], a["precision"]
            vs = a.get("vs_gpu") or {}
            body.append([ck.replace("__", " "), name, str(a["n"]),
                         f"{_fmt(r['value'], 3)} [{_fmt(r['ci95'][0], 3)}, {_fmt(r['ci95'][1], 3)}]",
                         f"{_fmt(f['value'], 3)} [{_fmt(f['ci95'][0], 3)}, {_fmt(f['ci95'][1], 3)}]",
                         _fmt(p["value"], 3), _fmt(a["f1"], 3), _fmt(a["format_adherence"]["value"], 3),
                         _fmt(_g(vs, "agreement", "value"), 3), "yes" if vs.get("qualifies") else (vs.get("reason") or "-"),
                         "fit" if a["fitness"]["fit"] else "below floors"])
            for h, v in a["hard_negative_fpr"].items():
                hard_rows.append([ck.replace("__", " "), name, h, str(v["n"]), _fmt(v["value"], 3)])
            if ck.startswith(("d1-", "d2-")):
                for cat, v in a["per_category_recall"].items():
                    cov_rows.append([ck.replace("__", " "), name, cat, str(v["n"]), _fmt(v["value"], 3)])
    out += [_md_table(["Cell", "Workload", "n", "Recall [95% CI]", "FPR [95% CI]", "Precision", "F1",
                       "Format adherence", "Agreement with GPU", "CPU qualifies", "Fitness"], body), ""]
    if hard_rows:
        out += ["### Hard negatives (false positive rate by type)", "",
                _md_table(["Cell", "Workload", "Hard case", "n", "FPR"], hard_rows), ""]
    if cov_rows:
        out += ["### D1/D2 coverage (recall by policy category)", "",
                "D1 has no health or confidential-business categories and D2 detects shared PII in chat, so "
                "their recall there is expected to be low.", "",
                _md_table(["Cell", "Workload", "Category", "n", "Recall"], cov_rows), ""]
    out += [MANAGED_PRICES, "", ATTRIBUTION, ""]
    return out


def translation_markdown(s: dict) -> list[str]:
    out = ["## Translation quality", "",
           "chrF (character n-gram F-score, 0-100) of each cell's translations against the synthetic reference, "
           "and against the model's GPU BF16 cell on the same items. Read the figures relative to each other: a "
           "paraphrase lowers chrF for every cell alike. Length ratio is translated characters over reference "
           "characters (1.0 = same length); truncated is the share of answers that hit the token limit.", ""]
    body = []
    for ck, per in sorted(s["translation_quality"].items()):
        for name, a in sorted(per.items()):
            vs = a.get("vs_reference_cell") or {}
            body.append([ck.replace("__", " "), name, str(a["n"]), _fmt(a["chrf"]), _fmt(a["chrf_p10"]),
                         _fmt(a["length_ratio"], 2), _fmt(a.get("truncated_pct"), 1), _fmt(vs.get("chrf")),
                         _fmt(vs.get("exact_match_rate"), 3), a.get("flag") or "-"])
    out += [_md_table(["Cell", "Direction", "n", "chrF vs reference", "chrF P10", "Length ratio", "Truncated %",
                       "chrF vs GPU BF16", "Identical answers", "Flag"], body), "", MANAGED_MT_PRICES, ""]
    return out


def embedding_markdown(s: dict) -> list[str]:
    out = ["## Embedding quality", "",
           "Cosine similarity of each cell's vectors to the GPU BF16 cell's vectors for the same texts (1.0 = "
           "identical direction), and cross-lingual retrieval: for every German or Spanish text, is the nearest "
           "English text in the pool its own translation (recall@1)? The retrieval score needs no reference cell.", ""]
    body = []
    for ck, a in sorted(s["embedding_quality"].items()):
        vs = a.get("vs_reference_cell") or {}
        rt = a.get("retrieval") or {}
        body.append([ck.replace("__", " "), str(a["n"]), str(a.get("dim")), _fmt(vs.get("cosine_mean"), 4), _fmt(vs.get("cosine_min"), 4),
                     _fmt((rt.get("de") or {}).get("recall_at_1"), 3), _fmt((rt.get("es") or {}).get("recall_at_1"), 3), a.get("flag") or "-"])
    out += [_md_table(["Cell", "n", "dim", "cosine vs GPU (mean)", "cosine vs GPU (min)", "DE->EN recall@1", "ES->EN recall@1", "Flag"], body), ""]
    return out


def intel_markdown(s: dict) -> list[str]:
    out = ["## Intel's published figures for this model", "",
           "From the Intel Software Catalog (config/intel_catalog_benchmarks.json): Intel's own measurements on two-socket "
           "Xeon 6 servers, vLLM v0.18, TP=4, at the highest concurrency that keeps TPOT <= 100 ms and TTFT <= 3 s. Compare "
           "per core: an AWS slice has a fraction of a socket's cores and of its memory bandwidth.", ""]
    body = [[r["model_id"], f"{r['sockets']}x {r['cpu_model']}", str(r["total_cores"]), f"{r['memory_gb']} GB DDR5-{r['ddr5_mt_s']}",
             str(r["max_concurrency"]), _fmt(r["throughput_tok_s"], 0), _fmt(r["tok_s_per_core"], 1), _fmt(r["tpot_ms"], 0), _fmt(r["ttft_ms"], 0),
             r.get("tested") or "-"] for r in s["intel_reference"]]
    out += [_md_table(["Model", "System", "Cores", "Memory", "Concurrency", "tok/s", "tok/s per core", "TPOT ms", "TTFT ms", "Tested"], body), ""]
    return out


def charts(s: dict) -> dict[str, bytes]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files: dict[str, bytes] = {}
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in s["rows"]:
        by_model[analysis_key(r).replace(" ", "-")].append(r)
    specs = [("e2e_p99", "P99 E2E latency (ms)"), ("e2e_p50", "P50 E2E latency (ms)"), ("ttft_p99", "P99 TTFT (ms)"),
             ("agg_out_tok_s", "Aggregate output tok/s"), ("usd_per_1m_tokens", "$ per 1M output tokens"),
             ("docs_s", "Documents/s"), ("usd_per_gb", "$ per GB classified"), ("cls_s", "Classifications/s"),
             ("chars_s", "Source characters translated/s"), ("usd_per_1m_chars", "$ per 1M characters translated")]
    for name, rows in sorted(by_model.items()):
        for metric, label in specs:
            pts = [r for r in rows if r.get(metric) is not None and r["batch"] in (1, None)]
            if len({r["conc"] for r in pts}) < 2:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            series: dict[str, list[tuple[int, float]]] = defaultdict(list)
            for r in pts:
                series[f"{r['target']} {r['precision']}{'' if r['variant'] == 'default' else ' ' + r['variant']}"].append(
                    (r["conc"], r[metric]))
            for lab, xy in sorted(series.items()):
                xy.sort()
                style = "-o" if "t1-" in lab else "--s"
                ax.plot([x for x, _ in xy], [y for _, y in xy], style, label=lab)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Concurrency")
            ax.set_ylabel(label)
            ax.set_title(f"{name}: {label}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=110)
            plt.close(fig)
            files[f"{name}-{metric}.png"] = buf.getvalue()
    return files


def render(s: dict, *, with_charts: bool = True) -> dict[str, bytes]:
    png = charts(s) if with_charts else {}
    files = {"summary.json": json.dumps(s, indent=1, sort_keys=True, default=str).encode(),
             "summary.csv": to_csv(s["rows"]), "summary.md": to_markdown(s, sorted(png)).encode()}
    files.update({f"charts/{k}": v for k, v in png.items()})
    return files
