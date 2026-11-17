"""Everything the wizard needs to offer choices, with plain-language names and explanations.
The reader is a product manager: comfortable with "10 concurrent requests" and "batch versus
single request", not with engine flags. Technical identifiers are kept alongside so the
generated run config maps back to the YAML files."""

from __future__ import annotations

import json
from pathlib import Path

from bench import catalog_store, pairing
from bench.config import load_targets

ROOT = Path(__file__).resolve().parents[2]

USE_CASES = {
    "standard": {
        "name": "Chat and text generation",
        "short": "How fast each hardware option answers typical assistant requests, and what it costs.",
        "long": ("Measures time to the first word, time per generated word, how many requests per second a "
                 "machine sustains, and the cost per million generated tokens. Also compares answer quality "
                 "with the GPU's. Use it for chatbots, summarization, drafting and similar generative work."),
        "icon": "chat",
    },
    "dspm": {
        "name": "Sensitive-data detection (DSPM / DLP)",
        "short": "How fast and how accurately each hardware option flags sensitive documents and messages.",
        "long": ("Classifies documents and messages as sensitive or not with a synthetic dataset of 3,000 "
                 "documents whose answers are known. Reports documents per second, cost per GB scanned and "
                 "detection accuracy with confidence intervals. Use it for data-store scanners and data-loss "
                 "prevention gates."),
        "icon": "shield",
    },
}

MODELS = {
    "m1-qwen2.5-3b": ("Qwen2.5 3B Instruct", "General chat model, 3 billion parameters. Research license: "
                                              "results are for evaluation only, not for a commercial decision on this model."),
    "m2-qwen3-4b-2507": ("Qwen3 4B Instruct", "General chat model, 4 billion parameters, Apache-2.0 license."),
    "m3-roberta-large-mnli": ("RoBERTa-large (classifier)", "Text classifier, 0.36 billion parameters. Does not generate "
                                                            "text; answers 'does this text match this statement'. Very "
                                                            "fast, lower accuracy."),
    "m4-gemma-2-2b": ("Gemma 2 2B Instruct", "Google chat model, 2.6 billion parameters, Gemma license."),
    "m5-flan-t5-xl": ("Flan-T5 XL", "Instruction-following model of the older encoder-decoder type, 2.85 billion "
                                    "parameters. Many production classifiers still use this family."),
    "d1-privacy-filter": ("OpenAI privacy-filter (PII detector)", "Purpose-built detector, 1.4 billion parameters: marks "
                                                                  "the exact words that are personal data. Sensitive-data use case only."),
    "d2-roblox-pii-v2": ("Roblox PII classifier v2", "Purpose-built classifier, 0.56 billion parameters: flags chat "
                                                     "messages with personal data. Sensitive-data use case only."),
}
MODEL_KIND = {"decoder": "Chat / generation", "encoder-decoder": "Instruction model (encoder-decoder)",
              "encoder-classifier": "Classifier", "token-classifier": "Detector (marks words)"}

ENGINES = {"vllm": ("vLLM", "The most common open-source serving software for chat models, on GPUs and Intel CPUs."),
           "ovms": ("OpenVINO", "Intel's optimized CPU serving stack (OpenVINO Model Server)."),
           "thin-openvino": ("OpenVINO", "OpenVINO through a small serving wrapper, for models the model server does not host."),
           "thin-torch": ("PyTorch", "Plain PyTorch through a small serving wrapper: the simplest, least optimized way to serve.")}

PRECISIONS = {
    "bf16": ("BF16 (standard)", "Full-quality number format on GPUs and AMX CPUs. The like-for-like baseline."),
    "bf16-ov": ("BF16 (OpenVINO)", "The same full-quality BF16 as the GPU side, served by OpenVINO on AMX: the "
                                   "like-for-like baseline on this option. Tick it next to INT8/INT4 to see what "
                                   "compression buys."),
    "fp16": ("FP16 (half precision)", "Half precision on the GPU. On CPUs it would need the AMX-FP16 feature, which the "
                                      "AWS Intel instances offered here do not expose, so it is greyed out for the CPU options."),
    "w8a8": ("INT8 compressed", "8-bit weights and activations: half the memory, faster on AMX, small quality loss."),
    "int8": ("INT8 compressed (OpenVINO)", "8-bit weights in OpenVINO's format: smaller and faster, small quality loss."),
    "int4": ("INT4 compressed (OpenVINO)", "4-bit weights: smallest and fastest, larger quality loss."),
    # The model catalog names the OpenVINO exports int8-ov / int4-ov.
    "int8-ov": ("INT8 compressed (OpenVINO)", "8-bit weights in OpenVINO's format: smaller and faster, small quality loss."),
    "int4-ov": ("INT4 compressed (OpenVINO)", "4-bit weights: smallest and fastest, larger quality loss."),
    "fp32": ("FP32 reference", "Full 32-bit precision; used only as the quality reference."),
}

PROMPT_SETS = {
    "short": ("Short prompts", "~128 tokens: a question or two sentences. Dominated by answer generation."),
    "medium": ("Medium prompts", "~1,024 tokens: about a page, like summarizing an email thread."),
    "long": ("Long prompts", "~4,096 tokens: a long document. Dominated by reading speed, which differs a lot "
                             "between CPU and GPU."),
}
WORKLOADS = {
    "W1": ("Bulk document scan: Sensitive / Not sensitive",
           "Whole document chunks (~500 and ~2,000 tokens) classified as sensitive or not, as a data-store scanner does."),
    "W2": ("Bulk document scan: which category",
           "Same documents; name the category: government ID, personal, financial, health, credentials, confidential, or none."),
    "W3": ("Inline DLP: Allow / Block a message",
           "Chat messages, emails and AI prompts, decided in real time before they leave the company."),
    "W4": ("Database columns: which category",
           "A table header plus sample rows; name the sensitive category of the table, or none."),
}
PRESETS = {
    "inline_strict": ("Inline, strict", "Every decision within 250 ms: blocking a message while the user waits.", 250),
    "inline_standard": ("Inline, standard", "Every decision within 1 second: real-time checks where a pause is acceptable.", 1000),
    "bulk": ("Bulk scan", "Within 30 seconds; maximize documents per second. Scanning data at rest.", 30000),
}
DEPTHS = {
    "quick": ("Quick check", "50 requests for each measurement, run once. Shows the shape of the result in under an "
                             "hour; the report marks every number as indicative only (SMOKE), not a basis for a decision.", 50, 1),
    "full": ("Full test", "200 requests for each measurement, repeated 3 times: the minimum for trustworthy percentiles "
                          "and for seeing whether a result repeats.", 200, 3),
}
REQUEST_MODES = {
    "single": ("Single request", "One request at a time. Shows the best-case latency of each machine."),
    "concurrent": ("Concurrent requests", "Several requests in flight at the same time, as a service under load. Pick "
                                          "how many; each number is measured separately, and the verdict looks for the "
                                          "highest one where the machine still meets your targets."),
    "batch": ("Batched requests", "Several inputs per request, N requests in flight. Available for classifier and "
                                  "detector models, which score inputs in batches; chat models take one prompt per "
                                  "request and the server batches concurrent requests itself."),
}
PURCHASE = {
    "spot": ("Spot", "Spare capacity at a discount; AWS may reclaim it with two minutes' notice. The run then launches "
                     "a replacement and re-runs only the unfinished measurements. Cost columns always use "
                     "On-Demand prices so results stay comparable."),
    "on-demand": ("On-Demand", "Guaranteed for the run, full price. Use when Spot keeps getting interrupted."),
}
CONCURRENCY_HINTS = {1: "one user", 4: "a small team", 8: "a department", 16: "a busy service", 32: "a large service",
                     64: "peak load"}
# Why a number format is greyed out for a CPU option: the CPU feature it needs (models.yaml
# requires_cpu_flag) is not in the option's cpu_features (targets.yaml, measured on booted hosts).
CPU_FLAG_NOTES = {
    "amx_fp16": ("needs the AMX-FP16 CPU feature, which the AWS instances this tool offers do not expose: every m8i and "
                 "c8i host it has booted (36 so far, Xeon 6975P-C) reports AMX-BF16 and AMX-INT8 but not AMX-FP16, and "
                 "m7i (Sapphire Rapids) never had it. No instance size changes this; keep BF16 or INT8 for the CPU side."),
}


def cpu_flag_note(flag: str) -> str:
    return CPU_FLAG_NOTES.get(flag, f"needs the {flag} CPU feature, which this option's instances do not have.")


def _targets():
    return {tid: t for tid, t in load_targets(ROOT / "config/targets.yaml").resolved().items() if t.enabled}


def hardware_options() -> list[dict]:
    """Deployed hardware options: one per software stack per accelerator. Targets that differ
    only by default instance size (t2-m8i-4xl / t2-c8i-4xl / t2-m8i-8xl are all vLLM on AMX)
    collapse into one option; the size is chosen separately in the wizard."""
    out = []
    seen: set[tuple] = set()
    for tid, t in _targets().items():
        key = (t.accelerator, tuple(sorted(t.engines)))
        if key in seen:
            continue
        seen.add(key)
        engines = list(dict.fromkeys(ENGINES[e][0] for e in t.engines))
        spec = pairing.spec(t.instance_type) or {}
        out.append({
            "id": tid, "accelerator": t.accelerator, "cpu_features": sorted(t.cpu_features),
            "software": engines, "engines": list(t.engines),
            "software_help": " ".join(ENGINES[e][1] for e in t.engines),
            "default_instance_type": t.instance_type, "alternates": list(t.alternates),
            "label": (f"GPU · NVIDIA · {' + '.join(engines)}" if t.accelerator == "gpu"
                      else f"CPU · Intel Xeon with AMX · {' + '.join(engines)}"),
            "default_spec": spec,
        })
    return out


def catalog() -> dict:
    models = catalog_store.load_catalog()
    cat = pairing.load_catalog()
    mods = []
    for m in models.models:
        if not m.enabled:
            continue
        name, desc = _display(m)
        precs = {}
        for p, spec in m.precisions.items():
            if spec.enabled:
                precs[p] = {"label": PRECISIONS.get(p, (p, ""))[0], "help": PRECISIONS.get(p, (p, ""))[1],
                            "engines": list(spec.engines), "quality_only": bool(spec.quality_only),
                            "requires_cpu_flag": spec.requires_cpu_flag}
        mods.append({"id": m.id, "name": name, "description": desc, "kind": MODEL_KIND.get(m.type, m.type),
                     "type": m.type, "params_b": round(m.params / 1e9, 2), "license": m.license,
                     "commercial_ok": m.license not in ("qwen-research",), "use_cases": list(m.suites),
                     "precisions": precs, "batch_capable": m.type not in ("decoder", "encoder-decoder"),
                     "batch_sizes_default": (m.classifier.batch_sizes if m.classifier else [1]),
                     "hf_repo": m.hf_repo, "user_added": m.user_added})
    insts = {t: {**s, "type": t} for t, s in cat["instances"].items() if s.get("usd_per_hour")}
    return {
        "use_cases": USE_CASES, "models": mods, "hardware": hardware_options(),
        "instances": insts, "price_snapshot": cat["price_snapshot"],
        "prompt_sets": {k: {"label": v[0], "help": v[1]} for k, v in PROMPT_SETS.items()},
        "workloads": {k: {"label": v[0], "help": v[1]} for k, v in WORKLOADS.items()},
        "presets": {k: {"label": v[0], "help": v[1], "p99_e2e_ms": v[2]} for k, v in PRESETS.items()},
        "depths": {k: {"label": v[0], "help": v[1], "requests": v[2], "passes": v[3]} for k, v in DEPTHS.items()},
        "request_modes": {k: {"label": v[0], "help": v[1]} for k, v in REQUEST_MODES.items()},
        "purchase": {k: {"label": v[0], "help": v[1]} for k, v in PURCHASE.items()},
        "concurrency_hints": CONCURRENCY_HINTS,
        "cpu_flag_notes": CPU_FLAG_NOTES,
        "precisions": {k: {"label": v[0], "help": v[1]} for k, v in PRECISIONS.items()},
        "engines": {k: {"label": v[0], "help": v[1]} for k, v in ENGINES.items()},
        "fair_ratio": list(pairing.FAIR_RATIO),
        "account": "346399954218", "region": "us-east-1",
    }


def _display(m) -> tuple[str, str]:
    """Name and one-line description: the shipped text for shipped models; for a model added in
    Setup, the repository's own name and the onboarding note."""
    if m.id in MODELS:
        return MODELS[m.id]
    return m.hf_repo.split("/")[-1], m.notes or "added from Hugging Face in Setup"


def model_name(mid: str) -> str:
    if mid in MODELS:
        return MODELS[mid][0]
    try:
        return _display(catalog_store.load_catalog().get(mid))[0]
    except KeyError:
        return mid


def dataset_info(use_case: str) -> dict:
    """What data the use case runs on, with counts; samples come from /api/data/sample."""
    if use_case == "standard":
        man_p = ROOT / "prompts/generated/manifest.json"
        if not man_p.exists():
            return {"available": False, "reason": "prompt sets are not built locally (tools/build_prompts.py)"}
        man = json.loads(man_p.read_text())
        sets = {}
        # The manifest synced from the bucket may carry only pool sizes and hashes (bench.pools.manifest);
        # the per-set statistics exist when tools/build_prompts.py or the onboarding job wrote them.
        for key, info in (man.get("sets") or {}).items():
            mid, name = key.split("/", 1)
            if name in PROMPT_SETS:
                sets.setdefault(name, {"label": PROMPT_SETS[name][0], "help": PROMPT_SETS[name][1], "models": {}})
                sets[name]["models"][mid] = {"n": info["n"], "min_tokens": info["min_tokens"],
                                             "max_tokens": info["max_tokens"], "target": info["target"]}
        return {"available": True, "kind": "prompts", "sets": sets,
                "sources": [{"name": "Project Gutenberg", "role": "prompt text", "license": "public domain"},
                            {"name": "databricks-dolly-15k", "role": "quality check (50 instructions)", "license": "CC BY-SA 3.0"},
                            {"name": "SNLI", "role": "quality check for classifiers (50 pairs)", "license": "CC BY-SA 4.0"}],
                "summary": ("Fixed prompts cut from public-domain books to a target length and wrapped in each model's own "
                            "chat format, so every model reads the same text. Answers are generated to a fixed length so "
                            "timings are comparable.")}
    man_p = ROOT / "data/dspm/v1/manifest.json"
    if not man_p.exists():
        return {"available": False, "reason": "the dataset is not built locally (tools/dspm/build_dataset.py)"}
    man = json.loads(man_p.read_text())
    items = _dspm_items()
    per = {}
    for w, (label, help_) in WORKLOADS.items():
        ws = [x for x in items if x["workload"] == w]
        if w == "W2":                        # W2 asks the category question on W1's short documents
            ws = [x for x in items if "W2" in (x.get("workloads") or [])]
        per[w] = {"label": label, "help": help_, "documents": len(ws), "positive": sum(1 for x in ws if x["sensitive"]),
                  "hard_negatives": sum(1 for x in ws if x.get("hard_case")),
                  "labels": sorted({x["label"] for x in ws}),
                  "hard_cases": sorted({x["hard_case"] for x in ws if x.get("hard_case")}),
                  "note": "Reuses the short documents of the bulk scan with a category question." if w == "W2" else None}
    return {"available": True, "kind": "documents", "version": man.get("version"),
            "fingerprint": str(man.get("dataset_sha256"))[:12], "documents": len(items), "workloads": per,
            "sources": [{"name": "NVIDIA Nemotron-PII", "role": "documents with marked personal data", "license": "CC BY 4.0"},
                        {"name": "Gretel synthetic PII finance", "role": "financial documents", "license": "Apache-2.0"},
                        {"name": "seeded generator (this repository)", "role": "chat, email, code, config, logs, tables, memos", "license": "-"}],
            "summary": ("3,000 synthetic test documents with known answers. About half contain sensitive data; of the "
                        "rest, about a third are hard negatives that look sensitive but are not (masked card numbers, "
                        "documentation placeholders, public business data, first names alone). All values are fake."),
            "policy": (ROOT / "config/dspm/policy_v1.md").read_text() if (ROOT / "config/dspm/policy_v1.md").exists() else None}


_DSPM_CACHE: list[dict] | None = None


def _dspm_items() -> list[dict]:
    global _DSPM_CACHE
    if _DSPM_CACHE is None:
        import gzip

        p = ROOT / "data/dspm/v1/test.jsonl.gz"
        _DSPM_CACHE = [json.loads(line) for line in gzip.open(p, "rt")] if p.exists() else []
    return _DSPM_CACHE


def sample(use_case: str, *, prompt_set: str | None = None, model: str | None = None, workload: str | None = None,
           label: str | None = None, hard_case: str | None = None, n: int = 5, seed: int = 0) -> list[dict]:
    import random

    rnd = random.Random(seed)  # noqa: S311 - display sampling
    if use_case == "standard":
        # Both values come from the request. Only catalog model ids and known prompt-set names name
        # a file; anything else (a path, "..", an unknown name) is an empty sample, never a read.
        model = model or "m1-qwen2.5-3b"
        prompt_set = prompt_set or "short"
        if model not in {m.id for m in catalog_store.load_catalog().models} or prompt_set not in PROMPT_SETS:
            return []
        base = (ROOT / "prompts/generated").resolve()
        p = (base / model / f"{prompt_set}.jsonl").resolve()
        if not p.is_relative_to(base) or not p.exists():
            return []
        lines = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        out = []
        for line in rnd.sample(lines, min(n, len(lines))):
            text = (f"Premise: {line['pair'][0]}\n\nHypothesis: {line['pair'][1]}" if "pair" in line else line.get("prompt", ""))
            out.append({"id": line["id"], "tokens": line.get("prompt_tokens_expected"), "text": text[:3000]})
        return out
    items = _dspm_items()
    if workload == "W2":
        items = [{**x, "label": x.get("category_top") or "NONE"} for x in items if "W2" in (x.get("workloads") or [])]
        workload = None
    pool = [x for x in items if (not workload or x["workload"] == workload) and (not label or x["label"] == label)
            and (not hard_case or hard_case == "any" or (hard_case == "none" and not x.get("hard_case"))
                 or x.get("hard_case") == hard_case)]
    out = []
    for x in rnd.sample(pool, min(n, len(pool))):
        toks = x["tokens"]
        toks = toks.get("m1-qwen2.5-3b", next(iter(toks.values()))) if isinstance(toks, dict) else toks
        out.append({"id": x["id"], "label": x["label"], "categories": x.get("categories") or [], "hard_case": x.get("hard_case"),
                    "tokens": toks, "bytes": x["bytes"], "source": x["source"], "text": x["text"][:4000],
                    "values": [{"category": s.get("category"), "text": x["text"][s["start"]:s["end"]][:60]}
                               for s in (x.get("spans") or [])[:8] if "start" in s]})
    return out
