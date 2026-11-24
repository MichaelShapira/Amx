"""Everything the wizard needs to offer choices, with plain-language names and explanations.
The reader is a product manager: comfortable with "10 concurrent requests" and "batch versus
single request", not with engine flags. Technical identifiers are kept alongside so the
generated run config maps back to the YAML files."""

from __future__ import annotations

import json
from pathlib import Path

from bench import catalog_store, pairing, translation
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
    "translation": {
        "name": "Translation (EN ⇄ DE, EN ⇄ ES)",
        "short": "How fast each hardware option translates business text between English, German and Spanish, and what it costs per million characters.",
        "long": ("Translates short messages or paragraphs from a synthetic parallel corpus (the same text written in all "
                 "three languages, so the reference translation is known). Reports characters translated per second, cost "
                 "per million characters (the unit managed translation services bill by), time to the first word and to "
                 "the whole translation, and translation quality (chrF) against the reference and against the GPU's output. "
                 "Use it for document, support-ticket and product-content translation."),
        "icon": "translate",
    },
    "embedding": {
        "name": "Embeddings (search, RAG, deduplication)",
        "short": "How fast each hardware option turns texts into embedding vectors, and what it costs per million texts.",
        "long": ("Embeds short messages or paragraphs from the synthetic EN/DE/ES corpus into vectors, the step behind "
                 "semantic search, retrieval-augmented generation and near-duplicate detection. Reports embeddings per "
                 "second, cost per million embeddings, time per request, whether the CPU's vectors match the GPU's "
                 "(cosine similarity) and whether a German or Spanish text retrieves its English translation. Embedding "
                 "is all reading and no writing: the whole text is processed at once and no token is generated, which is "
                 "the work Intel's AMX units are built for."),
        "icon": "vector",
    },
    "decision": {
        "name": "Agent decisions (routing, guardrails, triage)",
        "short": "How fast each hardware option answers the typed decisions inside an agent, and what it costs per million decisions.",
        "long": ("A decision model (Strands Decider) does not write text: given a state (a message or a ticket thread) it picks "
                 "one of N options, answers yes/no, or rates on a scale, each with a calibrated confidence, in one pass. These are "
                 "the decisions inside an agent: which team handles a request, whether an action needs a response within the hour, "
                 "how severe a problem is. The suite sends generated customer requests with their gold labels; it reports decisions "
                 "per second, cost per million decisions, time per decision, accuracy against the labels, and whether the CPU's "
                 "decisions agree with the GPU's (same model and server on both sides). All reading, no writing. Note: the model's "
                 "server runs the torso in fp32 on CPU (the project's own setting), so the CPU side uses AVX-512, not AMX."),
    },
    "intel": {
        "name": "Intel's published benchmark (reproduced)",
        "short": "The exact test Intel's Software Catalog publishes Xeon numbers for, run on your GPU and CPU options, with Intel's figures beside yours.",
        "long": ("Random prompts of 128 tokens, answers of exactly 128 tokens, requests sent as fast as the machine takes them, "
                 "one concurrency at a time; a machine passes a load setting when each generated token takes at most "
                 "100 ms and the first token arrives within 3 s. This is the method behind Intel's catalog figures for "
                 "Qwen3 and Llama 3.1 (two-socket Xeon 6 servers), shown next to your results per core so a cloud slice "
                 "and a whole server can be compared."),
        "icon": "intel",
    },
}
INTEL_IO = {"128/128": ("128 in / 128 out", "Intel's headline shape: a short prompt, a short answer."),
            "2048/128": ("2,048 in / 128 out", "Long prompt, short answer: reading-heavy."),
            "128/2048": ("128 in / 2,048 out", "Short prompt, long answer: writing-heavy."),
            "2048/2048": ("2,048 in / 2,048 out", "Long prompt, long answer.")}
# Translation use case: directions and text sizes (bench.plan.TRANSLATION_DIRECTIONS / _SIZES).
DIRECTIONS = {
    "en-de": ("English → German", "Source text in English, translated into German."),
    "de-en": ("German → English", "Source text in German, translated into English."),
    "en-es": ("English → Spanish", "Source text in English, translated into Spanish."),
    "es-en": ("Spanish → English", "Source text in Spanish, translated into English."),
}
TEXT_SIZES = {
    "sentence": ("Short messages", "One or two sentences (about 8-30 words): chat replies, subject lines, UI strings. "
                                   "Dominated by the time to the first word.", 192),
    "paragraph": ("Paragraphs", "Five to seven sentences (about 40-90 words): a support e-mail, a shipping notice, "
                                "release notes. Dominated by generation speed.", 448),
}

MODELS = {
    "m1-qwen3-1.7b": ("Qwen3 1.7B", "Alibaba's compact chat model, 1.7 billion parameters, Apache-2.0. The model Intel's "
                                    "catalog describes as built for extreme throughput on server CPUs and publishes Xeon "
                                    "benchmarks for. Thinking mode is off, as in Intel's recipe."),
    "m3-roberta-large-mnli": ("RoBERTa-large (classifier)", "Text classifier, 0.36 billion parameters. Does not generate "
                                                            "text; answers 'does this text match this statement'. Very "
                                                            "fast, lower accuracy."),
    "m5-flan-t5-xl": ("Flan-T5 XL", "Instruction-following model of the older encoder-decoder type, 2.85 billion "
                                    "parameters. Many production classifiers still use this family."),
    "m6-gemma-4-e4b": ("Gemma 4 E4B Instruct", "Google's newest small chat model: 8.0 billion stored parameters (Google "
                                                 "counts 4.5 billion effective), understands text, images and audio; text "
                                                 "only here. Apache-2.0. Served with the vLLM settings from Intel's Xeon recipe."),
    "m7-gemma-4-e2b": ("Gemma 4 E2B Instruct", "The compact Gemma 4: 5.1 billion stored parameters (2 billion effective), "
                                                 "text, image and audio upstream, text only here. Apache-2.0."),
    "d1-strands-decider-e4b": ("Strands Decider E4B", "AWS Strands Labs' decision model on Gemma 4 E4B: a LoRA adapter and a pointer "
                                                      "readout replace the text-writing head, so it picks between options and rates on "
                                                      "scales with a calibrated confidence (81% on JevBench). Served by the official "
                                                      "strands-decider server on both sides; fp32 on CPU. In the DSPM/DLP suite it answers "
                                                      "the policy question as one typed decision. Fits one L4 (g6.xlarge) and an m8i.4xlarge."),
    "d2-strands-decider-12b": ("Strands Decider 12B", "The same decision model on Gemma 4 12B (87% on JevBench): 24 GB of bf16 weights, "
                                                      "so the GPU side needs an L40S (pick the g6e.xlarge size) and the CPU side, which "
                                                      "holds the torso in fp32, an m8i.8xlarge (128 GiB): a fair pair at about $1.9 and "
                                                      "$1.7 per hour. Agent decisions and DSPM/DLP."),
    "e1-embeddinggemma-300m": ("EmbeddingGemma 300M", "Google's embedding model (Gemma-3-based), 0.3 billion parameters, "
                                                       "768-dimensional vectors, 100+ languages. The embedding model Intel's "
                                                       "catalog lists for Xeon. Embeddings use case only."),
}
MODEL_KIND = {"decoder": "Chat / generation", "encoder-decoder": "Instruction model (encoder-decoder)",
              "encoder-classifier": "Classifier", "token-classifier": "Detector (marks words)", "embedding": "Embedding model",
              "decision": "Decision model"}

ENGINES = {"vllm": ("vLLM", "The most common open-source serving software for chat models, on GPUs and Intel CPUs."),
           "ovms": ("OpenVINO", "Intel's optimized CPU serving stack (OpenVINO Model Server)."),
           "thin-openvino": ("OpenVINO", "OpenVINO through a small serving wrapper, for models the model server does not host."),
           "decider": ("Strands Decider server", "The official strands-decider server (PyTorch): the only engine that can run a decision "
                                                   "model, whose pointer readout replaces the text-writing head vLLM expects. The same code "
                                                   "serves the GPU (bf16) and the CPU (fp32, the project's setting)."),
           "thin-torch": ("PyTorch", "Plain PyTorch through a small serving wrapper, used on the GPU side only for Flan-T5, which vLLM does not serve. "
                                     "The CPU PyTorch option was removed on 2026-11-20: 3-10x below vLLM on the same instance, it compared engines, not CPU with GPU.")}

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
    "on-demand": ("On-Demand", "Not subject to Spot interruption; full price. Use when Spot keeps getting interrupted or refused."),
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
        # An option is named for its headline engine. vLLM's helper engines (the PyTorch thin server for the one
        # model vLLM cannot serve, the Strands Decider server for decision models) are model-specific, not a choice
        # the reader makes, so they stay out of the label and the sentences; the Technical page lists them.
        helpers = {"thin-torch", "decider"}
        named = [e for e in t.engines if not (e in helpers and "vllm" in t.engines)]
        engines = list(dict.fromkeys(ENGINES[e][0] for e in named))
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
                     "batch_sizes_default": (m.classifier.batch_sizes if m.classifier else m.embedding.batch_sizes if m.embedding else [1]),
                     "hf_repo": m.hf_repo, "user_added": m.user_added})
    insts = {t: {**s, "type": t} for t, s in cat["instances"].items() if s.get("usd_per_hour")}
    return {
        "use_cases": USE_CASES, "models": mods, "hardware": hardware_options(),
        "instances": insts, "price_snapshot": cat["price_snapshot"],
        "prompt_sets": {k: {"label": v[0], "help": v[1]} for k, v in PROMPT_SETS.items()},
        "workloads": {k: {"label": v[0], "help": v[1]} for k, v in WORKLOADS.items()},
        "directions": {k: {"label": v[0], "help": v[1]} for k, v in DIRECTIONS.items()},
        "intel_io": {k: {"label": v[0], "help": v[1]} for k, v in INTEL_IO.items()},
        "text_sizes": {k: {"label": v[0], "help": v[1], "answer_tokens": v[2]} for k, v in TEXT_SIZES.items()},
        "presets": {k: {"label": v[0], "help": v[1], "p99_e2e_ms": v[2]} for k, v in PRESETS.items()},
        "depths": {k: {"label": v[0], "help": v[1], "requests": v[2], "passes": v[3]} for k, v in DEPTHS.items()},
        "request_modes": {k: {"label": v[0], "help": v[1]} for k, v in REQUEST_MODES.items()},
        "purchase": {k: {"label": v[0], "help": v[1]} for k, v in PURCHASE.items()},
        "concurrency_hints": CONCURRENCY_HINTS,
        "cpu_flag_notes": CPU_FLAG_NOTES,
        "precisions": {k: {"label": v[0], "help": v[1]} for k, v in PRECISIONS.items()},
        "engines": {k: {"label": v[0], "help": v[1]} for k, v in ENGINES.items()},
        "fair_ratio": list(pairing.FAIR_RATIO),
        "region": "us-east-1",
    }


def _display(m) -> tuple[str, str]:
    """Name and one-line description: the shipped text for shipped models; for a model added in
    Setup, the repository's own name and the onboarding note."""
    if m.id in MODELS:
        return MODELS[m.id]
    return m.hf_repo.split("/")[-1], m.notes or "added from Hugging Face in Setup"


def hardware_label(hw_id: str) -> str:
    """A hardware option's software names ("vLLM", "OpenVINO + PyTorch"), for sentences."""
    for h in hardware_options():
        if h["id"] == hw_id:
            return " + ".join(h["software"])
    return hw_id


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
    if use_case == "translation":
        return _translation_info()
    if use_case == "embedding":
        info = _translation_info()
        if not info.get("available"):
            return info
        return {**info, "kind": "embedding", "prefix": _embedding_prefix(),
                "summary": (f"The same {info['items']} synthetic items as the translation use case, every text in English, German "
                            "and Spanish, embedded one by one. Each text is prefixed with the model's document prompt. Besides "
                            "speed, the report checks that the CPU's vectors point the same way as the GPU's and that a German or "
                            "Spanish text's nearest English neighbour is its own translation.")}
    if use_case == "decision":
        return _decision_info()
    if use_case == "intel":
        return _intel_info()
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


def _translation_info() -> dict:
    man_p = ROOT / "data/translation/v1/manifest.json"
    if not man_p.exists():
        return {"available": False, "reason": "the translation corpus is not built locally (tools/translation/build_dataset.py)"}
    man = json.loads(man_p.read_text())
    sizes = {}
    for k, (label, help_, tokens) in TEXT_SIZES.items():
        st = (man.get("sizes") or {}).get(k) or {}
        sizes[k] = {"label": label, "help": help_, "answer_tokens": tokens, "items": st.get("items"),
                    "sentences": st.get("sentences"), "words_en": st.get("words_en"), "chars_en": st.get("chars_en")}
    return {"available": True, "kind": "parallel", "version": man.get("version"),
            "fingerprint": str(man.get("dataset_sha256"))[:12], "items": man.get("items"), "sizes": sizes,
            "directions": {k: {"label": v[0], "help": v[1]} for k, v in DIRECTIONS.items()},
        "intel_io": {k: {"label": v[0], "help": v[1]} for k, v in INTEL_IO.items()},
            "domains": man.get("domains") or {}, "chars": man.get("chars") or {}, "words": man.get("words") or {},
            "sources": [{"name": "seeded template generator (this repository, tools/translation)",
                         "role": "every item written in English, German and Spanish from one template", "license": "-"}],
            "summary": (f"{man.get('items')} synthetic items, each written in English, German and Spanish from the same template "
                        "with the same names, dates, amounts and places, so each text is the reference translation of the other "
                        "two. Six business domains: customer support, logistics, HR notices, travel confirmations, product "
                        "release notes, invoices. Nothing is real; no third-party text is used. The model is asked to translate "
                        "the source text and reply with the translation only."),
            "instruction": translation.INSTRUCTION}


def _embedding_prefix() -> str:
    for m in catalog_store.load_catalog().models:
        if m.type == "embedding" and m.embedding:
            return m.embedding.prefix
    return ""


def _decision_info() -> dict:
    """The decision suite's data: generated agent decisions (bench/decision.py), two state sizes, gold labels."""
    from bench import decision
    sizes = {}
    for size in decision.SIZES:
        items = decision.generate(20, size)
        words = [len(x["state"].split()) for x in items]
        sizes[size] = {"label": {"short": "Short messages", "long": "Ticket threads"}[size],
                       "help": {"short": "One customer message of 25-50 words: the state an agent routes on first contact.",
                                "long": "A ticket thread of 200-250 words with the deciding facts in the middle, as an agent sees a case later."}[size],
                       "words": [min(words), max(words)]}
    return {"available": True, "kind": "decision", "sizes": sizes, "teams": list(decision.TEAMS), "severity": list(decision.SEVERITY),
            "questions": {"route": "choice: which of five teams should handle this (billing, security, shipping, technical, sales)",
                          "urgent": "noul (yes/no): does the writer need a response within the hour",
                          "severity": "score: how severe is the problem, on minor / moderate / major / critical"},
            "summary": ("Generated customer requests, balanced over five teams and twenty scenarios, each with the three typed "
                        "questions a Strands Decider answers and the gold answers. A request carries one state and 1 or 3 of the "
                        "questions (the state is read once; each question adds its own tokens). The report scores accuracy against "
                        "the gold answers and the CPU's agreement with the GPU's decisions.")}


def _intel_info() -> dict:
    man_p = ROOT / "prompts/generated/manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    sets = {k: v for k, v in (man.get("sets") or {}).items() if "/intel-random-" in k}
    ref_p = ROOT / "config/intel_catalog_benchmarks.json"
    ref = json.loads(ref_p.read_text()) if ref_p.exists() else {"rows": []}
    published: dict[str, list] = {}
    for r in ref["rows"]:
        published.setdefault(r["model_id"], []).append({k: r.get(k) for k in ("cpu_model", "sockets", "total_cores", "memory_gb", "ddr5_mt_s",
                                                                                "io", "max_concurrency", "throughput_tok_s", "tpot_ms", "ttft_ms", "tested")})
    return {"available": bool(sets), "kind": "intel", "sets": sets, "io": {k: {"label": v[0], "help": v[1]} for k, v in INTEL_IO.items()},
            "published": published, "source": ref.get("source"), "note": ref.get("note"),
            "reason": None if sets else "the random-prompt pools are not built locally (tools/build_prompts.py)",
            "summary": ("Random vocabulary tokens assembled into prompts of a fixed length and sent as plain completions, exactly as "
                        "`vllm bench serve --dataset-name random` does; every answer is cut at the same length (ignore_eos). Nothing "
                        "about the text matters: the test measures how many tokens a machine reads and writes per second under a "
                        "concurrency, and whether it stays within 100 ms per token and 3 s to the first token. Intel publishes the "
                        "same test for Qwen3 and Llama 3.1 on two-socket Xeon 6 servers; those figures are shown with yours."),
            "sources": [{"name": "Intel Software Catalog", "role": "method and published figures", "license": "-"}]}


_TRANSLATION_CACHE: list[dict] | None = None


def _translation_items() -> list[dict]:
    global _TRANSLATION_CACHE
    if _TRANSLATION_CACHE is None:
        import gzip

        p = ROOT / "data/translation/v1/items.jsonl.gz"
        _TRANSLATION_CACHE = [json.loads(line) for line in gzip.open(p, "rt", encoding="utf-8")] if p.exists() else []
    return _TRANSLATION_CACHE


_DSPM_CACHE: list[dict] | None = None


def _dspm_items() -> list[dict]:
    global _DSPM_CACHE
    if _DSPM_CACHE is None:
        import gzip

        p = ROOT / "data/dspm/v1/test.jsonl.gz"
        _DSPM_CACHE = [json.loads(line) for line in gzip.open(p, "rt")] if p.exists() else []
    return _DSPM_CACHE


def sample(use_case: str, *, prompt_set: str | None = None, model: str | None = None, workload: str | None = None,
           label: str | None = None, hard_case: str | None = None, n: int = 5, seed: int = 0,
           direction: str | None = None) -> list[dict]:
    import random

    rnd = random.Random(seed)  # noqa: S311 - display sampling
    if use_case == "embedding":
        size = prompt_set if prompt_set in TEXT_SIZES else "sentence"
        lang = direction if direction in ("en", "de", "es") else None
        pool = [x for x in _translation_items() if x["size"] == size]
        prefix = _embedding_prefix()
        out = []
        for x in rnd.sample(pool, min(n, len(pool))):
            for lg in ([lang] if lang else ("en", "de", "es")):
                out.append({"id": f"{x['id']}:{lg}", "domain": x["domain"], "lang": lg, "size": size, "text": prefix + x["text"][lg],
                            "chars": x["chars"][lg], "words": x["words"][lg]})
        return out
    if use_case == "decision":
        from bench import decision
        size = prompt_set if prompt_set in decision.SIZES else "short"
        items = decision.generate(300 if size == "short" else 120, size)
        return [{"id": x["id"], "size": size, "text": x["state"], "gold": x["gold"], "words": len(x["state"].split()),
                 "questions": {k: v["instructions"] for k, v in x["questions"].items()}} for x in rnd.sample(items, min(n, len(items)))]
    if use_case == "intel":
        base = (ROOT / "prompts/generated").resolve()
        ids = {m.id for m in catalog_store.load_catalog().models}
        mid = model if model in ids else next((m.id for m in catalog_store.load_catalog().models if "intel" in m.suites), None)
        n_tok = prompt_set if prompt_set in ("128", "2048") else "128"
        p = (base / (mid or "") / f"intel-random-{n_tok}.jsonl").resolve() if mid else None
        if not p or not p.is_relative_to(base) or not p.exists():
            return []
        lines = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        return [{"id": x["id"], "tokens": x.get("prompt_tokens_expected"), "text": x["prompt"][:1500]} for x in rnd.sample(lines, min(n, len(lines)))]
    if use_case == "translation":
        direction = direction if direction in DIRECTIONS else "en-de"
        size = prompt_set if prompt_set in TEXT_SIZES else "sentence"
        src, tgt = translation.split_direction(direction)
        pool = [x for x in _translation_items() if x["size"] == size]
        return [{"id": x["id"], "domain": x["domain"], "direction": direction, "size": size,
                 "text": translation.prompt_text(direction, x["text"][src]), "source": x["text"][src], "reference": x["text"][tgt],
                 "chars": x["chars"][src], "words": x["words"][src], "sentences": x["sentences"]}
                for x in rnd.sample(pool, min(n, len(pool)))]
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
