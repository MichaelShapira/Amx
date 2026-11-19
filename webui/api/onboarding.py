"""Adding a Hugging Face model to the benchmark (Setup → Models → Add).

A model is more than a name: the load generator needs prompt pools cut to token lengths with the
model's own tokenizer (short/medium/long, the 50-item quality set) and, for the DSPM suite,
the 3,000 documents rendered through its chat template. So "add" is a job:

  1. lookup    the repository on Hugging Face: exists, gated, license, latest commit (pinned)
  2. spec      the catalog entry from config.json: what kind of model, how big, context length,
               chat template, labels; which serving software is expected to run it
  3. pools     prompt pools with the model's tokenizer (the same code as tools/build_prompts.py
               and tools/dspm/build_requests.py), uploaded next to the shipped models' pools
  4. catalog   the entry appended to the live catalog in the bucket (bench/catalog_store.py)

What this cannot do, and the Setup page says so in plain words: prove that the serving software
runs the model. vLLM and OpenVINO each support a list of architectures; a new model's engines are
marked "expected to work" from its architecture family, and the first benchmark run is the real
test (the run page names the cause when a bring-up fails).

Jobs run in a thread of the web app process (one task, so one job at a time); status is polled
by the Setup page. The Hugging Face token is read from Secrets Manager for gated models and used
only as a download credential here; it is never returned or logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from bench import catalog_store
from bench.config import ModelCatalog, ModelSpec

log = logging.getLogger("onboarding")
ROOT = Path(__file__).resolve().parents[2]
REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]{0,95}/[A-Za-z0-9][\w.-]{0,95}$")
PROMPTS_DIR = ROOT / "prompts" / "generated"
DSPM_DIR = ROOT / "data" / "dspm" / "v1"
TRANSLATION_DIR = ROOT / "data" / "translation" / "v1"
CACHE_DIR = ROOT / "data" / "cache"

# What the architecture name says about the model, and which serving software usually runs that
# kind. "Expected", not proven: the first run is the test.
KIND_BY_SUFFIX = (("ForCausalLM", "decoder"), ("ForConditionalGeneration", "encoder-decoder"),
                  ("ForSequenceClassification", "encoder-classifier"), ("ForTokenClassification", "token-classifier"))
# Architecture families the CPU engines list as supported (vLLM CPU docs, optimum-intel model_configs).
VLLM_CPU_FAMILIES = ("llama", "qwen2", "qwen3", "mistral", "gemma", "gemma2", "gemma3", "phi3", "phi", "granite",
                     "deepseek", "olmo", "smollm", "falcon", "starcoder2", "gpt_oss", "internlm2", "exaone")
OPENVINO_FAMILIES = VLLM_CPU_FAMILIES + ("t5", "mt5", "bert", "roberta", "deberta", "deberta-v2", "xlm-roberta", "distilbert",
                                         "electra", "modernbert", "gpt2", "opt", "bloom", "chatglm", "baichuan", "mpt")
SIZE_WARN_B = 8.0           # above this a CPU benchmark gets slow; say so, do not refuse

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


class OnboardingError(Exception):
    """A reason the reader can act on."""


# ------------------------------------------------------------------------- 1. lookup

def lookup(repo: str, token: str | None) -> dict:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    repo = (repo or "").strip().removeprefix("https://huggingface.co/").strip("/")
    if not REPO_RE.match(repo):
        raise OnboardingError("Enter the repository id as it appears on Hugging Face, for example "
                              "mistralai/Mistral-7B-Instruct-v0.3.")
    api = HfApi(token=token)
    try:
        info = api.model_info(repo, files_metadata=True)
    except RepositoryNotFoundError as e:
        raise OnboardingError(f"Hugging Face has no model repository called {repo}. Check the spelling, or whether it "
                              "is private.") from e
    except GatedRepoError as e:
        raise OnboardingError(f"{repo} is gated and the stored Hugging Face token cannot access it. Accept the model's "
                              "license on Hugging Face with the account that owns the token, then try again.") from e
    gated = bool(info.gated)
    license_ = (info.card_data or {}).get("license") if info.card_data else None
    if not license_:
        tag = next((t for t in (info.tags or []) if t.startswith("license:")), None)
        license_ = tag.split(":", 1)[1] if tag else "unknown"
    if isinstance(license_, list):
        license_ = license_[0] if license_ else "unknown"
    params = (info.safetensors.total if info.safetensors else None)
    siblings = {s.rfilename for s in (info.siblings or [])}
    return {"repo": repo, "revision": info.sha, "gated": gated, "license": str(license_).lower(), "params": params,
            "config": info.config or {}, "files": siblings, "pipeline_tag": info.pipeline_tag}


def _download_json(repo: str, revision: str, filename: str, token: str | None) -> dict | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        p = hf_hub_download(repo, filename, revision=revision, token=token, cache_dir=str(CACHE_DIR / "hf"))
    except EntryNotFoundError:
        return None
    return json.loads(Path(p).read_text())


# ------------------------------------------------------------------------- 2. spec

def model_id_for(repo: str, existing: set[str]) -> str:
    base = "u-" + re.sub(r"[^a-z0-9.]+", "-", repo.split("/")[-1].lower()).strip("-")[:36]
    mid, n = base, 2
    while mid in existing:
        mid, n = f"{base}-{n}", n + 1
    return mid


def derive_spec(found: dict, config: dict, tokenizer_cfg: dict | None, existing: set[str], next_slot: str) -> tuple[ModelSpec, list[str]]:
    """The catalog entry and the plain-language notes the Setup page shows next to it."""
    archs = config.get("architectures") or found["config"].get("architectures") or []
    kind = next((k for a in archs for suffix, k in KIND_BY_SUFFIX if a.endswith(suffix)), None)
    if kind is None:
        raise OnboardingError(
            f"The benchmark cannot serve this kind of model ({', '.join(archs) or 'unknown architecture'}). It serves "
            "chat models (…ForCausalLM), instruction models of the encoder-decoder type (…ForConditionalGeneration), "
            "text classifiers (…ForSequenceClassification) and detectors that mark words (…ForTokenClassification).")
    if kind == "encoder-decoder" and not config.get("is_encoder_decoder", True):
        kind = "decoder"
    params = found["params"]
    if not params:
        raise OnboardingError("Hugging Face does not report this model's size (no safetensors metadata), so the benchmark "
                              "cannot size hardware for it. Models stored as .safetensors work.")
    ctx = int(config.get("max_position_embeddings") or config.get("n_positions") or config.get("seq_length")
              or config.get("max_sequence_length") or (512 if kind in ("encoder-classifier", "token-classifier") else 4096))
    family = (config.get("model_type") or "").lower()
    has_chat = bool((tokenizer_cfg or {}).get("chat_template")) or "chat_template.jinja" in found["files"]
    notes: list[str] = []
    size_b = round(params / 1e9, 2)
    if size_b > SIZE_WARN_B:
        notes.append(f"At {size_b:g} billion parameters this is large for a CPU benchmark: expect slow answers on the CPU "
                     "side and long model downloads. The comparison still runs.")

    expected = {"vllm": family in VLLM_CPU_FAMILIES, "ovms": family in OPENVINO_FAMILIES,
                "thin-openvino": family in OPENVINO_FAMILIES, "thin-torch": True}
    precisions: dict[str, dict] = {}
    if kind == "decoder":
        engines_bf16 = ["vllm", "thin-torch"]
        precisions["bf16"] = {"engines": engines_bf16, "weights": "bf16", "activations": "bf16"}
        precisions["bf16-ov"] = {"engines": ["ovms"], "export": {"weight_format": "fp16"}, "weights": "bf16", "activations": "bf16"}
        precisions["int8-ov"] = {"engines": ["ovms"], "export": {"weight_format": "int8"}, "weights": "int8",
                                 "activations": "bf16-or-int8-dynamic"}
        precisions["int4-ov"] = {"engines": ["ovms"], "export": {"weight_format": "int4", "group_size": 128, "ratio": 1.0, "sym": False},
                                 "weights": "int4-g128", "activations": "bf16-or-int8-dynamic"}
        max_input, max_len = min(4096, max(512, ctx - 512)), min(16384, ctx)
        prompt_format = {"kind": "chat_template"} if has_chat else {"kind": "plain"}
        suites = ["standard", "dspm", "translation", "intel"]
        if not has_chat:
            notes.append("This model has no chat template, so prompts are sent as plain text; a base (non-instruct) model "
                         "may ramble instead of answering.")
    elif kind == "encoder-decoder":
        precisions["bf16"] = {"engines": ["thin-torch"], "weights": "bf16", "activations": "bf16"}
        precisions["int8-ov"] = {"engines": ["thin-openvino"], "export": {"weight_format": "int8", "task": "text2text-generation-with-past"},
                                 "weights": "int8", "activations": "bf16-or-int8-dynamic"}
        max_input, max_len = min(1024, ctx), None
        prompt_format, suites = {"kind": "plain"}, ["standard", "dspm", "translation"]
    else:
        precisions["bf16"] = {"engines": ["thin-torch"] + (["thin-openvino"] if kind == "encoder-classifier" else []),
                              "weights": "bf16", "activations": "bf16"}
        if kind == "encoder-classifier":
            precisions["bf16"]["inference_precision"] = "bf16"
            precisions["int8-ov"] = {"engines": ["thin-openvino"], "export": {"weight_format": "int8", "task": "text-classification"},
                                     "weights": "int8", "activations": "bf16-or-int8-dynamic"}
        max_input, max_len = min(512 if kind == "encoder-classifier" else 2048, ctx), None
        labels = [str(v) for _, v in sorted(((int(k), v) for k, v in (config.get("id2label") or {}).items()))]
        nli = {x.lower() for x in labels} & {"entailment", "contradiction", "neutral"}
        prompt_format = {"kind": "pair"} if nli else {"kind": "plain"}
        suites = ["standard"]
        notes.append("Classifiers and detectors added here run the speed benchmark only: the sensitive-data (DSPM) "
                     "suite needs a hand-made mapping from the model's labels to the policy's categories.")
    spec: dict[str, Any] = {
        "id": model_id_for(found["repo"], existing), "slot": next_slot, "hf_repo": found["repo"],
        "revision": found["revision"], "type": kind, "license": found["license"], "gated": found["gated"],
        "params": int(params), "nominal_size_b": size_b, "suites": suites, "max_input_tokens": max_input,
        "prompt_format": prompt_format, "quality_reference": {"target": "t1-g6", "precision": "bf16"},
        "precisions": precisions, "notes": f"added from Hugging Face; architecture {archs[0] if archs else '?'} ({family or '?'})",
    }
    if max_len:
        spec["max_model_len"] = max_len
    if kind in ("encoder-classifier", "token-classifier"):
        labels = [str(v) for _, v in sorted(((int(k), v) for k, v in (config.get("id2label") or {}).items()))]
        spec["classifier"] = {"labels": labels, "max_length": max_input}
    not_expected = [e for e in {e for p in precisions.values() for e in p["engines"]} if not expected.get(e, True)]
    if not_expected:
        notes.append("Its architecture family is not on the support lists of " + ", ".join(sorted(not_expected)) +
                     ", so those options may fail to load it; the run page will say exactly why if they do.")
    notes.append("Nothing has run this model here yet. Which serving software really loads it is proven by the first "
                 "run, not by this page.")
    return ModelSpec.model_validate(spec), notes


# ------------------------------------------------------------------------- 3. pools

def build_pools(model: ModelSpec, token: str | None, bucket: str, progress) -> dict:
    """Prompt pools for one model, uploaded to the bucket; the shipped manifests are extended."""
    import sys

    from bench import pools
    from bench.prompts import tokenizer_for

    sys.path.insert(0, str(ROOT))
    from tools import build_prompts as bp
    from webui.api import aws

    s3 = aws.client("s3")
    progress("Downloading the tokenizer")
    tok = tokenizer_for(model, token)                           # the one cost of a gated model: the token
    progress("Loading the public-domain text corpus")
    sentences, _hashes = bp.gutenberg_corpus(CACHE_DIR / "gutenberg", bucket, s3)
    out = PROMPTS_DIR
    progress("Cutting prompts to the model's token lengths")
    manifest_sets = bp.build_length_sets([model], sentences, out, 100, 0, ["short", "medium", "long"], token)
    progress("Building the 50-item quality set")
    manifest_sets.update(bp.build_quality_sets([model], out, CACHE_DIR / "hf", 0, token))
    if "intel" in model.suites:
        progress("Building Intel's random-prompt benchmark pools")
        manifest_sets.update(bp.build_intel_sets([model], out, token))
    if "translation" in model.suites and (TRANSLATION_DIR / "items.jsonl.gz").exists():
        progress("Rendering the translation corpus through the model's prompt format")
        from tools.translation import build_requests as tr

        items, tman = tr.read_items(TRANSLATION_DIR)
        manifest_sets.update(tr.build_for_models(items, tman, out, [model], {model.id: tok}))
    # upload only this model's files; merge its sets into the shared manifest
    uploaded = 0
    for f in sorted((out / model.id).glob("*.jsonl")):
        s3.put_object(Bucket=bucket, Key=f"artifacts/prompts/{model.id}/{f.name}", Body=f.read_bytes())
        uploaded += 1
    man = _read_json(s3, bucket, "artifacts/prompts/manifest.json") or {"sets": {}, "pool_sizes": {}}
    man.setdefault("sets", {}).update(manifest_sets)
    std = {f"artifacts/prompts/{model.id}/{f.name}": f.read_bytes() for f in (out / model.id).glob("*.jsonl")}
    pm = pools.manifest(std)
    man.setdefault("pool_sizes", {}).update(pm["pool_sizes"])
    man.setdefault("sha256", {}).update(pm["sha256"])
    s3.put_object(Bucket=bucket, Key="artifacts/prompts/manifest.json", Body=json.dumps(man, indent=1).encode())
    result = {"prompt_files": uploaded, "dspm_files": 0}
    if "dspm" in model.suites and (DSPM_DIR / "test.jsonl.gz").exists():
        progress("Rendering the sensitive-data documents through the model's prompt format")
        from tools.dspm import build_requests as br

        req_out = DSPM_DIR / "requests"
        # merge: the container's copy of the manifest serves the plan for EVERY model; overwriting it with
        # one model's pools made every shipped DSPM pool "missing from the manifest" until a restart.
        local_manifest = br.build_for_models(DSPM_DIR, req_out, [model], {model.id: tok}, merge=True)
        for f in sorted((req_out / model.id).glob("*.jsonl")):
            s3.put_object(Bucket=bucket, Key=f"artifacts/dspm/v1/requests/{model.id}/{f.name}", Body=f.read_bytes())
            result["dspm_files"] += 1
        dman = _read_json(s3, bucket, "artifacts/dspm/v1/requests/manifest.json") or {"pool_sizes": {}, "pools": {}}
        dman.setdefault("pool_sizes", {}).update(local_manifest.get("pool_sizes", {}))
        dman.setdefault("pools", {}).update(local_manifest.get("pools", {}))
        s3.put_object(Bucket=bucket, Key="artifacts/dspm/v1/requests/manifest.json", Body=json.dumps(dman, indent=1, sort_keys=True).encode())
    return result


def _read_json(s3, bucket: str, key: str) -> dict | None:
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception:  # noqa: BLE001 - absent is fine
        return None


# ------------------------------------------------------------------------- the job

def start(repo: str, who: str | None) -> str:
    with _lock:
        running = [j for j in _jobs.values() if j["status"] == "running"]
        if running:
            raise OnboardingError(f"Another model ({running[0]['repo']}) is being added; wait for it to finish.")
        job_id = uuid.uuid4().hex[:12]
        _jobs[job_id] = {"id": job_id, "repo": repo, "status": "running", "stage": "Looking the model up on Hugging Face",
                         "started_at": time.time(), "by": who, "notes": [], "error": None, "model": None}
    threading.Thread(target=_run, args=(job_id, repo), daemon=True, name=f"onboard-{job_id}").start()
    return job_id


def status(job_id: str) -> dict | None:
    j = _jobs.get(job_id)
    if not j:
        return None
    out = {k: v for k, v in j.items() if k != "by"}
    out["elapsed_s"] = round(time.time() - j["started_at"], 1)
    return out


def _run(job_id: str, repo: str) -> None:
    from webui.api import settings

    j = _jobs[job_id]

    def progress(stage: str) -> None:
        j["stage"] = stage

    try:
        token = settings.hf_token_for_onboarding()
        found = lookup(repo, token)
        bucket = catalog_store.default_bucket()
        if not bucket:
            raise OnboardingError("this deployment has no bucket configured")
        catalog = catalog_store.load_catalog(fresh=True)
        if any(m.hf_repo.lower() == found["repo"].lower() for m in catalog.models):
            raise OnboardingError(f"{found['repo']} is already in the catalog.")
        progress("Reading the model's configuration")
        config = _download_json(found["repo"], found["revision"], "config.json", token) or found["config"]
        tok_cfg = _download_json(found["repo"], found["revision"], "tokenizer_config.json", token)
        existing = {m.id for m in catalog.models}
        n = 1 + sum(1 for m in catalog.models if m.user_added)
        spec, notes = derive_spec(found, config, tok_cfg, existing, f"U{n}")
        j["notes"] = notes
        j["model"] = {"id": spec.id, "name": found["repo"].split("/")[-1], "type": spec.type, "params_b": spec.nominal_size_b,
                      "license": spec.license, "gated": spec.gated}
        pools_result = build_pools(spec, token, bucket, progress)
        progress("Adding the model to the catalog")
        new_models = [*catalog.models, spec]
        new_cat = ModelCatalog(size_cap_b=max(catalog.size_cap_b, spec.nominal_size_b), models=new_models,
                               removed=list(catalog.removed))
        catalog_store.save_catalog(new_cat, bucket)
        j.update(status="done", stage="Done", pools=pools_result)
    except OnboardingError as e:
        j.update(status="failed", stage="Failed", error=str(e))
    except Exception as e:  # noqa: BLE001 - the reader gets the type and message, the log gets the trace
        log.error("onboarding %s failed: %s", repo, traceback.format_exc())
        j.update(status="failed", stage="Failed", error=f"{type(e).__name__}: {str(e)[:300]}")


def remove(model_id: str, who: str | None) -> dict:
    """Take a model out of the live catalog. Its prompt pools stay in the bucket (cheap, and a
    re-add reuses them); finished runs keep their data."""
    catalog = catalog_store.load_catalog(fresh=True)
    keep = [m for m in catalog.models if m.id != model_id]
    if len(keep) == len(catalog.models):
        raise OnboardingError("no such model")
    if len(keep) == 0:
        raise OnboardingError("The catalog cannot be empty; add another model before removing this one.")
    gone = next(m for m in catalog.models if m.id == model_id)
    # A removed shipped model is recorded by id: the live copy is merged with the seed on every read, and
    # without the record the next read would bring it back (bench.catalog_store.merge_with_seed).
    removed = sorted(set(catalog.removed) | ({model_id} if not gone.user_added else set()))
    catalog_store.save_catalog(ModelCatalog(size_cap_b=catalog.size_cap_b, models=keep, removed=removed),
                               catalog_store.default_bucket())
    return {"removed": model_id, "remaining": len(keep)}


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]
