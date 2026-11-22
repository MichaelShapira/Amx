"""Render DSPM/DLP request pools per model from the frozen dataset (design 24.4, task 21.1).

    .venv/bin/python -m tools.dspm.build_requests --dataset data/dspm/v1 \
        --hf-token-secret amx-gpu-bench/hf-token

Writes data/dspm/v1/requests/<model>/<workload>[-<bucket>].jsonl (test items) and
dev-<workload>... (dev items, used only to pick M3's threshold), plus manifest.json with
the S3 key and line count of every pool and the policy hashes. Each line holds the exact
prompt or classifier input, the expected prompt token count and the content bytes. Pools
are shuffled with a fixed seed so that any prefix is a fair sample (smoke runs use one).
Fails if an input exceeds a model's window (512 tokens for Flan-T5, RoBERTa-MNLI, Roblox).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import dspm  # noqa: E402
from bench.config import load_models  # noqa: E402
from bench.plan import SHORT_ONLY_SLOTS, WORKLOAD_BUCKETS, WORKLOAD_SLOTS, dspm_pool  # noqa: E402
from bench.prompts import count_tokens, render, roblox_format  # noqa: E402


def read_jsonl_gz(path: Path) -> list[dict]:
    """Same as tools.dspm.build_dataset.read_jsonl_gz, kept here so the web app's onboarding job
    (which renders requests for an added model) does not import the dataset generator and its
    dependencies (faker)."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


WINDOW = {"M5": 512, "M3": 512, "D2": 512, "DEC": 4096}     # DEC: the decider's validated max_length


def _order(item_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()


def items_for(items: list[dict], workload: str, bucket: str | None) -> list[dict]:
    if workload == "W2":
        return [x for x in items if "W2" in x["workloads"]]
    return [x for x in items if x["workload"] == workload and (bucket is None or x["bucket"] == bucket)]


def render_line(model, tok, policy: dspm.Policy, item: dict, workload: str) -> dict:
    text, slot = item["text"], model.dspm_profile
    line = {"id": item["id"], "item_id": item["id"], "bytes": item["bytes"]}
    if slot in ("M1", "M2", "M4"):
        r = render(model, tok, dspm.decoder_user_message(text, workload), system_text=policy.text)
        line.update(prompt=r["prompt"], prompt_tokens_expected=r["tokens"])
    elif slot == "M5":
        r = render(model, tok, dspm.t5_prompt(text, workload, policy))
        line.update(prompt=r["prompt"], prompt_tokens_expected=r["tokens"])
    elif slot == "M3":
        inputs = dspm.mnli_inputs(text)
        line.update(input=inputs, prompt_tokens_expected=sum(count_tokens(tok, s) for s in inputs),
                    max_input_tokens=max(count_tokens(tok, s) for s in inputs))
    elif slot == "D1":
        line.update(input=text, prompt_tokens_expected=count_tokens(tok, text))
    elif slot == "D2":
        s = roblox_format(text)
        line.update(input=s, prompt_tokens_expected=count_tokens(tok, s))
    elif slot == "DEC":
        # The decider's own request shape (state + typed questions, bench/decision.request_body); the token count
        # covers the state and the question text the server appends.
        state, qs = dspm.decision_state(text, workload, policy), dspm.decision_questions(workload)
        q_text = " ".join(f"{q['instructions']} {' '.join(q.get('criteria') or [])}" for q in qs.values())
        line.update(state=state, questions=qs, prompt_tokens_expected=count_tokens(tok, state + " " + q_text))
    else:
        raise ValueError(f"no DSPM request shape for slot {slot}")
    return line


def build(dataset: Path, out: Path, tokenizers: dict, seed: int = 20261109) -> dict:
    models = [m for m in load_models(ROOT / "config/models.yaml").models if m.enabled and "dspm" in m.suites]
    return build_for_models(dataset, out, models, tokenizers, seed)


def build_for_models(dataset: Path, out: Path, models: list, tokenizers: dict, seed: int = 20261109, *,
                     merge: bool = False) -> dict:
    """The request pools for the given models only (the Setup page adds one model at a time).
    merge=True keeps the other models' entries in out/manifest.json (the web app's copy of the
    shared manifest is what the plan reads for every model); the CLI rebuilds it from scratch."""
    policy = dspm.load_policy()
    splits = {"test": read_jsonl_gz(dataset / "test.jsonl.gz"), "dev": read_jsonl_gz(dataset / "dev.jsonl.gz")}
    sizes: dict[str, int] = {}
    stats: dict[str, dict] = {}
    for m in models:
        tok = tokenizers[m.id]
        for wl, slots in WORKLOAD_SLOTS.items():
            if m.dspm_profile not in slots or wl == "W5":
                continue
            if m.type == "token-classifier" and wl not in ("W1", "W3"):
                continue
            for bkt in WORKLOAD_BUCKETS[wl]:
                if bkt == "long" and (m.dspm_profile in SHORT_ONLY_SLOTS or m.max_input_tokens < 2560):
                    continue
                for split in ("test", "dev"):
                    if split == "dev" and m.dspm_profile != "M3":
                        continue
                    sel = sorted(items_for(splits[split], wl, bkt), key=lambda x: _order(x["id"], seed))
                    lines = [render_line(m, tok, policy, x, wl) for x in sel]
                    window = WINDOW.get(m.dspm_profile)
                    if window:
                        worst = max((x.get("max_input_tokens", x["prompt_tokens_expected"]) for x in lines), default=0)
                        if worst > window:
                            raise SystemExit(f"{m.id} {wl}/{bkt}/{split}: input of {worst} tokens exceeds {window}")
                    key = dspm_pool(m, wl if split == "test" else f"dev-{wl}", bkt)
                    path = out / Path(key).relative_to("artifacts/dspm/v1/requests")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in lines))
                    sizes[key] = len(lines)
                    toks = sorted(x["prompt_tokens_expected"] for x in lines)
                    stats[key] = {"n": len(lines), "tokens_min": toks[0] if toks else None,
                                  "tokens_p50": toks[len(toks) // 2] if toks else None,
                                  "tokens_max": toks[-1] if toks else None,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"pool_sizes": sizes, "pools": stats, "policy": policy.hashes(), "seed": seed,
                "dataset_manifest_sha256": hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest()}
    return write_manifest(out / "manifest.json", manifest, merge=merge)


def write_manifest(path: Path, manifest: dict, *, merge: bool) -> dict:
    """Write the requests manifest; with merge, the pools of models not rebuilt this time stay."""
    if merge and path.exists():
        prev = json.loads(path.read_text())
        manifest = {**prev, **manifest, "pool_sizes": {**prev.get("pool_sizes", {}), **manifest["pool_sizes"]},
                    "pools": {**prev.get("pools", {}), **manifest["pools"]}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv=None) -> int:
    from tools.dspm.build_dataset import load_tokenizers

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/dspm/v1")
    ap.add_argument("--hf-token-secret", default=None)
    ap.add_argument("--models", default=None,
                    help="comma-separated model ids: build these pools only and merge them into the existing manifest "
                         "(a new model added to the catalog); default: every enabled DSPM model, manifest from scratch")
    a = ap.parse_args(argv)
    token = None
    if a.hf_token_secret:
        import boto3

        token = boto3.client("secretsmanager", region_name="us-east-1").get_secret_value(
            SecretId=a.hf_token_secret)["SecretString"]
    ds = ROOT / a.dataset
    if a.models:
        from bench.config import load_models
        from bench.prompts import tokenizer_for
        cat = load_models(ROOT / "config/models.yaml")
        models = [cat.get(mid) for mid in a.models.split(",")]
        man = build_for_models(ds, ds / "requests", models, {m.id: tokenizer_for(m, token) for m in models}, merge=True)
    else:
        man = build(ds, ds / "requests", load_tokenizers(token))
    print(json.dumps({"pools": len(man["pool_sizes"]), "lines": sum(man["pool_sizes"].values())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
