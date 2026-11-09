"""Build the standard-suite prompt sets (Requirement 7.6) and quality sets (Requirement 13).

Length sets: ~128, ~1,024, ~4,096 (and optional ~14,000) prompt tokens per model
tokenizer, from public-domain Project Gutenberg books. Quality sets: 50 fixed
databricks-dolly-15k instructions (CC BY-SA 3.0) and 50 SNLI test pairs (CC BY-SA 4.0).

Usage:
  HF_TOKEN=... python tools/build_prompts.py --out prompts/generated
  python tools/build_prompts.py --hf-token-secret amx-gpu-bench/hf-token
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.config import load_models  # noqa: E402
from bench.prompts import count_tokens, render, tokenizer_for  # noqa: E402

GUTENBERG = {  # id -> title; public domain in the US
    1342: "Pride and Prejudice", 84: "Frankenstein", 2701: "Moby Dick",
    1661: "The Adventures of Sherlock Holmes", 98: "A Tale of Two Cities",
}
DOLLY = ("databricks/databricks-dolly-15k", "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
         "databricks-dolly-15k.jsonl")
SNLI = ("stanfordnlp/snli", "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b",
        "plain_text/test-00000-of-00001.parquet")
SETS = {"short": 128, "medium": 1024, "long": 4096, "xlong": 14000, "cls512": 500}
CLASSIFIER_SETS = ("short", "cls512")
INSTRUCTION = "Summarize the following passage in detail, then list the main characters.\n\n"
TOLERANCE = 0.02


def fetch(url: str) -> bytes:
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-https URL {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "amx-gpu-bench/0.1"})  # noqa: S310 - https only
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - https only, checked above
        return r.read()


def gutenberg_corpus(cache: Path) -> tuple[list[str], dict]:
    cache.mkdir(parents=True, exist_ok=True)
    sentences: list[str] = []
    hashes = {}
    for gid in sorted(GUTENBERG):
        path = cache / f"pg{gid}.txt"
        if not path.exists():
            path.write_bytes(fetch(f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"))
        raw = path.read_bytes()
        hashes[str(gid)] = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8", errors="ignore")
        start = text.find("*** START OF")
        end = text.find("*** END OF")
        body = text[text.find("\n", start) + 1 if start >= 0 else 0: end if end > 0 else None]
        body = re.sub(r"\s+", " ", body)
        sentences += [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if len(s.strip()) > 20]
    return sentences, hashes


def hf_file(repo: str, revision: str, filename: str, cache: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, revision=revision, repo_type="dataset",
                                cache_dir=str(cache)))


def fit(model, tok, sentences: list[str], start: int, target: int) -> tuple[str, int, int]:
    """Grow a passage from `start` until the rendered prompt reaches the target tokens."""
    words: list[str] = []
    i = start
    lo_target = int(target * (1 - TOLERANCE))
    while True:
        words.append(sentences[i % len(sentences)])
        i += 1
        if len(words) % 4 and len(words) < 3:
            continue
        r = render(model, tok, INSTRUCTION + " ".join(words))
        if r["tokens"] >= lo_target:
            break
    # Trim words from the end of the last sentence until within tolerance.
    text = " ".join(words)
    r = render(model, tok, INSTRUCTION + text)
    while r["tokens"] > target * (1 + TOLERANCE):
        text = text.rsplit(" ", max(1, (r["tokens"] - target) // 2))[0]
        r = render(model, tok, INSTRUCTION + text)
    return text, r["tokens"], i


def build_length_sets(models, sentences, out: Path, n: int, seed: int, sets: list[str], token):
    manifest = {}
    rng = random.Random(seed)
    starts = [rng.randrange(len(sentences)) for _ in range(n)]
    for m in models:
        tok = tokenizer_for(m, token)
        model_sets = CLASSIFIER_SETS if m.type == "encoder-classifier" else sets
        for name in model_sets:
            target = SETS[name]
            limit = (m.max_input_tokens if m.type != "decoder" else (m.max_model_len or 0) - 512)
            if target > max(m.max_input_tokens, 0) or (m.type == "decoder" and target > limit):
                continue
            path = out / m.id / f"{name}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for k, st in enumerate(starts):
                passage, ntok, _ = fit(m, tok, sentences, st, target)
                r = render(m, tok, INSTRUCTION + passage)
                row = {"id": f"{name}-{k:04d}", "prompt_tokens_expected": r["tokens"]}
                row.update({k2: v for k2, v in r.items() if k2 in ("prompt", "pair")})
                rows.append(row)
            data = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
            path.write_text(data, encoding="utf-8")
            toks = [x["prompt_tokens_expected"] for x in rows]
            manifest[f"{m.id}/{name}"] = {"n": len(rows), "target": target,
                                          "min_tokens": min(toks), "max_tokens": max(toks),
                                          "sha256": hashlib.sha256(data.encode()).hexdigest()}
            print(f"{m.id}/{name}: {len(rows)} prompts, {min(toks)}-{max(toks)} tokens")
    return manifest


def build_quality_sets(models, out: Path, cache: Path, seed: int, token):
    import pyarrow.parquet as pq

    manifest = {}
    rng = random.Random(seed)
    dolly = [json.loads(line) for line in hf_file(*DOLLY, cache).read_text().splitlines()]
    by_cat: dict[str, list[dict]] = {}
    for i, d in enumerate(dolly):
        d["_idx"] = i
        if len(d["instruction"]) + len(d.get("context") or "") < 2500:
            by_cat.setdefault(d["category"], []).append(d)
    picked = []
    cats = sorted(by_cat)
    while len(picked) < 50:
        for c in cats:
            if len(picked) < 50:
                picked.append(rng.choice(by_cat[c]))
    snli = pq.read_table(hf_file(*SNLI, cache)).to_pylist()
    snli_by_label = {lab: [r for r in snli if r["label"] == lab] for lab in (0, 1, 2)}
    pairs = []
    for k in range(50):
        pairs.append(rng.choice(snli_by_label[k % 3]))
    for m in models:
        tok = tokenizer_for(m, token)
        rows = []
        if m.type in ("decoder", "encoder-decoder"):
            for d in picked:
                text = d["instruction"] + (f"\n\nContext:\n{d['context']}" if d.get("context") else "")
                r = render(m, tok, text)
                rows.append({"id": f"dolly-{d['_idx']}", "category": d["category"],
                             "prompt": r["prompt"], "prompt_tokens_expected": r["tokens"]})
            name = "quality-50"
        else:
            for k, p in enumerate(pairs):
                rows.append({"id": f"snli-{k:02d}", "pair": [p["premise"], p["hypothesis"]],
                             "gold": p["label"],
                             "prompt_tokens_expected": count_tokens(tok, p["premise"], p["hypothesis"])})
            name = "quality-50-nli"
        path = out / m.id / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
        path.write_text(data, encoding="utf-8")
        manifest[f"{m.id}/{name}"] = {"n": len(rows),
                                      "sha256": hashlib.sha256(data.encode()).hexdigest()}
        print(f"{m.id}/{name}: {len(rows)} items")
    return manifest


def token_from_secret(name: str) -> str:
    import boto3

    return boto3.client("secretsmanager", region_name="us-east-1").get_secret_value(SecretId=name)["SecretString"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "prompts" / "generated"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--sets", nargs="*", default=["short", "medium", "long"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf-token-secret", default=None)
    args = ap.parse_args(argv)
    import os

    token = os.environ.get("HF_TOKEN")
    if args.hf_token_secret:
        token = token_from_secret(args.hf_token_secret)
    catalog = load_models(ROOT / "config" / "models.yaml")
    wanted = args.models or [m.id for m in catalog.models
                             if m.enabled and "standard" in m.suites]
    models = [catalog.get(i) for i in wanted]
    out, cache = Path(args.out), Path(args.cache)
    sentences, hashes = gutenberg_corpus(cache / "gutenberg")
    manifest = {"seed": args.seed, "n": args.n, "gutenberg_sha256": hashes,
                "dolly": {"repo": DOLLY[0], "revision": DOLLY[1], "license": "CC BY-SA 3.0"},
                "snli": {"repo": SNLI[0], "revision": SNLI[1], "license": "CC BY-SA 4.0"},
                "sets": {}}
    manifest["sets"].update(build_length_sets(models, sentences, out, args.n, args.seed, args.sets, token))
    manifest["sets"].update(build_quality_sets(models, out, cache / "hf", args.seed, token))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out / 'manifest.json'}")
    _ = io  # keep import for type checkers
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
