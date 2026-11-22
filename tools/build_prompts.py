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
# The exact texts the shipped pools were cut from (prompts/generated/manifest.json, gutenberg_sha256).
# A model added later gets pools from byte-identical text or none at all.
GUTENBERG_SHA256 = {
    84: "2b97b7f8aad508649481d8dd4f6b54f9b965ddb3ba634e9151c815f537b49e42",
    98: "d54c2b80d40a40b982cd88852c6180bb944d95acdb028af3d0e01a1750681784",
    1342: "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75",
    1661: "922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0",
    2701: "b5e88b71a1cda068e9ab1d94344c9d932338ba6da1a66def97a1aecd9cfcc87d",
}
CORPUS_PREFIX = "datasets/gutenberg/"       # the same five files, kept in the bench bucket for the web app
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


def gutenberg_corpus(cache: Path, bucket: str | None = None, s3=None) -> tuple[list[str], dict]:
    """The five texts, from the local cache, else the bench bucket (datasets/gutenberg/, where
    `publish_corpus` put them), else gutenberg.org (which refuses some clients with HTTP 503).
    Every file must match its pinned SHA-256: the pools of a model added later are cut from the
    same bytes as the shipped ones, or the add fails with a clear reason."""
    cache.mkdir(parents=True, exist_ok=True)
    sentences: list[str] = []
    hashes = {}
    for gid in sorted(GUTENBERG):
        path = cache / f"pg{gid}.txt"
        if not path.exists():
            path.write_bytes(_fetch_text(gid, bucket, s3))
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != GUTENBERG_SHA256[gid]:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"pg{gid}.txt ({GUTENBERG[gid]}) does not match the pinned text (sha256 {digest[:12]} != "
                               f"{GUTENBERG_SHA256[gid][:12]}); the prompt pools would differ from the shipped ones")
        hashes[str(gid)] = digest
        text = raw.decode("utf-8", errors="ignore")
        start = text.find("*** START OF")
        end = text.find("*** END OF")
        body = text[text.find("\n", start) + 1 if start >= 0 else 0: end if end > 0 else None]
        body = re.sub(r"\s+", " ", body)
        sentences += [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if len(s.strip()) > 20]
    return sentences, hashes


def _fetch_text(gid: int, bucket: str | None, s3) -> bytes:
    key = f"{CORPUS_PREFIX}pg{gid}.txt"
    errors = []
    if bucket:
        try:
            import boto3

            c = s3 or boto3.client("s3", region_name="us-east-1")
            return c.get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception as e:  # noqa: BLE001 - absent or unreadable: try the origin next
            errors.append(f"s3://{bucket}/{key}: {type(e).__name__}")
    try:
        return fetch(f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt")
    except Exception as e:  # noqa: BLE001
        errors.append(f"gutenberg.org: {type(e).__name__}: {str(e)[:80]}")
    raise RuntimeError(f"the public-domain text pg{gid}.txt ({GUTENBERG[gid]}) could not be fetched ({'; '.join(errors)}). "
                       f"Publish it to the bucket once: tools/build_prompts.py --publish-corpus <bucket>")


def publish_corpus(cache: Path, bucket: str, s3=None) -> int:
    """Copy the five verified texts from the local cache to the bench bucket (one-off, operator)."""
    import boto3

    c = s3 or boto3.client("s3", region_name="us-east-1")
    gutenberg_corpus(cache)                     # fetch if needed and verify every hash first
    for gid in sorted(GUTENBERG):
        c.put_object(Bucket=bucket, Key=f"{CORPUS_PREFIX}pg{gid}.txt", Body=(cache / f"pg{gid}.txt").read_bytes(),
                     ContentType="text/plain; charset=utf-8")
    return len(GUTENBERG)


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


def build_embedding_sets(models, items: list[dict], dataset_manifest: dict, out: Path, token) -> dict:
    """Embedding pools: every corpus text (translation corpus, all three languages) of a size, with the
    model's document prefix, under prompts/generated/<model>/embed-<size>.jsonl. item_id is
    "<corpus id>:<lang>" so the report can score cross-lingual retrieval against the corpus."""
    from bench.embedding import LANGUAGES, SIZES, item_key

    manifest = {}
    for m in models:
        if m.type != "embedding":
            continue
        tok = tokenizer_for(m, token)
        prefix = m.embedding.prefix if m.embedding else ""
        max_len = m.embedding.max_length if m.embedding else m.max_input_tokens
        for size in SIZES:
            rng = random.Random(f"embed|{size}")
            rows = []
            for it in items:
                if it["size"] != size:
                    continue
                for lang in LANGUAGES:
                    text = it["text"][lang]
                    rows.append({"item_id": item_key(it["id"], lang), "lang": lang, "size": size, "input": prefix + text,
                                 "prompt_tokens_expected": min(count_tokens(tok, prefix + text), max_len), "chars": len(text)})
            rng.shuffle(rows)                                        # any prefix is a fair language mix
            for k, r in enumerate(rows):
                r["id"] = f"embed-{size}-{k:04d}"
            path = out / m.id / f"embed-{size}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            data = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
            path.write_text(data, encoding="utf-8")
            toks = [x["prompt_tokens_expected"] for x in rows]
            manifest[f"{m.id}/embed-{size}"] = {"n": len(rows), "size": size, "min_tokens": min(toks), "max_tokens": max(toks),
                                                "languages": list(LANGUAGES), "sha256": hashlib.sha256(data.encode()).hexdigest(),
                                                "dataset_sha256": dataset_manifest["dataset_sha256"]}
            print(f"{m.id}/embed-{size}: {len(rows)} texts, {min(toks)}-{max(toks)} tokens")
    return manifest


DECISION_ITEMS = {"short": 600, "long": 240}


def build_decision_sets(models, out: Path, token, seed: int = 0) -> dict:
    """Decision pools (Strands Decider): generated agent decisions in two state sizes (bench/decision.py), one
    pool per size under prompts/generated/<model>/decide-<size>.jsonl. A line carries the state, the three
    typed questions, the gold labels and the token count of state + all questions (the tokenizer in the
    adapter repository is the torso's). The report scores answers against the gold labels and the GPU cell."""
    from bench import decision

    manifest = {}
    for m in models:
        if m.type != "decision":
            continue
        tok = tokenizer_for(m, token)
        for size, n in DECISION_ITEMS.items():
            items = decision.generate(n, size, seed)
            rows = []
            for it in items:
                q_text = " ".join(f"{q['instructions']} {json.dumps(q.get('criteria', ''))}" for q in it["questions"].values())
                rows.append({"id": it["id"], "item_id": it["id"], "size": size, "state": it["state"], "questions": it["questions"],
                             "gold": it["gold"], "prompt_tokens_expected": count_tokens(tok, it["state"] + " " + q_text),
                             "chars": len(it["state"])})
            path = out / m.id / f"decide-{size}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            data = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
            path.write_text(data, encoding="utf-8")
            toks = [x["prompt_tokens_expected"] for x in rows]
            manifest[f"{m.id}/decide-{size}"] = {"n": len(rows), "size": size, "min_tokens": min(toks), "max_tokens": max(toks),
                                                 "questions": list(decision.QUESTION_TYPES), "sha256": hashlib.sha256(data.encode()).hexdigest(),
                                                 "dataset_sha256": decision.dataset_sha256(items), "seed": seed}
            print(f"{m.id}/decide-{size}: {len(rows)} decisions, {min(toks)}-{max(toks)} tokens")
    return manifest


INTEL_INPUT_TOKENS = (128, 2048)
INTEL_POOL_SIZE = 512


def random_prompt(tok, rng: random.Random, n_tokens: int, model) -> tuple[str, int]:
    """A prompt of about n_tokens tokens made of random vocabulary tokens, the way `vllm bench serve
    --dataset-name random` builds its requests: sample ids, decode to text, re-count as the server will."""
    from bench.prompts import render

    special = set(tok.all_special_ids or [])
    vocab = tok.vocab_size
    ids = [i for i in (rng.randrange(vocab) for _ in range(n_tokens * 2)) if i not in special][:n_tokens]
    text = tok.decode(ids, skip_special_tokens=True)
    for _ in range(8):
        got = render(model, tok, text)["tokens"] if model.prompt_format.kind == "plain" else count_tokens(tok, text)
        if abs(got - n_tokens) <= max(2, n_tokens // 50):
            break
        if got > n_tokens:
            text = tok.decode(tok(text, add_special_tokens=False)["input_ids"][: n_tokens - (got - n_tokens) // 2 - 1],
                              skip_special_tokens=True)
        else:
            extra = [i for i in (rng.randrange(vocab) for _ in range(2 * (n_tokens - got) + 4)) if i not in special]
            text = text + tok.decode(extra[: n_tokens - got], skip_special_tokens=True)
    return text, count_tokens(tok, text)


def build_intel_sets(models, out: Path, token, seed: int = 0) -> dict:
    """Intel's benchmark pools: INTEL_POOL_SIZE random prompts of 128 and 2,048 tokens per generation
    model, sent as plain completions (no chat template, as `vllm bench serve` does)."""
    manifest = {}
    for m in models:
        if m.type not in ("decoder", "encoder-decoder") or "intel" not in m.suites:
            continue
        tok = tokenizer_for(m, token)
        for n in INTEL_INPUT_TOKENS:
            if n > m.max_input_tokens:
                continue
            rng = random.Random(f"intel|{seed}|{n}")
            rows = []
            for k in range(INTEL_POOL_SIZE):
                text, got = random_prompt(tok, rng, n, m)
                rows.append({"id": f"intel-random-{n}-{k:04d}", "prompt": text, "prompt_tokens_expected": got})
            path = out / m.id / f"intel-random-{n}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            data = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
            path.write_text(data, encoding="utf-8")
            toks = [x["prompt_tokens_expected"] for x in rows]
            manifest[f"{m.id}/intel-random-{n}"] = {"n": len(rows), "target": n, "min_tokens": min(toks), "max_tokens": max(toks),
                                                    "sha256": hashlib.sha256(data.encode()).hexdigest()}
            print(f"{m.id}/intel-random-{n}: {len(rows)} prompts, {min(toks)}-{max(toks)} tokens")
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
    ap.add_argument("--publish-corpus", metavar="BUCKET", default=None,
                    help="copy the five Gutenberg texts to s3://BUCKET/datasets/gutenberg/ and exit")
    args = ap.parse_args(argv)
    if args.publish_corpus:
        n = publish_corpus(Path(args.cache) / "gutenberg", args.publish_corpus)
        print(f"published {n} texts to s3://{args.publish_corpus}/{CORPUS_PREFIX}")
        return 0
    import os

    token = os.environ.get("HF_TOKEN")
    if args.hf_token_secret:
        token = token_from_secret(args.hf_token_secret)
    catalog = load_models(ROOT / "config" / "models.yaml")
    wanted = args.models or [m.id for m in catalog.models if m.enabled]
    models = [catalog.get(i) for i in wanted]
    std_models = [m for m in models if "standard" in m.suites]
    out, cache = Path(args.out), Path(args.cache)
    sentences, hashes = gutenberg_corpus(cache / "gutenberg")
    manifest = {"seed": args.seed, "n": args.n, "gutenberg_sha256": hashes,
                "dolly": {"repo": DOLLY[0], "revision": DOLLY[1], "license": "CC BY-SA 3.0"},
                "snli": {"repo": SNLI[0], "revision": SNLI[1], "license": "CC BY-SA 4.0"},
                "sets": {}}
    if args.models and (out / "manifest.json").exists():
        # A subset build (one added model) keeps the other models' entries; a full build starts clean.
        prev = json.loads((out / "manifest.json").read_text())
        manifest["sets"] = {k: v for k, v in prev.get("sets", {}).items() if k.split("/")[0] not in wanted}
        manifest.update({k: v for k, v in prev.items() if k not in manifest})
    manifest["sets"].update(build_length_sets(std_models, sentences, out, args.n, args.seed, args.sets, token))
    manifest["sets"].update(build_quality_sets(std_models, out, cache / "hf", args.seed, token))
    manifest["sets"].update(build_intel_sets(models, out, token, args.seed))
    manifest["sets"].update(build_decision_sets(models, out, token, args.seed))
    translation_dir = ROOT / "data" / "translation" / "v1"
    if (translation_dir / "items.jsonl.gz").exists():
        from tools.translation import build_requests as tr

        items, tman = tr.read_items(translation_dir)
        tmodels = [m for m in models if "translation" in m.suites]
        manifest["sets"].update(tr.build_for_models(items, tman, out, tmodels, {m.id: tokenizer_for(m, token) for m in tmodels}))
        manifest["sets"].update(build_embedding_sets(models, items, tman, out, token))
        manifest["translation"] = {"dataset_version": tman["version"], "dataset_sha256": tman["dataset_sha256"], "seed": tr.SEED}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out / 'manifest.json'}")
    _ = io  # keep import for type checkers
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
