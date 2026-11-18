"""Write the synthetic translation corpus (tools.translation.generator) as a frozen dataset.

    .venv/bin/python -m tools.translation.build_dataset --out data/translation/v1 --n 200

Writes items.jsonl.gz (one item per line: id, size, domain, text per language, character and
word counts), manifest.json (seed, counts, the dataset's SHA-256, the languages and directions)
and review_sample.md, a few items per domain in all three languages for a human check. The
request pools per model are rendered from this file by tools.translation.build_requests, and
`bench upload-artifacts` copies it to s3://<bucket>/datasets/translation/v1/ where the report
reads the references.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.translation import generator as g  # noqa: E402

VERSION = "v1"
DEFAULT_SEED = 20261119


def items_bytes(items: list[dict]) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:       # fixed mtime: the same items give the same bytes
        for it in items:
            gz.write((json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    return buf.getvalue()


def manifest(items: list[dict], raw: bytes, seed: int, n_per_size: int) -> dict:
    chars = {lang: sum(it["chars"][lang] for it in items) for lang in g.LANGUAGES}
    words = {lang: sum(it["words"][lang] for it in items) for lang in g.LANGUAGES}
    per_size = {}
    for size in g.SIZES:
        sel = [it for it in items if it["size"] == size]
        per_size[size] = {"items": len(sel), "sentences": [min(x["sentences"] for x in sel), max(x["sentences"] for x in sel)],
                          "words_en": [min(x["words"]["en"] for x in sel), max(x["words"]["en"] for x in sel)],
                          "chars_en": [min(x["chars"]["en"] for x in sel), max(x["chars"]["en"] for x in sel)]}
    return {"version": VERSION, "seed": seed, "n_per_size": n_per_size, "items": len(items),
            "languages": list(g.LANGUAGES), "directions": list(g.DIRECTIONS), "sizes": per_size,
            "domains": dict(sorted(Counter(it["domain"] for it in items).items())), "frames": g.frame_count(),
            "chars": chars, "words": words, "dataset_sha256": hashlib.sha256(raw).hexdigest(),
            "generator": "tools/translation/generator.py", "source": "synthetic (templated, seeded); no real data"}


def review_sample(items: list[dict], per_domain: int = 2) -> str:
    out = ["# Translation corpus: review sample", "",
           "Every item is the same template filled in English, German and Spanish, so each text is the",
           "reference translation of the other two. Check these for grammar and faithfulness.", ""]
    seen: Counter = Counter()
    for it in items:
        key = (it["size"], it["domain"])
        if seen[key] >= per_domain:
            continue
        seen[key] += 1
        out += [f"## {it['id']} ({it['size']}, {it['domain']})", ""]
        for lang in g.LANGUAGES:
            out += [f"- **{lang}** ({it['words'][lang]} words): {it['text'][lang]}"]
        out.append("")
    return "\n".join(out)


def build(out: Path, n_per_size: int, seed: int) -> dict:
    items = g.build_items(n_per_size, seed)
    raw = items_bytes(items)
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.jsonl.gz").write_bytes(raw)
    man = manifest(items, raw, seed, n_per_size)
    (out / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "review_sample.md").write_text(review_sample(items), encoding="utf-8")
    return man


def read_items(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "translation" / VERSION))
    ap.add_argument("--n", type=int, default=200, help="items per size (sentence, paragraph)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)
    man = build(Path(args.out), args.n, args.seed)
    print(f"{man['items']} items ({man['n_per_size']} per size), {man['frames']} frames, "
          f"sha256 {man['dataset_sha256'][:12]} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
