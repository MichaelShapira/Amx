"""Render the translation request pools per model from the frozen corpus.

    .venv/bin/python -m tools.translation.build_requests [--models m1-qwen3-1.7b ...]

Writes prompts/generated/<model>/translate-<direction>-<size>.jsonl: one line per corpus item of
that size, the source text wrapped in the translation instruction (bench.translation.INSTRUCTION)
and the model's own chat format, with the expected prompt token count, the source character and
word counts (chars/s and $ per 1M characters come from these) and the item id the report uses to
look up the reference. Lines are shuffled with a fixed seed so any prefix is a fair sample. The
pools live next to the standard prompt sets and are uploaded by `bench upload-artifacts` with them;
tools/build_prompts.py calls build_for_models for every model, the Setup page for one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import translation  # noqa: E402
from bench.config import load_models  # noqa: E402
from bench.plan import TRANSLATION_DIRECTIONS, TRANSLATION_SIZES, translation_pool  # noqa: E402
from bench.prompts import count_tokens, render, tokenizer_for  # noqa: E402

DATASET = ROOT / "data" / "translation" / "v1"
SEED = 20261119


def read_items(dataset: Path = DATASET) -> tuple[list[dict], dict]:
    from tools.translation.build_dataset import read_items as _read

    items = _read(dataset / "items.jsonl.gz")
    man = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    return items, man


def _order(item_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{item_id}".encode()).hexdigest()


def render_line(model, tok, item: dict, direction: str, k: int) -> dict:
    src, tgt = translation.split_direction(direction)
    text = item["text"][src]
    r = render(model, tok, translation.prompt_text(direction, text))
    return {"id": f"{direction}-{item['size']}-{k:04d}", "item_id": item["id"], "direction": direction,
            "size": item["size"], "domain": item["domain"], "prompt": r["prompt"], "prompt_tokens_expected": r["tokens"],
            "chars": item["chars"][src], "words": item["words"][src],
            "reference_tokens_expected": count_tokens(tok, item["text"][tgt])}


def build_for_models(items: list[dict], dataset_manifest: dict, out: Path, models: list, tokenizers: dict,
                     seed: int = SEED) -> dict:
    """Pools for the given models; returns the prompt-manifest `sets` entries (key <model>/<pool name>)."""
    sets: dict[str, dict] = {}
    for m in models:
        tok = tokenizers[m.id]
        for size in TRANSLATION_SIZES:
            sel = sorted((it for it in items if it["size"] == size), key=lambda x: _order(x["id"], seed))
            for d in TRANSLATION_DIRECTIONS:
                lines = [render_line(m, tok, it, d, k) for k, it in enumerate(sel)]
                worst = max(x["prompt_tokens_expected"] for x in lines)
                if worst > m.max_input_tokens:
                    raise SystemExit(f"{m.id} {d}/{size}: a prompt of {worst} tokens exceeds the model's {m.max_input_tokens}")
                key = translation_pool(m, d, size)
                path = out / Path(key).relative_to("artifacts/prompts")
                path.parent.mkdir(parents=True, exist_ok=True)
                data = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines)
                path.write_text(data, encoding="utf-8")
                toks = [x["prompt_tokens_expected"] for x in lines]
                refs = [x["reference_tokens_expected"] for x in lines]
                sets[f"{m.id}/{path.stem}"] = {
                    "n": len(lines), "direction": d, "size": size, "min_tokens": min(toks), "max_tokens": max(toks),
                    "ref_tokens_max": max(refs), "ref_tokens_p95": sorted(refs)[int(0.95 * (len(refs) - 1))],
                    "chars": sum(x["chars"] for x in lines), "sha256": hashlib.sha256(data.encode()).hexdigest(),
                    "dataset_sha256": dataset_manifest["dataset_sha256"]}
                print(f"{m.id}/{path.stem}: {len(lines)} prompts, {min(toks)}-{max(toks)} tokens, "
                      f"reference up to {max(refs)} tokens")
    return sets


def main(argv=None) -> int:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "prompts" / "generated"))
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--hf-token-secret", default=None)
    args = ap.parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    if args.hf_token_secret:
        from tools.build_prompts import token_from_secret

        token = token_from_secret(args.hf_token_secret)
    catalog = load_models(ROOT / "config" / "models.yaml")
    wanted = args.models or [m.id for m in catalog.models if m.enabled and "translation" in m.suites]
    models = [catalog.get(i) for i in wanted]
    items, man = read_items(Path(args.dataset))
    out = Path(args.out)
    sets = build_for_models(items, man, out, models, {m.id: tokenizer_for(m, token) for m in models})
    man_p = out / "manifest.json"
    pm = json.loads(man_p.read_text()) if man_p.exists() else {"sets": {}}
    pm.setdefault("sets", {}).update(sets)
    pm["translation"] = {"dataset_version": man["version"], "dataset_sha256": man["dataset_sha256"], "seed": SEED}
    man_p.write_text(json.dumps(pm, indent=2))
    print(f"wrote {len(sets)} pools; manifest {man_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
