"""Prompt rendering tests. They use ungated tokenizers from the local HF cache."""

import json
from pathlib import Path

import pytest

from bench.config import load_models
from bench.prompts import _adds_bos, count_tokens, render, tokenizer_for

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def catalog():
    return load_models(ROOT / "config" / "models.yaml")


def _tok(model):
    try:
        return tokenizer_for(model)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer not available offline: {e}")


def test_qwen_chat_render_counts_match(catalog):
    m = catalog.get("m1-qwen3-1.7b")
    tok = _tok(m)
    r = render(m, tok, "Is this sensitive?", system_text="POLICY")
    assert r["prompt"].startswith("<|im_start|>system\nPOLICY")
    # Qwen3 with enable_thinking=false (prompt_format.template_kwargs): the template closes an empty think block,
    # so the model answers at once instead of thinking first (Intel's recipe does the same server-side).
    assert r["prompt"].rstrip().endswith("<|im_start|>assistant\n<think>\n\n</think>")
    assert r["tokens"] == count_tokens(tok, r["prompt"])
    assert not _adds_bos(tok)


def test_t5_plain_render_adds_eos_only(catalog):
    m = catalog.get("m5-flan-t5-xl")
    tok = _tok(m)
    r = render(m, tok, "hello world")
    ids = tok(r["prompt"], add_special_tokens=True)["input_ids"]
    assert ids[-1] == tok.eos_token_id and r["tokens"] == len(ids)


def test_pair_render(catalog):
    m = catalog.get("m3-roberta-large-mnli")
    tok = _tok(m)
    r = render(m, tok, "A man plays a guitar.", system_text="A person makes music.")
    assert r["pair"] == ["A man plays a guitar.", "A person makes music."]
    assert r["tokens"] == count_tokens(tok, *r["pair"])


def test_generated_sets_match_manifest():
    manifest_path = ROOT / "prompts" / "generated" / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("prompt sets not built")
    import hashlib

    manifest = json.loads(manifest_path.read_text())
    for key, meta in manifest["sets"].items():
        path = ROOT / "prompts" / "generated" / f"{key}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"], key
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        assert len(rows) == meta["n"]
        if "target" in meta:
            assert meta["max_tokens"] <= meta["target"] * 1.02 + 1
