"""Client-side prompt rendering (Requirement 7.2).

Chat templates are applied here, never on the server, so every engine receives identical
text. When the tokenizer adds a BOS token itself, a BOS at the start of the rendered
template is removed; otherwise /v1/completions would see it twice (vLLM and OVMS both
tokenize completion prompts with special tokens).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from bench.config import ModelSpec

HYPOTHESIS_STANDARD = "This passage describes events and characters in a story."


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


@lru_cache(maxsize=16)
def _tokenizer(repo: str, revision: str, token: str | None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(repo, revision=revision, token=token)


def tokenizer_for(model: ModelSpec, token: str | None = None):
    return _tokenizer(model.hf_repo, model.revision, token if model.gated else None)


def _adds_bos(tok) -> bool:
    if tok.bos_token_id is None:
        return False
    ids = tok("x", add_special_tokens=True)["input_ids"]
    return bool(ids) and ids[0] == tok.bos_token_id


def render_chat(tok, user_text: str, system_text: str | None = None) -> str:
    messages: list[dict[str, Any]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001 - templates without a system role (Gemma 2) raise here
        if not system_text:
            raise
        merged = [{"role": "user", "content": f"{system_text}\n\n{user_text}"}]
        text = tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)
    if _adds_bos(tok) and tok.bos_token and text.startswith(tok.bos_token):
        text = text[len(tok.bos_token):]
    return text


def count_tokens(tok, text: str, pair: str | None = None) -> int:
    if pair is not None:
        return len(tok(text, pair, add_special_tokens=True)["input_ids"])
    return len(tok(text, add_special_tokens=True)["input_ids"])


def render(model: ModelSpec, tok, user_text: str, system_text: str | None = None) -> dict:
    """Render one request input for a model; returns {"prompt" | "pair", "tokens"}."""
    kind = model.prompt_format.kind
    if kind == "chat_template":
        prompt = render_chat(tok, user_text, system_text)
        return {"prompt": prompt, "tokens": count_tokens(tok, prompt)}
    if kind == "plain":
        prompt = f"{system_text}\n\n{user_text}" if system_text else user_text
        return {"prompt": prompt, "tokens": count_tokens(tok, prompt)}
    if kind == "prefix":
        prompt = (model.prompt_format.prefix or "") + user_text
        return {"prompt": prompt, "tokens": count_tokens(tok, prompt)}
    if kind == "pair":
        hyp = system_text or HYPOTHESIS_STANDARD
        return {"pair": [user_text, hyp], "tokens": count_tokens(tok, user_text, hyp)}
    if kind == "roblox_chat":
        prompt = ("Instruct: In the following chat messages from target speaker t and possibly "
                  "other speakers s1, s2, etc., detect abuse by speaker t.\nQuery:\n"
                  f"t: {user_text}")
        return {"prompt": prompt, "tokens": min(count_tokens(tok, prompt), 512)}
    raise ValueError(f"unknown prompt format {kind}")
