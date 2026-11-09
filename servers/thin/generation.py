"""Batched generation with per-row streaming, stop strings and per-row lengths (design 7.5).

Hugging Face generate() runs a batch to completion (static batching). A batch can mix
requests with different max_tokens, ignore_eos and stop strings: a per-row logits
processor keeps EOS suppressed until a row reaches its minimum length, a per-row stopping
criterion ends a row at its own max_tokens or stop string, and the multi-row streamer
sends each row's text deltas to that row's callback as soon as the step that produced
them finishes. Rows that end early keep computing padding until the batch ends, as in
any static batch; their output is not affected.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.generation.streamers import BaseStreamer


@dataclass
class GenJob:
    prompt: str
    max_tokens: int
    min_tokens: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    stop: list[str] = field(default_factory=list)
    on_text: Callable[[str], None] = lambda s: None
    on_done: Callable[[str, int, int], None] = lambda reason, n_out, n_in: None   # finish_reason, completion, prompt
    prompt_tokens: int = 0

    def sampling_key(self) -> tuple:
        """Requests in one generate() call must share the sampling settings."""
        if self.temperature <= 0:
            return ("greedy",)
        return ("sample", round(self.temperature, 6), round(self.top_p, 6), self.seed)


class _RowState:
    def __init__(self, jobs: list[GenJob], tok, eos_ids: set[int]):
        self.jobs, self.tok, self.eos = jobs, tok, eos_ids
        n = len(jobs)
        self.ids: list[list[int]] = [[] for _ in range(n)]
        self.sent: list[str] = [""] * n
        self.finished = [False] * n
        self.reason = ["length"] * n
        self.lock = threading.Lock()

    def _flush(self, r: int, final: bool) -> None:
        text = self.tok.decode(self.ids[r], skip_special_tokens=True)
        job = self.jobs[r]
        for s in job.stop:
            k = text.find(s)
            if k >= 0:
                text = text[:k]
                self.finished[r] = True
                self.reason[r] = "stop"
                final = True
                break
        if not final and text.endswith("\ufffd"):
            return       # incomplete UTF-8 sequence; wait for the next token
        if not final:
            # Hold back a suffix that could be the start of a stop string.
            hold = max((k for s in job.stop for k in range(1, len(s)) if text.endswith(s[:k])), default=0)
            text = text[:len(text) - hold]
        if len(text) > len(self.sent[r]) and text.startswith(self.sent[r]):
            delta = text[len(self.sent[r]):]
            self.sent[r] = text
            job.on_text(delta)

    def step(self, tokens: list[int]) -> None:
        with self.lock:
            for r, t in enumerate(tokens):
                if self.finished[r]:
                    continue
                self.ids[r].append(t)
                job = self.jobs[r]
                if t in self.eos and len(self.ids[r]) > job.min_tokens:
                    self.finished[r] = True
                    self.reason[r] = "stop"
                    self.ids[r].pop()
                    self._flush(r, final=True)
                    self.ids[r].append(t)       # EOS counts as a completion token, as in vLLM
                    continue
                self._flush(r, final=False)
                if not self.finished[r] and len(self.ids[r]) >= job.max_tokens:
                    self.finished[r] = True
                    self.reason[r] = "length"
                    self._flush(r, final=True)


class MultiRowStreamer(BaseStreamer):
    """generate() calls put() first with the prompt (decoder-only) or the decoder start
    tokens (encoder-decoder), then once per step with one new token per row."""

    def __init__(self, state: _RowState):
        self.state = state
        self.first = True

    def put(self, value) -> None:
        if self.first:
            self.first = False
            return
        v = value.reshape(-1).tolist()
        self.state.step([int(x) for x in v])

    def end(self) -> None:
        pass


class _RowMinTokens(LogitsProcessor):
    def __init__(self, state: _RowState, eos_ids: list[int]):
        self.state, self.eos = state, eos_ids

    def __call__(self, input_ids, scores):
        for r, job in enumerate(self.state.jobs):
            if len(self.state.ids[r]) < job.min_tokens:
                scores[r, self.eos] = -float("inf")
        return scores


class _RowDone(StoppingCriteria):
    def __init__(self, state: _RowState):
        self.state = state

    def __call__(self, input_ids, scores, **kwargs):
        return torch.tensor(self.state.finished, dtype=torch.bool, device=input_ids.device)


def eos_ids(model, tok) -> set[int]:
    ids: set[int] = set()
    for src in (getattr(model, "generation_config", None), getattr(model, "config", None)):
        e = getattr(src, "eos_token_id", None) if src is not None else None
        if isinstance(e, int):
            ids.add(e)
        elif isinstance(e, (list, tuple)):
            ids.update(int(x) for x in e)
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    return ids


def run_batch(model, tok, jobs: list[GenJob], *, device: str, encoder_decoder: bool,
              max_input_tokens: int) -> None:
    """Run one static batch to completion and report each row through its callbacks."""
    eos = eos_ids(model, tok)
    state = _RowState(jobs, tok, eos)
    enc = tok([j.prompt for j in jobs], return_tensors="pt", padding=True, truncation=True,
              max_length=max_input_tokens, add_special_tokens=True)
    lengths = enc["attention_mask"].sum(dim=1).tolist()
    for j, n in zip(jobs, lengths, strict=True):
        j.prompt_tokens = int(n)
    enc = {k: v.to(device) for k, v in enc.items()}
    kwargs = dict(max_new_tokens=max(j.max_tokens for j in jobs), streamer=MultiRowStreamer(state),
                  logits_processor=LogitsProcessorList([_RowMinTokens(state, sorted(eos))]),
                  stopping_criteria=StoppingCriteriaList([_RowDone(state)]),
                  pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else sorted(eos)[0])
    head = jobs[0]
    if head.temperature > 0:
        if head.seed is not None:
            torch.manual_seed(head.seed)
        kwargs.update(do_sample=True, temperature=head.temperature, top_p=head.top_p)
    else:
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    with torch.inference_mode():
        model.generate(**enc, **kwargs)
    for r, job in enumerate(jobs):
        if not state.finished[r]:
            state.finished[r] = True
            state._flush(r, final=True)
        job.on_done(state.reason[r], len(state.ids[r]), job.prompt_tokens)
