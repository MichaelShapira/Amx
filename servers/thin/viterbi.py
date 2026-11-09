"""Constrained BIOES Viterbi decoding and span extraction for D1 (openai/privacy-filter).

Follows the reference decoder in github.com/openai/privacy-filter (Apache-2.0,
opf/_core/decoding.py and opf/_core/spans.py at commit f7f00ca): emissions are per-token
log-probabilities; a path may start in O, B or S and end in O, E or S; O, E and S may be
followed by O, B or S; B and I must be followed by I or E of the same label. Six
transition biases (all zero at the default operating point in viterbi_calibration.json)
shift the background, entry, continuation and closure edges. Written with numpy so the
same code runs on CPU and after a GPU forward pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NEG_INF = -1e9
BIAS_KEYS = ("transition_bias_background_stay", "transition_bias_background_to_start",
             "transition_bias_inside_to_continue", "transition_bias_inside_to_end",
             "transition_bias_end_to_background", "transition_bias_end_to_start")


def load_biases(path: str | Path | None) -> dict[str, float]:
    if path is None or not Path(path).is_file():
        return {k: 0.0 for k in BIAS_KEYS}
    raw = json.loads(Path(path).read_text())
    biases = raw["operating_points"]["default"]["biases"]
    if set(biases) != set(BIAS_KEYS):
        raise ValueError(f"unexpected Viterbi bias keys in {path}: {sorted(biases)}")
    return {k: float(biases[k]) for k in BIAS_KEYS}


def _parse(label: str) -> tuple[str | None, str | None]:
    if label == "O":
        return None, None
    tag, _, name = label.partition("-")
    if tag not in ("B", "I", "E", "S") or not name:
        raise ValueError(f"not a BIOES label: {label!r}")
    return tag, name


@dataclass
class Decoder:
    id2label: dict[int, str]
    biases: dict[str, float]

    def __post_init__(self) -> None:
        k = len(self.id2label)
        parsed = [_parse(self.id2label[i]) for i in range(k)]
        self.tags = [p[0] for p in parsed]
        self.names = [p[1] for p in parsed]
        self.start = np.full(k, NEG_INF)
        self.end = np.full(k, NEG_INF)
        self.trans = np.full((k, k), NEG_INF)
        b = self.biases
        for i, (tag, name) in enumerate(parsed):
            if tag in (None, "B", "S"):
                self.start[i] = 0.0
            if tag in (None, "E", "S"):
                self.end[i] = 0.0
            for j, (ntag, nname) in enumerate(parsed):
                if tag in (None, "E", "S"):
                    if ntag is None:
                        self.trans[i, j] = b["transition_bias_background_stay"] if tag is None \
                            else b["transition_bias_end_to_background"]
                    elif ntag in ("B", "S"):
                        self.trans[i, j] = b["transition_bias_background_to_start"] if tag is None \
                            else b["transition_bias_end_to_start"]
                elif tag in ("B", "I") and nname == name and ntag in ("I", "E"):
                    self.trans[i, j] = b["transition_bias_inside_to_continue"] if ntag == "I" \
                        else b["transition_bias_inside_to_end"]

    def decode(self, logprobs: np.ndarray) -> list[int]:
        """logprobs: [T, K] per-token log-probabilities. Returns the best label path."""
        t_len = logprobs.shape[0]
        if t_len == 0:
            return []
        score = logprobs[0] + self.start
        back = np.empty((t_len - 1, logprobs.shape[1]), dtype=np.int64)
        for t in range(1, t_len):
            cand = score[:, None] + self.trans
            back[t - 1] = cand.argmax(axis=0)
            score = cand.max(axis=0) + logprobs[t]
        if not np.isfinite(score).any() or score.max() <= NEG_INF / 2:
            return logprobs.argmax(axis=1).tolist()
        score = score + self.end
        path = [int(score.argmax())]
        for t in range(t_len - 2, -1, -1):
            path.append(int(back[t, path[-1]]))
        return path[::-1]

    def spans(self, path: list[int]) -> list[tuple[str, int, int]]:
        """Token spans (label, start, end_exclusive) from a decoded path; tolerant of the
        malformed sequences an argmax fallback can produce."""
        out: list[tuple[str, int, int]] = []
        cur, start = None, None
        for t, i in enumerate(path):
            tag, name = self.tags[i], self.names[i]
            if tag is None:
                if cur is not None:
                    out.append((cur, start, t))
                cur = start = None
            elif tag == "S":
                if cur is not None:
                    out.append((cur, start, t))
                out.append((name, t, t + 1))
                cur = start = None
            elif tag == "B" or (tag == "I" and cur != name):
                if cur is not None:
                    out.append((cur, start, t))
                cur, start = name, t
            elif tag == "E":
                if cur == name:
                    out.append((cur, start, t + 1))
                else:
                    if cur is not None:
                        out.append((cur, start, t))
                    out.append((name, t, t + 1))
                cur = start = None
        if cur is not None:
            out.append((cur, start, len(path)))
        return out


def char_spans(text: str, offsets: list[tuple[int, int]], token_spans, probs: np.ndarray,
               path: list[int]) -> list[dict]:
    """Map token spans to character spans (whitespace trimmed) with a mean-probability score."""
    out = []
    for label, s, e in token_spans:
        valid = [k for k in range(s, e) if offsets[k][1] > offsets[k][0]]
        if not valid:
            continue
        cs, ce = offsets[valid[0]][0], offsets[valid[-1]][1]
        while cs < ce and text[cs].isspace():
            cs += 1
        while ce > cs and text[ce - 1].isspace():
            ce -= 1
        if ce <= cs:
            continue
        score = float(np.mean([probs[k, path[k]] for k in valid]))
        out.append({"start": cs, "end": ce, "label": label, "score": round(score, 6)})
    return out


def category_max(probs: np.ndarray, names: list[str | None], mask: np.ndarray) -> dict[str, float]:
    """Per-category maximum over tokens of the probability mass of that category's BIOES classes."""
    out: dict[str, float] = {}
    for cat in sorted({n for n in names if n}):
        cols = [i for i, n in enumerate(names) if n == cat]
        mass = probs[:, cols].sum(axis=1)
        mass = mass[mask] if mask.any() else mass[:0]
        out[cat] = round(float(mass.max()), 6) if mass.size else 0.0
    return out
