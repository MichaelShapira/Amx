"""Translation quality for the translation suite: chrF against the synthetic reference, and
agreement between a cell's translations and the GPU reference cell's.

chrF (Popović 2015) is the character n-gram F-score sacreBLEU reports as chrF2: character
n-grams of order 1-6 over the text with whitespace removed, precision and recall averaged over
the orders, recall weighted beta=2. Characters rather than words, so German compounds and
Spanish clitics do not need a tokenizer, and a near-miss ("Lieferschein" vs "Lieferscheins")
scores high instead of zero. Values are 0-100.

Scores are read as RELATIVE figures: the same model on two hardware options, or two number
formats of the same model, on the same items. A legitimate paraphrase lowers chrF against the
reference for every cell alike, so the flag compares each cell with the CPU BF16 cell of its
model, as the standard suite does with ROUGE-L.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

CHRF_ORDER = 6
CHRF_BETA = 2.0
CHRF_DROP_FLAG = 3.0           # chrF points below the model's CPU BF16 cell that flag a compressed format
DIRECTIONS = ("en-de", "de-en", "en-es", "es-en")
SIZES = ("sentence", "paragraph")
LANGUAGE = {"en": "English", "de": "German", "es": "Spanish"}
# The source chars a cell translated per second are read from the pool line (chars of the source text).
INSTRUCTION = ("Translate the following text from {src} to {tgt}. Reply with the {tgt} translation only, "
               "without explanations or notes.\n\n{text}")


def split_direction(direction: str) -> tuple[str, str]:
    src, tgt = direction.split("-")
    if src not in LANGUAGE or tgt not in LANGUAGE or src == tgt:
        raise ValueError(f"unknown translation direction {direction}")
    return src, tgt


def prompt_text(direction: str, text: str) -> str:
    src, tgt = split_direction(direction)
    return INSTRUCTION.format(src=LANGUAGE[src], tgt=LANGUAGE[tgt], text=text)


def _ngrams(text: str, n: int) -> Counter:
    s = "".join(text.split())
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(hypothesis: str, reference: str, order: int = CHRF_ORDER, beta: float = CHRF_BETA) -> float:
    """Sentence-level chrF in 0-100 (sacreBLEU's chrF2 with default settings, no word n-grams)."""
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    precisions, recalls = [], []
    for n in range(1, order + 1):
        h, r = _ngrams(hypothesis, n), _ngrams(reference, n)
        if not h or not r:
            continue
        match = sum((h & r).values())
        precisions.append(match / sum(h.values()))
        recalls.append(match / sum(r.values()))
    if not precisions:
        return 0.0
    p, r = float(np.mean(precisions)), float(np.mean(recalls))
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return 100.0 * (1 + b2) * p * r / (b2 * p + r)


def clean(text: str) -> str:
    """What is scored: the answer without a leading label such as
    'German:' and without a quotation wrapper. Models that obey "translation only" are unaffected."""
    t = (text or "").strip()
    for label in ("Translation:", "Übersetzung:", "Traducción:", "German:", "Deutsch:", "Spanish:", "Español:",
                  "English:", "Englisch:", "Inglés:"):
        if t.startswith(label):
            t = t[len(label):].strip()
    if t.startswith(("\"", "“", "„")) and t.endswith(("\"", "”", "“")) and len(t) > 2:
        t = t[1:-1].strip()
    return t

def score(outputs: dict[str, str], references: dict[str, str], *, max_tokens: int | None = None,
          completion_tokens: dict[str, int] | None = None) -> dict:
    """outputs/references: item_id -> text. Items missing on either side are skipped."""
    ids = sorted(set(outputs) & set(references))
    scores = [chrf(clean(outputs[i]), references[i]) for i in ids]
    ratios = [len(clean(outputs[i])) / len(references[i]) for i in ids if references[i]]
    out = {"n": len(ids), "chrf": float(np.mean(scores)) if ids else None,
           "chrf_p10": float(np.percentile(scores, 10)) if ids else None,
           "length_ratio": float(np.mean(ratios)) if ratios else None,
           "empty_pct": 100.0 * sum(1 for i in ids if not clean(outputs[i])) / len(ids) if ids else None}
    if max_tokens and completion_tokens:
        cut = sum(1 for i in ids if (completion_tokens.get(i) or 0) >= max_tokens)
        out["truncated_pct"] = 100.0 * cut / len(ids) if ids else None
    return out


def agreement(outputs: dict[str, str], reference_outputs: dict[str, str]) -> dict:
    """How close a cell's translations are to another cell's (the GPU BF16 reference): chrF of one
    against the other and the share of byte-identical answers."""
    ids = sorted(set(outputs) & set(reference_outputs))
    sc = [chrf(clean(outputs[i]), clean(reference_outputs[i])) for i in ids]
    same = [clean(outputs[i]) == clean(reference_outputs[i]) for i in ids]
    return {"n": len(ids), "chrf": float(np.mean(sc)) if ids else None,
            "exact_match_rate": float(np.mean(same)) if ids else None}


def flag(cell: dict, cpu_bf16: dict | None, *, precision: str) -> str | None:
    """A compressed format is flagged when its chrF against the reference falls more than
    CHRF_DROP_FLAG points below the model's CPU BF16 cell on the same direction."""
    from bench.quality import is_bf16

    if is_bf16(precision) or cpu_bf16 is None or cell.get("chrf") is None or cpu_bf16.get("chrf") is None:
        return None
    floor = cpu_bf16["chrf"] - CHRF_DROP_FLAG
    if cell["chrf"] < floor:
        return f"chrF {cell['chrf']:.1f} < CPU BF16 {cpu_bf16['chrf']:.1f} - {CHRF_DROP_FLAG:g}"
    return None
