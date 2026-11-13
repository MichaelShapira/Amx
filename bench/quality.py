"""Quality checks for the standard suite (design section 15, Requirement 13)."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")

ROUGE_DROP_FLAG = 0.10
LABEL_AGREEMENT_MIN = 0.98
PROB_DIFF_MAX = 0.05


def _tokens(text: str) -> list[str]:
    # Same normalization as rouge_score's default tokenizer (lowercase, alphanumeric runs).
    return _TOKEN.findall(text.lower())


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l_f1(candidate: str, reference: str) -> float:
    c, r = _tokens(candidate), _tokens(reference)
    if not c or not r:
        return 0.0
    lcs = _lcs(c, r)
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(c), lcs / len(r)
    return 2 * p * rec / (p + rec)


def score_generation(outputs: dict[str, str], reference: dict[str, str]) -> dict:
    ids = sorted(set(outputs) & set(reference))
    scores = [rouge_l_f1(outputs[i], reference[i]) for i in ids]
    exact = [outputs[i].strip() == reference[i].strip() for i in ids]
    return {"n": len(ids), "exact_match_rate": float(np.mean(exact)) if ids else None,
            "rouge_l_mean": float(np.mean(scores)) if ids else None,
            "rouge_l_p10": float(np.percentile(scores, 10)) if ids else None}


def score_classifier(outputs: dict[str, dict], reference: dict[str, dict]) -> dict:
    """outputs[id] = {"label": str, "probs": [float, ...]}"""
    ids = sorted(set(outputs) & set(reference))
    agree = [outputs[i]["label"] == reference[i]["label"] for i in ids]
    diffs = [float(np.max(np.abs(np.asarray(outputs[i]["probs"]) - np.asarray(reference[i]["probs"]))))
             for i in ids]
    return {"n": len(ids), "label_agreement": float(np.mean(agree)) if ids else None,
            "prob_diff_mean": float(np.mean(diffs)) if ids else None,
            "prob_diff_max": float(np.max(diffs)) if ids else None}


BF16_PRECISIONS = ("bf16", "bf16-ov")      # uncompressed BF16, whichever engine serves it


def is_bf16(precision: str) -> bool:
    return precision in BF16_PRECISIONS


def generation_flag(cell_score: dict, cpu_bf16_score: dict | None, *, precision: str) -> str | None:
    if is_bf16(precision) or cpu_bf16_score is None or cell_score.get("rouge_l_mean") is None:
        return None
    floor = cpu_bf16_score["rouge_l_mean"] - ROUGE_DROP_FLAG
    if cell_score["rouge_l_mean"] < floor:
        return f"rouge_l_mean {cell_score['rouge_l_mean']:.3f} < CPU BF16 {cpu_bf16_score['rouge_l_mean']:.3f} - {ROUGE_DROP_FLAG}"
    return None


def classifier_flag(vs_fp32: dict | None, *, precision: str) -> str | None:
    if precision in ("fp32",) or vs_fp32 is None or vs_fp32.get("label_agreement") is None:
        return None
    if vs_fp32["label_agreement"] < LABEL_AGREEMENT_MIN:
        return f"label agreement with FP32 {vs_fp32['label_agreement']:.3f} < {LABEL_AGREEMENT_MIN}"
    if vs_fp32["prob_diff_max"] > PROB_DIFF_MAX:
        return f"max probability difference {vs_fp32['prob_diff_max']:.3f} > {PROB_DIFF_MAX}"
    return None


def quality_delta(score: dict) -> float | None:
    if score.get("rouge_l_mean") is not None:
        return 1.0 - score["rouge_l_mean"]
    if score.get("label_agreement") is not None:
        return 1.0 - score["label_agreement"]
    return None
