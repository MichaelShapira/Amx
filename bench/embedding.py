"""Embedding quality for the embedding suite.

Two readings per cell. Agreement: the cosine similarity between a cell's vector for a text and the
GPU BF16 reference cell's vector for the same text (1.0 = identical direction); the mean and the
worst item say whether the CPU produces the same embeddings as the GPU. Retrieval: the embedding
pools hold every corpus text in English, German and Spanish, so for each German or Spanish text the
nearest English text (by cosine, over the whole pool) should be its own translation; the share for
which it is (recall@1) is a task score with a known answer, computed per cell without any reference
cell. Both are read relatively: the same model on two machines should score alike.
"""

from __future__ import annotations

import numpy as np

SIZES = ("sentence", "paragraph")
LANGUAGES = ("en", "de", "es")
COSINE_DROP_FLAG = 0.01          # mean cosine to the CPU BF16 cell below 1 - this flags a compressed format


def item_key(corpus_id: str, lang: str) -> str:
    return f"{corpus_id}:{lang}"


def split_key(item_id: str) -> tuple[str, str]:
    cid, _, lang = item_id.rpartition(":")
    return cid, lang


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(a)
    return a / n if n else a


def agreement(outputs: dict[str, list[float]], reference: dict[str, list[float]]) -> dict:
    ids = sorted(set(outputs) & set(reference))
    cos = [float(np.dot(_unit(outputs[i]), _unit(reference[i]))) for i in ids]
    return {"n": len(ids), "cosine_mean": float(np.mean(cos)) if ids else None,
            "cosine_min": float(np.min(cos)) if ids else None,
            "cosine_p10": float(np.percentile(cos, 10)) if ids else None}


def retrieval(outputs: dict[str, list[float]]) -> dict:
    """recall@1 of German and Spanish texts against the English texts of the same pool."""
    by_lang: dict[str, dict[str, np.ndarray]] = {}
    for k, v in outputs.items():
        cid, lang = split_key(k)
        by_lang.setdefault(lang, {})[cid] = _unit(v)
    en = by_lang.get("en") or {}
    out: dict[str, dict] = {}
    if not en:
        return out
    en_ids = sorted(en)
    en_mat = np.stack([en[i] for i in en_ids])
    for lang in LANGUAGES:
        if lang == "en" or lang not in by_lang:
            continue
        hits = total = 0
        for cid, vec in by_lang[lang].items():
            if cid not in en:
                continue
            total += 1
            best = en_ids[int(np.argmax(en_mat @ vec))]
            hits += best == cid
        out[lang] = {"recall_at_1": hits / total if total else None, "n": total}
    return out


def score(outputs: dict[str, list[float]], reference: dict[str, list[float]] | None) -> dict:
    sc: dict = {"n": len(outputs), "dim": len(next(iter(outputs.values()))) if outputs else None,
                "retrieval": retrieval(outputs)}
    if reference:
        sc["vs_reference_cell"] = agreement(outputs, reference)
    return sc


def flag(cell: dict, cpu_bf16: dict | None, *, precision: str) -> str | None:
    from bench.quality import is_bf16

    if is_bf16(precision) or not cpu_bf16:
        return None
    mine = (cell.get("vs_reference_cell") or {}).get("cosine_mean")
    base = (cpu_bf16.get("vs_reference_cell") or {}).get("cosine_mean")
    if mine is None or base is None:
        return None
    if mine < base - COSINE_DROP_FLAG:
        return f"cosine to GPU {mine:.4f} < CPU BF16 {base:.4f} - {COSINE_DROP_FLAG}"
    return None
