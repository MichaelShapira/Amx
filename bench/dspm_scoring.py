"""DSPM/DLP accuracy scoring (design section 24.7, Requirement 23)."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

BINARY_LABELS = {"W1": ("SENSITIVE", "NOT_SENSITIVE"), "W4": ("SENSITIVE", "NOT_SENSITIVE"),
                 "W3": ("BLOCK", "ALLOW")}
CATEGORIES = ("GOV_ID", "PERSONAL", "FINANCIAL", "HEALTH", "CREDENTIALS", "CONFIDENTIAL", "NONE")
PRIORITY = ("CREDENTIALS", "GOV_ID", "FINANCIAL", "HEALTH", "PERSONAL", "CONFIDENTIAL")


def positive_label(workload: str) -> str:
    return BINARY_LABELS.get(workload, ("SENSITIVE",))[0]


def labels_for(workload: str) -> tuple[str, ...]:
    if workload == "W2":
        return CATEGORIES
    return BINARY_LABELS[workload]


def top_category(categories: Sequence[str]) -> str:
    for c in PRIORITY:
        if c in categories:
            return c
    return "NONE"


def parse_label(text: str | None, workload: str) -> str | None:
    """Strict parser: the first word must be exactly one of the workload's labels."""
    if not text:
        return None
    first = text.strip().split()[0] if text.strip() else ""
    first = first.strip(".,:;\"'`*").upper()
    return first if first in labels_for(workload) else None


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"value": (k / n) if n else None, "k": k, "n": n, "ci95": [lo, hi]}


def score(items: Sequence[dict], predictions: dict[str, str | None], *, workload: str,
          fail_closed: bool = True) -> dict:
    """items: dataset items (id, label, categories, hard_case). predictions: id -> label or None.

    Unparseable outputs (None) count as format failures; with fail_closed they score as the
    positive label, as an inline DLP gate would.
    """
    pos = positive_label(workload) if workload != "W2" else None
    tp = fp = tn = fn = 0
    fmt_fail = 0
    per_cat_hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_cat_lenient: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    hard: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # type -> [false positives, n]
    strict_ok = lenient_ok = 0
    n = 0
    for it in items:
        if it["id"] not in predictions:
            continue
        n += 1
        pred = predictions[it["id"]]
        if pred is None:
            fmt_fail += 1
        if workload == "W2":
            gold_cats = it.get("categories") or []
            gold_top = top_category(gold_cats)
            p = pred if pred is not None else ("CREDENTIALS" if fail_closed else "NONE")
            strict_ok += p == gold_top
            lenient_ok += (p in gold_cats) or (p == "NONE" and not gold_cats)
            gold_pos = bool(gold_cats)
            pred_pos = p != "NONE"
        else:
            gold_pos = it["label"] == pos
            p = pred if pred is not None else (pos if fail_closed else None)
            pred_pos = p == pos
        if gold_pos and pred_pos:
            tp += 1
        elif gold_pos:
            fn += 1
        elif pred_pos:
            fp += 1
        else:
            tn += 1
        for cat in it.get("categories") or []:
            per_cat_hits[cat][1] += 1
            per_cat_hits[cat][0] += int(pred_pos)
            if workload == "W2":
                per_cat_lenient[cat][1] += 1
                per_cat_lenient[cat][0] += int(p == cat)
        if not gold_pos and it.get("hard_case"):
            hard[it["hard_case"]][1] += 1
            hard[it["hard_case"]][0] += int(pred_pos)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    out = {
        "workload": workload, "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": _rate(tp, tp + fn), "precision": _rate(tp, tp + fp),
        "fpr": _rate(fp, fp + tn), "f1": f1,
        "format_adherence": _rate(n - fmt_fail, n),
        "per_category_recall": {c: _rate(k, m) for c, (k, m) in sorted(per_cat_hits.items())},
        "hard_negative_fpr": {h: _rate(k, m) for h, (k, m) in sorted(hard.items())},
        "fail_closed": fail_closed,
    }
    if workload == "W2":
        out["category_accuracy_strict"] = _rate(strict_ok, n)
        out["category_accuracy_lenient"] = _rate(lenient_ok, n)
        out["per_category_exact"] = {c: _rate(k, m) for c, (k, m) in sorted(per_cat_lenient.items())}
    return out


def agreement(cpu_preds: dict[str, str | None], gpu_preds: dict[str, str | None]) -> dict:
    ids = sorted(set(cpu_preds) & set(gpu_preds))
    same = sum(1 for i in ids if cpu_preds[i] == gpu_preds[i])
    return _rate(same, len(ids))


def capacity_qualifies(cpu_score: dict, gpu_score: dict, agree: dict, *,
                       agreement_min: float = 0.98, recall_drop_max_pp: float = 1.0) -> tuple[bool, str]:
    """Requirement 23.3: no meaningful accuracy loss against the GPU cell of the same model."""
    a = agree.get("value")
    if a is None:
        return False, "no overlapping items with the GPU cell"
    if a < agreement_min:
        return False, f"label agreement with GPU {a:.3f} < {agreement_min}"
    cr, gr = cpu_score["recall"]["value"], gpu_score["recall"]["value"]
    if cr is not None and gr is not None and (gr - cr) * 100 > recall_drop_max_pp:
        return False, f"recall drop {(gr - cr) * 100:.2f} pp > {recall_drop_max_pp} pp"
    return True, "accuracy matches GPU"


def fitness(score_: dict, *, recall_min: float = 0.90, fpr_max: float = 0.05) -> dict:
    r = score_["recall"]["value"]
    f = score_["fpr"]["value"]
    return {"recall_ok": r is not None and r >= recall_min,
            "fpr_ok": f is not None and f <= fpr_max,
            "fit": r is not None and f is not None and r >= recall_min and f <= fpr_max,
            "recall_min": recall_min, "fpr_max": fpr_max}
