"""DSPM/DLP accuracy scoring (design section 24.7, Requirement 23).

Binary workloads (W1 bulk scan, W3 inline DLP) answer one of two labels. Category
workloads (W2 bulk scan by category, W4 tables) answer one category or NONE; for them
"positive" means any category other than NONE. An output that does not parse to a valid
label is a format failure; with fail_closed (the default, as an inline gate would do) it
counts as a positive prediction with no usable category.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

BINARY_LABELS = {"W1": ("SENSITIVE", "NOT_SENSITIVE"), "W3": ("BLOCK", "ALLOW")}
CATEGORY_WORKLOADS = ("W2", "W4")
CATEGORIES = ("GOV_ID", "PERSONAL", "FINANCIAL", "HEALTH", "CREDENTIALS", "CONFIDENTIAL", "NONE")
PRIORITY = ("CREDENTIALS", "GOV_ID", "FINANCIAL", "HEALTH", "PERSONAL", "CONFIDENTIAL")


def positive_label(workload: str) -> str:
    return BINARY_LABELS[workload][0]


def labels_for(workload: str) -> tuple[str, ...]:
    if workload in CATEGORY_WORKLOADS:
        return CATEGORIES
    return BINARY_LABELS[workload]


def top_category(categories: Sequence[str]) -> str:
    for c in PRIORITY:
        if c in categories:
            return c
    return "NONE"


def parse_label(text: str | None, workload: str) -> str | None:
    """Strict parser: the first word must be exactly one of the workload's labels
    (surrounding punctuation and case are ignored)."""
    if not text or not text.strip():
        return None
    first = text.strip().split()[0].strip(".,:;\"'`*()[]").upper()
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
    """items: dataset items (id, label, categories, hard_case). predictions: id -> parsed
    label, or None when the output did not parse. Items without a prediction are skipped."""
    category_mode = workload in CATEGORY_WORKLOADS
    pos = None if category_mode else positive_label(workload)
    tp = fp = tn = fn = fmt_fail = n = 0
    per_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])          # detected as sensitive
    per_cat_exact: dict[str, list[int]] = defaultdict(lambda: [0, 0])    # named the category
    hard_neg: dict[str, list[int]] = defaultdict(lambda: [0, 0])         # [false positives, n]
    hard_pos: dict[str, list[int]] = defaultdict(lambda: [0, 0])         # [detected, n]
    strict_ok = lenient_ok = 0
    for it in items:
        if it["id"] not in predictions:
            continue
        n += 1
        pred = predictions[it["id"]]
        if pred is None:
            fmt_fail += 1
        gold_cats = list(it.get("categories") or [])
        if category_mode:
            gold_pos = bool(gold_cats)
            pred_pos = (pred != "NONE") if pred is not None else fail_closed
            gold_top = top_category(gold_cats)
            strict_ok += pred is not None and pred == gold_top
            lenient_ok += pred is not None and (pred in gold_cats or (pred == "NONE" and not gold_cats))
        else:
            gold_pos = it["label"] == pos
            pred_pos = (pred == pos) if pred is not None else fail_closed
        if gold_pos and pred_pos:
            tp += 1
        elif gold_pos:
            fn += 1
        elif pred_pos:
            fp += 1
        else:
            tn += 1
        for cat in gold_cats:
            per_cat[cat][1] += 1
            per_cat[cat][0] += int(pred_pos)
            if category_mode:
                per_cat_exact[cat][1] += 1
                per_cat_exact[cat][0] += int(pred == cat)
        if it.get("hard_case"):
            bucket = hard_pos if gold_pos else hard_neg
            bucket[it["hard_case"]][1] += 1
            bucket[it["hard_case"]][0] += int(pred_pos)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    out = {
        "workload": workload, "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": _rate(tp, tp + fn), "precision": _rate(tp, tp + fp),
        "fpr": _rate(fp, fp + tn), "f1": f1,
        "format_adherence": _rate(n - fmt_fail, n),
        "per_category_recall": {c: _rate(k, m) for c, (k, m) in sorted(per_cat.items())},
        "hard_negative_fpr": {h: _rate(k, m) for h, (k, m) in sorted(hard_neg.items())},
        "hard_positive_recall": {h: _rate(k, m) for h, (k, m) in sorted(hard_pos.items())},
        "fail_closed": fail_closed,
    }
    if category_mode:
        out["category_accuracy_strict"] = _rate(strict_ok, n)
        out["category_accuracy_lenient"] = _rate(lenient_ok, n)
        out["per_category_exact"] = {c: _rate(k, m) for c, (k, m) in sorted(per_cat_exact.items())}
    return out


def agreement(cpu_preds: dict[str, str | None], gpu_preds: dict[str, str | None]) -> dict:
    """Per-item label agreement (Requirement 23.2); two format failures count as agreeing."""
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
    """Requirement 23.4: absolute floors, reported separately from the CPU-vs-GPU verdict."""
    r = score_["recall"]["value"]
    f = score_["fpr"]["value"]
    return {"recall_ok": r is not None and r >= recall_min,
            "fpr_ok": f is not None and f <= fpr_max,
            "fit": r is not None and f is not None and r >= recall_min and f <= fpr_max,
            "recall_min": recall_min, "fpr_max": fpr_max}


def choose_threshold(dev_scores: dict[str, float], dev_items: Sequence[dict], *, workload: str) -> dict:
    """Pick the decision threshold of a score-based classifier (M3's highest entailment
    probability) on the dev set only: the threshold with the best F1, ties broken by the
    lower false positive rate, then by the higher threshold."""
    pos = None if workload in CATEGORY_WORKLOADS else positive_label(workload)
    gold = {it["id"]: (bool(it.get("categories")) if pos is None else it["label"] == pos)
            for it in dev_items if it["id"] in dev_scores}
    if not gold:
        raise ValueError("no dev items with scores")
    best = None
    for t in sorted(set(dev_scores[i] for i in gold)):
        tp = sum(1 for i, g in gold.items() if g and dev_scores[i] >= t)
        fp = sum(1 for i, g in gold.items() if not g and dev_scores[i] >= t)
        fn = sum(1 for i, g in gold.items() if g and dev_scores[i] < t)
        tn = len(gold) - tp - fp - fn
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        key = (f1, -fpr, t)
        if best is None or key > best[0]:
            best = (key, {"threshold": t, "f1": f1, "fpr": fpr, "n": len(gold)})
    return best[1]
