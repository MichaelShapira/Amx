"""Dataset build checks (design 24.3). Any failure stops the build.

Per item: the label follows from the categories, the categories follow from the spans,
every positive card span passes the Luhn check, a negative contains no patterned
sensitive value, a positive contains none outside its labeled spans, a masked hard
negative contains none of the values it was made from.

Per dataset: quotas, no exact duplicates, no near-duplicates across splits, no source
document or template phrasing shared between dev and test.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np

from tools.dspm.fakes import luhn_ok
from tools.dspm.scan import findings

PRIORITY = ("CREDENTIALS", "GOV_ID", "FINANCIAL", "HEALTH", "PERSONAL", "CONFIDENTIAL")
CARD_KINDS = {"card", "credit_debit_card", "credit_card_number"}
HARD_NEGATIVE_TYPES = {
    "W1": ("masked", "look-alike-number", "documentation-value", "topic-only", "public-business",
           "first-name-only", "number-without-keyword", "indirect-reference"),
    "W3": ("masked", "look-alike-number", "documentation-value", "topic-only", "public-business",
           "first-name-only", "number-without-keyword", "indirect-reference"),
    "W4": ("masked", "look-alike-number", "documentation-value", "public-business", "first-name-only",
           "number-without-keyword", "indirect-reference"),
}
HARD_POSITIVE_TYPES = {"W1": ("test-labeled-secret",), "W3": ("test-labeled-secret", "keyword-proximity")}


class CheckError(AssertionError):
    pass


def top_category(categories: Iterable[str]) -> str:
    cats = set(categories)
    for c in PRIORITY:
        if c in cats:
            return c
    return "NONE"


def label_for(workload: str, categories: list[str]) -> str:
    if workload == "W3":
        return "BLOCK" if categories else "ALLOW"
    if workload == "W4":
        return top_category(categories)
    return "SENSITIVE" if categories else "NOT_SENSITIVE"


def check_item(item: dict, masked_originals: list[str] | None = None) -> list[str]:
    errs: list[str] = []
    text, spans, cats = item["text"], item["spans"], item["categories"]
    span_cats = sorted({s["category"] for s in spans}, key=PRIORITY.index)
    if list(cats) != span_cats:
        errs.append(f"categories {cats} != span categories {span_cats}")
    if item["sensitive"] != bool(cats):
        errs.append("sensitive flag disagrees with categories")
    if item["label"] != label_for(item["workload"], cats):
        errs.append(f"label {item['label']} != {label_for(item['workload'], cats)}")
    if item["category_top"] != top_category(cats):
        errs.append("category_top disagrees with categories")
    for s in spans:
        if not (0 <= s["start"] < s["end"] <= len(text)):
            errs.append(f"span out of range {s}")
            continue
        if s.get("kind") in CARD_KINDS and not luhn_ok(text[s["start"]:s["end"]]):
            errs.append(f"card span fails Luhn: {text[s['start']:s['end']]!r}")
    for kind, match in findings(text):
        hits = [m.start() for m in re.finditer(re.escape(match), text)]
        covered = any(any(s["start"] < h + len(match) and h < s["end"] for s in spans) for h in hits)
        if not cats or not covered:
            errs.append(f"unlabeled {kind}: {match[:40]!r}")
    if item.get("hard_case") == "masked" and masked_originals:
        for o in masked_originals:
            if len(o) >= 4 and o in text:
                errs.append(f"masked item still contains an original value {o[:12]!r}")
    if bool(item["hard_case"]) and item["sensitive"] and item["hard_case"] not in (
            HARD_POSITIVE_TYPES.get(item["workload"], ())):
        errs.append(f"positive item carries a hard-negative type {item['hard_case']}")
    return errs


# ---------------------------------------------------------------------------- near duplicates

_P = (1 << 31) - 1


def _shingles(text: str, n: int = 5) -> np.ndarray:
    words = re.sub(r"\s+", " ", text.lower()).split(" ")
    grams = {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}
    return np.array(sorted(int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4).digest(), "big")
                           for g in grams), dtype=np.uint64)


def _signature(sh: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a[:, None] * (sh[None, :] % _P) + b[:, None]) % _P).min(axis=1)


def near_duplicates(dev: list[dict], test: list[dict], threshold: float = 0.7,
                    perms: int = 128) -> list[tuple[str, str, float]]:
    rng = np.random.default_rng(12345)
    a = rng.integers(1, _P, size=perms, dtype=np.uint64)
    b = rng.integers(0, _P, size=perms, dtype=np.uint64)
    sh_dev = [_shingles(x["text"]) for x in dev]
    sh_test = [_shingles(x["text"]) for x in test]
    sig_dev = np.stack([_signature(s, a, b) for s in sh_dev]) if dev else np.zeros((0, perms))
    sig_test = np.stack([_signature(s, a, b) for s in sh_test]) if test else np.zeros((0, perms))
    out = []
    for i in range(len(dev)):
        est = (sig_test == sig_dev[i][None, :]).mean(axis=1)
        for j in np.nonzero(est >= threshold - 0.15)[0]:
            s1, s2 = set(sh_dev[i].tolist()), set(sh_test[j].tolist())
            jac = len(s1 & s2) / max(1, len(s1 | s2))
            if jac >= threshold:
                out.append((dev[i]["id"], test[j]["id"], round(jac, 3)))
    return out


# ---------------------------------------------------------------------------- dataset checks

def check_dataset(dev: list[dict], test: list[dict], *, min_pos_per_category: int,
                  min_pos_per_workload: int, min_hard_share: float = 0.30) -> dict:
    errors: list[str] = []
    report: dict = {}
    norm = Counter(re.sub(r"\s+", " ", x["text"].strip().lower()) for x in dev + test)
    dups = sum(c - 1 for c in norm.values() if c > 1)
    if dups:
        errors.append(f"{dups} exact duplicate texts")
    ids = Counter(x["id"] for x in dev + test)
    if any(c > 1 for c in ids.values()):
        errors.append("duplicate item ids")

    def comps(items):
        uids, tpls = set(), set()
        for x in items:
            for c in x["components"]:
                if c.get("uid"):
                    uids.add(c["uid"])
                if c.get("template") is not None:
                    tpls.add((c["family"], c["template"]))
        return uids, tpls

    du, dt = comps(dev)
    tu, tt = comps(test)
    if du & tu:
        errors.append(f"{len(du & tu)} source documents shared between dev and test")
    if dt & tt:
        errors.append(f"{len(dt & tt)} template phrasings shared between dev and test: {sorted(dt & tt)[:3]}")
    nd = near_duplicates(dev, test)
    report["near_duplicates_dev_test"] = len(nd)
    if nd:
        errors.append(f"{len(nd)} near-duplicates across dev and test, e.g. {nd[:3]}")

    by_wl: dict[str, list[dict]] = defaultdict(list)
    for x in test:
        by_wl[x["workload"]].append(x)
    cat_counts = Counter(c for x in test for c in x["categories"])
    report["test_positives_per_category"] = dict(sorted(cat_counts.items()))
    for c in PRIORITY:
        if cat_counts[c] < min_pos_per_category:
            errors.append(f"test positives for {c}: {cat_counts[c]} < {min_pos_per_category}")
    n_pos = sum(x["sensitive"] for x in test)
    report["test_sensitive_share"] = round(n_pos / max(1, len(test)), 4)
    if abs(n_pos / max(1, len(test)) - 0.5) > 0.03:
        errors.append(f"test set not balanced: {n_pos}/{len(test)} sensitive")
    for wl, items in sorted(by_wl.items()):
        pos = [x for x in items if x["sensitive"]]
        neg = [x for x in items if not x["sensitive"]]
        hard = [x for x in neg if x["hard_case"]]
        share = len(hard) / max(1, len(neg))
        types = Counter(x["hard_case"] for x in hard)
        report[f"{wl}"] = {"items": len(items), "positives": len(pos), "negatives": len(neg),
                           "hard_negative_share": round(share, 4), "hard_negative_types": dict(sorted(types.items())),
                           "hard_positive_types": dict(sorted(Counter(x["hard_case"] for x in pos if x["hard_case"]).items())),
                           "positives_per_category": dict(sorted(Counter(c for x in pos for c in x["categories"]).items()))}
        if len(pos) < min_pos_per_workload:
            errors.append(f"{wl}: {len(pos)} positives < {min_pos_per_workload}")
        if share < min_hard_share:
            errors.append(f"{wl}: hard negatives {share:.2%} < {min_hard_share:.0%} of negatives")
        missing = set(HARD_NEGATIVE_TYPES[wl]) - set(types)
        if missing:
            errors.append(f"{wl}: hard-negative types missing: {sorted(missing)}")
        missing_pos = set(HARD_POSITIVE_TYPES.get(wl, ())) - {x["hard_case"] for x in pos}
        if missing_pos:
            errors.append(f"{wl}: hard-positive types missing: {sorted(missing_pos)}")
    report["errors"] = errors
    return report


def wilson_halfwidth(n: int, p: float = 0.5) -> float:
    return 1.96 * math.sqrt(p * (1 - p) / max(1, n))
