"""DSPM/DLP scoring with hand-computed confusion matrices and intervals (task 22)."""

from __future__ import annotations

import pytest

from bench import dspm_scoring as S


def _items(spec):
    """spec: list of (id, label, categories, hard_case)."""
    return [{"id": i, "label": lab, "categories": cats, "hard_case": h} for i, lab, cats, h in spec]


def test_wilson_known_values():
    lo, hi = S.wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4) and hi == pytest.approx(0.9433, abs=1e-4)
    lo, hi = S.wilson(0, 20)
    assert lo == 0.0 and hi == pytest.approx(0.1611, abs=1e-4)
    assert S.wilson(0, 0) == (None, None)


@pytest.mark.parametrize("text,wl,expect", [
    ("SENSITIVE", "W1", "SENSITIVE"), (" not_sensitive.\n", "W1", "NOT_SENSITIVE"),
    ("BLOCK because", "W3", "BLOCK"), ("**ALLOW**", "W3", "ALLOW"), ("Sensitive data", "W1", "SENSITIVE"),
    ("The answer is BLOCK", "W3", None), ("", "W3", None), (None, "W1", None), ("YES", "W1", None),
    ("GOV_ID", "W2", "GOV_ID"), ("none", "W4", "NONE"), ("SSN", "W4", None),
])
def test_parse_label_is_strict(text, wl, expect):
    assert S.parse_label(text, wl) == expect


def test_binary_confusion_and_fail_closed():
    items = _items([
        ("a", "BLOCK", ["FINANCIAL"], None), ("b", "BLOCK", ["CREDENTIALS"], "test-labeled-secret"),
        ("c", "BLOCK", ["GOV_ID"], None), ("d", "ALLOW", [], "masked"), ("e", "ALLOW", [], None),
        ("f", "ALLOW", [], "masked"), ("g", "ALLOW", [], None),
    ])
    preds = {"a": "BLOCK", "b": None, "c": "ALLOW", "d": "BLOCK", "e": "ALLOW", "f": "ALLOW", "g": None}
    r = S.score(items, preds, workload="W3")
    # fail closed: b (gold BLOCK) -> tp, g (gold ALLOW) -> fp
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (2, 1, 2, 2)
    assert r["recall"]["value"] == pytest.approx(2 / 3)
    assert r["fpr"]["value"] == pytest.approx(2 / 4)
    assert r["precision"]["value"] == pytest.approx(2 / 4)
    assert r["f1"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert r["format_adherence"]["value"] == pytest.approx(5 / 7)
    assert r["hard_negative_fpr"]["masked"]["value"] == pytest.approx(0.5)
    assert r["hard_positive_recall"]["test-labeled-secret"]["value"] == 1.0
    assert r["per_category_recall"]["GOV_ID"]["value"] == 0.0
    open_ = S.score(items, preds, workload="W3", fail_closed=False)
    assert (open_["tp"], open_["fn"], open_["fp"], open_["tn"]) == (1, 2, 1, 3)


def test_category_workloads_strict_and_lenient():
    items = _items([
        ("a", "CREDENTIALS", ["CREDENTIALS", "PERSONAL"], None), ("b", "GOV_ID", ["GOV_ID"], None),
        ("c", "NONE", [], "look-alike-number"), ("d", "NONE", [], None), ("e", "HEALTH", ["HEALTH"], None),
    ])
    preds = {"a": "PERSONAL", "b": "GOV_ID", "c": "FINANCIAL", "d": "NONE", "e": None}
    for wl in ("W2", "W4"):
        r = S.score(items, preds, workload=wl)
        assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (3, 0, 1, 1)
        assert r["category_accuracy_strict"]["value"] == pytest.approx(2 / 5)   # b, d
        assert r["category_accuracy_lenient"]["value"] == pytest.approx(3 / 5)  # a, b, d
        assert r["per_category_exact"]["PERSONAL"]["value"] == 1.0
        assert r["per_category_exact"]["CREDENTIALS"]["value"] == 0.0
        assert r["hard_negative_fpr"]["look-alike-number"]["value"] == 1.0


def test_agreement_and_capacity_rule():
    cpu = {str(i): "BLOCK" for i in range(100)}
    gpu = dict(cpu)
    gpu["0"] = "ALLOW"
    a = S.agreement(cpu, gpu)
    assert a["value"] == pytest.approx(0.99)
    good = {"recall": {"value": 0.95}}
    assert S.capacity_qualifies(good, {"recall": {"value": 0.955}}, a)[0]
    ok, why = S.capacity_qualifies({"recall": {"value": 0.93}}, {"recall": {"value": 0.95}}, a)
    assert not ok and "recall drop" in why
    gpu.update({"1": "ALLOW", "2": "ALLOW"})
    ok, why = S.capacity_qualifies(good, good, S.agreement(cpu, gpu))
    assert not ok and "agreement" in why


def test_fitness_floors():
    f = S.fitness({"recall": {"value": 0.92}, "fpr": {"value": 0.06}})
    assert f["recall_ok"] and not f["fpr_ok"] and not f["fit"]


def test_threshold_chosen_on_dev():
    items = _items([("p1", "BLOCK", ["X"], None), ("p2", "BLOCK", ["X"], None), ("p3", "BLOCK", ["X"], None),
                    ("n1", "ALLOW", [], None), ("n2", "ALLOW", [], None), ("n3", "ALLOW", [], None)])
    scores = {"p1": 0.9, "p2": 0.8, "p3": 0.4, "n1": 0.5, "n2": 0.2, "n3": 0.1}
    t = S.choose_threshold(scores, items, workload="W3")
    # t=0.4 -> tp 3, fp 1, F1 6/7; t=0.8 -> tp 2, fp 0, F1 4/5; best is 0.4
    assert t["threshold"] == 0.4 and t["f1"] == pytest.approx(6 / 7)
