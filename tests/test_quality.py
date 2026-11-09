import pytest
from rouge_score import rouge_scorer

from bench import dspm_scoring as ds
from bench import quality as q


@pytest.mark.parametrize("cand,ref", [
    ("The cat sat on the mat.", "The cat is on the mat"),
    ("Hello world", "hello, World!"),
    ("completely different", "nothing shared here"),
    ("A B C D E F", "A C E G"),
])
def test_rouge_l_matches_reference_implementation(cand, ref):
    expected = rouge_scorer.RougeScorer(["rougeL"]).score(ref, cand)["rougeL"].fmeasure
    assert q.rouge_l_f1(cand, ref) == pytest.approx(expected)


def test_score_generation_and_flag():
    out = q.score_generation({"a": "x y z", "b": "same"}, {"a": "x y z", "b": "same"})
    assert out["exact_match_rate"] == 1.0 and out["rouge_l_mean"] == 1.0
    assert q.generation_flag({"rouge_l_mean": 0.7}, {"rouge_l_mean": 0.9}, precision="int4-ov")
    assert q.generation_flag({"rouge_l_mean": 0.85}, {"rouge_l_mean": 0.9}, precision="int4-ov") is None
    assert q.generation_flag({"rouge_l_mean": 0.1}, {"rouge_l_mean": 0.9}, precision="bf16") is None


def test_classifier_scores_and_flag():
    a = {"1": {"label": "X", "probs": [0.9, 0.1]}, "2": {"label": "Y", "probs": [0.4, 0.6]}}
    b = {"1": {"label": "X", "probs": [0.88, 0.12]}, "2": {"label": "X", "probs": [0.55, 0.45]}}
    s = q.score_classifier(a, b)
    assert s["label_agreement"] == 0.5
    assert s["prob_diff_max"] == pytest.approx(0.15)
    assert "label agreement" in q.classifier_flag(s, precision="fp16")


def test_wilson_interval():
    lo, hi = ds.wilson(95, 100)
    assert lo == pytest.approx(0.8882, abs=1e-3) and hi == pytest.approx(0.9785, abs=1e-3)
    assert ds.wilson(0, 0) == (None, None)


def test_parse_label_strict():
    assert ds.parse_label(" SENSITIVE\n", "W1") == "SENSITIVE"
    assert ds.parse_label("Block.", "W3") == "BLOCK"
    assert ds.parse_label("Yes it is sensitive", "W1") is None
    assert ds.parse_label("", "W1") is None
    assert ds.parse_label("CREDENTIALS", "W2") == "CREDENTIALS"


def _items():
    return [
        {"id": "p1", "label": "BLOCK", "categories": ["FINANCIAL"], "hard_case": None},
        {"id": "p2", "label": "BLOCK", "categories": ["CREDENTIALS"], "hard_case": None},
        {"id": "p3", "label": "BLOCK", "categories": ["GOV_ID"], "hard_case": None},
        {"id": "n1", "label": "ALLOW", "categories": [], "hard_case": "masked_value"},
        {"id": "n2", "label": "ALLOW", "categories": [], "hard_case": "documentation_value"},
        {"id": "n3", "label": "ALLOW", "categories": [], "hard_case": None},
    ]


def test_binary_confusion_and_fail_closed():
    preds = {"p1": "BLOCK", "p2": "BLOCK", "p3": "ALLOW", "n1": "BLOCK", "n2": None, "n3": "ALLOW"}
    s = ds.score(_items(), preds, workload="W3", fail_closed=True)
    # n2 is unparseable -> counted as BLOCK (false positive)
    assert (s["tp"], s["fn"], s["fp"], s["tn"]) == (2, 1, 2, 1)
    assert s["recall"]["value"] == pytest.approx(2 / 3)
    assert s["fpr"]["value"] == pytest.approx(2 / 3)
    assert s["format_adherence"]["value"] == pytest.approx(5 / 6)
    assert s["hard_negative_fpr"]["masked_value"]["value"] == 1.0
    assert s["per_category_recall"]["GOV_ID"]["value"] == 0.0
    open_ = ds.score(_items(), preds, workload="W3", fail_closed=False)
    assert open_["fp"] == 1


def test_category_scoring():
    items = [{"id": "a", "label": "SENSITIVE", "categories": ["PERSONAL", "CREDENTIALS"], "hard_case": None},
             {"id": "b", "label": "NOT_SENSITIVE", "categories": [], "hard_case": None}]
    s = ds.score(items, {"a": "PERSONAL", "b": "NONE"}, workload="W2")
    assert s["category_accuracy_strict"]["value"] == 0.5  # gold top is CREDENTIALS
    assert s["category_accuracy_lenient"]["value"] == 1.0
    assert s["recall"]["value"] == 1.0


def test_capacity_rule_and_fitness():
    preds = {"p1": "BLOCK", "p2": "BLOCK", "p3": "BLOCK", "n1": "ALLOW", "n2": "ALLOW", "n3": "ALLOW"}
    gpu = ds.score(_items(), preds, workload="W3")
    cpu_preds = dict(preds, p3="ALLOW")
    cpu = ds.score(_items(), cpu_preds, workload="W3")
    agree = ds.agreement(cpu_preds, preds)
    ok, why = ds.capacity_qualifies(cpu, gpu, agree)
    assert not ok and "agreement" in why
    ok2, _ = ds.capacity_qualifies(gpu, gpu, ds.agreement(preds, preds))
    assert ok2
    fit = ds.fitness(gpu)
    assert fit["fit"] is True
