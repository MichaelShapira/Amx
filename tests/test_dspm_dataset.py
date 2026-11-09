"""Tests for the DSPM/DLP dataset builder (task 20.5).

Source records are synthesized in memory, so these tests need neither the parquet files
nor the tokenizers. Real-format fake secrets are generated at run time; no such literal
appears in this file.
"""

from __future__ import annotations

import random
import re

import pytest

from bench.dspm import load_policy
from tools.dspm import build_dataset as B
from tools.dspm import checks, fakes, families, generator, sources, tables
from tools.dspm.scan import findings, masked_ok

WORDS = ("account review schedule meeting quarterly service update request portal office record "
         "summary payment policy report status customer manager project budget invoice support "
         "renewal contract travel training equipment delivery feedback survey analysis planning "
         "security network software license vendor approval holiday parking facility budget audit").split()


# ---------------------------------------------------------------------------- fakes and scanner

def test_check_digit_generators():
    rng = random.Random(1)
    for _ in range(300):
        card = fakes.card_number(rng)
        assert fakes.luhn_ok(card) and card not in fakes.DOC_TEST_CARDS
        assert not fakes.luhn_ok(fakes.luhn_break(card))
        assert fakes.iban_ok(fakes.iban(rng))
        assert fakes.aba_ok(fakes.aba_routing(rng))
        area, group, serial = fakes.ssn(rng).split("-")
        assert area not in ("000", "666") and not area.startswith("9") and group != "00" and serial != "0000"
        assert not fakes.luhn_ok(re.sub(r"\D", "", fakes.order_number16(rng)))
    assert all(fakes.luhn_ok(c) for c in fakes.DOC_TEST_CARDS)


def test_masks_show_at_most_four_digits():
    rng = random.Random(2)
    for _ in range(100):
        assert masked_ok(fakes.mask_ssn(fakes.ssn(rng)))
        assert masked_ok(fakes.mask_card(rng, fakes.card(rng)))
        assert masked_ok(fakes.mask_account(rng, fakes.bank_account(rng)))
        assert masked_ok(fakes.mask_phone("(415) 555-0199"))


@pytest.mark.parametrize("make", [
    lambda r: fakes.card(r), lambda r: fakes.ssn(r), lambda r: fakes.iban(r), lambda r: fakes.aws_key_id(r),
    lambda r: fakes.github_pat(r), lambda r: fakes.slack_token(r), lambda r: fakes.stripe_key(r),
    lambda r: fakes.google_api_key(r), lambda r: fakes.openai_key(r), lambda r: fakes.npm_token(r),
    lambda r: fakes.jwt(r), lambda r: fakes.pem_private_key(r), lambda r: fakes.db_uri(r, "app", "db.internal"),
    lambda r: "password: " + fakes.password(r), lambda r: "STRICTLY CONFIDENTIAL",
])
def test_scanner_finds_real_format_values(make):
    rng = random.Random(3)
    for _ in range(20):
        assert findings(f"note: {make(rng)} end"), make(rng)


@pytest.mark.parametrize("text", [
    "aws_access_key_id = AKIAIOSFODNN7EXAMPLE", "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "use 4242 4242 4242 4242 in staging", "api_key: YOUR_API_KEY", "password: <password>",
    "DB_PASSWORD=${DB_PASSWORD}", "SSN on file ***-**-6789", "card ending in 4242",
    "Ticket 412559087 closed", "order 4716 2389 1045 7732", "commit 5a4645c19a51f0d3683912781a6c0570940885402203",
    "how do I rotate API keys?", "the password is in the vault",
])
def test_scanner_ignores_placeholders_masks_and_look_alikes(text):
    assert findings(text) == []


# ---------------------------------------------------------------------------- templates

def test_families_validate_and_render_consistently():
    families.validate()
    for f in families.FAMILIES:
        for i, t in enumerate(f.templates):
            for k in range(5):
                text, spans = generator.render(t, generator.Ctx(generator.derive_seed(f.name, i, k)))
                assert "<<" not in text
                cats = {s["category"] for s in spans}
                assert cats == set(generator.template_categories(t))
                for s in spans:
                    if s["kind"] == "card":
                        assert fakes.luhn_ok(text[s["start"]:s["end"]])
                hits = findings(text)
                if f.kind == "neg":
                    assert hits == [], (f.name, i, hits)
                for _, m in hits:
                    pos = text.find(m)
                    assert any(s["start"] < pos + len(m) and pos < s["end"] for s in spans), (f.name, m)


def test_template_split_is_disjoint():
    for f in families.FAMILIES:
        dev = set(generator.templates_for(f, "dev"))
        test = set(generator.templates_for(f, "test"))
        assert dev and test and not dev & test


def test_tables_label_from_columns():
    for kind in tables.TABLES:
        for split in ("dev", "test"):
            c = tables.generate(kind, split, generator.derive_seed(kind, split))
            assert set(c.categories) == tables.categories_of(kind)
            assert c.text.count("\n") >= 15
            if not c.categories:
                assert findings(c.text) == [], kind


# ---------------------------------------------------------------------------- source transforms

def _record(uid: str, parts: list[tuple[str, str | None]], source: str, smap: sources.SpanMap) -> sources.Record:
    text, raw = "", []
    for chunk, label in parts:
        if label:
            raw.append({"start": len(text), "end": len(text) + len(chunk), "label": label})
        text += chunk
    spans = sources._make_spans(smap, source, text, raw)
    assert spans is not None
    return sources.Record(uid, source, "Form", "structured", "Banking", text, spans)


def _filler_words(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n)).capitalize() + "."


def fixture_records(n: int, seed: int, smap: sources.SpanMap, source: str = "S1") -> list[sources.Record]:
    rng = random.Random(seed)
    recs = []
    kinds = ["ssn", "card", "mrn", "password", "personal", "none", "none"]
    for i in range(n):
        ctx = generator.Ctx(generator.derive_seed("fixture", source, seed, i))
        first, last = ctx.get("first_name"), ctx.get("last_name")
        kind = kinds[i % len(kinds)]
        name_labels = ("name", None) if source == "S2" else ("first_name", "last_name")
        if source == "S2":
            parts = [(_filler_words(rng, 25) + "\nName: ", None), (f"{first} {last}", "name")]
        else:
            parts = [(_filler_words(rng, 25) + "\nFirst Name: ", None), (first, name_labels[0]),
                     ("\nLast Name: ", None), (last, name_labels[1])]
        if kind == "ssn":
            parts += [("\nSSN: ", None), (fakes.ssn(rng), "ssn")]
        elif kind == "card":
            label = "credit_card_number" if source == "S2" else "credit_debit_card"
            parts += [("\nCard number: ", None), (fakes.card(rng), label)]
        elif kind == "mrn" and source == "S1":
            parts += [("\nMedical record number: ", None), (str(rng.randint(10**7, 10**8 - 1)), "medical_record_number")]
        elif kind == "password":
            parts += [("\nPassword: ", None), (fakes.password(rng), "password")]
        elif kind == "personal":
            parts += [("\nDate of birth: ", None), (ctx.get("dob"), "date_of_birth")]
        parts += [("\n" + _filler_words(rng, 30), None)]
        recs.append(_record(f"{source.lower()}-fx-{seed}-{i}", parts, source, smap))
    return recs


@pytest.fixture(scope="module")
def smap():
    return sources.SpanMap()


def test_positive_sanitize_mask(smap):
    rng = random.Random(5)
    recs = fixture_records(40, 1, smap)
    card_recs = [r for r in recs if any(s.label == "credit_debit_card" for s in r.spans)]
    for r in recs:
        cats = smap.categories("nemotron-pii", r.text, r.spans)
        p = sources.positive(r, rng)
        for s in p.spans:
            if s.label == "credit_debit_card":
                assert fakes.luhn_ok(p.text[s.start:s.end])
        san = sources.sanitize(r, rng, smap)
        assert san is not None and not smap.categories("nemotron-pii", san.text, san.spans)
        assert findings(san.text) == []
        if cats:
            m = sources.mask(r, rng, smap)
            assert m is not None
            masked, originals = m
            assert all(o not in masked.text for o in originals if len(o) >= 4)
            assert findings(masked.text) == []
    assert card_recs


# ---------------------------------------------------------------------------- full build

SMALL = dict(test_size=240, dev_size=80, short=(60, 140), long=(260, 420), w3=(16, 160), w4=(60, 900),
             caps_short={}, caps_w3={}, table_rows=15)


def _build(seed: int, smap):
    s1 = fixture_records(900, 11, smap, "S1")
    s2 = fixture_records(300, 12, smap, "S2")
    cfg = B.BuildConfig(seed=seed, **SMALL)
    return B.build(cfg, s1, s2, B.TokenCounter(None), smap)


@pytest.fixture(scope="module")
def built(smap):
    return _build(7, smap)


def test_build_passes_all_checks(built):
    splits, failures, report = built
    assert failures == []
    assert report["errors"] == []
    assert len(splits["test"]) == SMALL["test_size"]
    assert len(splits["dev"]) == SMALL["dev_size"]


def test_same_seed_same_hash_and_bytes(built, smap, tmp_path):
    splits, _, _ = built
    again, _, _ = _build(7, smap)
    assert B.dataset_hash(splits) == B.dataset_hash(again)
    a = B.write_jsonl_gz(tmp_path / "a.jsonl.gz", splits["test"])
    b = B.write_jsonl_gz(tmp_path / "b.jsonl.gz", again["test"])
    assert a == b
    assert B.read_jsonl_gz(tmp_path / "a.jsonl.gz") == splits["test"]
    other, _, _ = _build(8, smap)
    assert B.dataset_hash(other) != B.dataset_hash(splits)


def test_quotas(built):
    splits, _, report = built
    test = splits["test"]
    assert abs(sum(x["sensitive"] for x in test) / len(test) - 0.5) <= 0.03
    for wl in ("W1", "W3", "W4"):
        r = report[wl]
        assert r["hard_negative_share"] >= 0.30
        assert set(checks.HARD_NEGATIVE_TYPES[wl]) <= set(r["hard_negative_types"])
        assert set(checks.HARD_POSITIVE_TYPES.get(wl, ())) <= set(r["hard_positive_types"])
    for c in checks.PRIORITY:
        assert report["test_positives_per_category"][c] >= 0.05 * len(test)
    w2 = [x for x in test if "W2" in x["workloads"]]
    assert w2 and all(x["bucket"] == "short" for x in w2)


def test_no_split_leakage(built):
    splits, _, _ = built
    def comps(items):
        return ({c["uid"] for x in items for c in x["components"] if c["uid"]},
                {(c["family"], c["template"]) for x in items for c in x["components"] if c["template"] is not None})
    du, dt = comps(splits["dev"])
    tu, tt = comps(splits["test"])
    assert du and tu and not du & tu
    assert dt and tt and not dt & tt
    texts = [" ".join(x["text"].lower().split()) for x in splits["dev"] + splits["test"]]
    assert len(texts) == len(set(texts))
    assert checks.near_duplicates(splits["dev"], splits["test"]) == []


def test_items_are_self_consistent(built):
    splits, _, _ = built
    for x in splits["dev"] + splits["test"]:
        assert checks.check_item(x) == []
        assert x["bytes"] == len(x["text"].encode())
        assert set(x["tokens"]) == set(B.DSPM_MODELS)
        if x["workload"] == "W4":
            assert x["label"] == x["category_top"]
        for s in x["spans"]:
            if s.get("kind") in checks.CARD_KINDS:
                assert fakes.luhn_ok(x["text"][s["start"]:s["end"]])


def test_check_item_catches_label_errors(built):
    splits, _, _ = built
    pos = next(x for x in splits["test"] if x["sensitive"] and x["workload"] == "W1")
    bad = {**pos, "label": "NOT_SENSITIVE"}
    assert checks.check_item(bad)
    neg = next(x for x in splits["test"] if not x["sensitive"] and x["workload"] == "W3")
    leaked = {**neg, "text": neg["text"] + " " + fakes.github_pat(random.Random(1))}
    assert any("unlabeled" in e for e in checks.check_item(leaked))
    masked = next(x for x in splits["test"] if x["hard_case"] == "masked")
    value = fakes.ssn(random.Random(2))
    assert checks.check_item({**masked, "text": masked["text"] + " " + value}, [value])


def test_review_sample(built):
    splits, _, _ = built
    md = B.review_sample(splits["test"], n=100, seed=1)
    assert md.count("\n## test-") == 100
    assert "⟦" in md and "CC BY 4.0" in md


def test_policy_hashes_are_stable():
    a, b = load_policy(), load_policy()
    assert a.sha256 == b.sha256 and len(a.sha256) == 64
    assert a.text.startswith("You are a data loss prevention classifier")
    assert "4111 1111 1111 1111" in a.text
    # The committed policy must hold no real-format secret or card (scanners would flag
    # the repository); the category name CONFIDENTIAL is expected.
    hits = [(k, m) for line in a.text.splitlines() for k, m in findings(line) if k != "confidential_marking"]
    assert hits == []
