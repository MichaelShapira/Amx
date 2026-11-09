"""Public synthetic sources S1 (Nemotron-PII) and S2 (Gretel PII finance), design 24.3.

A source document becomes a Record with role-tagged spans. Three transforms produce
dataset components from it:

- positive(): the document as it is, with card spans rewritten to numbers that pass the
  Luhn check (the sources' card values often do not), so FINANCIAL matches the policy;
- sanitize(): every span with a policy role rewritten into non-identifying text (first
  names kept, last names cut to an initial, values replaced by neutral words); weak spans
  are kept because the policy does not treat them as sensitive;
- mask(): every value masked or truncated (at most four digits visible) with names kept,
  which is the "masked" hard negative.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from tools.dspm import fakes
from tools.dspm.scan import CONF_MARK, clean, masked_ok

NEMOTRON_REPO = "nvidia/Nemotron-PII"
NEMOTRON_REV = "b70ffaf5ff39e079776134c5bf4381f00a9fd1ed"
NEMOTRON_FILE = "data/test-00000-of-00001.parquet"
GRETEL_REPO = "gretelai/synthetic_pii_finance_multilingual"
GRETEL_REV = "7b844d16738527a04264f50214cb426a4cea0897"
GRETEL_FILE = "data/English_test-00000-of-00001.parquet"

CARD_LABELS = {"credit_debit_card", "credit_card_number"}
FULL_MASK_LABELS = {"street_address", "coordinate", "local_latlng", "race_ethnicity", "religious_belief",
                    "political_view", "sexuality", "biometric_identifier", "blood_type", "password",
                    "api_key"}


@dataclass
class Span:
    start: int
    end: int
    label: str
    role: str                    # category | name | personal_attr | weak
    category: str | None = None


@dataclass
class Record:
    uid: str
    source: str                  # S1 | S2
    doc_type: str
    fmt: str                     # structured | unstructured
    domain: str
    text: str
    spans: list[Span] = field(default_factory=list)


class SpanMap:
    def __init__(self, path: str | Path = "config/dspm/span_label_map.yaml"):
        raw = yaml.safe_load(Path(path).read_text())
        self.raw = raw
        self.maps = {}
        for src, m in raw.items():
            roles = {lab: ("category", cat) for lab, cat in m.get("category", {}).items()}
            roles.update({lab: ("name", None) for lab in m.get("name", [])})
            roles.update({lab: ("personal_attr", None) for lab in m.get("personal_attr", [])})
            roles.update({lab: ("weak", None) for lab in m.get("weak", [])})
            self.maps[src] = {"roles": roles, "requires_name": set(m.get("requires_name", [])),
                              "full_name_labels": set(m.get("full_name_labels", []))}

    def role(self, source: str, label: str) -> tuple[str, str | None]:
        return self.maps[source]["roles"].get(label, ("weak", None))

    def categories(self, source: str, text: str, spans: list[Span]) -> set[str]:
        """Policy categories of one record (the PERSONAL combination rule applied here)."""
        m = self.maps[source]
        labels = {s.label for s in spans}
        has_name = any(s.role == "name" for s in spans)
        full = ({"first_name", "last_name"} <= labels) or any(
            s.label in m["full_name_labels"] and len(text[s.start:s.end].split()) >= 2 for s in spans)
        cats = {s.category for s in spans if s.role == "category"
                and (s.label not in m["requires_name"] or has_name)}
        if full and any(s.role == "personal_attr" for s in spans):
            cats.add("PERSONAL")
        return cats

    def evidence(self, source: str, text: str, spans: list[Span]) -> list[dict]:
        """Spans that support the record's categories, in the dataset item format."""
        cats = self.categories(source, text, spans)
        out = []
        for s in spans:
            if s.role == "category" and s.category in cats:
                out.append({"start": s.start, "end": s.end, "category": s.category, "kind": s.label})
            elif s.role in ("name", "personal_attr") and "PERSONAL" in cats:
                out.append({"start": s.start, "end": s.end, "category": "PERSONAL", "kind": s.label})
        return out


SOURCE_KEY = {"S1": "nemotron-pii", "S2": "gretel-pii-finance"}


def split_of(uid: str, dev_share: float = 0.1) -> str:
    h = int.from_bytes(hashlib.sha256(uid.encode()).digest()[:4], "big") / 2**32
    return "dev" if h < dev_share else "test"


def _dup_line_ratio(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return 1 - len(set(lines)) / max(1, len(lines))


def degenerate(text: str) -> bool:
    """Generation loops in the sources ("C,C,C,...", repeated lines) compress far better
    than real text; normal prose and forms stay above 0.2."""
    raw = text.encode("utf-8")
    return _dup_line_ratio(text) > 0.3 or len(zlib.compress(raw)) / max(1, len(raw)) < 0.2


# Keywords that must appear shortly before a category span for the source label to be
# trusted (Macie's keyword-proximity idea). Gretel in particular tags policy numbers as
# SSNs and serial numbers as bank accounts; such records are not used as positives or
# masked negatives (sanitizing them is still safe).
CONTEXT_KEYWORDS = {
    "ssn": ["ssn", "social security", "social", "national insurance", "nino"],
    "national_id": ["national", "identity", "identification", "id", "aadhaar", "nric", "dni", "cpf",
                    "bsn", "pesel", "personnummer", "citizen"],
    "tax_id": ["tax", "tin", "ein", "itin", "vat", "pan", "taxpayer"],
    "passport_number": ["passport"],
    "driver_license_number": ["driver", "license", "licence", "dl"],
    "credit_debit_card": ["card", "visa", "mastercard", "amex", "credit", "debit", "pan"],
    "credit_card_number": ["card", "visa", "mastercard", "amex", "credit", "debit", "pan"],
    "cvv": ["cvv", "cvc", "security code", "csc", "cvv2"],
    "credit_card_security_code": ["cvv", "cvc", "security code", "csc", "cvv2"],
    "account_number": ["account", "acct", "acctid", "a/c"],
    "bban": ["account", "acct", "acctid", "bban", "bank"],
    "iban": ["iban", "account", "bank"],
    "medical_record_number": ["medical record", "mrn", "record", "patient", "chart"],
    "health_plan_beneficiary_number": ["health plan", "beneficiary", "member", "insurance", "policy", "plan"],
    "blood_type": ["blood"],
    "password": ["password", "passcode", "pwd", "pass"],
    "api_key": ["api", "key", "token"],
    "pin": ["pin"],
    "account_pin": ["pin"],
}


def _keyword_ok(text: str, span: Span, window: int = 80) -> bool:
    words = CONTEXT_KEYWORDS.get(span.label)
    if not words:
        return True
    ctx = text[max(0, span.start - window):span.start].lower().replace("_", " ")
    return any(re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", ctx) for w in words)


def trusted(rec: Record) -> bool:
    """Every category span of the record has its keyword nearby."""
    return all(_keyword_ok(rec.text, s) for s in rec.spans if s.role == "category")


def _make_spans(smap: SpanMap, source: str, text: str, raw: list[dict]) -> list[Span] | None:
    spans = []
    for s in sorted(raw, key=lambda s: (s["start"], s["end"])):
        start, end = int(s["start"]), int(s["end"])
        if not (0 <= start < end <= len(text)):
            return None
        if "text" in s and text[start:end] != s["text"]:
            return None
        role, cat = smap.role(SOURCE_KEY[source], s["label"])
        spans.append(Span(start, end, s["label"], role, cat))
    for a, b in zip(spans, spans[1:], strict=False):
        if b.start < a.end:
            return None
    return spans


def load_nemotron(path: Path, smap: SpanMap, limit: int | None = None, seed: int = 0) -> list[Record]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path, columns=["uid", "domain", "document_type", "document_format",
                                        "locale", "text", "spans"]).to_pylist()
    rows.sort(key=lambda r: r["uid"])
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows:
        if limit is not None and len(out) >= limit:
            break
        text = r["text"]
        if CONF_MARK.search(text) or degenerate(text):
            continue
        spans = _make_spans(smap, "S1", text, ast.literal_eval(r["spans"]))
        if spans is None:
            continue
        out.append(Record(f"nemotron-{r['uid']}", "S1", r["document_type"], r["document_format"],
                          r["domain"], text, spans))
    return out


def load_gretel(path: Path, smap: SpanMap) -> list[Record]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path, columns=["document_type", "domain", "generated_text", "pii_spans",
                                        "__index_level_0__"]).to_pylist()
    out = []
    for r in rows:
        text = r["generated_text"]
        if len(text) < 200 or CONF_MARK.search(text) or degenerate(text):
            continue
        spans = _make_spans(smap, "S2", text, json.loads(r["pii_spans"]))
        if spans is None:
            continue
        out.append(Record(f"gretel-{r['__index_level_0__']}", "S2", r["document_type"],
                          "structured" if "\n:" in text or text.count(",") > 40 else "unstructured",
                          r["domain"], text, spans))
    out.sort(key=lambda r: r.uid)
    return out


# ---------------------------------------------------------------------------- rewriting

def _apply(rec: Record, repl: dict[int, str], extra_values: dict[str, str]) -> Record:
    """Replace span i's text with repl[i]; also replace other occurrences of the original
    values listed in extra_values (word-bounded), so a value repeated outside its span does
    not survive. Spans that were replaced become weak spans over the new text."""
    edits: list[tuple[int, int, str, int | None]] = []
    for i, s in enumerate(rec.spans):
        if i in repl:
            edits.append((s.start, s.end, repl[i], i))
    taken = [(s.start, s.end) for s in rec.spans]
    for orig, new in sorted(extra_values.items(), key=lambda kv: -len(kv[0])):
        if len(orig) < 3:
            continue
        for m in re.finditer(r"(?<!\w)" + re.escape(orig) + r"(?!\w)", rec.text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            edits.append((m.start(), m.end(), new, None))
            taken.append((m.start(), m.end()))
    edits.sort()
    text_parts, pos, shift_at = [], 0, []
    for start, end, new, _ in edits:
        text_parts.append(rec.text[pos:start])
        text_parts.append(new)
        shift_at.append((start, end, len(new) - (end - start)))
        pos = end
    text_parts.append(rec.text[pos:])
    new_text = "".join(text_parts)

    def moved(x: int) -> int:
        return x + sum(d for s, e, d in shift_at if e <= x)

    spans = []
    edit_by_span = {i: new for _, _, new, i in edits if i is not None}
    for i, s in enumerate(rec.spans):
        start = moved(s.start)
        if i in edit_by_span:
            spans.append(Span(start, start + len(edit_by_span[i]), s.label, "weak"))
        else:
            spans.append(replace(s, start=start, end=moved(s.end)))
    return replace(rec, text=new_text, spans=spans)


def _new_card_like(rng: random.Random, original: str) -> str:
    digits = re.sub(r"\D", "", original)
    if len(digits) == 15:
        num = fakes.card_number(rng, "amex")
    elif len(digits) == 16:
        num = fakes.card_number(rng, rng.choice(["visa", "mastercard", "discover"]))
    else:
        return fakes.format_card(fakes.card_number(rng, "visa"), "space")
    it = iter(num)
    return "".join(next(it) if c.isdigit() else c for c in original)


def positive(rec: Record, rng: random.Random) -> Record:
    repl, extra = {}, {}
    for i, s in enumerate(rec.spans):
        if s.label in CARD_LABELS:
            orig = rec.text[s.start:s.end]
            if orig not in extra:
                extra[orig] = _new_card_like(rng, orig)
            repl[i] = extra[orig]
    return _apply(rec, repl, extra) if repl else rec


_NEUTRAL = {
    "date_of_birth": ["on file", "not provided"],
    "street_address": ["the address on file", "the registered office"],
    "phone_number": ["the main office line", "not provided"],
    "email": ["contact@example.com", "support@example.org"],
    "coordinate": ["the site location", "nearby"],
    "local_latlng": ["the branch location"],
    "GOV_ID": ["on file", "provided separately", "not provided"],
    "FINANCIAL": ["on file", "the account on record", "not provided"],
    "HEALTH": ["not recorded", "on file"],
    "CREDENTIALS": ["provided separately", "set by the user"],
}


def sanitize(rec: Record, rng: random.Random, smap: SpanMap) -> Record | None:
    """Rewrite every span with a policy role into non-identifying text."""
    repl, extra = {}, {}
    for i, s in enumerate(rec.spans):
        orig = rec.text[s.start:s.end]
        if s.role == "name":
            if s.label == "first_name":
                continue
            parts = orig.split()
            if s.label == "last_name":
                new = orig[:1].upper() + "."
            elif len(parts) >= 2:
                new = parts[0] + " " + parts[-1][:1].upper() + "."
            else:
                continue
        elif s.role == "personal_attr":
            new = rng.choice(_NEUTRAL.get(s.label, ["not recorded"]))
        elif s.role == "category":
            new = rng.choice(_NEUTRAL.get(s.category, ["on file"]))
        else:
            continue
        repl[i] = new
        extra.setdefault(orig, new)
    out = _apply(rec, repl, extra)
    if smap.categories(SOURCE_KEY[rec.source], out.text, out.spans) or not clean(out.text):
        return None
    return out


def _mask_value(rng: random.Random, s: Span, orig: str) -> str:
    if s.label in CARD_LABELS:
        return fakes.mask_card(rng, orig)
    if s.label == "ssn":
        return fakes.mask_ssn(orig)
    if s.label in ("cvv", "credit_card_security_code", "pin", "account_pin"):
        return "***"
    if s.label == "date_of_birth":
        return rng.choice(["**/**/****", "XX/XX/XXXX", "[REDACTED]"])
    if s.label == "email":
        return fakes.mask_email(orig)
    if s.label == "phone_number":
        return fakes.mask_phone(orig)
    if s.label in FULL_MASK_LABELS:
        return fakes.mask_generic(rng)
    if sum(c.isdigit() for c in orig) >= 5:
        if s.category == "FINANCIAL":
            return fakes.mask_account(rng, orig)
        return "****" + fakes.last4(orig)
    return fakes.mask_generic(rng)


def mask(rec: Record, rng: random.Random, smap: SpanMap) -> tuple[Record, list[str]] | None:
    """Mask every value; names stay. Returns the record and the original values (for the
    build check that no unmasked value survives)."""
    if not smap.categories(SOURCE_KEY[rec.source], rec.text, rec.spans):
        return None
    repl, extra, originals = {}, {}, []
    for i, s in enumerate(rec.spans):
        if s.role not in ("category", "personal_attr"):
            continue
        orig = rec.text[s.start:s.end]
        new = extra.get(orig) or _mask_value(rng, s, orig)
        if not masked_ok(new):
            return None
        repl[i] = new
        extra.setdefault(orig, new)
        originals.append(orig)
    out = _apply(rec, repl, extra)
    if smap.categories(SOURCE_KEY[rec.source], out.text, out.spans) or not clean(out.text):
        return None
    if any(len(o) >= 4 and o in out.text for o in originals):
        return None
    return out, originals
