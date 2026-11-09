"""Seeded template generator for source S3 (design 24.3).

Templates use <<key>> placeholders; <<key:CATEGORY>> marks the inserted value as a
sensitive span of that policy category. A trailing digit selects another entity of the
same kind (<<first_name2>> is a second person). Values come from tools.dspm.fakes and a
pinned Faker; nothing depends on the current date or on Python's hash randomization, so
the same seed gives the same text.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta

from faker import Faker

from tools.dspm import fakes

PLACEHOLDER = re.compile(r"<<([a-z_]+?)(\d*)(?::([A-Z_]+))?>>")
CATEGORIES = ("GOV_ID", "PERSONAL", "FINANCIAL", "HEALTH", "CREDENTIALS", "CONFIDENTIAL")


def derive_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


_FAKERS: dict[str, Faker] = {}


def faker(locale: str) -> Faker:
    if locale not in _FAKERS:
        _FAKERS[locale] = Faker(locale)
    return _FAKERS[locale]


DIAGNOSES = ["type 2 diabetes", "major depressive disorder", "hypertension", "asthma",
             "generalized anxiety disorder", "rheumatoid arthritis", "Crohn's disease", "migraine",
             "atrial fibrillation", "hypothyroidism", "early-stage breast cancer", "COPD",
             "a herniated disc", "celiac disease", "ADHD", "chronic kidney disease, stage 3",
             "bipolar II disorder", "multiple sclerosis", "sleep apnea", "a torn ACL"]
MEDICATIONS = ["metformin 500 mg", "sertraline 50 mg", "lisinopril 10 mg", "an albuterol inhaler",
               "levothyroxine 75 mcg", "adalimumab", "sumatriptan", "apixaban 5 mg", "tamoxifen",
               "methotrexate", "lamotrigine 100 mg", "atorvastatin 20 mg", "prednisone 20 mg"]
PROCEDURES = ["an MRI of the lumbar spine", "a colonoscopy", "an echocardiogram", "physical therapy",
              "a knee arthroscopy", "a sleep study", "chemotherapy, cycle 2", "a CT scan"]
PRODUCTS = ["Atlas", "Beacon", "Cinder", "Drift", "Ember", "Fjord", "Glimmer", "Harbor", "Ion",
            "Juniper", "Keystone", "Lumen", "Mosaic", "Nimbus", "Orbit", "Pylon", "Quill", "Relay"]
TEAMS = ["platform", "payments", "growth", "data", "mobile", "security", "infra", "support",
         "finance", "legal", "people ops", "design"]
SERVICES = ["checkout", "ledger", "notifier", "search", "auth-gateway", "billing", "catalog",
            "reports", "ingest", "scheduler", "profile", "inventory"]
CONF_MARKERS = ["STRICTLY CONFIDENTIAL", "CONFIDENTIAL - INTERNAL USE ONLY",
                "Internal only - do not forward", "Company Confidential",
                "CONFIDENTIAL: board and executive staff only", "INTERNAL ONLY"]
EXEC_TITLES = ["Chief Executive Officer", "Chief Financial Officer", "Chief Technology Officer",
               "Chief Operating Officer", "General Counsel"]
BLOOD = ["A+", "A-", "B+", "O+", "O-", "AB+", "AB-"]


def _day(rng: random.Random, start: date, span_days: int) -> date:
    return start + timedelta(days=rng.randrange(span_days))


def _fmt_date(rng: random.Random, d: date) -> str:
    return rng.choice([d.isoformat(), d.strftime("%B %d, %Y").replace(" 0", " "), d.strftime("%m/%d/%Y"),
                       d.strftime("%d %b %Y")])


class Ctx:
    """Value cache for one generated record."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        locale = self.rng.choice(["en_US", "en_US", "en_US", "en_GB", "en_IN", "en_CA"])
        self.fk = faker(locale)
        self.fk.seed_instance(seed % (2**31))
        self.locale = locale
        self.cache: dict[str, str] = {}

    def get(self, key: str, idx: str = "") -> str:
        k = key + idx
        if k not in self.cache:
            fn = PROVIDERS.get(key)
            if fn is None:
                raise KeyError(f"unknown placeholder {key}")
            self.cache[k] = fn(self, idx)
        return self.cache[k]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())[:14] or "acme"


def _company(c: Ctx, i: str) -> str:
    name = c.fk.company().replace(",", "")
    return name


def _phone(c: Ctx, i: str) -> str:
    r = c.rng
    if c.locale == "en_GB":
        return f"07{r.randint(100, 999)} {r.randint(100000, 999999)}"
    if c.locale == "en_IN":
        return f"+91 {r.randint(70000, 99999)} {r.randint(10000, 99999)}"
    return r.choice([f"({r.randint(201, 989)}) {r.randint(200, 989)}-{r.randint(1000, 9999)}",
                     f"{r.randint(201, 989)}-{r.randint(200, 989)}-{r.randint(1000, 9999)}",
                     f"+1 {r.randint(201, 989)} {r.randint(200, 989)} {r.randint(1000, 9999)}"])


def _personal_email(c: Ctx, i: str) -> str:
    first, last = c.get("first_name", i).lower(), c.get("last_name", i).lower()
    first, last = _slug(first), _slug(last)
    domain = c.rng.choice(["gmail.com", "yahoo.com", "outlook.com", "icloud.com", "proton.me", "hotmail.co.uk"])
    return c.rng.choice([f"{first}.{last}{c.rng.randint(1, 99)}@{domain}", f"{first[:1]}{last}@{domain}",
                         f"{first}{c.rng.randint(1970, 2005)}@{domain}"])


def _home_address(c: Ctx, i: str) -> str:
    return c.fk.address().replace("\n", ", ")


def _office_address(c: Ctx, i: str) -> str:
    r = c.rng
    return f"{r.randint(1, 999)} {r.choice(['Market', 'Main', 'Harbor', 'Innovation', 'Commerce'])} " \
           f"{r.choice(['Street', 'Way', 'Plaza', 'Boulevard'])}, {c.get('city', i)}"


def _cf(fn: Callable[[random.Random], str]) -> Callable[[Ctx, str], str]:
    return lambda c, i: fn(c.rng)


def _deal(c: Ctx, i: str) -> str:
    r = c.rng
    return r.choice([
        f"we sign the acquisition of {c.get('company', '2')} for ${r.randint(40, 900)}M on {c.get('date', '9')}",
        f"the board approved a merger with {c.get('company', '2')}; the announcement is planned for {c.get('date', '9')}",
        f"we are in exclusive talks to buy {c.get('company', '2')} at {r.randint(4, 14)}x revenue",
    ])


def _unreleased(c: Ctx, i: str) -> str:
    r = c.rng
    q = r.randint(1, 4)
    return r.choice([
        f"Q{q} revenue came in at ${r.randint(20, 400)}.{r.randint(0, 9)}M, {r.randint(3, 19)}% below guidance; "
        f"results are not public until {c.get('date', '9')}",
        f"we will cut the {c.get('team', i)} team by {r.randint(8, 30)}% next quarter; not announced yet",
        f"unreleased Q{q} numbers: gross margin {r.randint(30, 70)}%, net loss ${r.randint(2, 60)}M",
    ])


def _roadmap(c: Ctx, i: str) -> str:
    r = c.rng
    return r.choice([
        f"Project {c.get('codename', i)} launches in Q{r.randint(1, 4)} {r.randint(2027, 2028)} and replaces "
        f"{c.get('product', i)}; pricing goes up {r.randint(10, 40)}%",
        f"we are exiting the {r.choice(['EMEA', 'LATAM', 'APAC'])} reseller channel by {c.get('date', '9')}",
        f"next year's roadmap: {c.get('product', i)} 3.0, a {c.get('product', '2')} acquisition and a "
        f"price change for enterprise tiers",
    ])


def _customer_list(c: Ctx, i: str) -> str:
    r = c.rng
    rows = [f"{faker('en_US').company()} - ARR ${r.randint(50, 2400)}K - renewal {_fmt_date(r, _day(r, date(2027, 1, 1), 365))}"
            for _ in range(r.randint(3, 5))]
    return "; ".join(rows)


PROVIDERS: dict[str, Callable[[Ctx, str], str]] = {
    # people
    "first_name": lambda c, i: c.fk.first_name(),
    "last_name": lambda c, i: c.fk.last_name(),
    "full_name": lambda c, i: f"{c.get('first_name', i)} {c.get('last_name', i)}",
    "dob": lambda c, i: _fmt_date(c.rng, _day(c.rng, date(1948, 1, 1), 365 * 58)),
    "home_address": _home_address,
    "phone": _phone,
    "personal_email": _personal_email,
    "geo": _cf(fakes.geo),
    "pronoun": lambda c, i: "they",
    # government IDs
    "ssn": _cf(fakes.ssn),
    "ssn_nodash": lambda c, i: fakes.ssn(c.rng, dashes=False),
    "ein": _cf(fakes.ein),
    "itin": _cf(fakes.itin),
    "passport": _cf(fakes.passport),
    "drivers_license": _cf(fakes.drivers_license),
    "nino": _cf(fakes.uk_nino),
    # financial
    "card": _cf(fakes.card),
    "card_tail": lambda c, i: fakes.last4(c.get("card", i)),
    "cvv": lambda c, i: fakes.cvv(c.rng),
    "expiry": _cf(fakes.expiry),
    "iban": lambda c, i: fakes.iban(c.rng),
    "routing": _cf(fakes.aba_routing),
    "bank_account": _cf(fakes.bank_account),
    "amount": lambda c, i: f"${c.rng.randint(40, 98000):,}.{c.rng.randint(0, 99):02d}",
    # health
    "diagnosis": lambda c, i: c.rng.choice(DIAGNOSES),
    "medication": lambda c, i: c.rng.choice(MEDICATIONS),
    "procedure": lambda c, i: c.rng.choice(PROCEDURES),
    "mrn": lambda c, i: c.rng.choice(["MRN-", "", "MR"]) + str(c.rng.randint(10_000_000, 99_999_999)),
    "insurance_id": lambda c, i: "".join(c.rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(3))
                                 + str(c.rng.randint(100_000_000, 999_999_999)),
    "blood_type": lambda c, i: c.rng.choice(BLOOD),
    # credentials
    "aws_key_id": _cf(fakes.aws_key_id),
    "aws_secret": _cf(fakes.aws_secret),
    "github_pat": _cf(fakes.github_pat),
    "slack_token": _cf(fakes.slack_token),
    "stripe_key": _cf(fakes.stripe_key),
    "google_api_key": _cf(fakes.google_api_key),
    "openai_key": _cf(fakes.openai_key),
    "npm_token": _cf(fakes.npm_token),
    "jwt": lambda c, i: fakes.jwt(c.rng),
    "pem_key": _cf(fakes.pem_private_key),
    "db_uri": lambda c, i: fakes.db_uri(c.rng, c.get("db_user", i), c.get("host", i)),
    "password": _cf(fakes.password),
    "session_cookie": _cf(fakes.session_cookie),
    "db_user": lambda c, i: c.rng.choice(["app", "svc_orders", "admin", "etl", "report_ro", "root"]),
    # confidential business content
    "conf_marker": lambda c, i: c.rng.choice(CONF_MARKERS),
    "deal": _deal,
    "unreleased": _unreleased,
    "roadmap": _roadmap,
    "customer_list": _customer_list,
    "codename": lambda c, i: c.rng.choice(["Falcon", "Nightjar", "Bluebird", "Cormorant", "Kestrel",
                                           "Osprey", "Starling", "Heron", "Tern", "Merlin"]),
    # benign
    "company": _company,
    "domain": lambda c, i: _slug(c.get("company", i)) + c.rng.choice([".com", ".io", ".co"]),
    "city": lambda c, i: c.fk.city(),
    "office_address": _office_address,
    "job": lambda c, i: c.fk.job().split(",")[0],
    "product": lambda c, i: c.rng.choice(PRODUCTS),
    "team": lambda c, i: c.rng.choice(TEAMS),
    "service": lambda c, i: c.rng.choice(SERVICES),
    "date": lambda c, i: _fmt_date(c.rng, _day(c.rng, date(2026, 1, 1), 730)),
    "weekday": lambda c, i: c.rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
    "time": lambda c, i: f"{c.rng.randint(8, 17)}:{c.rng.choice(['00', '15', '30', '45'])}",
    "iso_ts": lambda c, i: f"2026-{c.rng.randint(1, 12):02d}-{c.rng.randint(1, 28):02d}T"
                           f"{c.rng.randint(0, 23):02d}:{c.rng.randint(0, 59):02d}:{c.rng.randint(0, 59):02d}Z",
    "quarter": lambda c, i: f"Q{c.rng.randint(1, 4)}",
    "year": lambda c, i: str(c.rng.randint(2025, 2028)),
    "num": lambda c, i: str(c.rng.randint(2, 60)),
    "big_num": lambda c, i: f"{c.rng.randint(1200, 98000):,}",
    "percent": lambda c, i: f"{c.rng.randint(1, 95)}%",
    "ms": lambda c, i: str(c.rng.randint(3, 2400)),
    "status": lambda c, i: c.rng.choice(["200", "200", "200", "201", "204", "304", "400", "404", "500"]),
    "path": lambda c, i: c.rng.choice(["/api/v1/orders", "/api/v2/search", "/health", "/login",
                                       "/api/v1/cart/items", "/static/app.js", "/api/v1/reports"]),
    "order_ref": _cf(fakes.order_number16),
    "tracking": _cf(fakes.tracking_number),
    "invoice_no": _cf(fakes.invoice9),
    "ninedigit": _cf(fakes.invoice9),
    "uuid": _cf(fakes.uuid4),
    "sha": _cf(fakes.sha256_hex),
    "gitsha": _cf(fakes.git_sha),
    "ticket": _cf(fakes.ticket),
    "ip": _cf(fakes.private_ip),
    "username": lambda c, i: _slug(c.get("first_name", i)) + c.rng.choice(["", "_", "."]) + str(c.rng.randint(1, 999)),
    "work_alias": lambda c, i: c.rng.choice(["billing", "support", "team", "noreply", "hr", "it-help",
                                             "accounts", "info"]) + "@" + c.get("domain", i),
    "version": lambda c, i: f"{c.rng.randint(1, 9)}.{c.rng.randint(0, 30)}.{c.rng.randint(0, 9)}",
    "port": lambda c, i: str(c.rng.choice([80, 443, 5432, 6379, 8080, 8443, 9090, 3000])),
    "host": lambda c, i: f"{c.get('service', i)}-{c.rng.randint(1, 9)}.{c.rng.choice(['prod', 'stg', 'dev'])}.internal",
    "region": lambda c, i: c.rng.choice(["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-2"]),
    "env_name": lambda c, i: c.rng.choice(["production", "staging", "dev", "qa"]),
    "repo": lambda c, i: f"{_slug(c.get('company', i))}/{c.get('service', i)}",
    "exec_name": lambda c, i: f"{faker('en_US').first_name()} {faker('en_US').last_name()}",
    "exec_title": lambda c, i: c.rng.choice(EXEC_TITLES),
    "ticker": lambda c, i: "".join(c.rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(c.rng.randint(3, 4))),
    # masked
    "masked_ssn": lambda c, i: fakes.mask_ssn(c.get("ssn", i)),
    "masked_card": lambda c, i: fakes.mask_card(c.rng, c.get("card", i)),
    "masked_account": lambda c, i: fakes.mask_account(c.rng, c.get("bank_account", i)),
    "masked_email": lambda c, i: fakes.mask_email(c.get("personal_email", i)),
    "masked_phone": lambda c, i: fakes.mask_phone(c.get("phone", i)),
    "masked_dob": lambda c, i: c.rng.choice(["**/**/****", "XX/XX/19XX", "[REDACTED]"]),
    "masked_secret": lambda c, i: c.rng.choice(["********", "[REDACTED]", "****************"]),
    "redacted": lambda c, i: fakes.mask_generic(c.rng),
    # documentation and placeholder values (policy rule 2)
    "doc_aws_key_id": lambda c, i: fakes.DOC_AWS_KEY_ID,
    "doc_aws_secret": lambda c, i: fakes.DOC_AWS_SECRET,
    "doc_card": lambda c, i: c.rng.choice(["4111 1111 1111 1111", "4242 4242 4242 4242", "4111111111111111",
                                           "4242424242424242", "5555 5555 5555 4444"]),
    "placeholder": lambda c, i: c.rng.choice(fakes.DOC_PLACEHOLDERS),
}

# Faker calls inside exec_name and customer_list use the shared en_US instance; seed it from
# the record's rng first so they stay deterministic.
_UNSEEDED = {"exec_name", "customer_list"}


def render(template: str, ctx: Ctx) -> tuple[str, list[dict]]:
    parts: list[str] = []
    spans: list[dict] = []
    pos = 0
    length = 0
    for m in PLACEHOLDER.finditer(template):
        lit = template[pos:m.start()]
        parts.append(lit)
        length += len(lit)
        key, idx, cat = m.group(1), m.group(2), m.group(3)
        if key in _UNSEEDED and key + idx not in ctx.cache:
            faker("en_US").seed_instance(ctx.rng.getrandbits(31))
        val = ctx.get(key, idx)
        if cat:
            if cat not in CATEGORIES:
                raise ValueError(f"unknown category {cat} in template")
            spans.append({"start": length, "end": length + len(val), "category": cat, "kind": key})
        parts.append(val)
        length += len(val)
        pos = m.end()
    parts.append(template[pos:])
    return "".join(parts), spans


@dataclass(frozen=True)
class Family:
    name: str
    channel: str
    kind: str                       # pos | neg
    templates: tuple[str, ...]
    workloads: tuple[str, ...] = ("W1", "W3")
    category: str | None = None     # positives
    hard_case: str | None = None    # hard negatives and hard positives
    filler: bool = False            # benign records used to pad chunks and threads


@dataclass
class Component:
    text: str
    spans: list[dict]
    family: str
    channel: str
    source: str
    categories: list[str]
    hard_case: str | None = None
    uid: str | None = None
    template: int | None = None
    masked_originals: list[str] = field(default_factory=list)


def template_split(index: int) -> str:
    """Template phrasing split: templates 0, 4, 8, ... of a family are dev-only, the rest
    test-only. Small families give dev exactly one phrasing; large filler families give
    dev two, so dev threads and chunks do not repeat one sentence."""
    return "dev" if index % 4 == 0 else "test"


def template_categories(template: str) -> frozenset[str]:
    return frozenset(m.group(3) for m in PLACEHOLDER.finditer(template) if m.group(3))


def templates_for(family: Family, split: str, want: str | None = None) -> list[int]:
    return [i for i, t in enumerate(family.templates) if template_split(i) == split
            and (want is None or want in template_categories(t))]


def generate(family: Family, split: str, seed: int, want: str | None = None) -> Component:
    rng = random.Random(seed)
    choices = templates_for(family, split, want)
    if not choices:
        raise ValueError(f"family {family.name} has no {split} templates for {want}")
    ti = rng.choice(choices)
    text, spans = render(family.templates[ti], Ctx(seed))
    cats = sorted({s["category"] for s in spans})
    return Component(text, spans, family.name, family.channel, "S3", cats, family.hard_case, template=ti)
