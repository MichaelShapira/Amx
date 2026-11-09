"""W4 structured tables: a CSV header plus sample rows (design 24.1, CSA/Sentra's point that
column names and neighbouring columns carry the context a single value lacks)."""

from __future__ import annotations

import random

from tools.dspm.generator import Component, Ctx, derive_seed

# (header, provider key, category or None). Each table type has a dev variant (index 0 of
# the header choices) and test variants, so header phrasing follows the template split.
Col = tuple[tuple[str, ...], str, str | None]

TABLES: dict[str, dict] = {
    # positives
    "employees-ssn": {"cols": [(("employee_id", "emp_no", "staff_id"), "ticket", None),
                               (("full_name", "name", "employee_name"), "full_name", None),
                               (("ssn", "social_security_number", "ssn_full"), "ssn", "GOV_ID"),
                               (("department", "dept", "team"), "team", None),
                               (("hire_date", "start_date", "joined"), "date", None)]},
    "vendors-ein": {"cols": [(("vendor", "supplier", "payee"), "company", None),
                             (("contact", "contact_first_name", "rep"), "first_name", None),
                             (("ein", "tax_id", "federal_tax_id"), "ein", "GOV_ID"),
                             (("city", "location", "billing_city"), "city", None)]},
    "travelers-passport": {"cols": [(("traveler", "passenger", "guest_name"), "full_name", None),
                                    (("passport_number", "passport_no", "passport"), "passport", "GOV_ID"),
                                    (("departure", "travel_date", "flight_date"), "date", None),
                                    (("destination", "city", "arrival_city"), "city", None)]},
    "customers-contact": {"cols": [(("full_name", "customer_name", "name"), "full_name", "PERSONAL"),
                                   (("date_of_birth", "dob", "birth_date"), "dob", "PERSONAL"),
                                   (("personal_email", "email", "email_address"), "personal_email", "PERSONAL"),
                                   (("mobile", "phone", "cell"), "phone", "PERSONAL"),
                                   (("city", "home_city", "town"), "city", None)]},
    "members-address": {"cols": [(("first_name", "given_name", "fname"), "first_name", "PERSONAL"),
                                 (("last_name", "surname", "lname"), "last_name", "PERSONAL"),
                                 (("home_address", "address", "street_address"), "home_address", "PERSONAL"),
                                 (("member_since", "joined", "signup_date"), "date", None)]},
    "payments-card": {"cols": [(("customer", "cardholder", "name_on_card"), "full_name", None),
                               (("card_number", "pan", "cc_number"), "card", "FINANCIAL"),
                               (("expiry", "exp", "valid_thru"), "expiry", None),
                               (("amount", "charge", "total"), "amount", None)]},
    "payroll-bank": {"cols": [(("employee", "payee", "name"), "full_name", None),
                              (("routing_number", "aba", "routing"), "routing", "FINANCIAL"),
                              (("account_number", "acct", "bank_account"), "bank_account", "FINANCIAL"),
                              (("net_pay", "amount", "salary"), "amount", None)]},
    "suppliers-iban": {"cols": [(("supplier", "vendor", "beneficiary"), "company", None),
                                (("iban", "bank_iban", "iban_number"), "iban", "FINANCIAL"),
                                (("invoice", "invoice_no", "bill"), "invoice_no", None)]},
    "patients-dx": {"cols": [(("patient_name", "patient", "name"), "full_name", "HEALTH"),
                             (("mrn", "medical_record_number", "record_no"), "mrn", "HEALTH"),
                             (("diagnosis", "dx", "primary_diagnosis"), "diagnosis", "HEALTH"),
                             (("admitted", "admit_date", "visit_date"), "date", None)]},
    "claims-member": {"cols": [(("member_name", "insured", "patient"), "full_name", "HEALTH"),
                               (("member_id", "insurance_id", "policy_member_id"), "insurance_id", "HEALTH"),
                               (("procedure", "service", "treatment"), "procedure", "HEALTH"),
                               (("billed", "amount", "claim_amount"), "amount", None)]},
    "service-passwords": {"cols": [(("service", "system", "app"), "service", None),
                                   (("username", "login", "user"), "username", None),
                                   (("password", "pass", "pwd"), "password", "CREDENTIALS"),
                                   (("last_rotated", "updated", "changed_on"), "date", None)]},
    "integration-keys": {"cols": [(("integration", "provider", "service"), "service", None),
                                  (("api_key", "key", "secret_key"), "stripe_key", "CREDENTIALS"),
                                  (("owner_team", "owner", "team"), "team", None)]},
    "pipeline-confidential": {"marker": True,
                              "cols": [(("account", "customer", "company"), "company", None),
                                       (("deal_value", "arr", "contract_value"), "amount", "CONFIDENTIAL"),
                                       (("stage", "status", "pipeline_stage"), "stage", None),
                                       (("close_date", "expected_close", "renewal"), "date", None)]},
    "pricing-confidential": {"marker": True,
                             "cols": [(("sku", "product_code", "item"), "ticket", None),
                                      (("product", "name", "plan"), "product", None),
                                      (("floor_price", "min_price", "partner_floor"), "amount", "CONFIDENTIAL"),
                                      (("discount", "max_discount", "partner_discount"), "percent", None)]},
    # easy negatives
    "products": {"cols": [(("sku", "item_code", "product_id"), "ticket", None),
                          (("product", "name", "item"), "product", None),
                          (("price", "unit_price", "list_price"), "amount", None),
                          (("stock", "qty_on_hand", "inventory"), "big_num", None)]},
    "sales-region": {"cols": [(("region", "market", "territory"), "region", None),
                              (("quarter", "period", "qtr"), "quarter", None),
                              (("revenue", "sales", "bookings"), "amount", None),
                              (("growth", "yoy", "change"), "percent", None)]},
    "web-analytics": {"cols": [(("date", "day", "report_date"), "date", None),
                               (("page", "path", "url_path"), "path", None),
                               (("views", "pageviews", "hits"), "big_num", None),
                               (("bounce_rate", "bounce", "exit_rate"), "percent", None)]},
    "releases": {"cols": [(("service", "component", "app"), "service", None),
                          (("version", "release", "tag"), "version", None),
                          (("released", "date", "shipped_on"), "date", None),
                          (("owner", "team", "owning_team"), "team", None)]},
    "tickets": {"cols": [(("ticket", "case_id", "issue"), "ticket", None),
                         (("status", "state", "stage"), "stage", None),
                         (("team", "queue", "assignee_team"), "team", None),
                         (("opened", "created", "date"), "date", None)]},
    # hard negatives
    "masked-customers": {"hard": "masked",
                         "cols": [(("customer", "name", "account_holder"), "full_name", None),
                                  (("ssn", "ssn_masked", "social"), "masked_ssn", None),
                                  (("card", "card_masked", "payment_card"), "masked_card", None),
                                  (("dob", "birth_date", "date_of_birth"), "masked_dob", None)]},
    "orders-lookalike": {"hard": "look-alike-number",
                         "cols": [(("order_id", "order_number", "reference"), "order_ref", None),
                                  (("tracking", "tracking_number", "shipment"), "tracking", None),
                                  (("amount", "total", "order_total"), "amount", None),
                                  (("status", "state", "order_status"), "stage", None)]},
    "first-names": {"hard": "first-name-only",
                    "cols": [(("first_name", "name", "attendee"), "first_name", None),
                             (("team", "group", "squad"), "team", None),
                             (("tshirt", "shirt_size", "size"), "size", None),
                             (("dietary", "lunch", "meal"), "meal", None),
                             (("office", "site", "location"), "city", None),
                             (("registered", "signup_date", "date"), "date", None)]},
    "companies-public": {"hard": "public-business",
                         "cols": [(("company", "name", "issuer"), "company", None),
                                  (("headquarters", "hq_address", "hq"), "office_address", None),
                                  (("ceo", "chief_executive", "leader"), "exec_name", None),
                                  (("ticker", "symbol", "listing"), "ticker", None)]},
    "qa-test-cards": {"hard": "documentation-value",
                      "cols": [(("test_case", "case", "scenario"), "ticket", None),
                               (("card_number", "test_card", "pan"), "doc_card", None),
                               (("expected", "expected_result", "outcome"), "outcome", None)]},
    "refs-no-keyword": {"hard": "number-without-keyword",
                        "cols": [(("ref", "reference", "code"), "ninedigit", None),
                                 (("item", "product", "description"), "product", None),
                                 (("qty", "quantity", "units"), "num", None),
                                 (("warehouse", "site", "location"), "city", None),
                                 (("received", "date", "logged_on"), "date", None)]},
    "secret-refs": {"hard": "indirect-reference",
                    "cols": [(("service", "app", "system"), "service", None),
                             (("secret_ref", "credential_source", "secret_path"), "secret_ref", None),
                             (("owner", "team", "owning_team"), "team", None)]},
}

EXTRA = {
    "stage": lambda c, i: c.rng.choice(["open", "won", "lost", "negotiation", "closed", "pending"]),
    "size": lambda c, i: c.rng.choice(["S", "M", "L", "XL"]),
    "meal": lambda c, i: c.rng.choice(["vegetarian", "vegan", "none", "gluten-free"]),
    "outcome": lambda c, i: c.rng.choice(["approved", "declined", "requires_3ds", "insufficient_funds"]),
    "secret_ref": lambda c, i: c.rng.choice([
        f"vault:secret/data/{c.get('service', i)}/db", "${DB_PASSWORD}",
        f"secretsmanager:{c.get('service', i)}/api", f"ssm:/{c.get('service', i)}/token"]),
}


def _value(ctx: Ctx, key: str, idx: str) -> str:
    if key in EXTRA:
        k = key + idx
        if k not in ctx.cache:
            ctx.cache[k] = EXTRA[key](ctx, idx)
        return ctx.cache[k]
    return ctx.get(key, idx)


def _cell(v: str) -> tuple[str, int]:
    """CSV-quote a cell; returns the cell text and the offset of the value inside it."""
    if any(ch in v for ch in ',"\n'):
        return '"' + v.replace('"', '""') + '"', 1
    return v, 0


def positive_types() -> list[str]:
    return [k for k, t in TABLES.items() if any(c[2] for c in t["cols"])]


def easy_negative_types() -> list[str]:
    return [k for k, t in TABLES.items() if not any(c[2] for c in t["cols"]) and "hard" not in t]


def hard_types() -> dict[str, str]:
    return {k: t["hard"] for k, t in TABLES.items() if "hard" in t}


def categories_of(kind: str) -> set[str]:
    return {c[2] for c in TABLES[kind]["cols"] if c[2]}


def generate(kind: str, split: str, seed: int, rows: int = 15) -> Component:
    spec = TABLES[kind]
    ctx = Ctx(seed)
    rng = random.Random(derive_seed(seed, "hdr"))
    variant = 0 if split == "dev" else rng.choice([1, 2])
    parts: list[str] = []
    spans: list[dict] = []
    length = 0

    def emit(s: str) -> None:
        nonlocal length
        parts.append(s)
        length += len(s)

    if spec.get("marker"):
        marker = ctx.get("conf_marker")
        emit("# ")
        spans.append({"start": length, "end": length + len(marker), "category": "CONFIDENTIAL",
                      "kind": "conf_marker"})
        emit(marker + "\n")
    elif rng.random() < 0.3:
        emit(f"# export {ctx.get('date')}\n")
    emit(",".join(c[0][variant] for c in spec["cols"]) + "\n")
    for r in range(rows):
        for j, (_, key, cat) in enumerate(spec["cols"]):
            v = _value(ctx, key, str(r))
            cell, off = _cell(v)
            if cat:
                spans.append({"start": length + off, "end": length + off + len(v), "category": cat, "kind": key})
            emit(cell + ("," if j < len(spec["cols"]) - 1 else "\n"))
    text = "".join(parts).rstrip("\n")
    cats = sorted({s["category"] for s in spans})
    return Component(text, spans, f"table/{kind}", "table", "S3", cats, spec.get("hard"), template=variant)
