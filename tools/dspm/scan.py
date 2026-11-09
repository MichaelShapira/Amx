"""Pattern scanner for values that the policy treats as sensitive.

Used two ways: while generating, to reject a negative (or a filler around a positive)
that accidentally contains a sensitive value; and in the build checks, as an independent
second opinion on every label. It only covers patterned values. Combination rules such as
PERSONAL (full name plus a personal attribute) are checked from span metadata instead.
"""

from __future__ import annotations

import re

from tools.dspm.fakes import DOC_PLACEHOLDERS, DOC_TEST_CARDS, iban_ok, luhn_ok

SSN = re.compile(r"(?<![\w-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\w-])")
CARD = re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")
TOKENS = {
    "aws_key_id": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "github": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b"),
    "slack": re.compile(r"\bxox[abprs]-[0-9A-Za-z-]{10,}"),
    "stripe": re.compile(r"\b[rs]k_live_[0-9A-Za-z]{16,}"),
    "google": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"),
    "openai": re.compile(r"\bsk-proj-[A-Za-z0-9_\-]{20,}"),
    "npm": re.compile(r"\bnpm_[A-Za-z0-9]{36}\b"),
    "pem": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{16,}"),
}
URI_PASSWORD = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:([^\s@/]+)@")
ASSIGNMENT = re.compile(
    r"(?i)(?:(?<![a-z])(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?key|token)"
    r"(?:[_-][a-z0-9]+)*|\bpin\b)"
    r"[\"']?\s*[:=]\s*[\"']?([^\s'\",;}]{6,})")
CONF_MARK = re.compile(
    r"\bCONFIDENTIAL\b|(?i:\binternal[ -](?:use[ -])?only\b)|(?i:\bstrictly confidential\b)"
    r"|(?i:\bcompany confidential\b)|(?i:\bdo not (?:distribute|forward)\b)")


def _placeholder(value: str) -> bool:
    """Template variables, documentation values and masks are not secrets."""
    v = value.strip("'\"")
    return (v.startswith(("${", "$(", "%(", "<", "{{", "[")) or "EXAMPLE" in v or v in DOC_PLACEHOLDERS
            or set(v) <= set("x*X#-") or v.lower() in {"redacted", "removed"})


def _secret_like(value: str) -> bool:
    return any(c.isdigit() for c in value) and any(c.isalpha() for c in value)


def findings(text: str) -> list[tuple[str, str]]:
    """Return (kind, matched text) for every patterned sensitive value in text."""
    out: list[tuple[str, str]] = []
    for m in SSN.finditer(text):
        out.append(("ssn", m.group(0)))
    for m in CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn_ok(digits) and digits not in DOC_TEST_CARDS:
            out.append(("card", m.group(0)))
    for m in IBAN.finditer(text):
        if iban_ok(m.group(0)):
            out.append(("iban", m.group(0)))
    for kind, rx in TOKENS.items():
        for m in rx.finditer(text):
            if "EXAMPLE" not in m.group(0):
                out.append((kind, m.group(0)))
    for m in URI_PASSWORD.finditer(text):
        if not _placeholder(m.group(1)):
            out.append(("uri_password", m.group(0)))
    for m in ASSIGNMENT.finditer(text):
        if not _placeholder(m.group(1)) and _secret_like(m.group(1)):
            out.append(("assignment", m.group(0)))
    for m in CONF_MARK.finditer(text):
        out.append(("confidential_marking", m.group(0)))
    return out


def clean(text: str) -> bool:
    return not findings(text)


def masked_ok(value: str) -> bool:
    """A masked replacement may show at most four digits."""
    return sum(c.isdigit() for c in value) <= 4
