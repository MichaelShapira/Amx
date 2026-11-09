"""Fake values in real formats, check digits, masking and look-alike numbers.

Every value is random. Formats follow public specifications (Luhn for cards, ISO 13616
mod-97 for IBANs, the ABA routing checksum, the SSA area/group/serial ranges, vendor
token prefixes) so that models and pattern scanners see realistic values. This module
contains no literal secret in a real format; the only literals are the well-known
documentation placeholders, which the policy (rule 2) treats as not sensitive.
"""

from __future__ import annotations

import base64
import json
import random
import string
import uuid

ALNUM = string.ascii_letters + string.digits
B32 = string.ascii_uppercase + "234567"
B64 = string.ascii_letters + string.digits + "+/"
B64URL = string.ascii_letters + string.digits + "-_"

# Well-known test cards from payment processor documentation (policy rule 2).
DOC_TEST_CARDS = frozenset({
    "4111111111111111", "4242424242424242", "5555555555554444", "378282246310005",
    "6011111111111117", "4000056655665556", "5105105105105100", "4012888888881881",
})
DOC_AWS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
DOC_AWS_SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
DOC_PLACEHOLDERS = ("YOUR_API_KEY", "<password>", "<your-token>", "REPLACE_ME", "${API_KEY}",
                    "<YOUR_SECRET_HERE>", "xxxxxxxx")


def _rand(rng: random.Random, alphabet: str, n: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


# ---------------------------------------------------------------------------- check digits

def luhn_ok(number: str) -> bool:
    d = [int(c) for c in number if c.isdigit()]
    if len(d) < 12:
        return False
    total = 0
    for i, x in enumerate(reversed(d)):
        if i % 2:
            x *= 2
            if x > 9:
                x -= 9
        total += x
    return total % 10 == 0


def luhn_complete(partial: str) -> str:
    for c in string.digits:
        if luhn_ok(partial + c):
            return partial + c
    raise AssertionError("unreachable")


def luhn_break(number: str) -> str:
    """Change the last digit so the number fails the Luhn check."""
    head, last = number[:-1], int(number[-1])
    for step in range(1, 10):
        cand = head + str((last + step) % 10)
        if not luhn_ok(cand):
            return cand
    raise AssertionError("unreachable")


def iban_ok(iban: str) -> bool:
    s = iban.replace(" ", "").upper()
    if len(s) < 15 or not s[:2].isalpha() or not s[2:4].isdigit():
        return False
    moved = s[4:] + s[:4]
    num = "".join(str(int(c, 36)) for c in moved)
    return int(num) % 97 == 1


def _iban_check(country: str, bban: str) -> str:
    num = "".join(str(int(c, 36)) for c in bban + country + "00")
    return f"{98 - int(num) % 97:02d}"


def aba_ok(routing: str) -> bool:
    if len(routing) != 9 or not routing.isdigit():
        return False
    d = [int(c) for c in routing]
    return (3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + d[2] + d[5] + d[8]) % 10 == 0


# ---------------------------------------------------------------------------- financial

def card_number(rng: random.Random, brand: str | None = None) -> str:
    brand = brand or rng.choice(["visa", "visa", "mastercard", "mastercard", "amex", "discover"])
    while True:
        if brand == "visa":
            n = luhn_complete("4" + _digits(rng, 14))
        elif brand == "mastercard":
            n = luhn_complete(f"5{rng.randint(1, 5)}" + _digits(rng, 13))
        elif brand == "amex":
            n = luhn_complete(rng.choice(["34", "37"]) + _digits(rng, 12))
        else:
            n = luhn_complete("6011" + _digits(rng, 11))
        if n not in DOC_TEST_CARDS:
            return n


def format_card(number: str, style: str = "space") -> str:
    if len(number) == 15:
        parts = [number[:4], number[4:10], number[10:]]
    else:
        parts = [number[i:i + 4] for i in range(0, len(number), 4)]
    return {"space": " ".join(parts), "dash": "-".join(parts)}.get(style, number)


def card(rng: random.Random) -> str:
    return format_card(card_number(rng), rng.choice(["space", "space", "dash", "plain"]))


def cvv(rng: random.Random, amex: bool = False) -> str:
    return _digits(rng, 4 if amex else 3)


def expiry(rng: random.Random) -> str:
    return f"{rng.randint(1, 12):02d}/{rng.randint(27, 32)}"


def iban(rng: random.Random, country: str | None = None) -> str:
    country = country or rng.choice(["GB", "DE", "FR", "NL", "ES"])
    if country == "GB":
        bban = _rand(rng, string.ascii_uppercase, 4) + _digits(rng, 14)
    elif country == "DE":
        bban = _digits(rng, 18)
    elif country == "FR":
        bban = _digits(rng, 10) + _rand(rng, string.ascii_uppercase + string.digits, 11) + _digits(rng, 2)
    elif country == "NL":
        bban = _rand(rng, string.ascii_uppercase, 4) + _digits(rng, 10)
    else:
        bban = _digits(rng, 20)
    raw = country + _iban_check(country, bban) + bban
    return " ".join(raw[i:i + 4] for i in range(0, len(raw), 4))


def aba_routing(rng: random.Random) -> str:
    while True:
        head = f"{rng.choice([*range(1, 13), *range(21, 33)]):02d}" + _digits(rng, 6)
        for c in string.digits:
            if aba_ok(head + c):
                return head + c


def bank_account(rng: random.Random) -> str:
    return str(rng.randint(1, 9)) + _digits(rng, rng.randint(8, 11))


# ---------------------------------------------------------------------------- government IDs

def ssn(rng: random.Random, dashes: bool = True) -> str:
    while True:
        area = rng.randint(1, 899)
        if area != 666:
            break
    s = f"{area:03d}-{rng.randint(1, 99):02d}-{rng.randint(1, 9999):04d}"
    return s if dashes else s.replace("-", "")


def ein(rng: random.Random) -> str:
    return f"{rng.choice([12, 20, 27, 33, 45, 46, 52, 61, 74, 83, 88, 94])}-{_digits(rng, 7)}"


def itin(rng: random.Random) -> str:
    return f"9{_digits(rng, 2)}-{rng.randint(70, 88)}-{_digits(rng, 4)}"


def passport(rng: random.Random) -> str:
    return rng.choice([lambda: rng.choice("ACEFGH") + _digits(rng, 8), lambda: _digits(rng, 9),
                       lambda: _rand(rng, string.ascii_uppercase, 2) + _digits(rng, 7)])()


def drivers_license(rng: random.Random) -> str:
    return rng.choice([
        lambda: rng.choice(string.ascii_uppercase) + _digits(rng, 7),                 # CA
        lambda: f"{_digits(rng, 3)} {_digits(rng, 3)} {_digits(rng, 3)}",            # NY
        lambda: rng.choice(string.ascii_uppercase) + "-" + "-".join(
            [_digits(rng, 3), _digits(rng, 3), _digits(rng, 2), _digits(rng, 3)]),    # FL
        lambda: _digits(rng, 8),                                                      # TX
    ])()


def uk_nino(rng: random.Random) -> str:
    first = "ABCEGHJKLMNPRSTWXYZ"
    second = "ABCEGHJKLMNPRSTWXYZ".replace("O", "")
    return (rng.choice(first) + rng.choice(second) + " " + " ".join(_digits(rng, 2) for _ in range(3))
            + " " + rng.choice("ABCD"))


# ---------------------------------------------------------------------------- credentials

def aws_key_id(rng: random.Random) -> str:
    return rng.choice(["AKIA", "AKIA", "ASIA"]) + _rand(rng, B32, 16)


def aws_secret(rng: random.Random) -> str:
    return _rand(rng, B64, 40)


def github_pat(rng: random.Random) -> str:
    return rng.choice(["ghp_", "gho_", "ghs_"]) + _rand(rng, ALNUM, 36)


def slack_token(rng: random.Random) -> str:
    return f"xoxb-{_digits(rng, 12)}-{_digits(rng, 13)}-{_rand(rng, ALNUM, 24)}"


def stripe_key(rng: random.Random) -> str:
    return rng.choice(["sk_live_", "rk_live_"]) + _rand(rng, ALNUM, rng.choice([24, 32]))


def google_api_key(rng: random.Random) -> str:
    return "AIza" + _rand(rng, B64URL, 35)


def openai_key(rng: random.Random) -> str:
    return "sk-proj-" + _rand(rng, B64URL, 48)


def npm_token(rng: random.Random) -> str:
    return "npm_" + _rand(rng, ALNUM, 36)


def _b64url(obj: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj, separators=(",", ":")).encode()).decode().rstrip("=")


def jwt(rng: random.Random, subject: str = "user") -> str:
    iat = rng.randint(1_780_000_000, 1_800_000_000)
    head = _b64url({"alg": rng.choice(["HS256", "RS256"]), "typ": "JWT"})
    body = _b64url({"sub": f"{subject}-{_digits(rng, 6)}", "iat": iat, "exp": iat + 3600,
                    "scope": rng.choice(["read write", "admin", "payments:write"])})
    return f"{head}.{body}.{_rand(rng, B64URL, 43)}"


def pem_private_key(rng: random.Random) -> str:
    kind = rng.choice(["RSA PRIVATE KEY", "PRIVATE KEY", "EC PRIVATE KEY", "OPENSSH PRIVATE KEY"])
    lead = {"RSA PRIVATE KEY": "MIIEow", "PRIVATE KEY": "MIIEvQ", "EC PRIVATE KEY": "MHcCAQEE",
            "OPENSSH PRIVATE KEY": "b3BlbnNzaC1rZXktdjEAAAAA"}[kind]
    n_lines = rng.randint(5, 9)
    body = lead + _rand(rng, B64, 64 * n_lines - len(lead))
    lines = [body[i:i + 64] for i in range(0, len(body), 64)]
    return f"-----BEGIN {kind}-----\n" + "\n".join(lines) + f"\n-----END {kind}-----"


def password(rng: random.Random) -> str:
    style = rng.random()
    if style < 0.4:
        word = rng.choice(["Summer", "Falcon", "Orchid", "Rainbow", "Quartz", "Maple", "Harbor",
                           "Tiger", "Velvet", "Cobalt", "Juniper", "Atlas"])
        return f"{word}{rng.randint(1970, 2031)}{rng.choice('!@#$%&*?')}"
    n = rng.randint(10, 16)
    pool = string.ascii_letters + string.digits + "!@#$%^&*-_=+?"
    while True:
        p = _rand(rng, pool, n)
        if (any(c.islower() for c in p) and any(c.isupper() for c in p)
                and any(c.isdigit() for c in p) and any(c in "!@#$%^&*-_=+?" for c in p)):
            return p


def session_cookie(rng: random.Random) -> str:
    return rng.choice([
        lambda: "sessionid=" + _rand(rng, string.ascii_lowercase + string.digits, 32),
        lambda: "JSESSIONID=" + _rand(rng, "0123456789ABCDEF", 32),
        lambda: "connect.sid=s%3A" + _rand(rng, B64URL, 32) + "." + _rand(rng, B64URL, 43),
    ])()


def db_uri(rng: random.Random, user: str, host: str) -> str:
    pw = password(rng).replace("@", "#").replace("/", "-").replace(":", "!")
    return rng.choice([
        lambda: f"postgresql://{user}:{pw}@{host}:5432/{rng.choice(['orders', 'billing', 'crm', 'app'])}",
        lambda: f"mysql://{user}:{pw}@{host}:3306/{rng.choice(['shop', 'payroll', 'inventory'])}",
        lambda: f"mongodb+srv://{user}:{pw}@{host}/{rng.choice(['events', 'profiles'])}?retryWrites=true",
        lambda: f"redis://default:{pw}@{host}:6379/0",
    ])()


# ---------------------------------------------------------------------------- masking

def last4(value: str) -> str:
    d = [c for c in value if c.isdigit()]
    return "".join(d[-4:]) if len(d) >= 4 else "0000"


def mask_ssn(value: str) -> str:
    return "***-**-" + last4(value)


def mask_card(rng: random.Random, value: str) -> str:
    l4 = last4(value)
    return rng.choice([f"**** **** **** {l4}", f"XXXX-XXXX-XXXX-{l4}", f"card ending in {l4}",
                       f"************{l4}"])


def mask_account(rng: random.Random, value: str) -> str:
    return rng.choice(["****" + last4(value), "acct ****" + last4(value), "x" * 6 + last4(value)])


def mask_email(value: str) -> str:
    local, _, domain = value.partition("@")
    tld = domain.rsplit(".", 1)[-1] if "." in domain else "com"
    return f"{local[:1] or 'x'}***@*****.{tld}"


def mask_phone(value: str) -> str:
    return "(***) ***-" + last4(value)


def mask_secret(rng: random.Random, value: str) -> str:
    return rng.choice(["[REDACTED]", "********", value[:4] + "*" * 12 if len(value) > 8 else "****"])


def mask_generic(rng: random.Random) -> str:
    return rng.choice(["[REDACTED]", "[REMOVED]", "***", "<redacted>"])


# ---------------------------------------------------------------------------- look-alikes

def order_number16(rng: random.Random) -> str:
    """A 16-digit order or reference number that fails the Luhn check."""
    n = luhn_break(str(rng.randint(1, 9)) + _digits(rng, 15))
    return format_card(n, rng.choice(["space", "plain", "dash"]))


def tracking_number(rng: random.Random) -> str:
    return rng.choice([
        lambda: "1Z" + _rand(rng, string.ascii_uppercase + string.digits, 6) + _digits(rng, 10),
        lambda: _digits(rng, 22),
        lambda: _digits(rng, 12),
    ])()


def invoice9(rng: random.Random) -> str:
    return str(rng.randint(1, 9)) + _digits(rng, 8)


def uuid4(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def sha256_hex(rng: random.Random) -> str:
    return _rand(rng, "0123456789abcdef", 64)


def git_sha(rng: random.Random) -> str:
    return _rand(rng, "0123456789abcdef", 40)


def ticket(rng: random.Random) -> str:
    return rng.choice(["INC", "REQ", "CHG", "TKT"]) + _digits(rng, 7)


def private_ip(rng: random.Random) -> str:
    return f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def geo(rng: random.Random) -> str:
    return f"{rng.uniform(-50, 60):.5f}, {rng.uniform(-120, 140):.5f}"
