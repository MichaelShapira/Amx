"""DSPM/DLP policy files (design 24.2, Requirement 21.3).

The policy file is the single source of truth: it labels the dataset and, as the system
message, instructs the generative models. Both files are hashed; the hashes are stored in
the dataset manifest and with every result, so a result can always be traced to the
exact policy text it was produced with.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "config" / "dspm"
MAX_POLICY_CHARS = 4500        # about 900 tokens for the Qwen and Gemma tokenizers
MAX_SHORT_POLICY_CHARS = 650   # about 120 tokens for the T5 tokenizer


@dataclass(frozen=True)
class Policy:
    version: str
    text: str
    short_text: str
    sha256: str
    short_sha256: str

    def hashes(self) -> dict[str, str]:
        return {"version": self.version, "policy_sha256": self.sha256, "policy_short_sha256": self.short_sha256}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_policy(version: str = "v1", directory: Path = POLICY_DIR) -> Policy:
    full = (directory / f"policy_{version}.md").read_bytes()
    short = (directory / f"policy_{version}_short.md").read_bytes()
    text, short_text = full.decode("utf-8").strip(), short.decode("utf-8").strip()
    if len(text) > MAX_POLICY_CHARS:
        raise ValueError(f"policy_{version}.md is {len(text)} chars; the limit is {MAX_POLICY_CHARS}")
    if len(short_text) > MAX_SHORT_POLICY_CHARS:
        raise ValueError(f"policy_{version}_short.md is {len(short_text)} chars; the limit is {MAX_SHORT_POLICY_CHARS}")
    return Policy(version, text, short_text, _sha(full), _sha(short))


def policy_hashes(version: str = "v1") -> dict[str, str]:
    return load_policy(version).hashes()


# ============================================================================ request shapes
# (design 24.4, Requirement 21.3-21.4). The same text reaches every engine for a model.

BINARY = {"W1": ("SENSITIVE", "NOT_SENSITIVE"), "W3": ("BLOCK", "ALLOW")}
CATEGORY_LABELS = ("GOV_ID", "PERSONAL", "FINANCIAL", "HEALTH", "CREDENTIALS", "CONFIDENTIAL", "NONE")
CATEGORY_WORKLOADS = ("W2", "W4")

INSTRUCTIONS = {
    "W1": "The CONTENT is a chunk of a file found while scanning a data store.",
    "W2": "The CONTENT is a chunk of a file found while scanning a data store.",
    "W3": "The CONTENT is a message or prompt that a user is about to send outside the company. "
          "Answer BLOCK if it contains sensitive data and ALLOW otherwise.",
    "W4": "The CONTENT is a table: a CSV header followed by sample rows.",
}

# M3 (roberta-large-mnli): one hypothesis per category (design 24.4).
HYPOTHESES = {
    "GOV_ID": "This text contains a government ID number such as a social security or passport number.",
    "PERSONAL": "This text contains a named person's birth date, home address, phone number or personal email.",
    "FINANCIAL": "This text contains a payment card number or a bank account number.",
    "HEALTH": "This text contains medical or health information about a named person.",
    "CREDENTIALS": "This text contains a password, API key, access token or private key.",
    "CONFIDENTIAL": "This text contains confidential or internal-only business information.",
}
MNLI_ENTAILMENT = 2          # roberta-large-mnli labels: CONTRADICTION, NEUTRAL, ENTAILMENT

# M5 (Flan-T5): options in natural words, mapped back to labels.
T5_BINARY = {"yes": True, "no": False}
T5_CATEGORIES = {"government id": "GOV_ID", "personal data": "PERSONAL", "financial": "FINANCIAL",
                 "health": "HEALTH", "credentials": "CREDENTIALS", "confidential business": "CONFIDENTIAL",
                 "none": "NONE"}

# D1 (openai/privacy-filter) span labels to policy categories. A person counts as PERSONAL
# only together with one of the personal attributes, as in the policy.
D1_DIRECT = {"account_number": "FINANCIAL", "secret": "CREDENTIALS"}
D1_PERSONAL_ATTRS = {"private_address", "private_email", "private_phone", "private_date"}
D2_GIVING_PII = 1            # roblox-pii-classifier-v2 label order
D2_THRESHOLD = 0.55          # model card recommendation for privacy_giving_pii


def answer_line(workload: str) -> str:
    if workload in CATEGORY_WORKLOADS:
        return "Answer with exactly one label from: " + ", ".join(CATEGORY_LABELS) + \
            ". Use the highest category in the priority order when several apply, and NONE if nothing is sensitive."
    pos, neg = BINARY[workload]
    return f"Answer with exactly one label: {pos} or {neg}."


def decoder_user_message(text: str, workload: str) -> str:
    return (f"{INSTRUCTIONS[workload]}\n\nCONTENT:\n<<<\n{text}\n>>>\n\n{answer_line(workload)}")


def t5_prompt(text: str, workload: str, policy: Policy) -> str:
    if workload in CATEGORY_WORKLOADS:
        q = "Which kind of sensitive data does the text contain?"
        opts = list(T5_CATEGORIES)
    else:
        q = "Does the text contain sensitive data?"
        opts = list(T5_BINARY)
    return f"{policy.short_text}\n\nText: {text}\n\n{q}\nOPTIONS:\n" + "\n".join(f"- {o}" for o in opts)


def mnli_inputs(text: str) -> list[str]:
    """Premise/hypothesis pairs as single strings in RoBERTa's pair format, so vLLM /classify
    and the thin server receive identical text (tests: test_mnli_joined_string_equals_pair_encoding)."""
    return [f"{text}</s></s>{HYPOTHESES[c]}" for c in HYPOTHESES]


# ============================================================================ decisions

def _clean(text: str | None) -> str:
    return (text or "").strip().split("\n")[0].strip().strip(".,:;\"'`*()[]").strip()


def decide_generative(slot: str, workload: str, text: str | None) -> str | None:
    """Strict parsing (design 24.4): anything that is not exactly one label is a format
    failure (None), which the scorer counts as positive (fail closed)."""
    if slot == "M5":
        word = _clean(text).lower()
        if workload in CATEGORY_WORKLOADS:
            return T5_CATEGORIES.get(word)
        if word in T5_BINARY:
            return BINARY[workload][0] if T5_BINARY[word] else BINARY[workload][1]
        return None
    from bench.dspm_scoring import parse_label

    return parse_label(text, workload)


def mnli_score(outputs: list[dict]) -> float | None:
    """Highest entailment probability over the category hypotheses."""
    vals = [o["probs"][MNLI_ENTAILMENT] for o in outputs if o.get("probs")]
    return max(vals) if len(vals) == len(HYPOTHESES) else None


def decide_mnli(workload: str, outputs: list[dict], threshold: float) -> str | None:
    s = mnli_score(outputs)
    if s is None:
        return None
    return BINARY[workload][0] if s >= threshold else BINARY[workload][1]


def d1_categories(spans: list[dict]) -> set[str]:
    labels = {s["label"] for s in spans}
    cats = {D1_DIRECT[x] for x in labels if x in D1_DIRECT}
    if "private_person" in labels and labels & D1_PERSONAL_ATTRS:
        cats.add("PERSONAL")
    return cats


def decide_d1(workload: str, outputs: list[dict]) -> str | None:
    if not outputs or "spans" not in outputs[0]:
        return None
    return BINARY[workload][0] if d1_categories(outputs[0]["spans"]) else BINARY[workload][1]


def decide_d2(workload: str, outputs: list[dict]) -> str | None:
    if not outputs or not outputs[0].get("probs"):
        return None
    return BINARY[workload][0] if outputs[0]["probs"][D2_GIVING_PII] >= D2_THRESHOLD else BINARY[workload][1]


def decide(slot: str, workload: str, output: list[dict] | None, *, threshold: float | None = None) -> str | None:
    """One prediction from one accuracy-level record output (see lambdas/aggregate.py)."""
    if not output:
        return None
    if slot in ("M1", "M2", "M4", "M5"):
        return decide_generative(slot, workload, output[0].get("text"))
    if slot == "M3":
        if threshold is None:
            raise ValueError("M3 needs the threshold chosen on the dev set")
        return decide_mnli(workload, output, threshold)
    if slot == "D1":
        return decide_d1(workload, output)
    if slot == "D2":
        return decide_d2(workload, output)
    raise ValueError(f"no DSPM decision rule for slot {slot}")
