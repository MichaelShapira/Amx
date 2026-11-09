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
