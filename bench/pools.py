"""Request pools uploaded to s3://<bucket>/artifacts/ (design 10.1).

Standard-suite prompt sets come from tools/build_prompts.py (prompts/generated/) and DSPM
pools from tools/dspm/build_requests.py (data/dspm/v1/requests/). Classifier pairs are
joined into RoBERTa's pair string here, so vLLM /classify and the thin server receive the
same text. The manifest maps every S3 key to its line count; preflight reads it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIR_SEP = "</s></s>"     # RoBERTa / XLM-R pair separator (tests: test_mnli_joined_string_equals_pair_encoding)


def normalize_line(line: dict) -> dict:
    if "pair" in line and "input" not in line:
        a, b = line["pair"]
        line = {k: v for k, v in line.items() if k != "pair"}
        line["input"] = f"{a}{PAIR_SEP}{b}"
    return line


def standard_pools(prompts_dir: Path = ROOT / "prompts/generated") -> dict[str, bytes]:
    """S3 key -> JSONL bytes for every generated prompt set."""
    out: dict[str, bytes] = {}
    for f in sorted(prompts_dir.glob("*/*.jsonl")):
        lines = [normalize_line(json.loads(x)) for x in f.read_text().splitlines() if x.strip()]
        out[f"artifacts/prompts/{f.parent.name}/{f.name}"] = "".join(
            json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in lines).encode()
    return out


def dspm_pools(requests_dir: Path = ROOT / "data/dspm/v1/requests") -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for f in sorted(requests_dir.glob("*/*.jsonl")):
        out[f"artifacts/dspm/v1/requests/{f.parent.name}/{f.name}"] = f.read_bytes()
    return out


def manifest(pools: dict[str, bytes]) -> dict:
    return {"pool_sizes": {k: v.count(b"\n") for k, v in sorted(pools.items())},
            "sha256": {k: hashlib.sha256(v).hexdigest() for k, v in sorted(pools.items())}}


def local_pool_sizes() -> dict[str, int]:
    """Pool sizes from the local files, for `bench validate` and the UI before upload."""
    sizes = manifest(standard_pools())["pool_sizes"]
    req = ROOT / "data/dspm/v1/requests/manifest.json"
    if req.exists():
        sizes.update(json.loads(req.read_text())["pool_sizes"])
    return sizes
