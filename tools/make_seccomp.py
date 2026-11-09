"""Generate host/seccomp-vllm.json (design 7.3).

Docker's default seccomp profile (moby v25.0.8, checked by sha256) with one change:
`migrate_pages` joins the NUMA syscalls that the default profile already allows when the
container has CAP_SYS_NICE (get_mempolicy, mbind, set_mempolicy). vLLM's CPU backend
needs it for NUMA binding; this is the narrow alternative to `seccomp=unconfined`.

    .venv/bin/python tools/make_seccomp.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/moby/moby/v25.0.8/profiles/seccomp/default.json"
SHA256 = "9c1025c88ccaa517b648da571961838744ea2137f176bfe6a48b21294cae9c76"
OUT = Path(__file__).resolve().parents[1] / "host" / "seccomp-vllm.json"


def build(raw: bytes) -> dict:
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise SystemExit("moby default seccomp profile checksum mismatch")
    prof = json.loads(raw)
    group = next(s for s in prof["syscalls"] if "mbind" in s.get("names", [])
                 and s.get("includes", {}).get("caps") == ["CAP_SYS_NICE"])
    if "migrate_pages" not in group["names"]:
        group["names"] = sorted([*group["names"], "migrate_pages"])
    return prof


def main() -> int:
    with urllib.request.urlopen(URL, timeout=30) as r:  # noqa: S310 - fixed https URL, checksum verified
        prof = build(r.read())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(prof, indent="\t") + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
