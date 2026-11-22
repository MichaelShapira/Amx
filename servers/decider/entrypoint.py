"""Start the official Strands Decider server on staged weights, offline.

The host stages two Hugging Face snapshots under /models (host/stage_model.sh): the decider checkpoint (adapter,
readout, tokenizer, config) at DECIDER_CHECKPOINT and the Gemma torso at DECIDER_BASE_DIR. The decider loader
asks the Hub cache for the torso by repo id and revision (`AutoModel.from_pretrained(config.base_model,
revision=config.base_revision)`), so this script lays the staged torso into a Hub-cache layout under HF_HOME
with symlinks, pins the revision the catalog names, and then runs `strands-decider serve` with HF_HUB_OFFLINE
set. Nothing is downloaded on the host.

Environment (set by bench/cellspec.py):
  DECIDER_CHECKPOINT    /models/hf/<adapter repo>/<rev>
  DECIDER_BASE_DIR      /models/hf/<base repo>/<rev>
  DECIDER_BASE_REPO     google/gemma-4-E4B-it
  DECIDER_BASE_REVISION 40-hex commit
  DECIDER_MODEL_ID      the catalog id, echoed in /v1/models
  DECIDER_DEVICE        cuda | cpu
  DECIDER_PORT          8000
  DECIDER_MAX_BATCH     questions per forward (default 8)
  DECIDER_MAX_CONCURRENCY requests on the model at once (default 4)
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys


def log(msg: str) -> None:
    print(f"[decider-entrypoint] {msg}", flush=True)


def lay_hub_cache(repo: str, revision: str, snapshot: pathlib.Path, hub: pathlib.Path) -> None:
    """models--org--name/{refs/main, snapshots/<rev> -> symlinks to the staged files}."""
    if not snapshot.is_dir():
        raise SystemExit(f"staged torso not found at {snapshot}")
    if not (snapshot / ".complete").exists():
        raise SystemExit(f"staged torso at {snapshot} is incomplete (no .complete marker)")
    root = hub / ("models--" + repo.replace("/", "--"))
    snap = root / "snapshots" / revision
    (root / "refs").mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True, exist_ok=True)
    (root / "refs" / "main").write_text(revision)
    n = 0
    for f in snapshot.rglob("*"):
        if f.is_dir() or f.name in (".complete", ".revision"):
            continue
        rel = f.relative_to(snapshot)
        dst = snap / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(f)
        n += 1
    log(f"hub cache: {repo}@{revision[:10]} -> {n} files from {snapshot}")


def pin_base_revision(checkpoint: pathlib.Path, work: pathlib.Path, repo: str, revision: str) -> pathlib.Path:
    """A copy of the checkpoint dir whose config names the staged base repo and revision exactly, so the loader
    cannot resolve a different torso than the one the catalog pins."""
    cfg_name = "strands_decider_config.json" if (checkpoint / "strands_decider_config.json").exists() else "hobson_config.json"
    cfg = json.loads((checkpoint / cfg_name).read_text())
    if cfg.get("base_model") != repo:
        raise SystemExit(f"checkpoint base_model {cfg.get('base_model')!r} != catalog base_repo {repo!r}")
    work.mkdir(parents=True, exist_ok=True)
    for f in checkpoint.iterdir():
        if f.name in (".complete", ".revision", cfg_name):
            continue
        dst = work / f.name
        if not dst.exists():
            dst.symlink_to(f)
    cfg["base_revision"] = revision
    (work / cfg_name).write_text(json.dumps(cfg, indent=1))
    log(f"checkpoint: {checkpoint} (base {repo}@{revision[:10]} pinned)")
    return work


def main() -> int:
    env = os.environ
    checkpoint = pathlib.Path(env["DECIDER_CHECKPOINT"])
    base_dir = pathlib.Path(env["DECIDER_BASE_DIR"])
    repo, revision = env["DECIDER_BASE_REPO"], env["DECIDER_BASE_REVISION"]
    device = env.get("DECIDER_DEVICE", "cpu")
    port = env.get("DECIDER_PORT", "8000")
    hf_home = pathlib.Path(env.get("HF_HOME", "/tmp/cache/huggingface"))
    lay_hub_cache(repo, revision, base_dir, hf_home / "hub")
    ckpt = pin_base_revision(checkpoint, pathlib.Path("/tmp/decider-checkpoint"), repo, revision)
    cmd = ["strands-decider", "serve", str(ckpt), "--host", "0.0.0.0", "--port", port, "--device", device,  # noqa: S104 - in-container bind; published on the private IP only
           "--max-batch", env.get("DECIDER_MAX_BATCH", "8"), "--max-concurrency", env.get("DECIDER_MAX_CONCURRENCY", "4")]
    log("exec: " + " ".join(cmd))
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HOME": str(hf_home)})
    sys.stdout.flush()
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
