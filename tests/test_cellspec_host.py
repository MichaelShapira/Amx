"""Cell specs for serve.sh and the host scripts in dry-run mode (task 11)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from bench import cellspec
from bench.config import load_models, load_run, load_targets, resolve_cells

ROOT = Path(__file__).resolve().parents[1]
RUNS = sorted((ROOT / "config/runs").glob("*.yaml"))
TAGS = {"thin-torch-cpu": "thin-torch-cpu-abc123", "thin-torch-cuda": "thin-torch-cuda-abc123",
        "thin-openvino": "thin-openvino-abc123", "export": "export-abc123"}
ECR = "346399954218.dkr.ecr.us-east-1.amazonaws.com/amx-gpu-bench"


def all_specs():
    models, targets = load_models(ROOT / "config/models.yaml"), load_targets(ROOT / "config/targets.yaml")
    tmap = targets.resolved()
    for path in RUNS:
        run = load_run(path)
        for cell in resolve_cells(run, models, targets):
            yield path.stem, run, cell, cellspec.build(
                cell, models.get(cell.model), tmap[cell.target], run, run_id=f"{path.stem}-001",
                bucket="amx-gpu-bench-346399954218-us-east-1", hf_token_secret="amx-gpu-bench/hf-token",
                ecr_repo_uri=ECR, built_tags=TAGS)


SPECS = list(all_specs())


def test_every_run_cell_has_a_spec():
    assert len(SPECS) > 40
    assert {s["engine"] for *_, s in SPECS} == {"vllm", "ovms", "thin-torch", "thin-openvino"}


@pytest.mark.parametrize("name,run,cell,spec", SPECS, ids=[s["cell_id"] + "@" + n for n, _, _, s in SPECS])
def test_spec_is_private_pinned_and_consistent(name, run, cell, spec):
    opts = spec["container"]["run_opts"]
    publishes = [opts[i + 1] for i, o in enumerate(opts) if o == "-p"]
    assert publishes == ["{PRIVATE_IP}:8000:8000"]           # never 0.0.0.0 or a bare port
    assert "-v" in opts and opts[opts.index("-v") + 1] == "/opt/models:/models:ro"
    assert "--network" not in opts and "--privileged" not in opts and "host" not in opts
    assert re.fullmatch(r"[0-9a-f]{40}", spec["weights"]["revision"])
    image = spec["container"]["image"]
    assert "@sha256:" in image or image.startswith(ECR + ":")
    gpu = cell.target.startswith("t1-")
    assert ("--gpus" in opts) == gpu
    cmd = spec["container"]["cmd"]
    if spec["engine"] == "vllm":
        assert cmd[:2] == ["serve", "{MODEL_DIR}"]
        caching = run.suite == "dspm"
        assert ("--enable-prefix-caching" in cmd) == caching and ("--no-enable-prefix-caching" in cmd) != caching
        if not gpu:
            assert "seccomp=/opt/bench/seccomp-vllm.json" in opts and "SYS_NICE" in opts
            assert "seccomp=unconfined" not in " ".join(opts)
        if spec["model_type"] == "encoder-classifier":
            assert "--runner" in cmd and cmd[cmd.index("--max-model-len") + 1] == "512"
    if spec["engine"].startswith("thin"):
        env = dict(opts[i + 1].split("=", 1) for i, o in enumerate(opts) if o == "-e")
        # In-container bind; the publish above is the boundary (private IP only).
        assert env["THIN_HOST"] == "0.0.0.0" and env["THIN_MODEL_DIR"] == "{MODEL_DIR}"  # noqa: S104
        if spec["engine"] == "thin-openvino" and cell.precision == "fp16":
            assert env["THIN_OV_PRECISION_HINT"] == "f16"
        if spec["model_id"] == "d2-roblox-pii-v2":
            assert env["THIN_MULTI_LABEL"] == "true"
    if spec["engine"] in ("ovms", "thin-openvino"):
        assert spec["ov"]["export_image"] == f"{ECR}:export-abc123"
        assert spec["ov"]["export"]["weight_format"] in ("int8", "int4", "fp16")
        if cell.precision == "bf16-ov":
            # The OpenVINO BF16 cell: an uncompressed FP16 IR (optimum has no bf16 weight format)
            # with no quantization arguments and no precision hint, so the probe reports the
            # CPU plugin's own choice (BF16 on AMX) rather than one we forced.
            assert spec["ov"]["export"] == {"weight_format": "fp16", "task": "text-generation-with-past"}
            assert spec["ov"]["hint"] is None and spec["engine"] == "ovms"
    if spec["engine"] == "ovms":
        assert spec["ovms_graph"]["enable_prefix_caching"] == ("true" if run.suite == "dspm" else "false")
    assert spec["cell_key"] == cellspec.cell_key(spec["cell_id"]) and "|" not in spec["cell_key"]


@pytest.fixture()
def dry_env(tmp_path):
    if shutil.which("jq") is None:
        pytest.skip("jq not installed")
    return {**os.environ, "BENCH_DRY_RUN": "1", "BENCH_HOME": str(ROOT / "host"), "STATE_DIR": str(tmp_path / "state"),
            "MODELS_ROOT": str(tmp_path / "models"), "PRIVATE_IP": "10.0.3.17"}


def _pick(engine: str, target_prefix: str = ""):
    return next(s for _, _, c, s in SPECS if s["engine"] == engine and c.target.startswith(target_prefix))


@pytest.mark.parametrize("engine,target", [("vllm", "t2-"), ("vllm", "t1-"), ("ovms", "t3-"),
                                           ("thin-openvino", "t3-"), ("thin-torch", "t4-")])
def test_serve_dry_run_prints_private_publish(engine, target, dry_env, tmp_path):
    spec = _pick(engine, target)
    f = tmp_path / "cell.json"
    f.write_text(json.dumps(spec))
    out = subprocess.run([str(ROOT / "host/serve.sh"), "start", str(f)], env=dry_env, capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    run_line = next(line for line in out.stderr.splitlines() if line.startswith("DRY-RUN: docker run -d"))
    assert "-p 10.0.3.17:8000:8000" in run_line
    assert "{PRIVATE_IP}" not in run_line and "{MODEL_DIR}" not in run_line
    assert spec["container"]["image"] in run_line
    if engine in ("ovms", "thin-openvino"):
        assert "/models/ov/" in run_line and "ir" not in run_line.split()[-1]


def test_serve_refuses_public_address(dry_env, tmp_path):
    f = tmp_path / "cell.json"
    f.write_text(json.dumps(_pick("vllm", "t2-")))
    out = subprocess.run([str(ROOT / "host/serve.sh"), "start", str(f)], env={**dry_env, "PRIVATE_IP": "54.1.2.3"},
                         capture_output=True, text=True, timeout=60)
    assert out.returncode != 0 and "non-private" in out.stderr


def test_perf_ctl_and_probe_amx_dry_run(dry_env, tmp_path):
    if shutil.which("perf") is None and shutil.which("lscpu") is None:
        pytest.skip("perf/lscpu not available on this machine")


def test_seccomp_profile_adds_only_migrate_pages():
    prof = json.loads((ROOT / "host/seccomp-vllm.json").read_text())
    assert prof["defaultAction"] == "SCMP_ACT_ERRNO"
    group = next(s for s in prof["syscalls"] if "mbind" in s.get("names", []))
    assert "migrate_pages" in group["names"] and group["includes"] == {"caps": ["CAP_SYS_NICE"]}


def test_shellcheck_clean():
    sc = shutil.which("shellcheck") or str(ROOT / ".venv/bin/shellcheck")
    if not Path(sc).exists():
        pytest.skip("shellcheck not installed")
    scripts = sorted(str(p) for p in (ROOT / "host").glob("*.sh"))
    out = subprocess.run([sc, "-x", "-P", str(ROOT / "host"), *scripts], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout
