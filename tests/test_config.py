import copy

import pytest
from pydantic import ValidationError

from bench.config import (
    ModelCatalog,
    RunConfig,
    TargetCatalog,
    _strip_anchors,
    load_models,
    load_run,
    load_targets,
    load_yaml,
    resolve_cells,
)


@pytest.fixture(scope="module")
def models():
    return load_models("config/models.yaml")


@pytest.fixture(scope="module")
def targets():
    return load_targets("config/targets.yaml")


def test_catalogs_load(models, targets):
    assert {m.slot for m in models.models} >= {"M1", "M2", "M3", "M4", "M5", "D1", "D2"}
    resolved = targets.resolved()
    assert resolved["t2-c8i-4xl"].instance_type == "c8i.4xlarge"
    # inherited env survives a partial override
    assert resolved["t2-c8i-4xl"].engines["vllm"].env["VLLM_CPU_NUM_OF_RESERVED_CPU"] == "1"
    assert resolved["t2-c8i-4xl"].engines["vllm"].args["kv_cache_gib"] == 16
    assert resolved["t2-c8i-4xl"].engines["vllm"].args["max_num_seqs"] == 128
    assert "amx_fp16" not in resolved["t5-m7i-4xl"].cpu_features
    assert resolved["t5-m7i-4xl"].enabled is False


@pytest.mark.parametrize("path", ["config/runs/smoke-m1.yaml", "config/runs/full.yaml",
                                  "config/runs/dspm-smoke.yaml", "config/runs/dspm-full.yaml"])
def test_example_runs_resolve(path, models, targets):
    run = load_run(path)
    cells = resolve_cells(run, models, targets)
    assert cells


def test_unpinned_revision_rejected():
    raw = _strip_anchors(load_yaml("config/models.yaml"))
    bad = copy.deepcopy(raw)
    bad["models"][0]["revision"] = "main"
    with pytest.raises(ValidationError, match="40-hex"):
        ModelCatalog.model_validate(bad)


def test_size_cap_enforced():
    raw = _strip_anchors(load_yaml("config/models.yaml"))
    bad = copy.deepcopy(raw)
    bad["models"][0]["nominal_size_b"] = 7.6
    with pytest.raises(ValidationError, match="exceeds cap"):
        ModelCatalog.model_validate(bad)


def test_unpinned_image_rejected():
    raw = _strip_anchors(load_yaml("config/targets.yaml"))
    bad = copy.deepcopy(raw)
    bad["targets"][0]["engines"]["vllm"]["image"] = "vllm/vllm-openai:latest"
    with pytest.raises(ValidationError, match="pinned by digest"):
        TargetCatalog.model_validate(bad)


def _run(**over):
    base = dict(name="t", account="1", models=["m1-qwen2.5-3b"],
                cells=[{"target": "t1-g6", "precision": "bf16"}],
                concurrency=[1], per_level={"requests": 50}, repeats=1)
    base.update(over)
    return RunConfig.model_validate(base)


def test_full_mode_needs_methodology_defaults():
    with pytest.raises(ValidationError, match="200 requests"):
        _run(mode="full", per_level={"requests": 50}, repeats=3)
    with pytest.raises(ValidationError, match="3 repeats"):
        _run(mode="full", per_level={"requests": 200}, repeats=1)
    assert _run(mode="full", per_level={"requests": 200}, repeats=3).is_smoke is False


def test_smoke_label_when_below_defaults():
    assert _run().is_smoke is True


def test_launch_attempt_budget_is_bounded():
    from bench.config import PurchaseCfg

    assert PurchaseCfg().max_launch_attempts == 30 and PurchaseCfg().spot_capacity_window_s == 600
    assert PurchaseCfg(max_launch_attempts=1).max_launch_attempts == 1
    with pytest.raises(ValidationError):
        PurchaseCfg(max_launch_attempts=0)
    with pytest.raises(ValidationError):
        PurchaseCfg(max_launch_attempts=721)
    with pytest.raises(ValidationError):
        PurchaseCfg(spot_capacity_window_s=30)


def test_warmup_minimum_and_timeout_cap():
    with pytest.raises(ValidationError, match="20 warm-up"):
        _run(warmup_requests=5)
    with pytest.raises(ValidationError, match="request_timeout_s"):
        _run(request_timeout_s=900)


def test_fp16_refused_without_amx_fp16(models, targets):
    run = _run(models=["m3-roberta-large-mnli"],
               cells=[{"target": "t5-m7i-4xl", "precision": "fp16"}])
    raw = _strip_anchors(load_yaml("config/targets.yaml"))
    for t in raw["targets"]:
        if t["id"] == "t5-m7i-4xl":
            t["enabled"] = True
    tc = TargetCatalog.model_validate(raw)
    with pytest.raises(ValueError, match="amx_fp16"):
        resolve_cells(run, models, tc)


def test_engine_must_exist_on_target(models, targets):
    run = _run(cells=[{"target": "t3-m8i-4xl-ov", "precision": "bf16"}])
    with pytest.raises(ValueError, match="available on"):
        resolve_cells(run, models, targets)


def test_constrained_variant_only_on_vllm(models, targets):
    run = _run(models=["m5-flan-t5-xl"],
               cells=[{"target": "t4-m8i-4xl-torch", "precision": "bf16", "variant": "constrained"}])
    with pytest.raises(ValueError, match="constrained"):
        resolve_cells(run, models, targets)


def test_suite_membership(models, targets):
    run = _run(models=["d1-privacy-filter"], cells=[{"target": "t1-g6", "precision": "bf16"}])
    with pytest.raises(ValueError, match="standard suite"):
        resolve_cells(run, models, targets)
