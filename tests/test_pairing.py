"""Instance catalog, fair CPU/GPU pairing and per-run instance type overrides."""

from __future__ import annotations

import pytest

from bench import pairing
from bench.config import (
    load_models,
    load_run,
    load_targets,
    resolve_cells,
    run_alternates,
    run_instance_types,
)


def test_catalog_has_the_targets_and_prices():
    cat = pairing.load_catalog()
    for t in ("g6.xlarge", "g5.xlarge", "m8i.4xlarge", "c8i.4xlarge", "m8i.8xlarge"):
        s = cat["instances"][t]
        assert s["usd_per_hour"] > 0 and s["vcpus"] > 0
    assert cat["instances"]["m8i.4xlarge"]["amx"] is True and cat["instances"]["g6.xlarge"]["amx"] is False
    assert cat["instances"]["g6.xlarge"]["gpus"] == 1 and cat["instances"]["g6.48xlarge"]["gpus"] == 8
    assert pairing.types_for("cpu")[0] in cat["instances"] and all(
        cat["instances"][t]["amx"] for t in pairing.types_for("cpu"))
    assert all(cat["instances"][t]["accelerator"] == "gpu" for t in pairing.types_for("gpu"))


def test_judge_uses_the_hourly_price_band():
    ok = pairing.judge("m8i.4xlarge", "g6.xlarge")
    assert ok["fair"] is True and 0.9 < ok["ratio"] < 1.0
    big = pairing.judge("m8i.4xlarge", "g6.48xlarge")
    assert big["fair"] is False and big["ratio"] > 10 and "instance size" in big["reason"]
    small = pairing.judge("m8i.48xlarge", "g6.xlarge")
    assert small["fair"] is False and small["ratio"] < 0.5 and "oversized" in small["reason"]
    assert pairing.judge("m8i.4xlarge", "nope.xlarge")["fair"] is None


def test_propose_prefers_fair_single_gpu_then_closest_price():
    props = pairing.propose("m8i.4xlarge", "gpu", n=5)
    assert props[0]["instance_type"] == "g6.xlarge"
    assert all(p["fair"] for p in props) and all(p["gpus"] == 1 for p in props)
    ratios = [abs(p["ratio"] - 1) for p in props]
    assert ratios == sorted(ratios)
    back = pairing.propose("g6.48xlarge", "cpu", n=3)
    assert all(p["fair"] for p in back) and all(pairing.spec(p["instance_type"])["vcpus"] >= 96 for p in back)
    assert pairing.propose("nope.xlarge", "gpu") == []


def test_kv_cache_scales_with_memory():
    assert pairing.scale_kv_cache(32, "m8i.4xlarge", "m8i.4xlarge") == 32
    assert pairing.scale_kv_cache(32, "m8i.4xlarge", "m8i.8xlarge") == 64
    assert pairing.scale_kv_cache(32, "m8i.4xlarge", "c8i.4xlarge") == 16
    assert pairing.scale_kv_cache(32, "m8i.4xlarge", None) == 32


def test_run_instance_type_override_is_validated():
    models, targets = load_models("config/models.yaml"), load_targets("config/targets.yaml")
    tmap = targets.resolved()
    run = load_run("config/runs/smoke-m1.yaml")
    cells = resolve_cells(run, models, targets)
    assert run_instance_types(run, cells, tmap) == {"t1-g6": "g6.xlarge", "t2-m8i-4xl": "m8i.4xlarge"}
    assert run_alternates(run, "t1-g6", tmap) == ["g5.xlarge", "g6.2xlarge"]
    assert run_alternates(run, "t2-m8i-4xl", tmap) == tmap["t2-m8i-4xl"].alternates
    bigger = run.model_copy(update={"instance_types": {"t2-m8i-4xl": "m8i.8xlarge"}})
    assert run_instance_types(bigger, resolve_cells(bigger, models, targets), tmap)["t2-m8i-4xl"] == "m8i.8xlarge"
    with pytest.raises(ValueError, match="is a gpu instance; the target is cpu"):
        resolve_cells(run.model_copy(update={"instance_types": {"t2-m8i-4xl": "g6.xlarge"}}), models, targets)
    with pytest.raises(ValueError, match="not in the instance catalog"):
        resolve_cells(run.model_copy(update={"instance_types": {"t1-g6": "p5.48xlarge"}}), models, targets)
    with pytest.raises(ValueError, match="unknown target"):
        resolve_cells(run.model_copy(update={"instance_types": {"t9": "g6.xlarge"}}), models, targets)
    with pytest.raises(ValueError, match="has no cell"):
        resolve_cells(run.model_copy(update={"instance_types": {"t3-m8i-4xl-ov": "m8i.4xlarge"}}), models, targets)


def test_cellspec_scales_kv_cache_for_the_override():
    from bench import cellspec

    models, targets = load_models("config/models.yaml"), load_targets("config/targets.yaml")
    tmap = targets.resolved()
    run = load_run("config/runs/smoke-m1.yaml")
    cell = next(c for c in resolve_cells(run, models, targets) if c.target == "t2-m8i-4xl")
    kw = {"run_id": "r", "bucket": "b", "hf_token_secret": "arn"}
    base = cellspec.build(cell, models.get(cell.model), tmap[cell.target], run, **kw)
    big = cellspec.build(cell, models.get(cell.model), tmap[cell.target], run, instance_type="m8i.8xlarge", **kw)

    def kv(spec):
        cmd = spec["container"]["cmd"]
        return int(cmd[cmd.index("--kv-cache-memory-bytes") + 1]) // 2**30

    assert kv(base) == 32 and kv(big) == 64 and big["instance_type"] == "m8i.8xlarge"


def test_default_purchase_is_spot():
    from bench.config import PurchaseCfg

    assert PurchaseCfg().type == "spot"
    assert load_run("config/runs/smoke-m1.yaml").purchase.type == "spot"
