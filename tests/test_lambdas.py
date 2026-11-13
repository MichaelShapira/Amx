"""Lambda logic with moto (S3, DynamoDB) and the fake inference server (task 12.6)."""

from __future__ import annotations

import datetime as dt
import gzip
import json
import time

import boto3
import pytest

moto = pytest.importorskip("moto")

from tests import fake_server as fs  # noqa: E402

BUCKET = "amx-gpu-bench-123456789012-us-east-1"


class Ctx:
    """Minimal Lambda context."""

    def __init__(self, budget_s: float = 900):
        self.deadline = time.time() + budget_s

    def get_remaining_time_in_millis(self) -> int:
        return int((self.deadline - time.time()) * 1000)


@pytest.fixture()
def aws(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("BENCH_BUCKET", BUCKET)
    monkeypatch.setenv("BENCH_TABLE", "amx-gpu-bench")
    monkeypatch.setenv("AGENT_PUBLISH_DELAY_S", "0")       # no CloudWatch agent to wait for
    with moto.mock_aws():
        from lambdas import common

        common.client.cache_clear()
        common.client_in.cache_clear()
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        boto3.client("dynamodb", region_name="us-east-1").create_table(
            TableName="amx-gpu-bench", BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"},
                                  {"AttributeName": "sk", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}])
        yield common
        common.client.cache_clear()
        common.client_in.cache_clear()


@pytest.fixture()
def server():
    fs.Script.ttft_ms, fs.Script.itl_ms, fs.Script.tokens = 20.0, 2.0, 8
    fs.Script.tokens_per_event, fs.Script.status, fs.Script.omit_usage = 1, 200, False
    fs.Script.prompt_tokens = 42
    srv, url = fs.start()
    yield url
    srv.shutdown()


def _pool(common, key: str, n: int, kind: str = "prompt") -> dict:
    rows = [{"id": f"p{i:03d}", "item_id": f"item-{i:03d}", "prompt_tokens_expected": 42, "bytes": 100,
             **({"prompt": f"prompt {i}"} if kind == "prompt" else {"input": f"text {i}"})} for i in range(n)]
    common.put_jsonl_gz(key, rows)
    return {"bucket": BUCKET, "key": key, "size": n}


def _level(url: str, pool: dict, **kw) -> dict:
    lvl = {"run_id": "r1", "cell_key": "m1__t2__bf16", "level_key": "m1__t2__bf16__short__c4__r0", "conc": 4,
           "repeat": 0, "endpoint": url, "api_path": "/v1/completions", "api_task": "generate",
           "payload": {"model": "m1", "max_tokens": 8, "temperature": 0, "seed": 0, "ignore_eos": True},
           "pool": pool, "batch": 1, "request_timeout_s": 30, "health_path": "/health", "quota": 10,
           "target_id": "t2-m8i-4xl"}
    lvl.update(kw)
    return lvl


def test_plan_wave_quotas_sum_to_remaining(aws, server):
    from lambdas import waves

    lvl = _level(server, {"bucket": BUCKET, "key": "x", "size": 10}, quota=10)
    waves.init_level({"level": lvl})
    out = waves.plan_wave({"level": lvl, "wave": 0})
    assert sum(i["quota"] for i in out["items"]) == 10 and len(out["items"]) == 4
    assert all(i["start_at_ns"] > time.time_ns() for i in out["items"])
    aws.table().update_item(Key={"pk": "RUN#r1", "sk": f"LEVEL#{lvl['level_key']}"},
                            UpdateExpression="SET completed = :c", ExpressionAttributeValues={":c": 7})
    out = waves.plan_wave({"level": lvl, "wave": 1})
    assert [i["quota"] for i in out["items"]] == [1, 1, 1]
    assert waves.wave_check({"level": lvl, "wave": 1})["more"] is True


def test_worker_end_to_end_and_aggregate(aws, server):
    from lambdas import aggregate, waves, worker

    pool = _pool(aws, "artifacts/prompts/m1/short.jsonl", 20)
    lvl = _level(server, pool, quota=12)
    waves.init_level({"level": lvl})
    items = waves.plan_wave({"level": lvl, "wave": 0})["items"]
    for it in items:
        it["start_at_ns"] = time.time_ns() + 200_000_000
    outs = [worker.handler(it, Ctx()) for it in items]
    assert sum(o["n"] for o in outs) == 12 and all(o["errors"] == 0 for o in outs)
    state = aws.plain(aws.table().get_item(Key={"pk": "RUN#r1", "sk": f"LEVEL#{lvl['level_key']}"})["Item"])
    assert state["completed"] == 12
    assert waves.wave_check({"level": lvl, "wave": 0})["more"] is False
    recs = aggregate.level_records("r1", lvl["cell_key"], lvl["level_key"])
    assert len(recs) == 12
    assert {r["prompt_id"] for r in recs} == {f"p{i:03d}" for i in range(12)}     # deterministic striding
    r = recs[0]
    assert r["completion_tokens"] == 8 and r["prompt_tokens"] == r["prompt_tokens_expected"] == 42
    assert r["send_to_first_ns"] > 15_000_000 and r["conc"] == 4 and r["wave"] == 0
    summary = aggregate.aggregate_level({"level": lvl, "perf": {"counts": {"exe.amx_busy": 10, "cycles": 1000}}})
    assert summary["n_ok"] == 12 and summary["flags"] == [] or all("prompt_token" not in f for f in summary["flags"])
    m = aws.get_json(f"runs/r1/metrics/{lvl['cell_key']}/{lvl['level_key']}.json")
    assert m["amx_counter"]["ratio"] == pytest.approx(0.01) and m["ttft_ms"]["n"] == 12


def test_accuracy_level_covers_every_item_once_across_waves(aws, server):
    from lambdas import aggregate, waves, worker

    pool = _pool(aws, "artifacts/dspm/v1/requests/m1/W3.jsonl", 23)
    lvl = _level(server, pool, accuracy=True, keep_output=True, max_items_per_worker=4, workload="W3",
                 level_key="m1__t2__bf16__W3-accuracy__c4", name="W3-accuracy", quota=None)
    lvl.pop("quota")
    waves.init_level({"level": lvl})
    seen = []
    for wave in range(10):
        plan = waves.plan_wave({"level": lvl, "wave": wave})
        for it in plan["items"]:
            it["start_at_ns"] = time.time_ns()
            # the first worker of the first wave "runs out of time" after one request
            ctx = Ctx(30 + 30 + 0.05) if (wave == 0 and it["worker"] == 0) else Ctx()
            worker.handler(it, ctx)
        if not waves.wave_check({"level": lvl, "wave": wave})["more"]:
            break
    recs = aggregate.level_records("r1", lvl["cell_key"], lvl["level_key"])
    seen = sorted(i for r in recs for i in r["item_ids"])
    assert seen == [f"item-{i:03d}" for i in range(23)]
    assert all("output_text" in r for r in recs)
    aggregate.aggregate_level({"level": lvl})
    preds = aws.read_jsonl(f"runs/r1/predictions/{lvl['cell_key']}/W3-accuracy.jsonl.gz")
    assert len(preds) == 23 and preds[0]["output"][0]["text"].startswith("tok")


def test_classify_worker_batches_inputs(aws, server):
    from lambdas import aggregate, waves, worker

    pool = _pool(aws, "artifacts/prompts/m3/cls512.jsonl", 16, kind="input")
    lvl = _level(server, pool, api_path="/classify", api_task="classify", payload={"model": "m3"}, batch=4,
                 quota=4, conc=2, level_key="m3__t2__bf16__cls512__c2__b4__r0")
    waves.init_level({"level": lvl})
    for it in waves.plan_wave({"level": lvl, "wave": 0})["items"]:
        it["start_at_ns"] = time.time_ns()
        worker.handler(it, Ctx())
    recs = aggregate.level_records("r1", lvl["cell_key"], lvl["level_key"])
    assert len(recs) == 4 and all(r["batch_size"] == 4 and r["n_inputs"] == 4 for r in recs)
    assert all(r["prompt_tokens"] == 42 * 4 == r["prompt_tokens_expected"] for r in recs)
    s = aggregate.aggregate_level({"level": lvl})
    assert s["n_ok"] == 4


def test_warm_mode_returns_quickly(aws):
    from lambdas import worker

    t = time.time()
    assert worker.handler({"mode": "warm"}) == {"warm": True}
    assert 2.9 < time.time() - t < 4


def test_reaper_decisions():
    from lambdas.reaper import decide

    now = time.time()
    inst = {"LaunchTime": dt.datetime.fromtimestamp(now - 3600, dt.UTC)}
    assert decide(inst, {"last_activity_at": now - 60}, active=False, now=now, idle_min=30, max_uptime_h=8) is None
    assert "idle" in decide(inst, {"last_activity_at": now - 3000}, active=False, now=now, idle_min=30, max_uptime_h=8)
    assert decide(inst, {"last_activity_at": now - 3000}, active=True, now=now, idle_min=30, max_uptime_h=8) is None
    old = {"LaunchTime": dt.datetime.fromtimestamp(now - 10 * 3600, dt.UTC)}
    assert "running for more" in decide(old, {"last_activity_at": now - 60}, active=False, now=now, idle_min=30,
                                        max_uptime_h=8)


def test_reaper_terminates_idle_and_stopped_leftovers(aws):
    """On-Demand instances used to be *stopped* for reuse; nothing is reused now, so a stopped
    managed instance is terminated on sight, an idle one (no live run) is terminated, and an
    instance whose run is still going is left alone."""
    from lambdas import reaper

    ec2 = boto3.client("ec2", region_name="us-east-1")
    ami = ec2.describe_images(Owners=["amazon"])["Images"][0]["ImageId"]

    def launch(run):
        return ec2.run_instances(ImageId=ami, InstanceType="m8i.4xlarge", MinCount=1, MaxCount=1, TagSpecifications=[
            {"ResourceType": "instance", "Tags": [{"Key": "bench:managed", "Value": "true"}, {"Key": "bench:target", "Value": "t2-m8i-4xl"},
                                                  {"Key": "bench:run", "Value": run}]}])["Instances"][0]["InstanceId"]
    live, idle, left = launch("r-live"), launch("r-idle"), launch("r-left")
    ec2.stop_instances(InstanceIds=[left])
    aws.set_status("r-live", "META", "running")
    aws.set_status("r-idle", "META", "complete")
    aws.touch_target("t2-m8i-4xl", "r-live")
    boto3.resource("dynamodb", region_name="us-east-1").Table("amx-gpu-bench").put_item(
        Item={**aws.activity_key("t2-m8i-4xl", "r-idle"), "last_activity_at": int(time.time()) - 3600})
    out = reaper.handler({})
    acted = {x["instance_id"]: x for x in out["stopped"]}
    # The stopped leftover goes at once. The 'idle' one was launched seconds ago: the idle rule
    # counts from max(last heartbeat, launch), so a fresh instance still booting is never reaped.
    assert set(acted) == {left} and acted[left]["action"] == "terminated"
    assert "left behind" in acted[left]["reason"] and acted[left]["run"] == "r-left"
    states = {i["InstanceId"]: i["State"]["Name"] for res in ec2.describe_instances()["Reservations"] for i in res["Instances"]}
    assert states[live] == "running" and states[idle] == "running" and states[left] in ("shutting-down", "terminated")
    old_launch = dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)
    assert "idle" in reaper.decide({"LaunchTime": old_launch}, {"last_activity_at": time.time() - 3600}, active=False,
                                   now=time.time(), idle_min=30, max_uptime_h=8)
    # An unmanaged instance is never touched.
    other = ec2.run_instances(ImageId=ami, InstanceType="t3.micro", MinCount=1, MaxCount=1)["Instances"][0]["InstanceId"]
    reaper.handler({})
    assert {i["InstanceId"]: i["State"]["Name"] for res in ec2.describe_instances()["Reservations"] for i in res["Instances"]}[other] == "running"


def test_preflight_checks(tmp_path):
    from bench.config import load_models, load_run, load_targets, resolve_cells
    from lambdas.preflight import check_run

    run = load_run("config/runs/full.yaml")
    models, targets = load_models(), load_targets()
    cells = resolve_cells(run, models, targets)
    tmap = targets.resolved()
    ok = check_run(run, cells, models, tmap, reserved_worker_concurrency=80,
                   offerings={tmap[c.target].instance_type for c in cells}, host_flags={})
    assert ok == []
    errs = check_run(run, cells, models, tmap, reserved_worker_concurrency=32, offerings={"g6.xlarge"},
                     host_flags={"t4-m8i-4xl-torch": {"amx_bf16", "amx_int8", "amx_tile"}})
    text = " | ".join(errs)
    assert "reserved concurrency 32 < highest level 64" in text
    assert "not offered" in text and "m8i.4xlarge" in text
    assert "t4-m8i-4xl-torch|fp16: m8i.4xlarge has no amx_fp16" in text
    # A GPU cell is never subject to a CPU-feature requirement, whatever the GPU host's facts say
    # (2026-11-12: RoBERTa FP16 on g6.xlarge was refused with "host facts of t1-g6 lack amx_fp16").
    from bench.config import Cell
    gpu_run = run.model_copy(update={"models": ["m3-roberta-large-mnli"],
                                     "cells": [Cell(target="t1-g6", precision="fp16", model="m3-roberta-large-mnli")],
                                     "instance_types": {"t1-g6": "g6.xlarge"}})
    gpu_cells = resolve_cells(gpu_run, models, targets)
    assert gpu_cells and gpu_cells[0].precision == "fp16"
    errs = check_run(gpu_run, gpu_cells, models, tmap, reserved_worker_concurrency=80, offerings={"g6.xlarge"},
                     host_flags={"t1-g6": set()})
    assert errs == []
    # No host facts yet: the instance family decides (8i has AMX-FP16, 7i does not).
    from lambdas.preflight import expected_cpu_flags
    assert "amx_fp16" in expected_cpu_flags("c8i.4xlarge") and "amx_fp16" not in expected_cpu_flags("m7i.4xlarge")
    assert expected_cpu_flags("g6.xlarge") is None
    run7 = run.model_copy(update={"instance_types": {**run.instance_types, "t4-m8i-4xl-torch": "m7i.4xlarge"}})
    errs = check_run(run7, cells, models, tmap, reserved_worker_concurrency=80, offerings={"m7i.4xlarge", *(tmap[c.target].instance_type for c in cells)},
                     host_flags={})
    assert any("m7i.4xlarge has no amx_fp16" in e for e in errs)


def test_preflight_dspm_hashes_refuse_stale_pools():
    import hashlib

    from lambdas.preflight import dspm_hashes

    ds = json.dumps({"version": "v1", "dataset_sha256": "d" * 64, "seed": 7}).encode()
    req = {"policy": {"policy_sha256": "p" * 64}, "dataset_manifest_sha256": hashlib.sha256(ds).hexdigest()}
    hashes, errs = dspm_hashes(req, ds)
    assert errs == [] and hashes["dataset_sha256"] == "d" * 64 and hashes["policy_sha256"] == "p" * 64
    _, errs = dspm_hashes({**req, "dataset_manifest_sha256": "0" * 64}, ds)
    assert "different dataset" in errs[0]
    assert "upload-artifacts" in dspm_hashes(None, ds)[1][0]


def test_capacity_log_appends(aws):
    from lambdas import target_control

    target_control._log("r1", "t1-g6", "capacity", {"code": "InsufficientInstanceCapacity", "time": "x"})
    target_control._log("r1", "t1-g6", "capacity", {"code": "InsufficientInstanceCapacity", "time": "y"})
    body = boto3.client("s3", region_name="us-east-1").get_object(Bucket=BUCKET, Key="runs/r1/capacity/t1-g6.json")
    assert [e["time"] for e in json.loads(body["Body"].read())] == ["x", "y"]


def test_purchase_decisions():
    from lambdas.target_control import decide_after_failure, decide_market, market_options

    spot = {"type": "spot", "spot_capacity_window_s": 600, "spot_fallback_on_demand": True}
    assert market_options({"type": "on-demand"}) is None
    assert market_options(spot) == {"MarketType": "spot", "SpotOptions": {
        "SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}}
    assert market_options({**spot, "spot_max_price_usd": 0.5})["SpotOptions"]["MaxPrice"] == "0.5"
    assert decide_market({"type": "on-demand"}, 0, 10_000) == "on-demand"
    assert decide_market(spot, 1000, 1500) == "spot"            # inside the capacity window
    assert decide_market(spot, 1000, 1700) == "on-demand"       # window used up: fall back
    with pytest.raises(RuntimeError, match="spot_fallback_on_demand"):
        decide_market({**spot, "spot_fallback_on_demand": False}, 1000, 1700)
    running, gone = {"State": {"Name": "running"}}, {"State": {"Name": "terminated"}}
    assert decide_after_failure(gone, 0, spot, "levels") == "relaunch"
    assert decide_after_failure(None, 0, spot, "bringup") == "relaunch"
    assert decide_after_failure(running, 0, spot, "levels") == "resume"
    assert decide_after_failure(running, 0, spot, "bringup") == "fail"
    assert decide_after_failure(gone, 3, spot, "levels") == "fail"   # relaunch cap


def test_give_up_after_max_launch_attempts_records_why(aws, monkeypatch):
    """purchase.max_launch_attempts refused launches -> the target is given up with a plain
    reason: `gave_up` event, every cell `not_run` + reason, META.not_run, and the RuntimeError
    for the state machine's TargetFailed catch. Seen 2026-11-11 as a bare 'Not run' in the UI."""
    from lambdas import target_control as tc

    monkeypatch.setattr(tc, "find_instance", lambda tid, run_id: None)

    def refused(run_id, tid, purchase, market, instance_type=None):
        tc._log(run_id, tid, "capacity", {"time": f"2026-11-11T09:{40 + len(seen):02d}:00Z", "market": market,
                                          "code": "InsufficientInstanceCapacity", "instance_type": instance_type or "g6.2xlarge"})
        seen.append(market)
        return None

    seen: list[str] = []
    monkeypatch.setattr(tc, "_launch", refused)
    purchase = {"type": "spot", "spot_capacity_window_s": 600, "spot_fallback_on_demand": True, "max_launch_attempts": 2}
    cells = [{"cell_id": "d1|t1-g6|bf16"}, {"cell_id": "d2|t1-g6|bf16"}]
    aws.set_status("r9", "CELL#d1|t1-g6|bf16", "planned")
    aws.set_status("r9", "CELL#d2|t1-g6|bf16", "planned")
    aws.set_status("r9", "META", "running")
    ev = {"run_id": "r9", "target_id": "t1-g6", "purchase": purchase, "alternates": ["g5.xlarge"], "cells": cells}
    assert tc.ensure_running(ev) == {"ready": False, "reason": "capacity", "attempt": 1, "max_attempts": 2}
    assert tc.ensure_running(ev) == {"ready": False, "reason": "capacity", "attempt": 2, "max_attempts": 2}
    with pytest.raises(RuntimeError, match="No capacity: AWS refused 2 launch request"):
        tc.ensure_running(ev)                                   # third call: attempts == max -> give up, no launch
    assert len(seen) == 2
    tbl = boto3.resource("dynamodb", region_name="us-east-1").Table("amx-gpu-bench")
    for cid in ("d1|t1-g6|bf16", "d2|t1-g6|bf16"):
        item = tbl.get_item(Key={"pk": "RUN#r9", "sk": f"CELL#{cid}"})["Item"]
        assert item["status"] == "not_run" and "gave up after 2 attempt(s)" in item["reason"]
        assert "g6.2xlarge" in item["reason"] and "Spot" in item["reason"]
    meta = tbl.get_item(Key={"pk": "RUN#r9", "sk": "META"})["Item"]
    assert [x["target_id"] for x in meta["not_run"]] == ["t1-g6"] and int(meta["not_run"][0]["attempts"]) == 2
    events = aws.get_json("runs/r9/targets/t1-g6.json")
    assert events[-1]["event"] == "gave_up" and events[-1]["refusals"] == 2
    # The attempt counter was reset, so the next run on this target starts from zero.
    assert int(tc._activity("t1-g6", "r9").get("launch_attempts") or 0) == 0


def test_every_run_gets_its_own_instance(aws, monkeypatch):
    """2026-11-12: a second run adopted the first run's instance (lookup by target only), removed
    its serving container and replaced its instance. Instances are now per run (bench:run tag):
    two runs on the same hardware option each launch their own, with their own launch
    bookkeeping, and stopping one run leaves the other's instance alone."""
    from lambdas import target_control as tc

    ec2 = boto3.client("ec2", region_name="us-east-1")
    vpc = ec2.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]["VpcId"]
    subnet = ec2.create_subnet(VpcId=vpc, CidrBlock="10.0.1.0/24")["Subnet"]["SubnetId"]
    ami = ec2.describe_images(Owners=["amazon"])["Images"][0]["ImageId"]
    lt = ec2.create_launch_template(LaunchTemplateName="amx-gpu-bench-t1-g6", LaunchTemplateData={
        "ImageId": ami, "InstanceType": "g6.xlarge",
        "NetworkInterfaces": [{"DeviceIndex": 0, "SubnetId": subnet, "AssociatePublicIpAddress": False}]})[
        "LaunchTemplate"]["LaunchTemplateId"]
    boto3.client("ssm", region_name="us-east-1").put_parameter(Name="/amx-gpu-bench/targets/t1-g6/launch-template",
                                                              Value=lt, Type="String")
    tc._ssm_online = lambda iid, deadline: True
    tc._boot_complete = lambda iid, deadline: True
    purchase = {"type": "spot", "spot_capacity_window_s": 600, "spot_fallback_on_demand": True, "max_launch_attempts": 30}
    first = tc.ensure_running({"run_id": "r-first", "target_id": "t1-g6", "purchase": purchase})
    second = tc.ensure_running({"run_id": "r-second", "target_id": "t1-g6", "purchase": purchase})
    assert first["ready"] and second["ready"] and first["instance_id"] != second["instance_id"]
    # Each run finds only its own instance again; nothing is replaced or logged as reused.
    assert tc.ensure_running({"run_id": "r-first", "target_id": "t1-g6", "purchase": purchase})["instance_id"] == first["instance_id"]
    assert tc.ensure_running({"run_id": "r-second", "target_id": "t1-g6", "purchase": purchase})["instance_id"] == second["instance_id"]
    for r in ("r-first", "r-second"):
        assert [e["event"] for e in aws.get_json(f"runs/{r}/targets/t1-g6.json")] == ["launched"]
    assert tc.find_instance("t1-g6", "r-first")["InstanceId"] == first["instance_id"]
    assert tc.find_instance("t1-g6", "r-nobody") is None
    # Bookkeeping is per (target, run).
    assert aws.activity_key("t1-g6", "r-first") == {"pk": "TARGET#t1-g6", "sk": "ACTIVITY#r-first"}
    tc._set_activity("t1-g6", "r-first", launch_attempts=4)
    assert int(tc._activity("t1-g6", "r-first")["launch_attempts"]) == 4 and not tc._activity("t1-g6", "r-second").get("launch_attempts")
    # Stopping one run terminates only its own (Spot) instance.
    out = tc.stop({"target_id": "t1-g6", "run_id": "r-first"})
    assert out["terminated"] == first["instance_id"]
    states = {i["InstanceId"]: i["State"]["Name"] for res in ec2.describe_instances()["Reservations"] for i in res["Instances"]}
    assert states[first["instance_id"]] in ("shutting-down", "terminated") and states[second["instance_id"]] == "running"
    assert tc.stop({"target_id": "t1-g6", "run_id": "r-first"}) == {"terminated": None}


def test_error_message_unwraps_the_step_functions_envelope():
    from lambdas.target_control import error_message

    err = {"Error": "RuntimeError", "Cause": json.dumps({"errorMessage": "bring-up of c failed (Failed): tail", "stackTrace": ["x"]})}
    assert error_message(err) == "bring-up of c failed (Failed): tail"
    assert error_message({"Error": "States.Timeout", "Cause": "not json"}) == "not json"
    assert error_message({"Error": "States.Timeout"}) == "States.Timeout"
    assert error_message("plain") == "plain" and error_message(None) == ""


def test_capacity_reason_reads_like_a_sentence():
    from lambdas.target_control import capacity_reason

    caps = [{"time": "2026-11-11T09:38:00Z", "market": "spot", "instance_type": "g6.2xlarge"},
            {"time": "2026-11-11T09:50:00Z", "market": "on-demand", "instance_type": "g5.xlarge"},
            {"time": "2026-11-11T10:09:00Z", "market": "spot", "instance_type": "g5.4xlarge"}]
    r = capacity_reason(caps, 30, {"type": "spot"})
    assert r.startswith("No capacity: AWS refused 3 launch request(s) over 31 minutes for g5.4xlarge, g5.xlarge, g6.2xlarge "
                        "(On-Demand and Spot).")
    assert "gave up after 30 attempt(s), the configured maximum" in r
    assert "level" not in r and "pass" not in r
    assert "the requested size" in capacity_reason([], 5, {"type": "on-demand"})


def test_worker_raises_when_endpoint_is_dead(aws):
    """Connection failures in a row raise EndpointUnreachable (not counted toward the quota);
    HTTP errors and timeouts are measurements and do not."""
    from lambdas import worker

    down = [{"error": "client_ConnectError"}] * 5
    assert worker.endpoint_down(down) is True
    assert worker.endpoint_down(down[:4]) is False
    assert worker.endpoint_down([{"error": "http_503"}] * 5) is False
    assert worker.endpoint_down([{"error": "timeout"}] * 5) is False
    assert worker.endpoint_down([*down[:4], {"error": None}]) is False
    aws.put_jsonl_gz("artifacts/prompts/dead.jsonl", [{"id": f"p{i}", "prompt": "hi", "tokens": 1} for i in range(8)])
    aws.table().put_item(Item={"pk": "RUN#r9", "sk": "LEVEL#lk", "completed": 0, "errors": 0})
    ev = {"run_id": "r9", "cell_key": "ck", "level_key": "lk", "conc": 1, "wave": 0, "worker": 0, "quota": 8,
          "endpoint": "http://127.0.0.1:9", "api_path": "/v1/completions", "api_task": "generate",
          "payload": {"model": "m", "max_tokens": 1}, "pool": {"bucket": BUCKET, "key": "artifacts/prompts/dead.jsonl"},
          "start_at_ns": time.time_ns(), "request_timeout_s": 2}
    with pytest.raises(worker.EndpointUnreachable):
        worker.handler(ev)
    item = aws.plain(aws.table().get_item(Key={"pk": "RUN#r9", "sk": "LEVEL#lk"})["Item"])
    assert item.get("endpoint_down") is True and item["completed"] == 0


def test_capacity_fallback_rotates_alternates():
    from lambdas.target_control import pick_instance_type

    assert pick_instance_type(0, ["g5.xlarge"]) is None and pick_instance_type(2, ["g5.xlarge"]) is None
    assert pick_instance_type(3, ["g5.xlarge"]) == "g5.xlarge"       # after 3 failures on the template type
    assert pick_instance_type(4, ["g5.xlarge"]) is None               # then alternate again
    assert pick_instance_type(5, ["g5.xlarge"]) == "g5.xlarge"
    assert pick_instance_type(9, []) is None and pick_instance_type(9, None) is None


def test_spot_interruption_relaunches_and_resumes(aws):
    """Spot instance disappears mid-cell: check() terminates nothing twice, launches a
    replacement from the launch template, keeps done levels and re-runs the rest."""
    from lambdas import target_control as tc

    ec2 = boto3.client("ec2", region_name="us-east-1")
    vpc = ec2.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]["VpcId"]
    subnet = ec2.create_subnet(VpcId=vpc, CidrBlock="10.0.1.0/24")["Subnet"]["SubnetId"]
    ami = ec2.describe_images(Owners=["amazon"])["Images"][0]["ImageId"]
    lt = ec2.create_launch_template(LaunchTemplateName="amx-gpu-bench-t1-g6", LaunchTemplateData={
        "ImageId": ami, "InstanceType": "g6.xlarge",
        "NetworkInterfaces": [{"DeviceIndex": 0, "SubnetId": subnet, "AssociatePublicIpAddress": False}]})[
        "LaunchTemplate"]["LaunchTemplateId"]
    boto3.client("ssm", region_name="us-east-1").put_parameter(Name="/amx-gpu-bench/targets/t1-g6/launch-template",
                                                              Value=lt, Type="String")
    tc._ssm_online = lambda iid, deadline: True          # moto has no DescribeInstanceInformation
    tc._boot_complete = lambda iid, deadline: True
    purchase = {"type": "spot", "spot_capacity_window_s": 600, "spot_fallback_on_demand": True,
                "max_relaunches_per_cell": 3}
    first = tc.ensure_running({"run_id": "r1", "target_id": "t1-g6", "purchase": purchase})
    assert first["ready"] is True and first["lifecycle"] in ("spot", "on-demand")
    old = first["instance_id"]
    tags = {t["Key"]: t["Value"] for t in ec2.describe_instances(InstanceIds=[old])["Reservations"][0]["Instances"][0]["Tags"]}
    assert tags["bench:managed"] == "true" and tags["bench:target"] == "t1-g6" and tags["project"] == "amx-gpu-bench"
    # Plan with four levels: done; failed because the endpoint died (marked by the level state
    # machine); running (the Map had started it when the cell failed); not started.
    levels = [{"level_key": f"ck__short__c{c}__r0", "cell_key": "ck"} for c in (1, 4, 8, 16)]
    aws.put_json("runs/r1/plan/ck.json", {"levels": levels, "quality": None})
    aws.put_json("runs/r1/bringup/ck.json", {"api_path": "/v1/completions"})
    aws.set_status("r1", "LEVEL#ck__short__c1__r0", "done")
    aws.set_status("r1", "LEVEL#ck__short__c4__r0", "failed", error="EndpointUnreachable")
    aws.set_status("r1", "LEVEL#ck__short__c8__r0", "running")
    ec2.terminate_instances(InstanceIds=[old])                   # the interruption
    cell = {"cell_id": "c|t|p", "cell_key": "ck", "cpu": False}
    out = tc.check({"run_id": "r1", "target_id": "t1-g6", "instance_id": old, "private_ip": "10.0.1.5", "cell": cell,
                    "purchase": purchase, "recovery": {"attempt": 0, "interruption_logged": False},
                    "stage": "levels", "error": {"Error": "WaveFailed"}})
    assert out["decision"] == "relaunch" and out["ready"] is True and out["instance_id"] != old
    assert out["recovery"] == {"attempt": 1, "interruption_logged": False}
    events = aws.get_json("runs/r1/targets/t1-g6.json")
    assert [e["event"] for e in events] == ["launched", "interrupted", "launched"]
    assert events[1]["instance_id"] == old and events[1]["cell"] == "c|t|p"
    assert out["reset_levels"] == ["ck__short__c4__r0"]          # endpoint failure -> re-run
    remaining = [lvl["level_key"] for lvl in tc.remaining_levels("r1", levels)]
    assert remaining == ["ck__short__c4__r0", "ck__short__c8__r0", "ck__short__c16__r0"]
    # A level failing on its own on a healthy instance stays failed; the rest resume.
    aws.set_status("r1", "LEVEL#ck__short__c4__r0", "failed", error="WaveFailed")
    out2 = tc.check({"run_id": "r1", "target_id": "t1-g6", "instance_id": out["instance_id"], "private_ip": "x",
                     "cell": cell, "purchase": purchase, "recovery": out["recovery"], "stage": "levels",
                     "error": {"Error": "LevelFailed"}})
    assert out2["decision"] == "resume" and out2["recovery"]["attempt"] == 2
    assert [lvl["level_key"] for lvl in tc.remaining_levels("r1", levels)] == ["ck__short__c8__r0", "ck__short__c16__r0"]
    # mark_failed turns the workers' endpoint_down flag into the EndpointUnreachable error.
    from lambdas import waves

    boto3.resource("dynamodb", region_name="us-east-1").Table("amx-gpu-bench").update_item(
        Key={"pk": "RUN#r1", "sk": "LEVEL#ck__short__c8__r0"}, UpdateExpression="SET endpoint_down = :t",
        ExpressionAttributeValues={":t": True})
    m = waves.init_level({"action": "mark_failed", "level": {"run_id": "r1", "level_key": "ck__short__c8__r0"},
                          "error": {"Error": "WaveFailed", "Cause": "x"}})
    assert m["error"] == "EndpointUnreachable" and m["endpoint_failure"] is True
    # stop: this run's instance is terminated whatever its market; nothing is kept warm.
    res = tc.stop({"target_id": "t1-g6", "run_id": "r1"})
    assert res["terminated"] == out["instance_id"]


def test_jsonl_gz_round_trip(aws):
    aws.put_jsonl_gz("k.jsonl.gz", [{"a": 1}, {"b": 2}])
    raw = boto3.client("s3", region_name="us-east-1").get_object(Bucket=BUCKET, Key="k.jsonl.gz")["Body"].read()
    assert gzip.decompress(raw).count(b"\n") == 2
    assert aws.read_jsonl("k.jsonl.gz") == [{"a": 1}, {"b": 2}]


def test_bringup_error_keeps_the_line_that_names_the_failure():
    """2026-11-13: the first bf16-ov bring-up failed and the UI showed "graph.pbtxt has no" — a
    500-char window of stderr, later cut again to 500, lost the end of the sentence. The tail now
    drops tool chatter and is cut from the front, so the last line always survives."""
    from lambdas.target_control import error_message, stderr_tail

    stderr = "\n".join(
        [f"{i:012x}: Pulling fs layer" for i in range(40)]
        + [f"{i:012x}: Waiting" for i in range(40)] + [f"{i:012x}: Download complete" for i in range(40)]
        + ["Digest: sha256:196c999fea8b03470a5d24982c9c93fef8a8610eb5a950e60baa0b56dee82eaa",
           "Status: Downloaded newer image for 346399954218.dkr.ecr.us-east-1.amazonaws.com/x:y",
           "+ optimum-cli export openvino --model /models/hf/Qwen/Qwen2.5-3B-Instruct/aa8e --task text-generation-with-past "
           "--weight-format fp16 --trust-remote-code /models/ov/Qwen/Qwen2.5-3B-Instruct/aa8e/bf16-ov/0ec1e1dd4be9abf6/model",
           "`torch_dtype` is deprecated! Use `dtype` instead!",
           "Loading weights:   0%|          | 0/434 [00:00<?, ?it/s]\rLoading weights: 100%|██████████| 434/434 [00:00<00:00, 1095.55it/s]",
           "/usr/local/lib/python3.12/site-packages/optimum/exporters/openvino/patching_utils.py:247: TracerWarning: torch.tensor "
           "results are registered as constants in the trace.",
           "  torch.tensor(0.0, device=mask.device, dtype=dtype),",
           "+ /usr/local/bin/python /app/ovms/export_model.py text_generation --source_model /models/ov/Qwen/" + "x" * 300,
           "graph.pbtxt has no LLMCalculatorOptions block",
           "failed to run commands: exit status 1"])
    tail = stderr_tail(stderr)
    assert tail.endswith("graph.pbtxt has no LLMCalculatorOptions block\nfailed to run commands: exit status 1")
    assert len(tail) <= 400 and "Pulling fs layer" not in tail and "TracerWarning" not in tail and "it/s]" not in tail
    assert "+ /usr/local/bin/python" not in tail and "`torch_dtype` is deprecated" in tail
    # Through the state machine's error envelope and the 1000-char cap, the message stays whole.
    msg = f"bring-up of m1-qwen2.5-3b|t3-m8i-4xl-ov|bf16-ov failed (Failed): {tail}"
    cause = json.dumps({"errorMessage": msg, "errorType": "RuntimeError", "stackTrace": ["x"] * 20})
    stored = error_message({"Error": "RuntimeError", "Cause": cause})[:1000]
    assert stored == msg and stored.endswith("exit status 1")
    # Nothing but noise: fall back to the raw end so the reader still sees something.
    assert stderr_tail("a: Pulling fs layer\nb: Pull complete\n") == "a: Pulling fs layer\nb: Pull complete"
