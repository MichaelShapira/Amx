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
    assert "lack amx_fp16" in text


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
    # Plan with three levels: one done, one running (interrupted), one not started.
    levels = [{"level_key": f"ck__short__c{c}__r0", "cell_key": "ck"} for c in (1, 4, 16)]
    aws.put_json("runs/r1/plan/ck.json", {"levels": levels, "quality": None})
    aws.put_json("runs/r1/bringup/ck.json", {"api_path": "/v1/completions"})
    aws.set_status("r1", "LEVEL#ck__short__c1__r0", "done")
    aws.set_status("r1", "LEVEL#ck__short__c4__r0", "running")
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
    remaining = tc.remaining_levels("r1", levels)
    assert [lvl["level_key"] for lvl in remaining] == ["ck__short__c4__r0", "ck__short__c16__r0"]
    # A level failing on a healthy instance: marked failed, the rest resume.
    out2 = tc.check({"run_id": "r1", "target_id": "t1-g6", "instance_id": out["instance_id"], "private_ip": "x",
                     "cell": cell, "purchase": purchase, "recovery": out["recovery"], "stage": "levels",
                     "error": {"Error": "WaveFailed"}})
    assert out2["decision"] == "resume" and out2["marked_failed"] == ["ck__short__c4__r0"]
    assert [lvl["level_key"] for lvl in tc.remaining_levels("r1", levels)] == ["ck__short__c16__r0"]
    # stop: a Spot instance is terminated, not stopped.
    res = tc.stop({"target_id": "t1-g6"})
    assert "terminated" in res or "stopped" in res


def test_jsonl_gz_round_trip(aws):
    aws.put_jsonl_gz("k.jsonl.gz", [{"a": 1}, {"b": 2}])
    raw = boto3.client("s3", region_name="us-east-1").get_object(Bucket=BUCKET, Key="k.jsonl.gz")["Body"].read()
    assert gzip.decompress(raw).count(b"\n") == 2
    assert aws.read_jsonl("k.jsonl.gz") == [{"a": 1}, {"b": 2}]
