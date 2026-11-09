"""Shared helpers for the benchmark Lambdas (design sections 9 and 10).

Environment (set by the CDK CoreStack): BENCH_BUCKET, BENCH_TABLE, LEVEL_STATE_MACHINE_ARN,
SSM_DOCUMENT (AWS-RunShellScript), BENCH_HOME (/opt/bench on the instances).
"""

from __future__ import annotations

import gzip
import io
import json
import os
import time
from collections.abc import Iterable
from decimal import Decimal
from functools import cache
from typing import Any

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")


def env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if v is None:
        raise RuntimeError(f"environment variable {name} is not set")
    return v


@cache
def client(name: str):
    return boto3.client(name, region_name=REGION)


@cache
def client_in(name: str, region: str):
    """Client pinned to a region (the Price List API lives in us-east-1)."""
    return boto3.client(name, region_name=region)


def bucket() -> str:
    return env("BENCH_BUCKET")


def table():
    return boto3.resource("dynamodb", region_name=REGION).Table(env("BENCH_TABLE"))


# ---------------------------------------------------------------------------- S3

def put_json(key: str, obj: Any, *, bucket_name: str | None = None) -> None:
    client("s3").put_object(Bucket=bucket_name or bucket(), Key=key,
                            Body=json.dumps(obj, indent=1, sort_keys=True, default=_default).encode(),
                            ContentType="application/json")


def get_json(key: str, *, bucket_name: str | None = None) -> Any:
    return json.loads(get_bytes(key, bucket_name=bucket_name))


def get_bytes(key: str, *, bucket_name: str | None = None) -> bytes:
    return client("s3").get_object(Bucket=bucket_name or bucket(), Key=key)["Body"].read()


def put_jsonl_gz(key: str, rows: Iterable[dict], *, bucket_name: str | None = None) -> int:
    buf = io.BytesIO()
    n = 0
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        for r in rows:
            gz.write((json.dumps(r, separators=(",", ":"), default=_default) + "\n").encode())
            n += 1
    client("s3").put_object(Bucket=bucket_name or bucket(), Key=key, Body=buf.getvalue(),
                            ContentType="application/x-ndjson", ContentEncoding="gzip")
    return n


def read_jsonl(key: str, *, bucket_name: str | None = None) -> list[dict]:
    raw = get_bytes(key, bucket_name=bucket_name)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]


def list_keys(prefix: str, *, bucket_name: str | None = None) -> list[str]:
    keys: list[str] = []
    for page in client("s3").get_paginator("list_objects_v2").paginate(Bucket=bucket_name or bucket(), Prefix=prefix):
        keys += [o["Key"] for o in page.get("Contents", [])]
    return sorted(keys)


def delete_prefix(prefix: str) -> int:
    """Delete every object under a prefix (a re-run of a level must not read the raw
    records of the attempt a Spot interruption cut short)."""
    keys = list_keys(prefix)
    for i in range(0, len(keys), 1000):
        client("s3").delete_objects(Bucket=bucket(), Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]],
                                                             "Quiet": True})
    return len(keys)


def exists(key: str) -> bool:
    try:
        client("s3").head_object(Bucket=bucket(), Key=key)
        return True
    except client("s3").exceptions.ClientError:
        return False


def _default(o: Any):
    if isinstance(o, Decimal):
        return int(o) if o == o.to_integral_value() else float(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


# ---------------------------------------------------------------------------- run paths

def run_prefix(run_id: str) -> str:
    return f"runs/{run_id}"


def level_raw_prefix(run_id: str, cell_key: str, level_key: str) -> str:
    return f"runs/{run_id}/raw/{cell_key}/{level_key}"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------- SSM

def ssm_run(instance_id: str, commands: list[str], *, timeout_s: int = 3600, comment: str = "bench") -> dict:
    """Run shell commands on an instance through SSM and wait for the result."""
    ssm = client("ssm")
    resp = ssm.send_command(InstanceIds=[instance_id], DocumentName=env("SSM_DOCUMENT", "AWS-RunShellScript"),
                            Comment=comment[:100], TimeoutSeconds=min(timeout_s, 172800),
                            Parameters={"commands": commands, "executionTimeout": [str(timeout_s)]})
    cid = resp["Command"]["CommandId"]
    deadline = time.time() + timeout_s + 60
    while True:
        time.sleep(3)
        try:
            inv = ssm.get_command_invocation(CommandId=cid, InstanceId=instance_id)
        except ssm.exceptions.InvocationDoesNotExist:
            continue
        status = inv["Status"]
        if status in ("Success", "Failed", "Cancelled", "TimedOut", "Cancelling"):
            return {"status": status, "stdout": inv.get("StandardOutputContent", ""),
                    "stderr": inv.get("StandardErrorContent", ""), "command_id": cid}
        if time.time() > deadline:
            return {"status": "TimedOut", "stdout": "", "stderr": "wait deadline", "command_id": cid}


def ssm_start(instance_id: str, commands: list[str], *, timeout_s: int = 3600, comment: str = "bench") -> str:
    """Start an SSM command without waiting; returns the command id (long bring-ups)."""
    resp = client("ssm").send_command(InstanceIds=[instance_id], DocumentName=env("SSM_DOCUMENT", "AWS-RunShellScript"),
                                      Comment=comment[:100], TimeoutSeconds=min(timeout_s, 172800),
                                      Parameters={"commands": commands, "executionTimeout": [str(timeout_s)]})
    return resp["Command"]["CommandId"]


def ssm_poll(instance_id: str, command_id: str) -> dict:
    inv = client("ssm").get_command_invocation(CommandId=command_id, InstanceId=instance_id)
    return {"status": inv["Status"], "stdout": inv.get("StandardOutputContent", ""),
            "stderr": inv.get("StandardErrorContent", "")}


def bench_home() -> str:
    return env("BENCH_HOME", "/opt/bench")


# ---------------------------------------------------------------------------- DynamoDB

def touch_target(target_id: str, run_id: str | None = None) -> None:
    expr = "SET last_activity_at = :t"
    vals: dict[str, Any] = {":t": int(time.time())}
    if run_id is not None:
        expr += ", active_run_id = :r"
        vals[":r"] = run_id
    table().update_item(Key={"pk": f"TARGET#{target_id}", "sk": "ACTIVITY"}, UpdateExpression=expr,
                        ExpressionAttributeValues=vals)


def set_status(run_id: str, sk: str, status: str, **extra: Any) -> None:
    names = {"#s": "status", "#u": "updated_at"}
    vals: dict[str, Any] = {":s": status, ":u": now_iso()}
    sets = ["#s = :s", "#u = :u"]
    for i, (k, v) in enumerate(extra.items()):
        names[f"#k{i}"] = k
        vals[f":v{i}"] = _ddb(v)
        sets.append(f"#k{i} = :v{i}")
    table().update_item(Key={"pk": f"RUN#{run_id}", "sk": sk}, UpdateExpression="SET " + ", ".join(sets),
                        ExpressionAttributeNames=names, ExpressionAttributeValues=vals)


def plain(v: Any) -> Any:
    """DynamoDB items to plain Python: Decimal to int or float, recursively."""
    if isinstance(v, Decimal):
        return int(v) if v == v.to_integral_value() else float(v)
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, list):
        return [plain(x) for x in v]
    if isinstance(v, set):
        return sorted(plain(x) for x in v)
    return v


def _ddb(v: Any) -> Any:
    """DynamoDB wants Decimal, not float."""
    if isinstance(v, float):
        return Decimal(str(v))
    if isinstance(v, dict):
        return {k: _ddb(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_ddb(x) for x in v]
    return v
