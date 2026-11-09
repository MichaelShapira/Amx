"""CDK app (design section 3).

    cd infra && cdk synth                     # all stacks
    cdk deploy AmxBench-Network AmxBench-Core AmxBench-Target-t1-g6 AmxBench-Target-t2-m8i-4xl
    cdk deploy AmxBench-Target-t1-g6 -c type_t1-g6=g5.xlarge     # capacity fallback

Context: az_name (default: resolved from az_id use1-az1 at synth), skip_bundling,
build_images (default true), type_<target-id> (instance type override).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import aws_cdk as cdk  # noqa: E402

from bench.config import load_targets  # noqa: E402
from infra.stacks.core import CoreStack  # noqa: E402
from infra.stacks.network import NetworkStack  # noqa: E402
from infra.stacks.target import TargetStack  # noqa: E402

ACCOUNT, REGION, AZ_ID = "346399954218", "us-east-1", "use1-az1"
PROJECT_TAGS = {"project": "amx-gpu-bench", "owner": "operator", "purpose": "cpu-amx-vs-gpu-benchmark"}


def resolve_az_name(app: cdk.App) -> str:
    name = app.node.try_get_context("az_name")
    if name:
        return name
    try:
        import boto3

        zones = boto3.client("ec2", region_name=REGION).describe_availability_zones(ZoneIds=[AZ_ID])
        return zones["AvailabilityZones"][0]["ZoneName"]
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"cannot map {AZ_ID} to an AZ name ({e}); pass -c az_name=us-east-1x") from e


def harness_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build(app: cdk.App) -> dict:
    env = cdk.Environment(account=ACCOUNT, region=REGION)
    flag = lambda k, d: str(app.node.try_get_context(k) if app.node.try_get_context(k) is not None else d).lower() \
        in ("1", "true", "yes")  # noqa: E731
    net = NetworkStack(app, "AmxBench-Network", az_name=resolve_az_name(app), env=env)
    core = CoreStack(app, "AmxBench-Core", network=net, env=env, skip_bundling=flag("skip_bundling", False),
                     build_images=flag("build_images", True), harness_commit=harness_commit())
    stacks = {"network": net, "core": core}
    for tid, t in load_targets(ROOT / "config/targets.yaml").resolved().items():
        if not t.enabled:
            continue
        stacks[tid] = TargetStack(app, f"AmxBench-Target-{tid}", target=t, network=net, core=core, env=env,
                                  instance_type=app.node.try_get_context(f"type_{tid}"))
    for k, v in PROJECT_TAGS.items():
        cdk.Tags.of(app).add(k, v)
    return stacks


if __name__ == "__main__":
    a = cdk.App()
    build(a)
    a.synth()
