"""Refresh config/instance_catalog.json: specs (vCPUs, memory, GPUs, AMX) from EC2
DescribeInstanceTypes and On-Demand prices from the Price List API, for the instance
families the benchmark may run on. The UI, CLI and preflight read this file offline; the
price snapshot date is recorded so a stale file is visible.

    .venv/bin/python tools/refresh_instance_catalog.py            # us-east-1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import cost  # noqa: E402

# CPU families with Intel AMX (Sapphire Rapids = 7i, Granite Rapids = 8i) and the GPU families
# with one NVIDIA GPU at the small end. Anything outside this list is refused by preflight.
CPU_FAMILIES = ("m8i", "c8i", "r8i", "m7i", "c7i", "r7i")
GPU_FAMILIES = ("g6", "g5", "g6e")
OUT = ROOT / "config/instance_catalog.json"


def describe(ec2, families: tuple[str, ...]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    pages = ec2.get_paginator("describe_instance_types").paginate(
        Filters=[{"Name": "instance-type", "Values": [f"{f}.*" for f in families]}])
    for page in pages:
        for it in page["InstanceTypes"]:
            gpus = (it.get("GpuInfo") or {}).get("Gpus") or []
            family = it["InstanceType"].split(".")[0]
            out[it["InstanceType"]] = {
                "family": family,
                "vcpus": it["VCpuInfo"]["DefaultVCpus"],
                "memory_gib": round(it["MemoryInfo"]["SizeInMiB"] / 1024, 1),
                "accelerator": "gpu" if gpus else "cpu",
                "gpus": sum(g["Count"] for g in gpus),
                "gpu_model": gpus[0]["Name"] if gpus else None,
                "gpu_memory_gib": round(gpus[0]["MemoryInfo"]["SizeInMiB"] * gpus[0]["Count"] / 1024, 1) if gpus else None,
                # DescribeInstanceTypes does not expose AMX; every family in CPU_FAMILIES is
                # Sapphire Rapids or Granite Rapids, which have it (verified on m8i by lscpu).
                "amx": family in CPU_FAMILIES,
                "bare_metal": bool(it.get("BareMetal")),
                "current_generation": bool(it.get("CurrentGeneration", True)),
            }
    return out


def main(argv=None) -> int:
    import boto3

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    ec2 = boto3.client("ec2", region_name=a.region)
    specs = describe(ec2, CPU_FAMILIES + GPU_FAMILIES)
    specs = {k: v for k, v in specs.items() if not v["bare_metal"]}
    pr = boto3.client("pricing", region_name="us-east-1")
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    arn = cost.current_price_list_arn(pr, a.region, now)
    missing = []
    for t in sorted(specs):
        try:
            p = cost.fetch_price(pr, t, a.region, arn)
            specs[t]["usd_per_hour"] = p.usd_per_hour
        except cost.PriceUnavailable:
            missing.append(t)
            specs[t]["usd_per_hour"] = None
    doc = {"region": a.region, "price_snapshot": now, "price_list_arn": arn, "cpu_families": list(CPU_FAMILIES),
           "gpu_families": list(GPU_FAMILIES), "instances": dict(sorted(specs.items()))}
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"instances": len(specs), "without_price": missing, "out": a.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
