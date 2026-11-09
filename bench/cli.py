"""Command-line interface (design 17, Requirement 6.1, 15.4). Same YAML as the UI.

    bench validate config/runs/smoke-m1.yaml
    bench plan config/runs/smoke-m1.yaml                 # duration and cost estimate
    bench upload-artifacts                                # prompt sets, DSPM pools and dataset to S3
    bench run config/runs/smoke-m1.yaml                   # uploads run.yaml, starts the run state machine
    bench status <run-id>
    bench report <run-id> --out reports/<run-id>
    bench stop <run-id> --yes
    bench estimate project.yaml [--summary summary.json] [--journal journal.yaml]

AWS calls use the operator's credentials; deployment names come from the AmxBench-Core
stack outputs unless --bucket / --state-machine are given.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from bench.config import load_models, load_run, load_targets, resolve_cells

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:      # `bench status` reads lambdas.common from the repo checkout
    sys.path.insert(0, str(ROOT))
REGION = "us-east-1"
CORE_STACK = "AmxBench-Core"


def _outputs() -> dict[str, str]:
    import boto3

    st = boto3.client("cloudformation", region_name=REGION).describe_stacks(StackName=CORE_STACK)["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in st.get("Outputs", [])}


def _resolve(args, key: str, attr: str) -> str:
    v = getattr(args, attr, None)
    return v or _outputs()[key]


def cmd_validate(args) -> int:
    from bench.plan import levels_for_cell
    from bench.pools import local_pool_sizes

    run = load_run(args.run)
    models, targets = load_models(ROOT / "config/models.yaml"), load_targets(ROOT / "config/targets.yaml")
    cells = resolve_cells(run, models, targets)
    sizes = local_pool_sizes()
    problems = 0
    print(f"run {run.name}: suite {run.suite}, mode {run.mode}{' (labeled SMOKE)' if run.is_smoke else ''}")
    for c in cells:
        try:
            n = 0 if c.quality_only else len(levels_for_cell(run, c, models.get(c.model), run_id="validate",
                                                             bucket="local", pool_sizes=sizes))
            note = " (quality anchor only)" if c.quality_only else ""
            print(f"  {c.cell_id:55s} engine {c.engine:14s} levels {n}{note}")
        except ValueError as e:
            problems += 1
            print(f"  {c.cell_id:55s} ERROR {e}")
    return 1 if problems else 0


def cmd_plan(args) -> int:
    import boto3

    from bench import cost
    from bench.estimate_run import estimate
    from bench.pools import local_pool_sizes

    run = load_run(args.run)
    models, targets = load_models(ROOT / "config/models.yaml"), load_targets(ROOT / "config/targets.yaml")
    tmap = targets.resolved()
    cells = resolve_cells(run, models, targets)
    types = sorted({tmap[c.target].instance_type for c in cells})
    pr = boto3.client("pricing", region_name="us-east-1")
    prices = {t: cost.fetch_price(pr, t, run.region).usd_per_hour for t in types}
    out = estimate(run, cells, models, tmap, prices, local_pool_sizes())
    out["prices_usd_per_hour"] = prices
    print(json.dumps(out, indent=2))
    return 0


def cmd_upload(args) -> int:
    import boto3

    from bench import pools

    bucket = _resolve(args, "BucketName", "bucket")
    s3 = boto3.client("s3", region_name=REGION)
    std = pools.standard_pools()
    if not std:
        raise SystemExit("no prompt sets in prompts/generated/; run tools/build_prompts.py first")
    for key, body in std.items():
        s3.put_object(Bucket=bucket, Key=key, Body=body)
    s3.put_object(Bucket=bucket, Key="artifacts/prompts/manifest.json", Body=json.dumps(pools.manifest(std)).encode())
    n = len(std)
    ds = ROOT / "data/dspm/v1"
    if (ds / "requests/manifest.json").exists():
        dp = pools.dspm_pools(ds / "requests")
        for key, body in dp.items():
            s3.put_object(Bucket=bucket, Key=key, Body=body)
        s3.upload_file(str(ds / "requests/manifest.json"), bucket, "artifacts/dspm/v1/requests/manifest.json")
        for f in ("dev.jsonl.gz", "test.jsonl.gz", "manifest.json"):
            s3.upload_file(str(ds / f), bucket, f"datasets/dspm/v1/{f}")
        n += len(dp)
    print(f"uploaded {n} pools to s3://{bucket}/artifacts/")
    return 0


def cmd_run(args) -> int:
    import boto3

    body = Path(args.run).read_bytes()
    if getattr(args, "purchase", None):            # --purchase spot|on-demand overrides the YAML
        data = yaml.safe_load(body)
        data["run"]["purchase"] = {**data["run"].get("purchase", {}), "type": args.purchase}
        body = yaml.safe_dump(data, sort_keys=False).encode()
    run = load_run(args.run)
    run = run.model_copy(update={"purchase": run.purchase.model_copy(update={"type": args.purchase})}) \
        if getattr(args, "purchase", None) else run
    resolve_cells(run, load_models(ROOT / "config/models.yaml"), load_targets(ROOT / "config/targets.yaml"))
    bucket = _resolve(args, "BucketName", "bucket")
    arn = _resolve(args, "RunStateMachineArn", "state_machine")
    run_id = f"{run.name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    boto3.client("s3", region_name=REGION).put_object(Bucket=bucket, Key=f"runs/{run_id}/run.yaml", Body=body)
    ex = boto3.client("stepfunctions", region_name=REGION).start_execution(
        stateMachineArn=arn, name=run_id, input=json.dumps({"run_id": run_id}))
    print(json.dumps({"run_id": run_id, "purchase": run.purchase.type, "execution_arn": ex["executionArn"]}, indent=2))
    return 0


def cmd_status(args) -> int:
    import boto3
    from boto3.dynamodb.conditions import Key

    from lambdas.common import plain

    table = boto3.resource("dynamodb", region_name=REGION).Table(args.table)
    items = table.query(KeyConditionExpression=Key("pk").eq(f"RUN#{args.run_id}"))["Items"]
    for it in sorted(items, key=lambda x: x["sk"]):
        it = plain(it)
        extra = {k: it[k] for k in ("completed", "errors", "waves", "n_ok") if k in it}
        print(f"{it['sk']:90s} {it.get('status', '-'):14s} {json.dumps(extra) if extra else ''}")
    return 0


def cmd_report(args) -> int:
    import boto3

    bucket = _resolve(args, "BucketName", "bucket")
    s3 = boto3.client("s3", region_name=REGION)
    prefix = f"runs/{args.run_id}/report/"
    out = Path(args.out)
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            dest = out / o["Key"][len(prefix):]
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, o["Key"], str(dest))
            n += 1
    print(f"downloaded {n} files to {out}")
    return 0 if n else 1


def cmd_stop(args) -> int:
    import boto3

    if not args.yes:
        print("refusing to stop without --yes (this ends the run; finished levels are kept)")
        return 2
    arn = _resolve(args, "RunStateMachineArn", "state_machine").replace(":stateMachine:", ":execution:")
    boto3.client("stepfunctions", region_name=REGION).stop_execution(executionArn=f"{arn}:{args.run_id}",
                                                                     cause="stopped by operator")
    print(f"stopped {args.run_id}; targets stop through the reaper or `aws ec2 stop-instances`")
    return 0


def cmd_estimate(args) -> int:
    from bench import estimator

    project = estimator.ProjectInput.model_validate(yaml.safe_load(Path(args.project).read_text()))
    summary = json.loads(Path(args.summary).read_text()) if args.summary else None
    journal = [estimator.JournalEntry.model_validate(x)
               for x in (yaml.safe_load(Path(args.journal).read_text()) or [])] if args.journal else []
    print(json.dumps(estimator.estimate(project, summary=summary, journal=journal), indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("run")
        p.add_argument("--bucket")
        p.add_argument("--state-machine", dest="state_machine")
        if name == "run":
            p.add_argument("--purchase", choices=["on-demand", "spot"],
                           help="launch the target instances On-Demand or Spot (overrides run.purchase.type)")
    up = sub.add_parser("upload-artifacts")
    up.add_argument("--bucket")
    st = sub.add_parser("status")
    st.add_argument("run_id")
    st.add_argument("--table", default="amx-gpu-bench")
    rp = sub.add_parser("report")
    rp.add_argument("run_id")
    rp.add_argument("--out", default="reports")
    rp.add_argument("--bucket")
    sp = sub.add_parser("stop")
    sp.add_argument("run_id")
    sp.add_argument("--yes", action="store_true")
    sp.add_argument("--state-machine", dest="state_machine")
    es = sub.add_parser("estimate")
    es.add_argument("project")
    es.add_argument("--summary")
    es.add_argument("--journal")
    args = ap.parse_args(argv)
    return {"validate": cmd_validate, "plan": cmd_plan, "upload-artifacts": cmd_upload, "run": cmd_run,
            "status": cmd_status, "report": cmd_report, "stop": cmd_stop, "estimate": cmd_estimate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
