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

from bench import pairing
from bench.config import load_models, load_run, load_targets, resolve_cells, run_instance_types

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


def _apply_overrides(run, args):
    """--purchase and --instance-type TARGET=TYPE (repeatable) override the YAML for this invocation."""
    upd: dict = {}
    if getattr(args, "purchase", None):
        upd["purchase"] = run.purchase.model_copy(update={"type": args.purchase})
    if getattr(args, "instance_type", None):
        types = dict(run.instance_types)
        for item in args.instance_type:
            tid, _, itype = item.partition("=")
            if not tid or not itype:
                raise SystemExit(f"--instance-type expects TARGET=TYPE, got {item!r}")
            types[tid] = itype
        upd["instance_types"] = types
    return run.model_copy(update=upd) if upd else run


def cmd_pair(args) -> int:
    """Fair counterparts for an instance type (CPU -> GPU proposals, or GPU -> CPU)."""
    s = pairing.spec(args.instance_type)
    if s is None:
        print(f"{args.instance_type} is not in config/instance_catalog.json")
        return 1
    want = "gpu" if s["accelerator"] == "cpu" else "cpu"
    print(f"{args.instance_type}: {s['vcpus']} vCPU, {s['memory_gib']} GiB"
          + (f", {s['gpus']}x {s['gpu_model']}" if s["gpus"] else "") + f", ${s['usd_per_hour']:.4f}/h On-Demand")
    print(f"fair {want.upper()} counterparts (GPU/CPU price ratio within {pairing.FAIR_RATIO[0]}x-{pairing.FAIR_RATIO[1]}x; "
          "single-GPU first, then closest price):")
    for p in pairing.propose(args.instance_type, want, n=args.n):
        gpu = f"{p['gpus']}x {p['gpu_model']}" if want == "gpu" else "AMX"
        print(f"  {'fair  ' if p['fair'] else 'unfair'} {p['instance_type']:14s} {p['vcpus']:4d} vCPU {p['memory_gib']:7.1f} GiB "
              f"{gpu:10s} ${p['usd_per_hour']:.4f}/h  ratio {p['ratio']:.2f}x")
    return 0


def cmd_validate(args) -> int:
    from bench.plan import levels_for_cell
    from bench.pools import local_pool_sizes

    run = _apply_overrides(load_run(args.run), args)
    models, targets = load_models(ROOT / "config/models.yaml"), load_targets(ROOT / "config/targets.yaml")
    cells = resolve_cells(run, models, targets)
    tmap = targets.resolved()
    sizes = local_pool_sizes()
    problems = 0
    print(f"run {run.name}: suite {run.suite}, mode {run.mode}{' (labeled SMOKE)' if run.is_smoke else ''}, "
          f"instances {run.purchase.type}")
    rtypes = run_instance_types(run, cells, tmap)
    for tid, t in rtypes.items():
        print(f"  target {tid:16s} -> {t}{'  (run override)' if tid in run.instance_types else ''}")
    for j in pairing.judge_run(rtypes, {t: tmap[t].accelerator for t in rtypes}):
        print(f"  pairing {'OK     ' if j['fair'] else 'UNFAIR '} {j['reason']}")
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

    run = _apply_overrides(load_run(args.run), args)
    models, targets = load_models(ROOT / "config/models.yaml"), load_targets(ROOT / "config/targets.yaml")
    tmap = targets.resolved()
    cells = resolve_cells(run, models, targets)
    rtypes = run_instance_types(run, cells, tmap)
    pr = boto3.client("pricing", region_name="us-east-1")
    prices = {t: cost.fetch_price(pr, t, run.region).usd_per_hour for t in sorted(set(rtypes.values()))}
    out = estimate(run, cells, models, tmap, prices, local_pool_sizes(), instance_types=rtypes)
    out["prices_usd_per_hour"] = prices
    out["pairing"] = pairing.judge_run(rtypes, {t: tmap[t].accelerator for t in rtypes})
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
    man = pools.manifest(std)
    local = ROOT / "prompts/generated/manifest.json"
    if local.exists():        # the dataset fingerprint the translation pools were rendered from (preflight checks it)
        man.update({k: v for k, v in json.loads(local.read_text()).items() if k == "translation"})
    s3.put_object(Bucket=bucket, Key="artifacts/prompts/manifest.json", Body=json.dumps(man).encode())
    n = len(std)
    tds = ROOT / "data/translation/v1"
    if (tds / "manifest.json").exists():
        for f in ("items.jsonl.gz", "manifest.json"):
            s3.upload_file(str(tds / f), bucket, f"datasets/translation/v1/{f}")
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

    run = _apply_overrides(load_run(args.run), args)
    models, targets = load_models(ROOT / "config/models.yaml"), load_targets(ROOT / "config/targets.yaml")
    cells = resolve_cells(run, models, targets)
    tmap = targets.resolved()
    rtypes = run_instance_types(run, cells, tmap)
    unfair = [j for j in pairing.judge_run(rtypes, {t: tmap[t].accelerator for t in rtypes}) if j["fair"] is False]
    if unfair and not args.allow_unfair:
        for j in unfair:
            print(f"UNFAIR pairing: {j['reason']}")
        print("Refusing to start. Pick comparable instance types (see `bench pair <type>`) or pass --allow-unfair.")
        return 2
    # The uploaded run.yaml is what preflight validates, so it carries the CLI overrides.
    body = yaml.safe_dump({"run": run.model_dump(mode="json", exclude_defaults=True)}, sort_keys=False).encode()
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


def cmd_rejudge(args) -> int:
    """Re-parse the AMX traces of a finished run with the current parser, rewrite its evidence.json
    files and re-render the report (the AggregateRun Lambda). Needed once for the runs whose W8A8
    cells were INVALID because the brgemm micro-kernel lines were not recognized (2026-11-19)."""
    import gzip

    import boto3

    from bench import amx

    bucket = _resolve(args, "BucketName", "bucket")
    s3 = boto3.client("s3", region_name=REGION)
    prefix = f"runs/{args.run_id}/amx/"
    changed = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            if not o["Key"].endswith("/trace.log.gz"):
                continue
            cell = o["Key"][len(prefix):].split("/")[0]
            ev_key = f"{prefix}{cell}/evidence.json"
            ev = json.loads(s3.get_object(Bucket=bucket, Key=ev_key)["Body"].read())
            text = gzip.decompress(s3.get_object(Bucket=bucket, Key=o["Key"])["Body"].read()).decode(errors="replace")
            before = ev.get("trace") or {}
            ev["trace"] = amx.parse_onednn_verbose(text, set(ev.get("flags") or []))
            if ev["trace"] != before:
                s3.put_object(Bucket=bucket, Key=ev_key, Body=json.dumps(ev, indent=1, sort_keys=True).encode(),
                              ContentType="application/json")
                changed.append((cell, before.get("gemm_amx"), before.get("gemm_total"), ev["trace"]["gemm_amx"], ev["trace"]["gemm_total"]))
    for c in changed:
        print(f"{c[0]}: AMX GEMMs {c[1]}/{c[2]} -> {c[3]}/{c[4]}")
    if not changed:
        print("no evidence changed")
    if args.render:
        lam = boto3.client("lambda", region_name=REGION)
        r = lam.invoke(FunctionName="amx-gpu-bench-aggregate-run", Payload=json.dumps({"run_id": args.run_id}).encode())
        body = r["Payload"].read().decode()
        print("report re-rendered:" if r.get("FunctionError") is None else "re-render FAILED:", body[:400])
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
        p.add_argument("--purchase", choices=["spot", "on-demand"],
                       help="launch the target instances Spot (default) or On-Demand (overrides run.purchase.type)")
        p.add_argument("--instance-type", dest="instance_type", action="append", metavar="TARGET=TYPE",
                       help="instance type for a target in this run, e.g. t1-g6=g6.2xlarge (repeatable)")
        if name == "run":
            p.add_argument("--allow-unfair", action="store_true",
                           help="start even when a CPU/GPU pair is outside the fair price band")
    pp = sub.add_parser("pair", help="propose comparable GPU types for a CPU instance type (or the reverse)")
    pp.add_argument("instance_type")
    pp.add_argument("-n", type=int, default=5)
    up = sub.add_parser("upload-artifacts")
    up.add_argument("--bucket")
    st = sub.add_parser("status")
    st.add_argument("run_id")
    st.add_argument("--table", default="amx-gpu-bench")
    rp = sub.add_parser("report")
    rp.add_argument("run_id")
    rp.add_argument("--out", default="reports")
    rp.add_argument("--bucket")
    rj = sub.add_parser("rejudge-amx", help="re-parse a finished run's AMX traces and re-render its report")
    rj.add_argument("run_id")
    rj.add_argument("--bucket")
    rj.add_argument("--render", action="store_true", help="invoke the AggregateRun Lambda afterwards")
    sp = sub.add_parser("stop")
    sp.add_argument("run_id")
    sp.add_argument("--yes", action="store_true")
    sp.add_argument("--state-machine", dest="state_machine")
    es = sub.add_parser("estimate")
    es.add_argument("project")
    es.add_argument("--summary")
    es.add_argument("--journal")
    args = ap.parse_args(argv)
    return {"validate": cmd_validate, "plan": cmd_plan, "upload-artifacts": cmd_upload, "run": cmd_run, "pair": cmd_pair,
            "rejudge-amx": cmd_rejudge,
            "status": cmd_status, "report": cmd_report, "stop": cmd_stop, "estimate": cmd_estimate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
