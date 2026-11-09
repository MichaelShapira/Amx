"""Benchmark UI (design 17, Requirement 15). Local only: ui/.streamlit/config.toml binds
127.0.0.1, and every AWS call uses the operator's own credentials.

    cd ui && ../.venv/bin/streamlit run app.py
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import estimator, report  # noqa: E402
from bench.config import RunConfig, load_models, load_targets, resolve_cells  # noqa: E402
from bench.plan import levels_for_cell  # noqa: E402

REGION = "us-east-1"
st.set_page_config(page_title="AMX vs GPU benchmark", layout="wide")


@st.cache_resource
def catalogs():
    return load_models(ROOT / "config/models.yaml"), load_targets(ROOT / "config/targets.yaml")


@st.cache_data(ttl=300)
def stack_outputs() -> dict:
    import boto3

    try:
        stk = boto3.client("cloudformation", region_name=REGION).describe_stacks(StackName="AmxBench-Core")["Stacks"][0]
        return {o["OutputKey"]: o["OutputValue"] for o in stk.get("Outputs", [])}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def aws_ready() -> bool:
    out = stack_outputs()
    if "error" in out:
        st.info(f"AWS deployment not reachable with the current credentials: {out['error'][:200]}")
        return False
    return True


# ---------------------------------------------------------------------------- New run

def page_new_run():
    models, targets = catalogs()
    tmap = targets.resolved()
    st.header("New run")
    suite = st.radio("Suite", ["standard", "dspm"], horizontal=True)
    avail = [m.id for m in models.models if m.enabled and suite in m.suites]
    chosen = st.multiselect("Models", avail, default=avail[:1])
    cells = []
    st.subheader("Cells")
    for tid, t in tmap.items():
        if not t.enabled:
            continue
        with st.expander(f"{tid} ({t.instance_type}, {t.accelerator})"):
            for mid in chosen:
                m = models.get(mid)
                precs = [p for p, spec in m.precisions.items() if spec.enabled and any(e in t.engines for e in spec.engines)]
                for p in precs:
                    if st.checkbox(f"{mid} / {p}", key=f"{tid}-{mid}-{p}", value=(p == "bf16" and tid in ("t1-g6", "t2-m8i-4xl"))):
                        cells.append({"target": tid, "precision": p, "model": mid})
    c1, c2, c3 = st.columns(3)
    with c1:
        prompt_set = st.selectbox("Prompt set", ["short", "medium", "long", "custom"] if suite == "standard" else ["dspm"])
        custom = st.text_area("Custom prompt") if prompt_set == "custom" else None
        max_new = st.number_input("Max new tokens", 1, 4096, 128 if suite == "standard" else 4)
        temp = st.number_input("Temperature", 0.0, 2.0, 0.0, step=0.1)
    with c2:
        conc = st.text_input("Concurrency levels", "1, 4, 16")
        per_mode = st.radio("Per level", ["requests", "duration_s"], horizontal=True)
        per_val = st.number_input("Value", 1, 100000, 50)
        repeats = st.number_input("Repeats", 1, 10, 1)
    with c3:
        p99_ttft = st.number_input("SLA: P99 TTFT ms (0 = none)", 0, 600000, 2000)
        p50_tpot = st.number_input("SLA: P50 TPOT ms (0 = none)", 0, 10000, 150)
        p99_e2e = st.number_input("SLA: P99 E2E ms (0 = none)", 0, 600000, 0)
        preset = st.selectbox("DSPM SLA preset", ["inline_standard", "inline_strict", "bulk"]) if suite == "dspm" else None
    st.subheader("Instances")
    purchase = st.radio("Purchase option", ["on-demand", "spot"], horizontal=True,
                        help="Spot: one-time requests; an interrupted instance is replaced (Spot again, then "
                             "On-Demand once the capacity window is used up) and the cell's remaining levels are "
                             "re-run on the replacement. Cost columns always use On-Demand prices.")
    fallback = st.checkbox("Fall back to On-Demand when Spot capacity is short for 10 minutes", value=True) \
        if purchase == "spot" else True
    run = {"name": st.text_input("Run name", f"{suite}-ui"), "suite": suite,
           "mode": "full" if (per_mode == "requests" and per_val >= 200 and repeats >= 3) else "smoke",
           "account": "346399954218", "region": REGION, "az_id": "use1-az1", "models": chosen, "cells": cells,
           "prompt_set": prompt_set, "custom_prompt": custom,
           "sampling": {"temperature": temp, "max_new_tokens": int(max_new), "ignore_eos": suite == "standard", "seed": 0},
           "concurrency": [int(x) for x in conc.replace(" ", "").split(",") if x],
           "per_level": {per_mode: int(per_val)}, "repeats": int(repeats), "warmup_requests": 20,
           "quality": {"enabled": suite == "standard"},
           "sla": {k: v for k, v in {"p99_ttft_ms": p99_ttft, "p50_tpot_ms": p50_tpot, "p99_e2e_ms": p99_e2e}.items() if v}
           | {"max_error_pct": 1.0},
           "purchase": {"type": purchase, "spot_fallback_on_demand": bool(fallback)}}
    if suite == "dspm":
        run["dspm"] = {"sla_preset": preset}
    text = yaml.safe_dump({"run": {k: v for k, v in run.items() if v is not None}}, sort_keys=False)
    st.subheader("Run config (same YAML as the CLI)")
    st.code(text, language="yaml")
    try:
        rc = RunConfig.model_validate(yaml.safe_load(text)["run"])
        rcells = resolve_cells(rc, models, targets)
    except Exception as e:  # noqa: BLE001
        st.error(f"Invalid run: {e}")
        return
    from bench.pools import local_pool_sizes

    sizes = local_pool_sizes()
    plan_rows = []
    for c in rcells:
        try:
            n = len(levels_for_cell(rc, c, models.get(c.model), run_id="ui", bucket="ui", pool_sizes=sizes))
        except ValueError as e:
            n = f"missing pool ({e})"
        plan_rows.append({"cell": c.cell_id, "engine": c.engine, "levels": n})
    st.dataframe(plan_rows, width="stretch")
    if rc.is_smoke:
        st.warning("Below the methodology defaults: results will be labeled SMOKE.")
    if st.button("Estimate duration and cost") and aws_ready():
        import boto3

        from bench import cost
        from bench.estimate_run import estimate

        pr = boto3.client("pricing", region_name="us-east-1")
        prices = {t: cost.fetch_price(pr, t, REGION).usd_per_hour for t in {tmap[c.target].instance_type for c in rcells}}
        st.json(estimate(rc, rcells, models, tmap, prices, sizes))
    st.download_button("Download YAML", text, file_name=f"{run['name']}.yaml")
    if st.button("Start run", type="primary") and aws_ready():
        import boto3

        out = stack_outputs()
        run_id = f"{rc.name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        boto3.client("s3", region_name=REGION).put_object(Bucket=out["BucketName"], Key=f"runs/{run_id}/run.yaml",
                                                          Body=text.encode())
        boto3.client("stepfunctions", region_name=REGION).start_execution(
            stateMachineArn=out["RunStateMachineArn"], name=run_id, input=json.dumps({"run_id": run_id}))
        st.success(f"Started {run_id}")


# ---------------------------------------------------------------------------- Runs

def _runs() -> list[dict]:
    import boto3

    from lambdas.common import plain

    tbl = boto3.resource("dynamodb", region_name=REGION).Table("amx-gpu-bench")
    items, kw = [], {"FilterExpression": "sk = :m", "ExpressionAttributeValues": {":m": "META"}}
    while True:
        r = tbl.scan(**kw)
        items += [plain(x) for x in r["Items"]]
        if "LastEvaluatedKey" not in r:
            return sorted(items, key=lambda x: x.get("updated_at", ""), reverse=True)
        kw["ExclusiveStartKey"] = r["LastEvaluatedKey"]


def page_runs():
    st.header("Runs")
    if not aws_ready():
        return
    import boto3
    from boto3.dynamodb.conditions import Key

    from lambdas.common import plain

    runs = _runs()
    st.dataframe([{"run": r["pk"][4:], "status": r.get("status"), "updated": r.get("updated_at"),
                   "smoke": r.get("smoke"), "purchase": r.get("purchase")} for r in runs], width="stretch")
    rid = st.selectbox("Run", [r["pk"][4:] for r in runs]) if runs else None
    if not rid:
        return
    tbl = boto3.resource("dynamodb", region_name=REGION).Table("amx-gpu-bench")
    items = [plain(x) for x in tbl.query(KeyConditionExpression=Key("pk").eq(f"RUN#{rid}"))["Items"]]
    levels = [i for i in items if i["sk"].startswith("LEVEL#")]
    done = sum(1 for i in levels if i.get("status") == "done")
    st.progress(done / max(1, len(levels)), text=f"{done}/{len(levels)} levels done")
    st.dataframe([{"item": i["sk"], "status": i.get("status"), "completed": i.get("completed"), "errors": i.get("errors")}
                  for i in sorted(items, key=lambda x: x["sk"])], width="stretch")
    if st.checkbox("I want to stop this run") and st.button("Stop run", type="secondary"):
        arn = stack_outputs()["RunStateMachineArn"].replace(":stateMachine:", ":execution:")
        boto3.client("stepfunctions", region_name=REGION).stop_execution(executionArn=f"{arn}:{rid}",
                                                                         cause="stopped from the UI")
        st.success("Stopped. Targets stop through the reaper within 30 minutes, or stop them now with the CLI.")


# ---------------------------------------------------------------------------- Results

def _load_summary(rid: str) -> dict | None:
    import boto3

    try:
        body = boto3.client("s3", region_name=REGION).get_object(
            Bucket=stack_outputs()["BucketName"], Key=f"runs/{rid}/report/summary.json")["Body"].read()
        return json.loads(body)
    except Exception:  # noqa: BLE001
        return None


def page_results():
    st.header("Results")
    upload = st.file_uploader("Or open a local summary.json", type="json")
    summary = json.loads(upload.read()) if upload else None
    if summary is None and aws_ready():
        runs = [r["pk"][4:] for r in _runs() if r.get("status") == "complete"]
        rid = st.selectbox("Run", runs) if runs else None
        summary = _load_summary(rid) if rid else None
    if not summary:
        st.info("No summary loaded.")
        return
    if summary.get("smoke"):
        st.warning("SMOKE RUN: indicative numbers only.")
    rows = summary["rows"]
    st.subheader("SLA")
    c1, c2, c3, c4 = st.columns(4)
    sla = {"p99_ttft_ms": c1.number_input("P99 TTFT ms", 0.0, 1e6, 2000.0) or None,
           "p50_tpot_ms": c2.number_input("P50 TPOT ms", 0.0, 1e5, 150.0) or None,
           "p99_e2e_ms": c3.number_input("P99 E2E ms", 0.0, 1e7, 0.0) or None,
           "max_error_pct": c4.number_input("Max error %", 0.0, 100.0, 1.0)}
    if summary["run"].get("suite") == "dspm":
        preset = st.selectbox("DSPM preset", list(report.SLA_PRESETS))
        sla.update(report.SLA_PRESETS[preset])
        prov = summary.get("dspm_hashes") or {}
        st.caption(f"Dataset and policy hashes: {json.dumps(prov)}")
    for r in rows:
        r["meets_user_sla"], r["user_sla_reason"] = report.meets_sla(r, sla)
    models = sorted({r["model"] for r in rows})
    pick = st.multiselect("Models", models, default=models)
    view = [r for r in rows if r["model"] in pick]
    cols = ["model", "target", "instance_type", "engine", "precision", "workload", "conc", "batch", "e2e_p50", "e2e_p99",
            "ttft_p99", "tpot_p50", "agg_out_tok_s", "docs_s", "cls_s", "usd_per_1m_tokens", "usd_per_gb", "amx",
            "quality_delta", "meets_user_sla", "user_sla_reason"]
    st.dataframe([{k: r.get(k) for k in cols} for r in view], width="stretch")
    import pandas as pd

    df = pd.DataFrame(view)
    if not df.empty:
        metric = st.selectbox("Chart", ["e2e_p99", "e2e_p50", "ttft_p99", "agg_out_tok_s", "usd_per_1m_tokens", "docs_s",
                                        "usd_per_gb", "cls_s"])
        df["series"] = df["model"] + " " + df["target"] + " " + df["precision"]
        piv = df.pivot_table(index="conc", columns="series", values=metric, aggfunc="mean")
        st.line_chart(piv)
    st.subheader("Capacity equivalence")
    st.json(summary.get("per_model", {}))
    files = report.render(summary, with_charts=False)
    st.download_button("summary.csv", files["summary.csv"], "summary.csv")
    st.download_button("summary.md", files["summary.md"], "summary.md")
    if summary.get("warnings"):
        st.subheader("Warnings")
        for w in summary["warnings"]:
            st.write("- " + w)


# ---------------------------------------------------------------------------- Calculator

def page_calculator():
    st.header("Migration calculator")
    st.caption("Engineer-days from config/migration_effort.yaml; capacity decides the verdict, cost is shown only.")
    n = st.number_input("Models", 1, 20, 1)
    ms = []
    for i in range(int(n)):
        with st.expander(f"Model {i + 1}", expanded=i == 0):
            c1, c2, c3 = st.columns(3)
            ms.append(estimator.ModelInput(
                name=c1.text_input("Name", f"model-{i + 1}", key=f"n{i}"),
                model_type=c1.selectbox("Type", ["decoder", "encoder-decoder", "encoder-classifier", "token-classifier"],
                                        key=f"t{i}"),
                params_b=c2.number_input("Parameters (B)", 0.05, 500.0, 3.0, key=f"p{i}"),
                stack=c2.selectbox("Serving stack", ["vllm_gpu", "pytorch_transformers", "sagemaker_pytorch", "tgi",
                                                     "triton_trtllm"], key=f"s{i}"),
                cuda_deps=c3.multiselect("CUDA-only dependencies", ["flash_attention", "bitsandbytes", "custom_kernels",
                                                                     "tensorrt_engines"], key=f"d{i}"),
                precision=c3.selectbox("Target precision", ["bf16", "int8", "int4"], key=f"q{i}"),
                required_concurrency=c3.number_input("Required concurrency (0 = any)", 0, 1024, 0, key=f"c{i}") or None))
    c1, c2, c3 = st.columns(3)
    p = estimator.ProjectInput(
        models=ms, infrastructure=c1.selectbox("Infrastructure", ["ec2", "eks", "ecs", "sagemaker"]),
        iac_and_cicd_exist=c2.checkbox("IaC and CI/CD exist", True),
        familiarity=c3.selectbox("CPU-tuning familiarity", ["medium", "low", "high"]),
        sla=estimator.SlaInput(p99_ttft_ms=c1.number_input("SLA P99 TTFT ms", 0.0, 1e6, 2000.0) or None,
                               p50_tpot_ms=c2.number_input("SLA P50 TPOT ms", 0.0, 1e5, 150.0) or None))
    up = st.file_uploader("Benchmark summary.json (for the recommendation)", type="json", key="calc-summary")
    summary = json.loads(up.read()) if up else None
    out = estimator.estimate(p, summary=summary)
    st.metric("Likely engineer-days", out["likely"], help=f"range {out['low']} to {out['high']}")
    st.write(f"Low {out['low']}, likely {out['likely']}, high {out['high']} {out['units']}; overall **{out['overall']}**")
    st.dataframe(out["breakdown"], width="stretch")
    st.subheader("Risks")
    for r in out["risks"]:
        st.write("- " + r)
    st.subheader("Recommendation per model")
    st.json(out["recommendations"])


# ---------------------------------------------------------------------------- Journal

def page_journal():
    st.header("Bring-up journal")
    if not aws_ready():
        return
    import boto3
    from boto3.dynamodb.conditions import Key

    from lambdas.common import plain

    models, targets = catalogs()
    tbl = boto3.resource("dynamodb", region_name=REGION).Table("amx-gpu-bench")
    with st.form("journal"):
        c1, c2, c3 = st.columns(3)
        mid = c1.selectbox("Model", [m.id for m in models.models if m.enabled])
        tid = c2.selectbox("Target", list(targets.resolved()))
        prec = c3.text_input("Precision", "bf16")
        hours = c1.number_input("Engineer hours", 0.0, 1000.0, 2.0)
        rule = c2.selectbox("Calibrates rule", [r["id"] for r in estimator.load_config()["base_rules"]])
        predicted = c3.number_input("Predicted midpoint (days)", 0.0, 100.0, 5.0)
        issues = st.text_area("Issues hit")
        fixes = st.text_area("Fixes")
        if st.form_submit_button("Add entry"):
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            tbl.put_item(Item={"pk": f"JOURNAL#{mid}", "sk": f"{tid}#{prec}#{ts}", "hours": str(hours),
                               "rule_id": rule, "predicted_mid_days": str(predicted), "issues": issues, "fixes": fixes})
            st.success("Saved")
    entries = []
    for m in models.models:
        entries += [plain(x) for x in tbl.query(KeyConditionExpression=Key("pk").eq(f"JOURNAL#{m.id}"))["Items"]]
    st.dataframe(entries, width="stretch")
    buf = io.StringIO()
    yaml.safe_dump([{"rule_id": e["rule_id"], "actual_days": float(e["hours"]) / 8,
                     "predicted_mid_days": float(e["predicted_mid_days"]), "model": e["pk"][8:]} for e in entries], buf)
    st.download_button("Export for `bench estimate --journal`", buf.getvalue(), "journal.yaml")


PAGES = {"New run": page_new_run, "Runs": page_runs, "Results": page_results, "Migration calculator": page_calculator,
         "Bring-up journal": page_journal}
choice = st.sidebar.radio("Page", list(PAGES))
PAGES[choice]()
