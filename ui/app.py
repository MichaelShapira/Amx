"""Benchmark UI (design 17, Requirement 15). Local only: ui/.streamlit/config.toml binds
127.0.0.1, and every AWS call uses the operator's own credentials.

    cd ui && ../.venv/bin/streamlit run app.py

Written for a product or project manager choosing between GPU and AMX CPU instances: every
control carries a help icon (texts in ui/labels.py), the hardware options and models have
plain names, and the run definition is reviewed in words before it starts. The YAML the CLI
uses is available under "Advanced".
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

import labels as L  # noqa: E402

from bench import estimator, pairing, report  # noqa: E402
from bench.config import RunConfig, load_models, load_targets, resolve_cells, run_instance_types  # noqa: E402
from bench.plan import levels_for_cell  # noqa: E402

REGION = "us-east-1"
ACCOUNT = "346399954218"
st.set_page_config(page_title="GPU vs AMX CPU benchmark", layout="wide")


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
        st.info(f"The AWS deployment is not reachable with the current credentials: {out['error'][:200]}")
        return False
    return True


# ---------------------------------------------------------------------------- naming helpers

def model_label(mid: str, models) -> str:
    name, _ = L.MODELS.get(mid, (mid, ""))
    m = models.get(mid)
    return f"{name} · {m.nominal_size_b:g}B"


def model_help_text(models, ids: list[str]) -> str:
    lines = [L.MODELS_HELP, ""]
    for mid in ids:
        name, desc = L.MODELS.get(mid, (mid, ""))
        lines.append(f"- **{name}** ({mid}): {desc}")
    return "\n".join(lines)


def hardware_label(tid: str, t, instance_type: str) -> str:
    s = pairing.spec(instance_type) or {}
    if t.accelerator == "gpu":
        hw = f"GPU · NVIDIA {s.get('gpu_model', '?')} ({instance_type})"
    else:
        hw = f"CPU · Intel Xeon with AMX, {s.get('vcpus', '?')} vCPUs ({instance_type})"
    sw = " + ".join(dict.fromkeys(L.ENGINES.get(e, e) for e in t.engines))
    return f"{hw} · {sw}"


def fmt_type(t: str) -> str:
    s = pairing.spec(t) or {}
    gpu = f", {s['gpus']}x {s['gpu_model']}" if s.get("gpus") else ""
    return f"{t}  ·  {s.get('vcpus')} vCPU, {s.get('memory_gib')} GiB{gpu}  ·  ${s.get('usd_per_hour', 0):.2f}/hour"


def prec_label(p: str) -> str:
    return L.PRECISIONS.get(p, (p, ""))[0]


def hw_help_text(t) -> str:
    return "\n".join([L.HARDWARE_HELP, ""] + [f"- {L.ENGINE_HELP.get(e, e)}" for e in t.engines])


# ---------------------------------------------------------------------------- New run

def fairness_panel(hw_types: dict[str, str], tmap) -> None:
    """Every selected CPU option against every selected GPU option, in words."""
    cpus = [t for t in hw_types if tmap[t].accelerator == "cpu"]
    gpus = [t for t in hw_types if tmap[t].accelerator == "gpu"]
    st.markdown("**Is this a fair comparison?**", help=L.FAIRNESS_HELP)
    if not cpus or not gpus:
        st.info("Tick at least one GPU option and one CPU option to get a GPU-versus-CPU verdict.")
        return
    acc = {t: tmap[t].accelerator for t in hw_types}
    for j in pairing.judge_run(hw_types, acc):
        head = "Fair comparison" if j["fair"] else "Not a fair comparison"
        (st.success if j["fair"] else st.warning)(f"{head} — {j['cpu']} versus {j['gpu']}: {j['reason']}.")
        if not j["fair"]:
            props = pairing.propose(j["cpu"], "gpu")
            st.caption("GPU sizes closest in price to " + j["cpu"] + ": " + ", ".join(
                f"{p['instance_type']} (${p['usd_per_hour']:.2f}/h, {p['ratio']:.2f}x)" for p in props))


def page_new_run():
    models, targets = catalogs()
    tmap = {tid: t for tid, t in targets.resolved().items() if t.enabled}
    st.header("Set up a benchmark run")
    st.caption(L.ACCOUNT_NOTE)

    # 1. Workload
    st.subheader("1. What do you want to size hardware for?")
    suite = st.radio("Workload type", list(L.SUITES), format_func=lambda s: L.SUITES[s][0], horizontal=True,
                     help=L.SUITE_HELP, label_visibility="collapsed")
    st.caption(L.SUITES[suite][1])

    # 2. Models
    st.subheader("2. Which models?")
    avail = [m.id for m in models.models if m.enabled and suite in m.suites]
    chosen = st.multiselect("Models", avail, default=avail[:1], format_func=lambda m: model_label(m, models),
                            help=model_help_text(models, avail))
    if not chosen:
        st.warning("Pick at least one model.")
        return

    # 3. Hardware options
    st.subheader("3. Which hardware options to compare?")
    st.caption("Tick the options to include. Each becomes one test instance that runs every model you picked; "
               "the GPU option is the baseline the CPU options are measured against.")
    hw_types: dict[str, str] = {}
    chosen_precisions: dict[str, dict[str, list[str]]] = {}      # target -> model -> precisions
    default_on = {"t1-g6", "t2-m8i-4xl"}
    for tid, t in tmap.items():
        options = pairing.types_for(t.accelerator)
        default_type = t.instance_type if t.instance_type in options else options[0]
        current = st.session_state.get(f"itype-{tid}", default_type)
        on = st.checkbox(hardware_label(tid, t, current), value=tid in default_on, key=f"hw-{tid}",
                         help=hw_help_text(t))
        if not on:
            continue
        c1, c2 = st.columns([1, 1])
        with c1:
            sel = st.selectbox("Instance size", options, index=options.index(current) if current in options else 0,
                               key=f"itype-{tid}", format_func=fmt_type, help=L.INSTANCE_TYPE_HELP)
        hw_types[tid] = sel
        with c2:
            per_model: dict[str, list[str]] = {}
            for mid in chosen:
                m = models.get(mid)
                precs = [p for p, spec in m.precisions.items()
                         if spec.enabled and any(e in t.engines for e in spec.engines)]
                if not precs:
                    continue
                default = ["bf16"] if "bf16" in precs else precs[:1]
                per_model[mid] = st.multiselect(f"Number formats for {L.MODELS.get(mid, (mid, ''))[0]}", precs,
                                                default=default, format_func=prec_label, key=f"prec-{tid}-{mid}",
                                                help=L.PRECISION_HELP + "\n\n" + "\n".join(
                                                    f"- **{L.PRECISIONS[p][0]}**: {L.PRECISIONS[p][1]}"
                                                    for p in precs if p in L.PRECISIONS))
            chosen_precisions[tid] = per_model
    if not hw_types:
        st.warning("Tick at least one hardware option.")
        return
    fairness_panel(hw_types, tmap)
    cells = [{"target": tid, "precision": p, "model": mid}
             for tid, pm in chosen_precisions.items() for mid, ps in pm.items() for p in ps]
    itypes = {tid: it for tid, it in hw_types.items() if it != tmap[tid].instance_type}

    # 4. Load
    st.subheader("4. How hard to push?")
    c1, c2 = st.columns(2)
    with c1:
        conc = st.multiselect("Concurrent requests", list(L.CONCURRENCY_LEVELS), default=[1, 4, 16],
                              format_func=lambda c: L.CONCURRENCY_LEVELS[c], help=L.CONCURRENCY_HELP)
        depth = st.radio("How much data to collect", list(L.DEPTH), format_func=lambda d: L.DEPTH[d][0],
                         help=L.DEPTH_HELP + "\n\n" + "\n".join(f"- **{v[0]}**: {v[1]}" for v in L.DEPTH.values()))
        if depth == "quick":
            per_val, repeats = 50, 1
        elif depth == "full":
            per_val, repeats = 200, 3
        else:
            per_val = st.number_input("Requests per level", 1, 100000, 200, help=L.REQUESTS_HELP)
            repeats = st.number_input("Passes", 1, 10, 3, help=L.REPEATS_HELP)
    with c2:
        if suite == "standard":
            prompt_set = st.selectbox("Input length", list(L.PROMPT_SETS), format_func=lambda p: L.PROMPT_SETS[p][0],
                                      help=L.PROMPT_SET_HELP + "\n\n" + "\n".join(
                                          f"- **{v[0]}**: {v[1]}" for v in L.PROMPT_SETS.values()))
            custom = st.text_area("Your prompt", help="Used for every request.") if prompt_set == "custom" else None
            max_new = st.number_input("Answer length (tokens)", 1, 4096, 128, help=L.MAX_NEW_HELP)
            workloads, preset, items = None, None, None
        else:
            prompt_set, custom, max_new = "dspm", None, 4
            workloads = st.multiselect("Detection scenarios", list(L.DSPM_WORKLOADS), default=["W1", "W3"],
                                       format_func=lambda w: L.DSPM_WORKLOADS[w][0],
                                       help=L.DSPM_WORKLOADS_HELP + "\n\n" + "\n".join(
                                           f"- **{v[0]}**: {v[1]}" for v in L.DSPM_WORKLOADS.values()))
            items = st.select_slider("Documents in the accuracy check", [300, 1000, 3000], value=300,
                                     help=L.DSPM_ITEMS_HELP)
            preset = None
    with st.expander("Preview the test data for this choice"):
        st.caption("Full details, filters and the classification policy are on the Test data page.")
        if suite == "standard":
            preview_standard(models, prompt_set, chosen, n=3, key="run-std")
        elif workloads:
            preview_dspm(workloads, n=3, key="run-dspm", with_filters=False)

    # 5. Targets
    st.subheader("5. What counts as acceptable?", help=L.SLA_SECTION_HELP)
    sla: dict = {"max_error_pct": 1.0}
    if suite == "standard":
        c1, c2, c3 = st.columns(3)
        ttft = c1.number_input(L.SLA_HELP["ttft"][0], 0, 600000, 2000, help=L.SLA_HELP["ttft"][1])
        tpot = c2.number_input(L.SLA_HELP["tpot"][0], 0, 10000, 150, help=L.SLA_HELP["tpot"][1])
        e2e = c3.number_input(L.SLA_HELP["e2e"][0], 0, 600000, 0, help=L.SLA_HELP["e2e"][1])
        sla.update({k: v for k, v in {"p99_ttft_ms": ttft, "p50_tpot_ms": tpot, "p99_e2e_ms": e2e}.items() if v})
    else:
        preset = st.radio("Speed requirement", list(L.DSPM_PRESETS), index=1,
                          format_func=lambda p: L.DSPM_PRESETS[p][0],
                          help=L.DSPM_PRESET_HELP + "\n\n" + "\n".join(
                              f"- **{v[0]}**: {v[1]}" for v in L.DSPM_PRESETS.values()))
        sla["p99_e2e_ms"] = report.SLA_PRESETS[preset]["p99_e2e_ms"]

    # 6. Buying option
    st.subheader("6. How to buy the test instances?")
    purchase = st.radio("Purchase option", list(L.PURCHASE), format_func=lambda p: L.PURCHASE[p][0], horizontal=True,
                        help=L.PURCHASE_HELP + "\n\n" + "\n".join(f"- **{v[0]}**: {v[1]}" for v in L.PURCHASE.values()),
                        label_visibility="collapsed")
    fallback = st.checkbox("Fall back to On-Demand if Spot has no capacity for 10 minutes", value=True,
                           help=L.FALLBACK_HELP) if purchase == "spot" else True
    name = st.text_input("Run name", f"{suite}-ui", help=L.RUN_NAME_HELP)

    # Assemble and validate
    run = {"name": name, "suite": suite,
           "mode": "full" if (per_val >= 200 and repeats >= 3) else "smoke",
           "account": ACCOUNT, "region": REGION, "az_id": "use1-az1", "models": chosen, "cells": cells,
           "prompt_set": prompt_set, "custom_prompt": custom,
           "sampling": {"temperature": 0.0, "max_new_tokens": int(max_new), "ignore_eos": suite == "standard", "seed": 0},
           "concurrency": sorted(int(c) for c in conc), "per_level": {"requests": int(per_val)},
           "repeats": int(repeats), "warmup_requests": 20, "quality": {"enabled": suite == "standard"},
           "sla": sla, "purchase": {"type": purchase, "spot_fallback_on_demand": bool(fallback)}}
    if itypes:
        run["instance_types"] = itypes
    if suite == "dspm":
        run["dspm"] = {"workloads": workloads, "sla_preset": preset, "test_items": int(items), "prefix_caching": True}
        run["request_timeout_s"] = 120
    text = yaml.safe_dump({"run": {k: v for k, v in run.items() if v is not None}}, sort_keys=False)
    try:
        rc = RunConfig.model_validate(yaml.safe_load(text)["run"])
        rcells = resolve_cells(rc, models, targets)
    except Exception as e:  # noqa: BLE001
        st.error(f"This run cannot start yet: {e}")
        return

    # 7. Review
    st.subheader("7. Review")
    from bench.estimate_run import estimate
    from bench.pools import local_pool_sizes

    sizes = local_pool_sizes()
    rtypes = run_instance_types(rc, rcells, tmap)
    series = []
    missing = []
    for c in rcells:
        try:
            n = len(levels_for_cell(rc, c, models.get(c.model), run_id="ui", bucket="ui", pool_sizes=sizes))
        except ValueError as e:
            n, missing = 0, missing + [f"{c.cell_id}: {e}"]
        series.append({"Model": L.MODELS.get(c.model, (c.model, ""))[0],
                       "Hardware option": hardware_label(c.target, tmap[c.target], rtypes[c.target]),
                       "Number format": prec_label(c.precision), "Software": L.ENGINES.get(c.engine, c.engine),
                       "Measurements": n})
    st.dataframe(series, width="stretch", hide_index=True,
                 column_config={"Measurements": st.column_config.NumberColumn(
                     help="Concurrency levels x passes for this model on this hardware (plus the accuracy pass "
                          "and warm-ups in the sensitive-data suite).")})
    if missing:
        st.error("Some test data is missing locally: " + "; ".join(missing)[:400])
    cat = pairing.load_catalog()["instances"]
    prices = {t: cat[t]["usd_per_hour"] for t in set(rtypes.values()) if t in cat}
    try:
        est = estimate(rc, rcells, models, tmap, prices, sizes, instance_types=rtypes)
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated wall-clock time", f"{est['wall_clock_hours']:.1f} h",
                  help="Hardware options run in parallel; this is the longest one plus model loading. Speeds are "
                       "assumptions until a run has measured this model on this hardware.")
        c2.metric("Estimated cost", f"${est['usd']['total']:.2f}",
                  help=f"Instances ${est['usd']['instances']:.2f} (On-Demand rates; Spot is usually 30-70% less) + "
                       f"network ${est['usd']['nat_gateway']:.2f} + load generator ${est['usd']['lambda']:.2f}. "
                       "Excludes the standing network gateway cost between runs.")
        c3.metric("Result grade", "Decision-grade" if not rc.is_smoke else "Quick check", help=L.GRADE_HELP)
    except Exception as e:  # noqa: BLE001
        st.caption(f"No estimate: {e}")
    if rc.is_smoke:
        st.warning("Quick check: every number in the report will carry the SMOKE label (the engineering term for a "
                   "pipeline check). Use it to see the shape of the results, not to decide; choose 'Full methodology' "
                   "under step 4 for decision-grade numbers.")

    if st.button("Start run", type="primary", disabled=bool(missing)) and aws_ready():
        import boto3

        out = stack_outputs()
        run_id = f"{rc.name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        boto3.client("s3", region_name=REGION).put_object(Bucket=out["BucketName"], Key=f"runs/{run_id}/run.yaml",
                                                          Body=text.encode())
        boto3.client("stepfunctions", region_name=REGION).start_execution(
            stateMachineArn=out["RunStateMachineArn"], name=run_id, input=json.dumps({"run_id": run_id}))
        st.success(f"Started {run_id}. Follow it on the Runs page; the report appears under Results when it finishes.")
    with st.expander("Advanced: the run definition the command line uses"):
        st.caption("The same YAML can be started with `bench run <file>`; `bench validate <file>` checks it offline.")
        st.code(text, language="yaml")
        st.download_button("Download YAML", text, file_name=f"{run['name']}.yaml")


# ---------------------------------------------------------------------------- Test data

PROMPTS_DIR = ROOT / "prompts/generated"
DSPM_DIR = ROOT / "data/dspm/v1"


@st.cache_data
def prompt_manifest() -> dict | None:
    p = PROMPTS_DIR / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def prompt_set(model_id: str, name: str) -> list[dict]:
    p = PROMPTS_DIR / model_id / f"{name}.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


@st.cache_data
def dspm_items(split: str = "test") -> list[dict]:
    import gzip

    p = DSPM_DIR / f"{split}.jsonl.gz"
    if not p.exists():
        return []
    with gzip.open(p, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


@st.cache_data
def dspm_manifest() -> dict | None:
    p = DSPM_DIR / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


def _prompt_text(line: dict) -> str:
    if "pair" in line:
        return f"Premise: {line['pair'][0]}\n\nHypothesis: {line['pair'][1]}"
    return line.get("prompt", "")


def _sample(items: list, n: int, key: str) -> list:
    import random

    if st.button("Show other examples", key=f"btn-{key}", help=L.DATA_HELP["sample"]):
        st.session_state[f"seed-{key}"] = st.session_state.get(f"seed-{key}", 0) + 1
    rnd = random.Random(st.session_state.get(f"seed-{key}", 0))  # noqa: S311 - sampling for display
    return rnd.sample(items, min(n, len(items))) if items else []


def preview_standard(models, prompt_name: str, model_ids: list[str], n: int = 3, key: str = "std") -> None:
    man = prompt_manifest()
    if man is None:
        st.info("Prompt sets are not built locally (tools/build_prompts.py); the deployed runs use the copies in S3.")
        return
    mid = st.selectbox("Shown in the chat format of", model_ids, format_func=lambda m: L.MODELS.get(m, (m, ""))[0],
                       key=f"pm-{key}", help=L.DATA_HELP["model"])
    name = prompt_name if prompt_name != "custom" else "short"
    info = man["sets"].get(f"{mid}/{name}")
    if info:
        st.caption(f"{info['n']} prompts of {info['min_tokens']}-{info['max_tokens']} tokens (target {info['target']}); "
                   f"every request in a measurement draws from this set in a fixed order.")
    for line in _sample(prompt_set(mid, name), n, key):
        with st.expander(f"{line['id']} · {line.get('prompt_tokens_expected')} tokens"):
            st.text(_prompt_text(line)[:3000])
    qname = "quality-50-nli" if models.get(mid).type == "encoder-classifier" else "quality-50"
    q = prompt_set(mid, qname)
    if q:
        with st.expander(f"Quality check set: {len(q)} items ({qname})"):
            st.caption("Each model answers these once per hardware option; the report compares the answers with "
                       "the GPU's (exact match and ROUGE-L for text, label agreement for classifiers).")
            for line in q[:5]:
                st.text(f"[{line['id']}{' · ' + line['category'] if line.get('category') else ''}] "
                        f"{_prompt_text(line)[:400]}")


def preview_dspm(workloads: list[str], n: int = 5, key: str = "dspm", with_filters: bool = True) -> None:
    items = dspm_items("test")
    man = dspm_manifest()
    if not items or man is None:
        st.info("The dataset is not built locally (tools/dspm/build_dataset.py); the deployed runs use the copy in S3.")
        return
    counts = man.get("counts") or {}
    rows = [{"Scenario": L.DSPM_WORKLOADS.get(w, (w, ""))[0], "Documents": sum(
        1 for x in items if x["workload"] == w),
             "Sensitive / positive": sum(1 for x in items if x["workload"] == w and x["sensitive"]),
             "Hard negatives": sum(1 for x in items if x["workload"] == w and x.get("hard_case"))} for w in workloads]
    st.dataframe(rows, hide_index=True, width="stretch", column_config={
        "Hard negatives": st.column_config.NumberColumn(help=L.DATA_HELP["hard_case"])})
    dev = counts.get("split", {}).get("dev") if isinstance(counts.get("split"), dict) else None
    st.caption(f"Dataset {man.get('version')}, fingerprint {str(man.get('dataset_sha256'))[:12]}, "
               f"{len(items)} test documents" + (f", {dev} development documents (used only to tune the classifier "
                                                 f"threshold)." if dev else "."))
    pool = [x for x in items if x["workload"] in workloads]
    if with_filters:
        c1, c2, c3 = st.columns(3)
        wl = c1.selectbox("Scenario", workloads, format_func=lambda w: L.DSPM_WORKLOADS.get(w, (w, ""))[0],
                          key=f"wl-{key}", help=L.DATA_HELP["scenario"])
        labels_ = sorted({x["label"] for x in pool if x["workload"] == wl})
        lab = c2.selectbox("Expected answer", ["any", *labels_], key=f"lab-{key}", help=L.DATA_HELP["label"])
        hards = sorted({x["hard_case"] for x in pool if x["workload"] == wl and x.get("hard_case")})
        hard = c3.selectbox("Hard negative type", ["any", "none", *hards], key=f"hard-{key}", help=L.DATA_HELP["hard_case"])
        pool = [x for x in pool if x["workload"] == wl and (lab == "any" or x["label"] == lab)
                and (hard == "any" or (hard == "none" and not x.get("hard_case")) or x.get("hard_case") == hard)]
        st.caption(f"{len(pool)} documents match.")
    for x in _sample(pool, n, key):
        cats = ", ".join(x.get("categories") or []) or "none"
        hard = f" · hard negative: {x['hard_case']}" if x.get("hard_case") else ""
        toks = x["tokens"]
        toks = toks.get("m1-qwen2.5-3b", next(iter(toks.values()))) if isinstance(toks, dict) else toks
        title = (f"{x['id']} · expected **{x['label']}** · categories: {cats}{hard} · about {toks} tokens, "
                 f"{x['bytes']} bytes · source {x['source']}")
        with st.expander(title):
            st.text(x["text"][:4000] + ("\n[...]" if len(x["text"]) > 4000 else ""))
            if x.get("spans"):
                st.caption("Sensitive values in this document: " + "; ".join(
                    f"{s.get('category', '?')}: {x['text'][s['start']:s['end']][:40]}"
                    for s in x["spans"][:8] if "start" in s and "end" in s))


def page_test_data():
    st.header("Test data")
    st.caption("What the models are asked during a run. Each workload type has its own data; nothing is downloaded "
               "or generated at run time.")
    suite = st.radio("Workload type", list(L.SUITES), format_func=lambda s: L.SUITES[s][0], horizontal=True,
                     help=L.SUITE_HELP, key="data-suite")
    st.write(L.DATA_INTRO[suite])
    models, _ = catalogs()
    if suite == "standard":
        name = st.selectbox("Prompt set", ["short", "medium", "long"], format_func=lambda p: L.PROMPT_SETS[p][0],
                            help=L.DATA_HELP["set"])
        ids = [m.id for m in models.models if m.enabled and "standard" in m.suites
               and (PROMPTS_DIR / m.id / f"{name}.jsonl").exists()]
        if ids:
            preview_standard(models, name, ids, n=5, key="data-std")
        else:
            st.info("Prompt sets are not built locally (tools/build_prompts.py).")
    else:
        preview_dspm(list(L.DSPM_WORKLOADS), n=5, key="data-dspm")
        pol = ROOT / "config/dspm/policy_v1.md"
        if pol.exists():
            with st.expander("The classification policy the models are given", expanded=False):
                st.caption(L.DATA_HELP["policy"])
                st.text(pol.read_text())
        rs = DSPM_DIR / "review_sample.md"
        if rs.exists():
            st.download_button("Download the 100-document review sample (Markdown)", rs.read_bytes(), "review_sample.md",
                               help="A stratified sample with the sensitive values marked, for a human to check the "
                                    "labels against the policy.")


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


RUN_STATUS = {"running": "Running", "complete": "Finished, report ready", "refused": "Refused by the pre-checks",
              "planned": "Planned"}
CELL_STATUS = {"planned": "Planned", "configuring": "Starting the model server", "ready": "Serving",
               "bringup_failed": "Failed to start", "interrupted": "Instance lost, replacing it", "failed": "Failed"}
LEVEL_STATUS = {"running": "Running", "done": "Done", "failed": "Failed", "incomplete": "Stopped early",
                "pending": "Waiting to be re-run"}
RUN_STATUS_HELP = ("Running: in progress. Finished: the report is under Results. Refused: a pre-check failed before "
                   "anything started (missing capacity, price, data or access); the reason is listed below.")
CELL_STATUS_HELP = ("Planned: waiting for the instance. Starting the model server: instance up, loading the model. "
                    "Serving: measurements are running. Instance lost: a Spot interruption; a replacement is being "
                    "launched and unfinished measurements re-run. Failed: see the error.")
LEVEL_STATUS_HELP = ("Done: all requests sent and recorded. Running: in progress. Failed: see the error. Stopped early: "
                     "the request budget could not be finished in the allowed number of waves.")


def _hw_name(tid: str, tmap, instance_type: str | None = None) -> str:
    t = tmap.get(tid)
    return hardware_label(tid, t, instance_type or t.instance_type) if t else tid


def _parse_level(level_key: str, cell_key: str) -> dict:
    """<model>__<target>__<precision>__<name>__c<C>[__r<rep>] -> readable parts."""
    rest = level_key[len(cell_key) + 2:].split("__")
    name = rest[0]
    conc = next((int(x[1:]) for x in rest[1:] if x.startswith("c") and x[1:].isdigit()), None)
    rep = next((int(x[1:]) for x in rest[1:] if x.startswith("r") and x[1:].isdigit()), None)
    if name.endswith("-accuracy"):
        base = name[:-9]
        what = "Accuracy pass: " + (L.DSPM_WORKLOADS.get(base.split("-")[0], (base, ""))[0]
                                    + (f" ({base.split('-', 1)[1]} documents)" if "-" in base else ""))
    elif name in L.PROMPT_SETS:
        what = L.PROMPT_SETS[name][0]
    elif name.split("-")[0] in L.DSPM_WORKLOADS:
        what = L.DSPM_WORKLOADS[name.split("-")[0]][0] + (f" ({name.split('-', 1)[1]} documents)" if "-" in name else "")
    else:
        what = name
    return {"what": what, "conc": conc, "pass": None if rep is None else rep + 1}


def page_runs():
    st.header("Runs")
    if not aws_ready():
        return
    import boto3
    from boto3.dynamodb.conditions import Key

    from lambdas.common import plain

    models, targets = catalogs()
    tmap = targets.resolved()
    runs = _runs()
    st.dataframe([{"Run": r["pk"][4:], "Status": RUN_STATUS.get(r.get("status"), r.get("status")),
                   "Last update": r.get("updated_at"), "Grade": "Quick check" if r.get("smoke") else "Decision-grade",
                   "Instances": r.get("purchase")} for r in runs], width="stretch", hide_index=True,
                 column_config={"Status": st.column_config.TextColumn(help=RUN_STATUS_HELP),
                                "Grade": st.column_config.TextColumn(help=L.GRADE_HELP),
                                "Instances": st.column_config.TextColumn(help=L.PURCHASE_HELP),
                                "Last update": st.column_config.TextColumn(help="UTC time of the last status change.")})
    rid = st.selectbox("Run to inspect", [r["pk"][4:] for r in runs]) if runs else None
    if not rid:
        return
    tbl = boto3.resource("dynamodb", region_name=REGION).Table("amx-gpu-bench")
    items = [plain(x) for x in tbl.query(KeyConditionExpression=Key("pk").eq(f"RUN#{rid}"))["Items"]]
    meta = next((i for i in items if i["sk"] == "META"), {})
    if meta.get("status") == "refused":
        st.error("Refused by the pre-checks: " + "; ".join(str(e) for e in (meta.get("errors") or [])[:10]))
    cells = [i for i in items if i["sk"].startswith("CELL#")]
    levels = [i for i in items if i["sk"].startswith("LEVEL#")]
    done = sum(1 for i in levels if i.get("status") == "done")
    st.progress(done / max(1, len(levels)), text=f"{done} of {len(levels)} measurements done")

    st.markdown("**Models on hardware options**", help="One row per model served on one hardware option. The instance "
                "is started, the model loaded, then the measurements below run one after another.")
    cell_rows = []
    for c in sorted(cells, key=lambda x: x["sk"]):
        mid, tid, prec = (c["sk"][5:].split("|") + [None, None, None])[:3]
        cell_rows.append({"Model": L.MODELS.get(mid, (mid, ""))[0], "Hardware option": _hw_name(tid, tmap),
                          "Number format": prec_label(prec) if prec else "", "Software": L.ENGINES.get(c.get("engine"), c.get("engine")),
                          "Status": CELL_STATUS.get(c.get("status"), c.get("status")), "Measurements": c.get("n_levels"),
                          "Error": (c.get("error") or c.get("reason") or c.get("stderr") or "")[:160]})
    st.dataframe(cell_rows, width="stretch", hide_index=True,
                 column_config={"Status": st.column_config.TextColumn(help=CELL_STATUS_HELP)})

    st.markdown("**Measurements**", help="One row per concurrency level (and pass) of a model on a hardware option. "
                "'Requests done' counts the requests sent so far; the accuracy pass of the sensitive-data suite "
                "classifies every test document once.")
    level_rows = []
    for lv in sorted(levels, key=lambda x: x["sk"]):
        key = lv["sk"][6:]
        ck = lv.get("cell_key") or "__".join(key.split("__")[:3])
        mid, tid, _prec = (ck.split("__") + [None, None, None])[:3]
        parts = _parse_level(key, ck)
        level_rows.append({"Model": L.MODELS.get(mid, (mid, ""))[0], "Hardware option": _hw_name(tid, tmap),
                           "What": parts["what"], "Concurrent requests": parts["conc"], "Pass": parts["pass"],
                           "Status": LEVEL_STATUS.get(lv.get("status"), lv.get("status")),
                           "Requests done": lv.get("completed"), "Errors": lv.get("errors"),
                           "Detail": (lv.get("error") or lv.get("reason") or "")[:120]})
    st.dataframe(level_rows, width="stretch", hide_index=True,
                 column_config={"Status": st.column_config.TextColumn(help=LEVEL_STATUS_HELP),
                                "Concurrent requests": st.column_config.NumberColumn(help=L.CONCURRENCY_HELP),
                                "Pass": st.column_config.NumberColumn(help=L.REPEATS_HELP)})
    with st.expander("Advanced: raw status records"):
        st.dataframe([{"Item": i["sk"], "status": i.get("status"), "completed": i.get("completed"),
                       "errors": i.get("errors")} for i in sorted(items, key=lambda x: x["sk"])],
                     width="stretch", hide_index=True,
                     column_config={"Item": st.column_config.TextColumn(
                         help="Storage keys: META = the run, CELL# = one model on one hardware option, "
                              "LEVEL# = one concurrency level of it.")})
    if st.checkbox("I want to stop this run", help="Finished measurements are kept and reported.") \
            and st.button("Stop run", type="secondary"):
        arn = stack_outputs()["RunStateMachineArn"].replace(":stateMachine:", ":execution:")
        boto3.client("stepfunctions", region_name=REGION).stop_execution(executionArn=f"{arn}:{rid}",
                                                                         cause="stopped from the UI")
        st.success("Stopped. Test instances are stopped automatically within 30 minutes.")


# ---------------------------------------------------------------------------- GPU vs CPU comparison

GPU_COLORS = ["#2e7d32", "#66bb6a", "#1b5e20", "#a5d6a7"]
CPU_COLORS = ["#1565c0", "#42a5f5", "#0d47a1", "#90caf9", "#5c6bc0", "#7986cb"]


def _group_key(r: dict) -> str:
    """Rows that answer the same question: one model, one scenario (and length bucket)."""
    key = L.MODELS.get(r["model"], (r["model"], ""))[0]
    if r.get("workload"):
        key += " · " + L.DSPM_WORKLOADS.get(r["workload"], (r["workload"], ""))[0]
        if r.get("bucket"):
            key += f" ({r['bucket']} documents)"
    return key


def _unit(rows: list[dict]) -> tuple[str, str, str, str]:
    """(throughput field, throughput label, cost field, cost label) for a group."""
    r = rows[0]
    if r.get("workload"):
        return "docs_s", "documents per second", "usd_per_gb", "$ per GB scanned"
    if r.get("kind") == "classifier":
        return "cls_s", "classifications per second", "usd_per_1m_items", "$ per 1M classifications"
    return "agg_out_tok_s", "tokens per second", "usd_per_1m_tokens", "$ per 1M tokens"


def _series(r: dict) -> str:
    side = "GPU" if r["accelerator"] == "gpu" else "CPU"
    return f"{side} · {r['instance_type']} · {L.ENGINES.get(r['engine'], r['engine'])} · {r['precision']}"


def _best(rows: list[dict], thr: str):
    ok = [r for r in rows if r.get("meets_user_sla") and r.get(thr)]
    return max(ok, key=lambda r: r[thr]) if ok else None


def _fmt(v, nd=0) -> str:
    return "-" if v is None else f"{v:,.{nd}f}"


def compare_section(rows: list[dict], sla: dict, limit: int, accuracy: dict | None) -> None:
    import math

    import altair as alt
    import pandas as pd

    st.subheader("GPU versus CPU at a glance",
                 help="Everything here is computed against the targets you set above. GPU is green, CPU is blue. "
                      "'Best' means the highest-throughput measurement that still meets every target.")
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(_group_key(r), []).append(r)
    for gname, grows in groups.items():
        thr, thr_label, cost, cost_label = _unit(grows)
        lat = "e2e_p99" if grows[0].get("workload") else "ttft_p99"
        lat_label = "whole decision, 99th pct (ms)" if grows[0].get("workload") else "first word, 99th pct (ms)"
        lat_target = sla.get("p99_e2e_ms") if grows[0].get("workload") else sla.get("p99_ttft_ms")
        gpu_rows = [r for r in grows if r["accelerator"] == "gpu"]
        cpu_rows = [r for r in grows if r["accelerator"] == "cpu" and r.get("amx") != "INVALID"]
        st.markdown(f"#### {gname}")
        if not gpu_rows or not cpu_rows:
            st.info("This run has only one side (GPU or CPU) for this model; a comparison needs both.")
            continue
        best_gpu, best_cpu = _best(gpu_rows, thr), _best(cpu_rows, thr)

        # --- head-to-head cards
        c_gpu, c_cpu = st.columns(2)
        for col, side, best, other in ((c_gpu, "GPU", best_gpu, best_cpu), (c_cpu, "CPU", best_cpu, best_gpu)):
            with col:
                if best is None:
                    st.error(f"{side}: no measurement meets your targets.")
                    continue
                st.markdown(f"**{'🟩' if side == 'GPU' else '🟦'} {side}: {best['instance_type']} · "
                            f"{L.ENGINES.get(best['engine'], best['engine'])} · {prec_label(best['precision'])}** "
                            f"at {best['conc']} concurrent requests · ${best['usd_per_hour']:.2f}/hour")
                ratio = (best[thr] / other[thr]) if other and other.get(thr) else None
                m1, m2, m3 = st.columns(3)
                m1.metric("Throughput", f"{_fmt(best[thr])} {thr_label.split(' per ')[0]}/s",
                          delta=None if side == "GPU" or ratio is None else f"{ratio:.2f}x of GPU",
                          delta_color="normal", help=f"{thr_label} across all concurrent requests. Higher is better.")
                lr = (best[lat] / other[lat]) if other and other.get(lat) and best.get(lat) else None
                m2.metric("Responsiveness", f"{_fmt(best.get(lat))} ms",
                          delta=None if side == "GPU" or lr is None else f"{lr:.1f}x GPU's wait",
                          delta_color="inverse", help=f"{lat_label}. Lower is better.")
                cr = (best[cost] / other[cost]) if other and other.get(cost) and best.get(cost) else None
                m3.metric("Cost per unit", f"${_fmt(best.get(cost), 2)}",
                          delta=None if side == "GPU" or cr is None else f"{cr:.1f}x GPU's cost",
                          delta_color="inverse", help=f"{cost_label} at this load, On-Demand price. Lower is better.")
                if accuracy:
                    name = (best.get("workload") or "") + (f"-{best['bucket']}" if best.get("bucket") else "")
                    a = (accuracy.get(best.get("cell_key") or "") or {}).get(name)
                    if a:
                        st.caption(f"Detection accuracy on this scenario: recall {a['recall']['value']:.2f}, "
                                   f"false positives {a['fpr']['value']:.2f}, F1 {a['f1']:.2f}"
                                   + (f" · agreement with GPU {a['vs_gpu']['agreement']['value']:.2f}"
                                      if a.get("vs_gpu") else ""))
        if best_gpu and best_cpu:
            n = math.ceil(best_gpu[thr] / best_cpu[thr]) if best_cpu[thr] else None
            if n:
                cost_ratio = n * best_cpu["usd_per_hour"] / best_gpu["usd_per_hour"]
                verdict = ("within your limit: the CPU side is a viable alternative" if n <= limit
                           else f"above your limit of {limit}: conditional")
                st.markdown(f"**To match one {best_gpu['instance_type']} you need {n} × {best_cpu['instance_type']}**, "
                            f"costing {cost_ratio:.1f}x as much per hour, {verdict}.")

        # --- instances needed per CPU option
        eq = []
        if best_gpu:
            for r in cpu_rows:
                if r.get("meets_user_sla") and r.get(thr):
                    n = math.ceil(best_gpu[thr] / r[thr])
                    eq.append({"CPU option": f"{r['instance_type']} · {r['precision']} · {r['conc']} concurrent",
                               "Instances per GPU": n, "Hourly cost ratio": n * r["usd_per_hour"] / best_gpu["usd_per_hour"],
                               "Within limit": "yes" if n <= limit else "no"})
        if eq:
            bars = alt.Chart(pd.DataFrame(eq)).mark_bar().encode(
                y=alt.Y("CPU option:N", sort="-x", title=None),
                x=alt.X("Instances per GPU:Q", title=f"CPU instances needed to match one {best_gpu['instance_type']}"),
                color=alt.Color("Within limit:N", scale=alt.Scale(domain=["yes", "no"], range=["#1565c0", "#b0bec5"]),
                                legend=alt.Legend(title="Within your limit")),
                tooltip=["CPU option", "Instances per GPU", alt.Tooltip("Hourly cost ratio:Q", format=".2f")])
            rule = alt.Chart(pd.DataFrame({"limit": [limit]})).mark_rule(color="#c62828", strokeDash=[6, 4]).encode(x="limit:Q")
            st.altair_chart((bars + rule).properties(height=28 * len(eq) + 40), use_container_width=True)
            st.caption(f"Red line: your limit of {limit} CPU instances per GPU. Only CPU measurements that meet your "
                       "targets are shown.")

        # --- curves: throughput, responsiveness, cost vs concurrency
        df = pd.DataFrame(grows)
        df["Series"] = df.apply(_series, axis=1)
        df["Meets targets"] = df["meets_user_sla"].map({True: "yes", False: "no"})
        series = sorted(df["Series"].unique())
        colors = {}
        gi = ci = 0
        for s_ in series:
            if s_.startswith("GPU"):
                colors[s_] = GPU_COLORS[gi % len(GPU_COLORS)]
                gi += 1
            else:
                colors[s_] = CPU_COLORS[ci % len(CPU_COLORS)]
                ci += 1
        scale = alt.Scale(domain=series, range=[colors[s_] for s_ in series])

        def curve(field: str, title: str, target: float | None = None, log: bool = False, df=df, scale=scale):
            base = alt.Chart(df.dropna(subset=[field])).encode(
                x=alt.X("conc:O", title="Concurrent requests"),
                y=alt.Y(f"{field}:Q", title=title, scale=alt.Scale(type="log") if log else alt.Undefined),
                color=alt.Color("Series:N", scale=scale, legend=alt.Legend(orient="bottom", columns=1, title=None)),
                tooltip=["Series", "conc", alt.Tooltip(f"{field}:Q", format=",.2f"), "Meets targets"])
            chart = base.mark_line() + base.mark_point(size=90, filled=True).encode(
                shape=alt.Shape("Meets targets:N", scale=alt.Scale(domain=["yes", "no"], range=["circle", "cross"]),
                                legend=alt.Legend(orient="bottom", title="Meets targets")))
            if target:
                chart = chart + alt.Chart(pd.DataFrame({"t": [target]})).mark_rule(
                    color="#c62828", strokeDash=[6, 4]).encode(y="t:Q")
            return chart.properties(height=300)

        k1, k2, k3 = st.columns(3)
        with k1:
            st.caption("**Throughput** — higher is better")
            st.altair_chart(curve(thr, thr_label), use_container_width=True)
        with k2:
            st.caption("**Responsiveness** — lower is better; red line = your target")
            st.altair_chart(curve(lat, lat_label, lat_target), use_container_width=True)
        with k3:
            st.caption("**Cost per unit** — lower is better (log scale)")
            st.altair_chart(curve(cost, cost_label, log=True), use_container_width=True)


# ---------------------------------------------------------------------------- Results

def _load_summary(rid: str) -> dict | None:
    import boto3

    try:
        body = boto3.client("s3", region_name=REGION).get_object(
            Bucket=stack_outputs()["BucketName"], Key=f"runs/{rid}/report/summary.json")["Body"].read()
        return json.loads(body)
    except Exception:  # noqa: BLE001
        return None


def _verdict_text(summary: dict) -> list[str]:
    out = []
    for key, pm in sorted(summary.get("per_model", {}).items()):
        name = key.split(" ")[0]
        label = L.MODELS.get(name, (name, ""))[0] + key[len(name):]
        gpu = pm.get("best_gpu")
        if not gpu:
            out.append(f"**{label}**: {pm.get('note') or 'no GPU measurement met the targets'}.")
            continue
        eq = pm.get("capacity_equivalence") or []
        ok = [e for e in eq if e.get("within_limit")]
        best = f"best GPU: {gpu['instance_type']} at {gpu['conc']} concurrent requests"
        if ok:
            e = min(ok, key=lambda x: x["hourly_cost_ratio"])
            out.append(f"**{label}**: {best}. **{e['cpu_instances_per_gpu']} x {e['cpu_row']['instance_type']}** "
                       f"({e['cpu_row']['precision']}, {e['cpu_row']['conc']} concurrent) match it at "
                       f"{e['hourly_cost_ratio']:.1f}x the hourly cost, within the limit: CPU is a viable alternative.")
        elif eq:
            e = min(eq, key=lambda x: x["cpu_instances_per_gpu"])
            out.append(f"**{label}**: {best}. The closest CPU option needs {e['cpu_instances_per_gpu']} x "
                       f"{e['cpu_row']['instance_type']} ({e['hourly_cost_ratio']:.1f}x the cost), above the limit: "
                       f"conditional.")
        else:
            out.append(f"**{label}**: {best}; no CPU measurement met the targets.")
    return out


def page_results():
    st.header("Results")
    upload = st.file_uploader("Open a downloaded summary.json instead", type="json",
                              help="Reports are also saved to S3; this loads one from your disk.")
    summary = json.loads(upload.read()) if upload else None
    if summary is None and aws_ready():
        runs = [r["pk"][4:] for r in _runs() if r.get("status") == "complete"]
        rid = st.selectbox("Finished run", runs) if runs else None
        summary = _load_summary(rid) if rid else None
    if not summary:
        st.info("No finished run selected.")
        return
    if summary.get("smoke"):
        st.warning("Quick check: this run collected fewer than 200 requests per level or fewer than 3 passes, so "
                   "its numbers show the shape of the result, not a basis for a decision. The report labels them SMOKE.")
    rows = summary["rows"]
    for j in summary.get("pairing") or []:
        head = "Fair comparison" if j["fair"] else "Not a fair comparison"
        (st.success if j["fair"] else st.warning)(f"{head} — {j['cpu']} versus {j['gpu']}: {j['reason']}.")

    st.subheader("Your targets", help=L.SLA_SECTION_HELP)
    run_sla = (summary.get("run") or {}).get("sla") or {}
    err_help = "Measurements with more failed requests than this do not count."
    if summary["run"].get("suite") == "dspm":
        c1, c2 = st.columns([3, 1])
        presets = list(L.DSPM_PRESETS)
        run_preset = ((summary["run"].get("dspm") or {}).get("sla_preset")) or "inline_standard"
        preset = c1.radio("Speed requirement", presets, index=presets.index(run_preset) if run_preset in presets else 1,
                          format_func=lambda p: L.DSPM_PRESETS[p][0], help=L.DSPM_PRESET_HELP, horizontal=True)
        sla = {**report.SLA_PRESETS[preset],
               "max_error_pct": c2.number_input("Max failed requests (%)", 0.0, 100.0, 1.0, help=err_help)}
        prov = summary.get("dspm_hashes") or {}
        st.caption(f"Dataset version {prov.get('dataset_version')}, dataset fingerprint "
                   f"{str(prov.get('dataset_sha256'))[:12]}, policy fingerprint {str(prov.get('policy_sha256'))[:12]}",
                   help="Fingerprints identify the exact test data and classification policy this run used, so two "
                        "runs can be compared only when they match.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        sla = {"p99_ttft_ms": c1.number_input(L.SLA_HELP["ttft"][0], 0.0, 1e6, float(run_sla.get("p99_ttft_ms") or 2000),
                                              help=L.SLA_HELP["ttft"][1]) or None,
               "p50_tpot_ms": c2.number_input(L.SLA_HELP["tpot"][0], 0.0, 1e5, float(run_sla.get("p50_tpot_ms") or 150),
                                              help=L.SLA_HELP["tpot"][1]) or None,
               "p99_e2e_ms": c3.number_input(L.SLA_HELP["e2e"][0], 0.0, 1e7, float(run_sla.get("p99_e2e_ms") or 0),
                                             help=L.SLA_HELP["e2e"][1]) or None,
               "max_error_pct": c4.number_input("Max failed requests (%)", 0.0, 100.0, 1.0, help=err_help)}
        st.caption("Pre-filled with the targets the run was started with; change them to see how the verdict moves.")
    for r in rows:
        r["meets_user_sla"], r["user_sla_reason"] = report.meets_sla(r, sla)
    limit = int(((summary.get("run") or {}).get("capacity") or {}).get("max_cpu_instances_per_gpu", 4))
    compare_section(rows, sla, limit, summary.get("dspm_accuracy"))

    with st.expander("Verdict as written in the report (uses the run's own targets)"):
        for line in _verdict_text(summary):
            st.markdown("- " + line)

    st.subheader("Measurements")
    model_ids = sorted({r["model"] for r in rows})
    pick = st.multiselect("Models", model_ids, default=model_ids, format_func=lambda m: L.MODELS.get(m, (m, ""))[0])
    view = [r for r in rows if r["model"] in pick]
    cols = [c for c in L.RESULT_COLUMNS if any(r.get(c) not in (None, "", []) for r in view)]
    config = {k: st.column_config.Column(L.RESULT_COLUMNS[k][0], help=L.RESULT_COLUMNS[k][1]) for k in cols}
    st.dataframe([{k: r.get(k) for k in cols} for r in view], width="stretch", hide_index=True, column_config=config)
    import pandas as pd

    df = pd.DataFrame(view)
    if not df.empty:
        metric = st.selectbox("Chart", [m for m in L.CHARTS if m in df.columns and df[m].notna().any()],
                              format_func=lambda m: L.CHARTS[m],
                              help="One line per model, hardware option and number format; x axis = concurrent requests.")
        df["series"] = df["model"].map(lambda m: L.MODELS.get(m, (m, ""))[0]) + " · " + df["instance_type"] + " · " + df["precision"]
        piv = df.pivot_table(index="conc", columns="series", values=metric, aggfunc="mean")
        st.line_chart(piv, x_label="concurrent requests", y_label=L.CHARTS[metric])

    st.subheader("How many CPUs replace one GPU?", help=L.CAPACITY_HELP)
    eq_rows = []
    for key, pm in sorted(summary.get("per_model", {}).items()):
        for e in pm.get("capacity_equivalence") or []:
            eq_rows.append({"Model": L.MODELS.get(key.split(" ")[0], (key, ""))[0] + key[len(key.split(" ")[0]):],
                            "CPU instance": e["cpu_row"]["instance_type"], "Number format": e["cpu_row"]["precision"],
                            "Concurrent requests": e["cpu_row"]["conc"], "CPU instances per GPU": e["cpu_instances_per_gpu"],
                            "Standard vCPUs needed": e["standard_vcpus_needed"], "G vCPUs replaced": e["gpu_vcpus_replaced"],
                            "Hourly cost ratio": round(e["hourly_cost_ratio"], 2),
                            "Within limit": "yes" if e["within_limit"] else "no"})
    if eq_rows:
        st.dataframe(eq_rows, width="stretch", hide_index=True, column_config={
            "CPU instances per GPU": st.column_config.NumberColumn(help="Instances of this CPU type needed to match the best GPU measurement's throughput."),
            "Standard vCPUs needed": st.column_config.NumberColumn(help="vCPUs drawn from the Standard (M/C/R) On-Demand quota pool."),
            "G vCPUs replaced": st.column_config.NumberColumn(help="vCPUs the GPU instance draws from the separate G-family quota pool."),
            "Hourly cost ratio": st.column_config.NumberColumn(help="Cost of that many CPU instances divided by the GPU instance's cost, On-Demand rates."),
            "Within limit": st.column_config.TextColumn(help="Whether the count is within the run's maximum CPU instances per GPU (default 4).")})
    else:
        st.caption("Not computed: a GPU measurement that meets the targets is needed as the reference.")
    files = report.render(summary, with_charts=False)
    c1, c2 = st.columns(2)
    c1.download_button("Download table (CSV)", files["summary.csv"], "summary.csv")
    c2.download_button("Download report (Markdown)", files["summary.md"], "summary.md")
    if summary.get("warnings"):
        st.subheader("Notes and warnings")
        for w in summary["warnings"]:
            st.write("- " + w)
    with st.expander("Advanced: raw analysis (JSON)"):
        st.json(summary.get("per_model", {}))


# ---------------------------------------------------------------------------- Calculator

def page_calculator():
    st.header("Migration effort calculator")
    st.caption("How much engineering work it takes to move a model from NVIDIA GPUs to AMX CPUs, in engineer-days, "
               "and whether the benchmark results support the move. Rules come from config/migration_effort.yaml; "
               "capacity decides the verdict, cost is shown alongside.")
    n = st.number_input("Number of models to migrate", 1, 20, 1, help="One block of questions per model.")
    ms = []
    for i in range(int(n)):
        with st.expander(f"Model {i + 1}", expanded=i == 0):
            c1, c2, c3 = st.columns(3)
            ms.append(estimator.ModelInput(
                name=c1.text_input("Name", f"model-{i + 1}", key=f"n{i}", help=L.CALC_HELP["name"]),
                model_type=c1.selectbox("Model type", ["decoder", "encoder-decoder", "encoder-classifier", "token-classifier"],
                                        key=f"t{i}", help=L.CALC_HELP["type"]),
                params_b=c2.number_input("Size (billions of parameters)", 0.05, 500.0, 3.0, key=f"p{i}",
                                         help=L.CALC_HELP["params"]),
                stack=c2.selectbox("Served today with", ["vllm_gpu", "pytorch_transformers", "sagemaker_pytorch", "tgi",
                                                         "triton_trtllm"], key=f"s{i}", help=L.CALC_HELP["stack"]),
                cuda_deps=c3.multiselect("GPU-only dependencies", ["flash_attention", "bitsandbytes", "custom_kernels",
                                                                   "tensorrt_engines"], key=f"d{i}", help=L.CALC_HELP["cuda"]),
                precision=c3.selectbox("Number format on CPU", ["bf16", "int8", "int4"], key=f"q{i}",
                                       format_func=prec_label, help=L.CALC_HELP["precision"]),
                required_concurrency=c3.number_input("Required concurrent requests (0 = any)", 0, 1024, 0, key=f"c{i}",
                                                     help=L.CALC_HELP["conc"]) or None))
    c1, c2, c3 = st.columns(3)
    p = estimator.ProjectInput(
        models=ms,
        infrastructure=c1.selectbox("Runs on", ["ec2", "eks", "ecs", "sagemaker"], help=L.CALC_HELP["infra"]),
        iac_and_cicd_exist=c2.checkbox("Infrastructure-as-code and CI/CD exist", True, help=L.CALC_HELP["iac"]),
        familiarity=c3.selectbox("Team's CPU-tuning experience", ["medium", "low", "high"], help=L.CALC_HELP["familiarity"]),
        sla=estimator.SlaInput(p99_ttft_ms=c1.number_input(L.SLA_HELP["ttft"][0], 0.0, 1e6, 2000.0,
                                                           help=L.SLA_HELP["ttft"][1]) or None,
                               p50_tpot_ms=c2.number_input(L.SLA_HELP["tpot"][0], 0.0, 1e5, 150.0,
                                                           help=L.SLA_HELP["tpot"][1]) or None))
    up = st.file_uploader("Benchmark results (summary.json) for a hardware recommendation", type="json",
                          key="calc-summary", help=L.CALC_HELP["summary"])
    summary = json.loads(up.read()) if up else None
    out = estimator.estimate(p, summary=summary)
    c1, c2 = st.columns(2)
    c1.metric("Likely effort (engineer-days)", out["likely"],
              help=f"Range {out['low']} to {out['high']} engineer-days. One engineer-day is one person for one working day.")
    c2.metric("Overall", out["overall"], help="Go: the benchmark shows a CPU option meeting the targets within the "
                                              "capacity limit. Conditional: it does with caveats. No-go or Insufficient "
                                              "data: it does not, or no results were loaded.")
    st.dataframe(out["breakdown"], width="stretch", hide_index=True,
                 column_config={"days": st.column_config.Column(help="[low, high] engineer-days for this item.")})
    st.subheader("Risks")
    for r in out["risks"]:
        st.write("- " + r)
    st.subheader("Recommendation per model")
    for name, rec in out["recommendations"].items():
        st.markdown(f"- **{name}**: {rec.get('verdict')}. {rec.get('reason')}")
    with st.expander("Advanced: raw recommendation data"):
        st.json(out["recommendations"])


# ---------------------------------------------------------------------------- Journal

def page_journal():
    st.header("Bring-up journal")
    st.caption("Record the real effort of getting a model to run on a hardware option. Entries calibrate the "
               "migration calculator's rules against what actually happened.")
    if not aws_ready():
        return
    import boto3
    from boto3.dynamodb.conditions import Key

    from lambdas.common import plain

    models, targets = catalogs()
    tmap = targets.resolved()
    tbl = boto3.resource("dynamodb", region_name=REGION).Table("amx-gpu-bench")
    with st.form("journal"):
        c1, c2, c3 = st.columns(3)
        mid = c1.selectbox("Model", [m.id for m in models.models if m.enabled],
                           format_func=lambda m: L.MODELS.get(m, (m, ""))[0])
        tid = c2.selectbox("Hardware option", list(tmap), format_func=lambda t: hardware_label(t, tmap[t], tmap[t].instance_type))
        prec = c3.selectbox("Number format", list(L.PRECISIONS), format_func=prec_label)
        hours = c1.number_input("Engineer hours spent", 0.0, 1000.0, 2.0, help="Actual hands-on time, all people.")
        rule = c2.selectbox("Calculator rule this calibrates", [r["id"] for r in estimator.load_config()["base_rules"]],
                            help="The rule in config/migration_effort.yaml whose estimate this entry checks.")
        predicted = c3.number_input("What the calculator predicted (days)", 0.0, 100.0, 5.0,
                                    help="The midpoint the calculator gave before the work started.")
        issues = st.text_area("Issues hit", help="What went wrong, in a sentence each.")
        fixes = st.text_area("Fixes", help="What resolved each issue.")
        if st.form_submit_button("Add entry"):
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            tbl.put_item(Item={"pk": f"JOURNAL#{mid}", "sk": f"{tid}#{prec}#{ts}", "hours": str(hours),
                               "rule_id": rule, "predicted_mid_days": str(predicted), "issues": issues, "fixes": fixes})
            st.success("Saved")
    entries = []
    for m in models.models:
        entries += [plain(x) for x in tbl.query(KeyConditionExpression=Key("pk").eq(f"JOURNAL#{m.id}"))["Items"]]
    st.dataframe(entries, width="stretch", hide_index=True)
    buf = io.StringIO()
    yaml.safe_dump([{"rule_id": e["rule_id"], "actual_days": float(e["hours"]) / 8,
                     "predicted_mid_days": float(e["predicted_mid_days"]), "model": e["pk"][8:]} for e in entries], buf)
    st.download_button("Export for the calculator (journal.yaml)", buf.getvalue(), "journal.yaml",
                       help="Feed it to `bench estimate --journal` to calibrate the rules.")


PAGES = {"New run": page_new_run, "Test data": page_test_data, "Runs": page_runs, "Results": page_results,
         "Migration calculator": page_calculator, "Bring-up journal": page_journal}
st.sidebar.title("GPU vs AMX CPU")
st.sidebar.caption("Benchmark small models on NVIDIA GPU and Intel AMX CPU instances, then decide which to buy.")
choice = st.sidebar.radio("Page", list(PAGES), help="New run: define and start a benchmark. Test data: what the "
                          "models are asked. Runs: progress. Results: reports and the verdict. Migration calculator: "
                          "engineering effort. Journal: record real effort.")
PAGES[choice]()
