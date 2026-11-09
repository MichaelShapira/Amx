"""AggregateLevel and AggregateRun (design 9.1 step 5, 11-16, 24.7)."""

from __future__ import annotations

import datetime as dt
from typing import Any

from bench import metrics
from lambdas import common

CW_METRICS = (("cpu_usage_active", "cpu"), ("mem_used_percent", "mem"), ("nvidia_smi_utilization_gpu", "gpu"),
              ("nvidia_smi_memory_used", "gpu_mem"), ("nvidia_smi_power_draw", "gpu_power"))


def level_name(lvl: dict) -> str:
    """The pool name part of a level key (cell keys contain "__" themselves)."""
    if lvl.get("name"):
        return lvl["name"]
    return lvl["level_key"][len(lvl["cell_key"]) + 2:].split("__")[0]


def level_records(run_id: str, cell_key: str, level_key: str) -> list[dict]:
    recs: list[dict] = []
    for key in common.list_keys(common.level_raw_prefix(run_id, cell_key, level_key) + "/"):
        recs += common.read_jsonl(key)
    return recs


def resources(target_id: str, start_ns: int, end_ns: int) -> dict:
    """Mean, P95 and max of the CloudWatch agent metrics over the level window (design 13)."""
    if end_ns <= start_ns:
        return {}
    start = dt.datetime.fromtimestamp(start_ns / 1e9, dt.UTC)
    end = dt.datetime.fromtimestamp(end_ns / 1e9, dt.UTC) + dt.timedelta(seconds=10)
    queries = []
    for i, (name, _) in enumerate(CW_METRICS):
        for stat in ("Average", "p95", "Maximum"):
            queries.append({"Id": f"m{i}_{stat.lower()}", "MetricStat": {
                "Metric": {"Namespace": "AmxGpuBench", "MetricName": name,
                           "Dimensions": [{"Name": "target_id", "Value": target_id}]},
                "Period": 10, "Stat": stat}, "ReturnData": True})
    resp = common.client("cloudwatch").get_metric_data(MetricDataQueries=queries, StartTime=start, EndTime=end)
    out: dict[str, dict] = {}
    for r in resp.get("MetricDataResults", []):
        idx, stat = r["Id"][1:].split("_", 1)
        label = CW_METRICS[int(idx)][1]
        vals = r.get("Values") or []
        if not vals:
            continue
        agg = max(vals) if stat in ("maximum", "p95") else sum(vals) / len(vals)
        out.setdefault(label, {})[{"average": "mean", "p95": "p95", "maximum": "max"}[stat]] = agg
    return out


def aggregate_level(event: dict, context: Any = None) -> dict:
    lvl = event["level"]
    recs = level_records(lvl["run_id"], lvl["cell_key"], lvl["level_key"])
    kind = "generation" if lvl["api_task"] == "generate" else "classifier"
    pay = lvl.get("payload", {})
    summary = metrics.summarize_level(recs, concurrency=int(lvl["conc"]), kind=kind,
                                      ignore_eos=bool(pay.get("ignore_eos")), max_new_tokens=pay.get("max_tokens"))
    summary.update({k: lvl.get(k) for k in ("run_id", "cell_key", "level_key", "name", "conc", "batch", "repeat",
                                             "workload", "accuracy", "target_id")})
    if recs:
        start = min(int(r["start_at_ns"]) for r in recs)
        end = max(int(r["wall_send_ns"]) + int(r["send_to_end_ns"]) for r in recs)
        summary["window_wall_ns"] = [start, end]
        try:
            summary["resources"] = resources(lvl["target_id"], start, end)
        except Exception as e:  # noqa: BLE001 - resource metrics are context, not a reason to fail the level
            summary["resources_error"] = str(e)[:300]
    perf = (event.get("perf") or {}).get("counts")
    if perf:
        busy, cyc = perf.get("exe.amx_busy"), perf.get("cycles")
        summary["amx_counter"] = {"exe.amx_busy": busy, "cycles": cyc,
                                  "ratio": (busy / cyc) if busy is not None and cyc else None}
    common.put_json(f"runs/{lvl['run_id']}/metrics/{lvl['cell_key']}/{lvl['level_key']}.json", summary)
    if lvl.get("accuracy"):
        rows = []
        for r in recs:
            ids = r.get("item_ids") or []
            outs = r.get("output") or ([{"text": r.get("output_text")}] if "output_text" in r else [])
            if r.get("error"):
                rows += [{"item_id": i, "error": r["error"]} for i in ids]
            elif len(ids) == 1:
                rows.append({"item_id": ids[0], "output": outs})
        name = level_name(lvl)                           # e.g. W1-short-accuracy
        common.put_jsonl_gz(f"runs/{lvl['run_id']}/predictions/{lvl['cell_key']}/{name}.jsonl.gz", rows)
    common.set_status(lvl["run_id"], f"LEVEL#{lvl['level_key']}", "done", n_ok=summary["n_ok"],
                      flags=summary["flags"])
    return {"n_ok": summary["n_ok"], "n_requests": summary["n_requests"], "flags": summary["flags"]}


def aggregate_run(event: dict, context: Any = None) -> dict:
    """Collect everything the run produced and render the reports (bench/report.py)."""
    from bench import report

    run_id = event["run_id"]
    prefix = common.run_prefix(run_id)
    resolved = common.get_json(f"{prefix}/resolved.json")
    inputs = report.RunInputs(
        resolved=resolved,
        metrics=[common.get_json(k) for k in common.list_keys(f"{prefix}/metrics/") if k.endswith(".json")],
        evidence={k.split("/")[-2]: common.get_json(k) for k in common.list_keys(f"{prefix}/amx/")
                  if k.endswith("evidence.json")},
        bringup={k.split("/")[-1][:-5]: common.get_json(k) for k in common.list_keys(f"{prefix}/bringup/")},
        quality={k.split("/")[-2]: common.read_jsonl(k) for k in common.list_keys(f"{prefix}/quality/")
                 if k.endswith("outputs.jsonl.gz")},
        predictions={"/".join(k.split("/")[-2:]): common.read_jsonl(k)
                     for k in common.list_keys(f"{prefix}/predictions/")},
        capacity={k.split("/")[-1][:-5]: common.get_json(k) for k in common.list_keys(f"{prefix}/capacity/")},
        hosts={k.split("/")[-1][:-5]: common.get_json(k) for k in common.list_keys(f"{prefix}/hosts/")},
        dataset_items=_dataset_items(resolved, "test"),
        dev_items=_dataset_items(resolved, "dev"),
        raw_loader=lambda cell_key, level_key: level_records(run_id, cell_key, level_key),
    )
    resolved.update(target_provenance({k.split("/")[-1][:-5]: common.get_json(k)
                                       for k in common.list_keys(f"{prefix}/targets/")}))
    files = report.render(report.build_summary(inputs))
    for name, body in files.items():
        common.client("s3").put_object(Bucket=common.bucket(), Key=f"{prefix}/report/{name}", Body=body)
    common.set_status(run_id, "META", "complete", report_prefix=f"{prefix}/report/")
    return {"report_prefix": f"s3://{common.bucket()}/{prefix}/report/", "files": sorted(files)}


def target_provenance(events: dict[str, list[dict]]) -> dict:
    """From runs/<run>/targets/<target>.json (written by target_control): the instances that
    served each target, their market (spot / on-demand) and the interruptions seen."""
    instances, lifecycle, interruptions, itypes = {}, {}, {}, {}
    for tid, evs in sorted(events.items()):
        launched = [e for e in evs if e.get("event") in ("launched", "started")]
        instances[tid] = sorted({e["instance_id"] for e in launched})
        lifecycle[tid] = launched[-1]["lifecycle"] if launched else None
        itypes[tid] = launched[-1].get("instance_type") if launched else None
        interruptions[tid] = [{k: e.get(k) for k in ("time", "instance_id", "state_reason", "cell")}
                              for e in evs if e.get("event") == "interrupted"]
    return {"instances": instances, "lifecycle": lifecycle, "interruptions": interruptions, "instance_types": itypes}


def _dataset_items(resolved: dict, split: str) -> list[dict] | None:
    """Labels for scoring: test items for every model, dev items for M3's threshold."""
    if resolved["run"].get("suite") != "dspm":
        return None
    return common.read_jsonl(f"datasets/dspm/v1/{split}.jsonl.gz")
