"""Hardware utilization: the CloudWatch queries, merging across hosts, the per-measurement
statistics and the sizing verdict; the API's time series and the summary backfill (moto)."""
from __future__ import annotations

import datetime as dt
import json

import pytest

from bench import utilization as U

T0 = dt.datetime(2026, 11, 18, 7, 20, tzinfo=dt.UTC)


def ts(i: int) -> dt.datetime:
    return T0 + dt.timedelta(seconds=10 * i)


def test_queries_search_by_target_id_at_host_level_and_never_name_a_host():
    q = U.queries("t2-m8i-4xl", 10)
    assert [x["Id"] for x in q] == ["cpu", "mem", "gpu", "gpu_mem", "gpu_mem_total", "gpu_power"]
    cpu = q[0]["Expression"]
    assert cpu.startswith("SEARCH('{AmxGpuBench,cpu,host,target_id} MetricName=\"cpu_usage_active\" cpu=\"cpu-total\" target_id=\"t2-m8i-4xl\"'")
    assert cpu.endswith("'Average', 10)") and "host=" not in cpu
    assert "'Maximum', 60)" in U.queries("t1-g6", 60, ("gpu_mem_total",))[0]["Expression"]
    assert 'target_id="x"' in U.queries('x"', 10, ("mem",))[0]["Expression"]          # a quote cannot break out of the filter
    scoped = U.queries("t2", 60, ("cpu", "mem"), hosts=["ip-10-40-1-1.ec2.internal", 'ip-10-40-2-2"'])
    assert [q["Id"] for q in scoped] == ["cpu__0", "cpu__1", "mem__0", "mem__1"]
    assert ' host="ip-10-40-1-1.ec2.internal" target_id="t2"' in scoped[0]["Expression"] and ' host="ip-10-40-2-2" ' in scoped[1]["Expression"]
    merged = U.merge([{"Id": "cpu__0", "Label": "a", "Timestamps": [ts(0)], "Values": [10.0]}, {"Id": "cpu__1", "Label": "b", "Timestamps": [ts(1)], "Values": [20.0]}])
    assert merged["cpu"]["v"] == [10.0, 20.0] and merged["cpu"]["hosts"] == 2


def test_merge_combines_hosts_by_timestamp_and_stats_are_of_period_averages():
    res = [
        {"Id": "cpu", "Label": "cpu-total ip-a t2", "Timestamps": [ts(0), ts(1), ts(2)], "Values": [10.0, 30.0, 50.0]},
        {"Id": "cpu", "Label": "cpu-total ip-b t2", "Timestamps": [ts(2), ts(3)], "Values": [70.0, 90.0]},   # a replacement host
        {"Id": "cpu", "Label": "cpu-total ip-old t2", "Timestamps": [], "Values": []},                         # an earlier run's host
        {"Id": "gpu_mem", "Label": "L4", "Timestamps": [ts(0), ts(1)], "Values": [1000.0, 3000.0]},
        {"Id": "gpu_mem_total", "Label": "L4", "Timestamps": [ts(0), ts(1)], "Values": [23034.0, 23034.0]},
    ]
    series = U.merge(res)
    assert series["cpu"]["v"] == [10.0, 30.0, 60.0, 90.0] and series["cpu"]["hosts"] == 2       # overlap at ts(2) averaged
    st = U.summarize(series)
    assert st["cpu"] == {"mean": 47.5, "p95": 90.0, "max": 90.0, "n": 4}
    assert st["gpu_mem"]["max"] == 3000.0 and st["gpu_mem"]["total"] == 23034.0 and "gpu_mem_total" not in st
    c = U.compact(series)
    assert c["t"] == [int(ts(i).timestamp()) for i in range(4)] and c["cpu"] == [10.0, 30.0, 60.0, 90.0]
    assert c["gpu_mem"] == [1000.0, 3000.0, None, None] and c["period_s"] == 10 and "gpu_mem_total" not in c
    assert U.summarize({}) == {} and U.compact({}) is None and U.stats([]) is None


def test_hostname_is_the_private_dns_name_the_agent_publishes():
    assert U.hostname("http://10.40.16.97:8000") == "ip-10-40-16-97.ec2.internal"
    assert U.hostname("10.40.1.2") == "ip-10-40-1-2.ec2.internal" and U.hostname("10.0.0.1", "eu-west-1") == "ip-10-0-0-1.eu-west-1.compute.internal"
    assert U.hostname(None) is None and U.hostname("http://host:8000") is None


def test_source_tags_travel_through_legacy_and_merge():
    old = U.legacy({"cpu": {"mean": 34.0, "p95": 97.0, "max": 100.0}})
    assert old["_source"] == "level-rollup" and old["cpu"] == {"mean": 34.0}
    new = U.legacy({"cpu": {"mean": 30.0, "p95": 40.0, "max": 50.0, "n": 5}, "_source": "level-10s-host"})
    assert new["_source"] == "level-10s-host" and new["cpu"]["p95"] == 40.0
    m = U.merge_stats([old, new])
    assert m["_source"] == "level-rollup" and m["cpu"]["mean"] == (34.0 * 1 + 30.0 * 5) / 6
    cw = {"cpu": {"mean": 20.0, "p95": 25.0, "max": 30.0, "n": 2}, "_source": "cloudwatch-60s-host", "_rough": True}
    m2 = U.merge_stats([cw, cw])
    assert m2["_source"] == "cloudwatch-60s-host" and m2["_rough"] is True and m2["cpu"]["n"] == 4
    v = U.sizing([{**row("cpu", 4, 1, 25.0), "util": {**row("cpu", 4, 1, 25.0)["util"], "_rough": True, "_source": "cloudwatch-60s-host"}}], CPU_SPEC)
    assert any("approximate" in n for n in v["notes"]) and not any("mixed into" in n for n in v["notes"])
    # The Qwen run's GPU: 83% at 1 request (166 s), 50% at 4 (44 s, rough), 25% at 16 (14 s, rough). Judged on the 166 s
    # measurement; the two short ones are named with their lengths, not used as the verdict.
    def gpu(conc, busy_pct, win, rough):
        r = row("gpu", conc, 1, busy_pct)
        r["util"].update({"_source": "cloudwatch-60s-host", "_window_s": win, **({"_rough": True} if rough else {})})
        return r
    v = U.sizing([gpu(1, 83.0, 166, False), gpu(4, 50.0, 44, True), gpu(16, 25.0, 14, True)], GPU_SPEC)
    assert v["verdict"] == "limit" and v["load_top"] == "1 request at a time" and v["util_top"] == 83.0
    assert "too briefly" in v["text"] and "4 requests at a time (44 s, read 50%)" in v["text"] and "16 requests at a time (14 s, read 25%)" in v["text"]
    assert not any("approximate" in n for n in v["notes"])
    assert U.legacy({"cpu": {"mean": 1.0, "n": 2}, "_rough": True, "_window_s": 14.0, "_source": "x"})["_window_s"] == 14.0
    v = U.sizing([{**row("cpu", 4, 1, 25.0), "util": {**row("cpu", 4, 1, 25.0)["util"], "_source": "level-rollup"}}], CPU_SPEC)
    assert any("mixed into these figures" in n for n in v["notes"])


def test_compact_thins_long_series_evenly():
    series = {"cpu": {"t": [float(i) for i in range(1000)], "v": [float(i % 100) for i in range(1000)], "hosts": 1}}
    c = U.compact(series, limit=100)
    assert len(c["t"]) == 100 and c["t"][0] == 0 and c["t"][-1] == 990 and c["period_s"] == 10


def test_merge_stats_pools_repeats_and_legacy_rows_keep_only_their_mean():
    a = {"cpu": {"mean": 20.0, "p95": 40.0, "max": 50.0, "n": 10}, "gpu_mem": {"mean": 1.0, "p95": 2.0, "max": 2.0, "n": 10, "total": 23034.0}}
    b = {"cpu": {"mean": 40.0, "p95": 60.0, "max": 90.0, "n": 30}}
    m = U.merge_stats([a, b, {}])
    assert m["cpu"] == {"mean": 35.0, "p95": 60.0, "max": 90.0, "n": 40} and m["gpu_mem"]["total"] == 23034.0
    assert U.merge_stats([{}, None]) == {}
    # Before 2026-11-18 p95/max were the busiest single core (100% on every CPU host): dropped, the mean stands.
    old = U.legacy({"cpu": {"mean": 34.0, "p95": 97.0, "max": 100.0}, "mem": {"mean": 7.0, "p95": 7.0, "max": 7.0}, "gpu": {}})
    assert old == {"cpu": {"mean": 34.0}, "mem": {"mean": 7.0}, "_source": "level-rollup"}
    assert U.merge_stats([old, U.legacy({"cpu": {"mean": 36.0, "max": 100.0}})])["cpu"] == {"mean": 35.0}
    assert U.legacy(None) == {} and U.legacy({"cpu": {"mean": 5.0, "p95": 6.0, "max": 7.0, "n": 3}})["cpu"]["p95"] == 6.0


def test_cpu_busy_is_the_share_of_physical_cores():
    assert U.cpu_scale(CPU_SPEC) == 2.0 and U.cpu_scale(GPU_SPEC) == 1.0 and U.cpu_scale(None) == 1.0
    r = row("cpu", 4, 1, 45.0)                       # 45% of 32 vCPUs = 90% of the 16 physical cores
    assert U.busy(r, "cpu", 2.0) == {"mean": 90.0, "p95": 100.0, "max": 100.0}
    assert U.busy(row("cpu", 4, 1, None), "cpu", 2.0) == {"mean": None, "p95": None, "max": None}


def row(side, conc, batch=1, busy=None, meets=True, engine="vllm", mem=None, gmem=None, total=None):
    label = "gpu" if side == "gpu" else "cpu"
    util = {}
    if busy is not None:
        util[label] = {"mean": busy, "p95": min(100.0, busy + 15), "max": 100.0, "n": 5}
    if mem is not None:
        util["mem"] = {"mean": mem, "p95": mem, "max": mem, "n": 5}
    if gmem is not None:
        util["gpu_mem"] = {"mean": gmem, "p95": gmem, "max": gmem, "n": 5, "total": total}
    return {"accelerator": side, "instance_type": "c8i.8xlarge" if side == "cpu" else "g6.2xlarge", "engine": engine, "precision": "bf16",
            "conc": conc, "batch": batch, "meets_user_sla": meets, "util": util or None}


CPU_SPEC = {"vcpus": 32, "memory_gib": 64.0, "accelerator": "cpu"}
GPU_SPEC = {"gpus": 1, "gpu_model": "L4", "gpu_memory_gib": 22.0, "accelerator": "gpu"}


def test_sizing_verdicts_read_from_the_heaviest_load_setting():
    over = U.sizing([row("cpu", 1, 8, 10.0), row("cpu", 1, 32, 14.0), row("cpu", 4, 32, 15.5)], CPU_SPEC)    # host %, x2 = cores
    assert over["verdict"] == "over" and over["load_top"] == "4 requests at a time, 32 inputs per request"
    assert "busy 31% of the time on average, 61% at its busiest" in over["text"] and "smaller CPU size" in over["text"]
    assert "barely moved" not in over["text"] and over["size"] == "32 vCPUs, 64 GiB"
    assert [p["conc"] for p in over["points"]] == [1, 1, 4] and over["points"][2]["util_mean"] == 31.0
    assert over["busy_basis"].startswith("share of its 16 physical cores") and any("16 physical cores" in n for n in over["notes"])

    flat = U.sizing([row("cpu", 1, 1, 6.0), row("cpu", 16, 1, 9.0)], CPU_SPEC)
    assert flat["verdict"] == "over" and "barely moved between the lightest (12%)" in flat["text"]
    busier_light = U.sizing([row("gpu", 1, 1, 31.0), row("gpu", 16, 1, 17.0)], GPU_SPEC)            # the Qwen run's GPU
    assert busier_light["verdict"] == "over" and "barely moved" not in busier_light["text"]
    assert "busier at the lightest load setting (31%)" in busier_light["text"] and "limit elsewhere" in busier_light["text"]

    good = U.sizing([row("gpu", 1, 1, 30.0), row("gpu", 16, 1, 62.0)], GPU_SPEC)
    assert good["verdict"] == "good" and "room left over" in good["text"] and good["size"] == "1x L4 with 22 GiB"

    limit = U.sizing([row("gpu", 16, 1, 92.0, meets=True)], GPU_SPEC)
    assert limit["verdict"] == "limit" and "still meeting your targets" in limit["text"]

    under = U.sizing([row("cpu", 16, 1, 48.0, meets=False, engine="thin-torch")], CPU_SPEC)     # 48% of vCPUs = 96% of cores
    assert under["verdict"] == "under" and "too small for this load" in under["text"] and under["util_top"] == 96.0
    assert U.sizing([row("gpu", 16, 1, 62.0)], GPU_SPEC)["busy_basis"] == "GPU busy time"

    unknown = U.sizing([row("cpu", 1, 1, None), row("cpu", 4, 1, None)], CPU_SPEC)
    assert unknown["verdict"] == "unknown" and unknown["with_samples"] == 0 and unknown["measurements"] == 2
    assert U.sizing([], None) is not None and U.sizing([], None)["verdict"] == "unknown"


def test_sizing_notes_memory_and_explains_vllm_reservation():
    v = U.sizing([row("gpu", 4, 1, 50.0, engine="vllm", gmem=20700.0, total=23034.0)], GPU_SPEC)
    assert any("vLLM reserves most" in n and "90%" in n for n in v["notes"])
    v = U.sizing([row("gpu", 4, 1, 50.0, engine="thin-torch", gmem=4000.0, total=23034.0)], GPU_SPEC)
    assert any("most of it unused" in n for n in v["notes"])
    v = U.sizing([row("cpu", 4, 1, 25.0, mem=91.0)], CPU_SPEC)
    assert any("Host memory peaked at 91%" in n for n in v["notes"])
    v = U.sizing([row("cpu", 4, 1, 25.0, mem=69.0, engine="vllm")], CPU_SPEC)
    assert any("vLLM reserves space for its cache" in n for n in v["notes"])


def test_sizing_for_rows_groups_by_series_gpu_first():
    rows = [{**row("cpu", 1, 1, 20.0), "run_id": "r"}, {**row("gpu", 1, 1, 10.0), "run_id": "r"}, {**row("cpu", 8, 1, 30.0), "run_id": "r"}]
    out = U.sizing_for_rows(rows, lambda r: f"{r['accelerator']}·{r['instance_type']}", lambda t: GPU_SPEC if t.startswith("g") else CPU_SPEC)
    assert [v["series"] for v in out] == ["gpu·g6.2xlarge", "cpu·c8i.8xlarge"] and out[1]["verdict"] == "good" and out[1]["measurements"] == 2


# ----------------------------------------------------------------------------- through the API

pytest.importorskip("fastapi")


def test_compare_carries_utilization_sizing_and_the_summary_backfills_old_reports(monkeypatch):
    import boto3
    import moto
    from fastapi.testclient import TestClient

    from webui.api import main
    from webui.api.main import app

    c = TestClient(app)
    bucket = "amx-gpu-bench-123456789012-us-east-1"
    rows = []
    for conc in (1, 16):
        rows.append({"model": "m3-roberta-large-mnli", "slot": "M3", "license": "mit", "target": "t2-m8i-4xl", "target_label": "x",
                     "instance_type": "c8i.8xlarge", "accelerator": "cpu", "engine": "vllm", "precision": "bf16", "variant": None,
                     "workload": None, "bucket": None, "conc": conc, "batch": 8, "repeats": 2, "kind": "classifier", "n_ok": 50,
                     "n_requests": 50, "e2e_p50": 40.0, "e2e_p99": 80.0, "cls_s": 100.0 * conc, "usd_per_hour": 1.5, "error_pct": 0,
                     "amx": "VERIFIED", "usd_per_1m_items": 4.0, "cell_key": "m3__t2__bf16", "level_group": f"m3__t2__bf16__short__c{conc}__b8"})
    summary = {"run_id": "r-util", "rows": rows, "smoke": True, "run": {"suite": "standard", "concurrency": [1, 16], "instance_types": {"t2-m8i-4xl": "c8i.8xlarge"}}}
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)
        s3.put_object(Bucket=bucket, Key="runs/r-util/report/summary.json", Body=json.dumps(summary).encode())
        for i, (conc, rep, busy) in enumerate(((1, 0, 10.0), (1, 1, 14.0), (16, 0, 55.0), (16, 1, 61.0))):
            m = {"cell_key": "m3__t2__bf16", "level_key": f"m3__t2__bf16__short__c{conc}__b8__r{rep}", "target_id": "t2-m8i-4xl",
                 "name": "short", "conc": conc, "batch": 8, "repeat": rep, "window_wall_ns": [1_763_452_800_000_000_000 + i * 10**11, 1_763_452_860_000_000_000 + i * 10**11],
                 "resources": {"cpu": {"mean": busy, "p95": busy + 20, "max": 100.0, "n": 6}, "mem": {"mean": 8.0, "p95": 9.0, "max": 9.0, "n": 6}}}
            s3.put_object(Bucket=bucket, Key=f"runs/r-util/metrics/m3__t2__bf16/{m['level_key']}.json", Body=json.dumps(m).encode())
        s3.put_object(Bucket=bucket, Key="runs/r-util/targets/t2-m8i-4xl.json", Body=json.dumps([
            {"event": "launched", "instance_id": "i-aaa"}, {"event": "replaced", "instance_id": "i-bbb"}]).encode())
        s3.put_object(Bucket=bucket, Key="hosts/t2-m8i-4xl/i-aaa-20261118T070000Z.json", Body=json.dumps({"uname": "Linux ip-10-40-1-1.ec2.internal 6.12 x86_64"}).encode())
        s3.put_object(Bucket=bucket, Key="hosts/t2-m8i-4xl/i-bbb-20261118T073000Z.json", Body=json.dumps({"uname": "Linux ip-10-40-2-2.ec2.internal 6.12 x86_64"}).encode())
        monkeypatch.setattr(main, "_outputs", lambda: {"BucketName": bucket, "RunStateMachineArn": "arn:aws:states:us-east-1:1:stateMachine:x"})
        for k in list(main._CACHE):
            if k.startswith(("summary:", "metrics:", "util:", "hosts:")):
                main._CACHE.pop(k)
        out = c.post("/api/compare", json={"run_ids": ["r-util"]}).json()
        g = out["analysis"]["groups"][0]
        # backfilled from the metrics files, merged over repeats, CPU as the share of c8i.8xlarge's 16 physical cores (x2)
        assert g["points"][0]["util"]["busy_mean"] == 24.0 and g["points"][1]["util"]["busy_p95"] == 100.0 and g["points"][0]["util"]["cpu_mean"] == 12.0
        assert g["points"][1]["util"]["mem_max"] == 9.0
        s = g["sizing"]
        assert len(s) == 1 and s[0]["verdict"] == "limit" and s[0]["util_top"] == 100.0 and s[0]["side"] == "cpu"
        assert any("mixed into these figures" in n for n in s[0]["notes"])      # moto's CloudWatch had nothing: the rollup figures stand, and say so
        assert main._CACHE["lvlutil:r-util:m3__t2__bf16__short__c1__b8__r0"] is None
        assert s[0]["load_top"] == "16 requests at a time, 8 inputs per request" and s[0]["size"] == "32 vCPUs, 64 GiB"
        # The time series endpoint: moto has no SEARCH data, so each hardware option reports an empty series, never a 500.
        u = c.get("/api/runs/r-util/utilization").json()
        assert u["period_s"] in (60, 300, 3600) and [t["target"] for t in u["targets"]] == ["t2-m8i-4xl"]
        t = u["targets"][0]
        assert t["instance_type"] == "c8i.8xlarge" and t["accelerator"] == "cpu" and len(t["windows"]) == 4
        assert t["scoped_to_run"] and t["instances"] == 2                                  # the run's own two hosts, by name
        assert main._run_hosts("r-util", "t2-m8i-4xl") == ["ip-10-40-1-1.ec2.internal", "ip-10-40-2-2.ec2.internal"]
        assert main._run_hosts("r-util", "t9-none") == []
        assert [w["conc"] for w in t["windows"]] == [1, 1, 16, 16] and t["windows"][0]["level"] == "m3__t2__bf16__short__c1__b8"
        assert c.get("/api/runs/no.such/utilization").status_code == 404 and c.get("/api/runs/bad id/utilization").status_code == 400


# ----------------------------------------------------------------------------- what to use instead

def _cat():
    from bench import pairing
    return pairing.load_catalog()["instances"]


def test_idle_cpu_gets_the_smallest_same_family_size_the_load_still_fits():
    cat = _cat()
    # m8i.4xlarge (16 vCPUs = 8 cores) 30% busy of its cores, memory peaked at 20%: 8 cores x 0.3 / 0.6 = 4 cores -> m8i.2xlarge
    s = U.suggest("over", "m8i.4xlarge", cat["m8i.4xlarge"], cat, busy_top=30.0, mem_pct=20.0)
    assert s["instance_type"] == "m8i.2xlarge" and s["kind"] == "smaller" and s["usd_delta_per_hour"] < 0
    assert "about 60% busy" in s["why"] and "$0.42/h instead of $0.85/h" in s["why"]
    # memory binds: 69% of 64 GiB x 1.3 = 57 GiB; no smaller m8i has it
    s = U.suggest("over", "m8i.4xlarge", cat["m8i.4xlarge"], cat, busy_top=30.0, mem_pct=69.0)
    assert s["instance_type"] is None and s["kind"] == "none" and "no smaller m8i size has the memory" in s["why"]
    # the smallest size already: nothing to suggest
    assert U.suggest("over", "m8i.large", cat["m8i.large"], cat, busy_top=10.0, mem_pct=5.0) is None


def test_saturated_cpu_gets_the_next_size_up_and_a_fully_used_one_a_growth_note():
    cat = _cat()
    s = U.suggest("under", "c8i.8xlarge", cat["c8i.8xlarge"], cat, busy_top=96.0, mem_pct=30.0)
    assert s["instance_type"] == "c8i.16xlarge" and s["kind"] == "bigger" and "64 vCPUs" in s["why"]        # the size that doubles the cores, not 12xlarge
    s = U.suggest("limit", "m8i.4xlarge", cat["m8i.4xlarge"], cat, busy_top=85.0, mem_pct=30.0)
    assert s["instance_type"] == "m8i.8xlarge" and s["kind"] == "growth" and s["why"].startswith("Keep m8i.4xlarge")
    keep = U.suggest("good", "m8i.4xlarge", cat["m8i.4xlarge"], cat, busy_top=60.0, mem_pct=30.0)
    assert keep["instance_type"] == "m8i.4xlarge" and keep["kind"] == "keep" and keep["usd_delta_per_hour"] == 0
    g = U.suggest("limit", "g6.2xlarge", cat["g6.2xlarge"], cat, busy_top=85.0, mem_pct=30.0)
    assert g["instance_type"] == "g6e.xlarge" and g["kind"] == "growth" and g["why"].startswith("Keep g6.2xlarge")
    assert U.suggest("unknown", "m8i.4xlarge", cat["m8i.4xlarge"], cat, busy_top=None, mem_pct=None) is None
    assert U.suggest("over", "nope", None, cat, busy_top=10.0, mem_pct=1.0) is None


def test_idle_gpu_keeps_the_chip_and_shrinks_the_host_saturated_gpu_gets_a_stronger_chip():
    cat = _cat()
    s = U.suggest("over", "g6.2xlarge", cat["g6.2xlarge"], cat, busy_top=17.0, mem_pct=14.0, host_cpu_pct=10.0)
    assert s["instance_type"] == "g6.xlarge" and s["kind"] == "smaller" and "same L4" in s["why"] and "more requests at once" in s["why"]
    # the host itself was busy: no smaller size has the vCPUs
    s = U.suggest("over", "g6.2xlarge", cat["g6.2xlarge"], cat, busy_top=17.0, mem_pct=14.0, host_cpu_pct=70.0)
    assert s["instance_type"] is None and "same GPU" in s["why"]
    s = U.suggest("under", "g6.2xlarge", cat["g6.2xlarge"], cat, busy_top=97.0, mem_pct=20.0)
    assert s["instance_type"] == "g6e.xlarge" and s["kind"] == "stronger" and "L40S" in s["why"] and "would not help" in s["why"]
    assert U.suggest("under", "g6e.48xlarge", cat["g6e.48xlarge"], cat, busy_top=97.0, mem_pct=20.0) is None    # nothing stronger listed


def test_sizing_carries_the_suggestion_and_compare_judges_its_pairing():
    cat = _cat()
    rows = [{**row("cpu", 1, 1, 10.0), "instance_type": "m8i.4xlarge", "run_id": "r"}, {**row("cpu", 16, 1, 15.0), "instance_type": "m8i.4xlarge", "run_id": "r"}]
    v = U.sizing(rows, cat["m8i.4xlarge"], cat)
    assert v["verdict"] == "over" and v["suggestion"]["instance_type"] == "m8i.2xlarge" and v["run_id"] == "r"
    assert U.sizing(rows, cat["m8i.4xlarge"])["suggestion"] is None                     # no catalog: no suggestion
    from webui.api import compare
    common = {"kind": "generation", "model": "m1-qwen2.5-3b", "slot": "M1", "workload": None, "bucket": None, "e2e_p99": 100.0, "error_pct": 0,
              "amx": "VERIFIED", "quality_flag": None, "run_id": "r", "variant": None}
    g6 = {**common, "accelerator": "gpu", "target": "t1-g6", "instance_type": "g6.2xlarge", "engine": "vllm", "precision": "bf16", "conc": 16, "batch": 1,
          "util": {"gpu": {"mean": 83.0, "p95": 100.0, "max": 100.0, "n": 5}}, "usd_per_hour": 0.9776, "agg_out_tok_s": 500.0, "amx": "n/a"}
    cpu_rows = [{**common, **r, "target": "t2-m8i-4xl", "usd_per_hour": 0.847, "agg_out_tok_s": 100.0} for r in rows]
    out = compare.analyze([{"run_id": "r", "rows": [g6, *cpu_rows], "run": {"suite": "standard"}}], {"max_error_pct": 1.0})
    z = next(x for x in out["groups"][0]["sizing"] if x["side"] == "cpu")
    assert z["suggestion"]["instance_type"] == "m8i.2xlarge"
    assert z["suggestion"]["pairing"]["other"] == "g6.2xlarge" and z["suggestion"]["pairing"]["fair"] is False      # 0.98 / 0.42 = 2.3x
    assert "outside" in z["suggestion"]["pairing"]["text"]
