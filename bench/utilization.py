"""Hardware utilization: what the CloudWatch agent on every test host measured while the
benchmark ran, and what it says about the instance size.

The agent (host/cloudwatch-agent.json) publishes to the `AmxGpuBench` namespace every 10 s with a
`target_id` dimension: CPU busy per core and for the whole host (`cpu=cpu-total`), memory used,
and for GPUs the NVIDIA utilization, memory and power. Hosts come and go (one per hardware option
per run, a new one after a Spot interruption), so metrics are found with SEARCH expressions by
target_id and merged by timestamp; the queries never name a host.

Two uses:
* `aggregate_level` (Lambda) summarizes the window of one measurement: mean, 95th percentile and
  max of the host-level 10-second averages, plus a compact time series kept with the metrics.
* The web app reads a whole run's time series on demand (1-minute points; CloudWatch keeps them
  15 days, 5-minute ones 63 days) and derives a sizing verdict per hardware option from the
  utilization at the heaviest load setting: idle hardware is bigger than the load needs, hardware
  at its limit while missing the targets is too small.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

NAMESPACE = "AmxGpuBench"
# label -> (metric name, dimension schema of the raw metric, extra filter, statistic)
METRICS: dict[str, tuple[str, str, str, str]] = {
    "cpu": ("cpu_usage_active", "cpu,host,target_id", ' cpu="cpu-total"', "Average"),
    "mem": ("mem_used_percent", "host,target_id", "", "Average"),
    "gpu": ("nvidia_smi_utilization_gpu", "arch,host,index,name,target_id", "", "Average"),
    "gpu_mem": ("nvidia_smi_memory_used", "arch,host,index,name,target_id", "", "Average"),
    "gpu_mem_total": ("nvidia_smi_memory_total", "arch,host,index,name,target_id", "", "Maximum"),
    "gpu_power": ("nvidia_smi_power_draw", "arch,host,index,name,target_id", "", "Average"),
}
PERCENT = ("cpu", "mem", "gpu")
# Sizing thresholds on the mean utilization at the heaviest load setting (per cent).
IDLE_BELOW = 35.0
BUSY_ABOVE = 80.0
SERIES_POINTS = 360          # at most this many samples kept per measurement (an hour at 10 s)


def queries(target_id: str, period: int, labels: tuple[str, ...] | None = None, hosts: list[str] | None = None) -> list[dict]:
    """GetMetricData queries for one hardware option, one SEARCH per metric; with `hosts`, one per
    metric and host (the run's own instances, so a concurrent run on the same hardware option is
    not mixed in). Ids are `<label>` or `<label>__<i>`; `merge` reads the label back."""
    tid = target_id.replace('"', "")
    out = []
    for label in labels or tuple(METRICS):
        name, schema, extra, stat = METRICS[label]
        for i, host in enumerate(hosts or [None]):
            hf = f' host="{host.replace(chr(34), "")}"' if host else ""
            expr = f"SEARCH('{{{NAMESPACE},{schema}}} MetricName=\"{name}\"{extra}{hf} target_id=\"{tid}\"', '{stat}', {period})"
            out.append({"Id": label if host is None else f"{label}__{i}", "Expression": expr, "ReturnData": True})
    return out


def hostname(private_ip_or_endpoint: str | None, region: str = "us-east-1") -> str | None:
    """The CloudWatch agent's `host` dimension for an instance, from its private IP (or the
    http://ip:port endpoint a level carries): EC2's private DNS name, ip-A-B-C-D.ec2.internal in
    us-east-1 and ip-A-B-C-D.<region>.compute.internal elsewhere."""
    import re

    m = re.search(r"(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})", private_ip_or_endpoint or "")
    if not m:
        return None
    suffix = "ec2.internal" if region == "us-east-1" else f"{region}.compute.internal"
    return f"ip-{'-'.join(m.groups())}.{suffix}"


def merge(results: list[dict]) -> dict[str, dict[str, Any]]:
    """SEARCH returns one series per host; hosts active in the window are merged by timestamp
    (the mean where two overlap). -> label -> {"t": [epoch s], "v": [value], "hosts": n}"""
    acc: dict[str, dict[float, list[float]]] = {}
    hosts: dict[str, set[str]] = {}
    for r in results:
        label = r["Id"].split("__", 1)[0]
        ts, vals = r.get("Timestamps") or [], r.get("Values") or []
        if vals:
            hosts.setdefault(label, set()).add(r.get("Label", ""))
        for t, v in zip(ts, vals, strict=False):
            key = t.timestamp() if isinstance(t, dt.datetime) else float(t)
            acc.setdefault(label, {}).setdefault(key, []).append(float(v))
    out = {}
    for label, by_t in acc.items():
        ts_sorted = sorted(by_t)
        out[label] = {"t": ts_sorted, "v": [sum(by_t[t]) / len(by_t[t]) for t in ts_sorted], "hosts": len(hosts.get(label, ()))}
    return out


def stats(values: list[float]) -> dict[str, float] | None:
    """Mean, 95th percentile and max of a series of period averages."""
    if not values:
        return None
    s = sorted(values)
    return {"mean": sum(s) / len(s), "p95": s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))], "max": s[-1], "n": len(s)}


def summarize(series: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """resources: label -> stats, for the labels that had samples (gpu_mem_total folded into gpu_mem)."""
    out = {}
    for label, s in series.items():
        if label == "gpu_mem_total":
            continue
        st = stats(s["v"])
        if st:
            out[label] = st
    total = series.get("gpu_mem_total")
    if total and total["v"] and "gpu_mem" in out:
        out["gpu_mem"]["total"] = max(total["v"])
    return out


def compact(series: dict[str, dict[str, Any]], limit: int = SERIES_POINTS) -> dict[str, Any] | None:
    """The time series kept with a measurement: one timestamp axis, per-label values (null when a
    label had no sample at that second), thinned evenly to `limit` points."""
    labels = [k for k in ("cpu", "mem", "gpu", "gpu_mem", "gpu_power") if k in series and series[k]["v"]]
    if not labels:
        return None
    ts = sorted({t for k in labels for t in series[k]["t"]})
    if len(ts) > limit:
        step = len(ts) / limit
        ts = [ts[int(i * step)] for i in range(limit)]
    lookup = {k: dict(zip(series[k]["t"], series[k]["v"], strict=False)) for k in labels}
    return {"t": [int(t) for t in ts], **{k: [None if lookup[k].get(t) is None else round(lookup[k][t], 1) for t in ts] for k in labels},
            "period_s": _period(ts)}


def _period(ts: list[float]) -> int | None:
    if len(ts) < 2:
        return None
    gaps = sorted(b - a for a, b in zip(ts, ts[1:], strict=False) if b > a)
    return int(gaps[len(gaps) // 2]) if gaps else None


def legacy(resources: dict | None) -> dict:
    """Resources written before 2026-11-18 (no sample count `n`) came from the per-core rollup: their
    mean is the host average and stands, but p95 and max are the busiest single core, not the host.
    Keep only what is true."""
    out = {}
    for label, v in (resources or {}).items():
        if not isinstance(v, dict) or v.get("mean") is None:
            continue
        out[label] = v if "n" in v else {k: v[k] for k in ("mean", "total") if k in v}
    if out:
        out["_source"] = (resources or {}).get("_source") or "level-rollup"
        for k in ("_rough", "_window_s"):
            if (resources or {}).get(k) is not None:
                out[k] = resources[k]
    return out


def merge_stats(parts: list[dict[str, dict[str, float]] | None]) -> dict[str, dict[str, float]]:
    """Resources of repeated measurements pooled: mean of means (weighted by samples when known),
    max of the p95s and maxes that exist (none when no part has them)."""
    out: dict = {}
    sources = {p["_source"] for p in parts if p and isinstance(p.get("_source"), str)}
    rough = any(p.get("_rough") for p in parts if p)
    for label in {k for p in parts for k in (p or {}) if not k.startswith("_")}:
        xs = [p[label] for p in parts if p and isinstance(p.get(label), dict) and p[label].get("mean") is not None]
        if not xs:
            continue
        w = [x.get("n") or 1 for x in xs]
        agg: dict[str, float] = {"mean": sum(x["mean"] * n for x, n in zip(xs, w, strict=False)) / sum(w)}
        for k in ("p95", "max"):
            vals = [x[k] for x in xs if x.get(k) is not None]
            if vals:
                agg[k] = max(vals)
        if any(x.get("n") for x in xs):
            agg["n"] = sum(x.get("n") or 0 for x in xs)
        if any("total" in x for x in xs):
            agg["total"] = max(x["total"] for x in xs if "total" in x)
        out[label] = agg
    if out and sources:
        out["_source"] = "level-rollup" if "level-rollup" in sources else sorted(sources)[0]
    if out and rough:
        out["_rough"] = True
    windows = [p["_window_s"] for p in parts if p and p.get("_window_s") is not None]
    if out and windows:
        out["_window_s"] = min(windows)
    return out


THREADS_PER_CORE = 2         # every Intel instance here: two vCPUs per physical core


def cpu_scale(spec: dict | None) -> float:
    """The CPU engines run one thread per PHYSICAL core (OpenVINO and PyTorch by default, vLLM with
    OMP_THREADS_BIND=auto), so a fully loaded 16-vCPU host reads 50% in `cpu_usage_active`. CPU busy
    is therefore reported as the share of the physical cores: host average x threads per core."""
    if spec and spec.get("accelerator") == "cpu" and (spec.get("vcpus") or 0) > 1:
        return float(THREADS_PER_CORE)
    return 1.0


def busy(r: dict, label: str, scale: float) -> dict[str, float | None]:
    """mean/p95/max of the side's own processor for a row, scaled to physical cores for CPUs."""
    u = (r.get("util") or {}).get(label) or {}
    return {k: (min(100.0, u[k] * scale) if u.get(k) is not None else None) for k in ("mean", "p95", "max")}


# ----------------------------------------------------------------------------- sizing verdict

def _load_key(r: dict) -> tuple[int, int]:
    return (int(r.get("conc") or 1), int(r.get("batch") or 1))


def _load_text(r: dict) -> str:
    c, b = _load_key(r)
    s = f"{c} request{'s' if c != 1 else ''} at a time"
    return s + (f", {b} inputs per request" if b > 1 else "")


def sizing(rows: list[dict], spec: dict | None, catalog: dict | None = None) -> dict | None:
    """One hardware option's verdict from its measurements (rows of one series: same model,
    hardware, software and number format). `spec` is the instance catalog entry; with the whole
    `catalog`, the verdict carries which instance to use instead."""
    side = rows[0]["accelerator"] if rows else None
    label = "gpu" if side == "gpu" else "cpu"
    scale = cpu_scale(spec) if side == "cpu" else 1.0
    with_util = [r for r in rows if busy(r, label, scale)["mean"] is not None]
    engine = rows[0].get("engine") if rows else None
    cores = int((spec or {}).get("vcpus") or 0) // THREADS_PER_CORE if scale > 1 else None
    base = {"side": side, "instance_type": rows[0].get("instance_type") if rows else None, "engine": engine,
            "precision": rows[0].get("precision") if rows else None, "measurements": len(rows), "with_samples": len(with_util),
            "busy_basis": (f"share of its {cores} physical cores (the serving software runs one thread per core)" if cores
                           else "GPU busy time" if side == "gpu" else "host average"),
            "points": [{"conc": r.get("conc"), "batch": r.get("batch") or 1, "meets": bool(r.get("meets_user_sla")),
                        "util_mean": busy(r, label, scale)["mean"], "util_p95": busy(r, label, scale)["p95"],
                        "mem_pct": ((r.get("util") or {}).get("mem") or {}).get("max")}
                       for r in sorted(rows, key=_load_key)]}
    if not with_util:
        return {**base, "verdict": "unknown", "label": "No utilization samples",
                "text": "The host published no utilization for these measurements (a very short measurement, or the agent had "
                        "not flushed yet), so nothing can be said about the size."}
    # A measurement shorter than CloudWatch's 1-minute points (a quick check's heaviest setting can last
    # 15 s) reads low because the point mixes in idle time: judge on the reliable measurements and say
    # which heavier ones were too short to read.
    reliable = [r for r in with_util if not (r.get("util") or {}).get("_rough")]
    pool = reliable or with_util
    top = max(pool, key=_load_key)
    low = min(pool, key=_load_key)
    too_short = sorted((r for r in with_util if r not in pool and _load_key(r) > _load_key(top)), key=_load_key)
    u_top = busy(top, label, scale)["mean"]
    p95_top = busy(top, label, scale)["p95"]
    u_low = busy(low, label, scale)["mean"]
    meets = bool(top.get("meets_user_sla"))
    what = "GPU" if side == "gpu" else "CPU"
    size_words = ""
    if spec:
        size_words = (f" ({spec.get('gpus', 1)}x {spec.get('gpu_model')} with {spec.get('gpu_memory_gib', 0):.0f} GiB)" if side == "gpu"
                      else f" ({spec.get('vcpus')} vCPUs, {spec.get('memory_gib', 0):.0f} GiB)")
    head = f"At the heaviest load setting ({_load_text(top)}) the {what} was busy {u_top:.0f}% of the time on average"
    head += f", {p95_top:.0f}% at its busiest" if p95_top is not None else ""
    if u_top < IDLE_BELOW:
        verdict, lbl = "over", "Bigger than this load needs"
        text = (f"{head}. It spent most of the time waiting for work: for this load a smaller {what} size would very "
                f"likely give the same result for less money, or the test should send more requests at once to find where "
                f"this size tops out.")
    elif u_top <= BUSY_ABOVE:
        verdict, lbl = "good", "A good match for this load"
        text = (f"{head}: working hard with room left over. The size fits this load; there is headroom for a busier day.")
    elif meets:
        verdict, lbl = "limit", "Fully used, still meeting the targets"
        text = (f"{head}: at its limit while still meeting your targets. A good match for exactly this load, with no room "
                f"to grow; the next size up would be needed for more traffic.")
    else:
        verdict, lbl = "under", "Too small for this load"
        text = (f"{head}: saturated, and the measurement missed your targets. This size is too small for this load; the next "
                f"size up (or fewer requests at once) is needed.")
    if len(with_util) > 1 and abs(u_top - u_low) < 10 and u_top < IDLE_BELOW:
        text += (f" Utilization barely moved between the lightest ({u_low:.0f}%) and the heaviest load setting, which says the load "
                 f"settings tested are all light for this {what}.")
    elif len(with_util) > 1 and u_low >= u_top + 10:
        text += (f" It was busier at the lightest load setting ({u_low:.0f}%) than at the heaviest: as requests piled up the {what} "
                 f"waited more, which points to a limit elsewhere (memory bandwidth, the serving software's scheduling, or the "
                 f"request pipeline) rather than to a lack of {what} capacity.")
    if too_short:
        heavier = ", ".join(f"{_load_text(r)} ({(r.get('util') or {}).get('_window_s') or 0:.0f} s, read {busy(r, label, scale)['mean']:.0f}%)"
                            for r in too_short)
        text += (f" Heavier settings ran too briefly for CloudWatch's 1-minute points to read them reliably: {heavier}; those "
                 f"figures understate how busy the {what} was, so the verdict rests on the longest heavy measurement.")
    notes = []
    srcs = {(r.get("util") or {}).get("_source") for r in with_util}
    if not reliable and any((r.get("util") or {}).get("_rough") for r in with_util):
        notes.append("Every measurement here was short (under 90 s) and was read back from 1-minute CloudWatch points, so "
                     "the busy figures are approximate and lean low.")
    if "level-rollup" in srcs:
        notes.append("Recorded before 2026-11-18 without the host's name: if another instance ran under the same hardware option "
                     "at the same time, its idle or busy time is mixed into these figures.")
    if cores:
        notes.append(f"CPU busy is the share of the {cores} physical cores: the serving software runs one thread per core, so the "
                     f"{spec.get('vcpus')} vCPUs can never read above 50% in raw host terms; 100% here means every core was busy.")
    mem = (top.get("util") or {}).get("mem") or {}
    mem_val = mem.get("max", mem.get("mean"))
    if mem_val is not None and mem_val > 85:
        notes.append(f"Host memory peaked at {mem_val:.0f}%: close to full.")
    elif side == "cpu" and engine == "vllm" and mem_val is not None and mem_val > 50:
        notes.append(f"Host memory reached {mem_val:.0f}%; vLLM reserves space for its cache up front, so this is by design, "
                     f"not a sign of running out.")
    gmem = (top.get("util") or {}).get("gpu_mem") or {}
    gmem_val = gmem.get("max", gmem.get("mean"))
    if side == "gpu" and gmem_val is not None and gmem.get("total"):
        pct = 100 * gmem_val / gmem["total"]
        if engine == "vllm":
            notes.append(f"GPU memory in use peaked at {pct:.0f}% of {gmem['total'] / 1024:.0f} GiB; vLLM reserves most of the GPU's "
                         f"memory up front for its cache, so a high figure here is by design, not a sign of running out.")
        elif pct > 90:
            notes.append(f"GPU memory peaked at {pct:.0f}% of {gmem['total'] / 1024:.0f} GiB: nearly full.")
        elif pct < 25:
            notes.append(f"GPU memory peaked at {pct:.0f}% of {gmem['total'] / 1024:.0f} GiB: most of it unused.")
    suggestion = None
    if catalog:
        host_cpu = ((top.get("util") or {}).get("cpu") or {}).get("mean")
        suggestion = suggest(verdict, base["instance_type"], spec, catalog, busy_top=u_top, mem_pct=mem_val,
                             host_cpu_pct=host_cpu if side == "gpu" else None)
    return {**base, "verdict": verdict, "label": lbl, "size": size_words.strip(" ()"), "util_top": u_top, "util_p95_top": p95_top,
            "util_low": u_low, "load_top": _load_text(top), "meets_top": meets, "text": text, "notes": notes,
            "run_id": rows[0].get("run_id"), "suggestion": suggestion}


# ----------------------------------------------------------------------------- what to use instead

GPU_RANK = {"T4": 1, "L4": 2, "A10G": 3, "L40S": 4, "A100": 5, "H100": 6, "H200": 7, "B200": 8}   # rough compute order
TARGET_BUSY = 60.0           # a smaller size is chosen so the same load would run it about this busy
MEM_HEADROOM = 1.3           # host memory the smaller size must still have: peak use x this


def _size_rank(spec: dict) -> tuple:
    return (spec.get("gpus") or 0, spec.get("vcpus") or 0, spec.get("memory_gib") or 0)


def suggest(verdict: str, instance_type: str | None, spec: dict | None, catalog: dict, *, busy_top: float | None,
            mem_pct: float | None, host_cpu_pct: float | None = None) -> dict | None:
    """Which instance to use instead, from the catalog: for an idle machine the smallest same-family
    size the load would still fit (CPU: cores for ~60% busy and memory with headroom; GPU: the same
    GPU chip with fewer host cores); for a saturated one the next size up (CPU) or the cheapest
    stronger GPU. None when the current size is the right one or nothing in the catalog fits."""
    if not spec or not instance_type or instance_type not in catalog:
        return None
    side = spec.get("accelerator")
    fam = spec.get("family")
    price = spec["usd_per_hour"]

    def pack(t: str, why: str, kind: str) -> dict:
        c = catalog[t]
        return {"instance_type": t, "kind": kind, "why": why, "usd_per_hour": c["usd_per_hour"], "usd_delta_per_hour": c["usd_per_hour"] - price,
                "vcpus": c.get("vcpus"), "memory_gib": c.get("memory_gib"), "gpus": c.get("gpus") or 0, "gpu_model": c.get("gpu_model"),
                "gpu_memory_gib": c.get("gpu_memory_gib"), "pct_of_current": 100 * c["usd_per_hour"] / price}

    same_family = sorted(((t, c) for t, c in catalog.items() if c.get("family") == fam and c.get("usd_per_hour")), key=lambda x: _size_rank(x[1]))
    rank = GPU_RANK.get(spec.get("gpu_model") or "", 0)
    stronger = sorted(((t, c) for t, c in catalog.items() if c.get("accelerator") == "gpu" and c.get("usd_per_hour")
                       and GPU_RANK.get(c.get("gpu_model") or "", 0) > rank and (c.get("gpus") or 0) == 1
                       and (c.get("vcpus") or 0) >= (spec.get("vcpus") or 0) / 2),
                      key=lambda x: (not x[1].get("current_generation", True), x[1]["usd_per_hour"]))
    if verdict == "good":
        return pack(instance_type, f"Keep {instance_type}: it fits this load with headroom.", "keep")
    if verdict == "limit":
        if side == "cpu":
            up = [(t, c) for t, c in same_family if _size_rank(c) > _size_rank(spec)]
            doubled = [(t, c) for t, c in up if (c.get("vcpus") or 0) >= 2 * (spec.get("vcpus") or 0)]
            if up:
                t, c = (doubled or up)[0]
                return pack(t, f"Keep {instance_type} for this load. For more traffic the next size up is {t} ({c['vcpus']} vCPUs, "
                               f"${c['usd_per_hour']:.2f}/h).", "growth")
        elif stronger:
            t, c = stronger[0]
            return pack(t, f"Keep {instance_type} for this load. For more traffic a bigger {fam} size would not help (same "
                           f"{spec.get('gpu_model')}); the next stronger GPU is the {c.get('gpu_model')} in {t} (${c['usd_per_hour']:.2f}/h).", "growth")
        return pack(instance_type, f"Keep {instance_type}: fully used for this load, nothing stronger in the catalog.", "keep")
    if verdict == "over" and busy_top is not None:
        mem_need = ((mem_pct or 0) / 100) * (spec.get("memory_gib") or 0) * MEM_HEADROOM
        if side == "cpu":
            cores_need = max(1.0, (spec.get("vcpus") or 2) / THREADS_PER_CORE * busy_top / TARGET_BUSY)
            fits = [(t, c) for t, c in same_family if _size_rank(c) < _size_rank(spec)
                    and (c.get("vcpus") or 0) / THREADS_PER_CORE >= cores_need and (c.get("memory_gib") or 0) >= mem_need]
            if fits:
                t, c = fits[0]
                return pack(t, f"{t} ({c['vcpus']} vCPUs, {c['memory_gib']:.0f} GiB) would run this load about {min(100, busy_top * (spec['vcpus'] / c['vcpus'])):.0f}% "
                               f"busy and still holds the model with room (memory peaked at {mem_pct or 0:.0f}% here). It costs "
                               f"${c['usd_per_hour']:.2f}/h instead of ${price:.2f}/h.", "smaller")
            if any(_size_rank(c) < _size_rank(spec) for _, c in same_family):
                return {"instance_type": None, "kind": "none", "why": f"The cores sit idle, but no smaller {fam} size has the memory this model "
                        f"needs (it used {mem_pct or 0:.0f}% of {spec.get('memory_gib', 0):.0f} GiB here). Keep {instance_type}, or send more requests at once.",
                        "usd_per_hour": None, "usd_delta_per_hour": None}
            return None
        # GPU: the chip is the same across the family's single-GPU sizes; only the host part shrinks.
        fits = [(t, c) for t, c in same_family if (c.get("gpus") or 0) == (spec.get("gpus") or 0) and c.get("gpu_model") == spec.get("gpu_model")
                and _size_rank(c) < _size_rank(spec) and (c.get("memory_gib") or 0) >= mem_need
                and (host_cpu_pct is None or (c.get("vcpus") or 0) >= (spec.get("vcpus") or 0) * host_cpu_pct / 100 * 1.5)]
        if fits:
            t, c = fits[0]
            return pack(t, f"The GPU itself idled, so a bigger GPU is not the fix; {t} has the same {spec.get('gpu_model')} with fewer host "
                           f"cores ({c['vcpus']} vCPUs, {c['memory_gib']:.0f} GiB) for ${c['usd_per_hour']:.2f}/h instead of ${price:.2f}/h. "
                           f"The larger lever is to send more requests at once: this GPU has room for them.", "smaller")
        return {"instance_type": None, "kind": "none", "why": f"The GPU idled, but every smaller {fam} size has the same GPU and too little "
                f"host memory or CPU for what ran here. Keep {instance_type} and send more requests at once.", "usd_per_hour": None, "usd_delta_per_hour": None}
    if verdict == "under":
        if side == "cpu":
            up = [(t, c) for t, c in same_family if _size_rank(c) > _size_rank(spec)]
            doubled = [(t, c) for t, c in up if (c.get("vcpus") or 0) >= 2 * (spec.get("vcpus") or 0)]
            if up:
                t, c = (doubled or up)[0]
                return pack(t, f"{t} has {c['vcpus']} vCPUs and {c['memory_gib']:.0f} GiB (this one: {spec.get('vcpus')} and "
                               f"{spec.get('memory_gib', 0):.0f}) for ${c['usd_per_hour']:.2f}/h instead of ${price:.2f}/h; the saturated CPU is "
                               f"what held this load back.", "bigger")
            return None
        if stronger:
            t, c = stronger[0]
            return pack(t, f"A bigger {fam} size would not help: every single-GPU {fam} size has the same {spec.get('gpu_model')}. The next "
                           f"stronger GPU is the {c.get('gpu_model')} in {t} ({c.get('gpu_memory_gib', 0):.0f} GiB GPU memory) for "
                           f"${c['usd_per_hour']:.2f}/h instead of ${price:.2f}/h.", "stronger")
        return None
    return None


def sizing_for_rows(rows: list[dict], series_name, spec_for, catalog: dict | None = None) -> list[dict]:
    """One verdict per series (model x hardware x software x format), GPU first."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(series_name(r), []).append(r)
    out = []
    for name, rs in by.items():
        v = sizing(rs, spec_for(rs[0].get("instance_type")), catalog)
        if v:
            out.append({"series": name, **v})
    out.sort(key=lambda v: (v["side"] != "gpu", v["series"]))
    return out
