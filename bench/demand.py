"""The question the CPU can win: "what is the cheapest option that meets my targets at MY load?"

The capacity verdict (bench/report.py, webui/api/compare.py) asks how many CPU instances match one
GPU's peak throughput. On AWS On-Demand pricing a price-matched Xeon slice loses that on every
workload measured here (one L4 carries 7.5x the matrix arithmetic and 2.7x the memory bandwidth of
eight Xeon 6 cores at the same hourly price; docs/recipe-check-2026-11-19.md). Most deployments do
not need a GPU's peak: at a given load, the cheapest instance that still meets the latency targets
may be a small CPU slice, and the smallest GPU instance costs $0.80/h whatever the load.

This module answers that question from the measurements:

* `price_per_hour` applies a pricing basis to an instance type: On-Demand, a Compute or EC2 Instance
  Savings Plan (1 or 3 years, No Upfront, real rates from the Savings Plans API kept in
  config/instance_catalog.json by tools/refresh_instance_catalog.py), or an operator override per
  side (a reserved rate, or $0 for capacity already paid for).
* `capacity` reads one series' measurements: its capacity is the highest request rate at which it
  still met the targets.
* `plan` sizes every option for a required load (requests per second), prices it, names the
  cheapest per side and overall, and finds the crossover load above which the GPU side is cheaper.
  Smaller sizes of a measured CPU family are offered as ESTIMATES (capacity scaled by core count,
  memory checked against the model), marked as such; a measurement confirms them.
"""
from __future__ import annotations

import math
from typing import Any

HOURS_PER_MONTH = 730.0
THREADS_PER_CORE = 2
MEM_FLOOR_GIB = 4.0            # OS, engine, KV/activations headroom on top of the weights
MEM_PER_WEIGHT_GB = 2.5        # measured engine footprint is ~2.15x the checkpoint (Gemma 4 probe) plus slack

PRICING_BASES: dict[str, tuple[str, str | None]] = {
    "on_demand": ("On-Demand", None),
    "compute_1y": ("Compute Savings Plan, 1 year, no upfront", "compute_1y"),
    "compute_3y": ("Compute Savings Plan, 3 years, no upfront", "compute_3y"),
    "ec2_instance_1y": ("EC2 Instance Savings Plan, 1 year, no upfront", "ec2_instance_1y"),
    "ec2_instance_3y": ("EC2 Instance Savings Plan, 3 years, no upfront", "ec2_instance_3y"),
}

TYPICAL_LOADS: list[dict[str, Any]] = [
    {"key": "pilot", "label": "Pilot: 10 requests/minute", "requests_per_minute": 10,
     "help": "A handful of people trying the feature. One small instance of either kind is idle most of the time."},
    {"key": "team", "label": "Team: 60 requests/minute", "requests_per_minute": 60,
     "help": "One request per second: an internal tool for a department."},
    {"key": "service", "label": "Service: 600 requests/minute", "requests_per_minute": 600,
     "help": "Ten per second: a customer-facing feature with steady traffic."},
    {"key": "product", "label": "Product: 6,000 requests/minute", "requests_per_minute": 6000,
     "help": "A hundred per second: the regime where peak throughput per dollar decides, and the GPU wins."},
]


def price_per_hour(instance_type: str, accelerator: str, pricing: dict | None, catalog: dict) -> dict[str, Any]:
    """{usd_per_hour, basis, label, on_demand, override}. An override for the side wins over the basis."""
    pricing = pricing or {}
    spec = (catalog.get("instances") or {}).get(instance_type) or {}
    on_demand = spec.get("usd_per_hour")
    basis = pricing.get("basis") or "on_demand"
    if basis not in PRICING_BASES:
        basis = "on_demand"
    override = pricing.get(f"{accelerator}_usd_per_hour")
    if override is not None:
        return {"usd_per_hour": float(override), "basis": "override", "label": f"your rate for the {accelerator.upper()} side",
                "on_demand": on_demand, "override": True}
    label, key = PRICING_BASES[basis]
    rate = on_demand
    if key:
        sp = (spec.get("savings_plan_usd_per_hour") or {}).get(key)
        if sp is not None:
            rate = sp
        else:
            label += " (rate not in the catalog; On-Demand used)"
    return {"usd_per_hour": rate, "basis": basis, "label": label, "on_demand": on_demand, "override": False}


def savings_options(instance_type: str, catalog: dict) -> list[dict[str, Any]]:
    """The committed-use rates next to On-Demand, for display: [{basis, label, usd_per_hour, saving_pct}]."""
    spec = (catalog.get("instances") or {}).get(instance_type) or {}
    od = spec.get("usd_per_hour")
    out = []
    for basis, (label, key) in PRICING_BASES.items():
        rate = od if key is None else (spec.get("savings_plan_usd_per_hour") or {}).get(key)
        if rate is None or od is None:
            continue
        out.append({"basis": basis, "label": label, "usd_per_hour": rate, "saving_pct": round(100 * (1 - rate / od)) if od else None})
    return out


def _req_s(row: dict) -> float | None:
    v = row.get("req_s")
    return float(v) if v else None


def capacity(rows: list[dict]) -> dict[str, Any] | None:
    """One series' capacity: the highest request rate at which a measurement met the targets, and that
    measurement's load setting and busy-ness. None when nothing met them."""
    ok = [r for r in rows if r.get("meets_user_sla") and _req_s(r)]
    if not ok:
        return None
    top = max(ok, key=_req_s)
    return {"req_s": _req_s(top), "conc": top.get("conc"), "batch": top.get("batch") or 1,
            "busy_mean": top.get("gpu_util_mean") if top.get("accelerator") == "gpu" else top.get("cpu_util_mean"),
            "latency_p99_ms": top.get("e2e_p99"), "run_id": top.get("run_id")}


def _series_key(r: dict) -> tuple:
    return (r["accelerator"], r["instance_type"], r.get("engine"), r.get("precision"))


def _size_rank(spec: dict) -> tuple:
    return ((spec.get("vcpus") or 0), (spec.get("memory_gib") or 0))


def _estimated_smaller(option: dict, rows: list[dict], params: int | None, catalog: dict, pricing: dict | None) -> list[dict]:
    """Smaller same-family CPU sizes, capacity scaled by physical cores, memory checked against the
    model. Estimates, not measurements: the UI and the summary say so and offer to run them."""
    inst = catalog.get("instances") or {}
    spec = inst.get(option["instance_type"]) or {}
    fam, cores = spec.get("family"), (spec.get("vcpus") or 0) / THREADS_PER_CORE
    if not fam or not cores:
        return []
    weights_gb = (params or 0) * 2 / 1e9                       # BF16; INT8 halves it but we stay conservative
    mem_need = MEM_FLOOR_GIB + MEM_PER_WEIGHT_GB * weights_gb
    out = []
    for t, c in inst.items():
        if c.get("family") != fam or c.get("accelerator") != "cpu" or _size_rank(c) >= _size_rank(spec):
            continue
        if (c.get("memory_gib") or 0) < mem_need or not c.get("usd_per_hour"):
            continue
        scale = ((c.get("vcpus") or 0) / THREADS_PER_CORE) / cores
        pr = price_per_hour(t, "cpu", pricing, catalog)
        out.append({**option, "instance_type": t, "estimated": True, "capacity_req_s": option["capacity_req_s"] * scale,
                    "capacity_basis": f"estimated: {option['instance_type']}'s measured capacity scaled by core count "
                                      f"({int(cores * scale)} of {int(cores)} cores); memory {c.get('memory_gib')} GiB holds the model",
                    "usd_per_hour_each": pr["usd_per_hour"], "price_label": pr["label"], "vcpus": c.get("vcpus"),
                    "series": option["series"].replace(option["instance_type"], t) + " (estimated)"})
    return out


def _size(option: dict, demand_req_s: float) -> dict:
    n = max(1, math.ceil(demand_req_s / option["capacity_req_s"])) if demand_req_s > 0 else 1
    hourly = n * option["usd_per_hour_each"]
    return {**option, "instances": n, "usd_per_hour": hourly, "usd_per_month": hourly * HOURS_PER_MONTH,
            "headroom": option["capacity_req_s"] * n / demand_req_s - 1 if demand_req_s > 0 else None}


def plan(rows: list[dict], demand_req_s: float | None, pricing: dict | None, catalog: dict, series_name,
         params: int | None = None, *, include_estimates: bool = True) -> dict[str, Any] | None:
    """Options (one per series that met the targets, plus estimated smaller CPU sizes), each sized for
    the demand, the cheapest per side and overall, and the crossover load. rows: one group's rows with
    meets_user_sla set. Returns None when no demand is given."""
    if demand_req_s is None or demand_req_s <= 0:
        return None
    by_series: dict[tuple, list[dict]] = {}
    for r in rows:
        by_series.setdefault(_series_key(r), []).append(r)
    options: list[dict] = []
    for key, srows in by_series.items():
        cap = capacity(srows)
        if not cap:
            continue
        acc, itype, engine, precision = key
        pr = price_per_hour(itype, acc, pricing, catalog)
        if pr["usd_per_hour"] is None:
            continue
        opt = {"series": series_name(srows[0]), "side": acc, "instance_type": itype, "engine": engine, "precision": precision,
               "capacity_req_s": cap["req_s"], "capacity_at": {"conc": cap["conc"], "batch": cap["batch"], "busy_mean": cap["busy_mean"],
                                                               "latency_p99_ms": cap["latency_p99_ms"]},
               "capacity_basis": f"measured: {cap['req_s'] * 60:.0f} requests/minute at {cap['conc']} at a time"
                                 + (f" x {cap['batch']} per request" if (cap["batch"] or 1) > 1 else "") + " still met the targets",
               "usd_per_hour_each": pr["usd_per_hour"], "price_label": pr["label"], "estimated": False, "run_id": cap["run_id"]}
        options.append(opt)
        if include_estimates and acc == "cpu":
            options += _estimated_smaller(opt, srows, params, catalog, pricing)
    if not options:
        return {"demand_req_s": demand_req_s, "options": [], "cheapest": None, "cheapest_gpu": None, "cheapest_cpu": None,
                "crossover": None, "text": "No option met your targets at any measured load, so nothing can be sized for this demand."}
    sized = sorted((_size(o, demand_req_s) for o in options), key=lambda o: (o["usd_per_hour"], o["estimated"]))
    measured = [o for o in sized if not o["estimated"]]
    cheapest = sized[0]
    cheapest_measured = measured[0] if measured else None
    cheapest_gpu = next((o for o in sized if o["side"] == "gpu"), None)
    cheapest_cpu = next((o for o in sized if o["side"] == "cpu"), None)
    cheapest_cpu_measured = next((o for o in measured if o["side"] == "cpu"), None)
    crossover = _crossover(options, demand_req_s)
    return {"demand_req_s": demand_req_s, "demand_rpm": demand_req_s * 60, "options": sized, "cheapest": cheapest,
            "cheapest_measured": cheapest_measured, "cheapest_gpu": cheapest_gpu, "cheapest_cpu": cheapest_cpu,
            "cheapest_cpu_measured": cheapest_cpu_measured, "crossover": crossover,
            "curve": _curve(options, demand_req_s), "text": _text(demand_req_s, cheapest, cheapest_gpu, cheapest_cpu, crossover)}


def _cost_at(options: list[dict], side: str, d: float) -> float | None:
    costs = [_size(o, d)["usd_per_hour"] for o in options if o["side"] == side]
    return min(costs) if costs else None


def _crossover(options: list[dict], demand_req_s: float) -> dict | None:
    """The smallest load at which the cheapest GPU option costs no more than the cheapest CPU option
    (measured or estimated). None when one side is absent; 'cpu_never' / 'gpu_always' flags otherwise."""
    gpu = [o for o in options if o["side"] == "gpu"]
    cpu = [o for o in options if o["side"] == "cpu"]
    if not gpu or not cpu:
        return None
    top = max(max(o["capacity_req_s"] for o in gpu) * 2, demand_req_s * 2, 0.01)
    steps = 400
    first = None
    for i in range(1, steps + 1):
        d = top * i / steps
        g, c = _cost_at(options, "gpu", d), _cost_at(options, "cpu", d)
        if g is not None and c is not None and g <= c:
            first = d
            break
    if first is None:
        return {"req_s": None, "rpm": None, "kind": "cpu_cheaper_everywhere",
                "text": f"Up to {top * 60:.0f} requests/minute the CPU side stays cheaper."}
    if first <= top / steps:
        return {"req_s": first, "rpm": first * 60, "kind": "gpu_cheaper_everywhere",
                "text": "The GPU side is the cheaper one at every load, including the lightest."}
    return {"req_s": first, "rpm": first * 60, "kind": "crossover",
            "text": f"Below about {first * 60:.0f} requests/minute the CPU side is cheaper; above it the GPU side is."}


def _curve(options: list[dict], demand_req_s: float, points: int = 40) -> list[dict]:
    """Cheapest hourly cost per side over a range of loads, for the chart."""
    gpu_cap = [o["capacity_req_s"] for o in options if o["side"] == "gpu"]
    top = max(max(gpu_cap) * 1.5 if gpu_cap else 0, demand_req_s * 1.5, 0.01)
    out = []
    for i in range(0, points + 1):
        d = top * i / points if i else top / points / 100       # a near-zero first point: one instance of each side
        out.append({"rpm": d * 60 if i else 0.0, "gpu_usd_per_hour": _cost_at(options, "gpu", d), "cpu_usd_per_hour": _cost_at(options, "cpu", d),
                    "cpu_measured_usd_per_hour": min([_size(o, d)["usd_per_hour"] for o in options if o["side"] == "cpu" and not o["estimated"]] or [float("nan")])})
    return out


def _text(d: float, cheapest: dict, gpu: dict | None, cpu: dict | None, crossover: dict | None) -> str:
    rpm = d * 60
    def one(o: dict) -> str:
        est = " (estimated size; run it to confirm)" if o["estimated"] else ""
        return (f"{o['instances']} x {o['instance_type']} ({o['engine']}, {o['precision']}) at ${o['usd_per_hour']:.2f}/h, "
                f"${o['usd_per_month']:,.0f}/month{est}")
    s = f"At {rpm:,.0f} requests/minute the cheapest option that meets your targets is {one(cheapest)}."
    if gpu and cpu:
        other = cpu if cheapest["side"] == "gpu" else gpu
        ratio = other["usd_per_hour"] / cheapest["usd_per_hour"] if cheapest["usd_per_hour"] else None
        s += f" The cheapest {other['side'].upper()} option is {one(other)}" + (f", {ratio:.1f}x as much." if ratio else ".")
    if crossover and crossover.get("text"):
        s += " " + crossover["text"]
    return s
