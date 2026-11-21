"""Where the CPU is the better choice, and where it is not: one plain statement per comparison, computed
from the measurements, never asserted.

Three kinds of benefit are checked:

1. **Cost at a given load.** From bench/demand.plan: below the crossover load the cheapest CPU option
   that meets the targets costs less than the cheapest GPU option. The statement names the load range,
   the options and the monthly costs at the typical loads that fall inside it, and says when the
   cheapest CPU size is an estimate.
2. **Availability.** The runs' own capacity log (report `capacity_errors`): how often a GPU launch was
   refused for lack of capacity and fell back to another type, against the CPU launches. A benefit only
   when the data shows it.
3. **Granularity.** The smallest CPU slice that holds the model against the smallest GPU instance, when
   the CPU slice is cheaper: a workload too small for the smallest GPU still gets its own instance.

Everything the CPU does NOT win is listed next to it: peak throughput per dollar (always the GPU here),
groups where no CPU option met the targets, and the load above the crossover.
"""
from __future__ import annotations

from typing import Any

from bench import demand


def _money(h: float) -> str:
    return f"${h:.2f}/h (${h * demand.HOURS_PER_MONTH:,.0f}/month)"


def _plan_at(rows: list[dict], rpm: float, pricing: dict | None, catalog: dict, series_name, params) -> dict | None:
    return demand.plan(rows, rpm / 60, pricing, catalog, series_name, params=params)


def cost_wins(group: dict, grows: list[dict], pricing: dict | None, catalog: dict, series_name, params: int | None) -> dict[str, Any]:
    """The load range where the CPU side is cheaper for one group, with worked examples at the typical loads."""
    probe = _plan_at(grows, 60, pricing, catalog, series_name, params)
    out: dict[str, Any] = {"group": group["label"], "key": group["key"], "kind": "none", "examples": [], "text": ""}
    if not probe or not probe.get("options"):
        out["text"] = "No option on either side met your targets, so nothing can be compared here."
        return out
    if not probe.get("cheapest_cpu"):
        out["text"] = "No CPU option met your targets at any measured load: the GPU is the only option for this question."
        out["kind"] = "gpu_only"
        return out
    if not probe.get("cheapest_gpu"):
        out["text"] = "No GPU option met your targets at any measured load; the CPU options stand alone."
        out["kind"] = "cpu_only"
        return out
    cx = probe.get("crossover") or {}
    kind = cx.get("kind")
    examples = []
    for t in demand.TYPICAL_LOADS:
        rpm = t["requests_per_minute"]
        if kind == "crossover" and cx.get("rpm") and rpm >= cx["rpm"]:
            continue
        if kind == "gpu_cheaper_everywhere":
            continue
        p = _plan_at(grows, rpm, pricing, catalog, series_name, params)
        if not p or not p.get("cheapest_cpu") or not p.get("cheapest_gpu"):
            continue
        c, g = p["cheapest_cpu"], p["cheapest_gpu"]
        if c["usd_per_hour"] >= g["usd_per_hour"]:
            continue
        examples.append({"label": t["label"], "rpm": rpm, "cpu": f"{c['instances']} x {c['instance_type']}", "cpu_usd_per_hour": c["usd_per_hour"],
                         "cpu_usd_per_month": c["usd_per_month"], "gpu": f"{g['instances']} x {g['instance_type']}", "gpu_usd_per_hour": g["usd_per_hour"],
                         "gpu_usd_per_month": g["usd_per_month"], "ratio": g["usd_per_hour"] / c["usd_per_hour"] if c["usd_per_hour"] else None,
                         "estimated": c["estimated"]})
    out["examples"] = examples
    out["crossover_rpm"] = cx.get("rpm")
    if kind == "gpu_cheaper_everywhere" or not examples:
        out["kind"] = "gpu_always"
        out["text"] = ("The GPU side is cheaper at every load, including the lightest: the CPU options that meet your targets "
                       "cost more per hour than the smallest GPU instance even at one instance.")
        return out
    out["kind"] = "cpu_below_crossover" if kind == "crossover" else "cpu_everywhere"
    ex = examples[0]
    est = " (an estimated size: one run confirms it)" if ex["estimated"] else ""
    rng = (f"below about {cx['rpm']:,.0f} requests per minute" if kind == "crossover" and cx.get("rpm")
           else "at every load measured and sized here")
    out["text"] = (f"The CPU side is cheaper {rng}. At {ex['rpm']:,} requests/minute: {ex['cpu']} at {_money(ex['cpu_usd_per_hour'])}{est} "
                   f"against {ex['gpu']} at {_money(ex['gpu_usd_per_hour'])}, {ex['ratio']:.1f}x less." if ex["ratio"] else "")
    if kind == "crossover" and cx.get("rpm"):
        out["text"] += f" Above about {cx['rpm']:,.0f} requests/minute the GPU's throughput per dollar wins."
    return out


def availability(summaries: list[dict]) -> dict[str, Any]:
    """Capacity refusals per side across the selected runs, from the reports' capacity logs (split by market:
    a Spot refusal means the discounted pool was empty, an On-Demand refusal that the type itself was not
    available), plus the instance-type fallbacks that followed."""
    counts = {"gpu": {"spot": 0, "on-demand": 0}, "cpu": {"spot": 0, "on-demand": 0}}
    fallbacks: list[dict] = []
    for s in summaries:
        run = s.get("run") or {}
        acc = {r["target"]: r["accelerator"] for r in s.get("rows") or []}
        requested = run.get("instance_types") or {}
        actual = {r["target"]: r["instance_type"] for r in s.get("rows") or []}
        for tid, events in (s.get("capacity_errors") or {}).items():
            side = acc.get(tid) or ("gpu" if tid.startswith("t1") else "cpu")
            for e in events or []:
                market = "spot" if (e.get("market") or "").lower().startswith("spot") else "on-demand"
                counts[side][market] += 1
        for tid, a in actual.items():
            if requested.get(tid) and a != requested[tid]:
                fallbacks.append({"run_id": s["run_id"], "side": acc.get(tid), "requested": requested[tid], "actual": a})
    g, c = counts["gpu"], counts["cpu"]
    gpu_ref, cpu_ref = g["spot"] + g["on-demand"], c["spot"] + c["on-demand"]
    gpu_fb = [f for f in fallbacks if f["side"] == "gpu"]
    cpu_fb = [f for f in fallbacks if f["side"] == "cpu"]
    fb_text = (", ".join(sorted({f'{f["requested"]} -> {f["actual"]}' for f in gpu_fb})))
    if gpu_ref == 0 and cpu_ref == 0:
        kind, text = "none", ("Every instance in these runs launched at the requested type on the first try; the runs give no evidence "
                             "either way on availability.")
    elif g["on-demand"] > 0 and c["on-demand"] == 0:
        kind = "gpu_scarcer"
        text = (f"In these runs the GPU type was refused for lack of capacity {gpu_ref} time{'s' if gpu_ref != 1 else ''}: "
                f"{g['spot']} on Spot and {g['on-demand']} On-Demand, meaning the GPU type itself was not available in the zone, not only "
                f"the discounted pool"
                + (f"; the GPU launch only succeeded on a different type ({fb_text})" if gpu_fb else "")
                + f". The CPU type was refused {cpu_ref} time{'s' if cpu_ref != 1 else ''}, all on Spot, and launched On-Demand at the requested "
                "type. Availability is a real benefit here: when the GPU you sized for cannot be launched, the CPU option can, "
                "and at the loads where it is also cheaper nothing is given up.")
    elif gpu_ref > cpu_ref:
        kind = "gpu_scarcer_spot"
        text = (f"In these runs the GPU type was refused {gpu_ref} time{'s' if gpu_ref != 1 else ''} and the CPU type {cpu_ref}, "
                f"all on Spot (the discounted pool): a mild availability edge for the CPU, not a decisive one, since both launched On-Demand.")
    else:
        kind = "no_advantage"
        text = (f"In these runs the CPU type was refused {cpu_ref} time{'s' if cpu_ref != 1 else ''} and the GPU type {gpu_ref}: "
                "no availability advantage for the CPU is visible here.")
    return {"kind": kind, "gpu_refusals": gpu_ref, "cpu_refusals": cpu_ref, "by_market": counts, "gpu_fallbacks": gpu_fb,
            "cpu_fallbacks": cpu_fb, "text": text}


def granularity(plans: list[dict | None]) -> dict[str, Any] | None:
    """The smallest CPU option that met the targets (measured or estimated) against the smallest GPU instance, over the groups' plans."""
    smallest_cpu, smallest_gpu = None, None
    for p in plans:
        for o in (p or {}).get("options") or []:
            if o["side"] == "cpu" and (smallest_cpu is None or o["usd_per_hour_each"] < smallest_cpu["usd_per_hour_each"]):
                smallest_cpu = o
            if o["side"] == "gpu" and (smallest_gpu is None or o["usd_per_hour_each"] < smallest_gpu["usd_per_hour_each"]):
                smallest_gpu = o
    if not smallest_cpu or not smallest_gpu or smallest_cpu["usd_per_hour_each"] >= smallest_gpu["usd_per_hour_each"]:
        return None
    ratio = smallest_gpu["usd_per_hour_each"] / smallest_cpu["usd_per_hour_each"]
    return {"cpu": smallest_cpu["instance_type"], "cpu_usd_per_hour": smallest_cpu["usd_per_hour_each"], "gpu": smallest_gpu["instance_type"],
            "gpu_usd_per_hour": smallest_gpu["usd_per_hour_each"], "ratio": ratio, "estimated": smallest_cpu["estimated"],
            "text": (f"The smallest unit of GPU you can buy is one {smallest_gpu['instance_type']} at ${smallest_gpu['usd_per_hour_each']:.2f}/h, idle or not. "
                     f"The smallest CPU slice that holds this model and met the targets is {smallest_cpu['instance_type']} at "
                     f"${smallest_cpu['usd_per_hour_each']:.2f}/h, {ratio:.0f}x less{' (estimated size)' if smallest_cpu['estimated'] else ''}: "
                     "a workload too small for a GPU still gets its own instance, and many small workloads can each have one.")}


def build(groups: list[dict], grows_by_key: dict[str, list[dict]], summaries: list[dict], pricing: dict | None, catalog: dict,
          series_name, params_by_model: dict[str, int]) -> dict[str, Any]:
    """The whole statement: per-group cost findings, availability, granularity, and the explicit list of what the CPU does not win."""
    per_group, probes = [], []
    for g in groups:
        grows = grows_by_key.get(g["key"])
        if not grows:
            continue
        params = params_by_model.get(grows[0]["model"])
        per_group.append(cost_wins(g, grows, pricing, catalog, series_name, params))
        probes.append(_plan_at(grows, 60, pricing, catalog, series_name, params))
    wins = [x for x in per_group if x["kind"] in ("cpu_below_crossover", "cpu_everywhere")]
    losses = [x for x in per_group if x["kind"] in ("gpu_always", "gpu_only")]
    avail = availability(summaries)
    gran = granularity(probes)
    peak = []
    for g in groups:
        v = g.get("verdict") or {}
        if v.get("instances_per_gpu"):
            peak.append(f"{g['label']}: {v['instances_per_gpu']} CPU instances per GPU, {v['hourly_cost_ratio']:.1f}x the hourly cost")
    any_benefit = bool(wins) or avail["kind"] in ("gpu_scarcer", "gpu_scarcer_spot") or gran is not None
    if wins:
        headline = (f"The CPU is the better choice for {len(wins)} of {len(per_group)} question{'s' if len(per_group) != 1 else ''} here, "
                    "at low to moderate load; the GPU wins on peak throughput everywhere.")
    elif avail["kind"] == "gpu_scarcer":
        headline = ("On cost the GPU wins at every load measured here. The CPU's one benefit in these runs is availability: "
                    "it launched when the GPU type did not.")
    elif gran:
        headline = "On cost the GPU wins at every load measured here; the CPU's one benefit is a smaller minimum unit."
    else:
        headline = "In these runs the CPU has no benefit over the GPU: not on cost at any load, not on availability, not on size."
    return {"any_benefit": any_benefit, "headline": headline, "cost": per_group, "wins": wins, "losses": losses,
            "availability": avail, "granularity": gran,
            "not_won": {"peak_throughput": peak, "text": ("Peak throughput per dollar: the GPU wins every question here, so a workload that needs a "
                                                            "GPU's full capacity should run on the GPU." if peak else "")},
            "structural": [
                {"title": "Capacity you already pay for", "text": "If the CPU fleet exists and has idle capacity, its marginal cost is close to zero; "
                                                                   "enter your own CPU rate (0 for idle capacity) under 'Your load and your prices' to see that comparison."},
                {"title": "Same fleet, same tooling", "text": "A CPU instance needs no GPU driver or CUDA stack and can run inference next to the application on "
                                                             "the general-purpose fleet; this benchmark does not measure that, it is an operational fact."},
            ]}
