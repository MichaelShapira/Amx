"""Why a result came out the way it did, computed from the run (for the executive summary's
"Technical details" and the results page).

A request has two phases. Reading the input (prefill): every input token is processed at once, the
weights are fetched from memory once for the whole batch, and arithmetic throughput decides the
time; this is what Intel's AMX units add to a Xeon core. Writing the answer (decode): each output
token needs the whole set of weights streamed from memory again, so memory bandwidth decides the
time at low concurrency and arithmetic takes over only when many requests share each step. Memory
bandwidth is where a price-matched CPU slice is furthest behind a GPU (STREAM on an m8i.4xlarge:
107 GB/s; an L4: 300 GB/s), arithmetic less so at small batches. The share of a request spent
writing therefore predicts how far behind the CPU will be, and that is what this module quantifies:
the token shape, the writing share, the bytes a token moves, the bandwidth each side reached against
what it has, and the arithmetic rate at the heaviest load against each side's peak.
"""

from __future__ import annotations

from typing import Any

# Memory bandwidth (GB/s) and dense BF16 peak (TFLOPS) of the GPUs the instance catalog offers. Vendor
# figures; the harness measures the CPU side's bandwidth with STREAM on every host (hosts/membw-*).
GPU = {"T4": (320, 65), "L4": (300, 121), "A10G": (600, 70), "L40S": (864, 362), "A100": (1555, 312),
       "A100-80GB": (2039, 312), "H100": (3350, 989), "H200": (4800, 989), "V100": (900, 112)}
# A Xeon 6 core with AMX: 1,024 BF16 FLOPs per cycle at about 2 GHz under AMX load -> ~2 TFLOPS, theoretical.
AMX_TFLOPS_PER_CORE = 2.0
CPU_GB_S_PER_CORE_FALLBACK = 13.0          # STREAM triad per physical core measured on m8i.4xlarge (107 GB/s / 8)
BYTES_PER_WEIGHT = {"bf16": 2, "fp16": 2, "fp32": 4, "w8a8": 1, "int8-ov": 1, "int4-ov": 0.5, "awq-int4": 0.5, "bf16-ov": 2}


def _weights_gb(params: int | None, precision: str) -> float | None:
    if not params:
        return None
    return params * BYTES_PER_WEIGHT.get(precision, 2) / 1e9


def _side_hw(row: dict, hosts: dict, instances: dict) -> dict:
    """Bandwidth and arithmetic peak of the row's instance."""
    spec = instances.get(row.get("instance_type") or "") or {}
    if row.get("accelerator") == "gpu":
        bw, tf = GPU.get(spec.get("gpu_model") or "", (None, None))
        gpus = spec.get("gpus") or 1
        return {"bandwidth_gb_s": bw * gpus if bw else None, "peak_tflops": tf * gpus if tf else None,
                "bandwidth_source": f"vendor figure for {spec.get('gpu_model')}" if bw else None, "cores": None}
    cores = (spec.get("vcpus") or 0) // 2 or None
    mb = (hosts.get(f"membw-{row.get('instance_type')}") or {})
    bw = mb.get("triad_mb_s") / 1000 if mb.get("triad_mb_s") else (cores * CPU_GB_S_PER_CORE_FALLBACK if cores else None)
    return {"bandwidth_gb_s": bw, "peak_tflops": cores * AMX_TFLOPS_PER_CORE if cores else None,
            "bandwidth_source": "STREAM triad measured on this host" if mb.get("triad_mb_s") else "estimate (13 GB/s per core)",
            "cores": cores}


def _shape(rows: list[dict]) -> dict:
    """Tokens in and out per request and the share of a request spent writing, at the lowest concurrency."""
    r = min(rows, key=lambda x: (x["conc"], x.get("batch") or 1))
    gen = r["kind"] == "generation"
    req_s = r.get("req_s") or 0
    if gen:
        tin = (r.get("agg_in_tok_s") or 0) / req_s if req_s else None
        tout = (r.get("agg_out_tok_s") or 0) / req_s if req_s else None
        ttft, tpot, e2e = r.get("ttft_p50"), r.get("tpot_p50"), r.get("e2e_p50")
        if ttft and tpot and e2e and e2e > ttft:
            # Per-request medians, not the aggregate ratio: the steady-state window counts the tokens of requests
            # that straddle it but only the requests that finish inside it, so agg_tok_s / req_s overstates the
            # per-request token count at high concurrency (186 for a 128/128 run with 50 requests).
            if r.get("prefill_tok_s_p50"):
                tin = r["prefill_tok_s_p50"] * ttft / 1000
            tout = 1 + (e2e - ttft) / tpot
            share = 1 - ttft / e2e
        else:
            write_ms = (tpot or 0) * max((tout or 1) - 1, 0)
            share = min(1.0, write_ms / e2e) if e2e else None
    else:
        items_s = r.get("cls_s") or 0
        tin = (r.get("agg_in_tok_s") or 0) / items_s if items_s else None
        tout, share = 0.0, 0.0
    return {"input_tokens": tin, "output_tokens": tout, "writing_share": share, "generation": gen}


def explain_group(rows: list[dict], params: dict[str, int], hosts: dict, instances: dict) -> dict:
    """rows: the group's rows (one model, one question); params: model id -> parameter count."""
    shape = _shape(rows)
    sides = {}
    for side in ("gpu", "cpu"):
        srows = [r for r in rows if r["accelerator"] == side]
        if not srows:
            continue
        low = min(srows, key=lambda x: (x["conc"], x.get("batch") or 1))
        top = max(srows, key=lambda x: (x.get(_thr_key(x)) or 0))
        hw = _side_hw(low, hosts, instances)
        w = _weights_gb(params.get(low["model"]), low["precision"])
        out: dict[str, Any] = {"instance_type": low["instance_type"], "engine": low["engine"], "precision": low["precision"],
                               "weights_gb": w, **hw}
        if shape["generation"] and low.get("tpot_p50") and w:
            implied = w / (low["tpot_p50"] / 1000)                  # the weights stream once per generated token
            out["low_conc"] = low["conc"]
            out["implied_bandwidth_gb_s"] = implied
            out["bandwidth_utilization"] = implied / hw["bandwidth_gb_s"] if hw["bandwidth_gb_s"] else None
        if top.get("tpot_p50") and params.get(top["model"]) and shape["generation"]:
            tokens_per_step = top["conc"]
            flops = 2 * params[top["model"]] * tokens_per_step
            out["top_conc"] = top["conc"]
            out["achieved_tflops"] = flops / (top["tpot_p50"] / 1000) / 1e12
            out["compute_utilization"] = out["achieved_tflops"] / hw["peak_tflops"] if hw["peak_tflops"] else None
        if not shape["generation"] and top.get("agg_in_tok_s") and params.get(top["model"]):
            out["achieved_tflops"] = 2 * params[top["model"]] * top["agg_in_tok_s"] / 1e12
            out["compute_utilization"] = out["achieved_tflops"] / hw["peak_tflops"] if hw["peak_tflops"] else None
            out["top_conc"] = top["conc"]
        out["best_throughput"] = top.get(_thr_key(top))
        sides[side] = out
    return {"shape": shape, "sides": sides, "why": _why(shape, sides)}


def _thr_key(r: dict) -> str:
    if r.get("direction"):
        return "chars_s"
    if r.get("workload"):
        return "docs_s"
    return "agg_out_tok_s" if r["kind"] == "generation" else "cls_s"


def _pct(x: float | None) -> str:
    return f"{100 * x:.0f}%" if x is not None else "n/a"


def _why(shape: dict, sides: dict) -> str:
    g, c = sides.get("gpu"), sides.get("cpu")
    if not shape["generation"]:
        s = ("All reading, no writing: every input is processed at once and no token is generated, so the arithmetic "
             "units decide the speed, which is what AMX adds to a Xeon core.")
        if g and c and g.get("achieved_tflops") and c.get("achieved_tflops"):
            s += (f" At the heaviest load the GPU reached about {g['achieved_tflops']:.1f} TFLOPS ({_pct(g.get('compute_utilization'))} of its "
                  f"peak) and the CPU about {c['achieved_tflops']:.1f} TFLOPS ({_pct(c.get('compute_utilization'))} of its AMX peak).")
        return s
    share = shape.get("writing_share")
    tin, tout = shape.get("input_tokens"), shape.get("output_tokens")
    s = ""
    if tin is not None and tout is not None:
        s += f"Each request reads about {tin:.0f} tokens and writes about {tout:.0f}"
        if share is not None:
            s += f"; writing takes {_pct(share)} of the request time at the lightest load"
        s += ". "
    if g and c and g.get("implied_bandwidth_gb_s") and c.get("implied_bandwidth_gb_s"):
        s += (f"Every written token streams the model's weights ({c['weights_gb']:.1f} GB) through memory: the CPU reached "
              f"{c['implied_bandwidth_gb_s']:.0f} GB/s of its {c['bandwidth_gb_s']:.0f} GB/s ({_pct(c.get('bandwidth_utilization'))}, "
              f"{c['bandwidth_source']}), the GPU {g['implied_bandwidth_gb_s']:.0f} GB/s of its {g['bandwidth_gb_s']:.0f} GB/s "
              f"({_pct(g.get('bandwidth_utilization'))}). ")
        if c["bandwidth_gb_s"] and g["bandwidth_gb_s"]:
            s += (f"The GPU's memory is {g['bandwidth_gb_s'] / c['bandwidth_gb_s']:.1f}x faster, which is the size of the gap a "
                  f"writing-heavy task shows at low load; ")
            if share is not None and share < 0.4:
                s += "this task is mostly reading, so the gap should stay well below that ratio."
            else:
                s += "this task is mostly writing, so no setting on the CPU side closes it."
    elif c and c.get("implied_bandwidth_gb_s"):
        s += (f"The CPU streamed the weights at {c['implied_bandwidth_gb_s']:.0f} GB/s of its {c['bandwidth_gb_s']:.0f} GB/s "
              f"({_pct(c.get('bandwidth_utilization'))}): it runs at the slice's memory limit.")
    if g and c and g.get("achieved_tflops") and c.get("achieved_tflops"):
        s += (f" At the heaviest load the CPU did about {c['achieved_tflops']:.1f} TFLOPS ({_pct(c.get('compute_utilization'))} of its AMX "
              f"peak) against the GPU's {g['achieved_tflops']:.1f} TFLOPS ({_pct(g.get('compute_utilization'))} of its peak).")
    return s.strip()


def context(summaries: list[dict]) -> tuple[dict[str, int], dict, dict]:
    """(model id -> parameters, host facts incl. STREAM probes, instance specs) for explain_group."""
    from bench import catalog_store, pairing

    try:
        params = {m.id: m.params for m in catalog_store.load_catalog().models}
    except Exception:  # noqa: BLE001 - no catalog: no per-token bytes
        params = {}
    hosts: dict = {}
    for s in summaries:
        hosts.update(s.get("hosts") or {})
    return params, hosts, pairing.load_catalog()["instances"]
