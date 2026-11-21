"""Results analysis for the web UI: a run (or several) summarized into verdict cards, series
for charts, and a comparability check that is re-evaluated on every selection.

Two measurements are comparable when they answer the same question: same use case, same
data (prompt set and answer length, or scenarios plus dataset and policy fingerprints), the
same request shape (single request / N concurrent / batched), and the GPU and CPU instances
in a fair price band. Anything else is reported as a blocker or a warning, in words."""

from __future__ import annotations

import math
import re

from bench import cpu_case, demand, mechanics, pairing, report, utilization
from webui.api.catalog import (
    DIRECTIONS,
    INTEL_IO,
    MODELS,
    PRESETS,
    PROMPT_SETS,
    TEXT_SIZES,
    WORKLOADS,
)

# ----------------------------------------------------------------------------- run shape

def request_mode(run: dict) -> str:
    conc = run.get("concurrency") or []
    batches = run.get("batch_sizes")
    if batches and any(b > 1 for b in batches):
        return "batch"
    return "single" if conc == [1] else "concurrent"


def shape(summary: dict) -> dict:
    """The facts that decide whether two runs can be compared."""
    run = summary.get("run") or {}
    s = {"use_case": run.get("suite"), "request_mode": request_mode(run), "concurrency": sorted(run.get("concurrency") or []),
         "batch_sizes": run.get("batch_sizes"), "grade": "quick" if summary.get("smoke") else "full",
         "requests_per_measurement": (run.get("per_level") or {}).get("requests"), "repeats": run.get("repeats")}
    if run.get("suite") == "standard":
        s.update(prompt_set=run.get("prompt_set"), answer_tokens=(run.get("sampling") or {}).get("max_new_tokens"),
                 temperature=(run.get("sampling") or {}).get("temperature"))
    elif run.get("suite") == "translation":
        h = summary.get("dspm_hashes") or {}          # the pre-check stores the corpus fingerprint under the same key
        s.update(prompt_set=run.get("prompt_set"), answer_tokens=(run.get("sampling") or {}).get("max_new_tokens"),
                 directions=sorted((run.get("translation") or {}).get("directions") or []),
                 dataset=str(h.get("dataset_sha256"))[:12], test_items=(run.get("translation") or {}).get("test_items"))
    elif run.get("suite") == "embedding":
        s.update(prompt_set=run.get("prompt_set"), test_items=(run.get("embedding") or {}).get("test_items"))
    elif run.get("suite") == "intel":
        i = run.get("intel") or {}
        s.update(prompt_set="intel", io=f"{i.get('input_tokens', 128)}/{i.get('output_tokens', 128)}",
                 answer_tokens=i.get("output_tokens", 128))
    else:
        h = summary.get("dspm_hashes") or {}
        s.update(workloads=sorted((run.get("dspm") or {}).get("workloads") or []),
                 dataset=str(h.get("dataset_sha256"))[:12], policy=str(h.get("policy_sha256"))[:12],
                 test_items=(run.get("dspm") or {}).get("test_items"))
    return s


MODE_LABEL = {"single": "single request", "concurrent": "concurrent requests", "batch": "batched requests"}


def comparability(summaries: list[dict]) -> dict:
    """Blockers stop a comparison; warnings are shown next to it."""
    if len(summaries) < 2:
        return {"ok": True, "blockers": [], "warnings": [], "shapes": [shape(s) for s in summaries]}
    shapes = [shape(s) for s in summaries]
    names = [s["run_id"] for s in summaries]
    blockers, warnings = [], []

    def differ(key):
        vals = [str(sh.get(key)) for sh in shapes]
        return len(set(vals)) > 1, vals

    d, v = differ("use_case")
    if d:
        blockers.append(f"Different use cases ({', '.join(v)}): chat results and detection results measure different things.")
        return {"ok": False, "blockers": blockers, "warnings": warnings, "shapes": shapes}
    d, v = differ("request_mode")
    if d:
        blockers.append("Different request shapes (" + ", ".join(MODE_LABEL.get(x, x) for x in v) + "): a single request, "
                        "concurrent requests and batched requests load a machine differently, so their numbers are not comparable.")
    if shapes[0]["use_case"] == "standard":
        d, v = differ("prompt_set")
        if d:
            blockers.append("Different input lengths (" + ", ".join(PROMPT_SETS.get(x, (x,))[0] for x in v) + "): reading a long "
                            "document and answering a question stress the hardware differently.")
        d, v = differ("answer_tokens")
        if d:
            blockers.append(f"Different answer lengths ({', '.join(v)} tokens): throughput and cost per token depend on it.")
        d, v = differ("temperature")
        if d:
            warnings.append(f"Different sampling temperatures ({', '.join(v)}); timings are unaffected, answer quality is.")
    elif shapes[0]["use_case"] == "embedding":
        d, v = differ("prompt_set")
        if d:
            blockers.append("Different text sizes (" + ", ".join(TEXT_SIZES.get(x, (x,))[0] for x in v) + "): embedding a "
                            "sentence and a paragraph load the hardware differently.")
    elif shapes[0]["use_case"] == "intel":
        d, v = differ("io")
        if d:
            blockers.append("Different token shapes (" + ", ".join(INTEL_IO.get(x, (x,))[0] for x in v) + "): Intel's "
                            "figures are per shape.")
    elif shapes[0]["use_case"] == "translation":
        d, v = differ("prompt_set")
        if d:
            blockers.append("Different text sizes (" + ", ".join(TEXT_SIZES.get(x, (x,))[0] for x in v) + "): translating a "
                            "sentence and a paragraph load the hardware differently.")
        d, v = differ("dataset")
        if d:
            blockers.append(f"Different corpus versions (fingerprints {', '.join(v)}): the texts differ, so characters per "
                            "second and translation quality are not comparable.")
        sets = [set(sh.get("directions") or []) for sh in shapes]
        common = set.intersection(*sets) if sets else set()
        if not common:
            blockers.append("No translation direction in common between the runs.")
        elif any(s_ != common for s_ in sets):
            warnings.append("Only the directions all runs share are compared: " + ", ".join(DIRECTIONS[w][0] for w in sorted(common)) + ".")
    else:
        d, v = differ("dataset")
        if d:
            blockers.append(f"Different dataset versions (fingerprints {', '.join(v)}): the documents differ, so accuracy "
                            "and documents-per-second are not comparable.")
        d, v = differ("policy")
        if d:
            blockers.append(f"Different classification policies (fingerprints {', '.join(v)}): the models were asked "
                            "different questions.")
        sets = [set(sh.get("workloads") or []) for sh in shapes]
        common = set.intersection(*sets) if sets else set()
        if not common:
            blockers.append("No scenario in common between the runs.")
        elif any(s_ != common for s_ in sets):
            warnings.append("Only the scenarios all runs share are compared: " + ", ".join(WORKLOADS[w][0] for w in sorted(common)) + ".")
    d, v = differ("grade")
    if d:
        warnings.append("Mixing a quick check with a full test: the quick check's percentiles rest on far fewer requests.")
    d, v = differ("concurrency")
    if d and shapes[0]["request_mode"] == "concurrent":
        shared = set.intersection(*(set(sh["concurrency"]) for sh in shapes))
        if shared:
            warnings.append("Different numbers of concurrent requests; charts overlap only at " + ", ".join(map(str, sorted(shared))) + ".")
        else:
            blockers.append("No number of concurrent requests in common: the runs never loaded the machines the same way.")
    if shapes[0]["request_mode"] == "batch":
        d, v = differ("batch_sizes")
        if d:
            blockers.append(f"Different batch sizes ({', '.join(v)}).")
    return {"ok": not blockers, "blockers": blockers, "warnings": warnings, "shapes": shapes, "runs": names}


# ----------------------------------------------------------------------------- merged analysis

def _model_name(mid: str) -> str:
    return MODELS.get(mid, (mid, ""))[0]


def _group_key(r: dict) -> tuple[str, str]:
    key = r["model"]
    label = _model_name(r["model"])
    if r.get("direction"):
        key += "|" + r["direction"] + "|" + str(r.get("text_size"))
        label += " · " + DIRECTIONS.get(r["direction"], (r["direction"],))[0]
        label += f" ({TEXT_SIZES.get(r.get('text_size') or '', (r.get('text_size'),))[0]})"
        return key, label
    if r.get("model_type") == "embedding":
        size = r.get("text_size")
        if not size:                      # reports rendered before 2026-11-20 lost it when repeats were pooled
            m = re.search(r"__embed-([a-z]+)", r.get("level_group") or "")
            size = m.group(1) if m else None
        key += "|embed|" + str(size) + (f"|b{r['batch']}" if r.get("batch", 1) > 1 else "")
        label += " · embeddings" + (f", {TEXT_SIZES.get(size, (size,))[0]}" if size else "")
        if r.get("batch", 1) > 1:
            label += f" · batch of {r['batch']}"
        return key, label
    if r.get("workload"):
        key += "|" + r["workload"] + ("|" + r["bucket"] if r.get("bucket") else "")
        label += " · " + WORKLOADS.get(r["workload"], (r["workload"],))[0]
        if r.get("bucket"):
            label += f" ({r['bucket']} documents)"
    if r.get("kind") == "classifier" and not r.get("workload") and r.get("batch", 1) > 1:
        key += f"|b{r['batch']}"
        label += f" · batch of {r['batch']}"
    return key, label


def _unit(r: dict) -> dict:
    if r.get("model_type") == "embedding":
        return {"thr": "cls_s", "thr_label": "embeddings/s", "cost": "usd_per_1m_items", "cost_label": "$ per 1M embeddings",
                "lat": "e2e_p99", "lat_label": "whole request, 99th pct (ms)", "lat_sla": "p99_e2e_ms"}
    if r.get("direction"):
        return {"thr": "chars_s", "thr_label": "characters/s", "cost": "usd_per_1m_chars", "cost_label": "$ per 1M characters",
                "lat": "e2e_p99", "lat_label": "whole translation, 99th pct (ms)", "lat_sla": "p99_e2e_ms"}
    if r.get("workload"):
        return {"thr": "docs_s", "thr_label": "documents/s", "cost": "usd_per_gb", "cost_label": "$ per GB scanned",
                "lat": "e2e_p99", "lat_label": "whole decision, 99th pct (ms)", "lat_sla": "p99_e2e_ms"}
    if r.get("kind") == "classifier":
        return {"thr": "cls_s", "thr_label": "classifications/s", "cost": "usd_per_1m_items", "cost_label": "$ per 1M classifications",
                "lat": "e2e_p99", "lat_label": "whole answer, 99th pct (ms)", "lat_sla": "p99_e2e_ms"}
    return {"thr": "agg_out_tok_s", "thr_label": "tokens/s", "cost": "usd_per_1m_tokens", "cost_label": "$ per 1M tokens",
            "lat": "ttft_p99", "lat_label": "first word, 99th pct (ms)", "lat_sla": "p99_ttft_ms"}


def _series_name(r: dict) -> str:
    side = "GPU" if r["accelerator"] == "gpu" else "CPU"
    eng = {"vllm": "vLLM", "ovms": "OpenVINO", "thin-openvino": "OpenVINO", "thin-torch": "PyTorch"}.get(r["engine"], r["engine"])
    return f"{side} · {r['instance_type']} · {eng} · {r['precision']}"


def _util_point(r: dict) -> dict | None:
    """The utilization a chart point carries: the side's own processor (CPU as the share of physical
    cores, bench/utilization.cpu_scale), host memory, GPU memory."""
    u = r.get("util") or {}
    if not u:
        return None
    label = "gpu" if r["accelerator"] == "gpu" else "cpu"
    scale = utilization.cpu_scale(pairing.spec(r.get("instance_type"))) if label == "cpu" else 1.0
    own = utilization.busy(r, label, scale)
    gm = u.get("gpu_mem") or {}
    gm_val = gm.get("max", gm.get("mean"))
    mem = u.get("mem") or {}
    return {"busy_mean": own["mean"], "busy_p95": own["p95"], "busy_max": own["max"], "mem_max": mem.get("max", mem.get("mean")),
            "gpu_mem_pct": (100 * gm_val / gm["total"]) if gm_val is not None and gm.get("total") else None,
            "cpu_mean": (u.get("cpu") or {}).get("mean"), "gpu_mean": (u.get("gpu") or {}).get("mean"), "cpu_scale": scale}


def _best(rows: list[dict], thr: str):
    ok = [r for r in rows if r.get("meets_user_sla") and r.get(thr)]
    return max(ok, key=lambda r: r[thr]) if ok else None


def _acc(summary: dict, r: dict) -> dict | None:
    acc = summary.get("dspm_accuracy") or {}
    name = (r.get("workload") or "") + (f"-{r['bucket']}" if r.get("bucket") else "")
    a = (acc.get(r.get("cell_key") or "") or {}).get(name)
    if not a:
        return None
    return {"recall": a["recall"]["value"], "recall_ci": a["recall"]["ci95"], "fpr": a["fpr"]["value"], "fpr_ci": a["fpr"]["ci95"],
            "precision": a["precision"]["value"], "f1": a["f1"], "format_adherence": a["format_adherence"]["value"],
            "agreement_with_gpu": (a.get("vs_gpu") or {}).get("agreement", {}).get("value"),
            "qualifies_vs_gpu": (a.get("vs_gpu") or {}).get("qualifies"), "fit": bool(a["fitness"]["fit"])}


def _eq(summary: dict, r: dict) -> dict | None:
    """Embedding quality of this row's cell (report.embedding_quality)."""
    a = (summary.get("embedding_quality") or {}).get(r.get("cell_key") or "")
    if not a:
        return None
    vs = a.get("vs_reference_cell") or {}
    rt = a.get("retrieval") or {}
    return {"n": a.get("n"), "dim": a.get("dim"), "cosine_mean": vs.get("cosine_mean"), "cosine_min": vs.get("cosine_min"),
            "recall_de": (rt.get("de") or {}).get("recall_at_1"), "recall_es": (rt.get("es") or {}).get("recall_at_1"),
            "flag": a.get("flag")}


def _tq(summary: dict, r: dict) -> dict | None:
    """Translation quality of this row's cell on its direction (report.translation_quality)."""
    tq = summary.get("translation_quality") or {}
    if not r.get("direction"):
        return None
    a = (tq.get(r.get("cell_key") or "") or {}).get(f"{r['direction']}-{r.get('text_size')}")
    if not a:
        return None
    vs = a.get("vs_reference_cell") or {}
    return {"chrf": a.get("chrf"), "chrf_p10": a.get("chrf_p10"), "length_ratio": a.get("length_ratio"),
            "truncated_pct": a.get("truncated_pct"), "n": a.get("n"), "chrf_vs_gpu": vs.get("chrf"),
            "exact_match_vs_gpu": vs.get("exact_match_rate"), "flag": a.get("flag")}


_TARGETS: dict = {}


def _accelerator(tid: str) -> str:
    if not _TARGETS:
        try:
            _TARGETS.update({k: v.accelerator for k, v in
                             report.load_targets(report.ROOT / "config" / "targets.yaml").resolved().items()})
        except Exception as e:  # noqa: BLE001 - fall back to the id convention below
            _TARGETS["_error"] = str(e)[:80]
    return _TARGETS.get(tid) or ("gpu" if "-g" in tid else "cpu")


def not_run(summary: dict) -> list[dict]:
    """Hardware options the run planned but that produced no measurement (the GPU side with no
    capacity, for example), each with a plain-language reason. Reports written since 2026-11-11
    carry `not_run`; for older ones it is derived from the run's cells and capacity log."""
    if summary.get("not_run") is not None:
        return [{**nr, "run_id": summary["run_id"]} for nr in summary["not_run"]]
    run = summary.get("run") or {}
    planned = {c["target"] for c in run.get("cells") or []}
    ran = {r["target"] for r in summary.get("rows") or []}
    caps_all = summary.get("capacity_errors") or {}
    budget = (run.get("purchase") or {}).get("max_launch_attempts")
    out = []
    for tid in sorted(planned - ran):
        caps = caps_all.get(tid) or []
        side = "GPU" if _accelerator(tid) == "gpu" else "CPU"
        types = sorted({str(c.get("instance_type")) for c in caps if c.get("instance_type")})
        mk = {"spot": "Spot", "on-demand": "On-Demand"}
        markets = sorted({mk.get(str(c.get("market")), str(c.get("market"))) for c in caps if c.get("market")})
        if caps:
            reason = (f"No capacity: AWS refused {len(caps)} launch request(s) for {', '.join(types) or 'the requested size'} "
                      f"({' and '.join(markets) or 'requested market'}); the run gave up"
                      f"{f' after at most {budget} attempts' if budget else ''}. There are no {side} numbers in this report.")
        else:
            reason = f"No instance served it (no capacity refusal was logged). There are no {side} numbers in this report."
        out.append({"run_id": summary["run_id"], "target": tid, "side": side,
                    "instance_type": (run.get("instance_types") or {}).get(tid), "refusals": len(caps), "reason": reason})
    return out


COST_KEYS = ("usd_per_1m_tokens", "usd_per_1k_requests", "usd_per_1m_items", "usd_per_1m_docs", "usd_per_1m_chars")


def reprice(r: dict, pricing: dict | None, catalog: dict) -> dict:
    """The row under the chosen pricing basis (bench/demand.price_per_hour): hourly price and every
    per-unit cost scaled by the same factor. The report's own numbers are On-Demand; they stay in
    usd_per_hour_on_demand."""
    pr = demand.price_per_hour(r["instance_type"], r["accelerator"], pricing, catalog)
    od = r.get("usd_per_hour")
    r["usd_per_hour_on_demand"] = od
    r["price_label"] = pr["label"]
    if pr["usd_per_hour"] is None or not od:
        return r
    factor = pr["usd_per_hour"] / od
    r["usd_per_hour"] = pr["usd_per_hour"]
    for k in COST_KEYS:
        if r.get(k) is not None:
            r[k] = r[k] * factor
    return r


def analyze(summaries: list[dict], sla: dict, limit: int = 4, pricing: dict | None = None,
            demand_rpm: float | None = None) -> dict:
    """Rows from every run, re-judged against the given targets, grouped by question. pricing: the basis
    (On-Demand, a Savings Plan, or per-side overrides) applied to every price and cost; demand_rpm: the
    required load in requests per minute, for the cheapest-option-at-your-load plan per group."""
    catalog = pairing.load_catalog()
    rows = []
    for s in summaries:
        for r in s["rows"]:
            r = dict(r)
            r["run_id"] = s["run_id"]
            r["meets_user_sla"], r["user_sla_reason"] = report.meets_sla(r, sla)
            r["accuracy"] = _acc(s, r)
            r["translation"] = _tq(s, r)
            r["embedding"] = _eq(s, r)
            reprice(r, pricing, catalog)
            rows.append(r)
    missing = [nr for s in summaries for nr in not_run(s)]
    # A side counts as missing only when NO selected run measured it.
    measured_sides = {r["accelerator"] for r in rows}
    missing_side = {side: next((nr for nr in missing if nr["side"].lower() == side), None)
                    for side in ("gpu", "cpu") if side not in measured_sides}
    groups: dict[str, dict] = {}
    for r in rows:
        key, label = _group_key(r)
        groups.setdefault(key, {"key": key, "label": label, "rows": []})["rows"].append(r)
    mech_ctx = mechanics.context(summaries)
    out_groups = []
    for g in groups.values():
        grows = g["rows"]
        u = _unit(grows[0])
        gpu_rows = [r for r in grows if r["accelerator"] == "gpu"]
        cpu_rows = [r for r in grows if r["accelerator"] == "cpu" and r.get("amx") != "INVALID"]
        best_gpu, best_cpu = _best(gpu_rows, u["thr"]), _best(cpu_rows, u["thr"])

        def card(r, u=u):
            if r is None:
                return None
            return {"series": _series_name(r), "instance_type": r["instance_type"], "engine": r["engine"],
                    "precision": r["precision"], "conc": r["conc"], "batch": r.get("batch", 1), "usd_per_hour": r["usd_per_hour"],
                    "usd_per_hour_on_demand": r.get("usd_per_hour_on_demand"), "price_label": r.get("price_label"),
                    "savings": demand.savings_options(r["instance_type"], catalog),
                    "throughput": r.get(u["thr"]), "latency": r.get(u["lat"]), "cost": r.get(u["cost"]),
                    "quality_delta": r.get("quality_delta"), "quality_flag": r.get("quality_flag"), "amx": r.get("amx"),
                    "accuracy": r.get("accuracy"), "translation": r.get("translation"), "embedding": r.get("embedding"),
                    "run_id": r["run_id"], "ttft_p50": r.get("ttft_p50"), "tpot_p50": r.get("tpot_p50"),
                    "e2e_p50": r.get("e2e_p50"), "e2e_p99": r.get("e2e_p99")}

        verdict = None
        equivalence = []
        if best_gpu:
            for r in cpu_rows:
                if r.get("meets_user_sla") and r.get(u["thr"]):
                    n = math.ceil(best_gpu[u["thr"]] / r[u["thr"]])
                    equivalence.append({"series": _series_name(r), "instance_type": r["instance_type"], "precision": r["precision"],
                                        "conc": r["conc"], "instances_per_gpu": n,
                                        "hourly_cost_ratio": n * r["usd_per_hour"] / best_gpu["usd_per_hour"],
                                        "within_limit": n <= limit, "run_id": r["run_id"]})
            equivalence.sort(key=lambda e: e["instances_per_gpu"])
        if best_gpu and best_cpu:
            n = math.ceil(best_gpu[u["thr"]] / best_cpu[u["thr"]])
            ratio = n * best_cpu["usd_per_hour"] / best_gpu["usd_per_hour"]
            basis = (pricing or {}).get("basis") or "on_demand"
            basis_note = "" if basis == "on_demand" and not any((pricing or {}).get(k) is not None for k in ("cpu_usd_per_hour", "gpu_usd_per_hour")) \
                else f" (prices: {best_cpu.get('price_label')} for the CPU, {best_gpu.get('price_label')} for the GPU)"
            verdict = {"instances_per_gpu": n, "hourly_cost_ratio": ratio, "within_limit": n <= limit, "limit": limit,
                       "text": (f"To match one {best_gpu['instance_type']}'s peak throughput you need {n} × {best_cpu['instance_type']}, costing "
                                f"{ratio:.1f}x as much per hour{basis_note}, " + ("within your limit: the CPU side is a viable alternative."
                                                                                  if n <= limit else f"above your limit of {limit}: conditional."))}
        elif missing_side.get("gpu") and not gpu_rows:
            verdict = {"text": "The GPU side did not run (no capacity), so there is nothing to compare the CPU numbers "
                               "with. The CPU figures below stand alone; no verdict is possible.", "within_limit": None,
                       "missing_side": "gpu"}
        elif missing_side.get("cpu") and not cpu_rows:
            verdict = {"text": "The CPU side did not run (no capacity), so there is nothing to compare the GPU numbers "
                               "with. The GPU figures below stand alone; no verdict is possible.", "within_limit": None,
                       "missing_side": "cpu"}
        elif best_gpu and not best_cpu:
            verdict = {"text": "No CPU measurement meets your targets; the GPU is the only option that does.", "within_limit": False}
        elif best_cpu and not best_gpu:
            verdict = {"text": "No GPU measurement meets your targets; only CPU measurements do.", "within_limit": None}
        else:
            verdict = {"text": "No measurement on either side meets your targets. Relax them to see the picture.", "within_limit": None}
        points = [{"series": _series_name(r), "side": r["accelerator"], "conc": r["conc"], "batch": r.get("batch", 1),
                   "throughput": r.get(u["thr"]), "latency": r.get(u["lat"]), "cost": r.get(u["cost"]),
                   "meets": bool(r.get("meets_user_sla")), "reason": r.get("user_sla_reason"), "run_id": r["run_id"],
                   "instance_type": r["instance_type"], "precision": r["precision"], "engine": r["engine"],
                   "e2e_p50": r.get("e2e_p50"), "e2e_p99": r.get("e2e_p99"), "ttft_p99": r.get("ttft_p99"),
                   "tpot_p50": r.get("tpot_p50"), "req_s": r.get("req_s"), "error_pct": r.get("error_pct"),
                   "amx": r.get("amx"), "quality_delta": r.get("quality_delta"), "quality_flag": r.get("quality_flag"),
                   "accuracy": r.get("accuracy"), "translation": r.get("translation"), "embedding": r.get("embedding"),
                   "util": _util_point(r)}
                  for r in grows]
        sizing = utilization.sizing_for_rows(grows, _series_name, lambda t: pairing.spec(t), pairing.load_catalog()["instances"])
        for z in sizing:
            sug = z.get("suggestion")
            other = best_cpu if z["side"] == "gpu" else best_gpu
            if sug and sug.get("instance_type") and other:
                j = pairing.judge(sug["instance_type"], other["instance_type"]) if z["side"] == "cpu" else \
                    pairing.judge(other["instance_type"], sug["instance_type"])
                sug["pairing"] = {"other": other["instance_type"], "ratio": j.get("ratio"), "fair": j.get("fair"),
                                  "text": (f"Pairs fairly with {other['instance_type']} on the other side (GPU price {j['ratio']}x the CPU's)." if j.get("fair")
                                           else f"Against {other['instance_type']} on the other side the GPU price would be {j['ratio']}x the CPU's, outside "
                                                f"this tool's fair band (0.5x-2x); change the other side too for a like-for-like comparison.")}
        try:
            mech = mechanics.explain_group(grows, *mech_ctx)
        except Exception as e:  # noqa: BLE001 - an explanation never blocks the results
            mech = {"error": str(e)[:200]}
        try:
            dplan = demand.plan(grows, (demand_rpm or 0) / 60 or None, pricing, catalog, _series_name,
                                params=mech_ctx[0].get(grows[0]["model"]))
        except Exception as e:  # noqa: BLE001 - the plan never blocks the results
            dplan = {"error": str(e)[:200]}
        out_groups.append({"key": g["key"], "label": g["label"], "units": u, "best_gpu": card(best_gpu), "best_cpu": card(best_cpu),
                           "verdict": verdict, "equivalence": equivalence, "points": points, "sizing": sizing, "mechanics": mech,
                           "demand_plan": dplan,
                           "missing": {"gpu": (missing_side.get("gpu") or {}).get("reason") if not gpu_rows else None,
                                       "cpu": (missing_side.get("cpu") or {}).get("reason") if not cpu_rows else None},
                           "series": sorted({_series_name(r) for r in grows})})
    pairs = []
    for s in summaries:
        for j in s.get("pairing") or []:
            pairs.append({**j, "run_id": s["run_id"]})
    # cross-run pairs: every GPU type against every CPU type that appears
    gpu_types = sorted({r["instance_type"] for r in rows if r["accelerator"] == "gpu"})
    cpu_types = sorted({r["instance_type"] for r in rows if r["accelerator"] == "cpu"})
    cross = [pairing.judge(c, g) for c in cpu_types for g in gpu_types]
    intel_ref = [r for s in summaries for r in (s.get("intel_reference") or [])]
    seen = set()
    intel_ref = [r for r in intel_ref if not (tuple(sorted(r.items())) in seen or seen.add(tuple(sorted(r.items()))))]
    basis = (pricing or {}).get("basis") or "on_demand"
    try:
        where = cpu_case.build(out_groups, {g["key"]: g["rows"] for g in groups.values()}, summaries, pricing, catalog, _series_name, mech_ctx[0])
    except Exception as e:  # noqa: BLE001 - the statement never blocks the results
        where = {"error": str(e)[:200]}
    return {"groups": out_groups, "pairing": cross, "run_pairing": pairs, "not_run": missing, "where_cpu_wins": where,
            "warnings": [w for s in summaries for w in (s.get("warnings") or [])],
            "smoke": any(s.get("smoke") for s in summaries), "intel_reference": intel_ref,
            "pricing": {"basis": basis, "label": demand.PRICING_BASES.get(basis, ("On-Demand",))[0],
                        "cpu_usd_per_hour": (pricing or {}).get("cpu_usd_per_hour"), "gpu_usd_per_hour": (pricing or {}).get("gpu_usd_per_hour"),
                        "snapshot": catalog.get("price_snapshot")},
            "demand_rpm": demand_rpm, "typical_loads": demand.TYPICAL_LOADS,
            "how_to_read": HOW_TO_READ}


HOW_TO_READ = (
    "On AWS On-Demand pricing a GPU instance beats a price-matched Intel AMX CPU instance on peak throughput for every "
    "workload this tool has measured: at the same hourly price one L4 carries about 7.5x the matrix arithmetic and 2.7x the "
    "memory bandwidth of eight Xeon 6 cores, and the CPU side here runs at 70-80% of its own hardware limits with AMX "
    "verified. 'How many CPUs replace one GPU' therefore always favours the GPU. The CPU case is a different question: "
    "the cheapest option that meets your targets at the load you actually have (a small CPU slice costs a quarter of the "
    "smallest GPU instance), capacity you already pay for, or no GPU available. Set your load and prices above to see it.")


def default_sla(summaries: list[dict]) -> dict:
    run = (summaries[0].get("run") or {}) if summaries else {}
    if run.get("suite") == "dspm":
        preset = (run.get("dspm") or {}).get("sla_preset") or "inline_standard"
        return {"p99_e2e_ms": PRESETS[preset][2], "max_error_pct": 1.0, "preset": preset}
    sla = run.get("sla") or {}
    return {k: sla.get(k) for k in ("p99_ttft_ms", "p50_tpot_ms", "p99_e2e_ms")} | {"max_error_pct": sla.get("max_error_pct", 1.0)}
