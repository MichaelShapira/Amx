"""Results analysis for the web UI: a run (or several) summarized into verdict cards, series
for charts, and a comparability check that is re-evaluated on every selection.

Two measurements are comparable when they answer the same question: same use case, same
data (prompt set and answer length, or scenarios plus dataset and policy fingerprints), the
same request shape (single request / N concurrent / batched), and the GPU and CPU instances
in a fair price band. Anything else is reported as a blocker or a warning, in words."""

from __future__ import annotations

import math

from bench import pairing, report
from webui.api.catalog import MODELS, PRESETS, PROMPT_SETS, WORKLOADS

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
         "requests_per_level": (run.get("per_level") or {}).get("requests"), "passes": run.get("repeats")}
    if run.get("suite") == "standard":
        s.update(prompt_set=run.get("prompt_set"), answer_tokens=(run.get("sampling") or {}).get("max_new_tokens"),
                 temperature=(run.get("sampling") or {}).get("temperature"))
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


def analyze(summaries: list[dict], sla: dict, limit: int = 4) -> dict:
    """Rows from every run, re-judged against the given targets, grouped by question."""
    rows = []
    for s in summaries:
        for r in s["rows"]:
            r = dict(r)
            r["run_id"] = s["run_id"]
            r["meets_user_sla"], r["user_sla_reason"] = report.meets_sla(r, sla)
            r["accuracy"] = _acc(s, r)
            rows.append(r)
    groups: dict[str, dict] = {}
    for r in rows:
        key, label = _group_key(r)
        groups.setdefault(key, {"key": key, "label": label, "rows": []})["rows"].append(r)
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
                    "throughput": r.get(u["thr"]), "latency": r.get(u["lat"]), "cost": r.get(u["cost"]),
                    "quality_delta": r.get("quality_delta"), "amx": r.get("amx"), "accuracy": r.get("accuracy"),
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
            verdict = {"instances_per_gpu": n, "hourly_cost_ratio": ratio, "within_limit": n <= limit, "limit": limit,
                       "text": (f"To match one {best_gpu['instance_type']} you need {n} × {best_cpu['instance_type']}, costing "
                                f"{ratio:.1f}x as much per hour, " + ("within your limit: the CPU side is a viable alternative."
                                                                     if n <= limit else f"above your limit of {limit}: conditional."))}
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
                   "amx": r.get("amx"), "quality_delta": r.get("quality_delta"), "accuracy": r.get("accuracy")}
                  for r in grows]
        out_groups.append({"key": g["key"], "label": g["label"], "units": u, "best_gpu": card(best_gpu), "best_cpu": card(best_cpu),
                           "verdict": verdict, "equivalence": equivalence, "points": points,
                           "series": sorted({_series_name(r) for r in grows})})
    pairs = []
    for s in summaries:
        for j in s.get("pairing") or []:
            pairs.append({**j, "run_id": s["run_id"]})
    # cross-run pairs: every GPU type against every CPU type that appears
    gpu_types = sorted({r["instance_type"] for r in rows if r["accelerator"] == "gpu"})
    cpu_types = sorted({r["instance_type"] for r in rows if r["accelerator"] == "cpu"})
    cross = [pairing.judge(c, g) for c in cpu_types for g in gpu_types]
    return {"groups": out_groups, "pairing": cross, "run_pairing": pairs,
            "warnings": [w for s in summaries for w in (s.get("warnings") or [])],
            "smoke": any(s.get("smoke") for s in summaries)}


def default_sla(summaries: list[dict]) -> dict:
    run = (summaries[0].get("run") or {}) if summaries else {}
    if run.get("suite") == "dspm":
        preset = (run.get("dspm") or {}).get("sla_preset") or "inline_standard"
        return {"p99_e2e_ms": PRESETS[preset][2], "max_error_pct": 1.0, "preset": preset}
    sla = run.get("sla") or {}
    return {k: sla.get(k) for k in ("p99_ttft_ms", "p50_tpot_ms", "p99_e2e_ms")} | {"max_error_pct": sla.get("max_error_pct", 1.0)}
