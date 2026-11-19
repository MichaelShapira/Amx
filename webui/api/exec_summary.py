"""Executive summary of a results analysis, written by the newest Claude Opus on Amazon
Bedrock. The model is given only the computed findings (verdicts, best measurements,
equivalence, accuracy, comparability, caveats) and is told to use no number that is not in
them. The model is the one chosen in Setup (webui/api/settings.py); default: the newest Opus."""

from __future__ import annotations

import json
import time

SYSTEM = """You write executive summaries for product and business decision makers choosing between
NVIDIA GPU instances and Intel AMX CPU instances on AWS for running AI models. The reader is not an
engineer. Write in plain English, short paragraphs, no jargon without a one-clause explanation.

Rules:
- Use only the numbers and facts in the FINDINGS JSON. Never invent, round away a caveat, or extrapolate.
- When a finding is a quick check (smoke), say clearly that it is indicative and not decision-grade.
- Capacity (how many CPU machines replace one GPU, and whether that is within the stated limit) is the
  decision criterion; cost is context.
- Mention accuracy and answer-quality findings where present; speed alone is not a recommendation. For
  translation findings, "translation_quality.chrf" is the character-level similarity (0-100) of the option's
  translations to the reference text and "chrf_vs_gpu" the similarity to the GPU's own translations; read
  them relative to each other (the same model on two machines should score alike), never as an absolute
  grade, and name a flagged option as having lost quality.
- Say whether each machine was the right size, from "hardware_sizing": it gives, per hardware option, how
  busy the GPU or CPU was at the heaviest load setting and a verdict (over = mostly idle, bigger than this
  load needs; good = working hard with headroom; limit = fully used and still meeting the targets; under =
  saturated and missing the targets; unknown = no samples). Quote the busy percentage. A mostly idle
  machine means a smaller size would do for this load, or that the test's load was too light to show the
  machine's limit; say which reading applies given the load settings tested. Never call a comparison
  settled when one side was mostly idle and the other at its limit. When "suggested_instance" names a
  size, recommend it by name with its hourly price and say why (the tool chose it from the same family
  so the same load would run about 60% busy, or the next size up / stronger GPU for a saturated one);
  repeat its pairing note when the change would break the fair price band.
- If "hardware_that_did_not_run" is not empty, say so in the first sentence of the bottom line: which side
  (GPU or CPU) did not run and why (no capacity), and that the other side's numbers therefore stand alone
  with no comparison and no GPU-vs-CPU verdict. Never present a one-sided result as a comparison.
- Structure, with these exact headings in Markdown: "## Bottom line", "## What was tested",
  "## Findings", "## Was the hardware the right size?", "## Caveats", "## Recommendation",
  "## Suggested next step", "## Technical details".
- Bottom line: at most three sentences. Whole summary before "Technical details": at most 520 words.
- "## Technical details" (120-220 words) explains WHY the CPU did well or badly here, for a reader who is
  not an engineer, from each finding's "why_it_came_out_this_way" facts. Use this picture: a model answers
  in two phases, READING the input (all input words processed at once, the model's weights fetched from
  memory once, limited by arithmetic, which Intel's AMX units are built for) and WRITING the answer (one
  word at a time, the whole set of weights re-read from memory for every word, limited by memory
  bandwidth, where a price-matched CPU slice is furthest behind a GPU). State the task's reading/writing
  shape, quote the bandwidth figures and shares given, say which side ran at its hardware limit, and
  conclude whether a different setting could change the result (it cannot when both sides sit at their
  memory limit) and which kinds of work would favor the CPU (reading-heavy: classification, embeddings,
  extraction, detection). When "intel_published_figures" is present, put the run's per-core throughput
  next to Intel's per-core figures and explain the difference by core count, memory bandwidth and
  concurrency, not by software.
- Say "measurement" or "load setting", never "level"; say "repeated" rather than "passes".
"""


def _compact(analysis: dict, comparability: dict, sla: dict, runs: list[dict]) -> dict:
    """Only what the summary needs; the full analysis is large."""
    groups = []
    for g in analysis["groups"]:
        def card(c):
            if not c:
                return None
            out = {k: c.get(k) for k in ("instance_type", "engine", "precision", "conc", "batch", "usd_per_hour",
                                         "throughput", "latency", "cost", "quality_delta", "amx")}
            if c.get("accuracy"):
                a = c["accuracy"]
                out["accuracy"] = {k: a.get(k) for k in ("recall", "fpr", "f1", "agreement_with_gpu", "fit")}
            if c.get("translation"):
                t = c["translation"]
                out["translation_quality"] = {k: t.get(k) for k in ("chrf", "chrf_vs_gpu", "exact_match_vs_gpu",
                                                                     "length_ratio", "truncated_pct", "flag")}
            if c.get("embedding"):
                e = c["embedding"]
                out["embedding_quality"] = {k: e.get(k) for k in ("cosine_mean", "cosine_min", "recall_de", "recall_es", "flag")}
            return out
        groups.append({"question": g["label"], "units": {k: g["units"][k] for k in ("thr_label", "lat_label", "cost_label")},
                       "verdict": g["verdict"], "best_gpu": card(g["best_gpu"]), "best_cpu": card(g["best_cpu"]),
                       "cpu_options_meeting_targets": [{k: e[k] for k in ("instance_type", "precision", "conc",
                                                                           "instances_per_gpu", "hourly_cost_ratio", "within_limit")}
                                                       for e in g["equivalence"]],
                       "measurements_not_meeting_targets": [f"{p['series']} at {p['conc']}: {p['reason']}"
                                                            for p in g["points"] if not p["meets"]][:12],
                       "hardware_sizing": [{**{k: v.get(k) for k in ("series", "instance_type", "size", "verdict", "label", "load_top",
                                                                      "util_top", "util_p95_top", "util_low", "meets_top", "notes")},
                                            "suggested_instance": ({k: v["suggestion"].get(k) for k in ("instance_type", "kind", "why", "usd_per_hour",
                                                                                                        "usd_delta_per_hour")}
                                                                   | {"pairing": (v["suggestion"].get("pairing") or {}).get("text")})
                                            if v.get("suggestion") else None}
                                           for v in g.get("sizing") or []],
                       "why_it_came_out_this_way": _mechanics(g.get("mechanics") or {})})
    return {"targets": sla, "quick_check_only": analysis.get("smoke", False),
            "runs": [{"run_id": r["run_id"], **r["shape"]} for r in runs],
            "comparability": {k: comparability.get(k) for k in ("ok", "blockers", "warnings")},
            "hardware_that_did_not_run": [{k: nr.get(k) for k in ("run_id", "side", "target", "instance_type", "reason")}
                                          for nr in analysis.get("not_run") or []],
            "instance_pairing": analysis.get("pairing"), "findings": groups,
            "intel_published_figures": [{k: r.get(k) for k in ("model_id", "cpu_model", "sockets", "total_cores", "memory_gb", "max_concurrency",
                                                               "throughput_tok_s", "tok_s_per_core", "tpot_ms", "ttft_ms")}
                                        for r in (analysis.get("intel_reference") or [])] or None,
            "notes_from_reports": sorted(set(analysis.get("warnings") or []))[:12]}


def _mechanics(m: dict) -> dict | None:
    """The reading/writing facts of a finding (bench/mechanics.py), rounded for the prompt."""
    if not m or "shape" not in m:
        return None
    def r(x, nd=0):
        return round(x, nd) if isinstance(x, (int, float)) else x
    sides = {}
    for side, v in (m.get("sides") or {}).items():
        sides[side] = {"instance_type": v.get("instance_type"), "precision": v.get("precision"), "weights_gb": r(v.get("weights_gb"), 1),
                       "memory_bandwidth_gb_s": r(v.get("bandwidth_gb_s")), "bandwidth_source": v.get("bandwidth_source"),
                       "bandwidth_reached_gb_s_at_lowest_load": r(v.get("implied_bandwidth_gb_s")),
                       "share_of_bandwidth_used": r(v.get("bandwidth_utilization"), 2),
                       "arithmetic_tflops_at_heaviest_load": r(v.get("achieved_tflops"), 1),
                       "share_of_arithmetic_peak": r(v.get("compute_utilization"), 2), "cpu_cores": v.get("cores")}
    sh = m["shape"]
    return {"input_tokens_per_request": r(sh.get("input_tokens")), "output_tokens_per_request": r(sh.get("output_tokens")),
            "share_of_time_spent_writing": r(sh.get("writing_share"), 2), "generates_text": sh.get("generation"),
            "sides": sides, "plain_language": m.get("why")}


def write(analysis: dict, comparability: dict, sla: dict, runs: list[dict]) -> dict:
    from webui.api import aws

    rt = aws.client("bedrock-runtime")
    from webui.api import settings

    model = settings.reasoning_model()                 # Setup → Reasoning model; default newest Opus
    payload = _compact(analysis, comparability, sla, runs)
    prompt = ("FINDINGS JSON:\n" + json.dumps(payload, indent=1, default=str) +
              "\n\nWrite the executive summary now.")
    t0 = time.time()
    r = rt.converse(modelId=model["call_id"], system=[{"text": SYSTEM}],
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={"maxTokens": 3200})
    text = "".join(c.get("text", "") for c in r["output"]["message"]["content"])
    return {"markdown": text, "model": model, "usage": r.get("usage"), "seconds": round(time.time() - t0, 1),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
