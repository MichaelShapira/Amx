"""Executive summary of a results analysis, written by the newest Claude Opus on Amazon
Bedrock. The model is given only the computed findings (verdicts, best measurements,
equivalence, accuracy, comparability, caveats) and is told to use no number that is not in
them. The model is the one chosen in Setup (webui/api/settings.py); default: the newest Opus."""

from __future__ import annotations

import json
import re
import time

SYSTEM = """You write executive summaries for product and business decision makers choosing between
NVIDIA GPU instances and Intel AMX CPU instances on AWS for running AI models. The reader is not an
engineer. Write in plain English, short paragraphs, no jargon without a one-clause explanation.

The frame (state it, do not argue with it):
- On AWS pay-as-you-go pricing a GPU instance beats a price-matched Intel CPU instance on PEAK throughput for
  every workload this tool has measured, because at the same hourly price the GPU hardware carries several
  times the arithmetic and memory bandwidth; the CPU side here runs close to its own hardware limits with
  AMX verified, so no setting changes that. "How many CPU machines replace one GPU" therefore favours the
  GPU; report it, but do not present it as the decision.
- The decision is the cheapest option that meets the reader's targets AT THE READER'S LOAD. When
  "at_your_load" is present in a finding, it is the headline: name the cheapest option (instances,
  type, hourly and monthly cost), the cheapest option on the other side and the ratio, and the crossover
  load above which the GPU becomes cheaper. Say plainly when the cheapest option is an ESTIMATED smaller
  size ("estimated": true) that has not been measured, and that one run confirms it. When no load is
  given, say that the verdict assumes the reader needs a GPU's full peak and that setting a load changes
  the picture.
- "pricing" says which prices were used (On-Demand, a Savings Plan commitment, or the reader's own rate).
  Name it once. Where "savings_plans" are given for the best options, state the hourly price under the 1-
  and 3-year commitments and the saving in percent; a commitment lowers both sides and usually the CPU
  side a little more.

Rules:
- Use only the numbers and facts in the FINDINGS JSON. Never invent, round away a caveat, or extrapolate.
- Cover EVERY entry in "findings", not only the first: the "## Findings" section has one short paragraph or
  table row per finding, in the order given, each with its question, the best measurement on each side,
  the capacity ratio and, when present, the at-your-load result. A reader must be able to find each
  finding by its question.
- When a finding is a quick check (smoke), say clearly that it is indicative and not decision-grade.
- Mention accuracy and answer-quality findings where present; speed alone is not a recommendation. For
  translation findings, "translation_quality.chrf" is the character-level similarity (0-100) of the option's
  translations to the reference text and "chrf_vs_gpu" the similarity to the GPU's own translations; read
  them relative to each other (the same model on two machines should score alike), never as an absolute
  grade. A "quality_flag" is a warning that travels with the row, not an exclusion: say what it flags.
- Say whether each machine was the right size, from "hardware_sizing": it gives, per hardware option, how
  busy the GPU or CPU was at the heaviest load setting and a verdict (over = mostly idle, bigger than this
  load needs; good = working hard with headroom; limit = fully used and still meeting the targets; under =
  saturated and missing the targets; unknown = no samples). Quote the busy percentage. Never call a
  comparison settled when one side was mostly idle and the other at its limit. When "suggested_instance"
  names a size, recommend it by name with its hourly price and repeat its pairing note.
- If "hardware_that_did_not_run" is not empty, say so in the first sentence of the bottom line: which side
  did not run and why, and that the other side's numbers stand alone with no comparison.
- Structure, with these exact headings in Markdown: "## Bottom line", "## What was tested",
  "## Findings", "## Where the CPU is the better choice", "## At your load",
  "## Was the hardware the right size?", "## Caveats", "## Recommendation", "## Suggested next step",
  "## Technical details".
- "## Where the CPU is the better choice" is the section the reader came for. Build it ONLY from
  "where_cpu_wins": start with its headline; then one line per question from "cost_by_question" (the load
  range where the CPU is cheaper with the worked example, or the plain statement that the GPU is cheaper at
  every load, or that no CPU option met the targets); then availability exactly as its text says (when the
  runs show GPU launches refused for capacity, say that availability is a real benefit; when they show no
  difference, say the runs give no evidence); then the smallest-unit point when present; then the two
  structural points in one sentence each; and close with "not_won": peak throughput per dollar goes to the
  GPU on every question. When "any_benefit" is false, say in one sentence that the CPU has no benefit in
  these runs. Never add a benefit the data does not contain.
- Bottom line: at most three sentences. Whole summary before "Technical details": at most 800 words.
- "## At your load": when no finding has "at_your_load", one sentence saying that no load was set and
  what setting one would show; otherwise the headline result per finding.
- "## Technical details" (120-220 words) explains WHY the CPU did well or badly here, for a reader who is
  not an engineer, from each finding's "why_it_came_out_this_way" facts. Use this picture: a model answers
  in two phases, READING the input (all input words processed at once, the model's weights fetched from
  memory once, limited by arithmetic, which Intel's AMX units are built for) and WRITING the answer (one
  word at a time, the whole set of weights re-read from memory for every word, limited by memory
  bandwidth, where a price-matched CPU slice is furthest behind a GPU). State the task's reading/writing
  shape, quote the bandwidth and arithmetic shares given, say which side ran at its hardware limit, and
  conclude whether a different setting could change the result (it cannot when a side sits at its
  hardware limit) and which kinds of work and which loads favour the CPU. When "intel_published_figures"
  is present, put the run's per-core throughput next to Intel's per-core figures and explain the
  difference by core count, memory bandwidth and concurrency, not by software.
- Say "measurement" or "load setting", never "level"; say "repeated" rather than "passes".
- Output raw Markdown only: no preamble, and never wrap the text (or a table) in a ``` code fence; the page
  renders Markdown directly and a fence would show it as plain text.
"""
_FENCE = re.compile(r"\A\s*```[a-zA-Z]*[ \t]*\n(.*?)\n[ \t]*```\s*\Z", re.DOTALL)
def clean_markdown(text: str) -> str:
    """The model's output as the page should render it. A whole answer wrapped in one ```markdown fence
    (seen as 'the summary is displayed as plain text') is unwrapped; fences inside the text are left alone."""
    m = _FENCE.match(text)
    return m.group(1).strip() if m else text.strip()


def _compact(analysis: dict, comparability: dict, sla: dict, runs: list[dict]) -> dict:
    """Only what the summary needs; the full analysis is large."""
    groups = []
    for g in analysis["groups"]:
        def card(c):
            if not c:
                return None
            out = {k: c.get(k) for k in ("instance_type", "engine", "precision", "conc", "batch", "usd_per_hour",
                                         "throughput", "latency", "cost", "quality_delta", "quality_flag", "amx", "price_label")}
            if c.get("savings"):
                out["savings_plans"] = [{k: x.get(k) for k in ("label", "usd_per_hour", "saving_pct")} for x in c["savings"] if x.get("basis") != "on_demand"]
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
                       "why_it_came_out_this_way": _mechanics(g.get("mechanics") or {}),
                       "at_your_load": _plan(g.get("demand_plan"))})
    return {"targets": sla, "quick_check_only": analysis.get("smoke", False),
            "pricing": analysis.get("pricing"), "load_requests_per_minute": analysis.get("demand_rpm"),
            "frame": analysis.get("how_to_read"), "where_cpu_wins": _where(analysis.get("where_cpu_wins")),
            "runs": [{"run_id": r["run_id"], **r["shape"]} for r in runs],
            "comparability": {k: comparability.get(k) for k in ("ok", "blockers", "warnings")},
            "hardware_that_did_not_run": [{k: nr.get(k) for k in ("run_id", "side", "target", "instance_type", "reason")}
                                          for nr in analysis.get("not_run") or []],
            "instance_pairing": analysis.get("pairing"), "findings": groups,
            "intel_published_figures": [{k: r.get(k) for k in ("model_id", "cpu_model", "sockets", "total_cores", "memory_gb", "max_concurrency",
                                                               "throughput_tok_s", "tok_s_per_core", "tpot_ms", "ttft_ms")}
                                        for r in (analysis.get("intel_reference") or [])] or None,
            "notes_from_reports": sorted(set(analysis.get("warnings") or []))[:12]}


def _where(w: dict | None) -> dict | None:
    """bench/cpu_case.build, compacted: the computed statement of where the CPU is the better choice."""
    if not w or w.get("error"):
        return None
    return {"headline": w.get("headline"), "any_benefit": w.get("any_benefit"),
            "cost_by_question": [{"question": c["group"], "finding": c["kind"], "text": c["text"],
                                  "examples": [{k: e.get(k) for k in ("label", "cpu", "cpu_usd_per_month", "gpu", "gpu_usd_per_month", "ratio", "estimated")}
                                               for e in c.get("examples") or []][:3]} for c in w.get("cost") or []],
            "availability": {k: (w.get("availability") or {}).get(k) for k in ("kind", "gpu_refusals", "cpu_refusals", "text")},
            "smallest_unit": (w.get("granularity") or {}).get("text"),
            "not_won": w.get("not_won"), "structural": w.get("structural")}


def _plan(d: dict | None) -> dict | None:
    """The cheapest-at-your-load result of a finding (bench/demand.plan), compacted."""
    if not d or d.get("error") or not d.get("options"):
        return None
    def opt(o):
        if not o:
            return None
        return {k: o.get(k) for k in ("side", "instance_type", "engine", "precision", "instances", "usd_per_hour", "usd_per_month",
                                      "estimated", "capacity_basis", "price_label")}
    return {"requests_per_minute": d.get("demand_rpm"), "cheapest": opt(d.get("cheapest")), "cheapest_measured": opt(d.get("cheapest_measured")),
            "cheapest_gpu": opt(d.get("cheapest_gpu")), "cheapest_cpu": opt(d.get("cheapest_cpu")),
            "crossover": (d.get("crossover") or {}).get("text"), "plain_language": d.get("text")}


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
    # The budget covers the model's reasoning as well as the visible text: the newest Opus spent ~2,000 of a
    # 3,200-token budget thinking and the summary was cut mid-sentence in "Technical details" (2026-11-19,
    # outputTokens == maxTokens, 4,089 characters of text). Start high and go higher once if it still hits
    # the limit; a truncated result is marked so the caller does not cache it.
    for max_tokens in (8000, 16000):
        r = rt.converse(modelId=model["call_id"], system=[{"text": SYSTEM}],
                        messages=[{"role": "user", "content": [{"text": prompt}]}],
                        inferenceConfig={"maxTokens": max_tokens})
        if r.get("stopReason") != "max_tokens":
            break
    content = r["output"]["message"]["content"]
    text = clean_markdown("".join(c.get("text", "") for c in content))
    truncated = r.get("stopReason") == "max_tokens"
    if truncated:
        text += ("\n\n> The summary was cut at the model's output limit before it finished. Press Rewrite to generate "
                 "it again; if it happens twice, choose a smaller comparison (fewer runs or models).")
    return {"markdown": text, "model": model, "usage": r.get("usage"), "seconds": round(time.time() - t0, 1),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stop_reason": r.get("stopReason"),
            "truncated": truncated, "reasoning_blocks": sum(1 for c in content if "reasoningContent" in c)}
