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
- Mention accuracy and answer-quality findings where present; speed alone is not a recommendation.
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
  "## Suggested next step".
- Bottom line: at most three sentences. Whole summary: at most 520 words.
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
                                           for v in g.get("sizing") or []]})
    return {"targets": sla, "quick_check_only": analysis.get("smoke", False),
            "runs": [{"run_id": r["run_id"], **r["shape"]} for r in runs],
            "comparability": {k: comparability.get(k) for k in ("ok", "blockers", "warnings")},
            "hardware_that_did_not_run": [{k: nr.get(k) for k in ("run_id", "side", "target", "instance_type", "reason")}
                                          for nr in analysis.get("not_run") or []],
            "instance_pairing": analysis.get("pairing"), "findings": groups,
            "notes_from_reports": sorted(set(analysis.get("warnings") or []))[:12]}


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
                    inferenceConfig={"maxTokens": 2500})
    text = "".join(c.get("text", "") for c in r["output"]["message"]["content"])
    return {"markdown": text, "model": model, "usage": r.get("usage"), "seconds": round(time.time() - t0, 1),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
