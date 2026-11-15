"""Executive summary of a results analysis, written by the newest Claude Opus on Amazon
Bedrock. The model is given only the computed findings (verdicts, best measurements,
equivalence, accuracy, comparability, caveats) and is told to use no number that is not in
them. The model id is discovered at call time, so a newer Opus is picked up automatically."""

from __future__ import annotations

import json
import re
import time

_MODEL_CACHE: dict = {}

SYSTEM = """You write executive summaries for product and business decision makers choosing between
NVIDIA GPU instances and Intel AMX CPU instances on AWS for running AI models. The reader is not an
engineer. Write in plain English, short paragraphs, no jargon without a one-clause explanation.

Rules:
- Use only the numbers and facts in the FINDINGS JSON. Never invent, round away a caveat, or extrapolate.
- When a finding is a quick check (smoke), say clearly that it is indicative and not decision-grade.
- Capacity (how many CPU machines replace one GPU, and whether that is within the stated limit) is the
  decision criterion; cost is context.
- Mention accuracy and answer-quality findings where present; speed alone is not a recommendation.
- If "hardware_that_did_not_run" is not empty, say so in the first sentence of the bottom line: which side
  (GPU or CPU) did not run and why (no capacity), and that the other side's numbers therefore stand alone
  with no comparison and no GPU-vs-CPU verdict. Never present a one-sided result as a comparison.
- Structure, with these exact headings in Markdown: "## Bottom line", "## What was tested",
  "## Findings", "## Caveats", "## Recommendation", "## Suggested next step".
- Bottom line: at most three sentences. Whole summary: at most 450 words.
- Say "measurement" or "load setting", never "level"; say "repeated" rather than "passes".
"""


def newest_opus(bedrock) -> dict:
    """Pick the newest Anthropic Opus the account can call: highest version numbers in the model
    id (claude-opus-5-5 > claude-opus-5 > claude-opus-4-8 ...), ACTIVE, with a us. inference
    profile when the model is profile-only."""
    if _MODEL_CACHE.get("at", 0) > time.time() - 3600:
        return _MODEL_CACHE["model"]
    models = bedrock.list_foundation_models(byProvider="anthropic")["modelSummaries"]
    profiles = {p["inferenceProfileId"] for p in bedrock.list_inference_profiles()["inferenceProfileSummaries"]}

    def version(mid: str) -> tuple:
        m = re.search(r"opus-(\d+)(?:-(\d+))?", mid)
        if not m:
            return (0,)
        major, minor = int(m.group(1)), int(m.group(2) or 0)
        if major == 3:              # claude-3-opus is the oldest family
            return (3, 0)
        return (major, minor)

    cands = []
    for m in models:
        mid = m["modelId"]
        if "opus" not in mid or m.get("modelLifecycle", {}).get("status") != "ACTIVE":
            continue
        types = set(m.get("inferenceTypesSupported") or [])
        call_id = mid if "ON_DEMAND" in types else (f"us.{mid}" if f"us.{mid}" in profiles else None)
        if call_id:
            cands.append((version(mid), mid, call_id, m.get("modelName", mid)))
    if not cands:
        raise RuntimeError("no callable Anthropic Opus model found in Bedrock for this account and region")
    cands.sort(reverse=True)
    v, mid, call_id, name = cands[0]
    out = {"model_id": mid, "call_id": call_id, "name": name, "version": ".".join(map(str, v))}
    _MODEL_CACHE.update(model=out, at=time.time())
    return out


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
                                                            for p in g["points"] if not p["meets"]][:12]})
    return {"targets": sla, "quick_check_only": analysis.get("smoke", False),
            "runs": [{"run_id": r["run_id"], **r["shape"]} for r in runs],
            "comparability": {k: comparability.get(k) for k in ("ok", "blockers", "warnings")},
            "hardware_that_did_not_run": [{k: nr.get(k) for k in ("run_id", "side", "target", "instance_type", "reason")}
                                          for nr in analysis.get("not_run") or []],
            "instance_pairing": analysis.get("pairing"), "findings": groups,
            "notes_from_reports": sorted(set(analysis.get("warnings") or []))[:12]}


def write(analysis: dict, comparability: dict, sla: dict, runs: list[dict]) -> dict:
    from webui.api import aws

    bedrock = aws.client("bedrock")
    rt = aws.client("bedrock-runtime")
    model = newest_opus(bedrock)
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
