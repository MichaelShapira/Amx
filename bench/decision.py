"""The decision suite: synthetic agent decisions for a Strands Decider, and the scoring of its answers.

A decision model answers typed questions about a state in one forward pass (no generation):
  noul   a yes/no question -> the probability of "yes"
  choice one of N options  -> the chosen option, its confidence, and a probability per option
  score  a level on an ordered scale -> the expected level, its confidence, a probability per level
The payload is the server's own (`POST /v1/systemone`, strands-labs/strands-decider docs/inference.md):
  {"state": "<text>", "questions": {"<name>": {"type": "noul"|"choice"|"score", "instructions": "...",
                                               "criteria": {option: description} | [levels]}}}
and the answer carries, per question, `noul`, or `choice` + `confidence` + `probabilities`, or `score` +
`confidence` + `probabilities`, plus `usage.input_tokens`.

The workload is the model's intended use: the decisions inside an agent (routing a request to a team,
checking whether a tool call should proceed, rating severity). Every item is generated here from templates
with a deterministic seed, carries gold labels for the three questions, and exists in two state sizes:
`short` (a message, 30-80 words) and `long` (a ticket thread of 400-700 words with the deciding facts
spread through it). Items are JSON-serialisable and tokenizer-free; the pool builder adds token counts.

Scoring: a cell's answers are compared (a) with the gold labels (accuracy per question type) and (b)
with the GPU reference cell's answers (agreement, and the mean absolute probability gap), since the two
sides run the same code and the project reports CPU-vs-GPU probabilities within 0.05.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

SIZES = ("short", "long")
QUESTION_TYPES = ("noul", "choice", "score")

TEAMS = {"billing": "payments, invoices, refunds, charges", "security": "account access, suspicious activity, credentials",
         "shipping": "delivery, tracking, lost or damaged parcels", "technical": "errors, crashes, integrations, API",
         "sales": "quotes, upgrades, pricing, new contracts"}
SEVERITY = ["minor", "moderate", "major", "critical"]

# One scenario family per team; each carries the facts that decide the three questions.
SCENARIOS: dict[str, list[dict[str, Any]]] = {
    "billing": [
        {"issue": "was charged twice for the same order", "detail": "two identical charges of {amt} on {day}", "urgent": True, "sev": "major"},
        {"issue": "have not received a refund promised {n} days ago", "detail": "refund reference {ref}", "urgent": False, "sev": "moderate"},
        {"issue": "noticed a small rounding difference on an invoice", "detail": "invoice {ref}, off by a few cents", "urgent": False, "sev": "minor"},
        {"issue": "found that all payouts to my account have failed for {n} days", "detail": "every payout since {day} rejected", "urgent": True, "sev": "critical"},
    ],
    "security": [
        {"issue": "received a login alert from a country I have never visited", "detail": "login at {time} from an unknown device", "urgent": True, "sev": "critical"},
        {"issue": "want to turn on two-factor authentication", "detail": "no incident, a preference", "urgent": False, "sev": "minor"},
        {"issue": "think a former employee still has access", "detail": "the user left on {day} and the account is still active", "urgent": True, "sev": "major"},
        {"issue": "forgot my password and the reset email never arrives", "detail": "three reset attempts since {time}", "urgent": False, "sev": "moderate"},
    ],
    "shipping": [
        {"issue": "see my parcel marked as delivered but nothing arrived", "detail": "tracking {ref} says delivered on {day}", "urgent": True, "sev": "major"},
        {"issue": "want to change the delivery address before dispatch", "detail": "order placed {n} hours ago, not yet shipped", "urgent": True, "sev": "moderate"},
        {"issue": "would like to know when a back-ordered item will ship", "detail": "no date on the order page", "urgent": False, "sev": "minor"},
        {"issue": "received a damaged item", "detail": "box crushed, photos attached, order {ref}", "urgent": False, "sev": "moderate"},
    ],
    "technical": [
        {"issue": "get a 500 error on every API call since this morning", "detail": "production integration down since {time}", "urgent": True, "sev": "critical"},
        {"issue": "noticed a typo in the dashboard footer", "detail": "cosmetic, no functional impact", "urgent": False, "sev": "minor"},
        {"issue": "find that exports time out for large accounts", "detail": "fails above about {n} thousand rows", "urgent": False, "sev": "moderate"},
        {"issue": "cannot log in on the mobile app after the update", "detail": "web works, app version {ref} crashes at login", "urgent": True, "sev": "major"},
    ],
    "sales": [
        {"issue": "would like a quote for {n} additional seats", "detail": "renewal due on {day}", "urgent": False, "sev": "minor"},
        {"issue": "want to upgrade before a launch this week", "detail": "launch on {day}, needs the higher tier by then", "urgent": True, "sev": "moderate"},
        {"issue": "would like to know whether a discount applies to annual billing", "detail": "pricing question, no deadline", "urgent": False, "sev": "minor"},
        {"issue": "have been offered a lower price by a competitor and may cancel", "detail": "contract value {amt}, decision by {day}", "urgent": True, "sev": "major"},
    ],
}

OPENERS = ["Hi,", "Hello support,", "To whom it may concern,", "Hey team,", "Good morning,", "Urgent:", "Quick question:"]
CLOSERS = ["Thanks.", "Please advise.", "Regards, {name}", "Let me know what you need from me.", "Best, {name}", ""]
NAMES = ["Dana", "Priya", "Marco", "Lea", "Tomás", "Hannah", "Yusuf", "Ines", "Noah", "Mei"]
FILLER_COMMON = [
    "I have been a customer for a few years and this is the first time I write to you.",
    "I checked the help centre first but did not find anything that matched.",
    "For context, we use the service for our small team of about a dozen people.",
    "I am including the details below so you do not have to ask for them.",
    "If it helps, I can send screenshots or hop on a call.",
    "I am writing from the account registered under the same email as this message.",
]
FILLER_BY_TEAM = {
    "billing": ["Our finance team will need a written confirmation once this is sorted.", "The card on file has not changed."],
    "security": ["I have not shared my credentials with anyone.", "I changed my password just in case, but I am not sure that is enough."],
    "shipping": ["The order confirmation and tracking link are in my account.", "Nobody at the address signed for anything."],
    "technical": ["I also tried from another browser and a different network, same result.", "Nothing changed in our configuration recently."],
    "sales": ["We would move forward quickly once we have the numbers.", "The decision sits with our procurement team."],
}


def _filler(team: str, r: random.Random, k: int) -> list[str]:
    pool = FILLER_COMMON + FILLER_BY_TEAM[team]
    return r.sample(pool, k)


THREAD_TURNS = [
    "Agent: Thank you for reaching out. Could you confirm the account email and the approximate time this started?",
    "Customer: Sure, it is the email on this account, and it started {time} as far as I can tell.",
    "Agent: Thanks. I am checking our systems now; this may take a moment.",
    "Customer: Any update? I have a meeting in an hour where this will come up.",
    "Agent: Still looking. Can you tell me whether anything changed on your side recently?",
    "Customer: Nothing on our side. We did not change any settings.",
    "Agent: Understood. I am escalating this internally and will update you here.",
    "Customer: Thank you. Please also note the reference numbers I gave above.",
]


def _fill(tpl: str, rng: random.Random) -> str:
    return tpl.format(amt=f"${rng.choice([49, 120, 480, 1250, 3900]):,}", day=rng.choice(["Monday", "Tuesday", "the 3rd", "the 14th", "last Friday"]),
                      n=rng.choice([2, 3, 5, 7, 10, 25]), ref=f"{rng.choice('ABCDEFG')}{rng.randint(10000, 99999)}",
                      time=rng.choice(["around 9am", "last night", "this morning", "at 14:30", "yesterday afternoon"]),
                      name=rng.choice(NAMES))


def questions_for(item: dict) -> dict[str, dict]:
    """The three typed questions every item carries, in the server's request shape."""
    return {
        "route": {"type": "choice", "instructions": "Which team should handle this request?", "criteria": dict(TEAMS)},
        "urgent": {"type": "noul", "instructions": "Does the writer need a response within the hour?"},
        "severity": {"type": "score", "instructions": "How severe is the problem described?", "criteria": list(SEVERITY)},
    }


def generate(n: int, size: str, seed: int = 0) -> list[dict]:
    """n items of one size, deterministic in (n, size, seed), balanced over teams and scenarios."""
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}")
    rng = random.Random(f"decision|{size}|{seed}")
    items = []
    teams = list(TEAMS)
    for i in range(n):
        team = teams[i % len(teams)]
        sc = SCENARIOS[team][(i // len(teams)) % len(SCENARIOS[team])]
        r = random.Random(f"{seed}|{size}|{i}")
        issue, detail = _fill(sc["issue"], r), _fill(sc["detail"], r)
        opener, closer = r.choice(OPENERS), _fill(r.choice(CLOSERS), r)
        if size == "short":
            body = f"{opener} I {issue}. {detail.capitalize()}. {_filler(team, r, 1)[0]} {closer}".strip()
        else:
            fill = _filler(team, r, 6)
            turns = [_fill(t, r) for t in THREAD_TURNS]
            # the deciding facts sit in the middle of the thread, not at the top
            body = "\n".join([f"Ticket #{r.randint(100000, 999999)} - opened by {r.choice(NAMES)}", "",
                              f"Customer: {opener} {fill[0]} {fill[1]} I am writing because I {issue}.", *turns[:3],
                              f"Customer: {detail.capitalize()}. {fill[2]} {fill[3]}", *turns[3:6], f"Customer: {fill[4]} {fill[5]}",
                              *turns[6:], f"Customer: {closer}"]).strip()
        items.append({"id": f"dec-{size}-{i:04d}", "size": size, "state": body, "questions": questions_for({}),
                      "gold": {"route": team, "urgent": bool(sc["urgent"]), "severity": sc["sev"]}})
    rng.shuffle(items)
    return items


def dataset_sha256(items: list[dict]) -> str:
    return hashlib.sha256("".join(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n" for x in items).encode()).hexdigest()


def request_body(item: dict, n_questions: int) -> dict:
    """The POST /v1/systemone body for an item with its first n_questions questions (1 = route only)."""
    names = list(item["questions"])[:max(1, min(n_questions, len(item["questions"])))]
    return {"state": item["state"], "questions": {k: item["questions"][k] for k in names}}


# ----------------------------------------------------------------------------- scoring
def parse_answers(resp: dict) -> dict[str, dict]:
    """Normalise the server's answer block to {name: {type, value, confidence, probabilities}}."""
    out = {}
    for name, a in (resp.get("answers") or {}).items():
        t = a.get("type")
        if t == "noul":
            out[name] = {"type": t, "value": float(a.get("noul")), "confidence": None, "probabilities": None}
        elif t == "choice":
            out[name] = {"type": t, "value": a.get("choice"), "confidence": a.get("confidence"), "probabilities": a.get("probabilities")}
        elif t == "score":
            out[name] = {"type": t, "value": a.get("score"), "confidence": a.get("confidence"), "probabilities": a.get("probabilities"),
                         "level": a.get("level") or a.get("label")}
    return out


def _level_index(level_name: str | None, score: float | None) -> int | None:
    if level_name in SEVERITY:
        return SEVERITY.index(level_name)
    if score is None:
        return None
    return max(0, min(len(SEVERITY) - 1, int(round(float(score)))))


def score_against_gold(answers: dict[str, dict], gold: dict) -> dict[str, bool | None]:
    """Right or wrong per question: route (exact), urgent (noul >= 0.5), severity (nearest level)."""
    out: dict[str, bool | None] = {}
    r = answers.get("route")
    out["route"] = (r["value"] == gold["route"]) if r else None
    u = answers.get("urgent")
    out["urgent"] = ((u["value"] >= 0.5) == bool(gold["urgent"])) if u else None
    s = answers.get("severity")
    out["severity"] = (_level_index(s.get("level"), s.get("value")) == SEVERITY.index(gold["severity"])) if s else None
    return out


def score_cell(records: list[dict], items_by_id: dict[str, dict], reference: dict[str, dict] | None = None) -> dict[str, Any]:
    """records: kept outputs of a cell's accuracy level ({item_id, output: <server response>}). reference: the
    GPU cell's parsed answers by item id, for agreement. Returns accuracy per question, overall, mean
    confidence, and against the reference the agreement rate and mean absolute probability gap."""
    right: dict[str, list[bool]] = {q: [] for q in ("route", "urgent", "severity")}
    conf, agree, gaps, n = [], [], [], 0
    parsed_by_id: dict[str, dict] = {}
    for rec in records:
        item = items_by_id.get(rec.get("item_id") or "")
        resp = rec.get("output")
        if not item or not isinstance(resp, dict) or not resp.get("answers"):
            continue
        ans = parse_answers(resp)
        parsed_by_id[item["id"]] = ans
        n += 1
        for q, ok in score_against_gold(ans, item["gold"]).items():
            if ok is not None:
                right[q].append(ok)
        for a in ans.values():
            if a.get("confidence") is not None:
                conf.append(float(a["confidence"]))
        if reference and item["id"] in reference:
            ref = reference[item["id"]]
            for q, a in ans.items():
                b = ref.get(q)
                if not b:
                    continue
                if a["type"] == "noul":
                    agree.append((a["value"] >= 0.5) == (b["value"] >= 0.5))
                    gaps.append(abs(a["value"] - b["value"]))
                elif a["type"] == "choice":
                    agree.append(a["value"] == b["value"])
                    if a.get("probabilities") and b.get("probabilities"):
                        gaps.append(max(abs(float(a["probabilities"].get(k, 0)) - float(b["probabilities"].get(k, 0))) for k in a["probabilities"]))
                else:
                    agree.append(_level_index(a.get("level"), a.get("value")) == _level_index(b.get("level"), b.get("value")))
                    if a.get("value") is not None and b.get("value") is not None:
                        gaps.append(abs(float(a["value"]) - float(b["value"])) / max(1, len(SEVERITY) - 1))
    acc = {q: (sum(v) / len(v) if v else None) for q, v in right.items()}
    allv = [x for v in right.values() for x in v]
    out: dict[str, Any] = {"n": n, "accuracy": acc, "accuracy_overall": (sum(allv) / len(allv) if allv else None),
                           "confidence_mean": (sum(conf) / len(conf) if conf else None), "answers": parsed_by_id}
    if reference:
        out["vs_reference_cell"] = {"agreement": (sum(agree) / len(agree) if agree else None),
                                    "prob_gap_mean": (sum(gaps) / len(gaps) if gaps else None),
                                    "prob_gap_max": (max(gaps) if gaps else None), "n": len(agree)}
    return out


AGREEMENT_FLOOR = 0.95       # same code both sides: the project reports probabilities within 0.05 of each other


def flag(score: dict, *, precision: str) -> str | None:
    """A warning (never an exclusion) when a cell disagrees with the GPU reference more than the same-code floor."""
    vs = score.get("vs_reference_cell") or {}
    if vs.get("agreement") is not None and vs["agreement"] < AGREEMENT_FLOOR:
        return f"agreement with the GPU's decisions {vs['agreement']:.3f} < {AGREEMENT_FLOOR}"
    return None
