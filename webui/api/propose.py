"""Instance proposals with reasons (wizard step 5).

Given the models in the run and what the user has chosen so far, propose instance types for
the GPU side, the CPU side, or both, and say why each one fits: does the model fit in memory
with room for the KV cache, is the hourly price in the fair band against the other side, is it
a single GPU (the comparison is per accelerator), how many vCPUs feed AMX, is it the current
generation, and what the recent runs saw for capacity in this availability zone."""

from __future__ import annotations

import math

from bench import pairing

BYTES_PER_PARAM = {"bf16": 2.0, "bf16-ov": 2.0, "fp16": 2.0, "w8a8": 1.0, "int8": 1.0, "int4": 0.5, "fp32": 4.0}
GPU_HEADROOM = 0.75          # weights may use at most this share of GPU memory (rest: KV cache, activations, runtime)
CPU_WEIGHTS_SHARE = 0.4      # weights may use at most this share of host memory (rest: KV cache budget, OS, Docker)
MIN_CPU_VCPUS = 8


def _usd(v: float) -> str:
    return "$" + f"{v:.2f}" + "/h"


def _weights_gib(models: list[dict], precision: str = "bf16") -> float:
    bpp = BYTES_PER_PARAM.get(precision, 2.0)
    return max((m["params_b"] * 1e9 * bpp / 2**30 for m in models), default=0.0)


def _largest(models: list[dict]) -> dict | None:
    return max(models, key=lambda m: m["params_b"]) if models else None


def _fit(ok: bool, warn: bool = False) -> str:
    return "recommended" if ok and not warn else ("possible" if ok else "not_advisable")


def gpu_candidates(models: list[dict], precision: str, cpu_type: str | None, capacity_notes: dict[str, int],
                   catalog: dict | None = None, n: int = 6) -> list[dict]:
    cat = catalog or pairing.load_catalog()
    need = _weights_gib(models, precision)
    big = _largest(models)
    cpu = pairing.spec(cpu_type, cat) if cpu_type else None
    out = []
    for t in pairing.types_for("gpu", cat):
        s = cat["instances"][t]
        reasons, ok, warn = [], True, False
        mem = s.get("gpu_memory_gib") or 0
        if big:
            if need <= mem * GPU_HEADROOM:
                reasons.append({"kind": "memory", "ok": True,
                                "text": f"{big['name']} needs about {need:.1f} GiB of GPU memory in {precision.upper()}; "
                                        f"the {s['gpu_model']} has {mem:.0f} GiB, leaving room for the KV cache."})
            else:
                ok = False
                reasons.append({"kind": "memory", "ok": False,
                                "text": f"{big['name']} needs about {need:.1f} GiB in {precision.upper()}, more than "
                                        f"{GPU_HEADROOM:.0%} of the {s['gpu_model']}'s {mem:.0f} GiB; it would not load reliably."})
        if s["gpus"] == 1:
            reasons.append({"kind": "gpus", "ok": True, "text": "One GPU: the comparison is per accelerator, so a single-GPU "
                                                                  "instance keeps the GPU side one unit."})
        else:
            warn = True
            reasons.append({"kind": "gpus", "ok": False, "text": f"{s['gpus']} GPUs: a multi-GPU instance measures the box, "
                                                                   "not the accelerator; prefer a single-GPU size."})
        if cpu:
            j = pairing.judge(cpu_type, t, cat)
            mine, theirs = _usd(s["usd_per_hour"]), _usd(cpu["usd_per_hour"])
            verdict = ("a fair pairing (within 0.5x-2x)." if j["fair"]
                       else "outside the fair band, so the comparison would measure size, not architecture.")
            reasons.append({"kind": "price", "ok": bool(j["fair"]),
                            "text": f"{mine} is {j['ratio']:.2f}x the {cpu_type} price ({theirs}): {verdict}"})
            if not j["fair"]:
                warn = True
        if not s.get("current_generation", True):
            warn = True
            reasons.append({"kind": "generation", "ok": False,
                            "text": f"Previous generation ({s['gpu_model']}); a current-generation L4 (g6) is the more "
                                    "representative buy, but g5 is often easier to get."})
        else:
            reasons.append({"kind": "generation", "ok": True, "text": f"Current generation ({s['gpu_model']})."})
        if capacity_notes.get(t):
            reasons.append({"kind": "capacity", "ok": False,
                            "text": f"Recent runs were refused capacity for {t} in this zone {capacity_notes[t]} time(s); "
                                    "keep an alternate."})
        score = (0 if ok else 100) + (10 if warn else 0) + (abs(math.log(pairing.judge(cpu_type, t, cat)["ratio"])) if cpu else 0) \
            + (0 if s["gpus"] == 1 else 5) + s["usd_per_hour"] / 10
        out.append({"instance_type": t, "spec": s, "fit": _fit(ok, warn), "reasons": reasons, "_score": score})
    out.sort(key=lambda x: x["_score"])
    for x in out:
        del x["_score"]
    return out[:n]


def cpu_candidates(models: list[dict], precision: str, gpu_type: str | None, capacity_notes: dict[str, int],
                   catalog: dict | None = None, n: int = 6) -> list[dict]:
    cat = catalog or pairing.load_catalog()
    need = _weights_gib(models, precision)
    big = _largest(models)
    gpu = pairing.spec(gpu_type, cat) if gpu_type else None
    out = []
    for t in pairing.types_for("cpu", cat):
        s = cat["instances"][t]
        reasons, ok, warn = [], True, False
        mem = s["memory_gib"]
        if big:
            if need <= mem * CPU_WEIGHTS_SHARE:
                reasons.append({"kind": "memory", "ok": True,
                                "text": f"{big['name']} needs about {need:.1f} GiB for weights in {precision.upper()}; "
                                        f"{mem:.0f} GiB of memory leaves room for the KV cache and the OS."})
            else:
                ok = False
                reasons.append({"kind": "memory", "ok": False,
                                "text": f"{big['name']} needs about {need:.1f} GiB in {precision.upper()}; with "
                                        f"{mem:.0f} GiB there is too little left for the KV cache."})
        reasons.append({"kind": "amx", "ok": True, "text": f"{s['family']} is an Intel Xeon generation with AMX, the "
                                                           "matrix-math units this benchmark is about."})
        if s["vcpus"] < MIN_CPU_VCPUS:
            warn = True
            reasons.append({"kind": "vcpus", "ok": False, "text": f"{s['vcpus']} vCPUs: too few cores to keep a 3-4B model "
                                                                    "responsive under concurrent load."})
        else:
            reasons.append({"kind": "vcpus", "ok": True, "text": f"{s['vcpus']} vCPUs ({s['vcpus'] // 2} physical cores "
                                                                   "with AMX) share the load; throughput scales with cores."})
        if gpu:
            j = pairing.judge(t, gpu_type, cat)
            mine, theirs = _usd(s["usd_per_hour"]), _usd(gpu["usd_per_hour"])
            verdict = ("a fair pairing (within 0.5x-2x)." if j["fair"]
                       else "outside the fair band, so the comparison would measure size, not architecture.")
            reasons.append({"kind": "price", "ok": bool(j["fair"]),
                            "text": f"{mine} against {gpu_type} at {theirs} (ratio {j['ratio']:.2f}x): {verdict}"})
            if not j["fair"]:
                warn = True
        if s["family"].startswith(("m7", "c7", "r7")):
            warn = True
            reasons.append({"kind": "generation", "ok": False,
                            "text": "Sapphire Rapids (previous generation). Granite Rapids (m8i/c8i/r8i) has faster AMX "
                                    "and memory; use 7i only as a fallback."})
        else:
            reasons.append({"kind": "generation", "ok": True, "text": "Granite Rapids (current generation AMX)."})
        if s["family"].startswith("r"):
            reasons.append({"kind": "memory", "ok": True, "text": "Memory-optimized: more memory per core than needed for "
                                                                    "a 4B model; m-family gives the same cores for less."})
            warn = True
        if capacity_notes.get(t):
            reasons.append({"kind": "capacity", "ok": False,
                            "text": f"Recent runs were refused capacity for {t} in this zone {capacity_notes[t]} time(s)."})
        score = (0 if ok else 100) + (10 if warn else 0) + (abs(math.log(pairing.judge(t, gpu_type, cat)["ratio"])) if gpu else 0) \
            + s["usd_per_hour"] / 10
        out.append({"instance_type": t, "spec": s, "fit": _fit(ok, warn), "reasons": reasons, "_score": score})
    out.sort(key=lambda x: x["_score"])
    for x in out:
        del x["_score"]
    return out[:n]


# ----------------------------------------------------------------------------- auto-suggest

# What the model's architecture means for CPU sizing, in one sentence each, and the smallest
# AMX core count that keeps that kind of model responsive under concurrent load.
ARCH = {
    "decoder": ("a chat (decoder) model: every generated token re-reads all the weights, so CPU speed is set by "
                "memory bandwidth and the number of AMX cores working in parallel", 16),
    "encoder-decoder": ("an encoder-decoder model: one pass over the input, then token-by-token generation; "
                        "AMX cores and memory bandwidth both matter", 16),
    "encoder-classifier": ("a classifier: one pass per input and no generation, so it batches well and a modest "
                           "number of AMX cores goes a long way", 8),
    "token-classifier": ("a detector that labels each word in one pass; batches well, a modest number of AMX "
                         "cores goes a long way", 8),
}
ROLES = {
    "match": ("Best match", "closest in hourly price to the GPU you chose: the fairest like-for-like comparison"),
    "headroom": ("More CPU headroom", "the largest size still inside the fair band (up to 2x the GPU price): shows "
                                      "how far the CPU side can go"),
    "economy": ("Cheaper fair option", "the smallest size still inside the fair band (at least half the GPU price): "
                                       "the lowest-cost CPU that is still a fair comparison"),
}


def _cpu_fit(models: list[dict], precision: str, t: str, s: dict, gpu_type: str, cat: dict,
             capacity_notes: dict[str, int]) -> tuple[bool, bool, list[dict]]:
    """ok (fits the models), warn (fair but with a caveat), and the reasons for one CPU size."""
    need = _weights_gib(models, precision)
    big = _largest(models)
    min_vcpus = max((ARCH.get(m.get("type"), ("", MIN_CPU_VCPUS))[1] for m in models), default=MIN_CPU_VCPUS)
    reasons, ok, warn = [], True, False
    mem = s["memory_gib"]
    if big:
        if need <= mem * CPU_WEIGHTS_SHARE:
            reasons.append({"kind": "memory", "ok": True,
                            "text": f"{big['name']} needs about {need:.1f} GiB for weights in {precision.upper()}; "
                                    f"{mem:.0f} GiB leaves room for the KV cache and the OS."})
        else:
            ok = False
            reasons.append({"kind": "memory", "ok": False,
                            "text": f"{big['name']} needs about {need:.1f} GiB in {precision.upper()}; with {mem:.0f} GiB "
                                    "there is too little left for the KV cache."})
    if s["vcpus"] < min_vcpus:
        ok = False
        reasons.append({"kind": "vcpus", "ok": False,
                        "text": f"{s['vcpus']} vCPUs ({s['vcpus'] // 2} AMX cores): below the {min_vcpus} vCPUs this kind of "
                                "model needs to stay responsive under concurrent load."})
    else:
        reasons.append({"kind": "vcpus", "ok": True,
                        "text": f"{s['vcpus']} vCPUs ({s['vcpus'] // 2} physical cores with AMX) share the load."})
    j = pairing.judge(t, gpu_type, cat)
    g = cat["instances"][gpu_type]
    if j["fair"]:
        reasons.append({"kind": "price", "ok": True,
                        "text": f"{_usd(s['usd_per_hour'])} against {gpu_type} at {_usd(g['usd_per_hour'])}: the GPU costs "
                                f"{j['ratio']:.2f}x as much, inside the fair band (0.5x-2x)."})
    else:
        warn = True
        reasons.append({"kind": "price", "ok": False,
                        "text": f"{_usd(s['usd_per_hour'])} against {gpu_type} at {_usd(g['usd_per_hour'])}: ratio "
                                f"{j['ratio']:.2f}x, outside the fair band, so the test would measure size, not architecture."})
    if s["family"].startswith(("m7", "c7", "r7")):
        warn = True
        reasons.append({"kind": "generation", "ok": False,
                        "text": "Sapphire Rapids (previous generation): slower AMX and memory than Granite Rapids (m8i/c8i)."})
    else:
        reasons.append({"kind": "generation", "ok": True, "text": "Granite Rapids: the current AMX generation."})
    if s["family"].startswith("r"):
        warn = True
        reasons.append({"kind": "memory", "ok": False,
                        "text": "Memory-optimized: you pay for memory this model does not need; m8i gives the same cores for less."})
    elif big and s["family"][0] == FAMILY_PREF.get(big.get("type"), "m") and s["family"][1] == "8":
        reasons.append({"kind": "family", "ok": True, "text": FAMILY_WHY[s["family"][0]]})
    if capacity_notes.get(t):
        reasons.append({"kind": "capacity", "ok": False,
                        "text": f"Recent runs were refused capacity for {t} in this zone {capacity_notes[t]} time(s)."})
    return ok, warn, reasons


GPU_HOST_MEM_FACTOR = 2.0    # vLLM stages the weights in host memory before moving them to the GPU ...
GPU_HOST_MEM_BASE_GIB = 8.0  # ... plus the OS, Docker, the tokenizer and the server itself
GPU_MIN_VCPUS = {"decoder": 8, "encoder-decoder": 8, "encoder-classifier": 4, "token-classifier": 4}
FAMILY_PREF = {"decoder": "m", "encoder-decoder": "m", "encoder-classifier": "c", "token-classifier": "c"}
FAMILY_WHY = {
    "m": "General-purpose m-family: 4 GiB of memory per vCPU leaves room for the KV cache, which grows with every "
         "concurrent chat request.",
    "c": "Compute-optimized c-family: 2 GiB per vCPU is plenty for a model this small, and it is the cheapest "
         "Granite Rapids core, so the same spend buys more AMX cores.",
}
FAMILY_PENALTY = 0.12        # in log-price-ratio units: the preferred family wins unless it is clearly further from parity


def _gpu_fit(models: list[dict], precision: str, t: str, s: dict, capacity_notes: dict[str, int]) -> tuple[bool, bool, list[dict]]:
    """ok (the model runs well on it), warn (fits but a worse buy), reasons for one GPU size."""
    need = _weights_gib(models, precision)
    big = _largest(models)
    kind = (big or {}).get("type")
    min_vcpus = GPU_MIN_VCPUS.get(kind, 8)
    reasons, ok, warn = [], True, False
    mem = s.get("gpu_memory_gib") or 0
    if big:
        if need <= mem * GPU_HEADROOM:
            reasons.append({"kind": "memory", "ok": True,
                            "text": f"{big['name']} needs about {need:.1f} GiB of GPU memory in {precision.upper()}; the "
                                    f"{s['gpu_model']} has {mem:.0f} GiB, leaving room for the KV cache."})
        else:
            ok = False
            reasons.append({"kind": "memory", "ok": False,
                            "text": f"{big['name']} needs about {need:.1f} GiB in {precision.upper()}, more than {GPU_HEADROOM:.0%} "
                                    f"of the {s['gpu_model']}'s {mem:.0f} GiB; it would not load reliably."})
        host_need = need * GPU_HOST_MEM_FACTOR + GPU_HOST_MEM_BASE_GIB
        if s["memory_gib"] >= host_need and s["vcpus"] >= min_vcpus:
            reasons.append({"kind": "host", "ok": True,
                            "text": f"{s['vcpus']} vCPUs and {s['memory_gib']:.0f} GiB of host memory: enough to stage the weights "
                                    f"(about {host_need:.0f} GiB needed) and run the tokenizer and scheduler."})
        else:
            ok = False
            what = (f"only {s['memory_gib']:.0f} GiB of host memory (about {host_need:.0f} GiB needed to stage the weights)"
                    if s["memory_gib"] < host_need else f"only {s['vcpus']} vCPUs ({min_vcpus} needed for tokenizing and scheduling)")
            reasons.append({"kind": "host", "ok": False, "text": f"Same GPU, but {what}: the GPU would wait on the host."})
    if s["gpus"] == 1:
        reasons.append({"kind": "gpus", "ok": True, "text": "One GPU: the comparison is per accelerator."})
    else:
        ok = False
        reasons.append({"kind": "gpus", "ok": False, "text": f"{s['gpus']} GPUs: measures the box, not the accelerator."})
    if s.get("current_generation", True):
        reasons.append({"kind": "generation", "ok": True, "text": f"Current generation ({s['gpu_model']})."})
    else:
        warn = True
        reasons.append({"kind": "generation", "ok": False,
                        "text": f"Previous generation ({s['gpu_model']}); fine as a capacity fallback, the L4 (g6) is the representative buy."})
    if capacity_notes.get(t):
        reasons.append({"kind": "capacity", "ok": False,
                        "text": f"Recent runs were refused capacity for {t} in this zone {capacity_notes[t]} time(s); keep an alternate."})
    return ok, warn, reasons


def suggest_gpu(models: list[dict], precision: str, capacity_notes: dict[str, int] | None = None,
                catalog: dict | None = None) -> dict:
    """The GPU size for the model: the cheapest single-GPU, current-generation instance whose GPU
    memory holds the weights with headroom AND whose host side (memory, vCPUs) can feed it. A 0.4B
    classifier gets g6.xlarge; a 3–4B chat model needs g6.2xlarge because vLLM stages ~8 GiB of
    weights in host memory and g6.xlarge has 16 GiB. Alternates (for capacity fallback) are the
    next fitting sizes closest in price, previous generation included."""
    cat = catalog or pairing.load_catalog()
    notes = capacity_notes or {}
    big = _largest(models)
    need = _weights_gib(models, precision)
    basis = []
    if big:
        kind = big.get("type")
        basis.append(f"{big['name']} is {ARCH.get(kind, ('a model', 0))[0]}; in {precision.upper()} its weights take about "
                     f"{need:.1f} GiB of GPU memory and about {need * GPU_HOST_MEM_FACTOR + GPU_HOST_MEM_BASE_GIB:.0f} GiB of host "
                     f"memory while loading, and the host needs at least {GPU_MIN_VCPUS.get(kind, 8)} vCPUs.")
    fitting, rejected = [], []
    for t in pairing.types_for("gpu", cat):
        s = cat["instances"][t]
        ok, warn, reasons = _gpu_fit(models, precision, t, s, notes)
        c = {"instance_type": t, "spec": s, "reasons": reasons, "fit": _fit(ok, warn), "ratio": None, "fair": True}
        (fitting if ok else rejected).append(c)
    current = [c for c in fitting if c["spec"].get("current_generation", True)]
    pool = current or fitting
    best = min(pool, key=lambda c: c["spec"]["usd_per_hour"]) if pool else None
    alternatives = []
    if best:
        others = [c for c in fitting if c["instance_type"] != best["instance_type"]]
        others.sort(key=lambda c: abs(math.log(c["spec"]["usd_per_hour"] / best["spec"]["usd_per_hour"])))
        alternatives = others[:2]
        best = {**best, "role": "match", "role_label": "Best fit", "role_help": "The cheapest single current-generation GPU that holds the model and whose host side can feed it."}
        for a in alternatives:
            a.update(role="headroom", role_label="Capacity fallback",
                     role_help="Same class of GPU, closest in price: tried automatically when the best fit is refused for capacity.")
    return {"best": best, "alternatives": alternatives, "others": [c for c in fitting if c not in alternatives and (not best or c["instance_type"] != best["instance_type"])][:6],
            "rejected": rejected[:6], "basis": basis, "weights_gib": round(need, 2),
            "error": None if best else "No single-GPU size in the catalog holds this model."}


def initial_selection(models: list[dict], precision: str, capacity_notes: dict[str, int] | None = None,
                      catalog: dict | None = None) -> dict:
    """What the hardware step pre-selects for the chosen model: the GPU best fit, then the CPU best
    match against that GPU, with the reasons for both and a one-line summary."""
    gpu = suggest_gpu(models, precision, capacity_notes, catalog)
    cpu = suggest_cpu(models, precision, gpu["best"]["instance_type"], capacity_notes, catalog) if gpu["best"] else None
    big = _largest(models)
    summary = None
    if gpu["best"] and cpu and cpu["best"]:
        g, c = gpu["best"], cpu["best"]
        summary = (f"For {big['name'] if big else 'this model'}: {g['instance_type']} ({g['spec']['gpus']}x {g['spec']['gpu_model']}, "
                   f"{_usd(g['spec']['usd_per_hour'])}) against {c['instance_type']} ({c['spec']['vcpus']} vCPUs with AMX, "
                   f"{_usd(c['spec']['usd_per_hour'])}); the GPU costs {c['ratio']:.2f}x as much, a fair pairing.")
    return {"gpu": gpu, "cpu": cpu, "summary": summary,
            "gpu_alternates": [a["instance_type"] for a in gpu["alternatives"]]}


def suggest_cpu(models: list[dict], precision: str, gpu_type: str, capacity_notes: dict[str, int] | None = None,
                catalog: dict | None = None) -> dict:
    """CPU sizes for the chosen GPU and models: one best match plus at most two alternatives, each
    with a role and reasons, and the rest of the fair sizes folded away. Price parity with the GPU
    decides the best match (that is what makes the comparison fair); memory and AMX core count
    decide what is eligible at all; the model's architecture sets the core-count floor."""
    cat = catalog or pairing.load_catalog()
    notes = capacity_notes or {}
    g = cat["instances"].get(gpu_type)
    if not g or not g.get("usd_per_hour"):
        return {"gpu_type": gpu_type, "best": None, "alternatives": [], "others": [], "basis": [],
                "error": f"{gpu_type} is not in the instance catalog"}
    big = _largest(models)
    basis = []
    if big:
        arch = ARCH.get(big.get("type"), ("a model", MIN_CPU_VCPUS))
        basis.append(f"{big['name']} is {arch[0]}; it needs about {_weights_gib(models, precision):.1f} GiB of memory for "
                     f"its weights in {precision.upper()} and at least {arch[1]} vCPUs.")
    basis.append(f"{gpu_type}: {g['gpus']}x {g['gpu_model']} ({g['gpu_memory_gib']:.0f} GiB GPU memory), {g['vcpus']} vCPUs, "
                 f"{_usd(g['usd_per_hour'])}. A fair CPU partner costs between {_usd(g['usd_per_hour'] / 2)} and "
                 f"{_usd(g['usd_per_hour'] * 2)}; the closer to {_usd(g['usd_per_hour'])}, the fairer.")
    if len(models) > 1:
        basis.append(f"Sized for the largest of the {len(models)} chosen models; the smaller ones fit wherever it fits.")
    eligible, rejected = [], []
    for t in pairing.types_for("cpu", cat):
        s = cat["instances"][t]
        ok, warn, reasons = _cpu_fit(models, precision, t, s, gpu_type, cat, notes)
        j = pairing.judge(t, gpu_type, cat)
        c = {"instance_type": t, "spec": s, "reasons": reasons, "fit": _fit(ok, warn), "ratio": j["ratio"],
             "fair": bool(j["fair"]), "_dist": abs(math.log(j["ratio"]))}
        (eligible if ok and j["fair"] else rejected).append(c)
    # Current generation first, then price parity; r-family only when nothing else fits.
    current = [c for c in eligible if not c["spec"]["family"].startswith(("m7", "c7", "r7"))]
    pool = [c for c in (current or eligible) if not c["spec"]["family"].startswith("r")] or current or eligible
    chosen: list[dict] = []
    if pool:
        # Price parity decides, with a thumb on the scale for the family that suits the model's
        # architecture: c-family (cheapest AMX cores) for classifiers, m-family (memory for the
        # KV cache) for chat models. Both are Granite Rapids; the reasons on the card say why.
        pref = FAMILY_PREF.get((big or {}).get("type"), "m")
        best = min(pool, key=lambda c: (c["_dist"] + (0 if c["spec"]["family"][0] == pref else FAMILY_PENALTY), c["spec"]["usd_per_hour"]))
        chosen.append({**best, "role": "match"})
        bigger = [c for c in pool if c["spec"]["usd_per_hour"] > best["spec"]["usd_per_hour"] * 1.15]
        smaller = [c for c in pool if c["spec"]["usd_per_hour"] < best["spec"]["usd_per_hour"] / 1.15]
        if bigger:
            chosen.append({**max(bigger, key=lambda c: c["spec"]["usd_per_hour"]), "role": "headroom"})
        if smaller:
            chosen.append({**min(smaller, key=lambda c: c["spec"]["usd_per_hour"]), "role": "economy"})
    picked = {c["instance_type"] for c in chosen}
    others = sorted((c for c in eligible if c["instance_type"] not in picked), key=lambda c: c["_dist"])
    # Near misses first: sizes in the price band that fail on memory or cores, then the closest in price.
    rejected.sort(key=lambda c: (not c["fair"], c["_dist"]))
    for c in chosen + others + rejected:
        c.pop("_dist", None)
    for c in chosen:
        c["role_label"], c["role_help"] = ROLES[c["role"]]
    return {"gpu_type": gpu_type, "gpu_spec": g, "best": chosen[0] if chosen else None, "alternatives": chosen[1:],
            "others": others, "rejected": rejected[:8], "basis": basis,
            "weights_gib": round(_weights_gib(models, precision), 2),
            "error": None if chosen else "No AMX CPU size fits both the model and the fair price band for this GPU; "
                                        "pick another GPU size."}


def propose(models: list[dict], precision: str, gpu_type: str | None, cpu_type: str | None,
            capacity_notes: dict[str, int] | None = None, catalog: dict | None = None) -> dict:
    """Both lists; each side is proposed against the other side's current choice when there is one."""
    notes = capacity_notes or {}
    return {"gpu": gpu_candidates(models, precision, cpu_type, notes, catalog),
            "cpu": cpu_candidates(models, precision, gpu_type, notes, catalog),
            "pairing": pairing.judge(cpu_type, gpu_type, catalog) if cpu_type and gpu_type else None,
            "weights_gib": round(_weights_gib(models, precision), 2)}
