"""Instance proposals with reasons (wizard step 5).

Given the models in the run and what the user has chosen so far, propose instance types for
the GPU side, the CPU side, or both, and say why each one fits: does the model fit in memory
with room for the KV cache, is the hourly price in the fair band against the other side, is it
a single GPU (the comparison is per accelerator), how many vCPUs feed AMX, is it the current
generation, and what the recent runs saw for capacity in this availability zone."""

from __future__ import annotations

import math

from bench import pairing

BYTES_PER_PARAM = {"bf16": 2.0, "fp16": 2.0, "w8a8": 1.0, "int8": 1.0, "int4": 0.5, "fp32": 4.0}
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


def propose(models: list[dict], precision: str, gpu_type: str | None, cpu_type: str | None,
            capacity_notes: dict[str, int] | None = None, catalog: dict | None = None) -> dict:
    """Both lists; each side is proposed against the other side's current choice when there is one."""
    notes = capacity_notes or {}
    return {"gpu": gpu_candidates(models, precision, cpu_type, notes, catalog),
            "cpu": cpu_candidates(models, precision, gpu_type, notes, catalog),
            "pairing": pairing.judge(cpu_type, gpu_type, catalog) if cpu_type and gpu_type else None,
            "weights_gib": round(_weights_gib(models, precision), 2)}
