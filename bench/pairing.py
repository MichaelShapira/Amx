"""Instance catalog and fair CPU/GPU pairing.

The catalog (`config/instance_catalog.json`, refreshed by tools/refresh_instance_catalog.py)
holds specs and On-Demand prices for the AMX CPU families and the small NVIDIA GPU families.

A pairing is **fair** when the two instances cost about the same per hour: the GPU
instance's On-Demand price is within FAIR_RATIO of the CPU instance's. The benchmark asks
"for capacity I can buy at a similar spend, what does each side deliver?"; a g6.48xlarge
(8 GPUs, $13/h) against an m8i.xlarge ($0.21/h) only measures size. Among fair candidates,
single-GPU instances come first (the comparison is per accelerator, not per chassis), then
the closest price. The verdict logic is unchanged: capacity decides, cost is informational;
this module only keeps the two sides comparable.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "config/instance_catalog.json"
FAIR_RATIO = (0.5, 2.0)         # gpu_usd / cpu_usd


@lru_cache(maxsize=1)
def load_catalog(path: Path = CATALOG) -> dict:
    return json.loads(path.read_text())


def spec(instance_type: str, catalog: dict | None = None) -> dict | None:
    return (catalog or load_catalog())["instances"].get(instance_type)


def types_for(accelerator: str, catalog: dict | None = None, *, amx_only: bool = True) -> list[str]:
    """Selectable instance types for a target of this accelerator, cheapest first."""
    c = catalog or load_catalog()
    out = [t for t, s in c["instances"].items() if s["accelerator"] == accelerator
           and (accelerator == "gpu" or not amx_only or s["amx"]) and s.get("usd_per_hour")]
    return sorted(out, key=lambda t: c["instances"][t]["usd_per_hour"])


def validate_for_target(instance_type: str, accelerator: str, cpu_features: list[str],
                        catalog: dict | None = None) -> None:
    s = spec(instance_type, catalog)
    if s is None:
        fams = (catalog or load_catalog())
        raise ValueError(f"{instance_type} is not in the instance catalog (families {fams['cpu_families']} and "
                         f"{fams['gpu_families']}; refresh with tools/refresh_instance_catalog.py)")
    if s["accelerator"] != accelerator:
        raise ValueError(f"{instance_type} is a {s['accelerator']} instance; the target is {accelerator}")
    if accelerator == "cpu" and cpu_features and not s["amx"]:
        raise ValueError(f"{instance_type} has no AMX; the target requires {cpu_features}")


def _ratio(gpu: dict, cpu: dict) -> float:
    return gpu["usd_per_hour"] / cpu["usd_per_hour"]


def judge(cpu_type: str, gpu_type: str, catalog: dict | None = None) -> dict:
    """Is this CPU/GPU pair comparable? ratio = GPU $/h over CPU $/h."""
    c, g = spec(cpu_type, catalog), spec(gpu_type, catalog)
    if not c or not g or not c.get("usd_per_hour") or not g.get("usd_per_hour"):
        return {"cpu": cpu_type, "gpu": gpu_type, "ratio": None, "fair": None,
                "reason": "one of the instance types is not in the catalog"}
    r = _ratio(g, c)
    lo, hi = FAIR_RATIO
    fair = lo <= r <= hi
    if fair:
        reason = (f"{gpu_type} (${g['usd_per_hour']:.3f}/h, {g['gpus']}x {g['gpu_model']}) costs {r:.2f}x "
                  f"{cpu_type} (${c['usd_per_hour']:.3f}/h, {c['vcpus']} vCPUs): within {lo}x-{hi}x")
    elif r > hi:
        reason = (f"{gpu_type} costs {r:.1f}x {cpu_type}; above {hi}x the comparison measures instance size, "
                  f"not the accelerator. Pick a smaller GPU instance or a larger CPU one")
    else:
        reason = (f"{gpu_type} costs only {r:.2f}x {cpu_type}; below {lo}x the CPU side is oversized. "
                  f"Pick a smaller CPU instance or a larger GPU one")
    return {"cpu": cpu_type, "gpu": gpu_type, "ratio": round(r, 3), "fair": fair, "reason": reason,
            "cpu_usd_per_hour": c["usd_per_hour"], "gpu_usd_per_hour": g["usd_per_hour"],
            "cpu_vcpus": c["vcpus"], "gpu_vcpus": g["vcpus"], "gpus": g["gpus"], "gpu_model": g["gpu_model"]}


def propose(instance_type: str, want: str, catalog: dict | None = None, n: int = 3) -> list[dict]:
    """Fair counterparts for an instance: want="gpu" proposes GPU types for a CPU instance and
    want="cpu" the reverse. Sorted: fair first, single-GPU first, then closest price."""
    cat = catalog or load_catalog()
    base = spec(instance_type, cat)
    if not base or not base.get("usd_per_hour"):
        return []
    out = []
    for t in types_for(want, cat):
        s = cat["instances"][t]
        if not s.get("usd_per_hour"):
            continue
        j = judge(instance_type, t, cat) if want == "gpu" else judge(t, instance_type, cat)
        gpus = s["gpus"] if want == "gpu" else base["gpus"]
        out.append({**j, "instance_type": t, "vcpus": s["vcpus"], "memory_gib": s["memory_gib"],
                    "usd_per_hour": s["usd_per_hour"], "gpus": gpus, "gpu_model": s.get("gpu_model") or base.get("gpu_model"),
                    "_sort": (not j["fair"], gpus != 1, abs(math.log(j["ratio"])))})
    out.sort(key=lambda x: x["_sort"])
    for x in out:
        del x["_sort"]
    return out[:n]


def judge_run(instance_types: dict[str, str], accelerators: dict[str, str], catalog: dict | None = None) -> list[dict]:
    """Every CPU target against every GPU target of a run (instance types already resolved)."""
    cpus = [t for t, a in accelerators.items() if a == "cpu"]
    gpus = [t for t, a in accelerators.items() if a == "gpu"]
    return [{"cpu_target": c, "gpu_target": g, **judge(instance_types[c], instance_types[g], catalog)}
            for c in sorted(cpus) for g in sorted(gpus)]


def scale_kv_cache(kv_gib: int, default_type: str | None, actual_type: str | None, catalog: dict | None = None) -> int:
    """A target's vLLM CPU KV-cache budget was chosen for its default instance size; keep the
    same share of memory when the run overrides the instance type (64 GiB -> 32 GiB KV on
    m8i.4xlarge stays 50% on m8i.8xlarge: 64 GiB)."""
    if not default_type or not actual_type or default_type == actual_type:
        return kv_gib
    d, a = spec(default_type, catalog), spec(actual_type, catalog)
    if not d or not a:
        return kv_gib
    return max(1, int(kv_gib * a["memory_gib"] / d["memory_gib"]))
