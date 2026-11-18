"""Metric definitions (design section 11).

Raw records are dicts with absolute wall-clock send time and monotonic offsets from send:
  wall_send_ns, send_to_first_ns, send_to_last_ns, send_to_end_ns, start_at_ns,
  prompt_tokens, completion_tokens, error, timeout, worker, wave, conc, batch_size (classifier)
Percentiles are exact over all successful measured requests (Requirement 8.7).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np

NS = 1_000_000_000
MS = 1_000_000

START_SPREAD_FLAG_MS = 250.0
LATE_WORKER_MS = 2000.0
LATE_WAVE_FRACTION = 0.10
EFFECTIVE_CONC_FLAG = 0.9
CV_FLAG = 0.10


def pct(values: Iterable[float]) -> dict:
    """P1, P50, P90, P99, mean, min, max and n, using linear interpolation."""
    arr = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "p1": None, "p50": None, "p90": None, "p99": None,
                "mean": None, "min": None, "max": None}
    p1, p50, p90, p99 = np.percentile(arr, [1, 50, 90, 99], method="linear")
    return {"n": int(arr.size), "p1": float(p1), "p50": float(p50), "p90": float(p90),
            "p99": float(p99), "mean": float(arr.mean()), "min": float(arr.min()),
            "max": float(arr.max())}


def ok(r: dict) -> bool:
    return not r.get("error") and not r.get("timeout")


def times(r: dict) -> tuple[int, int | None, int | None, int]:
    s = int(r["wall_send_ns"])
    f = s + int(r["send_to_first_ns"]) if r.get("send_to_first_ns") is not None else None
    last = s + int(r["send_to_last_ns"]) if r.get("send_to_last_ns") is not None else None
    e = s + int(r["send_to_end_ns"])
    return s, f, last, e


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def e2e_ms(r: dict) -> float:
    return r["send_to_end_ns"] / MS


def ttft_ms(r: dict) -> float | None:
    v = r.get("send_to_first_ns")
    return None if v is None else v / MS


def tpot_ms(r: dict) -> float | None:
    n = r.get("completion_tokens") or 0
    if n < 2 or r.get("send_to_first_ns") is None or r.get("send_to_last_ns") is None:
        return None
    return (r["send_to_last_ns"] - r["send_to_first_ns"]) / MS / (n - 1)


def prefill_tok_s(r: dict) -> float | None:
    f = r.get("send_to_first_ns")
    p = r.get("prompt_tokens")
    if not f or not p:
        return None
    return p / (f / NS)


def wave_window(records: Sequence[dict]) -> tuple[int, int] | None:
    """W = [start_at, earliest time any worker finished its last request]."""
    if not records:
        return None
    start = min(int(r["start_at_ns"]) for r in records)
    last_end_by_worker: dict[int, int] = {}
    for r in records:
        _, _, _, e = times(r)
        w = r["worker"]
        last_end_by_worker[w] = max(last_end_by_worker.get(w, 0), e)
    end = min(last_end_by_worker.values())
    if end <= start:
        return None
    return start, end


def wave_aggregate(records: Sequence[dict], kind: str = "generation") -> dict:
    """Throughput figures inside one wave's window."""
    win = wave_window(records)
    if win is None:
        return {"window_s": 0.0, "out_tokens": 0.0, "in_tokens": 0.0, "requests": 0,
                "items": 0, "bytes": 0, "chars": 0, "conc_integral_s": 0.0}
    w0, w1 = win
    out_tokens = 0.0
    in_tokens = 0.0
    requests = 0
    items = 0
    nbytes = 0
    nchars = 0
    conc = 0.0
    for r in records:
        s, f, last, e = times(r)
        conc += overlap(s, e, w0, w1) / NS
        if not ok(r):
            continue
        if w0 <= e <= w1:
            requests += 1
            items += int(r.get("batch_size") or 1)
            nbytes += int(r.get("content_bytes") or 0)
            nchars += int(r.get("content_chars") or 0)
        if kind == "generation" and f is not None:
            n = int(r.get("completion_tokens") or 0)
            if w0 <= f <= w1:
                out_tokens += 1 if n >= 1 else 0
                in_tokens += int(r.get("prompt_tokens") or 0)
            if n >= 2 and last is not None and last > f:
                out_tokens += (n - 1) * overlap(f, last, w0, w1) / (last - f)
        elif kind == "classifier" and w0 <= e <= w1:
            in_tokens += int(r.get("prompt_tokens") or 0)
    return {"window_s": (w1 - w0) / NS, "out_tokens": out_tokens, "in_tokens": in_tokens,
            "requests": requests, "items": items, "bytes": nbytes, "chars": nchars, "conc_integral_s": conc}


def wave_flags(records: Sequence[dict]) -> list[str]:
    flags = []
    first_send: dict[int, int] = {}
    late = set()
    for r in records:
        w = r["worker"]
        first_send[w] = min(first_send.get(w, r["wall_send_ns"]), r["wall_send_ns"])
        if (r.get("late_start_ms") or 0) > LATE_WORKER_MS:
            late.add(w)
    if first_send:
        spread_ms = (max(first_send.values()) - min(first_send.values())) / MS
        if spread_ms > START_SPREAD_FLAG_MS:
            flags.append(f"start_spread_{spread_ms:.0f}ms")
        if len(late) > LATE_WAVE_FRACTION * len(first_send):
            flags.append(f"late_workers_{len(late)}")
    return flags


def summarize_level(records: Sequence[dict], *, concurrency: int, kind: str = "generation",
                    ignore_eos: bool = False, max_new_tokens: int | None = None) -> dict:
    """Metrics for one level and one repeat (all waves)."""
    by_wave: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_wave[int(r.get("wave", 0))].append(r)
    totals = defaultdict(float)
    flags: list[str] = []
    for wave_records in by_wave.values():
        agg = wave_aggregate(wave_records, kind)
        for k, v in agg.items():
            totals[k] += v
        flags += wave_flags(wave_records)
    good = [r for r in records if ok(r)]
    n_all = len(records)
    errors = sum(1 for r in records if r.get("error") and not r.get("timeout"))
    timeouts = sum(1 for r in records if r.get("timeout"))
    win = totals["window_s"] or float("nan")
    out = {
        "concurrency": concurrency,
        "kind": kind,
        "n_requests": n_all,
        "n_ok": len(good),
        "error_pct": 100.0 * errors / n_all if n_all else 0.0,
        "timeout_pct": 100.0 * timeouts / n_all if n_all else 0.0,
        "e2e_ms": pct(e2e_ms(r) for r in good),
        "requests_per_s": totals["requests"] / win if totals["window_s"] else None,
        "effective_concurrency": totals["conc_integral_s"] / win if totals["window_s"] else None,
        "window_s": totals["window_s"],
        "bytes_per_s": totals["bytes"] / win if totals["window_s"] and totals["bytes"] else None,
        # Source characters translated per second (translation suite; pool lines carry `chars`).
        "chars_per_s": totals["chars"] / win if totals["window_s"] and totals["chars"] else None,
    }
    if kind == "generation":
        out.update({
            "ttft_ms": pct(ttft_ms(r) for r in good),
            "tpot_ms": pct(tpot_ms(r) for r in good),
            "out_tok_s_per_request": pct(
                (r.get("completion_tokens") or 0) / (r["send_to_end_ns"] / NS) for r in good),
            "prefill_tok_s_per_request": pct(prefill_tok_s(r) for r in good),
            "decode_tok_s_per_request": pct(
                (1000.0 / t) if t else None for t in (tpot_ms(r) for r in good)),
            "agg_out_tok_s": totals["out_tokens"] / win if totals["window_s"] else None,
            "agg_in_tok_s": totals["in_tokens"] / win if totals["window_s"] else None,
            "documents_per_s": totals["items"] / win if totals["window_s"] else None,
        })
        mism = sum(1 for r in good if r.get("prompt_tokens_expected") is not None
                   and r.get("prompt_tokens") != r.get("prompt_tokens_expected"))
        if mism:
            flags.append(f"prompt_token_mismatch_{mism}")
        if ignore_eos and max_new_tokens:
            short = sum(1 for r in good if r.get("completion_tokens") != max_new_tokens)
            if short:
                flags.append(f"output_length_mismatch_{short}")
    else:
        out.update({
            "classifications_per_s": totals["items"] / win if totals["window_s"] else None,
            "agg_in_tok_s": totals["in_tokens"] / win if totals["window_s"] else None,
        })
    eff = out["effective_concurrency"]
    if eff is not None and eff < EFFECTIVE_CONC_FLAG * concurrency:
        flags.append(f"effective_concurrency_{eff:.2f}")
    out["flags"] = sorted(set(flags))
    return out


HEADLINE_KEYS = (("e2e_ms", "p50"), ("e2e_ms", "p99"), ("agg_out_tok_s", None),
                 ("classifications_per_s", None), ("documents_per_s", None))


def _get(summary: dict, key: str, sub: str | None):
    v = summary.get(key)
    if sub and isinstance(v, dict):
        v = v.get(sub)
    return v


def combine_repeats(per_repeat: Sequence[dict], all_records: Sequence[dict], *,
                    concurrency: int, kind: str = "generation", ignore_eos: bool = False,
                    max_new_tokens: int | None = None) -> dict:
    """Pooled percentiles across repeats plus repeat-to-repeat variance (Requirement 7.5)."""
    pooled = summarize_level(all_records, concurrency=concurrency, kind=kind,
                             ignore_eos=ignore_eos, max_new_tokens=max_new_tokens)
    # Throughput is averaged over repeats rather than recomputed across waves of different repeats.
    for key in ("agg_out_tok_s", "agg_in_tok_s", "requests_per_s", "classifications_per_s",
                "documents_per_s", "effective_concurrency", "bytes_per_s", "chars_per_s"):
        vals = [r.get(key) for r in per_repeat if r.get(key) is not None]
        if key in pooled:
            pooled[key] = float(np.mean(vals)) if vals else None
    variance = {}
    flags = list(pooled["flags"])
    for key, sub in HEADLINE_KEYS:
        vals = [_get(r, key, sub) for r in per_repeat]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 2:
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1))
            cv = std / mean if mean else None
            variance[f"{key}{'.' + sub if sub else ''}"] = {"mean": mean, "std": std, "cv": cv}
            if cv is not None and cv > CV_FLAG:
                flags.append(f"cv_{key}{'_' + sub if sub else ''}_{cv:.2f}")
    pooled["repeats"] = len(per_repeat)
    pooled["variance"] = variance
    pooled["flags"] = sorted(set(flags))
    return pooled
