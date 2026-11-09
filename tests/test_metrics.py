import math

import pytest

from bench import metrics as m

NS = m.NS
MS = m.MS


def rec(worker, s_ms, first_ms, last_ms, end_ms, n=10, p=100, wave=0, start_ms=0, **kw):
    r = {"worker": worker, "wave": wave, "start_at_ns": start_ms * MS,
         "wall_send_ns": s_ms * MS, "send_to_first_ns": (first_ms - s_ms) * MS,
         "send_to_last_ns": (last_ms - s_ms) * MS, "send_to_end_ns": (end_ms - s_ms) * MS,
         "completion_tokens": n, "prompt_tokens": p, "error": None, "timeout": False}
    r.update(kw)
    return r


def test_pct_exact_values():
    out = m.pct(range(1, 101))
    assert out["n"] == 100
    assert out["p50"] == pytest.approx(50.5)
    assert out["p99"] == pytest.approx(99.01)
    assert out["p1"] == pytest.approx(1.99)
    assert out["min"] == 1 and out["max"] == 100


def test_pct_empty():
    assert m.pct([])["p50"] is None


def test_per_request_metrics():
    r = rec(0, 0, 100, 1000, 1010, n=10, p=200)
    assert m.e2e_ms(r) == 1010
    assert m.ttft_ms(r) == 100
    assert m.tpot_ms(r) == pytest.approx(100.0)  # 900 ms over 9 tokens
    assert m.prefill_tok_s(r) == pytest.approx(2000.0)
    assert m.tpot_ms(rec(0, 0, 100, 100, 110, n=1)) is None


def test_window_is_earliest_worker_finish():
    recs = [rec(0, 0, 10, 90, 100), rec(0, 100, 110, 190, 200),
            rec(1, 0, 10, 140, 150)]
    assert m.wave_window(recs) == (0, 150 * MS)


def test_aggregate_tokens_by_overlap():
    # Two workers, window [0, 100 ms]. Worker 1's request decodes from 50 to 250 ms.
    recs = [rec(0, 0, 10, 100, 100, n=11), rec(1, 0, 50, 250, 300, n=21)]
    agg = m.wave_aggregate(recs)
    assert agg["window_s"] == pytest.approx(0.1)
    # worker 0: all 11 tokens; worker 1: first token + 20 * (50/200) = 6
    assert agg["out_tokens"] == pytest.approx(11 + 6)
    assert agg["requests"] == 1
    # concurrency integral: worker 0 0.1 s, worker 1 0.1 s overlap
    assert agg["conc_integral_s"] == pytest.approx(0.2)


def test_summarize_level_steady_state():
    recs = []
    for w in range(4):
        t = 0
        for _ in range(10):
            recs.append(rec(w, t, t + 20, t + 100, t + 100, n=9, p=50))
            t += 100
    s = m.summarize_level(recs, concurrency=4)
    assert s["effective_concurrency"] == pytest.approx(4.0)
    assert s["requests_per_s"] == pytest.approx(40.0)
    assert s["agg_out_tok_s"] == pytest.approx(4 * 9 * 10)  # 4 workers x 90 tok/s
    assert s["e2e_ms"]["p50"] == pytest.approx(100)
    assert s["tpot_ms"]["p50"] == pytest.approx(10)
    assert not s["flags"]


def test_errors_and_flags():
    recs = [rec(0, 0, 10, 50, 60), rec(0, 60, 70, 110, 120, error="http_500"),
            rec(1, 0, 10, 50, 60, timeout=True, error="timeout"),
            rec(1, 60, 70, 110, 120, prompt_tokens_expected=99)]
    s = m.summarize_level(recs, concurrency=2)
    assert s["error_pct"] == pytest.approx(25.0)
    assert s["timeout_pct"] == pytest.approx(25.0)
    assert "prompt_token_mismatch_1" in s["flags"]


def test_low_effective_concurrency_flagged():
    recs = [rec(0, 0, 10, 90, 100), rec(1, 900, 910, 990, 1000)]
    s = m.summarize_level(recs, concurrency=2)
    assert any(f.startswith("effective_concurrency") for f in s["flags"])


def test_start_spread_flag():
    recs = [rec(0, 0, 10, 90, 100), rec(1, 400, 410, 490, 500)]
    assert any(f.startswith("start_spread") for f in m.wave_flags(recs))


def test_classifier_summary():
    recs = []
    for s_ms in (0, 50):
        r = rec(0, s_ms, s_ms, s_ms, s_ms + 50, batch_size=8, p=512)
        r["send_to_first_ns"] = None
        r["send_to_last_ns"] = None
        recs.append(r)
    s = m.summarize_level(recs, concurrency=1, kind="classifier")
    assert s["classifications_per_s"] == pytest.approx(160.0)
    assert s["e2e_ms"]["p50"] == pytest.approx(50)


def test_combine_repeats_variance():
    reps, allr = [], []
    for i, lat in enumerate([100, 100, 130]):
        rs = [rec(0, k * lat, k * lat + 10, (k + 1) * lat, (k + 1) * lat, n=10, wave=i)
              for k in range(5)]
        reps.append(m.summarize_level(rs, concurrency=1))
        allr += rs
    out = m.combine_repeats(reps, allr, concurrency=1)
    v = out["variance"]["e2e_ms.p50"]
    assert v["mean"] == pytest.approx(110)
    assert v["cv"] == pytest.approx(math.sqrt(300) / 110, rel=1e-6)
    assert any(f.startswith("cv_e2e_ms") for f in out["flags"])
    assert out["repeats"] == 3
