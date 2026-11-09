import httpx
import pytest

from bench import client as c
from bench import metrics as m
from tests import fake_server as fs

# Sleep-based scripts on a laptop jitter by a few ms; the tolerance covers that.
TOL_MS = 15.0


@pytest.fixture()
def server():
    fs.Script.ttft_ms, fs.Script.itl_ms, fs.Script.tokens = 50.0, 10.0, 10
    fs.Script.tokens_per_event, fs.Script.status, fs.Script.omit_usage = 1, 200, False
    srv, url = fs.start()
    yield url
    srv.shutdown()


def test_streaming_timing_matches_script(server):
    with httpx.Client() as cl:
        cl.get(server + "/health")
        payload = c.completion_payload("m", "hi", max_tokens=10, temperature=0, seed=0,
                                       ignore_eos=True)
        r = c.stream_completion(cl, server + "/v1/completions", payload, timeout_s=10)
    assert r["error"] is None
    assert r["completion_tokens"] == 10 and r["prompt_tokens"] == 42
    assert m.ttft_ms(r) == pytest.approx(50.0, abs=TOL_MS)
    assert m.tpot_ms(r) == pytest.approx(10.0, abs=TOL_MS / 9 + 2)
    assert r["n_events"] == 11  # 10 token events + usage


def test_multi_token_events_use_server_counts(server):
    fs.Script.tokens_per_event = 4
    fs.Script.tokens = 12
    with httpx.Client() as cl:
        payload = c.completion_payload("m", "hi", max_tokens=12, temperature=0, seed=0,
                                       ignore_eos=True)
        r = c.stream_completion(cl, server + "/v1/completions", payload, timeout_s=10)
    assert r["completion_tokens"] == 12
    # 3 events: 4 tokens each; last event at ~ttft + 2 * 40 ms
    assert m.tpot_ms(r) == pytest.approx(80.0 / 11, abs=3)


def test_missing_usage_is_error(server):
    fs.Script.omit_usage = True
    with httpx.Client() as cl:
        payload = c.completion_payload("m", "hi", max_tokens=10, temperature=0, seed=0,
                                       ignore_eos=True)
        r = c.stream_completion(cl, server + "/v1/completions", payload, timeout_s=10)
    assert r["error"] == "no_usage"


def test_http_error_recorded(server):
    fs.Script.status = 503
    with httpx.Client() as cl:
        r = c.stream_completion(cl, server + "/v1/completions", {"model": "m"}, timeout_s=10)
    assert r["error"] == "http_503" and r["timeout"] is False


def test_timeout_recorded(server):
    fs.Script.ttft_ms = 500
    with httpx.Client() as cl:
        r = c.stream_completion(cl, server + "/v1/completions", {"model": "m"}, timeout_s=0.1)
    assert r["timeout"] is True


def test_keep_text_and_classify(server):
    with httpx.Client() as cl:
        payload = c.completion_payload("m", "hi", max_tokens=2, temperature=0, seed=0,
                                       ignore_eos=False)
        fs.Script.tokens = 2
        r = c.stream_completion(cl, server + "/v1/completions", payload, timeout_s=10,
                                keep_text=True)
        assert r["output_text"] == "tok tok "
        rec, body = c.post_json(cl, server + "/classify", c.classify_payload("m", ["a", "b"]),
                                timeout_s=10)
    assert rec["error"] is None and len(body["data"]) == 2
    assert rec["prompt_tokens"] == 84
    # The fake server sleeps 50 ms; macOS timer coalescing can delay its wake-up by tens of
    # milliseconds under load, so only the lower bound is tight here.
    assert 50 - 1 <= rec["send_to_end_ns"] / m.MS <= 50 + 3 * TOL_MS


def test_constrained_payload():
    p = c.completion_payload("m", "x", max_tokens=4, temperature=0, seed=0, ignore_eos=False,
                             stop=["\n"], choices=["ALLOW", "BLOCK"])
    assert p["structured_outputs"] == {"choice": ["ALLOW", "BLOCK"]}
    assert "ignore_eos" not in p
