"""Contract tests for the thin server with tiny checkpoints on CPU (task 9.5, 21.2).

The server runs in-process on 127.0.0.1 with a random port. Requests go through the
task-5 client (bench.client) and the OpenAI Python client, so the wire format matches
what the load generator and standard tooling expect. Tiny random models produce
meaningless text; the tests check shapes, counts, timing fields and batching, not output.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from bench import client as bc  # noqa: E402
from servers.thin import viterbi  # noqa: E402
from servers.thin.app import create_app  # noqa: E402
from servers.thin.batcher import Job, MicroBatcher  # noqa: E402
from servers.thin.settings import Settings  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache" / "hf"
TINY = {
    "seq2seq": ("hf-internal-testing/tiny-random-T5ForConditionalGeneration", "b12e41904f9f5c33909e8f6a21cf638eca966e74"),
    "causal": ("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", "edca82771e1fad8f0b7ce110ffe770c682043fec"),
    "seq-cls": ("hf-internal-testing/tiny-random-RobertaForSequenceClassification", "0e0860b1a5e9f6691cefeaee44746aab514ce7ac"),
}
D1 = ("openai/privacy-filter", "7ffa9a043d54d1be65afb281eddf0ffbe629385b")
MNLI = ("FacebookAI/roberta-large-mnli", "2a8f12d27941090092df78e4ba6f0928eb5eac98")


def _snapshot(repo: str, rev: str, patterns=None) -> str:
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, revision=rev, cache_dir=CACHE, allow_patterns=patterns)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot download {repo}: {e}")


def _tiny_privacy_filter(tmp: Path) -> str:
    """A 2-layer OpenAIPrivacyFilter with D1's real tokenizer and label set, random weights."""
    from transformers import (
        AutoTokenizer,
        OpenAIPrivacyFilterConfig,
        OpenAIPrivacyFilterForTokenClassification,
    )

    src = Path(_snapshot(*D1, patterns=["config.json", "tokenizer*.json", "viterbi_calibration.json"]))
    real = json.loads((src / "config.json").read_text())
    cfg = OpenAIPrivacyFilterConfig(hidden_size=32, intermediate_size=32, head_dim=8, num_attention_heads=4,
                                    num_key_value_heads=2, num_hidden_layers=2, num_local_experts=4,
                                    num_experts_per_tok=2, vocab_size=real["vocab_size"],
                                    id2label={int(k): v for k, v in real["id2label"].items()},
                                    label2id=real["label2id"], pad_token_id=real.get("pad_token_id"))
    torch.manual_seed(0)
    model = OpenAIPrivacyFilterForTokenClassification(cfg)
    out = tmp / "tiny-pf"
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(src).save_pretrained(out)
    (out / "viterbi_calibration.json").write_text((src / "viterbi_calibration.json").read_text())
    return str(out)


class Server:
    def __init__(self, settings: Settings):
        import uvicorn

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        self.app = create_app(settings)
        self.server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                if httpx.get(self.url("/health"), timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        raise RuntimeError("thin server did not become healthy")

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def _settings(model_dir: str, task: str, **kw) -> Settings:
    return Settings(model_dir=model_dir, model_id=f"tiny-{task}", task=task, dtype=kw.pop("dtype", "fp32"),
                    max_batch=kw.pop("max_batch", 8), max_wait_ms=kw.pop("max_wait_ms", 20),
                    max_input_tokens=kw.pop("max_input_tokens", 256), **kw)


@pytest.fixture(scope="module")
def t5_server():
    s = Server(_settings(_snapshot(*TINY["seq2seq"]), "seq2seq"))
    yield s
    s.stop()


@pytest.fixture(scope="module")
def causal_server():
    s = Server(_settings(_snapshot(*TINY["causal"]), "causal"))
    yield s
    s.stop()


@pytest.fixture(scope="module")
def cls_server():
    s = Server(_settings(_snapshot(*TINY["seq-cls"]), "seq-cls"))
    yield s
    s.stop()


@pytest.fixture(scope="module")
def det_server(tmp_path_factory):
    s = Server(_settings(_tiny_privacy_filter(tmp_path_factory.mktemp("pf")), "token-cls", dtype="fp32"))
    yield s
    s.stop()


# ---------------------------------------------------------------------------- generation

@pytest.mark.parametrize("srv", ["t5_server", "causal_server"])
def test_health_models_and_listener(srv, request):
    s = request.getfixturevalue(srv)
    h = httpx.get(s.url("/health")).json()
    assert h["status"] == "ok" and h["max_batch"] == 8
    m = httpx.get(s.url("/v1/models")).json()
    assert m["data"][0]["id"].startswith("tiny-")


@pytest.mark.parametrize("srv", ["t5_server", "causal_server"])
def test_stream_with_bench_client_counts_tokens(srv, request):
    s = request.getfixturevalue(srv)
    model = httpx.get(s.url("/v1/models")).json()["data"][0]["id"]
    payload = bc.completion_payload(model, "Summarize: the cat sat on the mat.", max_tokens=12,
                                    temperature=0.0, seed=0, ignore_eos=True)
    with httpx.Client() as c:
        rec = bc.stream_completion(c, s.url("/v1/completions"), payload, timeout_s=60, keep_text=True)
    assert rec["error"] is None, rec
    assert rec["completion_tokens"] == 12
    assert rec["prompt_tokens"] > 5
    assert rec["n_events"] >= 2
    if rec["output_text"]:      # the tiny random T5 emits only padding, which decodes to ""
        assert rec["send_to_first_ns"] is not None and rec["send_to_last_ns"] >= rec["send_to_first_ns"]
        assert rec["send_to_end_ns"] >= rec["send_to_last_ns"]
    else:
        assert srv == "t5_server"


def test_non_stream_and_openai_client(t5_server):
    openai = pytest.importorskip("openai")
    r = httpx.post(t5_server.url("/v1/completions"), json={
        "model": "tiny-seq2seq", "prompt": "hello world", "max_tokens": 6, "temperature": 0, "ignore_eos": True}).json()
    assert r["object"] == "text_completion" and r["usage"]["completion_tokens"] == 6
    assert r["choices"][0]["finish_reason"] == "length"
    oc = openai.OpenAI(base_url=t5_server.url("/v1"), api_key="unused-by-the-thin-server")
    chunks = list(oc.completions.create(model="tiny-seq2seq", prompt="hello world", max_tokens=6, temperature=0,
                                        stream=True, stream_options={"include_usage": True},
                                        extra_body={"ignore_eos": True}))
    assert chunks[-1].usage.completion_tokens == 6
    assert chunks[-2].choices[0].finish_reason == "length"


def test_greedy_is_deterministic(t5_server):
    body = {"model": "tiny-seq2seq", "prompt": "translate: good morning", "max_tokens": 8, "temperature": 0}
    a = httpx.post(t5_server.url("/v1/completions"), json=body).json()
    b = httpx.post(t5_server.url("/v1/completions"), json=body).json()
    assert a["choices"][0]["text"] == b["choices"][0]["text"]


def test_concurrent_requests_are_batched(t5_server):
    body = {"model": "tiny-seq2seq", "prompt": "count", "max_tokens": 8, "temperature": 0, "ignore_eos": True}
    before = httpx.get(t5_server.url("/health")).json()["batches"]
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda _: httpx.post(t5_server.url("/v1/completions"), json=body, timeout=60).json(),
                           range(8)))
    assert all(o["usage"]["completion_tokens"] == 8 for o in outs)
    assert httpx.get(t5_server.url("/health")).json()["batches"] - before < 8


def test_mixed_lengths_in_one_batch(t5_server):
    with ThreadPoolExecutor(3) as ex:
        outs = list(ex.map(lambda n: httpx.post(t5_server.url("/v1/completions"), json={
            "model": "tiny-seq2seq", "prompt": "x " * n, "max_tokens": n, "temperature": 0, "ignore_eos": True},
            timeout=60).json(), [2, 5, 9]))
    assert [o["usage"]["completion_tokens"] for o in outs] == [2, 5, 9]


@pytest.mark.parametrize("body,status", [
    ({"prompt": "", "max_tokens": 4}, 400),
    ({"prompt": "hi", "max_tokens": 0}, 400),
    ({"prompt": "hi", "max_tokens": 4, "n": 2}, 400),
    ({"prompt": "hi", "max_tokens": 4, "structured_outputs": {"choice": ["A", "B"]}}, 400),
    ({"prompt": "word " * 600, "max_tokens": 4}, 400),
    ({"model": "other", "prompt": "hi", "max_tokens": 4}, 404),
])
def test_completion_rejects(t5_server, body, status):
    assert httpx.post(t5_server.url("/v1/completions"), json=body).status_code == status


def test_wrong_endpoint_for_task(t5_server, cls_server):
    assert httpx.post(t5_server.url("/classify"), json={"input": "x"}).status_code == 404
    assert httpx.post(cls_server.url("/v1/completions"), json={"prompt": "x"}).status_code == 404
    assert httpx.post(cls_server.url("/detect"), json={"input": "x"}).status_code == 404


# ---------------------------------------------------------------------------- classification

def test_classify_shapes(cls_server):
    with httpx.Client() as c:
        rec, body = bc.post_json(c, cls_server.url("/classify"),
                                 bc.classify_payload("tiny-seq-cls", ["a short text", "another one", "third"]),
                                 timeout_s=30)
    assert rec["error"] is None and rec["prompt_tokens"] > 0
    assert body["object"] == "list" and len(body["data"]) == 3
    for d in body["data"]:
        assert abs(sum(d["probs"]) - 1) < 1e-4 and d["num_classes"] == len(d["probs"])
        assert d["label"] in json.dumps(httpx.get(cls_server.url("/health")).json()["labels"])
    pair = httpx.post(cls_server.url("/classify"), json={"input": [["premise text", "hypothesis"]]}).json()
    assert len(pair["data"]) == 1
    assert httpx.post(cls_server.url("/classify"), json={"input": ["a", ["b", "c"]]}).status_code == 400


def test_multi_label_uses_sigmoid():
    from servers.thin import backends

    s = _settings(_snapshot(*TINY["seq-cls"]), "seq-cls", multi_label=True)
    b = backends.load(s)
    probs, _ = b.classify(["one", "two"])
    p = np.array(probs)
    assert ((p > 0) & (p < 1)).all() and not np.allclose(p.sum(axis=1), 1.0)


def test_mnli_joined_string_equals_pair_encoding():
    """M3 requests send premise</s></s>hypothesis as one string so vLLM /classify and the
    thin server receive the same text; it must tokenize exactly like a sentence pair."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_snapshot(*MNLI, patterns=["*.json", "*.txt", "merges.txt", "vocab.json"]))
    for a, b in [("The cat sat on the mat.", "A cat is sitting."), ("SSN 000-00-0000 for payroll", "This text contains a government ID number.")]:
        assert tok(a, b)["input_ids"] == tok(a + "</s></s>" + b)["input_ids"]


# ---------------------------------------------------------------------------- detection (D1)

def test_detect_returns_spans_and_category_scores(det_server):
    texts = ["My name is Alice Smith and my email is alice@example.com", "nothing here"]
    body = httpx.post(det_server.url("/detect"), json={"input": texts}, timeout=60).json()
    assert len(body["data"]) == 2 and body["decoder"]["type"] == "viterbi-bioes"
    cats = set(body["data"][0]["category_max"])
    assert cats == {"account_number", "private_address", "private_date", "private_email", "private_person",
                    "private_phone", "private_url", "secret"}
    for d, t in zip(body["data"], texts, strict=True):
        for sp in d["spans"]:
            assert 0 <= sp["start"] < sp["end"] <= len(t) and not t[sp["start"]].isspace()
    assert body["usage"]["prompt_tokens"] > 0


def _labels():
    labs = ["O"]
    for name in ("person", "secret"):
        labs += [f"{t}-{name}" for t in "BIES"]
    return {i: x for i, x in enumerate(labs)}


def test_viterbi_enforces_bioes():
    dec = viterbi.Decoder(_labels(), viterbi.load_biases(None))
    idx = {v: k for k, v in _labels().items()}
    # Per-token argmax would give I-person, I-person (invalid start); Viterbi must pick a valid path.
    lp = np.log(np.full((3, len(idx)), 0.01))
    lp[0, idx["I-person"]] = np.log(0.6)
    lp[0, idx["B-person"]] = np.log(0.3)
    lp[1, idx["I-person"]] = np.log(0.9)
    lp[2, idx["E-person"]] = np.log(0.9)
    path = dec.decode(lp)
    assert [_labels()[i] for i in path] == ["B-person", "I-person", "E-person"]
    assert dec.spans(path) == [("person", 0, 3)]
    # A single-token secret, then background.
    lp2 = np.log(np.full((2, len(idx)), 0.01))
    lp2[0, idx["S-secret"]] = np.log(0.9)
    lp2[1, idx["O"]] = np.log(0.9)
    assert dec.spans(dec.decode(lp2)) == [("secret", 0, 1)]


def test_viterbi_biases_shift_the_tradeoff():
    labels = _labels()
    idx = {v: k for k, v in labels.items()}
    lp = np.log(np.full((1, len(idx)), 0.01))
    lp[0, idx["O"]] = np.log(0.5)
    lp[0, idx["S-secret"]] = np.log(0.4)
    plain = viterbi.Decoder(labels, viterbi.load_biases(None))
    assert plain.spans(plain.decode(lp)) == []
    eager = viterbi.Decoder(labels, {**viterbi.load_biases(None)})
    eager.start[idx["S-secret"]] += 1.0   # start terms are what a span-entry bias moves for a 1-token input
    assert eager.spans(eager.decode(lp)) == [("secret", 0, 1)]


# ---------------------------------------------------------------------------- batcher

def test_batcher_groups_by_key_and_size():
    seen: list[list[int]] = []
    done = threading.Event()

    def run(batch):
        seen.append([j.payload for j in batch])
        if sum(len(b) for b in seen) == 7:
            done.set()

    mb = MicroBatcher(run, max_batch=3, max_wait_ms=50, on_error=lambda b, e: None)
    for i in range(7):
        mb.submit(Job(i, 1, ("a",) if i != 3 else ("b",)))
    assert done.wait(5)
    mb.close()
    assert all(len(b) <= 3 for b in seen)
    assert [3] in seen                         # the job with another key ran on its own
    flat = [x for b in seen for x in b]
    assert sorted(flat) == list(range(7))
