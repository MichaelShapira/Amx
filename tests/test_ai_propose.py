"""Auto-suggest and the AI review of instance selections: offline (fake Bedrock, fake web)."""
from __future__ import annotations

import json

import httpx
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from webui.api import ai_propose as A  # noqa: E402
from webui.api import catalog as C  # noqa: E402
from webui.api import propose  # noqa: E402
from webui.api.main import app  # noqa: E402

c = TestClient(app)
CAT = C.catalog()


def models(*ids):
    return [m for m in CAT["models"] if m["id"] in ids]


def test_auto_suggest_gives_one_best_match_and_few_alternatives():
    """A 4B chat model against g6.2xlarge: price parity picks m8i.4xlarge; the 8xlarge is the
    headroom option; nothing under 16 vCPUs is offered for a decoder; everything else is folded."""
    s = propose.suggest_cpu(models("m2-qwen3-4b-2507"), "bf16", "g6.2xlarge", {})
    assert s["best"]["instance_type"] == "m8i.4xlarge" and s["best"]["role"] == "match"
    assert [a["instance_type"] for a in s["alternatives"]] == ["m8i.8xlarge"]
    assert s["alternatives"][0]["role"] == "headroom"
    assert all(x["spec"]["vcpus"] >= 16 for x in [s["best"], *s["alternatives"], *s["others"]])
    assert all(x["fair"] for x in [s["best"], *s["alternatives"], *s["others"]])
    assert len([s["best"], *s["alternatives"]]) <= 3 and s["error"] is None
    assert "chat (decoder) model" in s["basis"][0] and "7.5 GiB" in s["basis"][0] and "at least 16 vCPUs" in s["basis"][0]
    assert "between $0.49/h and $1.96/h" in s["basis"][1]
    rejected = {x["instance_type"]: x for x in s["rejected"]}
    assert "m8i.2xlarge" in rejected and any("below the 16 vCPUs" in r["text"] for r in rejected["m8i.2xlarge"]["reasons"])
    # The best match is never a previous-generation or memory-optimized size while a Granite Rapids m/c size fits.
    assert not s["best"]["spec"]["family"].startswith(("m7", "c7", "r"))


def test_auto_suggest_classifier_allows_smaller_sizes_and_offers_an_economy_pick():
    s = propose.suggest_cpu(models("m3-roberta-large-mnli"), "bf16", "g6.xlarge", {})
    roles = {a["role"]: a["instance_type"] for a in s["alternatives"]}
    assert s["best"]["instance_type"] == "m8i.4xlarge"
    assert roles["economy"] == "m8i.2xlarge" and roles["headroom"].endswith("8xlarge")
    assert "classifier" in s["basis"][0] and "at least 8 vCPUs" in s["basis"][0]


def test_auto_suggest_sizes_for_the_largest_model_and_handles_unknown_gpu():
    s = propose.suggest_cpu(models("m2-qwen3-4b-2507", "m3-roberta-large-mnli"), "bf16", "g5.4xlarge", {})
    assert s["best"]["instance_type"] == "m8i.8xlarge" and "largest of the 2 chosen models" in s["basis"][-1]
    bad = propose.suggest_cpu(models("m2-qwen3-4b-2507"), "bf16", "p9.nothing", {})
    assert bad["best"] is None and "not in the instance catalog" in bad["error"]


def test_suggest_endpoint_validates_input():
    r = c.post("/api/suggest", json={"models": ["m2-qwen3-4b-2507"], "gpu_type": "g6.2xlarge"}).json()
    assert r["best"]["instance_type"] == "m8i.4xlarge" and r["best"]["role_label"] == "Best match"
    assert c.post("/api/suggest", json={"models": [], "gpu_type": "g6.2xlarge"}).status_code == 400
    assert c.post("/api/suggest", json={"models": ["m2-qwen3-4b-2507"], "gpu_type": "x.large"}).status_code == 400


DDG_HTML = """<div class="result results_links web-result">
<h2 class="result__title"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Faws.amazon.com%2Fec2%2Finstance-types%2Fg6%2F&amp;rut=abc">Amazon EC2 G6 Instances</a></h2>
<a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Faws.amazon.com%2Fec2%2Finstance-types%2Fg6%2F">G6 instances powered by <b>NVIDIA L4</b> GPUs &amp; more</a>
</div>
<div class="result"><h2 class="result__title"><a class="result__a" href="https://example.com/dup">Dup</a></h2></div>
<div class="result"><h2 class="result__title"><a class="result__a" href="https://example.com/dup">Dup again</a></h2></div>
<div class="result"><h2 class="result__title"><a class="result__a" href="javascript:void(0)">Bad</a></h2></div>"""


def test_search_results_are_unwrapped_deduplicated_and_unescaped():
    out = A.parse_search_html(DDG_HTML)
    assert [r["url"] for r in out] == ["https://aws.amazon.com/ec2/instance-types/g6/", "https://example.com/dup"]
    assert out[0]["title"] == "Amazon EC2 G6 Instances" and out[0]["snippet"] == "G6 instances powered by NVIDIA L4 GPUs & more"


def test_fetch_refuses_local_and_private_addresses():
    for url in ("http://169.254.169.254/latest/meta-data/", "http://localhost:8601/api/catalog", "http://127.0.0.1/",
                "ftp://example.com/x", "http://10.0.0.1/", "http://foo.internal/"):
        with pytest.raises(ValueError):
            A.check_public_url(url)
        assert A.fetch_page(url).get("error")


def test_html_to_text_keeps_the_document_title_and_drops_boilerplate():
    doc = ("<html><head><title>Amazon M8i</title><style>x{}</style></head><body><nav>menu</nav>"
           "<svg><title>icon</title></svg><h1>M8i instances</h1><p>Custom Intel Xeon 6 with <b>AMX</b>.</p>"
           "<script>alert(1)</script><footer>legal</footer></body></html>")
    title, text = A.html_to_text(doc)
    assert title == "Amazon M8i" and "menu" not in text and "alert" not in text and "legal" not in text
    assert "M8i instances" in text and "Custom Intel Xeon 6 with AMX." in text


class FakeRuntime:
    """Converse that asks for a search, then a page, then answers."""

    def __init__(self):
        self.calls = 0
        self.seen_tool_config = None

    def converse(self, **kw):
        self.calls += 1
        self.seen_tool_config = kw.get("toolConfig")
        assert "temperature" not in kw.get("inferenceConfig", {})
        if self.calls == 1:
            return {"stopReason": "tool_use", "usage": {"inputTokens": 100, "outputTokens": 10},
                    "output": {"message": {"role": "assistant", "content": [
                        {"text": "Let me check the price."},
                        {"toolUse": {"toolUseId": "t1", "name": "web_search", "input": {"query": "g6.2xlarge price us-east-1"}}}]}}}
        if self.calls == 2:
            last = kw["messages"][-1]["content"][0]["toolResult"]
            assert last["toolUseId"] == "t1" and last["content"][0]["json"]["results"][0]["url"].startswith("https://aws.amazon.com")
            return {"stopReason": "tool_use", "usage": {"inputTokens": 200, "outputTokens": 10},
                    "output": {"message": {"role": "assistant", "content": [
                        {"toolUse": {"toolUseId": "t2", "name": "fetch_page", "input": {"url": "https://aws.amazon.com/ec2/instance-types/g6/"}}},
                        {"toolUse": {"toolUseId": "t3", "name": "fetch_page", "input": {"url": "http://127.0.0.1/secret"}}}]}}}
        last = kw["messages"][-1]["content"]
        assert last[0]["toolResult"]["content"][0]["json"]["title"] == "Amazon EC2 G6 Instances"
        assert "non-public" in last[1]["toolResult"]["content"][0]["json"]["error"]
        return {"stopReason": "end_turn", "usage": {"inputTokens": 300, "outputTokens": 200},
                "output": {"message": {"role": "assistant", "content": [{"text":
                    "## Verdict\nClose match.\n## What I checked\n- price ok ([AWS](https://aws.amazon.com/ec2/pricing/on-demand/))\n"
                    "## Recommendation\nKeep m8i.4xlarge.\n## Sources\n- https://aws.amazon.com/ec2/instance-types/g6/"}]}}}


class FakeBedrock:
    def list_foundation_models(self, **kw):
        return {"modelSummaries": [{"modelId": "anthropic.claude-opus-5-5", "modelName": "Claude Opus 5.5",
                                    "modelLifecycle": {"status": "ACTIVE"}, "inferenceTypesSupported": ["INFERENCE_PROFILE"]}]}

    def list_inference_profiles(self, **kw):
        return {"inferenceProfileSummaries": [{"inferenceProfileId": "us.anthropic.claude-opus-5-5"}]}


def fake_web(request: httpx.Request) -> httpx.Response:
    if "duckduckgo.com" in request.url.host:
        return httpx.Response(200, text=DDG_HTML)
    if request.url.host == "aws.amazon.com":
        return httpx.Response(200, headers={"content-type": "text/html"},
                              text="<html><head><title>Amazon EC2 G6 Instances</title></head><body><p>L4 GPU, 24 GB.</p></body></html>")
    return httpx.Response(404)


def test_review_drives_the_tool_loop_and_records_steps_and_sources(monkeypatch):
    from webui.api import exec_summary
    from webui.api import main as M

    exec_summary._MODEL_CACHE.clear()
    monkeypatch.setattr(A, "check_public_url", lambda url: url if "127.0.0.1" not in url else (_ for _ in ()).throw(ValueError("resolves to a non-public address")))
    http = httpx.Client(transport=httpx.MockTransport(fake_web))
    w = M.Wizard(models=["m2-qwen3-4b-2507"], gpu={"hardware": "t1-g6", "instance_type": "g6.2xlarge", "precisions": ["bf16"]},
                 cpus=[{"hardware": "t2-m8i-4xl", "instance_type": "m8i.4xlarge", "precisions": ["bf16"]}])
    ctx = M._ai_context(w)
    ctx.pop("_module")
    prompt = A.build_prompt(ctx)
    for must in ("Qwen3 4B Instruct (Qwen/Qwen3-4B-Instruct-2507)", "4.02 B parameters", "g6.2xlarge: 8 vCPUs", "1x L4 with 22 GiB",
                 "m8i.4xlarge: 16 vCPUs", "AMX", "GPU price is 1.155x the CPU price -> fair", "best-match CPU size for this GPU: m8i.4xlarge",
                 "ALLOWED INSTANCE SIZES", "Short prompts, answers of 128 tokens", "requests at a time"):
        assert must in prompt, must
    job = A._new_job()
    rt = FakeRuntime()
    out = A.review(ctx, job, bedrock=FakeBedrock(), runtime=rt, http=http)
    assert rt.calls == 3 and out["tool_calls"] == 3 and out["model"]["call_id"] == "us.anthropic.claude-opus-5-5"
    assert [s["kind"] for s in out["steps"]] == ["search", "fetch", "fetch"] and out["steps"][2]["error"]
    assert out["markdown"].startswith("## Verdict") and out["usage"] == {"inputTokens": 600, "outputTokens": 220}
    urls = [s["url"] for s in out["sources"]]
    assert "https://aws.amazon.com/ec2/instance-types/g6/" in urls and "https://aws.amazon.com/ec2/pricing/on-demand/" in urls
    assert {t["toolSpec"]["name"] for t in rt.seen_tool_config["tools"]} == {"web_search", "fetch_page"}
    assert "untrusted data" in A.SYSTEM and "## Verdict" in A.SYSTEM and "never \"level\"" in A.SYSTEM


def test_ai_job_lifecycle_through_the_api(monkeypatch):
    done = {}

    def runner(ctx, job):
        job["steps"].append({"kind": "search", "text": "x", "n": 1})
        done["ctx"] = ctx
        return {"markdown": "## Verdict\nok", "model": {"name": "m"}, "usage": {}, "tool_calls": 1, "seconds": 0.1,
                "sources": [], "steps": job["steps"], "generated_at": "now"}

    monkeypatch.setattr(A, "review", runner)
    w = {"models": ["m2-qwen3-4b-2507"], "gpu": {"hardware": "t1-g6", "instance_type": "g6.2xlarge", "precisions": ["bf16"]},
         "cpus": [{"hardware": "t2-m8i-4xl", "instance_type": "m8i.4xlarge", "precisions": ["bf16"]}]}
    r = c.post("/api/propose/ai", json=w).json()
    assert "job_id" in r and "THE TEST" in r["prompt_preview"]
    import time
    for _ in range(50):
        s = c.get(f"/api/propose/ai/{r['job_id']}").json()
        if s["status"] != "running":
            break
        time.sleep(0.05)
    assert s["status"] == "done" and s["result"]["markdown"].startswith("## Verdict") and s["steps"][0]["kind"] == "search"
    assert done["ctx"]["gpu"]["instance_type"] == "g6.2xlarge"
    assert c.get("/api/propose/ai/nope").status_code == 404
    assert c.post("/api/propose/ai", json={**w, "cpus": []}).status_code == 400
    assert c.post("/api/propose/ai", json={**w, "models": []}).status_code == 400
    json.dumps(s)   # the status is plain JSON
