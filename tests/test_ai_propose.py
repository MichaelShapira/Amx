"""The AI hardware proposal: offline (a fake Bedrock runtime, a fake AgentCore web search)."""
from __future__ import annotations

import json

import pytest
from botocore.exceptions import ClientError

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from webui.api import ai_propose as A  # noqa: E402
from webui.api.main import app  # noqa: E402

c = TestClient(app)

PROPOSAL = {
    "gpu": {"instance_type": "g6.2xlarge", "why": "One L4 GPU with 24 GB holds the 4 billion parameter model with room to spare.",
            "alternates": ["g6.4xlarge", "p9.nothing", "g6.2xlarge"]},
    "cpu": {"instance_type": "m8i.4xlarge", "why": "16 cores of the current Intel generation with AMX, priced close to the GPU."},
    "summary": "For about the same hourly spend, this pair shows what each architecture delivers.",
    "checked": ["g6.2xlarge is $0.9776/h On-Demand in us-east-1 (https://aws.amazon.com/ec2/pricing/on-demand/)"],
    "caveats": ["The CPU price on the AWS page is a cent lower than the catalog's snapshot."],
    "confidence": "high",
    "sources": ["https://aws.amazon.com/ec2/pricing/on-demand/", "https://aws.amazon.com/ec2/instance-types/g6/", "ftp://nope"],
}


class FakeSearch:
    """bedrock-agentcore InvokeWebSearch."""

    def __init__(self, fail_code: str | None = None):
        self.queries: list[dict] = []
        self.fail_code = fail_code

    def invoke_web_search(self, **kw):
        self.queries.append(kw)
        if self.fail_code:
            raise ClientError({"Error": {"Code": self.fail_code, "Message": "no"}}, "InvokeWebSearch")
        assert kw["webSearchId"] == "web-search.v1" and 1 <= kw["maxResults"] <= 25 and len(kw["searchQuery"]) <= 200
        import datetime as dt
        return {"results": [
            {"title": "Amazon EC2 G6 Instances", "url": "https://aws.amazon.com/ec2/instance-types/g6/",
             "text": "G6 instances feature NVIDIA L4 Tensor Core GPUs with 24 GB of memory. " * 60,
             "publishedAt": dt.datetime(2026, 3, 1, tzinfo=dt.UTC)},
            {"title": "", "url": "javascript:alert(1)", "text": "ignored"},
            {"url": "https://aws.amazon.com/ec2/pricing/on-demand/", "text": "g6.2xlarge $0.9776 per hour"},
        ]}


class FakeRuntime:
    """Converse that searches twice, then answers with the JSON block."""

    def __init__(self, answer: str | None = None):
        self.calls = 0
        self.seen_tool_config = None
        self.answer = answer or "Here is my proposal.\n```json\n" + json.dumps(PROPOSAL) + "\n```"

    def converse(self, **kw):
        self.calls += 1
        self.seen_tool_config = kw.get("toolConfig")
        assert "temperature" not in kw.get("inferenceConfig", {})
        if self.calls == 1:
            return {"stopReason": "tool_use", "usage": {"inputTokens": 100, "outputTokens": 10},
                    "output": {"message": {"role": "assistant", "content": [
                        {"text": "Let me check the price."},
                        {"toolUse": {"toolUseId": "t1", "name": "web_search", "input": {"query": "g6.2xlarge price us-east-1"}}},
                        {"toolUse": {"toolUseId": "t2", "name": "fetch_page", "input": {"url": "https://x"}}}]}}}
        if self.calls == 2:
            last = kw["messages"][-1]["content"]
            res = last[0]["toolResult"]["content"][0]["json"]
            assert last[0]["toolResult"]["toolUseId"] == "t1"
            assert [r["url"] for r in res["results"]] == ["https://aws.amazon.com/ec2/instance-types/g6/", "https://aws.amazon.com/ec2/pricing/on-demand/"]
            assert len(res["results"][0]["text"]) <= A.RESULT_CHARS and res["results"][0]["published"] == "2026-03-01"
            assert "unknown tool" in last[1]["toolResult"]["content"][0]["json"]["error"]
            return {"stopReason": "tool_use", "usage": {"inputTokens": 200, "outputTokens": 10},
                    "output": {"message": {"role": "assistant", "content": [
                        {"toolUse": {"toolUseId": "t3", "name": "web_search", "input": {"query": "vLLM CPU supported models Qwen3"}}}]}}}
        return {"stopReason": "end_turn", "usage": {"inputTokens": 300, "outputTokens": 200},
                "output": {"message": {"role": "assistant", "content": [{"text": self.answer}]}}}


class FakeBedrock:
    def list_foundation_models(self, **kw):
        if kw.get("byProvider") != "anthropic":
            return {"modelSummaries": []}
        return {"modelSummaries": [{"modelId": "anthropic.claude-opus-5-5", "modelName": "Claude Opus 5.5",
                                    "modelLifecycle": {"status": "ACTIVE"}, "inferenceTypesSupported": ["INFERENCE_PROFILE"]}]}

    def list_inference_profiles(self, **kw):
        return {"inferenceProfileSummaries": [{"inferenceProfileId": "us.anthropic.claude-opus-5-5"}]}


def wizard(**over):
    from webui.api import main as M
    return M.Wizard(**{"models": ["m7-gemma-4-e2b"], **over})


@pytest.fixture
def settings_default(monkeypatch):
    from webui.api import settings as S
    S.reset_for_tests()
    monkeypatch.setattr(S, "get", lambda: {})          # nothing chosen in Setup -> the newest Opus
    yield
    S.reset_for_tests()


def test_web_search_trims_results_and_reports_failures_as_data():
    s = FakeSearch()
    out = A.web_search("  g6.2xlarge   price " + "x" * 300, n=40, client=s)
    assert len(s.queries[0]["searchQuery"]) == 200 and s.queries[0]["maxResults"] == 25
    assert len(out["results"]) == 2 and out["results"][1]["title"] == "" and "error" not in out
    assert A.web_search("", client=s)["error"] == "empty query"
    denied = A.web_search("anything", client=FakeSearch("AccessDeniedException"))
    assert denied["results"] == [] and denied["error"] == "search unavailable (AccessDeniedException)"


def test_prompt_carries_the_model_the_test_every_allowed_size_and_the_fairness_rule(settings_default):
    from webui.api import main as M
    ctx = M._ai_context(wizard())
    ctx.pop("_module")
    assert ctx["current"] is None and ctx["fair_ratio"] == [0.5, 2.0]
    prompt = A.build_prompt(ctx)
    for must in ("Gemma 4 E2B Instruct (google/gemma-4-E2B-it)", "5.12 B parameters", "ALLOWED GPU SIZES", "ALLOWED CPU SIZES",
                 "g6.2xlarge: 8 vCPUs", "1x L4 with 22 GiB", "m8i.4xlarge: 16 vCPUs", "FAIRNESS RULE: GPU hourly price / CPU hourly price must be between 0.5 and 2.0",
                 "Short prompts, answers of 128 tokens", "TASK: Propose the GPU size and the CPU size"):
        assert must in prompt, must
    assert "CURRENTLY SELECTED" not in prompt and "added by the user" not in prompt
    # With a selection on the page, the model sees it and the tool's price check of it.
    ctx2 = M._ai_context(wizard(gpu={"hardware": "t1-g6", "instance_type": "g6.2xlarge", "precisions": ["bf16"]},
                                cpus=[{"hardware": "t2-m8i-4xl", "instance_type": "m8i.4xlarge", "precisions": ["bf16"]}]))
    p2 = A.build_prompt(ctx2)
    assert "CURRENTLY SELECTED" in p2 and "GPU: g6.2xlarge" in p2 and "m8i.4xlarge vs g6.2xlarge: GPU price is" in p2
    assert "untrusted data" in A.SYSTEM and "half to double" in A.SYSTEM and "```json" in A.SYSTEM and "never \"level\"" in A.SYSTEM
    assert {t["toolSpec"]["name"] for t in A.TOOLS["tools"]} == {"web_search"}      # web search only; no page fetching


def test_parse_proposal_validates_sizes_against_the_allowed_lists():
    allowed = {"gpu": [("g6.2xlarge", {}), ("g6.4xlarge", {})], "cpu": [("m8i.4xlarge", {})]}
    p, problem = A.parse_proposal("text before\n```json\n" + json.dumps(PROPOSAL) + "\n```", allowed)
    assert problem is None and p["gpu"]["instance_type"] == "g6.2xlarge" and p["cpu"]["instance_type"] == "m8i.4xlarge"
    assert p["gpu"]["alternates"] == ["g6.4xlarge"]                 # unknown and duplicate alternates dropped
    assert p["sources"] == PROPOSAL["sources"][:2] and p["confidence"] == "high" and len(p["checked"]) == 1
    bad = {**PROPOSAL, "cpu": {"instance_type": "c7i.48xlarge", "why": "big"}}
    p, problem = A.parse_proposal("```json\n" + json.dumps(bad) + "\n```", allowed)
    assert p is None and problem == "c7i.48xlarge is not an allowed CPU size"
    assert A.parse_proposal("I cannot decide.", allowed) == (None, "the model answered without the expected JSON block")
    assert A.parse_proposal("```json\n{broken\n```", allowed)[0] is None
    assert A.parse_proposal(json.dumps({**PROPOSAL, "confidence": "sure"}), allowed)[0]["confidence"] == "medium"
    # Unfenced JSON followed by prose, and JSON with prose before it, both parse (raw_decode stops at the object).
    p, problem = A.parse_proposal("Here you go.\n" + json.dumps(PROPOSAL) + "\n\nLet me know if you want changes.", allowed)
    assert problem is None and p["gpu"]["instance_type"] == "g6.2xlarge"
    # A fence whose content is cut off (the real failure: maxTokens reached mid-string) is reported as cut off.
    cut = "```json\n" + json.dumps(PROPOSAL)[:-40]
    p, problem = A.parse_proposal(cut, allowed)
    assert p is None and problem.startswith("the model's JSON was cut off")


def test_propose_repairs_a_cut_off_answer_once(settings_default):
    from webui.api import main as M
    ctx = M._ai_context(wizard())
    ctx.pop("_module")

    class CutThenGood(FakeRuntime):
        def converse(self, **kw):
            self.calls += 1
            if self.calls == 1:            # the first answer is cut off by the token limit
                assert kw["inferenceConfig"]["maxTokens"] == A.MAX_TOKENS
                return {"stopReason": "max_tokens", "usage": {"inputTokens": 100, "outputTokens": 100},
                        "output": {"message": {"role": "assistant", "content": [{"text": "```json\n" + json.dumps(PROPOSAL)[:-30]}]}}}
            last = kw["messages"][-1]["content"][0]["text"]
            assert "could not be used: the model's JSON was cut off" in last and "ONLY the JSON block" in last
            return {"stopReason": "end_turn", "usage": {"inputTokens": 200, "outputTokens": 50},
                    "output": {"message": {"role": "assistant", "content": [{"text": json.dumps(PROPOSAL)}]}}}

    rt = CutThenGood()
    out = A.propose(ctx, A._new_job(), bedrock=FakeBedrock(), runtime=rt, search=FakeSearch())
    assert rt.calls == 2 and out["repaired"] and out["problem"] is None and out["proposal"]["cpu"]["instance_type"] == "m8i.4xlarge"
    assert [s["kind"] for s in out["steps"]] == ["repair"] and out["stop_reason"] == "end_turn"


def test_propose_drives_the_search_loop_and_returns_a_validated_proposal_with_sources(settings_default):
    from webui.api import main as M
    ctx = M._ai_context(wizard())
    ctx.pop("_module")
    job, rt, search = A._new_job(), FakeRuntime(), FakeSearch()
    out = A.propose(ctx, job, bedrock=FakeBedrock(), runtime=rt, search=search)
    assert rt.calls == 3 and out["tool_calls"] == 2 and out["model"]["call_id"] == "us.anthropic.claude-opus-5-5"
    assert [s["kind"] for s in out["steps"]] == ["search", "search"] and out["steps"][0]["n"] == 2
    assert out["problem"] is None and out["markdown"] == "" and out["usage"] == {"inputTokens": 600, "outputTokens": 220}
    assert out["proposal"]["gpu"]["instance_type"] == "g6.2xlarge" and out["proposal"]["cpu"]["instance_type"] == "m8i.4xlarge"
    assert [s["url"] for s in out["sources"]] == PROPOSAL["sources"][:2]
    assert out["sources"][1]["title"] == "Amazon EC2 G6 Instances"                 # titled from the search result it came from
    assert rt.seen_tool_config == A.TOOLS


def test_propose_without_a_usable_answer_keeps_the_text_and_says_what_went_wrong(settings_default):
    from webui.api import main as M
    ctx = M._ai_context(wizard())
    ctx.pop("_module")
    rt = FakeRuntime(answer="I would pick p5.48xlarge, see https://aws.amazon.com/ec2/instance-types/p5/ .")
    out = A.propose(ctx, A._new_job(), bedrock=FakeBedrock(), runtime=rt, search=FakeSearch())
    assert out["proposal"] is None and out["problem"] == "the model answered without the expected JSON block"
    assert out["markdown"].startswith("I would pick") and [s["url"] for s in out["sources"]] == ["https://aws.amazon.com/ec2/instance-types/p5/"]


def test_ai_job_lifecycle_through_the_api(monkeypatch):
    done = {}

    def runner(ctx, job):
        job["steps"].append({"kind": "search", "text": "x", "n": 1})
        done["ctx"] = ctx
        return {"proposal": {"gpu": {"instance_type": "g6.2xlarge"}}, "problem": None, "markdown": "", "model": {"name": "m"}, "usage": {},
                "tool_calls": 1, "seconds": 0.1, "sources": [], "steps": job["steps"], "generated_at": "now"}

    monkeypatch.setattr(A, "propose", runner)
    w = {"models": ["m7-gemma-4-e2b"]}
    r = c.post("/api/propose/ai", json=w).json()
    assert "job_id" in r and "THE TEST" in r["prompt_preview"]
    import time
    for _ in range(50):
        s = c.get(f"/api/propose/ai/{r['job_id']}").json()
        if s["status"] != "running":
            break
        time.sleep(0.05)
    assert s["status"] == "done" and s["result"]["proposal"]["gpu"]["instance_type"] == "g6.2xlarge" and s["steps"][0]["kind"] == "search"
    assert done["ctx"]["current"] is None and [t for t, _ in done["ctx"]["allowed"]["gpu"]]
    assert c.get("/api/propose/ai/nope").status_code == 404
    assert c.post("/api/propose/ai", json={"models": []}).status_code == 400
    json.dumps(s)   # the status is plain JSON


HISTORY = [{"run_id": "gemma-dspm-20261122T085327Z", "date": "2026-11-22", "grade": "full", "use_case": "Sensitive-data detection (DSPM / DLP)",
            "models": ["Gemma 4 E4B Instruct"], "gpu_type": "g6.2xlarge", "cpu_type": "m8i.4xlarge",
            "questions": [{"label": "Gemma 4 E4B · Inline DLP", "verdict": "To match one g6.2xlarge you need 15 x m8i.4xlarge.",
                           "best_gpu": {"instance_type": "g6.2xlarge", "load": "16 at a time", "throughput": "79.5 documents/s", "latency": "207 ms", "cost": "$7.72 per GB"},
                           "best_cpu": {"instance_type": "m8i.4xlarge", "load": "16 at a time", "throughput": "5.6 documents/s", "latency": "3,789 ms", "cost": "$94.84 per GB"},
                           "sizing": [{"side": "gpu", "instance_type": "g6.2xlarge", "label": "Bigger than this load needs", "busy": "29%",
                                       "suggestion": {"instance_type": "g6.xlarge", "kind": "smaller"}},
                                      {"side": "cpu", "instance_type": "m8i.4xlarge", "label": "Too small for this load", "busy": "100%",
                                       "suggestion": {"instance_type": "m8i.8xlarge", "kind": "bigger"}}]}],
            "notes": ["AWS refused for capacity: g6.xlarge x4"]}]


class HistoryRuntime(FakeRuntime):
    """Searches once, proposes g6.2xlarge / m8i.4xlarge from the web, then changes the CPU to m8i.8xlarge once it sees the runs."""
    def __init__(self):
        super().__init__()
        self.prompts: list[str] = []
    def converse(self, **kw):
        last = kw["messages"][-1]["content"]
        if "text" in last[0]:
            self.prompts.append(last[0]["text"])
        if self.calls >= 1 and any("EARLIER RUNS ON THIS TOOL" in t for t in self.prompts[-1:]):
            self.calls += 1
            final = {**PROPOSAL, "cpu": {"instance_type": "m8i.8xlarge", "why": "The 4xlarge ran saturated at 100% in the earlier run."},
                     "history": "The earlier DSPM run saturated the m8i.4xlarge and left the g6.2xlarge mostly idle; the CPU moves up one size."}
            return {"stopReason": "end_turn", "usage": {"inputTokens": 500, "outputTokens": 150},
                    "output": {"message": {"role": "assistant", "content": [{"text": "```json\n" + json.dumps(final) + "\n```"}]}}}
        return super().converse(**kw)


def test_history_prompt_carries_sizes_verdicts_sizing_and_refusals():
    txt = A.build_history_prompt(HISTORY)
    for part in ("EARLIER RUNS ON THIS TOOL (1", "gemma-dspm-20261122T085327Z", "GPU g6.2xlarge, CPU m8i.4xlarge", "Inline DLP",
                 "best GPU: g6.2xlarge at 16 at a time: 79.5 documents/s", "sizing CPU m8i.4xlarge: Too small for this load (100% busy",
                 "suggests m8i.8xlarge (bigger)", "AWS refused for capacity: g6.xlarge x4", "FINAL VERDICT"):
        assert part in txt, part


def test_propose_weighs_earlier_runs_after_the_web_search_and_reports_what_changed(settings_default):
    from webui.api import main as M
    ctx = M._ai_context(wizard())
    ctx.pop("_module")
    ctx["history"] = HISTORY
    job, rt = A._new_job(), HistoryRuntime()
    out = A.propose(ctx, job, bedrock=FakeBedrock(), runtime=rt, search=FakeSearch())
    kinds = [s["kind"] for s in out["steps"]]
    assert kinds == ["search", "search", "history"] and out["steps"][2]["n"] == 1 and "m8i.4xlarge vs g6.2xlarge" in out["steps"][2]["text"]
    assert any("EARLIER RUNS ON THIS TOOL" in t for t in rt.prompts)        # the history went to the model AFTER the searches
    assert rt.prompts.index(next(t for t in rt.prompts if "EARLIER RUNS" in t)) > 0
    p = out["proposal"]
    assert p["cpu"]["instance_type"] == "m8i.8xlarge" and p["gpu"]["instance_type"] == "g6.2xlarge"
    assert "saturated" in p["history"]
    assert out["history_runs"] == 1 and out["web_only_proposal"] == {"gpu": "g6.2xlarge", "cpu": "m8i.4xlarge"}


def test_propose_without_earlier_runs_asks_once_and_has_no_history(settings_default):
    from webui.api import main as M
    ctx = M._ai_context(wizard())
    ctx.pop("_module")
    ctx["history"] = []
    job, rt = A._new_job(), HistoryRuntime()
    out = A.propose(ctx, job, bedrock=FakeBedrock(), runtime=rt, search=FakeSearch())
    assert not any("EARLIER RUNS" in t for t in rt.prompts) and out["history_runs"] == 0 and out["web_only_proposal"] is None
    assert out["proposal"]["cpu"]["instance_type"] == "m8i.4xlarge" and out["proposal"]["history"] is None
