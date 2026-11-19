"""Thin inference server: OpenAI-shaped /v1/completions, vLLM-shaped /classify, and
/detect for span detectors (design 7.5 and 24.4).

    THIN_MODEL_DIR=/models/m5 THIN_TASK=seq2seq python -m servers.thin.app

The server binds 127.0.0.1 unless THIN_HOST says otherwise. In the benchmark the
container binds all interfaces inside the container and serve.sh publishes the port on
the instance's private IP only; the security group admits only the load-generator
workers (design section 6). There is no application-level authentication, by decision
(design 23.1).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from servers.thin import backends
from servers.thin.batcher import Job, MicroBatcher
from servers.thin.generation import GenJob
from servers.thin.settings import Settings

log = logging.getLogger("thin")


def _err(status: int, message: str, code: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": code, "code": status}}, status_code=status)


class _Rejected(Exception):
    def __init__(self, status: int, message: str):
        self.status, self.message = status, message


def create_app(settings: Settings, backend: backends.Backend | None = None) -> FastAPI:
    state: dict[str, Any] = {"backend": backend, "batcher": None, "ready": False}

    def run(batch: list[Job]) -> None:
        b = state["backend"]
        kind = batch[0].payload["kind"]
        if kind == "gen":
            b.generate([j.payload["job"] for j in batch])
            return
        inputs = [x for j in batch for x in j.payload["inputs"]]
        if kind == "cls":
            probs, ntok = b.classify(inputs)
            results = [{"probs": p, "tokens": n} for p, n in zip(probs, ntok, strict=True)]
        elif kind == "emb":
            vecs, ntok = b.embed(inputs)
            results = [{"embedding": v, "tokens": n} for v, n in zip(vecs, ntok, strict=True)]
        else:
            dets, ntok = b.detect(inputs)
            results = [{**d, "tokens": n} for d, n in zip(dets, ntok, strict=True)]
        k = 0
        for j in batch:
            n = len(j.payload["inputs"])
            j.payload["resolve"](results[k:k + n])
            k += n

    def on_error(batch: list[Job], e: BaseException) -> None:
        log.exception("batch failed: %s", e)
        for j in batch:
            j.payload["fail"](e)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["backend"] is None:
            state["backend"] = await asyncio.to_thread(backends.load, settings)
        state["batcher"] = MicroBatcher(run, max_batch=settings.max_batch, max_wait_ms=settings.max_wait_ms,
                                        on_error=on_error)
        state["ready"] = True
        log.info("ready: %s", json.dumps(state["backend"].describe()))
        yield
        state["ready"] = False
        state["batcher"].close()

    app = FastAPI(title="bench-thin", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def task() -> str:
        return settings.task

    @app.get("/health")
    async def health():
        if not state["ready"]:
            return _err(503, "loading", "unavailable")
        return {"status": "ok", "model": settings.model_id, "batches": state["batcher"].batches,
                **state["backend"].describe()}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": settings.model_id, "object": "model", "owned_by": "bench",
                                            "created": 0, "max_model_len": settings.max_input_tokens}]}

    def _check_model(body: dict) -> None:
        m = body.get("model")
        if m is not None and m != settings.model_id:
            raise _Rejected(404, f"model {m!r} not served here; this server serves {settings.model_id!r}")

    def _parse_completion(body: dict) -> GenJob:
        _check_model(body)
        prompt = body.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str):
            prompt = prompt[0]
        if not isinstance(prompt, str) or not prompt:
            raise _Rejected(400, "prompt must be a non-empty string")
        for unsupported in ("suffix", "logprobs", "best_of", "logit_bias"):
            if body.get(unsupported) not in (None, 0, 1, {}):
                raise _Rejected(400, f"{unsupported} is not supported")
        if body.get("n", 1) != 1 or body.get("echo"):
            raise _Rejected(400, "n must be 1 and echo is not supported")
        if "structured_outputs" in body or "guided_choice" in body:
            raise _Rejected(400, "structured outputs are not supported by the thin server")
        max_tokens = body.get("max_tokens", 16)
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= settings.max_new_tokens_limit:
            raise _Rejected(400, f"max_tokens must be an integer in 1..{settings.max_new_tokens_limit}")
        temperature = float(body.get("temperature", 1.0))
        if temperature < 0:
            raise _Rejected(400, "temperature must be >= 0")
        stop = body.get("stop") or []
        stop = [stop] if isinstance(stop, str) else list(stop)
        min_tokens = max_tokens if body.get("ignore_eos") else int(body.get("min_tokens", 0) or 0)
        return GenJob(prompt=prompt, max_tokens=max_tokens, min_tokens=min(min_tokens, max_tokens),
                      temperature=temperature, top_p=float(body.get("top_p", 1.0)),
                      seed=body.get("seed"), stop=stop)

    @app.post("/v1/completions")
    async def completions(request: Request):
        if task() not in ("seq2seq", "causal"):
            return _err(404, f"/v1/completions is not available for task {task()}", "not_found")
        try:
            body = await request.json()
            job = _parse_completion(body)
        except _Rejected as r:
            return _err(r.status, r.message)
        except (ValueError, TypeError) as e:
            return _err(400, f"bad request: {e}")
        b = state["backend"]
        n_prompt = await asyncio.to_thread(b.count_tokens, job.prompt)
        if n_prompt > settings.max_input_tokens:
            return _err(400, f"prompt has {n_prompt} tokens; the maximum is {settings.max_input_tokens}")
        loop = asyncio.get_running_loop()
        events: asyncio.Queue = asyncio.Queue()
        job.on_text = lambda s: loop.call_soon_threadsafe(events.put_nowait, ("text", s))
        job.on_done = lambda reason, n_out, n_in: loop.call_soon_threadsafe(
            events.put_nowait, ("done", reason, n_out, n_in))
        fail = lambda e: loop.call_soon_threadsafe(events.put_nowait, ("error", str(e)))  # noqa: E731
        state["batcher"].submit(Job({"kind": "gen", "job": job, "fail": fail}, 1, ("gen", *job.sampling_key())))
        rid = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        def chunk(text: str, finish: str | None) -> str:
            return "data: " + json.dumps({"id": rid, "object": "text_completion", "created": created,
                                          "model": settings.model_id,
                                          "choices": [{"index": 0, "text": text, "logprobs": None,
                                                       "finish_reason": finish}]}) + "\n\n"

        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

            async def gen():
                while True:
                    ev = await events.get()
                    if ev[0] == "text":
                        yield chunk(ev[1], None)
                    elif ev[0] == "done":
                        yield chunk("", ev[1])
                        if include_usage:
                            yield "data: " + json.dumps({"id": rid, "object": "text_completion", "created": created,
                                                         "model": settings.model_id, "choices": [],
                                                         "usage": {"prompt_tokens": ev[3], "completion_tokens": ev[2],
                                                                   "total_tokens": ev[2] + ev[3]}}) + "\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    else:
                        yield "data: " + json.dumps({"error": {"message": ev[1], "type": "server_error"}}) + "\n\n"
                        return

            return StreamingResponse(gen(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        parts: list[str] = []
        while True:
            ev = await events.get()
            if ev[0] == "text":
                parts.append(ev[1])
            elif ev[0] == "done":
                return {"id": rid, "object": "text_completion", "created": created, "model": settings.model_id,
                        "choices": [{"index": 0, "text": "".join(parts), "logprobs": None, "finish_reason": ev[1]}],
                        "usage": {"prompt_tokens": ev[3], "completion_tokens": ev[2], "total_tokens": ev[2] + ev[3]}}
            else:
                return _err(500, ev[1], "server_error")

    async def _batched(kind: str, inputs: list) -> list[dict]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        resolve = lambda res: loop.call_soon_threadsafe(fut.set_result, res)  # noqa: E731
        fail = lambda e: loop.call_soon_threadsafe(fut.set_exception, RuntimeError(str(e)))  # noqa: E731
        state["batcher"].submit(Job({"kind": kind, "inputs": inputs, "resolve": resolve, "fail": fail},
                                    len(inputs), (kind,)))
        return await fut

    def _inputs(body: dict, allow_pairs: bool) -> list:
        _check_model(body)
        x = body.get("input")
        if isinstance(x, str):
            x = [x]
        if not isinstance(x, list) or not x:
            raise _Rejected(400, "input must be a string or a non-empty list")
        for v in x:
            ok = isinstance(v, str) and v or (allow_pairs and isinstance(v, list) and len(v) == 2
                                              and all(isinstance(s, str) for s in v))
            if not ok:
                raise _Rejected(400, "each input must be a non-empty string" + (" or a [text, text] pair"
                                                                                   if allow_pairs else ""))
        if allow_pairs and any(isinstance(v, list) for v in x) and not all(isinstance(v, list) for v in x):
            raise _Rejected(400, "mix of strings and pairs in one request")
        if len(x) > 1024:
            raise _Rejected(400, "at most 1024 inputs per request")
        return x

    def _usage(results: list[dict]) -> dict:
        n = int(sum(r["tokens"] for r in results))
        return {"prompt_tokens": n, "total_tokens": n, "completion_tokens": 0, "prompt_tokens_details": None}

    @app.post("/classify")
    async def classify(request: Request):
        if task() != "seq-cls":
            return _err(404, f"/classify is not available for task {task()}", "not_found")
        try:
            inputs = _inputs(await request.json(), allow_pairs=True)
        except _Rejected as r:
            return _err(r.status, r.message)
        except ValueError as e:
            return _err(400, f"bad request: {e}")
        labels = state["backend"].labels
        try:
            results = await _batched("cls", inputs)
        except RuntimeError as e:
            return _err(500, str(e), "server_error")
        data = []
        for i, r in enumerate(results):
            probs = r["probs"]
            k = max(range(len(probs)), key=probs.__getitem__)
            data.append({"index": i, "label": labels[k] if k < len(labels) else str(k), "probs": probs,
                         "num_classes": len(probs)})
        return {"id": f"classify-{uuid.uuid4().hex}", "object": "list", "created": int(time.time()),
                "model": settings.model_id, "data": data, "usage": _usage(results)}

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        """OpenAI-shaped embeddings, the same request and response vLLM's pooling runner serves."""
        if task() != "embed":
            return _err(404, f"/v1/embeddings is not available for task {task()}", "not_found")
        try:
            inputs = _inputs(await request.json(), allow_pairs=False)
        except _Rejected as r:
            return _err(r.status, r.message)
        except ValueError as e:
            return _err(400, f"bad request: {e}")
        try:
            results = await _batched("emb", inputs)
        except RuntimeError as e:
            return _err(500, str(e), "server_error")
        data = [{"index": i, "object": "embedding", "embedding": r["embedding"]} for i, r in enumerate(results)]
        return {"id": f"embd-{uuid.uuid4().hex}", "object": "list", "created": int(time.time()),
                "model": settings.model_id, "data": data, "usage": _usage(results)}

    @app.post("/detect")
    async def detect(request: Request):
        if task() != "token-cls":
            return _err(404, f"/detect is not available for task {task()}", "not_found")
        try:
            inputs = _inputs(await request.json(), allow_pairs=False)
        except _Rejected as r:
            return _err(r.status, r.message)
        except ValueError as e:
            return _err(400, f"bad request: {e}")
        try:
            results = await _batched("det", inputs)
        except RuntimeError as e:
            return _err(500, str(e), "server_error")
        data = [{"index": i, "spans": r["spans"], "category_max": r["category_max"]} for i, r in enumerate(results)]
        return {"id": f"detect-{uuid.uuid4().hex}", "object": "list", "created": int(time.time()),
                "model": settings.model_id, "data": data, "usage": _usage(results),
                "decoder": {"type": "viterbi-bioes", "biases": state["backend"].viterbi_biases}}

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    s = Settings.from_env()
    uvicorn.run(create_app(s), host=s.host, port=s.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
