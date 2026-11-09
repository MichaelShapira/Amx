"""OpenAI-shaped fake server with scripted delays, for client timing tests."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Script:
    ttft_ms = 50.0
    itl_ms = 10.0
    tokens = 10
    tokens_per_event = 1
    prompt_tokens = 42
    status = 200
    omit_usage = False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    script = Script

    def log_message(self, *args):  # silence test output
        pass

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def do_GET(self):  # noqa: N802
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n) or b"{}")
        s = self.script
        if s.status != 200:
            body = b'{"error":"scripted"}'
            self.send_response(s.status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.endswith("/classify"):
            inputs = req.get("input") or []
            time.sleep(s.ttft_ms / 1000)
            out = {"data": [{"index": i, "label": "ENTAILMENT", "probs": [0.1, 0.2, 0.7]}
                            for i, _ in enumerate(inputs)],
                   "usage": {"prompt_tokens": s.prompt_tokens * len(inputs)}}
            body = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        time.sleep(s.ttft_ms / 1000)
        sent = 0
        first = True
        while sent < s.tokens:
            if not first:
                time.sleep(s.itl_ms * s.tokens_per_event / 1000)
            k = min(s.tokens_per_event, s.tokens - sent)
            ev = {"choices": [{"index": 0, "text": "tok " * k}], "usage": None}
            self._chunk(f"data: {json.dumps(ev)}\n\n".encode())
            sent += k
            first = False
        if not s.omit_usage:
            ev = {"choices": [], "usage": {"prompt_tokens": s.prompt_tokens,
                                           "completion_tokens": s.tokens}}
            self._chunk(f"data: {json.dumps(ev)}\n\n".encode())
        self._chunk(b"data: [DONE]\n\n")
        self._chunk(b"")


def start() -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"
