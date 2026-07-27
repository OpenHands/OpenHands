"""Mock OpenAI-compatible LLM server powered by openhands-sdk TestLLM.

Serves scripted trajectories as OpenAI /v1/chat/completions responses.
The agent-server's litellm layer talks to this instead of a real LLM provider.

Usage:
    python mock-llm-server.py [--port PORT]

The server defines a single trajectory: one terminal tool call followed by a
text reply. Extend TRAJECTORY to test richer scenarios (multi-turn, errors, etc).
"""

import json
import os
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"

from openhands.sdk.llm import Message, MessageToolCall, TextContent
from openhands.sdk.llm.exceptions import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMContextWindowExceedError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from openhands.sdk.testing import TestLLM, TestLLMExhaustedError

BASH_TOKEN = "MOCK_LLM_E2E_BASH_OK"
REPLY_TOKEN = "MOCK_LLM_E2E_REPLY_OK"

# SDK exception → (HTTP status, OpenAI error type)
ERROR_MAP: dict[type, tuple[int, str]] = {
    LLMAuthenticationError: (401, "invalid_api_key"),
    LLMRateLimitError: (429, "rate_limit_exceeded"),
    LLMContextWindowExceedError: (400, "context_length_exceeded"),
    LLMBadRequestError: (400, "invalid_request_error"),
    LLMTimeoutError: (408, "timeout"),
    LLMServiceUnavailableError: (503, "server_error"),
}


def build_trajectory() -> list[Message | Exception]:
    """Build the scripted trajectory for the E2E test.

    Turn 1: Agent calls the terminal tool with a printf command.
    Turn 2: Agent replies with the expected token and finishes.
    """
    return [
        Message(
            role="assistant",
            content=[TextContent(text="")],
            tool_calls=[
                MessageToolCall(
                    id="call_mock_001",
                    name="terminal",
                    arguments=json.dumps(
                        {"command": f"printf '{BASH_TOKEN}\\n'"}
                    ),
                    origin="completion",
                )
            ],
        ),
        Message(
            role="assistant",
            content=[TextContent(text=REPLY_TOKEN)],
        ),
    ]


class MockLLMHandler(BaseHTTPRequestHandler):
    test_llm: TestLLM  # set by serve()
    # Named trajectories that tests can register via the admin API and then
    # activate with POST /admin/trajectory/activate.
    _named_trajectories: dict[str, list[Message | Exception]] = {}
    # All completion request bodies since the last /admin/reset.
    # Tests read them via GET /admin/requests to verify image / content details.
    # Stored as a list so assertions survive even when the agent-server makes
    # multiple LLM calls (e.g., internal condenser calls after the main turn).
    _completion_requests: list = []
    _lock = threading.Lock()

    def do_GET(self):
        """Health check and admin read endpoints."""
        path = self.path.rstrip("/").split("?")[0]
        if path == "/admin/requests":
            with self._lock:
                payload = list(MockLLMHandler._completion_requests)
            self._send_json(200, {"requests": payload})
            return
        # Default: health check — Playwright's webServer probes GET / to detect readiness.
        self._send_json(200, {"status": "ok", "server": "mock-llm"})

    def do_POST(self):
        path = self.path.rstrip("/")

        # ── Admin API: reset trajectory to default ──
        if path == "/admin/reset":
            with self._lock:
                MockLLMHandler.test_llm = TestLLM.from_messages(build_trajectory())
                MockLLMHandler._named_trajectories.clear()
                MockLLMHandler._completion_requests.clear()
                remaining = MockLLMHandler.test_llm.remaining_responses
            self._send_json(200, {
                "status": "reset",
                "remaining": remaining,
            })
            return

        # ── Admin API: register a named trajectory ──
        if path == "/admin/trajectory/register":
            body = self._read_body()
            if body is None:
                return  # error response already sent
            name = body.get("name", "")
            raw_turns = body.get("turns", [])
            if not name or not raw_turns:
                self._send_error(400, "bad_request", "need 'name' and 'turns'")
                return
            try:
                messages = _parse_trajectory_turns(raw_turns)
            except ValueError as exc:
                self._send_error(400, "bad_request", str(exc))
                return
            with self._lock:
                MockLLMHandler._named_trajectories[name] = messages
            self._send_json(200, {"status": "registered", "name": name, "turns": len(messages)})
            return

        # ── Admin API: activate a named trajectory ──
        if path == "/admin/trajectory/activate":
            body = self._read_body()
            if body is None:
                return  # error response already sent
            name = body.get("name", "")
            with self._lock:
                msgs = MockLLMHandler._named_trajectories.get(name)
            if msgs is None:
                self._send_error(404, "not_found", f"trajectory '{name}' not registered")
                return
            with self._lock:
                MockLLMHandler.test_llm = TestLLM.from_messages(list(msgs))
                remaining = MockLLMHandler.test_llm.remaining_responses
            self._send_json(200, {
                "status": "activated",
                "name": name,
                "remaining": remaining,
            })
            return

        # ── Reject unknown paths with a clear 404 ──
        COMPLETION_PATHS = ("/v1/chat/completions", "/chat/completions", "/completions", "")
        if path not in COMPLETION_PATHS:
            self._send_error(404, "not_found", f"Unknown path: {path}")
            return

        # ── Normal chat completion ──
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        # Append to request history for test verification.
        # Tests can GET /admin/requests to confirm image content was included.
        with self._lock:
            MockLLMHandler._completion_requests.append(body)

        try:
            response = self.test_llm.completion([])
        except TestLLMExhaustedError:
            self._send_error(
                500,
                "server_error",
                f"Mock LLM exhausted after {self.test_llm.call_count} calls",
            )
            return
        except tuple(ERROR_MAP.keys()) as exc:
            status, error_type = ERROR_MAP[type(exc)]
            self._send_error(status, error_type, str(exc))
            return

        raw = response.raw_response.model_dump()

        if body.get("stream"):
            stream_options = body.get("stream_options") or {}
            self._send_streaming(
                raw, include_usage=bool(stream_options.get("include_usage"))
            )
        else:
            self._send_json(200, raw)

    def _send_streaming(self, raw: dict, include_usage: bool = False):
        """SSE streaming: emit content chunk + finish chunk + [DONE]."""
        choice = raw["choices"][0]
        message = choice["message"]
        base = {
            "id": raw["id"],
            "object": "chat.completion.chunk",
            "created": raw.get("created", int(time.time())),
            "model": raw["model"],
        }

        finish_reason = "stop"
        tool_calls = message.get("tool_calls") or []
        chunks = []
        if tool_calls:
            finish_reason = "tool_calls"
            chunks.append(
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": message.get("role", "assistant")},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            for i, tool_call in enumerate(tool_calls):
                function = tool_call.get("function", {})
                chunks.append(
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": i,
                                            "id": tool_call.get("id"),
                                            "type": tool_call.get("type", "function"),
                                            "function": {
                                                "name": function.get("name"),
                                                "arguments": "",
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                if function.get("arguments"):
                    chunks.append(
                        {
                            **base,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": i,
                                                "function": {
                                                    "arguments": function["arguments"]
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
        else:
            chunks.append(
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": message.get("role", "assistant"),
                                "content": message.get("content") or "",
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )

        finish_chunk = {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for chunk in [*chunks, finish_chunk]:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        if include_usage:
            usage_chunk = {
                **base,
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
            self.wfile.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, error_type: str, message: str):
        self._send_json(
            status,
            {"error": {"message": message, "type": error_type, "code": error_type}},
        )

    def _read_body(self) -> dict | None:
        """Read and parse JSON body. Returns None (after sending 400) on parse failure."""
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            self._send_error(400, "invalid_json", str(exc))
            return None

    def log_message(self, format, *args):
        print(f"[mock-llm] {args[0]}", file=sys.stderr, flush=True)


def _parse_trajectory_turns(raw_turns: list[dict]) -> list[Message | Exception]:
    """Convert JSON turn descriptors into Message objects.

    Each turn is a dict with either:
      - {"tool_call": {"name": ..., "arguments": ...}}  → tool-call message
      - {"text": "..."}  → text reply message
    """
    messages: list[Message | Exception] = []
    for i, turn in enumerate(raw_turns):
        if "tool_call" in turn:
            tc = turn["tool_call"]
            messages.append(
                Message(
                    role="assistant",
                    content=[TextContent(text="")],
                    tool_calls=[
                        MessageToolCall(
                            id=f"call_dyn_{i:03d}",
                            name=tc["name"],
                            arguments=(
                                json.dumps(tc["arguments"])
                                if isinstance(tc["arguments"], dict)
                                else tc["arguments"]
                            ),
                            origin="completion",
                        )
                    ],
                )
            )
        elif "text" in turn:
            messages.append(
                Message(
                    role="assistant",
                    content=[TextContent(text=turn["text"])],
                )
            )
        else:
            raise ValueError(
                f"[mock-llm] turn {i} has neither 'tool_call' nor 'text': {turn!r}"
            )
    return messages


def serve(port: int = 9999):
    test_llm = TestLLM.from_messages(build_trajectory())
    MockLLMHandler.test_llm = test_llm

    server = HTTPServer(("127.0.0.1", port), MockLLMHandler)
    print(f"Mock LLM server ready on http://127.0.0.1:{port}", flush=True)
    print(f"Trajectory: {test_llm.remaining_responses} scripted turns", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mock OpenAI LLM server")
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()
    serve(args.port)
