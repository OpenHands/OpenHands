"""Run the repository's mock LLM server, adding token usage to completions.

Evidence-only helper (not part of any PR). The repository mock server returns
no usage for non-streaming completions, so the agent server would record 0
tokens per call. This wrapper derives deterministic counts from the request
and response sizes (about 4 characters per token) so per-call token records
differ, the same way a real provider's would.

Usage:
    python mock_llm_with_usage.py <path/to/mock-llm-server.py> [--port PORT]
"""

import importlib.util
import io
import json
import math
import sys


def load_mock_module(path: str):
    spec = importlib.util.spec_from_file_location("mock_llm_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def estimate_usage(request_body: dict, raw: dict) -> dict:
    prompt_chars = len(json.dumps(request_body.get("messages", [])))
    prompt_chars += len(json.dumps(request_body.get("tools", [])))
    message = (raw.get("choices") or [{}])[0].get("message") or {}
    completion_chars = len(message.get("content") or "")
    completion_chars += len(json.dumps(message.get("tool_calls") or []))
    prompt_tokens = math.ceil(prompt_chars / 4)
    completion_tokens = math.ceil(completion_chars / 4) + 12
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def main() -> None:
    module = load_mock_module(sys.argv[1])
    handler = module.MockLLMHandler
    original_do_post = handler.do_POST
    original_send_json = handler._send_json
    original_send_streaming = handler._send_streaming

    def do_post(self):
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length else b""
        try:
            self._usage_request_body = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError:
            self._usage_request_body = {}
        self.rfile = io.BytesIO(raw_body)
        original_do_post(self)

    def send_json(self, status, payload):
        if status == 200 and isinstance(payload, dict) and payload.get("choices"):
            payload = {
                **payload,
                "usage": estimate_usage(
                    getattr(self, "_usage_request_body", {}), payload
                ),
            }
        original_send_json(self, status, payload)

    def send_streaming(self, raw, include_usage=False):
        usage = estimate_usage(getattr(self, "_usage_request_body", {}), raw)
        real_write = self.wfile.write

        def write(data: bytes):
            text = data.decode()
            if '"usage"' in text and text.startswith("data: "):
                chunk = json.loads(text[len("data: ") :])
                chunk["usage"] = usage
                data = f"data: {json.dumps(chunk)}\n\n".encode()
            return real_write(data)

        self.wfile.write = write
        try:
            original_send_streaming(self, raw, include_usage=include_usage)
        finally:
            self.wfile.write = real_write

    handler.do_POST = do_post
    handler._send_json = send_json
    handler._send_streaming = send_streaming

    port = 9999
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    module.serve(port)


if __name__ == "__main__":
    main()
