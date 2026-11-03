"""Regression tests for the mock server's profile-validation boundary."""

import importlib.util
import json
from pathlib import Path
import threading
from urllib.request import Request, urlopen

import pytest


spec = importlib.util.spec_from_file_location(
    "mock_llm_server", Path(__file__).with_name("mock-llm-server.py")
)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


@pytest.fixture
def url():
    handler = server.MockLLMHandler
    handler.test_llm = server.TestLLM.from_messages(server.build_trajectory())
    handler._completion_requests = []
    httpd = server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def text(value, as_parts):
    return [{"type": "text", "text": value}] if as_parts else value


def complete(url, messages, max_tokens=100):
    request = Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(
            {"model": "test", "messages": messages, "max_tokens": max_tokens}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=5) as response:
        return json.load(response)["choices"][0]["message"]


def recorded(url):
    with urlopen(f"{url}/admin/requests", timeout=5) as response:
        return json.load(response)["requests"]


@pytest.mark.parametrize("as_parts", [False, True], ids=["string", "parts"])
@pytest.mark.parametrize("with_system", [False, True], ids=["user-only", "system+user"])
def test_validation_ping_leaves_the_trajectory_intact(url, as_parts, with_system):
    messages = [{"role": "user", "content": text("ping", as_parts)}]
    if with_system:
        system = {"role": "system", "content": text("Reply with one token.", as_parts)}
        messages.insert(0, system)

    assert complete(url, messages, max_tokens=1)["content"] == "pong"
    assert recorded(url) == []

    conversation = [{"role": "user", "content": "Run the scripted command."}]
    tool_call = complete(url, conversation)["tool_calls"][0]["function"]
    assert server.BASH_TOKEN in tool_call["arguments"]
    assert complete(url, conversation)["content"] == server.REPLY_TOKEN


def test_one_token_conversation_is_not_mistaken_for_validation(url):
    messages = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "ping"},
    ]

    assert "tool_calls" in complete(url, messages, max_tokens=1)
    assert recorded(url)[0]["messages"] == messages
