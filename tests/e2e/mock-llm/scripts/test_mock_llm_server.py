"""Regression tests for the mock server's profile-validation boundary."""

import importlib.util
import json
from pathlib import Path
import threading
import unittest
from http.server import HTTPServer
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


class TitleIsolationTest(unittest.TestCase):
    def test_titles_do_not_consume_conversation_trajectory(self):
        httpd = HTTPServer(("127.0.0.1", 0), server.MockLLMHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(httpd.shutdown)
        url = f"http://127.0.0.1:{httpd.server_port}"

        def post(path, body):
            with urlopen(
                Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
            ) as response:
                return response.read().decode()

        post("/admin/reset", {})
        post(
            "/admin/trajectory/register",
            {"name": "owned", "turns": [{"text": "FIRST_AGENT_REPLY"}, {"text": "SECOND_AGENT_REPLY"}]},
        )
        post("/admin/trajectory/activate", {"name": "owned"})
        title = {
            "model": "mock",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant that generates concise, "
                            "descriptive titles for conversations with OpenHands.",
                        }
                    ],
                },
                {"role": "user", "content": "Generate a title (maximum 50 characters) for this message"},
            ],
        }
        self.assertEqual(
            json.loads(post("/v1/chat/completions", title))["choices"][0]["message"]["content"], "Mock conversation"
        )
        reply = json.loads(post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Please respond"}]}))
        self.assertEqual(reply["choices"][0]["message"]["content"], "FIRST_AGENT_REPLY")
        self.assertIn("Mock conversation", post("/v1/chat/completions", {**title, "stream": True}))
        # The same words inside a normal agent/tool request are not housekeeping.
        reply = json.loads(post("/v1/chat/completions", {**title, "tools": [{"type": "function"}]}))
        self.assertEqual(reply["choices"][0]["message"]["content"], "SECOND_AGENT_REPLY")
        with urlopen(url + "/admin/requests") as response:
            self.assertEqual(len(json.load(response)["requests"]), 2)


@pytest.mark.parametrize("content", [None, [], [{"type": "image_url", "image_url": {"url": "unused"}}]])
@pytest.mark.parametrize("message_index", [0, 1], ids=["system", "user"])
def test_title_detection_rejects_missing_text(content, message_index):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates concise, descriptive titles for conversations with OpenHands."},
        {"role": "user", "content": "Generate a title (maximum 50 characters) for this message"},
    ]
    messages[message_index]["content"] = content

    assert server._is_title_request({"messages": messages}) is False
