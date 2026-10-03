"""Regression tests for the mock server's profile-validation boundary."""

import importlib.util
import json
from pathlib import Path
import threading
import unittest
from urllib.request import Request, urlopen


spec = importlib.util.spec_from_file_location(
    "mock_llm_server", Path(__file__).with_name("mock-llm-server.py")
)
server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_module)


class ProfileValidationTests(unittest.TestCase):
    def setUp(self):
        handler = server_module.MockLLMHandler
        handler.test_llm = server_module.TestLLM.from_messages(
            server_module.build_trajectory()
        )
        handler._completion_requests = []
        self.server = server_module.HTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def complete(self, messages, max_tokens=100):
        request = Request(
            f"{self.url}/v1/chat/completions",
            data=json.dumps(
                {"model": "test", "messages": messages, "max_tokens": max_tokens}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            return json.load(response)["choices"][0]["message"]

    def test_validation_pings_leave_the_conversation_trajectory_intact(self):
        for as_parts in (False, True):
            def content(text):
                return [{"type": "text", "text": text}] if as_parts else text

            ping = {"role": "user", "content": content("ping")}
            system = {"role": "system", "content": content("Reply with one token.")}
            for messages in ([ping], [system, ping]):
                with self.subTest(as_parts=as_parts, messages=len(messages)):
                    response = self.complete(messages, max_tokens=1)
                    self.assertEqual(response["content"], "pong")

        with urlopen(f"{self.url}/admin/requests", timeout=5) as response:
            self.assertEqual(json.load(response)["requests"], [])

        messages = [{"role": "user", "content": "Run the scripted command."}]
        tool_call = self.complete(messages)["tool_calls"][0]["function"]
        self.assertIn(server_module.BASH_TOKEN, tool_call["arguments"])
        self.assertEqual(
            self.complete(messages)["content"], server_module.REPLY_TOKEN
        )

    def test_one_token_conversation_is_not_mistaken_for_validation(self):
        messages = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "ping"},
        ]
        self.assertIn("tool_calls", self.complete(messages, max_tokens=1))
        with urlopen(f"{self.url}/admin/requests", timeout=5) as response:
            self.assertEqual(json.load(response)["requests"][0]["messages"], messages)


if __name__ == "__main__":
    unittest.main()
