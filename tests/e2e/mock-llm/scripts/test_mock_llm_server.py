"""Run with: <mock-LLM Python> -m unittest discover -s tests/e2e/mock-llm/scripts -p 'test_*.py'."""

import importlib.util
import json
from pathlib import Path
import threading
import unittest
from urllib.request import Request, urlopen
from http.server import HTTPServer

spec = importlib.util.spec_from_file_location("mock_llm_server", Path(__file__).with_name("mock-llm-server.py"))
server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_module)


class TitleIsolationTest(unittest.TestCase):
    def test_titles_do_not_consume_conversation_trajectory(self):
        server = HTTPServer(("127.0.0.1", 0), server_module.MockLLMHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_port}"

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


if __name__ == "__main__":
    unittest.main()
