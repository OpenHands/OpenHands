from __future__ import annotations

import unittest
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

from openhands.agenthub.gemini_native_codeact_agent.gemini_native_codeact_agent import (
    GeminiNativeCodeActAgent,
)
from openhands.events.observation.mcp import MCPImage, MCPObservation


# -------------------------
# Minimal stubs for google.genai.types
# -------------------------


@dataclass
class _FunctionResponseBlob:
    mime_type: str
    data: str


@dataclass
class _FunctionResponsePart:
    inline_data: _FunctionResponseBlob


@dataclass
class _FunctionResponse:
    name: str
    response: dict
    parts: list[_FunctionResponsePart] | None = None


@dataclass
class _Part:
    function_response: _FunctionResponse


@dataclass
class _Content:
    role: str
    parts: list[_Part]


def _make_agent(*, multimodal: bool, pending_names: list[str], batch_remaining: int) -> GeminiNativeCodeActAgent:
    """Construct an agent instance without running __init__ (no external deps)."""
    agent = GeminiNativeCodeActAgent.__new__(GeminiNativeCodeActAgent)
    agent._types = SimpleNamespace(  # type: ignore[attr-defined]
        FunctionResponseBlob=_FunctionResponseBlob,
        FunctionResponsePart=_FunctionResponsePart,
        FunctionResponse=_FunctionResponse,
        Part=_Part,
        Content=_Content,
    )
    agent._use_multimodal_function_response = multimodal  # type: ignore[attr-defined]
    agent._pending_fc_names = deque(pending_names)  # type: ignore[attr-defined]
    agent._batch_remaining = batch_remaining  # type: ignore[attr-defined]
    agent._batch_response_parts = []  # type: ignore[attr-defined]
    agent._history = []  # type: ignore[attr-defined]
    agent._last_history_len = 0  # type: ignore[attr-defined]
    return agent


class TestGeminiNativeCodeActAgentToolLoop(unittest.TestCase):
    def test_ingest_batches_function_responses_and_includes_text_content(self) -> None:
        agent = _make_agent(multimodal=False, pending_names=["computer", "text_search"], batch_remaining=2)

        obs1 = MCPObservation(
            content="PAGE_STATE_TEXT",
            images=[MCPImage(data="AAA", mime_type="image/png")],
            is_error=False,
        )
        obs2 = MCPObservation(content='[{"id": 1}]', images=[], is_error=True)

        agent._ingest_new_tool_observations([obs1, obs2])  # type: ignore[attr-defined]

        # One batched Content(role="model") with two function_response parts
        self.assertEqual(len(agent._history), 1)  # type: ignore[attr-defined]
        content = agent._history[0]  # type: ignore[attr-defined]
        self.assertEqual(content.role, "model")
        self.assertEqual(len(content.parts), 2)

        fr1 = content.parts[0].function_response
        self.assertEqual(fr1.name, "computer")
        self.assertEqual(fr1.response["status"], "success")
        self.assertEqual(fr1.response["content"], "PAGE_STATE_TEXT")
        # Full history stores image parts for Fleet UI even in API-key mode
        self.assertIsNotNone(fr1.parts)
        self.assertEqual(len(fr1.parts or []), 1)
        self.assertEqual(fr1.parts[0].inline_data.mime_type, "image/png")  # type: ignore[index]
        self.assertEqual(fr1.parts[0].inline_data.data, "AAA")  # type: ignore[index]

        fr2 = content.parts[1].function_response
        self.assertEqual(fr2.name, "text_search")
        self.assertEqual(fr2.response["status"], "error")
        self.assertEqual(fr2.response["content"], '[{"id": 1}]')

        # Sanitized history strips images for Gemini API-key requests
        sanitized = agent._build_request_history_for_gemini()  # type: ignore[attr-defined]
        self.assertEqual(len(sanitized), 1)
        s_fr1 = sanitized[0].parts[0].function_response
        self.assertIsNone(s_fr1.parts)
        self.assertEqual(s_fr1.response["content"], "PAGE_STATE_TEXT")

    def test_ingest_multimodal_attaches_first_valid_image(self) -> None:
        agent = _make_agent(multimodal=True, pending_names=["computer"], batch_remaining=1)

        # First image has empty data -> should skip to the next image with real data
        obs = MCPObservation(
            content="STATE",
            images=[
                MCPImage(data="", mime_type="image/png"),
                MCPImage(data="BBB", mime_type="image/jpeg"),
            ],
            is_error=False,
        )

        agent._ingest_new_tool_observations([obs])  # type: ignore[attr-defined]

        self.assertEqual(len(agent._history), 1)  # type: ignore[attr-defined]
        fr = agent._history[0].parts[0].function_response  # type: ignore[attr-defined]
        self.assertEqual(fr.name, "computer")
        self.assertEqual(fr.response["status"], "success")
        self.assertEqual(fr.response["content"], "STATE")
        self.assertIsNotNone(fr.parts)
        self.assertEqual(len(fr.parts or []), 1)
        self.assertEqual(fr.parts[0].inline_data.mime_type, "image/jpeg")  # type: ignore[index]
        self.assertEqual(fr.parts[0].inline_data.data, "BBB")  # type: ignore[index]

        # When multimodal is enabled, sanitized history still strips image parts (by design),
        # but production only uses sanitizer in API-key mode.
        # Here, multimodal=True => request history is full history (no stripping)
        request_history = agent._build_request_history_for_gemini()  # type: ignore[attr-defined]
        self.assertIsNotNone(request_history[0].parts[0].function_response.parts)

    def test_ingest_skips_when_not_in_pending_batch(self) -> None:
        agent = _make_agent(multimodal=False, pending_names=[], batch_remaining=0)

        obs = MCPObservation(content="STATE", images=[], is_error=False)
        agent._ingest_new_tool_observations([obs])  # type: ignore[attr-defined]

        # No pending tool calls => ignore observation as a tool response
        self.assertEqual(agent._history, [])  # type: ignore[attr-defined]

