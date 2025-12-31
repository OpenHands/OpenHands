from __future__ import annotations

from typing import Protocol, TypeGuard

from litellm import ChatCompletionToolParam, Message as LiteLLMMessage
from litellm.types.utils import Choices, ModelResponse, StreamingChoices

from openhands.sdk.llm.exceptions import LLMNoResponseError
from openhands.sdk.llm.mixins.fn_call_converter import (
    STOP_WORDS,
    convert_fncall_messages_to_non_fncall_messages,
    convert_non_fncall_messages_to_fncall_messages,
)
from openhands.sdk.llm.utils.model_features import get_features


class _HostSupports(Protocol):
    model: str
    disable_stop_word: bool | None
    native_tool_calling: bool


class NonNativeToolCallingMixin:
    """Mixin providing prompt-mocked tool-calling support when native FC is off.

    Host requirements:
    - self.model: str
    - self.disable_stop_word: bool | None
    - self.native_tool_calling -> bool
    """

    def should_mock_tool_calls(
        self: _HostSupports, tools: list[ChatCompletionToolParam] | None
    ) -> bool:
        return bool(tools) and not self.native_tool_calling

    def pre_request_prompt_mock(
        self: _HostSupports,
        messages: list[dict],
        tools: list[ChatCompletionToolParam],
        kwargs: dict,
    ) -> tuple[list[dict], dict]:
        """Convert to non-fncall prompting when native tool-calling is off."""
        # Skip in-context learning examples for models that understand the format
        # or have limited context windows
        add_iclex = not any(
            s in self.model for s in ("openhands-lm", "devstral", "nemotron")
        )
        messages = convert_fncall_messages_to_non_fncall_messages(
            messages, tools, add_in_context_learning_example=add_iclex
        )
        if get_features(self.model).supports_stop_words and not self.disable_stop_word:
            kwargs = dict(kwargs)
            kwargs["stop"] = STOP_WORDS

        # Ensure we don't send tool_choice when mocking
        kwargs.pop("tool_choice", None)
        return messages, kwargs

    def post_response_prompt_mock(
        self: _HostSupports,
        resp: ModelResponse,
        nonfncall_msgs: list[dict],
        tools: list[ChatCompletionToolParam],
    ) -> ModelResponse:
        if len(resp.choices) < 1:
            raise LLMNoResponseError(
                "Response choices is less than 1 (seen in some providers). Resp: "
                + str(resp)
            )

        def _all_choices(
            items: list[Choices | StreamingChoices],
        ) -> TypeGuard[list[Choices]]:
            return all(isinstance(c, Choices) for c in items)

        if not _all_choices(resp.choices):
            raise AssertionError(
                "Expected non-streaming Choices when post-processing mocked tools"
            )

        # Preserve provider-specific reasoning fields before conversion
        orig_msg = resp.choices[0].message
        non_fn_message: dict = orig_msg.model_dump()
        fn_msgs: list[dict] = convert_non_fncall_messages_to_fncall_messages(
            nonfncall_msgs + [non_fn_message], tools
        )
        last: dict = fn_msgs[-1]

        for name in ("reasoning_content", "provider_specific_fields"):
            val = getattr(orig_msg, name, None)
            if not val:
                continue
            last[name] = val

        resp.choices[0].message = LiteLLMMessage.model_validate(last)
        return resp
