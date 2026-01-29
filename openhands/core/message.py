# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from enum import Enum
from typing import Any, Literal

from litellm import ChatCompletionMessageToolCall
from pydantic import BaseModel, Field, model_serializer


class ContentType(Enum):
    TEXT = 'text'
    IMAGE_URL = 'image_url'


class Content(BaseModel):
    type: str
    cache_prompt: bool = False

    @model_serializer(mode='plain')
    def serialize_model(
        self,
    ) -> dict[str, str | dict[str, str]] | list[dict[str, str | dict[str, str]]]:
        raise NotImplementedError('Subclasses should implement this method.')


class TextContent(Content):
    type: str = ContentType.TEXT.value
    text: str

    @model_serializer(mode='plain')
    def serialize_model(self) -> dict[str, str | dict[str, str]]:
        data: dict[str, str | dict[str, str]] = {
            'type': self.type,
            'text': self.text,
        }
        if self.cache_prompt:
            data['cache_control'] = {'type': 'ephemeral'}
        return data


class ImageContent(Content):
    type: str = ContentType.IMAGE_URL.value
    image_urls: list[str]

    @model_serializer(mode='plain')
    def serialize_model(self) -> list[dict[str, str | dict[str, str]]]:
        images: list[dict[str, str | dict[str, str]]] = []
        for url in self.image_urls:
            images.append({'type': self.type, 'image_url': {'url': url}})
        if self.cache_prompt and images:
            images[-1]['cache_control'] = {'type': 'ephemeral'}
        return images


class Message(BaseModel):
    # NOTE: this is not the same as EventSource
    # These are the roles in the LLM's APIs
    role: Literal['user', 'system', 'assistant', 'tool']
    content: list[TextContent | ImageContent] = Field(default_factory=list)
    cache_enabled: bool = False
    vision_enabled: bool = False
    # function calling
    function_calling_enabled: bool = False
    # - tool calls (from LLM)
    tool_calls: list[ChatCompletionMessageToolCall] | None = None
    # - tool execution result (to LLM)
    tool_call_id: str | None = None
    name: str | None = None  # name of the tool
    # force string serializer
    force_string_serializer: bool = False

    @property
    def contains_image(self) -> bool:
        return any(isinstance(content, ImageContent) for content in self.content)

    @model_serializer(mode='plain')
    def serialize_model(self) -> dict[str, Any]:
        # We need two kinds of serializations:
        # - into a single string: for providers that don't support list of content items (e.g. no vision, no tool calls)
        # - into a list of content items: the new APIs of providers with vision/prompt caching/tool calls
        # NOTE: remove this when litellm or providers support the new API
        if not self.force_string_serializer and (
            self.cache_enabled or self.vision_enabled or self.function_calling_enabled
        ):
            return self._list_serializer()
        # some providers, like HF and Groq/llama, don't support a list here, but a single string
        return self._string_serializer()

    def _string_serializer(self) -> dict[str, Any]:
        # convert content to a single string
        content = '\n'.join(
            item.text for item in self.content if isinstance(item, TextContent)
        )
        message_dict: dict[str, Any] = {'content': content, 'role': self.role}

        # add tool call keys if we have a tool call or response
        return self._add_tool_call_keys(message_dict)

    def _list_serializer(self) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        role_tool_with_prompt_caching = False
        for item in self.content:
            d = item.model_dump()
            # We have to remove cache_prompt for tool content and move it up to the message level
            # See discussion here for details: https://github.com/BerriAI/litellm/issues/6422#issuecomment-2438765472
            if self.role == 'tool' and item.cache_prompt:
                role_tool_with_prompt_caching = True
                if isinstance(item, TextContent):
                    d.pop('cache_control', None)
                elif isinstance(item, ImageContent):
                    # ImageContent.model_dump() always returns a list
                    # We know d is a list of dicts for ImageContent
                    if hasattr(d, '__iter__'):
                        for d_item in d:
                            if hasattr(d_item, 'pop'):
                                d_item.pop('cache_control', None)

            if isinstance(item, TextContent):
                content.append(d)
            elif isinstance(item, ImageContent) and self.vision_enabled:
                # ImageContent.model_dump() always returns a list
                # We know d is a list for ImageContent
                content.extend([d] if isinstance(d, dict) else d)

        message_dict: dict[str, Any] = {'content': content, 'role': self.role}

        if role_tool_with_prompt_caching:
            message_dict['cache_control'] = {'type': 'ephemeral'}

        # add tool call keys if we have a tool call or response
        return self._add_tool_call_keys(message_dict)

    def _add_tool_call_keys(self, message_dict: dict[str, Any]) -> dict[str, Any]:
        """Add tool call keys if we have a tool call or response.

        NOTE: this is necessary for both native and non-native tool calling
        """
        # an assistant message calling a tool
        if self.tool_calls is not None:
            message_dict['tool_calls'] = [
                {
                    'id': tool_call.id,
                    'type': 'function',
                    'function': {
                        'name': tool_call.function.name,
                        'arguments': tool_call.function.arguments,
                    },
                }
                for tool_call in self.tool_calls
            ]

        # an observation message with tool response
        if self.tool_call_id is not None:
            assert self.name is not None, (
                'name is required when tool_call_id is not None'
            )
            message_dict['tool_call_id'] = self.tool_call_id
            message_dict['name'] = self.name

        return message_dict
