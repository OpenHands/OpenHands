import json
import types
from collections.abc import Sequence
from typing import (
    Annotated,
    Any,
    Union,
    get_args,
    get_origin,
    overload,
)

from openhands.sdk.context.condenser.base import CondenserBase
from openhands.sdk.context.view import View
from openhands.sdk.conversation.types import ConversationTokenCallbackType
from openhands.sdk.event.base import Event, LLMConvertibleEvent
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.llm import LLM, LLMResponse, Message
from openhands.sdk.tool import Action, ToolDefinition


def fix_malformed_tool_arguments(
    arguments: dict[str, Any], action_type: type[Action]
) -> dict[str, Any]:
    """Fix malformed tool arguments by decoding JSON strings for list/dict fields.

    This function handles cases where certain LLMs (such as GLM 4.6) incorrectly
    encode array/object parameters as JSON strings when using native function calling.

    Example raw LLM output from GLM 4.6:
    {
        "role": "assistant",
        "content": "I'll view the file for you.",
        "tool_calls": [{
            "id": "call_ef8e",
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "arguments": '{
                    "command": "view",
                    "path": "/tmp/test.txt",
                    "view_range": "[1, 5]"
                }'
            }
        }]
    }

    Expected output: `"view_range" : [1, 5]`

    Note: The arguments field is a JSON string. When decoded, view_range is
    incorrectly a string "[1, 5]" instead of the proper array [1, 5].
    This function automatically fixes this by detecting that view_range
    expects a list type and decoding the JSON string to get the actual array.

    Args:
        arguments: The parsed arguments dict from json.loads(tool_call.arguments).
        action_type: The action type that defines the expected schema.

    Returns:
        The arguments dict with JSON strings decoded where appropriate.
    """
    if not isinstance(arguments, dict):
        return arguments

    fixed_arguments = arguments.copy()

    # Use model_fields to properly handle aliases and inherited fields
    for field_name, field_info in action_type.model_fields.items():
        # Check both the field name and its alias (if any)
        data_key = field_info.alias if field_info.alias else field_name
        if data_key not in fixed_arguments:
            continue

        value = fixed_arguments[data_key]
        # Skip if value is not a string
        if not isinstance(value, str):
            continue

        expected_type = field_info.annotation

        # Unwrap Annotated types - only the first arg is the actual type
        if get_origin(expected_type) is Annotated:
            type_args = get_args(expected_type)
            expected_type = type_args[0] if type_args else expected_type

        # Get the origin of the expected type (e.g., list from list[str])
        origin = get_origin(expected_type)

        # For Union types, we need to check all union members
        if origin is Union or origin is types.UnionType:
            # For Union types, check each union member
            type_args = get_args(expected_type)
            expected_origins = [get_origin(arg) or arg for arg in type_args]
        else:
            # For non-Union types, just check the origin
            expected_origins = [origin or expected_type]

        # Check if any of the expected types is list or dict
        if any(exp in (list, dict) for exp in expected_origins):
            # Try to parse the string as JSON
            try:
                parsed_value = json.loads(value)
                # json.loads() returns dict, list, str, int, float, bool, or None
                # Only use parsed value if it matches expected collection types
                if isinstance(parsed_value, (list, dict)):
                    fixed_arguments[data_key] = parsed_value
            except (json.JSONDecodeError, ValueError):
                # If parsing fails, leave the original value
                # Pydantic will raise validation error if needed
                pass

    return fixed_arguments


@overload
def prepare_llm_messages(
    events: Sequence[Event],
    condenser: None = None,
    additional_messages: list[Message] | None = None,
    llm: LLM | None = None,
) -> list[Message]: ...


@overload
def prepare_llm_messages(
    events: Sequence[Event],
    condenser: CondenserBase,
    additional_messages: list[Message] | None = None,
    llm: LLM | None = None,
) -> list[Message] | Condensation: ...


def prepare_llm_messages(
    events: Sequence[Event],
    condenser: CondenserBase | None = None,
    additional_messages: list[Message] | None = None,
    llm: LLM | None = None,
) -> list[Message] | Condensation:
    """Prepare LLM messages from conversation context.

    This utility function extracts the common logic for preparing conversation
    context that is shared between agent.step() and ask_agent() methods.
    It handles condensation internally and calls the callback when needed.

    Args:
        events: Sequence of events to prepare messages from
        condenser: Optional condenser for handling context window limits
        additional_messages: Optional additional messages to append
        llm: Optional LLM instance from the agent, passed to condenser for
            token counting or other LLM features

    Returns:
        List of messages ready for LLM completion, or a Condensation event
        if condensation is needed

    Raises:
        RuntimeError: If condensation is needed but no callback is provided
    """

    view = View.from_events(events)
    llm_convertible_events: list[LLMConvertibleEvent] = view.events

    # If a condenser is registered, we need to give it an
    # opportunity to transform the events. This will either
    # produce a list of events, exactly as expected, or a
    # new condensation that needs to be processed
    if condenser is not None:
        condensation_result = condenser.condense(view, agent_llm=llm)

        match condensation_result:
            case View():
                llm_convertible_events = condensation_result.events

            case Condensation():
                return condensation_result

    # Convert events to messages
    messages = LLMConvertibleEvent.events_to_messages(llm_convertible_events)

    # Add any additional messages (e.g., user question for ask_agent)
    if additional_messages:
        messages.extend(additional_messages)

    return messages


def make_llm_completion(
    llm: LLM,
    messages: list[Message],
    tools: list[ToolDefinition] | None = None,
    on_token: ConversationTokenCallbackType | None = None,
) -> LLMResponse:
    """Make an LLM completion call with the provided messages and tools.

    Args:
        llm: The LLM instance to use for completion
        messages: The messages to send to the LLM
        tools: Optional list of tools to provide to the LLM
        on_token: Optional callback for streaming token updates

    Returns:
        LLMResponse from the LLM completion call

    Note:
        Always exposes a 'security_risk' parameter in tool schemas via
        add_security_risk_prediction=True. This ensures the schema remains
        consistent, even if the security analyzer is disabled. Validation of
        this field happens dynamically at runtime depending on the analyzer
        configured. This allows weaker models to omit risk field and bypass
        validation requirements when analyzer is disabled. For detailed logic,
        see `_extract_security_risk` method in agent.py.
    """
    if llm.uses_responses_api():
        return llm.responses(
            messages=messages,
            tools=tools or [],
            include=None,
            store=False,
            add_security_risk_prediction=True,
            on_token=on_token,
        )
    else:
        return llm.completion(
            messages=messages,
            tools=tools or [],
            add_security_risk_prediction=True,
            on_token=on_token,
        )
