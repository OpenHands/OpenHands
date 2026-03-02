# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
"""Utility for converting OpenHands events to OpenAI message format."""

from typing import Any

from openhands.core.logger import openhands_logger as logger
from openhands.events.action.message import MessageAction, SystemMessageAction
from openhands.events.event import EventSource
from openhands.events.observation.browse import BrowserOutputObservation
from openhands.events.observation.commands import (
    CmdOutputObservation,
    IPythonRunCellObservation,
)
from openhands.events.observation.file_download import FileDownloadObservation
from openhands.events.observation.files import (
    FileEditObservation,
    FileReadObservation,
    FileWriteObservation,
)
from openhands.events.observation.mcp import MCPObservation
from openhands.events.observation.observation import Observation


def convert_events_to_openai_messages(events: list[Any]) -> list[dict[str, Any]]:
    """Convert OpenHands events to OpenAI message format for LLM APIs."""
    openai_messages = []

    logger.info(f'Converting {len(events)} events to OpenAI messages')

    for i, event in enumerate(events):
        event_type = type(event).__name__

        # Skip agent_state_changed events and internal system actions
        if event_type in [
            'AgentStateChangedObservation',
            'ChangeAgentStateAction',
            'RecallAction',
            'RecallObservation',
            'TaskTrackingAction',
        ]:
            continue

        # Handle system messages
        if isinstance(event, SystemMessageAction):
            msg = {'role': 'system', 'content': event.content}
            openai_messages.append(msg)
        # Handle content messages
        elif isinstance(event, MessageAction):
            source = getattr(event, '_source', getattr(event, 'source', None))
            if source == EventSource.USER:
                msg = {'role': 'user', 'content': event.content}
                (msg['role'], msg['content'])
                openai_messages.append(msg)

            elif source == EventSource.AGENT:
                msg = {'role': 'assistant', 'content': event.content}
                (msg['role'], msg['content'])
                openai_messages.append(msg)

        # Handle tool calls
        elif (
            not isinstance(event, Observation)
            and hasattr(event, 'tool_call_metadata')
            and event.tool_call_metadata
            and getattr(event, '_source', getattr(event, 'source', None))
            == EventSource.AGENT
        ):
            tool_metadata = event.tool_call_metadata
            model_response = getattr(tool_metadata, 'model_response', {}) or {}
            choices = model_response.get('choices', [])

            if choices:
                choice = choices[0]
                message_data = choice.get('message', {})

                tool_calls = message_data.get('tool_calls')
                if tool_calls:
                    serializable_tool_calls = []
                    for tc in tool_calls:
                        if hasattr(tc, 'id'):
                            tc_dict = {
                                'id': tc.id,
                                'type': getattr(tc, 'type', 'function'),
                                'function': {
                                    'name': tc.function.name,
                                    'arguments': tc.function.arguments,
                                },
                            }
                            # Remove security_risk from arguments to avoid biasing the analysis
                            try:
                                import json

                                args = json.loads(tc.function.arguments)
                                if 'security_risk' in args:
                                    del args['security_risk']
                                tc_dict['function']['arguments'] = json.dumps(args)
                            except (json.JSONDecodeError, KeyError):
                                pass
                            serializable_tool_calls.append(tc_dict)
                        else:
                            serializable_tool_calls.append(tc)

                    assistant_msg = {
                        'role': 'assistant',
                        'content': message_data.get('content', ''),
                        'tool_calls': serializable_tool_calls,
                    }

                    openai_messages.append(assistant_msg)

        # Handle tool responses
        elif isinstance(
            event,
            (
                FileReadObservation,
                FileWriteObservation,
                FileEditObservation,
                CmdOutputObservation,
                IPythonRunCellObservation,
                BrowserOutputObservation,
                MCPObservation,
                FileDownloadObservation,
            ),
        ):
            # Skip observations from ENVIRONMENT source
            source = getattr(event, '_source', getattr(event, 'source', None))
            if source == EventSource.ENVIRONMENT:
                continue

            tool_call_id = None
            if hasattr(event, 'tool_call_metadata') and event.tool_call_metadata:
                tool_call_id = getattr(event.tool_call_metadata, 'tool_call_id', None)

            if tool_call_id:
                content = (
                    str(event.content) if hasattr(event, 'content') else str(event)
                )
                msg = {'role': 'tool', 'content': content, 'tool_call_id': tool_call_id}

                openai_messages.append(msg)
            else:
                logger.warning(
                    f'Could not find tool_call_id for observation {event_type}'
                )

    return openai_messages
