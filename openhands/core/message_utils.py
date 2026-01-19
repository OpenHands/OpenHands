# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from openhands.events.event import Event
from openhands.llm.metrics import Metrics, TokenUsage


def get_token_usage_for_event(event: Event, metrics: Metrics) -> TokenUsage | None:
    """Returns at most one token usage record for either:
      - `tool_call_metadata.model_response.id`, if possible
      - otherwise event.response_id, if set

    If neither exist or none matches in metrics.token_usages, returns None.
    """
    # 1) Use the tool_call_metadata's response.id if present
    if event.tool_call_metadata and event.tool_call_metadata.model_response:
        tool_response_id = event.tool_call_metadata.model_response.get('id')
        if tool_response_id:
            usage_rec = next(
                (u for u in metrics.token_usages if u.response_id == tool_response_id),
                None,
            )
            if usage_rec is not None:
                return usage_rec

    # 2) Fallback to the top-level event.response_id if present
    if event.response_id:
        return next(
            (u for u in metrics.token_usages if u.response_id == event.response_id),
            None,
        )

    return None


def get_token_usage_for_event_id(
    events: list[Event], event_id: int, metrics: Metrics
) -> TokenUsage | None:
    """Starting from the event with .id == event_id and moving backwards in `events`,
    find the first TokenUsage record (if any) associated either with:
      - tool_call_metadata.model_response.id, or
      - event.response_id
    Returns the first match found, or None if none is found.
    """
    # Find the index of the event with the given id
    idx = next((i for i, e in enumerate(events) if e.id == event_id), None)
    if idx is None:
        return None

    # Search backward from idx down to 0
    for i in range(idx, -1, -1):
        usage = get_token_usage_for_event(events[i], metrics)
        if usage is not None:
            return usage
    return None
