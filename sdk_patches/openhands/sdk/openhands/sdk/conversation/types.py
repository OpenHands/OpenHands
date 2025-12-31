import uuid
from collections.abc import Callable

from pydantic import BaseModel, Field

from openhands.sdk.event.base import Event
from openhands.sdk.llm.streaming import TokenCallbackType


ConversationCallbackType = Callable[[Event], None]
"""Type alias for event callback functions."""

ConversationTokenCallbackType = TokenCallbackType
"""Callback type invoked for streaming LLM deltas."""

ConversationID = uuid.UUID
"""Type alias for conversation IDs."""


class StuckDetectionThresholds(BaseModel):
    """Configuration for stuck detection thresholds.

    Attributes:
        action_observation: Number of repetitions before triggering
            action-observation loop detection
        action_error: Number of repetitions before triggering
            action-error loop detection
        monologue: Number of consecutive agent messages before triggering
            monologue detection
        alternating_pattern: Number of repetitions before triggering
            alternating pattern detection
    """

    action_observation: int = Field(
        default=4, ge=1, description="Threshold for action-observation loop detection"
    )
    action_error: int = Field(
        default=3, ge=1, description="Threshold for action-error loop detection"
    )
    monologue: int = Field(
        default=3, ge=1, description="Threshold for agent monologue detection"
    )
    alternating_pattern: int = Field(
        default=6, ge=1, description="Threshold for alternating pattern detection"
    )
