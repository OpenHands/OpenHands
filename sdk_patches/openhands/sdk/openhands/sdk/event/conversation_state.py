"""Events related to conversation state updates."""

import uuid
from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator

from openhands.sdk.event.base import Event
from openhands.sdk.event.types import SourceType


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState

FULL_STATE_KEY = "full_state"


class ConversationStateUpdateEvent(Event):
    """Event that contains conversation state updates.

    This event is sent via websocket whenever the conversation state changes,
    allowing remote clients to stay in sync without making REST API calls.

    All fields are serialized versions of the corresponding ConversationState fields
    to ensure compatibility with websocket transmission.
    """

    source: SourceType = "environment"
    key: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique key for this state update event",
    )
    value: Any = Field(
        default_factory=dict,
        description="Serialized conversation state updates",
    )

    @field_validator("key")
    def validate_key(cls, key):
        if not isinstance(key, str):
            raise ValueError("Key must be a string")
        # Allow special key "full_state" for full state snapshots
        if key == FULL_STATE_KEY:
            return key
        # Allow any string key for flexibility (testing, future extensibility)
        # In practice, keys should match ConversationState fields,
        # but we don't enforce it
        return key

    @field_validator("value")
    def validate_value(cls, value, info):
        # Prevent circular import
        from openhands.sdk.conversation.conversation_stats import ConversationStats

        # For ConversationStats, use snapshot serialization to avoid
        # sending lengthy lists over WebSocket
        if isinstance(value, ConversationStats):
            return value.model_dump(mode="json", context={"use_snapshot": True})

        key = info.data.get("key")
        if key is None:
            # Allow value without key for flexibility
            return value

        # Skip validation for special "full_state" key
        if key == FULL_STATE_KEY:
            return value

        # Prevent circular import
        from openhands.sdk.conversation.state import ConversationState

        field_info = ConversationState.model_fields.get(key)
        if field_info is None:
            # Allow arbitrary keys for testing/future extensibility
            return value

        # Skip type validation - just accept any value
        # The actual type conversion will happen when the state is updated
        return value

    @classmethod
    def from_conversation_state(
        cls, state: "ConversationState"
    ) -> "ConversationStateUpdateEvent":
        """Create a state update event from a ConversationState object.

        This creates an event containing a snapshot of important state fields.

        Args:
            state: The ConversationState to serialize
            conversation_id: The conversation ID for the event

        Returns:
            A ConversationStateUpdateEvent with serialized state data
        """
        # Create a snapshot with all important state fields
        # Use mode='json' to ensure proper serialization including SecretStr
        state_snapshot = state.model_dump(mode="json", exclude_none=True)

        # Use a special key "full_state" to indicate this is a full snapshot
        return cls(key=FULL_STATE_KEY, value=state_snapshot)

    def __str__(self) -> str:
        return f"ConversationStateUpdate(key={self.key}, value={self.value})"
