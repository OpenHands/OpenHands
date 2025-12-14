"""Data models for public conversation sharing."""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from openhands.integrations.service_types import ProviderType
from openhands.storage.data_models.conversation_metadata import ConversationTrigger
from openhands.storage.data_models.conversation_status import ConversationStatus


@dataclass
class PublicConversationInfo:
    """Public-safe conversation information with sensitive data filtered out."""

    conversation_id: str
    title: str
    status: ConversationStatus = ConversationStatus.STOPPED
    selected_repository: str | None = None
    selected_branch: str | None = None
    git_provider: ProviderType | None = None
    trigger: ConversationTrigger | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated_at: datetime | None = None
    shared_at: datetime | None = None
    # Note: Excludes user_id, session_api_key, llm_model, cost metrics, etc.


@dataclass
class PublicMessageInfo:
    """Public-safe message information with sensitive content filtered."""

    id: str
    timestamp: datetime
    source: str  # 'user' or 'assistant'
    content: str
    # Note: Excludes sensitive action details, API keys, tokens, etc.


@dataclass
class PublicConversationDetail:
    """Complete public conversation with messages."""

    conversation: PublicConversationInfo
    messages: list[PublicMessageInfo]
