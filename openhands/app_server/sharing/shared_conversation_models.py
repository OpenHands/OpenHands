from datetime import datetime
from enum import Enum

# Simplified imports to avoid dependency chain issues
# from openhands.integrations.service_types import ProviderType
# from openhands.sdk.llm import MetricsSnapshot
# from openhands.storage.data_models.conversation_metadata import ConversationTrigger
# For now, use Any to avoid import issues
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from openhands.agent_server.utils import OpenHandsUUID, utc_now

ProviderType = Any
MetricsSnapshot = Any
ConversationTrigger = Any


class SharedConversation(BaseModel):
    """Shared conversation info model with all fields from AppConversationInfo."""

    id: OpenHandsUUID = Field(default_factory=uuid4)

    created_by_user_id: str | None
    sandbox_id: str

    selected_repository: str | None = None
    selected_branch: str | None = None
    git_provider: ProviderType | None = None
    title: str | None = None
    pr_number: list[int] = Field(default_factory=list)
    llm_model: str | None = None

    metrics: MetricsSnapshot | None = None

    parent_conversation_id: OpenHandsUUID | None = None
    sub_conversation_ids: list[OpenHandsUUID] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SharedConversationSortOrder(Enum):
    CREATED_AT = 'CREATED_AT'
    CREATED_AT_DESC = 'CREATED_AT_DESC'
    UPDATED_AT = 'UPDATED_AT'
    UPDATED_AT_DESC = 'UPDATED_AT_DESC'
    TITLE = 'TITLE'
    TITLE_DESC = 'TITLE_DESC'


class SharedConversationPage(BaseModel):
    items: list[SharedConversation]
    next_page_id: str | None = None
