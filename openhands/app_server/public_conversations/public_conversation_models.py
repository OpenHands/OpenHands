from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from openhands.agent_server.utils import OpenHandsUUID, utc_now
from openhands.integrations.service_types import ProviderType
from openhands.sdk.llm import MetricsSnapshot
from openhands.storage.data_models.conversation_metadata import ConversationTrigger


class PublicConversationInfo(BaseModel):
    """Base conversation info which does not contain status or user-specific data."""

    id: OpenHandsUUID = Field(default_factory=uuid4)

    sandbox_id: str

    selected_repository: str | None = None
    selected_branch: str | None = None
    git_provider: ProviderType | None = None
    title: str | None = None
    trigger: ConversationTrigger | None = None
    pr_number: list[int] = Field(default_factory=list)
    llm_model: str | None = None

    metrics: MetricsSnapshot | None = None

    parent_conversation_id: OpenHandsUUID | None = None
    sub_conversation_ids: list[OpenHandsUUID] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class PublicConversationSortOrder(Enum):
    CREATED_AT = 'CREATED_AT'
    CREATED_AT_DESC = 'CREATED_AT_DESC'
    UPDATED_AT = 'UPDATED_AT'
    UPDATED_AT_DESC = 'UPDATED_AT_DESC'
    TITLE = 'TITLE'
    TITLE_DESC = 'TITLE_DESC'


class PublicConversationInfoPage(BaseModel):
    items: list[PublicConversationInfo]
    next_page_id: str | None = None