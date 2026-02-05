# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
from dataclasses import dataclass, field
from datetime import datetime, timezone

from openhands.integrations.service_types import ProviderType
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.storage.data_models.conversation_metadata import ConversationTrigger
from openhands.storage.data_models.conversation_status import ConversationStatus


@dataclass
class ConversationInfo:
    """Information about a conversation. This combines conversation metadata with
    information on whether a conversation is currently running
    """

    conversation_id: str
    title: str
    last_updated_at: datetime | None = None
    status: ConversationStatus = ConversationStatus.STOPPED
    runtime_status: RuntimeStatus | None = None
    selected_repository: str | None = None
    selected_branch: str | None = None
    git_provider: ProviderType | None = None
    trigger: ConversationTrigger | None = None
    num_connections: int = 0
    url: str | None = None
    session_api_key: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    pr_number: list[int] = field(default_factory=list)
    conversation_version: str = 'V0'
    sub_conversation_ids: list[str] = field(default_factory=list)
    public: bool | None = None
