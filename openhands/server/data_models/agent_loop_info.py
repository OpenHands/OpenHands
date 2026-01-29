# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
from dataclasses import dataclass, field

from openhands.events.event_store_abc import EventStoreABC
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.storage.data_models.conversation_status import ConversationStatus


@dataclass
class AgentLoopInfo:
    """Information about an agent loop - the URL on which to locate it and the event store"""

    conversation_id: str
    url: str | None
    session_api_key: str | None
    event_store: EventStoreABC | None
    status: ConversationStatus = field(default=ConversationStatus.RUNNING)
    runtime_status: RuntimeStatus | None = None
