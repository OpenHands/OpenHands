from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from openhands.app_server.app_conversation.app_conversation_models import (
    ConversationTrigger,
)
from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.user.user_models import UserInfo
from openhands.sdk.secret import SecretSource


@dataclass
class ConversationSecretEnrichment:
    secrets: dict[str, SecretSource] = field(default_factory=dict)
    system_message_suffix: str | None = None


class ConversationSecretEnricher:
    """Extension point for integration-scoped conversation secrets.

    This intentionally runs only while building a conversation start request.
    It should not be used for persisted custom secrets or sandbox settings
    secret listing.
    """

    async def enrich(
        self,
        *,
        user_context: UserContext,
        user: UserInfo,
        trigger: ConversationTrigger | None,
        system_message_suffix: str | None,
        web_url: str | None,
        jwt_service: JwtService,
        access_token_hard_timeout: timedelta | None,
    ) -> ConversationSecretEnrichment:
        return ConversationSecretEnrichment(system_message_suffix=system_message_suffix)
