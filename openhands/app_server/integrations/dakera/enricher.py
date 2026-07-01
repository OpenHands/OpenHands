"""Dakera ConversationSecretEnricher implementation.

Registers with OpenHands via ``ServerConfig.conversation_secret_enricher_class``::

    server_config.conversation_secret_enricher_class = (
        "openhands.app_server.integrations.dakera.enricher"
        ".DakeraConversationSecretEnricher"
    )

On each conversation start the enricher:

1. Reads the user's pending message / conversation title as the recall query
   (falling back to a generic query when no text is available).
2. Searches Dakera for the most relevant memories.
3. Formats them as a brief ``<memory>`` block appended to the system message.

The enricher degrades gracefully: if Dakera is unreachable or disabled it
passes the original ``system_message_suffix`` through unchanged so the
conversation still starts normally.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from openhands.app_server.app_conversation.app_conversation_models import (
    ConversationTrigger,
)
from openhands.app_server.app_conversation.conversation_secret_enricher import (
    ConversationSecretEnrichment,
    ConversationSecretEnricher,
)
from openhands.app_server.integrations.dakera.config import DakeraConfig
from openhands.app_server.integrations.dakera.memory_client import DakeraMemoryClient
from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.user.user_models import UserInfo

_logger = logging.getLogger(__name__)

# Maximum character length of individual memory content included in the prompt
_MAX_MEMORY_CONTENT_CHARS = 500

# System-message block template
_MEMORY_BLOCK_TEMPLATE = """\

<dakera_memory>
The following memories from previous sessions may be relevant to this task:

{entries}
</dakera_memory>
"""

_MEMORY_ENTRY_TEMPLATE = '[{index}] (relevance {score:.2f}) {content}'


def _build_memory_block(hits: list[dict[str, Any]]) -> str:
    """Format a list of memory hits into a system-prompt block."""
    entries: list[str] = []
    for i, hit in enumerate(hits, start=1):
        mem = hit.get('memory', {})
        content = str(mem.get('content', '')).strip()
        if not content:
            continue
        # Truncate very long memories so the prompt stays bounded
        if len(content) > _MAX_MEMORY_CONTENT_CHARS:
            content = content[:_MAX_MEMORY_CONTENT_CHARS].rstrip() + ' …'
        score = float(hit.get('score', 0.0))
        entries.append(
            _MEMORY_ENTRY_TEMPLATE.format(index=i, score=score, content=content)
        )
    if not entries:
        return ''
    return _MEMORY_BLOCK_TEMPLATE.format(entries='\n'.join(entries))


def _derive_recall_query(user: UserInfo) -> str:
    """Extract a recall query from the user's conversation context.

    Falls back to a generic query when no text is available so the enricher
    still attempts to surface generally useful memories even without a task
    description.
    """
    # UserInfo carries the initial message text in conversation_settings when
    # the user started the conversation from the UI.
    try:
        settings = getattr(user, 'conversation_settings', None)
        if settings is not None:
            initial = getattr(settings, 'initial_message', None)
            if initial:
                text = getattr(initial, 'text', None) or str(initial)
                if text:
                    return text[:1000]  # cap query length
    except Exception:
        pass
    return 'recent tasks and context'


class DakeraConversationSecretEnricher(ConversationSecretEnricher):
    """Inject relevant Dakera memories into every conversation's system prompt.

    Configuration is read from environment variables at instantiation time.
    Pass a ``DakeraConfig`` (and optionally a shared ``DakeraMemoryClient``)
    to the constructor for testing or fine-grained control.
    """

    def __init__(
        self,
        config: DakeraConfig | None = None,
        client: DakeraMemoryClient | None = None,
    ) -> None:
        self._config = config or DakeraConfig()
        self._client = client or DakeraMemoryClient(self._config)

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
        """Search Dakera for relevant memories and prepend them to the system suffix.

        If Dakera is disabled or the search returns no hits the original
        ``system_message_suffix`` is returned unchanged.
        """
        if not self._config.enabled:
            _logger.debug('Dakera memory integration is disabled — skipping enrichment')
            return ConversationSecretEnrichment(
                system_message_suffix=system_message_suffix
            )

        try:
            query = _derive_recall_query(user)
            _logger.debug(
                'Querying Dakera for memories (agent_id=%s, top_k=%d, query=%r)',
                self._config.agent_id,
                self._config.top_k,
                query[:80],
            )

            hits = await self._client.search(query)

            if not hits:
                _logger.debug('Dakera returned no memories for query')
                return ConversationSecretEnrichment(
                    system_message_suffix=system_message_suffix
                )

            memory_block = _build_memory_block(hits)
            if not memory_block:
                return ConversationSecretEnrichment(
                    system_message_suffix=system_message_suffix
                )

            _logger.info(
                'Injecting %d Dakera memories into system prompt', len(hits)
            )

            # Prepend the memory block to any existing suffix so downstream
            # enrichers can still append their own content after us.
            combined = memory_block
            if system_message_suffix:
                combined = combined + '\n' + system_message_suffix

            return ConversationSecretEnrichment(system_message_suffix=combined)

        except Exception as exc:
            _logger.warning(
                'Dakera enricher failed — continuing without memories: %s', exc
            )
            return ConversationSecretEnrichment(
                system_message_suffix=system_message_suffix
            )
