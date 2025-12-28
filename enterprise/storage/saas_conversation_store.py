from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from datetime import UTC

from sqlalchemy.orm import sessionmaker
from storage.database import session_maker
from storage.stored_conversation_metadata import StoredConversationMetadata

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.integrations.provider import ProviderType
from openhands.storage.conversation.conversation_store import ConversationStore
from openhands.storage.data_models.conversation_metadata import (
    ConversationMetadata,
    ConversationTrigger,
)
from openhands.storage.data_models.conversation_metadata_result_set import (
    ConversationMetadataResultSet,
)
from openhands.utils.async_utils import call_sync_from_async
from openhands.utils.search_utils import offset_to_page_id, page_id_to_offset

logger = logging.getLogger(__name__)


@dataclass
class SaasConversationStore(ConversationStore):
    user_id: str
    session_maker: sessionmaker

    def _select_by_id(self, session, conversation_id: str):
        return (
            session.query(StoredConversationMetadata)
            .filter(StoredConversationMetadata.user_id == self.user_id)
            .filter(StoredConversationMetadata.conversation_id == conversation_id)
            .filter(StoredConversationMetadata.conversation_version == 'V0')
        )

    def _to_external_model(self, conversation_metadata: StoredConversationMetadata):
        kwargs = {
            c.name: getattr(conversation_metadata, c.name)
            for c in StoredConversationMetadata.__table__.columns
            if c.name != 'github_user_id'  # Skip github_user_id field
        }
        # TODO: I'm not sure why the timezone is not set on the dates coming back out of the db
        kwargs['created_at'] = kwargs['created_at'].replace(tzinfo=UTC)
        kwargs['last_updated_at'] = kwargs['last_updated_at'].replace(tzinfo=UTC)
        if kwargs['trigger']:
            kwargs['trigger'] = ConversationTrigger(kwargs['trigger'])
        if kwargs['git_provider'] and isinstance(kwargs['git_provider'], str):
            # Convert string to ProviderType enum
            kwargs['git_provider'] = ProviderType(kwargs['git_provider'])

        # Remove V1 attributes
        kwargs.pop('max_budget_per_task', None)
        kwargs.pop('cache_read_tokens', None)
        kwargs.pop('cache_write_tokens', None)
        kwargs.pop('reasoning_tokens', None)
        kwargs.pop('context_window', None)
        kwargs.pop('per_turn_token', None)
        kwargs.pop('parent_conversation_id', None)
        kwargs.pop('public')

        return ConversationMetadata(**kwargs)

    async def save_metadata(self, metadata: ConversationMetadata):
        kwargs = dataclasses.asdict(metadata)
        kwargs['user_id'] = self.user_id

        # Convert ProviderType enum to string for storage
        if kwargs.get('git_provider') is not None:
            kwargs['git_provider'] = (
                kwargs['git_provider'].value
                if hasattr(kwargs['git_provider'], 'value')
                else kwargs['git_provider']
            )

        stored_metadata = StoredConversationMetadata(**kwargs)

        def _save_metadata():
            with self.session_maker() as session:
                session.merge(stored_metadata)
                session.commit()

        await call_sync_from_async(_save_metadata)

    async def get_metadata(self, conversation_id: str) -> ConversationMetadata:
        def _get_metadata():
            with self.session_maker() as session:
                conversation_metadata = self._select_by_id(
                    session, conversation_id
                ).first()
                if not conversation_metadata:
                    raise FileNotFoundError(conversation_id)
                return self._to_external_model(conversation_metadata)

        return await call_sync_from_async(_get_metadata)

    async def delete_metadata(self, conversation_id: str) -> None:
        def _delete_metadata():
            with self.session_maker() as session:
                self._select_by_id(session, conversation_id).delete()
                session.commit()

        await call_sync_from_async(_delete_metadata)

    async def exists(self, conversation_id: str) -> bool:
        def _exists():
            with self.session_maker() as session:
                result = self._select_by_id(session, conversation_id).scalar()
                return bool(result)

        return await call_sync_from_async(_exists)

    async def search(
        self,
        page_id: str | None = None,
        limit: int = 20,
    ) -> ConversationMetadataResultSet:
        offset = page_id_to_offset(page_id)

        def _search():
            with self.session_maker() as session:
                conversations = (
                    session.query(StoredConversationMetadata)
                    .filter(StoredConversationMetadata.user_id == self.user_id)
                    .filter(StoredConversationMetadata.conversation_version == 'V0')
                    .order_by(StoredConversationMetadata.created_at.desc())
                    .offset(offset)
                    .limit(limit + 1)
                    .all()
                )
                conversations = [self._to_external_model(c) for c in conversations]
                current_page_size = len(conversations)
                next_page_id = offset_to_page_id(
                    offset + limit, current_page_size > limit
                )
                conversations = conversations[:limit]
                return ConversationMetadataResultSet(conversations, next_page_id)

        return await call_sync_from_async(_search)

    @classmethod
    async def get_instance(
        cls, config: OpenHandsConfig, user_id: str | None
    ) -> ConversationStore:
        # user_id should not be None in SaaS, should we raise?
        return SaasConversationStore(str(user_id), session_maker)
