from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from datetime import UTC
from uuid import UUID

from sqlalchemy.orm import sessionmaker
from storage.database import session_maker
from storage.stored_conversation_metadata import StoredConversationMetadata
from storage.stored_conversation_metadata_saas import StoredConversationMetadataSaas
from storage.user_store import UserStore

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
    org_id: UUID | None = None  # will be fetched automatically

    def __init__(self, user_id: str, org_id: UUID, session_maker: sessionmaker):
        self.user_id = user_id
        self.org_id = org_id
        self.session_maker = session_maker

    def _select_by_id(self, session, conversation_id: str):
        # Join StoredConversationMetadata with ConversationMetadataSaas to filter by user/org
        query = (
            session.query(StoredConversationMetadata)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .filter(StoredConversationMetadataSaas.user_id == UUID(self.user_id))
            .filter(StoredConversationMetadata.conversation_id == conversation_id)
            .filter(StoredConversationMetadata.conversation_version == 'V0')
        )

        if self.org_id is not None:
            query = query.filter(StoredConversationMetadataSaas.org_id == self.org_id)

        return query

    def _to_external_model(self, conversation_metadata: StoredConversationMetadata):
        kwargs = {
            c.name: getattr(conversation_metadata, c.name)
            for c in StoredConversationMetadata.__table__.columns
        }
        # TODO: I'm not sure why the timezone is not set on the dates coming back out of the db
        kwargs['created_at'] = kwargs['created_at'].replace(tzinfo=UTC)
        kwargs['last_updated_at'] = kwargs['last_updated_at'].replace(tzinfo=UTC)
        if kwargs['trigger']:
            kwargs['trigger'] = ConversationTrigger(kwargs['trigger'])
        if kwargs['git_provider'] and isinstance(kwargs['git_provider'], str):
            # Convert string to ProviderType enum
            kwargs['git_provider'] = ProviderType(kwargs['git_provider'])

        kwargs['user_id'] = self.user_id

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

        # Remove user_id and org_id from kwargs since they're no longer in StoredConversationMetadata
        kwargs.pop('user_id', None)
        kwargs.pop('org_id', None)

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
                # Save the main conversation metadata
                session.merge(stored_metadata)

                # Create or update the SaaS metadata record
                saas_metadata = (
                    session.query(StoredConversationMetadataSaas)
                    .filter(
                        StoredConversationMetadataSaas.conversation_id
                        == stored_metadata.conversation_id
                    )
                    .first()
                )

                if not saas_metadata:
                    saas_metadata = StoredConversationMetadataSaas(
                        conversation_id=stored_metadata.conversation_id,
                        user_id=UUID(self.user_id),
                        org_id=self.org_id,
                    )
                    session.add(saas_metadata)
                else:
                    # Validate
                    expected_user_id = UUID(self.user_id)
                    expected_org_id = self.org_id

                    if saas_metadata.user_id != expected_user_id:
                        raise ValueError(
                            f'Existing user_id ({saas_metadata.user_id}) does not match expected value ({expected_user_id}).'
                        )

                    if expected_org_id and saas_metadata.org_id != expected_org_id:
                        raise ValueError(
                            f'Existing org_id ({saas_metadata.org_id}) does not match expected value ({expected_org_id}).'
                        )

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
                saas_record = (
                    session.query(StoredConversationMetadataSaas)
                    .filter(
                        StoredConversationMetadataSaas.conversation_id
                        == conversation_id,
                        StoredConversationMetadataSaas.user_id == UUID(self.user_id),
                        StoredConversationMetadataSaas.org_id == self.org_id,
                    )
                    .first()
                )

                if saas_record:
                    # Delete both records, but only if the SaaS one exists
                    session.query(StoredConversationMetadata).filter(
                        StoredConversationMetadata.conversation_id == conversation_id,
                    ).delete()

                    session.delete(saas_record)

                    session.commit()
                else:
                    # No SaaS record found → skip deleting main metadata
                    session.rollback()

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
                    .join(
                        StoredConversationMetadataSaas,
                        StoredConversationMetadata.conversation_id
                        == StoredConversationMetadataSaas.conversation_id,
                    )
                    .filter(
                        StoredConversationMetadataSaas.user_id == UUID(self.user_id)
                    )
                    .filter(StoredConversationMetadataSaas.org_id == self.org_id)
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
        user = await UserStore.get_user_by_id_async(user_id)
        org_id = user.current_org_id if user else None
        return SaasConversationStore(str(user_id), org_id, session_maker)
