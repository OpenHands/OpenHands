"""Enterprise injector for SQLAppConversationInfoService with SAAS filtering."""

from datetime import datetime
from typing import AsyncGenerator, cast
from uuid import UUID, uuid4

from fastapi import Request
from sqlalchemy import ColumnElement, delete, func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from storage.stored_conversation_metadata import StoredConversationMetadata
from storage.stored_conversation_metadata_saas import StoredConversationMetadataSaas
from storage.user import User

from openhands.agent_server.utils import utc_now
from openhands.app_server.app_conversation.app_conversation_info_service import (
    AppConversationInfoService,
    AppConversationInfoServiceInjector,
    ManagedCredentialConversationRef,
)
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
    AppConversationInfoPage,
    AppConversationSortOrder,
    has_managed_codex_credential,
)
from openhands.app_server.app_conversation.sql_app_conversation_info_service import (
    APP_CONVERSATION_RESERVATION_TOKEN_KEY,
    APP_CONVERSATION_RESERVATION_TTL,
    APP_CONVERSATION_RESERVATION_VERSION,
    SQLAppConversationInfoService,
)
from openhands.app_server.errors import AuthError
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import ADMIN, SandboxUserContext


class SaasSQLAppConversationInfoService(SQLAppConversationInfoService):
    """Extended SQLAppConversationInfoService with user and organization-based filtering and SAAS metadata handling."""

    async def _get_current_user(self) -> User | None:
        """Get the current user using the existing db_session.

        Uses self.db_session to avoid opening a separate database session.

        Returns:
            User object or None if no user_id is available
        """
        user_id_str = await self.user_context.get_user_id()
        if not user_id_str:
            return None

        user_id_uuid = UUID(user_id_str)
        result = await self.db_session.execute(
            select(User).where(User.id == user_id_uuid)
        )
        return result.scalars().first()

    async def _apply_user_and_org_filter(self, query):
        """Apply tenant filters to ensure conversation isolation.

        Filters conversations by:
        - user_id: Only show conversations belonging to the current user
        - sandbox_id: Authenticated sandbox webhooks stay within that sandbox
        - org_id: Other requests use the request's
          *effective* organization (honors ``X-Org-Id`` and API-key org
          binding; falls back to ``user.current_org_id``).

        Args:
            query: SQLAlchemy query to apply filters to

        Returns:
            Query with user and organization filters applied

        Raises:
            AuthError: If no user_id is available (secure default: deny access)
        """
        # For internal operations such as getting a conversation by session_api_key
        # we need a mode that does not have filtering. The dependency `as_admin()`
        # is used to enable it
        if self.user_context == ADMIN:
            return query

        user_id_str = await self.user_context.get_user_id()
        if not user_id_str:
            # Secure default: no user means no access, not "show everything"
            raise AuthError('User authentication required')

        user_id_uuid = UUID(user_id_str)
        query = query.where(StoredConversationMetadataSaas.user_id == user_id_uuid)

        if isinstance(self.user_context, SandboxUserContext):
            return query.where(
                StoredConversationMetadata.sandbox_id == self.user_context.sandbox_id
            )

        # Filter by the *effective* organization id (X-Org-Id override or
        # API-key binding take precedence over user.current_org_id).
        effective_org_id = await self._get_effective_org_id()
        if effective_org_id is not None:
            query = query.where(
                StoredConversationMetadataSaas.org_id == effective_org_id
            )

        return query

    async def _get_effective_org_id(self) -> UUID | None:
        """Resolve the effective org id for the active user context.

        Returns the request's effective org id (X-Org-Id > api_key_org_id >
        user.current_org_id) when the user is authenticated via SAAS auth,
        otherwise falls back to the user's persisted current_org_id.
        """
        get_effective_org_id = getattr(self.user_context, 'get_effective_org_id', None)
        if callable(get_effective_org_id):
            effective_org_id = await get_effective_org_id()
            if effective_org_id is not None:
                return effective_org_id
        else:
            user_auth = getattr(self.user_context, 'user_auth', None)
            get_effective_org_id = getattr(user_auth, 'get_effective_org_id', None)
            if callable(get_effective_org_id):
                effective_org_id = await get_effective_org_id()
                if effective_org_id is not None:
                    return effective_org_id
        user = await self._get_current_user()
        return user.current_org_id if user else None

    async def _secure_select(self):
        query = (
            select(StoredConversationMetadata)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .where(StoredConversationMetadata.conversation_version == 'V1')
        )
        return await self._apply_user_and_org_filter(query)

    async def _secure_select_with_saas_metadata(self):
        """Select query that includes SAAS metadata for retrieving user_id."""
        query = (
            select(StoredConversationMetadata, StoredConversationMetadataSaas)
            .outerjoin(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .where(StoredConversationMetadata.conversation_version == 'V1')
        )
        return await self._apply_user_and_org_filter(query)

    async def get_managed_credential_conversations_for_sandbox(
        self, sandbox_id: str
    ) -> list[ManagedCredentialConversationRef]:
        query = (
            select(StoredConversationMetadata, StoredConversationMetadataSaas)
            .outerjoin(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .where(
                StoredConversationMetadata.conversation_version == 'V1',
                StoredConversationMetadata.sandbox_id == sandbox_id,
            )
        )
        query = await self._apply_user_and_org_filter(query)
        rows = (await self.db_session.execute(query)).all()
        refs = []
        for stored, saas in rows:
            if not has_managed_codex_credential(stored.tags or {}):
                continue
            refs.append(
                ManagedCredentialConversationRef(
                    conversation_id=UUID(stored.conversation_id),
                    created_by_user_id=(
                        str(saas.user_id) if saas and saas.user_id else None
                    ),
                    organization_id=saas.org_id if saas else None,
                    owner_resolved=saas is not None,
                )
            )
        return refs

    async def search_app_conversation_info(
        self,
        title__contains: str | None = None,
        created_at__gte: datetime | None = None,
        created_at__lt: datetime | None = None,
        updated_at__gte: datetime | None = None,
        updated_at__lt: datetime | None = None,
        sandbox_id__eq: str | None = None,
        sort_order: AppConversationSortOrder = AppConversationSortOrder.CREATED_AT_DESC,
        page_id: str | None = None,
        limit: int = 100,
        include_sub_conversations: bool = False,
    ) -> AppConversationInfoPage:
        """Search for conversations with user_id from SAAS metadata."""
        query = await self._secure_select_with_saas_metadata()

        # Conditionally exclude sub-conversations based on the parameter
        if not include_sub_conversations:
            # Exclude sub-conversations (only include top-level conversations)
            query = query.where(
                StoredConversationMetadata.parent_conversation_id.is_(None)
            )

        query = self._apply_filters_with_saas_metadata(
            query=query,
            title__contains=title__contains,
            created_at__gte=created_at__gte,
            created_at__lt=created_at__lt,
            updated_at__gte=updated_at__gte,
            updated_at__lt=updated_at__lt,
            sandbox_id__eq=sandbox_id__eq,
        )

        # Add sort order
        if sort_order == AppConversationSortOrder.CREATED_AT:
            query = query.order_by(StoredConversationMetadata.created_at)
        elif sort_order == AppConversationSortOrder.CREATED_AT_DESC:
            query = query.order_by(StoredConversationMetadata.created_at.desc())
        elif sort_order == AppConversationSortOrder.UPDATED_AT:
            query = query.order_by(StoredConversationMetadata.last_updated_at)
        elif sort_order == AppConversationSortOrder.UPDATED_AT_DESC:
            query = query.order_by(StoredConversationMetadata.last_updated_at.desc())
        elif sort_order == AppConversationSortOrder.TITLE:
            query = query.order_by(StoredConversationMetadata.title)
        elif sort_order == AppConversationSortOrder.TITLE_DESC:
            query = query.order_by(StoredConversationMetadata.title.desc())

        # Apply pagination
        if page_id is not None:
            try:
                offset = int(page_id)
                query = query.offset(offset)
            except ValueError:
                # If page_id is not a valid integer, start from beginning
                offset = 0
        else:
            offset = 0

        # Apply limit and get one extra to check if there are more results
        query = query.limit(limit + 1)

        result = await self.db_session.execute(query)
        rows = result.all()

        # Check if there are more results
        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]

        items = [
            self._to_info_with_user_id(stored_metadata, saas_metadata)
            for stored_metadata, saas_metadata in rows
        ]

        # Calculate next page ID
        next_page_id = None
        if has_more:
            next_page_id = str(offset + limit)

        return AppConversationInfoPage(items=items, next_page_id=next_page_id)

    async def count_app_conversation_info(
        self,
        title__contains: str | None = None,
        created_at__gte: datetime | None = None,
        created_at__lt: datetime | None = None,
        updated_at__gte: datetime | None = None,
        updated_at__lt: datetime | None = None,
        sandbox_id__eq: str | None = None,
    ) -> int:
        """Count conversations matching the given filters with SAAS metadata."""
        query = (
            select(func.count(StoredConversationMetadata.conversation_id))
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .where(StoredConversationMetadata.conversation_version == 'V1')
        )

        # Apply user and organization filtering
        query = await self._apply_user_and_org_filter(query)

        query = self._apply_filters_with_saas_metadata(
            query=query,
            title__contains=title__contains,
            created_at__gte=created_at__gte,
            created_at__lt=created_at__lt,
            updated_at__gte=updated_at__gte,
            updated_at__lt=updated_at__lt,
            sandbox_id__eq=sandbox_id__eq,
        )

        result = await self.db_session.execute(query)
        count = result.scalar()
        return count or 0

    def _apply_filters_with_saas_metadata(
        self,
        query,
        title__contains: str | None = None,
        created_at__gte: datetime | None = None,
        created_at__lt: datetime | None = None,
        updated_at__gte: datetime | None = None,
        updated_at__lt: datetime | None = None,
        sandbox_id__eq: str | None = None,
    ):
        """Apply filters to query that includes SAAS metadata."""
        # Apply the same filters as the base class
        conditions: list[ColumnElement[bool]] = []
        if title__contains is not None:
            conditions.append(
                StoredConversationMetadata.title.like(f'%{title__contains}%')
            )

        if created_at__gte is not None:
            conditions.append(StoredConversationMetadata.created_at >= created_at__gte)

        if created_at__lt is not None:
            conditions.append(StoredConversationMetadata.created_at < created_at__lt)

        if updated_at__gte is not None:
            conditions.append(
                StoredConversationMetadata.last_updated_at >= updated_at__gte
            )

        if updated_at__lt is not None:
            conditions.append(
                StoredConversationMetadata.last_updated_at < updated_at__lt
            )

        if sandbox_id__eq is not None:
            conditions.append(StoredConversationMetadata.sandbox_id == sandbox_id__eq)

        if conditions:
            query = query.where(*conditions)
        return query

    async def get_app_conversation_info(
        self, conversation_id: UUID
    ) -> AppConversationInfo | None:
        """Get conversation info with user_id from SAAS metadata."""
        query = await self._secure_select_with_saas_metadata()
        query = query.where(
            StoredConversationMetadata.conversation_id == str(conversation_id)
        )
        result_set = await self.db_session.execute(query)
        result = result_set.first()
        if result:
            stored_metadata, saas_metadata = result
            # Fetch sub-conversation IDs
            sub_conversation_ids = await self.get_sub_conversation_ids(conversation_id)
            return self._to_info_with_user_id(
                stored_metadata,
                saas_metadata,
                sub_conversation_ids=sub_conversation_ids,
            )
        return None

    async def batch_get_app_conversation_info(
        self, conversation_ids: list[UUID]
    ) -> list[AppConversationInfo | None]:
        """Batch get conversation info with user_id from SAAS metadata."""
        conversation_id_strs = [
            str(conversation_id) for conversation_id in conversation_ids
        ]
        query = await self._secure_select_with_saas_metadata()
        query = query.where(
            StoredConversationMetadata.conversation_id.in_(conversation_id_strs)
        )
        result = await self.db_session.execute(query)
        rows = result.all()

        # Create a mapping of conversation_id to (metadata, saas_metadata)
        info_by_id = {}
        for stored_metadata, saas_metadata in rows:
            info_by_id[stored_metadata.conversation_id] = (
                stored_metadata,
                saas_metadata,
            )

        results: list[AppConversationInfo | None] = []
        for conversation_id in conversation_id_strs:
            if conversation_id in info_by_id:
                stored_metadata, saas_metadata = info_by_id[conversation_id]
                # Fetch sub-conversation IDs for each conversation
                sub_conversation_ids = await self.get_sub_conversation_ids(
                    UUID(conversation_id)
                )
                results.append(
                    self._to_info_with_user_id(
                        stored_metadata,
                        saas_metadata,
                        sub_conversation_ids=sub_conversation_ids,
                    )
                )
            else:
                results.append(None)

        return results

    async def try_reserve_app_conversation_id(
        self,
        conversation_id: UUID,
        created_by_user_id: str | None = None,
    ) -> bool:
        user_id_str = await self.user_context.get_user_id() or created_by_user_id
        if not user_id_str:
            return False
        user_id = UUID(user_id_str)
        user = await self.db_session.get(User, user_id)
        if user is None:
            raise AuthError()
        organization_id = await self._get_effective_org_id() or user.current_org_id
        if organization_id is None:
            raise AuthError()
        token = uuid4().hex
        if await self._insert_app_conversation_id_reservation_with_owner(
            conversation_id,
            user_id,
            organization_id,
            token,
        ):
            self._reservation_tokens[conversation_id] = token
            return True

        cutoff = utc_now() - APP_CONVERSATION_RESERVATION_TTL
        now = utc_now()
        result = cast(
            CursorResult,
            await self.db_session.execute(
                update(StoredConversationMetadata)
                .where(
                    StoredConversationMetadata.conversation_id == str(conversation_id),
                    StoredConversationMetadata.conversation_version
                    == APP_CONVERSATION_RESERVATION_VERSION,
                    StoredConversationMetadata.last_updated_at < cutoff,
                )
                .values(
                    tags={APP_CONVERSATION_RESERVATION_TOKEN_KEY: token},
                    last_updated_at=now,
                    created_at=now,
                )
            ),
        )
        if not result.rowcount:
            await self.db_session.rollback()
            return False
        await self.db_session.merge(
            StoredConversationMetadataSaas(
                conversation_id=str(conversation_id),
                user_id=user_id,
                org_id=organization_id,
            )
        )
        await self.db_session.commit()
        self._reservation_tokens[conversation_id] = token
        return True

    async def _insert_app_conversation_id_reservation_with_owner(
        self,
        conversation_id: UUID,
        user_id: UUID,
        organization_id: UUID,
        token: str,
    ) -> bool:
        self.db_session.add(
            StoredConversationMetadata(
                conversation_id=str(conversation_id),
                conversation_version=APP_CONVERSATION_RESERVATION_VERSION,
                tags={APP_CONVERSATION_RESERVATION_TOKEN_KEY: token},
            )
        )
        self.db_session.add(
            StoredConversationMetadataSaas(
                conversation_id=str(conversation_id),
                user_id=user_id,
                org_id=organization_id,
            )
        )
        try:
            await self.db_session.flush()
            await self.db_session.commit()
        except IntegrityError:
            await self.db_session.rollback()
            return False
        return True

    async def release_app_conversation_id_reservation(
        self, conversation_id: UUID
    ) -> None:
        token = self._reservation_tokens.get(conversation_id)
        await self.db_session.rollback()
        if token is None:
            return
        existing = (
            await self.db_session.execute(
                select(
                    StoredConversationMetadata.conversation_version,
                    StoredConversationMetadata.tags,
                )
                .where(
                    StoredConversationMetadata.conversation_id == str(conversation_id)
                )
                .with_for_update(of=StoredConversationMetadata)
            )
        ).first()
        if (
            existing is None
            or existing.conversation_version != APP_CONVERSATION_RESERVATION_VERSION
            or (existing.tags or {}).get(APP_CONVERSATION_RESERVATION_TOKEN_KEY)
            != token
        ):
            self._reservation_tokens.pop(conversation_id, None)
            await self.db_session.rollback()
            return
        await self.db_session.execute(
            delete(StoredConversationMetadataSaas).where(
                StoredConversationMetadataSaas.conversation_id == str(conversation_id)
            )
        )
        await self.db_session.execute(
            delete(StoredConversationMetadata).where(
                StoredConversationMetadata.conversation_id == str(conversation_id),
                StoredConversationMetadata.conversation_version
                == APP_CONVERSATION_RESERVATION_VERSION,
            )
        )
        await self.db_session.commit()
        self._reservation_tokens.pop(conversation_id, None)

    async def delete_app_conversation_info(self, conversation_id: UUID) -> bool:
        query = await self._secure_select_with_saas_metadata()
        existing = (
            await self.db_session.execute(
                query.where(
                    StoredConversationMetadata.conversation_id == str(conversation_id)
                ).with_for_update(of=StoredConversationMetadata)
            )
        ).first()
        if existing is None:
            await self.db_session.rollback()
            return False
        await self.db_session.execute(
            delete(StoredConversationMetadataSaas).where(
                StoredConversationMetadataSaas.conversation_id == str(conversation_id)
            )
        )
        await self.db_session.execute(
            delete(StoredConversationMetadata).where(
                StoredConversationMetadata.conversation_id == str(conversation_id)
            )
        )
        await self.db_session.commit()
        return True

    async def save_app_conversation_info(
        self,
        info: AppConversationInfo,
        allow_reservation_handoff: bool = False,
    ) -> AppConversationInfo:
        user_id_str = await self.user_context.get_user_id()
        if not user_id_str and info.created_by_user_id:
            user_id_str = info.created_by_user_id
        user_id = UUID(user_id_str) if user_id_str else None
        user = await self.db_session.get(User, user_id) if user_id else None
        if user_id is not None and user is None:
            raise AuthError()
        organization_id = (
            await self._get_effective_org_id() if user_id is not None else None
        )
        if organization_id is None and user is not None:
            organization_id = user.current_org_id

        existing = (
            await self.db_session.execute(
                select(
                    StoredConversationMetadata.sandbox_id,
                    StoredConversationMetadata.conversation_version,
                    StoredConversationMetadataSaas.user_id,
                    StoredConversationMetadataSaas.org_id,
                )
                .outerjoin(
                    StoredConversationMetadataSaas,
                    StoredConversationMetadata.conversation_id
                    == StoredConversationMetadataSaas.conversation_id,
                )
                .where(StoredConversationMetadata.conversation_id == str(info.id))
                .with_for_update(of=StoredConversationMetadata)
            )
        ).first()

        if existing is not None and self.user_context != ADMIN:
            if existing.user_id is None:
                if not isinstance(self.user_context, SandboxUserContext):
                    raise AuthError()
            elif existing.user_id != user_id:
                raise AuthError()
            if (
                existing.user_id is not None
                and not isinstance(self.user_context, SandboxUserContext)
                and (existing.org_id != organization_id)
            ):
                raise AuthError()

        if isinstance(self.user_context, SandboxUserContext):
            if not user_id_str:
                raise AuthError('Sandbox owner required')
            if existing is not None and existing.user_id is None:
                raise AuthError('Conversation organization is unavailable')
            if (
                info.sandbox_id != self.user_context.sandbox_id
                or info.created_by_user_id != user_id_str
            ):
                raise AuthError()
            if existing and (
                existing.sandbox_id not in (None, self.user_context.sandbox_id)
                or existing.user_id not in (None, user_id)
            ):
                raise AuthError()

        if existing is not None and existing.user_id is not None:
            user_id = existing.user_id
            organization_id = existing.org_id

        await self._merge_app_conversation_info(info, allow_reservation_handoff)
        if (existing is None or existing.user_id is None) and user_id is not None:
            assert organization_id is not None
            self.db_session.add(
                StoredConversationMetadataSaas(
                    conversation_id=str(info.id),
                    user_id=user_id,
                    org_id=organization_id,
                )
            )
        await self.db_session.commit()
        return info

    def _to_info_with_user_id(
        self,
        stored: StoredConversationMetadata,
        saas_metadata: StoredConversationMetadataSaas | None,
        sub_conversation_ids: list[UUID] | None = None,
    ) -> AppConversationInfo:
        """Convert stored metadata to AppConversationInfo with user_id from SAAS metadata."""
        # Use the base _to_info method to get the basic info
        info = self._to_info(stored, sub_conversation_ids=sub_conversation_ids)

        # Override the created_by_user_id with the user_id from SAAS metadata
        info.created_by_user_id = (
            str(saas_metadata.user_id)
            if saas_metadata is not None and saas_metadata.user_id
            else None
        )

        return info


class SaasAppConversationInfoServiceInjector(AppConversationInfoServiceInjector):
    """Enterprise injector for SQLAppConversationInfoService with SAAS filtering."""

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[AppConversationInfoService, None]:
        from openhands.app_server.config import (
            get_db_session,
            get_user_context,
        )

        async with (
            get_user_context(state, request) as user_context,
            get_db_session(state, request) as db_session,
        ):
            service = SaasSQLAppConversationInfoService(
                db_session=db_session, user_context=user_context
            )
            yield service
