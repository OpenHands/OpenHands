"""
Store class for managing organizations.
"""

from typing import Optional
from uuid import UUID

from server.constants import (
    LITE_LLM_API_URL,
    ORG_SETTINGS_VERSION,
    get_default_litellm_model,
)
from server.routes.org_models import OrgLLMSettingsUpdate, OrphanedUserError
from sqlalchemy import select, text
from sqlalchemy.orm import joinedload
from storage.database import a_session_maker, session_maker
from storage.lite_llm_manager import LiteLlmManager
from storage.org import Org
from storage.org_member import OrgMember
from storage.user import User
from storage.user_settings import UserSettings

from openhands.core.logger import openhands_logger as logger
from openhands.storage.data_models.settings import Settings


class OrgStore:
    """Store for managing organizations."""

    @staticmethod
    def create_org(
        kwargs: dict,
    ) -> Org:
        """Create a new organization."""
        with session_maker() as session:
            org = Org(**kwargs)
            org.org_version = ORG_SETTINGS_VERSION
            org.default_llm_model = get_default_litellm_model()
            session.add(org)
            session.commit()
            session.refresh(org)
            return org

    @staticmethod
    def get_org_by_id(org_id: UUID) -> Org | None:
        """Get organization by ID."""
        org = None
        with session_maker() as session:
            org = session.query(Org).filter(Org.id == org_id).first()
        return OrgStore._validate_org_version(org)

    @staticmethod
    def get_current_org_from_keycloak_user_id(keycloak_user_id: str) -> Org | None:
        with session_maker() as session:
            user = (
                session.query(User)
                .options(joinedload(User.org_members))
                .filter(User.id == UUID(keycloak_user_id))
                .first()
            )
            if not user:
                logger.warning(f'User not found for ID {keycloak_user_id}')
                return None
            org_id = user.current_org_id
            org = session.query(Org).filter(Org.id == org_id).first()
            if not org:
                logger.warning(
                    f'Org not found for ID {org_id} as the current org for user {keycloak_user_id}'
                )
                return None
            return OrgStore._validate_org_version(org)

    @staticmethod
    def get_org_by_name(name: str) -> Org | None:
        """Get organization by name."""
        org = None
        with session_maker() as session:
            org = session.query(Org).filter(Org.name == name).first()
        return OrgStore._validate_org_version(org)

    @staticmethod
    def _validate_org_version(org: Org) -> Org | None:
        """Check if we need to update org version."""
        if org and org.org_version < ORG_SETTINGS_VERSION:
            org = OrgStore.update_org(
                org.id,
                {
                    'org_version': ORG_SETTINGS_VERSION,
                    'default_llm_model': get_default_litellm_model(),
                    'llm_base_url': LITE_LLM_API_URL,
                },
            )
        return org

    @staticmethod
    def list_orgs() -> list[Org]:
        """List all organizations."""
        with session_maker() as session:
            orgs = session.query(Org).all()
            return orgs

    @staticmethod
    def get_user_orgs_paginated(
        user_id: UUID, page_id: str | None = None, limit: int = 100
    ) -> tuple[list[Org], str | None]:
        """
        Get paginated list of organizations for a user.

        Args:
            user_id: User UUID
            page_id: Optional page ID (offset as string) for pagination
            limit: Maximum number of organizations to return

        Returns:
            Tuple of (list of Org objects, next_page_id or None)
        """
        with session_maker() as session:
            # Build query joining OrgMember with Org
            query = (
                session.query(Org)
                .join(OrgMember, Org.id == OrgMember.org_id)
                .filter(OrgMember.user_id == user_id)
                .order_by(Org.name)
            )

            # Apply pagination offset
            if page_id is not None:
                try:
                    offset = int(page_id)
                    query = query.offset(offset)
                except ValueError:
                    # If page_id is not a valid integer, start from beginning
                    offset = 0
            else:
                offset = 0

            # Fetch limit + 1 to check if there are more results
            query = query.limit(limit + 1)
            orgs = query.all()

            # Check if there are more results
            has_more = len(orgs) > limit
            if has_more:
                orgs = orgs[:limit]

            # Calculate next page ID
            next_page_id = None
            if has_more:
                next_page_id = str(offset + limit)

            # Validate org versions
            validated_orgs = [
                OrgStore._validate_org_version(org) for org in orgs if org
            ]
            validated_orgs = [org for org in validated_orgs if org is not None]

            return validated_orgs, next_page_id

    @staticmethod
    def update_org(
        org_id: UUID,
        kwargs: dict,
    ) -> Optional[Org]:
        """Update organization details."""
        with session_maker() as session:
            org = session.query(Org).filter(Org.id == org_id).first()
            if not org:
                return None

            if 'id' in kwargs:
                kwargs.pop('id')
            for key, value in kwargs.items():
                if hasattr(org, key):
                    setattr(org, key, value)

            session.commit()
            session.refresh(org)
            return org

    @staticmethod
    def get_kwargs_from_settings(settings: Settings):
        kwargs = {}

        for c in Org.__table__.columns:
            # Normalize for lookup
            normalized = (
                c.name.removeprefix('_default_').removeprefix('default_').lstrip('_')
            )

            if not hasattr(settings, normalized):
                continue

            # ---- FIX: Output key should drop *only* leading "_" but preserve "default" ----
            key = c.name
            if key.startswith('_'):
                key = key[1:]  # remove only the very first leading underscore

            kwargs[key] = getattr(settings, normalized)

        return kwargs

    @staticmethod
    def get_kwargs_from_user_settings(user_settings: UserSettings):
        kwargs = {}

        for c in Org.__table__.columns:
            # Normalize for lookup
            normalized = (
                c.name.removeprefix('_default_').removeprefix('default_').lstrip('_')
            )

            if not hasattr(user_settings, normalized):
                continue

            # ---- FIX: Output key should drop *only* leading "_" but preserve "default" ----
            key = c.name
            if key.startswith('_'):
                key = key[1:]  # remove only the very first leading underscore

            kwargs[key] = getattr(user_settings, normalized)

        kwargs['org_version'] = user_settings.user_version
        return kwargs

    @staticmethod
    def persist_org_with_owner(
        org: Org,
        org_member: OrgMember,
    ) -> Org:
        """
        Persist organization and owner membership in a single transaction.

        Args:
            org: Organization entity to persist
            org_member: Organization member entity to persist

        Returns:
            Org: The persisted organization object

        Raises:
            Exception: If database operations fail
        """
        with session_maker() as session:
            session.add(org)
            session.add(org_member)
            session.commit()
            session.refresh(org)
            return org

    @staticmethod
    async def delete_org_cascade(org_id: UUID) -> Org | None:
        """
        Delete organization and all associated data in cascade, including external LiteLLM cleanup.

        Args:
            org_id: UUID of the organization to delete

        Returns:
            Org: The deleted organization object, or None if not found

        Raises:
            Exception: If database operations or LiteLLM cleanup fail
        """
        with session_maker() as session:
            # First get the organization to return it
            org = session.query(Org).filter(Org.id == org_id).first()
            if not org:
                return None

            try:
                # 1. Delete conversation data for organization conversations
                session.execute(
                    text("""
                    DELETE FROM conversation_metadata
                    WHERE conversation_id IN (
                        SELECT conversation_id FROM conversation_metadata_saas WHERE org_id = :org_id
                    )
                    """),
                    {'org_id': str(org_id)},
                )

                session.execute(
                    text("""
                    DELETE FROM app_conversation_start_task
                    WHERE app_conversation_id::text IN (
                        SELECT conversation_id FROM conversation_metadata_saas WHERE org_id = :org_id
                    )
                    """),
                    {'org_id': str(org_id)},
                )

                # 2. Delete organization-owned data tables (direct org_id foreign keys)
                session.execute(
                    text('DELETE FROM billing_sessions WHERE org_id = :org_id'),
                    {'org_id': str(org_id)},
                )
                session.execute(
                    text(
                        'DELETE FROM conversation_metadata_saas WHERE org_id = :org_id'
                    ),
                    {'org_id': str(org_id)},
                )
                session.execute(
                    text('DELETE FROM custom_secrets WHERE org_id = :org_id'),
                    {'org_id': str(org_id)},
                )
                session.execute(
                    text('DELETE FROM api_keys WHERE org_id = :org_id'),
                    {'org_id': str(org_id)},
                )
                session.execute(
                    text('DELETE FROM slack_conversation WHERE org_id = :org_id'),
                    {'org_id': str(org_id)},
                )
                session.execute(
                    text('DELETE FROM slack_users WHERE org_id = :org_id'),
                    {'org_id': str(org_id)},
                )
                session.execute(
                    text('DELETE FROM stripe_customers WHERE org_id = :org_id'),
                    {'org_id': str(org_id)},
                )

                # 3. Handle users with this as current_org_id BEFORE deleting memberships
                # Single query to find orphaned users (those with no alternative org)
                orphaned_users = session.execute(
                    text("""
                        SELECT u.id
                        FROM "user" u
                        WHERE u.current_org_id = :org_id
                        AND NOT EXISTS (
                            SELECT 1 FROM org_member om
                            WHERE om.user_id = u.id AND om.org_id != :org_id
                        )
                    """),
                    {'org_id': str(org_id)},
                ).fetchall()

                if orphaned_users:
                    raise OrphanedUserError([str(row[0]) for row in orphaned_users])

                # Batch update: reassign current_org_id to an alternative org for all affected users
                session.execute(
                    text("""
                        UPDATE "user" u
                        SET current_org_id = (
                            SELECT om.org_id FROM org_member om
                            WHERE om.user_id = u.id AND om.org_id != :org_id
                            LIMIT 1
                        )
                        WHERE u.current_org_id = :org_id
                    """),
                    {'org_id': str(org_id)},
                )

                # 4. Delete organization memberships (now safe)
                session.execute(
                    text('DELETE FROM org_member WHERE org_id = :org_id'),
                    {'org_id': str(org_id)},
                )

                # 5. Finally delete the organization
                session.delete(org)

                # 6. Clean up LiteLLM team before committing transaction
                logger.info(
                    'Deleting LiteLLM team within database transaction',
                    extra={'org_id': str(org_id)},
                )
                await LiteLlmManager.delete_team(str(org_id))

                # 7. Commit all changes only if everything succeeded
                session.commit()

                logger.info(
                    'Successfully deleted organization and all associated data including LiteLLM team',
                    extra={'org_id': str(org_id), 'org_name': org.name},
                )

                return org

            except Exception as e:
                session.rollback()
                logger.error(
                    'Failed to delete organization - transaction rolled back',
                    extra={'org_id': str(org_id), 'error': str(e)},
                )
                raise

    @staticmethod
    async def get_org_by_id_async(org_id: UUID) -> Org | None:
        """Get organization by ID (async version)."""
        async with a_session_maker() as session:
            result = await session.execute(select(Org).filter(Org.id == org_id))
            org = result.scalars().first()
        return OrgStore._validate_org_version(org) if org else None

    @staticmethod
    async def update_org_llm_settings_async(
        org_id: UUID,
        llm_settings: OrgLLMSettingsUpdate,
    ) -> Org | None:
        """Update organization LLM settings and propagate to members (async version).

        Args:
            org_id: Organization ID
            llm_settings: Typed LLM settings update model

        Returns:
            Updated Org or None if not found
        """
        from storage.org_member_store import OrgMemberStore

        async with a_session_maker() as session:
            result = await session.execute(select(Org).filter(Org.id == org_id))
            org = result.scalars().first()
            if not org:
                return None

            # Apply updates to org
            llm_settings.apply_to_org(org)

            # Propagate relevant settings to all org members
            member_updates = llm_settings.get_member_updates()
            if member_updates:
                await OrgMemberStore.update_all_members_llm_settings_async(
                    session, org_id, member_updates
                )

            await session.commit()
            await session.refresh(org)
            return org
