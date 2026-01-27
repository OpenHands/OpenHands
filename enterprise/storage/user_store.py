"""
Store class for managing users.
"""

import asyncio
import uuid
from typing import Optional

from server.auth.token_manager import TokenManager
from server.constants import (
    LITE_LLM_API_URL,
    ORG_SETTINGS_VERSION,
    PERSONAL_WORKSPACE_VERSION_TO_MODEL,
    get_default_litellm_model,
)
from server.logger import logger
from sqlalchemy import select, text
from sqlalchemy.orm import joinedload
from storage.database import a_session_maker, session_maker
from storage.encrypt_utils import decrypt_legacy_model
from storage.org import Org
from storage.org_member import OrgMember
from storage.role_store import RoleStore
from storage.user import User
from storage.user_settings import UserSettings

from openhands.utils.async_utils import GENERAL_TIMEOUT, call_async_from_sync

# The max possible time to wait for another process to finish creating a user before retrying
_REDIS_CREATE_TIMEOUT_SECONDS = 30
# The delay to wait for another process to finish creating a user before trying to load again
_RETRY_LOAD_DELAY_SECONDS = 2
# Redis key prefix for user creation locks
_REDIS_USER_CREATION_KEY_PREFIX = 'create_user:'


class UserStore:
    """Store for managing users."""

    @staticmethod
    async def create_user(
        user_id: str,
        user_info: dict,
        role_id: Optional[int] = None,
    ) -> User | None:
        """Create a new user."""
        with session_maker() as session:
            # create personal org
            org = Org(
                id=uuid.UUID(user_id),
                name=f'user_{user_id}_org',
                contact_name=user_info['preferred_username'],
                contact_email=user_info['email'],
                v1_enabled=True,
            )
            session.add(org)

            settings = await UserStore.create_default_settings(
                org_id=str(org.id), user_id=user_id
            )

            if not settings:
                return None

            from storage.org_store import OrgStore

            org_kwargs = OrgStore.get_kwargs_from_settings(settings)
            for key, value in org_kwargs.items():
                if hasattr(org, key):
                    setattr(org, key, value)

            user_kwargs = UserStore.get_kwargs_from_settings(settings)
            user = User(
                id=uuid.UUID(user_id),
                current_org_id=org.id,
                role_id=role_id,
                **user_kwargs,
            )
            session.add(user)

            role = RoleStore.get_role_by_name('owner')

            from storage.org_member_store import OrgMemberStore

            org_member_kwargs = OrgMemberStore.get_kwargs_from_settings(settings)
            # avoid setting org member llm fields to use org defaults on user creation
            del org_member_kwargs['llm_model']
            del org_member_kwargs['llm_base_url']
            org_member = OrgMember(
                org_id=org.id,
                user_id=user.id,
                role_id=role.id,  # owner of your own org.
                status='active',
                **org_member_kwargs,
            )
            session.add(org_member)
            session.commit()
            session.refresh(user)
            user.org_members  # load org_members
            return user

    @staticmethod
    def _get_redis_client():
        """Get the Redis client from the Socket.IO manager."""
        from openhands.server.shared import sio

        return getattr(sio.manager, 'redis', None)

    @staticmethod
    async def _acquire_user_creation_lock(user_id: str) -> bool:
        """Attempt to acquire a distributed lock for user creation.

        Returns True if the lock was acquired or if Redis is unavailable (fallback to no locking).
        Returns False if another process holds the lock.
        """
        redis_client = UserStore._get_redis_client()
        if redis_client is None:
            logger.warning(
                'user_store:_acquire_user_creation_lock:no_redis_client',
                extra={'user_id': user_id},
            )
            return True  # Proceed without locking if Redis is unavailable

        user_key = f'{_REDIS_USER_CREATION_KEY_PREFIX}{user_id}'
        lock_acquired = await redis_client.set(
            user_key, 1, nx=True, ex=_REDIS_CREATE_TIMEOUT_SECONDS
        )
        return bool(lock_acquired)

    @staticmethod
    async def migrate_user(
        user_id: str,
        user_settings: UserSettings,
        user_info: dict,
    ) -> User:
        if not user_id or not user_settings:
            return None

        kwargs = decrypt_legacy_model(
            [
                'llm_api_key',
                'llm_api_key_for_byor',
                'search_api_key',
                'sandbox_api_key',
            ],
            user_settings,
        )
        decrypted_user_settings = UserSettings(**kwargs)
        with session_maker() as session:
            # create personal org
            org = Org(
                id=uuid.UUID(user_id),
                name=f'user_{user_id}_org',
                org_version=user_settings.user_version,
                contact_name=user_info['username'],
                contact_email=user_info['email'],
            )
            session.add(org)

            from storage.lite_llm_manager import LiteLlmManager

            logger.debug(
                'user_store:migrate_user:calling_litellm_migrate_entries',
                extra={'user_id': user_id},
            )
            await LiteLlmManager.migrate_entries(
                str(org.id),
                user_id,
                decrypted_user_settings,
            )

            logger.debug(
                'user_store:migrate_user:done_litellm_migrate_entries',
                extra={'user_id': user_id},
            )
            custom_settings = UserStore._has_custom_settings(
                decrypted_user_settings, user_settings.user_version
            )

            # avoids circular reference. This migrate method is temprorary until all users are migrated.
            from integrations.stripe_service import migrate_customer

            logger.debug(
                'user_store:migrate_user:calling_stripe_migrate_customer',
                extra={'user_id': user_id},
            )
            await migrate_customer(session, user_id, org)
            logger.debug(
                'user_store:migrate_user:done_stripe_migrate_customer',
                extra={'user_id': user_id},
            )

            from storage.org_store import OrgStore

            org_kwargs = OrgStore.get_kwargs_from_user_settings(decrypted_user_settings)
            org_kwargs.pop('id', None)

            # if user has custom settings, set org defaults to current version
            if custom_settings:
                org_kwargs['default_llm_model'] = get_default_litellm_model()
                org_kwargs['llm_base_url'] = LITE_LLM_API_URL
                org_kwargs['org_version'] = ORG_SETTINGS_VERSION

            for key, value in org_kwargs.items():
                if hasattr(org, key):
                    setattr(org, key, value)

            user_kwargs = UserStore.get_kwargs_from_user_settings(
                decrypted_user_settings
            )
            user_kwargs.pop('id', None)
            user = User(
                id=uuid.UUID(user_id),
                current_org_id=org.id,
                role_id=None,
                **user_kwargs,
            )
            session.add(user)

            logger.debug(
                'user_store:migrate_user:calling_get_role_by_name',
                extra={'user_id': user_id},
            )
            role = await RoleStore.get_role_by_name_async('owner')
            logger.debug(
                'user_store:migrate_user:done_get_role_by_name',
                extra={'user_id': user_id},
            )

            from storage.org_member_store import OrgMemberStore

            org_member_kwargs = OrgMemberStore.get_kwargs_from_user_settings(
                decrypted_user_settings
            )

            # if the user did not have custom settings in the old model,
            # then use the org defaults by not setting org_member fields
            if not custom_settings:
                del org_member_kwargs['llm_model']
                del org_member_kwargs['llm_base_url']
                del org_member_kwargs['llm_api_key_for_byor']

            org_member = OrgMember(
                org_id=org.id,
                user_id=user.id,
                role_id=role.id,  # owner of your own org.
                status='active',
                **org_member_kwargs,
            )
            session.add(org_member)

            # Mark the old user_settings as migrated instead of deleting
            user_settings.already_migrated = True
            session.merge(user_settings)
            session.flush()
            logger.debug(
                'user_store:migrate_user:session_flush_complete',
                extra={'user_id': user_id},
            )

            # need to migrate conversation metadata
            session.execute(
                text("""
                    INSERT INTO conversation_metadata_saas (conversation_id, user_id, org_id)
                    SELECT
                        conversation_id,
                        :user_id,
                        :user_id
                    FROM conversation_metadata
                    WHERE user_id = :user_id
                """),
                {'user_id': user_id},
            )

            # Update org_id for tables that had org_id added
            user_uuid = uuid.UUID(user_id)

            # Update stripe_customers
            session.execute(
                text(
                    'UPDATE stripe_customers SET org_id = :org_id WHERE keycloak_user_id = :user_id'
                ),
                {'org_id': user_uuid, 'user_id': user_uuid},
            )

            # Update slack_users
            session.execute(
                text(
                    'UPDATE slack_users SET org_id = :org_id WHERE keycloak_user_id = :user_id'
                ),
                {'org_id': user_uuid, 'user_id': user_uuid},
            )

            # Update slack_conversation
            session.execute(
                text(
                    'UPDATE slack_conversation SET org_id = :org_id WHERE keycloak_user_id = :user_id'
                ),
                {'org_id': user_uuid, 'user_id': user_uuid},
            )

            # Update api_keys
            session.execute(
                text('UPDATE api_keys SET org_id = :org_id WHERE user_id = :user_id'),
                {'org_id': user_uuid, 'user_id': user_uuid},
            )

            # Update custom_secrets
            session.execute(
                text(
                    'UPDATE custom_secrets SET org_id = :org_id WHERE keycloak_user_id = :user_id'
                ),
                {'org_id': user_uuid, 'user_id': user_uuid},
            )

            # Update billing_sessions
            session.execute(
                text(
                    'UPDATE billing_sessions SET org_id = :org_id WHERE user_id = :user_id'
                ),
                {'org_id': user_uuid, 'user_id': user_uuid},
            )

            session.commit()
            session.refresh(user)
            user.org_members  # load org_members
            logger.debug(
                'user_store:migrate_user:session_committed',
                extra={'user_id': user_id},
            )
            return user

    @staticmethod
    def get_user_by_id(user_id: str) -> Optional[User]:
        """Get user by Keycloak user ID (sync version).

        Note: This method uses call_async_from_sync internally which creates a new
        event loop. If you're already in an async context, use get_user_by_id_async
        instead to avoid event loop conflicts.
        """
        with session_maker() as session:
            user = (
                session.query(User)
                .options(joinedload(User.org_members))
                .filter(User.id == uuid.UUID(user_id))
                .first()
            )
            if user:
                return user

            # Check if we need to migrate from user_settings
            while not call_async_from_sync(
                UserStore._acquire_user_creation_lock, GENERAL_TIMEOUT, user_id
            ):
                # The user is already being created in another thread / process
                logger.info(
                    'user_store:create_default_settings:waiting_for_lock',
                    extra={'user_id': user_id},
                )
                call_async_from_sync(
                    asyncio.sleep, GENERAL_TIMEOUT, _RETRY_LOAD_DELAY_SECONDS
                )

            # Check for user again as migration could have happened while trying to get the lock.
            user = (
                session.query(User)
                .options(joinedload(User.org_members))
                .filter(User.id == uuid.UUID(user_id))
                .first()
            )
            if user:
                return user

            user_settings = (
                session.query(UserSettings)
                .filter(
                    UserSettings.keycloak_user_id == user_id,
                    UserSettings.already_migrated.is_(False),
                )
                .first()
            )
            if user_settings:
                token_manager = TokenManager()
                user_info = call_async_from_sync(
                    token_manager.get_user_info_from_user_id,
                    GENERAL_TIMEOUT,
                    user_id,
                )
                user = call_async_from_sync(
                    UserStore.migrate_user,
                    GENERAL_TIMEOUT,
                    user_id,
                    user_settings,
                    user_info,
                )
                return user
            else:
                return None

    @staticmethod
    async def get_user_by_id_async(user_id: str) -> Optional[User]:
        """Get user by Keycloak user ID (async version).

        This is the preferred method when calling from an async context as it
        avoids event loop conflicts that can occur with the sync version.
        """
        async with a_session_maker() as session:
            result = await session.execute(
                select(User)
                .options(joinedload(User.org_members))
                .filter(User.id == uuid.UUID(user_id))
            )
            user = result.scalars().first()
            if user:
                return user

            # Check if we need to migrate from user_settings
            while not await UserStore._acquire_user_creation_lock(user_id):
                # The user is already being created in another thread / process
                logger.info(
                    'user_store:get_user_by_id_async:waiting_for_lock',
                    extra={'user_id': user_id},
                )
                await asyncio.sleep(_RETRY_LOAD_DELAY_SECONDS)

            # Check for user again as migration could have happened while trying to get the lock.
            result = await session.execute(
                select(User)
                .options(joinedload(User.org_members))
                .filter(User.id == uuid.UUID(user_id))
            )
            user = result.scalars().first()
            if user:
                return user

            logger.info(
                'user_store:get_user_by_id_async:start_migration',
                extra={'user_id': user_id},
            )
            result = await session.execute(
                select(UserSettings).filter(
                    UserSettings.keycloak_user_id == user_id,
                    UserSettings.already_migrated.is_(False),
                )
            )
            user_settings = result.scalars().first()
            if user_settings:
                token_manager = TokenManager()
                user_info = await token_manager.get_user_info_from_user_id(user_id)
                logger.info(
                    'user_store:get_user_by_id_async:calling_migrate_user',
                    extra={'user_id': user_id},
                )
                user = await UserStore.migrate_user(
                    user_id,
                    user_settings,
                    user_info,
                )
                return user
            else:
                return None

    @staticmethod
    def list_users() -> list[User]:
        """List all users."""
        with session_maker() as session:
            return session.query(User).all()

    # Prevent circular imports
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from openhands.storage.data_models.settings import Settings

    @staticmethod
    async def create_default_settings(
        org_id: str, user_id: str, create_user: bool = True
    ) -> Optional['Settings']:
        logger.info(
            'UserStore:create_default_settings:start',
            extra={'org_id': org_id, 'user_id': user_id},
        )
        # You must log in before you get default settings
        if not org_id:
            return None

        from openhands.storage.data_models.settings import Settings

        settings = Settings(language='en', enable_proactive_conversation_starters=True)

        from storage.lite_llm_manager import LiteLlmManager

        settings = await LiteLlmManager.create_entries(
            org_id, user_id, settings, create_user
        )
        if not settings:
            logger.info(
                'UserStore:create_default_settings:litellm_create_failed',
                extra={'org_id': org_id},
            )
            return None

        return settings

    @staticmethod
    def get_kwargs_from_settings(settings: 'Settings'):
        kwargs = {
            normalized: getattr(settings, normalized)
            for c in User.__table__.columns
            if (normalized := c.name.lstrip('_')) and hasattr(settings, normalized)
        }
        return kwargs

    @staticmethod
    def get_kwargs_from_user_settings(user_settings: UserSettings):
        kwargs = {
            normalized: getattr(user_settings, normalized)
            for c in User.__table__.columns
            if (normalized := c.name.lstrip('_')) and hasattr(user_settings, normalized)
        }
        return kwargs

    @staticmethod
    def _has_custom_settings(
        user_settings: UserSettings, old_user_version: int | None
    ) -> bool:
        """
        Check if user has custom LLM settings that should be preserved.
        Returns True if user customized either model or base_url.

        Args:
            settings: The user's current settings
            old_user_version: The user's old settings version, if any

        Returns:
            True if user has custom settings, False if using old defaults
        """
        # Normalize values
        user_model = (
            user_settings.llm_model.strip() or None if user_settings.llm_model else None
        )
        user_base_url = (
            user_settings.llm_base_url.strip() or None
            if user_settings.llm_base_url
            else None
        )

        # Custom base_url = definitely custom settings (BYOK)
        if user_base_url and user_base_url != LITE_LLM_API_URL:
            return True

        # No model set = using defaults
        if not user_model:
            return False

        # Check if model matches old version's default
        if (
            old_user_version
            and old_user_version <= ORG_SETTINGS_VERSION
            and old_user_version in PERSONAL_WORKSPACE_VERSION_TO_MODEL
        ):
            old_default_base = PERSONAL_WORKSPACE_VERSION_TO_MODEL[old_user_version]
            user_model_base = user_model.split('/')[-1]
            if user_model_base == old_default_base:
                return False  # Matches old default

        return True  # Custom model
