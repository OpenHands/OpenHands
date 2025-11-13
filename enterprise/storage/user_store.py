"""
Store class for managing users.
"""

import uuid
from typing import Optional

from server.logger import logger
from sqlalchemy import text
from sqlalchemy.orm import joinedload
from storage.database import session_maker
from storage.encrypt_utils import decrypt_model
from storage.org import Org
from storage.org_member import OrgMember
from storage.role_store import RoleStore
from storage.user import User
from storage.user_settings import UserSettings


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

            role = RoleStore.get_role_by_name('admin')

            from storage.org_member_store import OrgMemberStore

            org_member_kwargs = OrgMemberStore.get_kwargs_from_settings(settings)
            org_member = OrgMember(
                org_id=org.id,
                user_id=user.id,
                role_id=role.id,  # admin of your own org.
                status='active',
                **org_member_kwargs,
            )
            session.add(org_member)
            session.commit()
            session.refresh(user)
            user.org_members  # load org_members
            return user

    @staticmethod
    async def migrate_user(
        user_id: str,
        user_settings: UserSettings,
        user_info: dict,
    ) -> User:
        if not user_id or not user_settings:
            return None

        # Check if user is already migrated to prevent double migration
        if user_settings.migration_status is True:
            logger.warning(f'User {user_id} already migrated, skipping')
            return UserStore.get_user_by_id(user_id)
        kwargs = decrypt_model(
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
                contact_name=user_info['preferred_username'],
                contact_email=user_info['email'],
            )
            session.add(org)

            from storage.lite_llm_manager import LiteLlmManager

            await LiteLlmManager.migrate_entries(
                str(org.id), user_id, decrypted_user_settings
            )

            # avoids circular reference. This migrate method is temprorary until all users are migrated.
            from integrations.stripe_service import migrate_customer

            await migrate_customer(session, user_id, org)

            from storage.org_store import OrgStore

            org_kwargs = OrgStore.get_kwargs_from_user_settings(decrypted_user_settings)
            org_kwargs.pop('id', None)
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

            role = RoleStore.get_role_by_name('admin')

            from storage.org_member_store import OrgMemberStore

            org_member_kwargs = OrgMemberStore.get_kwargs_from_user_settings(
                decrypted_user_settings
            )
            org_member = OrgMember(
                org_id=org.id,
                user_id=user.id,
                role_id=role.id,  # admin of your own org.
                status='active',
                **org_member_kwargs,
            )
            session.add(org_member)

            # Mark the old user_settings as migrated instead of deleting
            user_settings.migration_status = True
            session.merge(user_settings)
            session.flush()

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
            return user

    @staticmethod
    def get_user_by_id(user_id: str) -> Optional[User]:
        """Get user by Keycloak user ID."""
        with session_maker() as session:
            return (
                session.query(User)
                .options(joinedload(User.org_members))
                .filter(User.id == uuid.UUID(user_id))
                .first()
            )

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
        org_id: str, user_id: str
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

        settings = await LiteLlmManager.create_entries(org_id, user_id, settings)
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
