"""
Store class for managing users.
"""

import uuid
from typing import Optional

from integrations.stripe_service import migrate_customer
from server.logger import logger
from sqlalchemy.orm import joinedload
from storage.database import session_maker
from storage.encrypt_utils import decrypt_model
from storage.lite_llm_manager import LiteLlmManager
from storage.org import Org
from storage.org_member import OrgMember
from storage.org_store import OrgStore
from storage.role_store import RoleStore
from storage.user import User
from storage.user_settings import UserSettings

from openhands.storage.data_models.settings import Settings


class UserStore:
    """Store for managing users."""

    @staticmethod
    async def create_user(
        keycloak_user_id: str,
        user_info: dict,
        role_id: Optional[int] = None,
    ) -> User | None:
        """Create a new user."""
        with session_maker() as session:
            # create personal org
            org = Org(
                id=uuid.UUID(keycloak_user_id),
                name=f'user_{keycloak_user_id}_org',
                contact_name=user_info['preferred_username'],
                contact_email=user_info['email'],
            )
            session.add(org)

            settings = await UserStore.create_default_settings(
                org_id=str(org.id), keycloak_user_id=keycloak_user_id
            )

            if not settings:
                return None

            org_kwargs = OrgStore.get_kwargs_from_settings(settings)
            for key, value in org_kwargs.items():
                if hasattr(org, key):
                    setattr(org, key, value)

            user_kwargs = UserStore.get_kwargs_from_settings(settings)
            user = User(
                id=uuid.UUID(keycloak_user_id),
                current_org_id=org.id,
                role_id=role_id,
                **user_kwargs,
            )
            session.add(user)

            role = RoleStore.get_role_by_name('admin')

            org_member = OrgMember(
                org_id=org.id,
                user_id=user.id,
                role_id=role.id,  # admin of your own org.
                llm_api_key=settings.llm_api_key,  # type: ignore[union-attr]
                status='active',
            )
            session.add(org_member)
            session.commit()
            session.refresh(user)
            user.org_members  # load org_members
            return user

    @staticmethod
    async def migrate_user(
        keycloak_user_id: str,
        user_settings: UserSettings,
        user_info: dict,
    ) -> User:
        if not keycloak_user_id or not user_settings:
            return None

        # Check if user is already migrated to prevent double migration
        if user_settings.migration_status is True:
            logger.warning(f'User {keycloak_user_id} already migrated, skipping')
            return UserStore.get_user_by_id(keycloak_user_id)
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
                id=uuid.UUID(keycloak_user_id),
                name=f'user_{keycloak_user_id}_org',
                contact_name=user_info['preferred_username'],
                contact_email=user_info['email'],
            )
            session.add(org)

            await LiteLlmManager.migrate_entries(
                str(org.id), keycloak_user_id, decrypted_user_settings
            )

            await migrate_customer(session, keycloak_user_id, org)

            org_kwargs = {
                c.name: getattr(decrypted_user_settings, c.name)
                for c in Org.__table__.columns
                if c.name != 'id' and hasattr(decrypted_user_settings, c.name)
            }
            for key, value in org_kwargs.items():
                if hasattr(org, key):
                    setattr(org, key, value)

            user_kwargs = {
                c.name: getattr(decrypted_user_settings, c.name)
                for c in User.__table__.columns
                if c.name != 'id' and hasattr(decrypted_user_settings, c.name)
            }
            user = User(
                id=uuid.UUID(keycloak_user_id),
                current_org_id=org.id,
                role_id=None,
                **user_kwargs,
            )
            session.add(user)

            role = RoleStore.get_role_by_name('admin')

            org_member = OrgMember(
                org_id=org.id,
                user_id=user.id,
                role_id=role.id,  # admin of your own org.
                llm_api_key=decrypted_user_settings.llm_api_key,  # type: ignore[union-attr]
                status='active',
            )
            session.add(org_member)

            # Mark the old user_settings as migrated instead of deleting
            user_settings.migration_status = True
            session.commit()
            session.refresh(user)
            user.org_members  # load org_members
            return user

    @staticmethod
    def get_user_by_id(keycloak_user_id: str) -> Optional[User]:
        """Get user by Keycloak user ID."""
        with session_maker() as session:
            return (
                session.query(User)
                .options(joinedload(User.org_members))
                .filter(User.id == uuid.UUID(keycloak_user_id))
                .first()
            )

    @staticmethod
    def list_users() -> list[User]:
        """List all users."""
        with session_maker() as session:
            return session.query(User).all()

    @staticmethod
    async def create_default_settings(
        org_id: str, keycloak_user_id: str
    ) -> Optional[Settings]:
        logger.info(
            'UserStore:create_default_settings:start',
            extra={'org_id': org_id, 'user_id': keycloak_user_id},
        )
        # You must log in before you get default settings
        if not org_id:
            return None

        settings = Settings(language='en', enable_proactive_conversation_starters=True)

        settings = await LiteLlmManager.create_entries(
            org_id, keycloak_user_id, settings
        )
        if not settings:
            logger.info(
                'UserStore:create_default_settings:litellm_create_failed',
                extra={'org_id': org_id},
            )
            return None

        return settings

    @staticmethod
    def get_kwargs_from_settings(settings: Settings):
        kwargs = {
            c.name: getattr(settings, normalized)
            for c in User.__table__.columns
            if (normalized := c.name.lstrip('_')) and hasattr(settings, normalized)
        }
        return kwargs
