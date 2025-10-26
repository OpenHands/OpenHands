from __future__ import annotations

import binascii
import hashlib
import uuid
from base64 import b64decode, b64encode
from dataclasses import dataclass

from cryptography.fernet import Fernet
from pydantic import SecretStr
from server.logger import logger
from sqlalchemy.orm import joinedload, sessionmaker
from storage.database import session_maker
from storage.org import Org
from storage.org_member import OrgMember
from storage.org_store import OrgStore
from storage.user import User
from storage.user_settings import UserSettings
from storage.user_store import UserStore

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.server.settings import Settings
from openhands.storage.settings.settings_store import SettingsStore as OssSettingsStore


@dataclass
class SaasSettingsStore(OssSettingsStore):
    user_id: str
    session_maker: sessionmaker
    config: OpenHandsConfig
    ENCRYPT_VALUES = ['llm_api_key', 'llm_api_key_for_byor', 'search_api_key']

    def get_user_settings_by_keycloak_id(
        self, keycloak_user_id: str, session=None
    ) -> UserSettings | None:
        """
        Get UserSettings by keycloak_user_id.

        Args:
            keycloak_user_id: The keycloak user ID to search for
            session: Optional existing database session. If not provided, creates a new one.

        Returns:
            UserSettings object if found, None otherwise
        """
        if not keycloak_user_id:
            return None

        def _get_settings():
            if session:
                # Use provided session
                return (
                    session.query(UserSettings)
                    .filter(UserSettings.keycloak_user_id == keycloak_user_id)
                    .first()
                )
            else:
                # Create new session
                with self.session_maker() as new_session:
                    return (
                        new_session.query(UserSettings)
                        .filter(UserSettings.keycloak_user_id == keycloak_user_id)
                        .first()
                    )

        return _get_settings()

    async def load(self) -> Settings | None:
        user = UserStore.get_user_by_id(self.user_id)
        if not user:
            # Check if we need to migrate from user_settings
            user_settings = None
            with session_maker() as session:
                user_settings = (
                    session.query(UserSettings)
                    .filter(
                        UserSettings.keycloak_user_id == self.user_id,
                        UserSettings.migration_status.is_(False),
                    )
                    .first()
                )
            if user_settings:
                user = await UserStore.migrate_user(self.user_id, user_settings)
            else:
                logger.error(f'User not found for ID {self.user_id}')
                return None

        org_id = user.current_org_id
        org_member: OrgMember = None
        for om in user.org_members:
            if om.org_id == org_id:
                org_member = om
                break
        if not org_member or not org_member.llm_api_key:
            return None
        org = OrgStore.get_org_by_id(org_id)
        if not org:
            logger.error(
                f'Org not found for ID {org_id} as the current org for user {self.user_id}'
            )
            return None
        kwargs = {
            **{
                normalized: getattr(org, c.name)
                for c in Org.__table__.columns
                if (
                    normalized := c.name.removeprefix('_default_')
                    .removeprefix('default_')
                    .lstrip('_')
                )
                in Settings.model_fields
            },
            **{
                normalized: getattr(user, c.name)
                for c in User.__table__.columns
                if (normalized := c.name.lstrip('_')) in Settings.model_fields
            },
        }
        kwargs['llm_api_key'] = org_member.llm_api_key
        if org_member.max_iterations:
            kwargs['max_iterations'] = org_member.max_iterations
        if org_member.llm_model:
            kwargs['llm_model'] = org_member.llm_model
        if org_member.llm_api_key_for_byor:
            kwargs['llm_api_key_for_byor'] = org_member.llm_api_key_for_byor
        if org_member.llm_base_url:
            kwargs['llm_base_url'] = org_member.llm_base_url

        settings = Settings(**kwargs)
        return settings

    async def store(self, item: Settings):
        # Call the static store method from SettingsStore
        with self.session_maker() as session:
            if not item:
                return None
            kwargs = item.model_dump(context={'expose_secrets': True})
            user = (
                session.query(User)
                .options(joinedload(User.org_members))
                .filter(User.id == uuid.UUID(self.user_id))
            ).first()

            if not user:
                # Check if we need to migrate from user_settings
                user_settings = None
                with session_maker() as session:
                    user_settings = self.get_user_settings_by_keycloak_id(
                        self.user_id, session
                    )
                if user_settings:
                    user = await UserStore.migrate_user(self.user_id, user_settings)
                else:
                    logger.error(f'User not found for ID {self.user_id}')
                    return None

            org_id = user.current_org_id
            org_member = None
            for om in user.org_members:
                if om.org_id == org_id:
                    org_member = om
                    break
            if not org_member or not org_member.llm_api_key:
                return None
            org = session.query(Org).filter(Org.id == org_id).first()
            if not org:
                logger.error(
                    f'Org not found for ID {org_id} as the current org for user {self.user_id}'
                )
                return None

            for model in (user, org, org_member):
                for key, value in kwargs.items():
                    if hasattr(model, key):
                        setattr(model, key, value)

            session.commit()

    @classmethod
    async def get_instance(
        cls,
        config: OpenHandsConfig,
        user_id: str,  # type: ignore[override]
    ) -> SaasSettingsStore:
        logger.debug(f'saas_settings_store.get_instance::{user_id}')
        return SaasSettingsStore(user_id, session_maker, config)

    def _should_encrypt(self, key):
        return key in self.ENCRYPT_VALUES

    def _decrypt_kwargs(self, kwargs: dict):
        fernet = self._fernet()
        for key, value in kwargs.items():
            try:
                if value is None:
                    continue
                if self._should_encrypt(key):
                    if isinstance(value, SecretStr):
                        value = fernet.decrypt(
                            b64decode(value.get_secret_value().encode())
                        ).decode()
                    else:
                        value = fernet.decrypt(b64decode(value.encode())).decode()
                    kwargs[key] = value
            except binascii.Error:
                pass  # Key is in legacy format...

    def _encrypt_kwargs(self, kwargs: dict):
        fernet = self._fernet()
        for key, value in kwargs.items():
            if value is None:
                continue

            if isinstance(value, dict):
                self._encrypt_kwargs(value)
                continue

            if self._should_encrypt(key):
                if isinstance(value, SecretStr):
                    value = b64encode(
                        fernet.encrypt(value.get_secret_value().encode())
                    ).decode()
                else:
                    value = b64encode(fernet.encrypt(value.encode())).decode()
                kwargs[key] = value

    def _fernet(self):
        if not self.config.jwt_secret:
            raise ValueError('jwt_secret must be defined on config')
        jwt_secret = self.config.jwt_secret.get_secret_value()
        fernet_key = b64encode(hashlib.sha256(jwt_secret.encode()).digest())
        return Fernet(fernet_key)
