from __future__ import annotations

import binascii
import hashlib
import json
import os
from base64 import b64decode, b64encode
from dataclasses import dataclass

import httpx
from cryptography.fernet import Fernet
from integrations import stripe_service
from pydantic import SecretStr
from server.auth.token_manager import TokenManager
from server.constants import (
    CURRENT_USER_SETTINGS_VERSION,
    DEFAULT_INITIAL_BUDGET,
    LITE_LLM_API_KEY,
    LITE_LLM_API_URL,
    LITE_LLM_TEAM_ID,
    REQUIRE_PAYMENT,
    get_default_litellm_model,
)
from server.logger import logger
from sqlalchemy.orm import sessionmaker
from storage.database import session_maker
from storage.user_settings import UserSettings

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.server.settings import Settings
from openhands.storage import get_file_store
from openhands.storage.settings.settings_store import SettingsStore
from openhands.utils.async_utils import call_sync_from_async
from openhands.utils.http_session import httpx_verify_option


@dataclass
class SaasSettingsStore(SettingsStore):
    user_id: str
    session_maker: sessionmaker
    config: OpenHandsConfig

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
        if not self.user_id:
            return None
        with self.session_maker() as session:
            settings = self.get_user_settings_by_keycloak_id(self.user_id, session)

            if not settings or settings.user_version != CURRENT_USER_SETTINGS_VERSION:
                logger.info(
                    'saas_settings_store:load:triggering_migration',
                    extra={'user_id': self.user_id},
                )
                return await self.create_default_settings(settings)
            kwargs = {
                c.name: getattr(settings, c.name)
                for c in UserSettings.__table__.columns
                if c.name in Settings.model_fields
            }
            self._decrypt_kwargs(kwargs)
            settings = Settings(**kwargs)

            return settings

    async def store(self, item: Settings):
        # Check if provider is OpenHands and generate API key if needed
        if item and self._is_openhands_provider(item):
            await self._ensure_openhands_api_key(item)

        with self.session_maker() as session:
            existing = None
            kwargs = {}
            if item:
                kwargs = item.model_dump(context={'expose_secrets': True})
                self._encrypt_kwargs(kwargs)
                # First check if we have an existing entry in the new table
                existing = self.get_user_settings_by_keycloak_id(self.user_id, session)

            kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in UserSettings.__table__.columns
            }
            if existing:
                # Update existing entry
                for key, value in kwargs.items():
                    setattr(existing, key, value)
                existing.user_version = CURRENT_USER_SETTINGS_VERSION
                session.merge(existing)
            else:
                kwargs['keycloak_user_id'] = self.user_id
                kwargs['user_version'] = CURRENT_USER_SETTINGS_VERSION
                kwargs.pop('secrets_store', None)  # Don't save secrets_store to db
                settings = UserSettings(**kwargs)
                session.add(settings)
            session.commit()

    async def create_default_settings(self, user_settings: UserSettings | None):
        logger.info(
            'saas_settings_store:create_default_settings:start',
            extra={'user_id': self.user_id},
        )
        # You must log in before you get default settings
        if not self.user_id:
            return None

        # Only users that have specified a payment method get default settings
        if REQUIRE_PAYMENT and not await stripe_service.has_payment_method(
            self.user_id
        ):
            logger.info(
                'saas_settings_store:create_default_settings:no_payment',
                extra={'user_id': self.user_id},
            )
            return None
        settings: Settings | None = None
        if user_settings is None:
            settings = Settings(
                language='en',
                enable_proactive_conversation_starters=True,
            )
        elif isinstance(user_settings, UserSettings):
            # Convert UserSettings (SQLAlchemy model) to Settings (Pydantic model)
            kwargs = {
                c.name: getattr(user_settings, c.name)
                for c in UserSettings.__table__.columns
                if c.name in Settings.model_fields
            }
            self._decrypt_kwargs(kwargs)
            settings = Settings(**kwargs)

        if settings:
            settings = await self.update_settings_with_litellm_default(settings)
        if settings is None:
            logger.info(
                'saas_settings_store:create_default_settings:litellm_update_failed',
                extra={'user_id': self.user_id},
            )
            return None

        await self.store(settings)
        return settings

    async def load_legacy_file_store_settings(self, github_user_id: str):
        if not github_user_id:
            return None

        file_store = get_file_store(self.config.file_store, self.config.file_store_path)
        path = f'users/github/{github_user_id}/settings.json'

        try:
            json_str = await call_sync_from_async(file_store.read, path)
            logger.info(
                'saas_settings_store:load_legacy_file_store_settings:found',
                extra={'github_user_id': github_user_id},
            )
            kwargs = json.loads(json_str)
            self._decrypt_kwargs(kwargs)
            settings = Settings(**kwargs)
            return settings
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(
                'saas_settings_store:load_legacy_file_store_settings:error',
                extra={'github_user_id': github_user_id, 'error': str(e)},
            )
            return None

    async def update_settings_with_litellm_default(
        self, settings: Settings
    ) -> Settings | None:
        logger.info(
            'saas_settings_store:update_settings_with_litellm_default:start',
            extra={'user_id': self.user_id},
        )
        if LITE_LLM_API_KEY is None or LITE_LLM_API_URL is None:
            return None
        local_deploy = os.environ.get('LOCAL_DEPLOYMENT', None)
        key = LITE_LLM_API_KEY
        if not local_deploy:
            # Get user info to add to litellm
            token_manager = TokenManager()
            keycloak_user_info = (
                await token_manager.get_user_info_from_user_id(self.user_id) or {}
            )

            async with httpx.AsyncClient(
                verify=httpx_verify_option(),
                headers={
                    'x-goog-api-key': LITE_LLM_API_KEY,
                },
            ) as client:
                # Get the previous max budget to prevent accidental loss
                # In Litellm a get always succeeds, regardless of whether the user actually exists
                response = await client.get(
                    f'{LITE_LLM_API_URL}/user/info?user_id={self.user_id}'
                )
                response.raise_for_status()
                response_json = response.json()
                user_info = response_json.get('user_info') or {}
                logger.info(
                    f'creating_litellm_user: {self.user_id}; prev_max_budget: {user_info.get("max_budget")}; prev_metadata: {user_info.get("metadata")}'
                )
                max_budget = user_info.get('max_budget') or DEFAULT_INITIAL_BUDGET
                spend = user_info.get('spend') or 0

                with session_maker() as session:
                    user_settings = self.get_user_settings_by_keycloak_id(
                        self.user_id, session
                    )
                    # In upgrade to V4, we no longer use billing margin, but instead apply this directly
                    # in litellm. The default billing marign was 2 before this (hence the magic numbers below)
                    if (
                        user_settings
                        and user_settings.user_version < 4
                        and user_settings.billing_margin
                        and user_settings.billing_margin != 1.0
                    ):
                        billing_margin = user_settings.billing_margin
                        logger.info(
                            'user_settings_v4_budget_upgrade',
                            extra={
                                'max_budget': max_budget,
                                'billing_margin': billing_margin,
                                'spend': spend,
                            },
                        )
                        max_budget *= billing_margin
                        spend *= billing_margin
                        user_settings.billing_margin = 1.0
                        session.commit()

                email = keycloak_user_info.get('email')

                # We explicitly delete here to guard against odd inherited settings on upgrade.
                # We don't care if this fails with a 404
                await client.post(
                    f'{LITE_LLM_API_URL}/user/delete', json={'user_ids': [self.user_id]}
                )

                # Create the new litellm user
                response = await self._create_user_in_lite_llm(
                    client, email, max_budget, spend
                )
                if not response.is_success:
                    logger.warning(
                        'duplicate_user_email',
                        extra={'user_id': self.user_id, 'email': email},
                    )
                    # Litellm insists on unique email addresses - it is possible the email address was registered with a different user.
                    response = await self._create_user_in_lite_llm(
                        client, None, max_budget, spend
                    )

                # User failed to create in litellm - this is an unforseen error state...
                if not response.is_success:
                    logger.error(
                        'error_creating_litellm_user',
                        extra={
                            'status_code': response.status_code,
                            'text': response.text,
                            'user_id': [self.user_id],
                            'email': email,
                            'max_budget': max_budget,
                            'spend': spend,
                        },
                    )
                    return None

                response_json = response.json()
                key = response_json['key']

                logger.info(
                    'saas_settings_store:update_settings_with_litellm_default:user_created',
                    extra={'user_id': self.user_id},
                )

        settings.agent = 'CodeActAgent'
        # Use the model corresponding to the current user settings version
        settings.llm_model = get_default_litellm_model()
        settings.llm_api_key = SecretStr(key)
        settings.llm_base_url = LITE_LLM_API_URL
        return settings

    @classmethod
    async def get_instance(
        cls,
        config: OpenHandsConfig,
        user_id: str,  # type: ignore[override]
    ) -> SaasSettingsStore:
        logger.debug(f'saas_settings_store.get_instance::{user_id}')
        return SaasSettingsStore(user_id, session_maker, config)

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

    def _should_encrypt(self, key: str) -> bool:
        return key in ('llm_api_key', 'llm_api_key_for_byor', 'search_api_key')

    def _is_openhands_provider(self, item: Settings) -> bool:
        """Check if the settings use the OpenHands provider."""
        return bool(item.llm_model and item.llm_model.startswith('openhands/'))

    async def _ensure_openhands_api_key(self, item: Settings) -> None:
        """Generate and set the OpenHands API key for the given settings.

        First checks if an existing key with the OpenHands alias exists,
        and reuses it if found. Otherwise, generates a new key.
        """
        # Generate new key if none exists
        generated_key = await self._generate_openhands_key()
        if generated_key:
            item.llm_api_key = SecretStr(generated_key)
            logger.info(
                'saas_settings_store:store:generated_openhands_key',
                extra={'user_id': self.user_id},
            )
        else:
            logger.warning(
                'saas_settings_store:store:failed_to_generate_openhands_key',
                extra={'user_id': self.user_id},
            )

    async def _create_user_in_lite_llm(
        self, client: httpx.AsyncClient, email: str | None, max_budget: int, spend: int
    ):
        response = await client.post(
            f'{LITE_LLM_API_URL}/user/new',
            json={
                'user_email': email,
                'models': [],
                'max_budget': max_budget,
                'spend': spend,
                'user_id': str(self.user_id),
                'teams': [LITE_LLM_TEAM_ID],
                'auto_create_key': True,
                'send_invite_email': False,
                'metadata': {
                    'version': CURRENT_USER_SETTINGS_VERSION,
                    'model': get_default_litellm_model(),
                },
                'key_alias': f'OpenHands Cloud - user {self.user_id}',
            },
        )
        return response

    async def _generate_openhands_key(self) -> str | None:
        """Generate a new OpenHands provider key for a user."""
        if not (LITE_LLM_API_KEY and LITE_LLM_API_URL):
            logger.warning(
                'saas_settings_store:_generate_openhands_key:litellm_config_not_found',
                extra={'user_id': self.user_id},
            )
            return None

        try:
            async with httpx.AsyncClient(
                verify=httpx_verify_option(),
                headers={
                    'x-goog-api-key': LITE_LLM_API_KEY,
                },
            ) as client:
                response = await client.post(
                    f'{LITE_LLM_API_URL}/key/generate',
                    json={
                        'user_id': self.user_id,
                        'metadata': {'type': 'openhands'},
                    },
                )
                response.raise_for_status()
                response_json = response.json()
                key = response_json.get('key')

                if key:
                    logger.info(
                        'saas_settings_store:_generate_openhands_key:success',
                        extra={
                            'user_id': self.user_id,
                            'key_length': len(key) if key else 0,
                            'key_prefix': (
                                key[:10] + '...' if key and len(key) > 10 else key
                            ),
                        },
                    )
                    return key
                else:
                    logger.error(
                        'saas_settings_store:_generate_openhands_key:no_key_in_response',
                        extra={'user_id': self.user_id, 'response_json': response_json},
                    )
                    return None
        except Exception as e:
            logger.exception(
                'saas_settings_store:_generate_openhands_key:error',
                extra={'user_id': self.user_id, 'error': str(e)},
            )
            return None
