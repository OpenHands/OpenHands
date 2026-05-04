# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from types import MappingProxyType

import httpx
from fastapi import Request
from pydantic import SecretStr

from openhands.core.config.utils import load_openhands_config
from openhands.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
    CustomSecret,
    ProviderToken,
)
from openhands.integrations.service_types import ProviderType
from openhands.server import shared
from openhands.server.settings import Settings
from openhands.server.user_auth.user_auth import UserAuth
from openhands.storage.data_models.secrets import (
    WELL_KNOWN_SECRET_GITHUB_TOKEN,
    WELL_KNOWN_SECRET_LLM_API_KEY,
    Secrets,
)
from openhands.storage.secrets.secrets_store import SecretsStore
from openhands.storage.settings.settings_store import SettingsStore

logger = logging.getLogger(__name__)

_SESSION_COOKIES = ('__Secure-b1.session_token', 'b1.session_token')


@dataclass
class DefaultUserAuth(UserAuth):
    """Default user authentication mechanism"""

    _settings: Settings | None = None
    _settings_store: SettingsStore | None = None
    _secrets_store: SecretsStore | None = None
    _secrets: Secrets | None = None
    _better_auth_user: dict | None = None
    _session_cookie: tuple[str, str] | None = field(default=None, repr=False)
    _forwarded_host: str | None = field(default=None, repr=False)
    _forwarded_proto: str | None = field(default=None, repr=False)

    async def get_user_id(self) -> str | None:
        # Single-user deployment; V1 OSS storage doesn't track per-user
        # ownership, so returning an id only makes owner checks fail.
        return None

    async def get_user_email(self) -> str | None:
        if self._better_auth_user:
            return self._better_auth_user.get('email')
        return None

    async def get_access_token(self) -> SecretStr | None:
        """The default implementation does not support multi tenancy, so access_token is always None"""
        return None

    async def get_user_settings_store(self) -> SettingsStore:
        settings_store = self._settings_store
        if settings_store:
            return settings_store
        user_id = await self.get_user_id()
        settings_store = await shared.SettingsStoreImpl.get_instance(
            shared.config, user_id
        )
        if settings_store is None:
            raise ValueError('Failed to get settings store instance')
        self._settings_store = settings_store
        return settings_store

    async def get_user_settings(self) -> Settings | None:
        settings = self._settings
        if settings:
            return settings
        settings_store = await self.get_user_settings_store()
        settings = await settings_store.load()

        # Merge config.toml / env var settings with stored settings
        if settings:
            settings = settings.merge_with_config_settings()
        else:
            # No stored settings — fall back to config.toml / environment variables
            settings = Settings.from_config()

        # Use anthropic-api-key custom secret as LLM API key fallback
        if not settings or not settings.llm_api_key:
            secrets = await self.get_secrets()
            if secrets and WELL_KNOWN_SECRET_LLM_API_KEY in secrets.custom_secrets:
                custom = secrets.custom_secrets[WELL_KNOWN_SECRET_LLM_API_KEY]
                if not settings:
                    # Create default settings populated from config (model, agent, etc.)
                    app_config = load_openhands_config()
                    llm_config = app_config.get_llm_config()
                    settings = Settings(
                        llm_model=llm_config.model,
                        llm_base_url=llm_config.base_url,
                        agent=app_config.default_agent,
                        max_iterations=app_config.max_iterations,
                    )
                settings.llm_api_key = custom.secret

        self._settings = settings
        return settings

    async def get_secrets_store(self) -> SecretsStore:
        secrets_store = self._secrets_store
        if secrets_store:
            return secrets_store
        user_id = await self.get_user_id()
        secret_store = await shared.SecretsStoreImpl.get_instance(
            shared.config, user_id
        )
        if secret_store is None:
            raise ValueError('Failed to get secrets store instance')
        self._secrets_store = secret_store
        return secret_store

    async def _fetch_auth_server_secrets(self) -> Secrets | None:
        """Fetch user secrets from the Better Auth server's secrets API."""
        better_auth_url = os.environ.get('BETTER_AUTH_URL', '').rstrip('/')
        if not better_auth_url or not self._session_cookie:
            return None

        cookie_name, cookie_value = self._session_cookie
        forwarded_headers = (
            {
                'x-forwarded-host': self._forwarded_host,
                'x-forwarded-proto': self._forwarded_proto or 'https',
            }
            if self._forwarded_host
            else {}
        )

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Fetch the list of secret keys
                keys_resp = await client.get(
                    f'{better_auth_url}/api/secrets/keys',
                    cookies={cookie_name: cookie_value},
                    headers=forwarded_headers,
                )
                if keys_resp.status_code != 200:
                    logger.warning(
                        'Failed to fetch secret keys from auth server: %s',
                        keys_resp.status_code,
                    )
                    return None

                keys_data = keys_resp.json()
                keys: list[str] = keys_data.get('keys', [])
                if not keys:
                    return None

                # Fetch individual secrets in parallel
                async def _fetch_one(key: str) -> tuple[str, str | None]:
                    try:
                        resp = await client.get(
                            f'{better_auth_url}/api/secrets/key/{key}',
                            cookies={cookie_name: cookie_value},
                            headers=forwarded_headers,
                        )
                        if resp.status_code != 200:
                            return key, None
                        data = resp.json()
                        # Extract the secret value (field name: "secret" or "value")
                        raw_secret = ''
                        if isinstance(data, dict):
                            raw_secret = (
                                data.get('secret', '') or data.get('value', '') or ''
                            )
                        if not raw_secret:
                            return key, None
                        # The secret value may be JSON-stringified or a plain string
                        try:
                            parsed = json.loads(raw_secret)
                            if isinstance(parsed, dict):
                                token = parsed.get('token', '')
                            else:
                                token = str(parsed)
                        except (json.JSONDecodeError, TypeError):
                            token = raw_secret
                        return key, token if token else None
                    except Exception:
                        logger.warning(
                            'Failed to fetch secret %r from auth server',
                            key,
                            exc_info=True,
                        )
                        return key, None

                results = await asyncio.gather(*[_fetch_one(k) for k in keys])

            custom_secrets: dict[str, CustomSecret] = {}
            for key, token_value in results:
                if token_value:
                    custom_secrets[key] = CustomSecret(secret=SecretStr(token_value))

            if not custom_secrets:
                return None

            return Secrets(custom_secrets=MappingProxyType(custom_secrets))

        except Exception:
            logger.warning('Error fetching secrets from auth server', exc_info=True)
            return None

    async def get_secrets(self) -> Secrets | None:
        user_secrets = self._secrets
        if user_secrets:
            return user_secrets
        secrets_store = await self.get_secrets_store()
        user_secrets = await secrets_store.load()

        # Merge secrets from the Better Auth server (takes precedence)
        auth_server_secrets = await self._fetch_auth_server_secrets()
        if auth_server_secrets:
            if user_secrets:
                # Merge: auth server secrets override file store secrets
                merged_custom = dict(user_secrets.custom_secrets)
                merged_custom.update(auth_server_secrets.custom_secrets)
                merged_tokens = dict(user_secrets.provider_tokens)
                merged_tokens.update(auth_server_secrets.provider_tokens)
                user_secrets = Secrets(
                    provider_tokens=MappingProxyType(merged_tokens),
                    custom_secrets=MappingProxyType(merged_custom),
                )
            else:
                user_secrets = auth_server_secrets

        # Auto-populate GitHub provider token from well-known custom secret
        if user_secrets:
            user_secrets = _resolve_github_token_from_custom_secret(user_secrets)

        self._secrets = user_secrets
        return user_secrets

    async def get_provider_tokens(self) -> PROVIDER_TOKEN_TYPE | None:
        user_secrets = await self.get_secrets()
        if user_secrets is None:
            return None
        return user_secrets.provider_tokens

    async def get_mcp_api_key(self) -> str | None:
        return None

    @classmethod
    async def get_instance(cls, request: Request) -> UserAuth:
        user_auth = DefaultUserAuth()
        better_auth_user = getattr(request.state, 'better_auth_user', None)
        if better_auth_user:
            user_auth._better_auth_user = better_auth_user
            # Store session cookie for downstream API calls to the auth server
            for name in _SESSION_COOKIES:
                token = request.cookies.get(name)
                if token:
                    user_auth._session_cookie = (name, token)
                    break
            # Capture the public origin so we can forward it to the auth
            # server (newer Better Auth resolves baseURL per-request from
            # x-forwarded-host / x-forwarded-proto).
            user_auth._forwarded_host = request.headers.get(
                'x-forwarded-host'
            ) or request.headers.get('host')
            user_auth._forwarded_proto = (
                request.headers.get('x-forwarded-proto') or request.url.scheme
            )
        return user_auth

    @classmethod
    async def get_for_user(cls, user_id: str) -> UserAuth:
        assert user_id == 'root'
        return DefaultUserAuth()


def _resolve_github_token_from_custom_secret(secrets: Secrets) -> Secrets:
    """If there is no GitHub provider token but a well-known ``github-token``
    custom secret exists, create a GitHub provider token from it."""
    github_provider = secrets.provider_tokens.get(ProviderType.GITHUB)
    github_token = github_provider.token if github_provider else None
    has_github_token = bool(github_token and github_token.get_secret_value())
    has_custom_secret = WELL_KNOWN_SECRET_GITHUB_TOKEN in secrets.custom_secrets

    if has_custom_secret and not has_github_token:
        custom = secrets.custom_secrets[WELL_KNOWN_SECRET_GITHUB_TOKEN]
        new_token = ProviderToken(token=custom.secret)
        updated_tokens = dict(secrets.provider_tokens)
        updated_tokens[ProviderType.GITHUB] = new_token
        secrets = secrets.model_copy(
            update={'provider_tokens': MappingProxyType(updated_tokens)}
        )

    return secrets
