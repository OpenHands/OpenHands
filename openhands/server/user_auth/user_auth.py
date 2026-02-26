# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from fastapi import Request
from pydantic import SecretStr

from openhands.integrations.provider import PROVIDER_TOKEN_TYPE
from openhands.server.settings import Settings
from openhands.server.shared import server_config
from openhands.storage.data_models.secrets import Secrets
from openhands.storage.secrets.secrets_store import SecretsStore
from openhands.storage.settings.settings_store import SettingsStore
from openhands.utils.import_utils import get_impl


class AuthType(Enum):
    COOKIE = 'cookie'
    BEARER = 'bearer'


class UserAuth(ABC):
    """Abstract base class for user authentication.

    This is an extension point in OpenHands that allows applications to provide their own
    authentication mechanisms. Applications can substitute their own implementation by:
    1. Creating a class that inherits from UserAuth
    2. Implementing all required methods
    3. Setting server_config.user_auth_class to the fully qualified name of the class

    The class is instantiated via get_impl() in openhands.server.shared.py.
    """

    _settings: Settings | None

    @abstractmethod
    async def get_user_id(self) -> str | None:
        """Get the unique identifier for the current user"""

    @abstractmethod
    async def get_user_email(self) -> str | None:
        """Get the email for the current user"""

    @abstractmethod
    async def get_access_token(self) -> SecretStr | None:
        """Get the access token for the current user"""

    @abstractmethod
    async def get_provider_tokens(self) -> PROVIDER_TOKEN_TYPE | None:
        """Get the provider tokens for the current user."""

    @abstractmethod
    async def get_user_settings_store(self) -> SettingsStore:
        """Get the settings store for the current user."""

    async def get_user_settings(self) -> Settings | None:
        """Get the user settings for the current user"""
        settings = self._settings
        if settings:
            return settings
        settings_store = await self.get_user_settings_store()
        settings = await settings_store.load()
        self._settings = settings
        return settings

    @abstractmethod
    async def get_secrets_store(self) -> SecretsStore:
        """Get secrets store"""

    @abstractmethod
    async def get_secrets(self) -> Secrets | None:
        """Get the user's secrets"""

    def get_auth_type(self) -> AuthType | None:
        return None

    @abstractmethod
    async def get_mcp_api_key(self) -> str | None:
        """Get an mcp api key for the user"""

    @classmethod
    @abstractmethod
    async def get_instance(cls, request: Request) -> UserAuth:
        """Get an instance of UserAuth from the request given"""

    @classmethod
    @abstractmethod
    async def get_for_user(cls, user_id: str) -> UserAuth:
        """Get an instance of UserAuth for the user given"""


async def get_user_auth(request: Request) -> UserAuth:
    user_auth: UserAuth | None = getattr(request.state, 'user_auth', None)
    if user_auth:
        return user_auth
    impl_name = server_config.user_auth_class
    impl = get_impl(UserAuth, impl_name)
    user_auth = await impl.get_instance(request)
    if user_auth is None:
        raise ValueError('Failed to get user auth instance')
    request.state.user_auth = user_auth
    return user_auth


async def get_for_user(user_id: str) -> UserAuth:
    impl_name = server_config.user_auth_class
    impl = get_impl(UserAuth, impl_name)
    user_auth = await impl.get_for_user(user_id)
    return user_auth
