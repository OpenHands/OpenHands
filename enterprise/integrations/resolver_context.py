from uuid import UUID

from openhands.app_server.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
    ProviderHandler,
)
from openhands.app_server.integrations.service_types import ProviderType, UserGitInfo
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.user.user_models import UserInfo
from openhands.app_server.user_auth.user_auth import UserAuth
from openhands.sdk.secret import SecretSource, StaticSecret


class ResolverUserContext(UserContext):
    """User context for resolver operations that inherits from UserContext."""

    def __init__(
        self,
        saas_user_auth: UserAuth,
        resolver_org_id: UUID | None = None,
    ):
        self.saas_user_auth = saas_user_auth
        self.resolver_org_id = resolver_org_id
        self._provider_handler: ProviderHandler | None = None
        # SaasUserAuth supports this resolver override; other UserAuth
        # implementations may not, so keep the call defensive.
        set_effective_org_id_override = getattr(
            type(self.saas_user_auth), 'set_effective_org_id_override', None
        )
        if resolver_org_id is not None and callable(set_effective_org_id_override):
            set_effective_org_id_override(self.saas_user_auth, resolver_org_id)

    async def get_user_id(self) -> str | None:
        return await self.saas_user_auth.get_user_id()

    async def get_user_email(self) -> str | None:
        return await self.saas_user_auth.get_user_email()

    async def get_user_info(self) -> UserInfo:
        user_settings = await self.saas_user_auth.get_user_settings()
        user_id = await self.saas_user_auth.get_user_id()
        if user_settings:
            return UserInfo(
                id=user_id,
                **user_settings.model_dump(context={'expose_secrets': True}),
            )

        return UserInfo(id=user_id)

    async def _get_provider_handler(self) -> ProviderHandler:
        """Get or create a ProviderHandler for git operations."""
        if self._provider_handler is None:
            provider_tokens = await self.saas_user_auth.get_provider_tokens()
            if provider_tokens is None:
                raise ValueError('No provider tokens available')
            user_id = await self.saas_user_auth.get_user_id()
            self._provider_handler = ProviderHandler(
                provider_tokens=provider_tokens, external_auth_id=user_id
            )
        return self._provider_handler

    async def get_authenticated_git_url(
        self, repository: str, is_optional: bool = False
    ) -> str:
        provider_handler = await self._get_provider_handler()
        url = await provider_handler.get_authenticated_git_url(
            repository, is_optional=is_optional
        )
        return url

    async def get_latest_token(self, provider_type: ProviderType) -> str | None:
        # Return the appropriate token string from git_provider_tokens
        provider_tokens = await self.saas_user_auth.get_provider_tokens()
        if provider_tokens:
            provider_token = provider_tokens.get(provider_type)
            if provider_token and provider_token.token:
                return provider_token.token.get_secret_value()
        return None

    async def get_provider_tokens(
        self, as_env_vars: bool = False
    ) -> PROVIDER_TOKEN_TYPE | dict[str, str] | None:
        return await self.saas_user_auth.get_provider_tokens()

    async def get_secrets(self) -> dict[str, SecretSource]:
        """Get secrets for the user, including custom secrets."""
        secrets = await self.saas_user_auth.get_secrets()
        if secrets:
            # Convert custom secrets to StaticSecret objects for SDK compatibility
            # secrets.custom_secrets is of type Mapping[str, CustomSecret]
            converted_secrets = {}
            for key, custom_secret in secrets.custom_secrets.items():
                # Extract the secret value from CustomSecret and convert to StaticSecret
                secret_source: SecretSource = StaticSecret(value=custom_secret.secret)
                converted_secrets[key] = secret_source
            return converted_secrets
        return {}

    async def get_mcp_api_key(self) -> str | None:
        return await self.saas_user_auth.get_mcp_api_key()

    async def get_user_git_info(self) -> UserGitInfo | None:
        return await self.saas_user_auth.get_user_git_info()

    async def get_default_sandbox_spec_id(self) -> str | None:
        user_info = await self.get_user_info()
        return user_info.default_sandbox_spec_id
