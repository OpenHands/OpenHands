from openhands.app_server.user.user_context import UserContext
from openhands.app_server.user.user_models import UserInfo
from openhands.integrations.provider import PROVIDER_TOKEN_TYPE
from openhands.integrations.service_types import ProviderType
from openhands.sdk.conversation.secret_source import SecretSource, StaticSecret
from openhands.server.user_auth.user_auth import UserAuth


class ResolverUserContext(UserContext):
    """User context for resolver operations that inherits from UserContext."""

    def __init__(
        self,
        saas_user_auth: UserAuth,
    ):
        self.saas_user_auth = saas_user_auth

    async def get_user_id(self) -> str | None:
        return await self.saas_user_auth.get_user_id()

    async def get_user_info(self) -> UserInfo:
        user_settings = await self.saas_user_auth.get_user_settings()
        user_id = await self.saas_user_auth.get_user_id()
        if user_settings:
            return UserInfo(
                id=user_id,
                **user_settings.model_dump(context={'expose_secrets': True}),
            )

        return UserInfo(id=user_id)

    async def get_authenticated_git_url(self, repository: str) -> str:
        # This would need to be implemented based on the git provider tokens
        # For now, return a basic HTTPS URL
        return f'https://github.com/{repository}.git'

    async def get_latest_token(self, provider_type: ProviderType) -> str | None:
        # Return the appropriate token from git_provider_tokens

        provider_tokens = await self.saas_user_auth.get_provider_tokens()
        if provider_tokens:
            return provider_tokens.get(provider_type)
        return None

    async def get_provider_tokens(self) -> PROVIDER_TOKEN_TYPE | None:
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
                secret_value = custom_secret.secret.get_secret_value()
                converted_secrets[key] = StaticSecret(value=secret_value)
            return converted_secrets
        return {}

    async def get_mcp_api_key(self) -> str | None:
        return await self.saas_user_auth.get_mcp_api_key()
