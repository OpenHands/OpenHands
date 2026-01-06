from dataclasses import dataclass

from fastapi import Request

from openhands.app_server.errors import OpenHandsError
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.user.user_models import UserInfo
from openhands.integrations.provider import PROVIDER_TOKEN_TYPE, ProviderType
from openhands.sdk.secret import SecretSource


@dataclass(frozen=True)
class SpecifyUserContext(UserContext):
    """User context for use in admin operations which allows access beyond the scope of a single user"""

    user_id: str | None

    async def get_user_id(self) -> str | None:
        return self.user_id

    async def get_user_info(self) -> UserInfo:
        raise NotImplementedError()

    async def get_authenticated_git_url(self, repository: str) -> str:
        raise NotImplementedError()

    async def get_provider_tokens(self) -> PROVIDER_TOKEN_TYPE | None:
        raise NotImplementedError()

    async def get_latest_token(self, provider_type: ProviderType) -> str | None:
        raise NotImplementedError()

    async def get_secrets(self) -> dict[str, SecretSource]:
        raise NotImplementedError()

    async def get_mcp_api_key(self) -> str | None:
        raise NotImplementedError()


USER_CONTEXT_ATTR = 'user_context'
ADMIN = SpecifyUserContext(user_id=None)


def as_admin(request: Request):
    """Service the request as an admin user without restrictions. The endpoint should
    handle security."""
    user_context = getattr(request.state, USER_CONTEXT_ATTR, None)
    if user_context not in (None, ADMIN):
        raise OpenHandsError(
            'Non admin context already present! '
            '(Do you need to move the as_admin dependency to the start of the args?)'
        )
    setattr(request.state, USER_CONTEXT_ATTR, ADMIN)
    return ADMIN
