"""Tests for the GET /api/v1/users/git-organizations SAAS endpoint.

This endpoint returns the Git organizations / groups / workspaces the user
belongs to on their active provider. In SAAS mode users sign in with one
provider at a time.
"""

from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, status
from pydantic import SecretStr

from openhands.app_server.user.user_context import UserContext
from openhands.integrations.provider import ProviderToken
from openhands.integrations.service_types import ProviderType


def _make_user_context(provider_tokens, user_id: str = 'user-1') -> UserContext:
    """Build a mock UserContext with the given provider tokens."""
    context = MagicMock(spec=UserContext)
    context.get_provider_tokens = AsyncMock(return_value=provider_tokens)
    context.get_user_id = AsyncMock(return_value=user_id)
    return context


@pytest.mark.asyncio
async def test_raises_401_when_no_provider_tokens():
    """Without provider tokens the endpoint refuses the request."""
    # Arrange
    from server.routes.users_v1 import get_current_user_git_organizations

    user_context = _make_user_context(provider_tokens=None)

    # Act
    with pytest.raises(HTTPException) as excinfo:
        await get_current_user_git_organizations(user_context=user_context)

    # Assert
    assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_raises_400_when_provider_unsupported():
    """An active provider with no organizations concept surfaces a 400."""
    # Arrange
    from server.routes.users_v1 import get_current_user_git_organizations

    user_context = _make_user_context(
        provider_tokens=MappingProxyType(
            {ProviderType.AZURE_DEVOPS: ProviderToken(token=SecretStr('az-token'))}
        )
    )

    # Act
    with pytest.raises(HTTPException) as excinfo:
        await get_current_user_git_organizations(user_context=user_context)

    # Assert
    assert excinfo.value.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'provider, service_method, service_return',
    [
        (
            ProviderType.GITHUB,
            'get_organizations_from_installations',
            ['All-Hands-AI', 'OpenHands'],
        ),
        (
            ProviderType.GITLAB,
            'get_user_groups',
            ['my-team', 'open-source'],
        ),
        (
            ProviderType.BITBUCKET,
            'get_installations',
            ['my-workspace'],
        ),
    ],
    ids=['github', 'gitlab', 'bitbucket'],
)
async def test_returns_organizations_for_supported_provider(
    provider, service_method, service_return
):
    """Each supported provider routes to its service method and is returned in the response."""
    # Arrange
    from server.routes.users_v1 import get_current_user_git_organizations

    user_context = _make_user_context(
        provider_tokens=MappingProxyType(
            {provider: ProviderToken(token=SecretStr('token'))}
        )
    )

    with patch(
        'openhands.integrations.provider.ProviderHandler.get_service'
    ) as mock_get_service:
        mock_service = mock_get_service.return_value
        setattr(mock_service, service_method, AsyncMock(return_value=service_return))

        # Act
        result = await get_current_user_git_organizations(user_context=user_context)

    # Assert
    assert result.provider == provider
    assert result.organizations == service_return
