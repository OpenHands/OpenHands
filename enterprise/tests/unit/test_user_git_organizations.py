"""Tests for the GET /api/user/git-organizations endpoint.

This endpoint returns git organizations for the user's active provider
in SaaS mode (single provider at a time).
"""

from types import MappingProxyType
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.responses import JSONResponse
from pydantic import SecretStr

from openhands.integrations.provider import ProviderToken
from openhands.integrations.service_types import ProviderType


@pytest.fixture
def github_provider_tokens():
    return MappingProxyType(
        {ProviderType.GITHUB: ProviderToken(token=SecretStr('gh-token'))}
    )


@pytest.fixture
def gitlab_provider_tokens():
    return MappingProxyType(
        {ProviderType.GITLAB: ProviderToken(token=SecretStr('gl-token'))}
    )


@pytest.fixture
def bitbucket_provider_tokens():
    return MappingProxyType(
        {ProviderType.BITBUCKET: ProviderToken(token=SecretStr('bb-token'))}
    )


@pytest.fixture
def azure_devops_provider_tokens():
    return MappingProxyType(
        {ProviderType.AZURE_DEVOPS: ProviderToken(token=SecretStr('az-token'))}
    )


@pytest.fixture
def mock_check_idp():
    with patch('server.routes.user._check_idp', new_callable=AsyncMock) as mock_fn:
        yield mock_fn


@pytest.mark.asyncio
async def test_no_provider_tokens_falls_back_to_idp(mock_check_idp):
    """When no provider tokens exist, falls back to IDP check."""
    from server.routes.user import saas_get_user_git_organizations

    mock_check_idp.return_value = {}

    result = await saas_get_user_git_organizations(
        provider_tokens=None,
        access_token=SecretStr('token'),
        user_id='user-1',
    )

    assert result == {}
    mock_check_idp.assert_called_once()


@pytest.mark.asyncio
async def test_unsupported_provider_returns_400(azure_devops_provider_tokens):
    """Unsupported provider returns a 400 error."""
    from server.routes.user import saas_get_user_git_organizations

    with patch('server.routes.user.ProviderHandler'):
        result = await saas_get_user_git_organizations(
            provider_tokens=azure_devops_provider_tokens,
            access_token=SecretStr('token'),
            user_id='user-1',
        )

    assert isinstance(result, JSONResponse)
    assert result.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'provider_tokens_fixture, mock_method, mock_return, expected_provider',
    [
        (
            'github_provider_tokens',
            'get_organizations_from_installations',
            ['All-Hands-AI', 'OpenHands'],
            'github',
        ),
        (
            'gitlab_provider_tokens',
            'get_user_groups',
            ['my-team', 'open-source'],
            'gitlab',
        ),
        (
            'bitbucket_provider_tokens',
            'get_installations',
            ['my-workspace'],
            'bitbucket',
        ),
    ],
    ids=['github', 'gitlab', 'bitbucket'],
)
async def test_provider_routing_with_real_handler(
    provider_tokens_fixture,
    mock_method,
    mock_return,
    expected_provider,
    request,
):
    """Each provider routes to the correct service method and returns the expected JSON structure.

    Uses a real ProviderHandler so the endpoint's if/elif routing and ProviderHandler's
    delegation are both exercised. Only the low-level git service call is mocked.
    """
    from server.routes.user import saas_get_user_git_organizations

    provider_tokens = request.getfixturevalue(provider_tokens_fixture)

    with patch(
        'openhands.integrations.provider.ProviderHandler.get_service'
    ) as mock_get_service:
        mock_service = mock_get_service.return_value
        setattr(mock_service, mock_method, AsyncMock(return_value=mock_return))

        result = await saas_get_user_git_organizations(
            provider_tokens=provider_tokens,
            access_token=SecretStr('token'),
            user_id='user-1',
        )

    assert result == {
        'provider': expected_provider,
        'organizations': mock_return,
    }
