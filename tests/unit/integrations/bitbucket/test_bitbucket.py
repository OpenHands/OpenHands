"""Tests for Bitbucket integration."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from openhands.app_server.integrations.bitbucket.bitbucket_service import (
    BitBucketService,
)
from openhands.app_server.integrations.provider import ProviderToken, ProviderType
from openhands.app_server.integrations.service_types import OwnerType, Repository
from openhands.app_server.integrations.service_types import (
    ProviderType as ServiceProviderType,
)
from openhands.app_server.integrations.utils import validate_provider_token
from openhands.app_server.secrets.secrets_router import check_provider_tokens
from openhands.app_server.settings.settings_models import POSTProviderModel
from openhands.app_server.types import AppMode


# Provider Token Validation Tests
@pytest.mark.asyncio
async def test_validate_provider_token_with_bitbucket_token():
    """Test that validate_provider_token correctly identifies a Bitbucket token.

    Ensures GitHub and GitLab validators are not invoked.
    """
    # Mock the service classes to avoid actual API calls
    with (
        patch(
            'openhands.app_server.integrations.utils.GitHubService'
        ) as mock_github_service,
        patch(
            'openhands.app_server.integrations.utils.GitLabService'
        ) as mock_gitlab_service,
        patch(
            'openhands.app_server.integrations.utils.BitBucketService'
        ) as mock_bitbucket_service,
    ):
        # Set up the mocks
        github_instance = AsyncMock()
        github_instance.verify_access.side_effect = Exception('Invalid GitHub token')
        mock_github_service.return_value = github_instance

        gitlab_instance = AsyncMock()
        gitlab_instance.get_user.side_effect = Exception('Invalid GitLab token')
        mock_gitlab_service.return_value = gitlab_instance

        bitbucket_instance = AsyncMock()
        bitbucket_instance.get_user.return_value = {'username': 'test_user'}
        mock_bitbucket_service.return_value = bitbucket_instance

        # Test with a Bitbucket token
        token = SecretStr('username:app_password')
        result = await validate_provider_token(token)

        # Verify that all services were tried
        mock_github_service.assert_called_once()
        mock_gitlab_service.assert_called_once()
        mock_bitbucket_service.assert_called_once()

        # Verify that the token was identified as a Bitbucket token
        assert result == ProviderType.BITBUCKET


@pytest.mark.asyncio
async def test_check_provider_tokens_with_only_bitbucket():
    """Test that check_provider_tokens ignores GitHub/GitLab tokens when only Bitbucket is provided."""
    # Create a mock validate_provider_token function
    mock_validate = AsyncMock()
    mock_validate.return_value = ProviderType.BITBUCKET

    # Create provider tokens with only Bitbucket
    provider_tokens = {
        ProviderType.BITBUCKET: ProviderToken(
            token=SecretStr('username:app_password'), host='bitbucket.org'
        ),
        ProviderType.GITHUB: ProviderToken(token=SecretStr(''), host='github.com'),
        ProviderType.GITLAB: ProviderToken(token=SecretStr(''), host='gitlab.com'),
    }

    # Create the POST model
    post_model = POSTProviderModel(provider_tokens=provider_tokens)

    # Call check_provider_tokens with the patched validate_provider_token
    with patch(
        'openhands.app_server.secrets.secrets_router.validate_provider_token',
        mock_validate,
    ):
        await check_provider_tokens(post_model, None)

        # Verify that validate_provider_token was called only once (for Bitbucket)
        assert mock_validate.call_count == 1

        # Verify that the token passed to validate_provider_token was the Bitbucket token
        args, kwargs = mock_validate.call_args
        assert args[0].get_secret_value() == 'username:app_password'


@pytest.mark.asyncio
async def test_bitbucket_sort_parameter_mapping():
    """Test that the Bitbucket service correctly maps sort parameters."""
    # Create a service instance
    service = BitBucketService(token=SecretStr('test-token'))

    # Mock the _make_request method to avoid actual API calls
    with patch.object(service, '_make_request') as mock_request:
        # Mock workspaces response
        mock_request.side_effect = [
            # First call: workspaces
            ({'values': [{'slug': 'test-workspace', 'name': 'Test Workspace'}]}, {}),
            # Second call: repositories with mapped sort parameter
            ({'values': []}, {}),
        ]

        # Call get_repositories with sort='pushed'
        await service.get_all_repositories('pushed', AppMode.SAAS)

        # Verify that the second call used 'updated_on' instead of 'pushed'
        assert mock_request.call_count == 2

        # Check the second call (repositories call)
        second_call_args = mock_request.call_args_list[1]
        url, params = second_call_args[0]

        # Verify the sort parameter was mapped correctly (with descending order)
        assert params['sort'] == '-updated_on'
        assert 'repositories/test-workspace' in url


@pytest.mark.asyncio
async def test_bitbucket_pagination():
    """Test that the Bitbucket service correctly handles pagination for repositories."""
    # Create a service instance
    service = BitBucketService(token=SecretStr('test-token'))

    # Mock the _make_request method to simulate paginated responses
    with patch.object(service, '_make_request') as mock_request:
        # Mock responses for pagination test
        mock_request.side_effect = [
            # First call: workspaces
            ({'values': [{'slug': 'test-workspace', 'name': 'Test Workspace'}]}, {}),
            # Second call: first page of repositories
            (
                {
                    'values': [
                        {
                            'uuid': 'repo-1',
                            'slug': 'repo1',
                            'workspace': {'slug': 'test-workspace'},
                            'is_private': False,
                            'updated_on': '2023-01-01T00:00:00Z',
                        },
                        {
                            'uuid': 'repo-2',
                            'slug': 'repo2',
                            'workspace': {'slug': 'test-workspace'},
                            'is_private': True,
                            'updated_on': '2023-01-02T00:00:00Z',
                        },
                    ],
                    'next': 'https://api.bitbucket.org/2.0/repositories/test-workspace?page=2',
                },
                {},
            ),
            # Third call: second page of repositories
            (
                {
                    'values': [
                        {
                            'uuid': 'repo-3',
                            'slug': 'repo3',
                            'workspace': {'slug': 'test-workspace'},
                            'is_private': False,
                            'updated_on': '2023-01-03T00:00:00Z',
                        }
                    ],
                    # No 'next' URL indicates this is the last page
                },
                {},
            ),
        ]

        # Call get_repositories
        repositories = await service.get_all_repositories('pushed', AppMode.SAAS)

        # Verify that all three requests were made (workspaces + 2 pages of repos)
        assert mock_request.call_count == 3

        # Verify that we got all repositories from both pages
        assert len(repositories) == 3
        assert repositories[0].id == 'repo-1'
        assert repositories[1].id == 'repo-2'
        assert repositories[2].id == 'repo-3'

        # Verify repository properties
        assert repositories[0].full_name == 'test-workspace/repo1'
        assert repositories[0].is_public is True
        assert repositories[1].is_public is False
        assert repositories[2].is_public is True


@pytest.mark.asyncio
async def test_validate_provider_token_with_empty_tokens():
    """Test that validate_provider_token handles empty tokens correctly."""
    # Create a mock for each service
    with (
        patch(
            'openhands.app_server.integrations.utils.GitHubService'
        ) as mock_github_service,
        patch(
            'openhands.app_server.integrations.utils.GitLabService'
        ) as mock_gitlab_service,
        patch(
            'openhands.app_server.integrations.utils.BitBucketService'
        ) as mock_bitbucket_service,
    ):
        # Configure mocks to raise exceptions for invalid tokens
        mock_github_service.return_value.verify_access.side_effect = Exception(
            'Invalid token'
        )
        mock_gitlab_service.return_value.verify_access.side_effect = Exception(
            'Invalid token'
        )
        mock_bitbucket_service.return_value.verify_access.side_effect = Exception(
            'Invalid token'
        )

        # Test with an empty token
        token = SecretStr('')
        result = await validate_provider_token(token)

        # Services should be tried but fail with empty tokens
        mock_github_service.assert_called_once()
        mock_gitlab_service.assert_called_once()
        mock_bitbucket_service.assert_called_once()

        # Result should be None for invalid tokens
        assert result is None

        # Reset mocks for second test
        mock_github_service.reset_mock()
        mock_gitlab_service.reset_mock()
        mock_bitbucket_service.reset_mock()

        # Test with a whitespace-only token
        token = SecretStr('   ')
        result = await validate_provider_token(token)

        # Services should be tried but fail with whitespace tokens
        mock_github_service.assert_called_once()
        mock_gitlab_service.assert_called_once()
        mock_bitbucket_service.assert_called_once()

        # Result should be None for invalid tokens
        assert result is None


@pytest.mark.asyncio
async def test_bitbucket_get_repositories_with_user_owner_type():
    """Test that get_repositories correctly sets owner_type field for user repositories."""
    service = BitBucketService(token=SecretStr('test-token'))

    # Mock repository data for user repositories (private workspace)
    mock_workspaces = [{'slug': 'test-user', 'name': 'Test User'}]
    mock_repos = [
        {
            'uuid': 'repo-1',
            'slug': 'user-repo1',
            'workspace': {'slug': 'test-user', 'is_private': True},
            'is_private': False,
            'updated_on': '2023-01-01T00:00:00Z',
        },
        {
            'uuid': 'repo-2',
            'slug': 'user-repo2',
            'workspace': {'slug': 'test-user', 'is_private': True},
            'is_private': True,
            'updated_on': '2023-01-02T00:00:00Z',
        },
    ]

    with patch.object(service, '_fetch_paginated_data') as mock_fetch:
        mock_fetch.side_effect = [mock_workspaces, mock_repos]

        repositories = await service.get_all_repositories('pushed', AppMode.SAAS)

        # Verify we got the expected number of repositories
        assert len(repositories) == 2

        # Verify owner_type is correctly set for user repositories (private workspace)
        for repo in repositories:
            assert repo.owner_type == OwnerType.ORGANIZATION
            assert isinstance(repo, Repository)
            assert repo.git_provider == ServiceProviderType.BITBUCKET


@pytest.mark.asyncio
async def test_bitbucket_get_repositories_with_organization_owner_type():
    """Test that get_repositories correctly sets owner_type field for organization repositories."""
    service = BitBucketService(token=SecretStr('test-token'))

    # Mock repository data for organization repositories (public workspace)
    mock_workspaces = [{'slug': 'test-org', 'name': 'Test Organization'}]
    mock_repos = [
        {
            'uuid': 'repo-3',
            'slug': 'org-repo1',
            'workspace': {'slug': 'test-org', 'is_private': False},
            'is_private': False,
            'updated_on': '2023-01-03T00:00:00Z',
        },
        {
            'uuid': 'repo-4',
            'slug': 'org-repo2',
            'workspace': {'slug': 'test-org', 'is_private': False},
            'is_private': True,
            'updated_on': '2023-01-04T00:00:00Z',
        },
    ]

    with patch.object(service, '_fetch_paginated_data') as mock_fetch:
        mock_fetch.side_effect = [mock_workspaces, mock_repos]

        repositories = await service.get_all_repositories('pushed', AppMode.SAAS)

        # Verify we got the expected number of repositories
        assert len(repositories) == 2

        # Verify owner_type is correctly set for organization repositories (public workspace)
        for repo in repositories:
            assert repo.owner_type == OwnerType.ORGANIZATION
            assert isinstance(repo, Repository)
            assert repo.git_provider == ServiceProviderType.BITBUCKET


@pytest.mark.asyncio
async def test_bitbucket_get_repositories_mixed_owner_types():
    """Test that get_repositories correctly handles mixed user and organization repositories."""
    service = BitBucketService(token=SecretStr('test-token'))

    # Mock repository data with mixed workspace types
    mock_workspaces = [
        {'slug': 'test-user', 'name': 'Test User'},
        {'slug': 'test-org', 'name': 'Test Organization'},
    ]

    # First workspace (user) repositories
    mock_user_repos = [
        {
            'uuid': 'repo-1',
            'slug': 'user-repo',
            'workspace': {'slug': 'test-user', 'is_private': True},
            'is_private': False,
            'updated_on': '2023-01-01T00:00:00Z',
        }
    ]

    # Second workspace (organization) repositories
    mock_org_repos = [
        {
            'uuid': 'repo-2',
            'slug': 'org-repo',
            'workspace': {'slug': 'test-org', 'is_private': False},
            'is_private': False,
            'updated_on': '2023-01-02T00:00:00Z',
        }
    ]

    with patch.object(service, '_fetch_paginated_data') as mock_fetch:
        mock_fetch.side_effect = [mock_workspaces, mock_user_repos, mock_org_repos]

        repositories = await service.get_all_repositories('pushed', AppMode.SAAS)

        # Verify we got repositories from both workspaces
        assert len(repositories) == 2

        # Verify owner_type is correctly set for each repository
        user_repo = next(repo for repo in repositories if 'user-repo' in repo.full_name)
        org_repo = next(repo for repo in repositories if 'org-repo' in repo.full_name)

        assert user_repo.owner_type == OwnerType.ORGANIZATION
        assert org_repo.owner_type == OwnerType.ORGANIZATION


# ── Bitbucket email fallback tests ──


@pytest.mark.asyncio
async def test_resolve_primary_email_selects_primary_confirmed():
    """_resolve_primary_email returns the email marked primary and confirmed."""
    from openhands.app_server.integrations.bitbucket.service.base import (
        BitBucketMixinBase,
    )

    emails = [
        {'email': 'secondary@example.com', 'is_primary': False, 'is_confirmed': True},
        {'email': 'primary@example.com', 'is_primary': True, 'is_confirmed': True},
        {
            'email': 'unconfirmed@example.com',
            'is_primary': False,
            'is_confirmed': False,
        },
    ]
    result = BitBucketMixinBase._resolve_primary_email(emails)
    assert result == 'primary@example.com'


@pytest.mark.asyncio
async def test_resolve_primary_email_returns_none_when_no_primary():
    """_resolve_primary_email returns None when no email is marked primary."""
    from openhands.app_server.integrations.bitbucket.service.base import (
        BitBucketMixinBase,
    )

    emails = [
        {'email': 'a@example.com', 'is_primary': False, 'is_confirmed': True},
        {'email': 'b@example.com', 'is_primary': False, 'is_confirmed': True},
    ]
    result = BitBucketMixinBase._resolve_primary_email(emails)
    assert result is None


@pytest.mark.asyncio
async def test_resolve_primary_email_returns_none_when_primary_not_confirmed():
    """_resolve_primary_email returns None when primary email is not confirmed."""
    from openhands.app_server.integrations.bitbucket.service.base import (
        BitBucketMixinBase,
    )

    emails = [
        {'email': 'primary@example.com', 'is_primary': True, 'is_confirmed': False},
        {'email': 'other@example.com', 'is_primary': False, 'is_confirmed': True},
    ]
    result = BitBucketMixinBase._resolve_primary_email(emails)
    assert result is None


@pytest.mark.asyncio
async def test_resolve_primary_email_returns_none_for_empty_list():
    """_resolve_primary_email returns None for an empty list."""
    from openhands.app_server.integrations.bitbucket.service.base import (
        BitBucketMixinBase,
    )

    result = BitBucketMixinBase._resolve_primary_email([])
    assert result is None


@pytest.mark.asyncio
async def test_get_user_emails():
    """get_user_emails calls /user/emails and returns the values list."""
    service = BitBucketService(token=SecretStr('test-token'))

    mock_response = {
        'values': [
            {'email': 'primary@example.com', 'is_primary': True, 'is_confirmed': True},
            {
                'email': 'secondary@example.com',
                'is_primary': False,
                'is_confirmed': True,
            },
        ]
    }

    with patch.object(service, '_make_request', return_value=(mock_response, {})):
        emails = await service.get_user_emails()

        assert emails == mock_response['values']


@pytest.mark.asyncio
async def test_get_user_falls_back_to_user_emails():
    """get_user calls /user/emails to resolve email (Bitbucket /user never returns email)."""
    service = BitBucketService(token=SecretStr('test-token'))

    mock_user_response = {
        'account_id': '123',
        'username': 'testuser',
        'display_name': 'Test User',
        'links': {'avatar': {'href': 'https://example.com/avatar.jpg'}},
    }

    mock_emails = [
        {'email': 'secondary@example.com', 'is_primary': False, 'is_confirmed': True},
        {'email': 'primary@example.com', 'is_primary': True, 'is_confirmed': True},
    ]

    with (
        patch.object(service, '_make_request', return_value=(mock_user_response, {})),
        patch.object(service, 'get_user_emails', return_value=mock_emails),
    ):
        user = await service.get_user()

        assert user.email == 'primary@example.com'


@pytest.mark.asyncio
async def test_get_user_handles_user_emails_api_failure():
    """get_user handles /user/emails failure gracefully — email stays None."""
    service = BitBucketService(token=SecretStr('test-token'))

    mock_user_response = {
        'account_id': '123',
        'username': 'testuser',
        'display_name': 'Test User',
        'links': {'avatar': {'href': 'https://example.com/avatar.jpg'}},
    }

    with (
        patch.object(service, '_make_request', return_value=(mock_user_response, {})),
        patch.object(
            service,
            'get_user_emails',
            side_effect=Exception('API Error'),
        ),
    ):
        user = await service.get_user()

        # Email should remain None — no crash
        assert user.email is None
        assert user.login == 'testuser'
        assert user.name == 'Test User'
