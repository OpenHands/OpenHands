"""Tests for Bitbucket integration."""

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from openhands.integrations.bitbucket.bitbucket_service import BitBucketService
from openhands.integrations.provider import ProviderToken, ProviderType
from openhands.integrations.service_types import OwnerType, Repository, RequestMethod
from openhands.integrations.service_types import ProviderType as ServiceProviderType
from openhands.integrations.utils import validate_provider_token
from openhands.microagent.types import MicroagentContentResponse
from openhands.resolver.interfaces.bitbucket import BitbucketIssueHandler
from openhands.resolver.interfaces.issue import Issue
from openhands.resolver.interfaces.issue_definitions import ServiceContextIssue
from openhands.resolver.send_pull_request import PR_SIGNATURE, send_pull_request
from openhands.runtime.base import Runtime
from openhands.server.routes.secrets import check_provider_tokens
from openhands.server.settings import POSTProviderModel
from openhands.server.types import AppMode


# BitbucketIssueHandler Tests
@pytest.fixture
def bitbucket_handler():
    return BitbucketIssueHandler(
        owner='test-workspace',
        repo='test-repo',
        token='test-token',
        user_id='test-user',
    )


def test_init():
    handler = BitbucketIssueHandler(
        owner='test-workspace',
        repo='test-repo',
        token='test-token',
        user_id='test-user',
    )

    assert handler.owner == 'test-workspace'
    assert handler.repo == 'test-repo'
    assert handler.token == 'test-token'
    assert handler.user_id == 'test-user'
    assert handler.base_domain == 'bitbucket.org'
    assert handler.base_url == 'https://api.bitbucket.org/2.0'
    assert (
        handler.download_url
        == 'https://bitbucket.org/test-workspace/test-repo/get/master.zip'
    )
    assert handler.clone_url == 'https://bitbucket.org/test-workspace/test-repo.git'
    assert handler.headers == {
        'Authorization': 'Bearer test-token',
        'Accept': 'application/json',
    }


def test_init_server_mode():
    handler = BitbucketIssueHandler(
        owner='PROJ',
        repo='test-repo',
        token='user:pass',
        user_id=None,
        base_domain='bitbucket.example.com',
        bitbucket_mode='server',
    )

    assert handler.base_url == 'https://bitbucket.example.com/rest/api/1.0'
    assert (
        handler.download_url
        == 'https://bitbucket.example.com/rest/api/latest/projects/PROJ/repos/test-repo/archive?format=zip'
    )
    assert handler.clone_url == 'https://bitbucket.example.com/scm/PROJ/test-repo.git'
    assert (
        handler.get_repo_url()
        == 'https://bitbucket.example.com/projects/PROJ/repos/test-repo'
    )
    assert handler.get_issue_url(42) == (
        'https://bitbucket.example.com/projects/PROJ/repos/test-repo/pull-requests/42'
    )
    assert handler.headers['Authorization'].startswith('Basic ')


@pytest.mark.asyncio
async def test_server_get_branches_uses_rest_endpoint(monkeypatch):
    service = BitBucketService(
        user_id='user',
        token=SecretStr('user:pass'),
        base_domain='bitbucket.example.com',
        bitbucket_mode='server',
    )

    fetch_mock = AsyncMock(
        return_value=[
            {
                'displayId': 'main',
                'latestCommit': 'abc123',
                'metadata': {'last-modified': [{'authorTimestamp': 1700000000000}]},
            }
        ]
    )
    monkeypatch.setattr(service, '_fetch_paginated_data', fetch_mock)

    branches = await service.get_branches('PROJ/repo')

    fetch_mock.assert_awaited_once()
    called_url, params, max_items = fetch_mock.call_args[0]
    assert (
        called_url
        == 'https://bitbucket.example.com/rest/api/1.0/projects/PROJ/repos/repo/branches'
    )
    assert params['limit'] == 100
    assert params['orderBy'] == 'MODIFICATION'
    assert max_items == 1000
    assert branches[0].name == 'main'
    assert branches[0].commit_sha == 'abc123'
    assert branches[0].last_push_date == '2023-11-14T22:13:20+00:00'


@pytest.mark.asyncio
async def test_server_get_paginated_branches(monkeypatch):
    service = BitBucketService(
        user_id='user',
        token=SecretStr('user:pass'),
        base_domain='bitbucket.example.com',
        bitbucket_mode='server',
    )

    response = {
        'values': [{'displayId': 'feature', 'latestCommit': 'def456'}],
        'isLastPage': False,
        'size': 40,
    }
    make_request = AsyncMock(return_value=(response, {}))
    monkeypatch.setattr(service, '_make_request', make_request)

    result = await service.get_paginated_branches('PROJ/repo', page=2, per_page=20)

    make_request.assert_awaited_once()
    called_url, params = make_request.call_args[0]
    assert (
        called_url
        == 'https://bitbucket.example.com/rest/api/1.0/projects/PROJ/repos/repo/branches'
    )
    assert params['start'] == 20
    assert params['limit'] == 20
    assert params['orderBy'] == 'MODIFICATION'
    assert result.has_next_page is True
    assert result.total_count == 40
    assert result.branches[0].name == 'feature'
    assert result.branches[0].commit_sha == 'def456'


@pytest.mark.asyncio
async def test_server_search_branches(monkeypatch):
    service = BitBucketService(
        user_id='user',
        token=SecretStr('user:pass'),
        base_domain='bitbucket.example.com',
        bitbucket_mode='server',
    )

    response = {
        'values': [{'displayId': 'bugfix', 'latestCommit': 'abc999'}],
    }
    make_request = AsyncMock(return_value=(response, {}))
    monkeypatch.setattr(service, '_make_request', make_request)

    branches = await service.search_branches('PROJ/repo', query='bug', per_page=50)

    make_request.assert_awaited_once()
    called_url, params = make_request.call_args[0]
    assert (
        called_url
        == 'https://bitbucket.example.com/rest/api/1.0/projects/PROJ/repos/repo/branches'
    )
    assert params['filterText'] == 'bug'
    assert params['limit'] == 50
    assert params['orderBy'] == 'MODIFICATION'
    assert branches[0].name == 'bugfix'
    assert branches[0].commit_sha == 'abc999'


@pytest.mark.asyncio
async def test_cloud_create_pr_uses_v2_endpoint(monkeypatch):
    service = BitBucketService(token=SecretStr('token'))

    response = {
        'links': {
            'html': {'href': 'https://bitbucket.org/workspace/repo/pull-requests/42'}
        }
    }
    make_request = AsyncMock(return_value=(response, {}))
    monkeypatch.setattr(service, '_make_request', make_request)

    pr_url = await service.create_pr(
        'workspace/repo', 'feature', 'main', 'Title', 'Body', draft=True
    )

    assert pr_url == 'https://bitbucket.org/workspace/repo/pull-requests/42'
    call_kwargs = make_request.call_args.kwargs
    assert (
        call_kwargs['url']
        == 'https://api.bitbucket.org/2.0/repositories/workspace/repo/pullrequests'
    )
    assert call_kwargs['params']['source']['branch']['name'] == 'feature'
    assert call_kwargs['params']['destination']['branch']['name'] == 'main'
    assert call_kwargs['params']['draft'] is True
    assert call_kwargs['method'] == RequestMethod.POST


@pytest.mark.asyncio
async def test_server_create_pr_uses_v1_endpoint(monkeypatch):
    service = BitBucketService(
        user_id='user',
        token=SecretStr('user:pass'),
        base_domain='bitbucket.example.com',
        bitbucket_mode='server',
    )

    response = {
        'links': {
            'self': [
                {
                    'href': (
                        'https://bitbucket.example.com/projects/PROJ/repos/repo/'
                        'pull-requests/7'
                    )
                }
            ]
        }
    }
    make_request = AsyncMock(return_value=(response, {}))
    monkeypatch.setattr(service, '_make_request', make_request)

    pr_url = await service.create_pr(
        'PROJ/repo', 'feature/server', 'main', 'Server Title', 'Body'
    )

    assert (
        pr_url
        == 'https://bitbucket.example.com/projects/PROJ/repos/repo/pull-requests/7'
    )
    call_kwargs = make_request.call_args.kwargs
    assert (
        call_kwargs['url']
        == 'https://bitbucket.example.com/rest/api/1.0/projects/PROJ/repos/repo/pull-requests'
    )
    assert call_kwargs['params']['fromRef']['id'] == 'refs/heads/feature/server'
    assert call_kwargs['params']['toRef']['id'] == 'refs/heads/main'
    assert call_kwargs['method'] == RequestMethod.POST


@pytest.mark.asyncio
async def test_server_get_pr_details_uses_v1_endpoint(monkeypatch):
    service = BitBucketService(
        user_id='user',
        token=SecretStr('user:pass'),
        base_domain='bitbucket.example.com',
        bitbucket_mode='server',
    )

    make_request = AsyncMock(return_value=({'id': 1}, {}))
    monkeypatch.setattr(service, '_make_request', make_request)

    await service.get_pr_details('PROJ/repo', 12)

    call_args = make_request.call_args
    assert (
        call_args.args[0]
        == 'https://bitbucket.example.com/rest/api/1.0/projects/PROJ/repos/repo/pull-requests/12'
    )


@pytest.mark.asyncio
async def test_cloud_get_pr_details_uses_v2_endpoint(monkeypatch):
    service = BitBucketService(token=SecretStr('token'))

    make_request = AsyncMock(return_value=({'id': 1}, {}))
    monkeypatch.setattr(service, '_make_request', make_request)

    await service.get_pr_details('workspace/repo', 5)

    call_args = make_request.call_args
    assert (
        call_args.args[0]
        == 'https://api.bitbucket.org/2.0/repositories/workspace/repo/pullrequests/5'
    )


@pytest.mark.asyncio
async def test_server_microagent_content_uses_browse_endpoint(monkeypatch):
    service = BitBucketService(
        user_id='user',
        token=SecretStr('user:pass'),
        base_domain='bitbucket.example.com',
        bitbucket_mode='server',
    )

    repo_details = Repository(
        id='1',
        full_name='PROJ/repo',
        git_provider=ServiceProviderType.BITBUCKET,
        is_public=False,
        owner_type=OwnerType.ORGANIZATION,
        main_branch='main',
    )
    monkeypatch.setattr(
        service,
        'get_repository_details_from_repo_name',
        AsyncMock(return_value=repo_details),
    )

    response = {'lines': [{'text': 'line1'}, {'text': 'line2'}]}
    make_request = AsyncMock(return_value=(response, {}))
    monkeypatch.setattr(service, '_make_request', make_request)

    expected_content = MicroagentContentResponse(
        content='parsed', path='microagents/file.md', triggers=[]
    )
    parse_mock = MagicMock(return_value=expected_content)
    monkeypatch.setattr(service, '_parse_microagent_content', parse_mock)

    result = await service.get_microagent_content('PROJ/repo', 'microagents/file.md')

    call_args = make_request.call_args
    assert (
        call_args.args[0]
        == 'https://bitbucket.example.com/rest/api/1.0/projects/PROJ/repos/repo/browse/microagents/file.md'
    )
    assert call_args.kwargs['params']['at'] == 'refs/heads/main'
    parse_mock.assert_called_once_with('line1\nline2', 'microagents/file.md')
    assert result is expected_content


@pytest.mark.asyncio
async def test_cloud_microagent_content_uses_src_endpoint(monkeypatch):
    service = BitBucketService(token=SecretStr('token'))

    repo_details = Repository(
        id='1',
        full_name='workspace/repo',
        git_provider=ServiceProviderType.BITBUCKET,
        is_public=True,
        owner_type=OwnerType.ORGANIZATION,
        main_branch='main',
    )
    monkeypatch.setattr(
        service,
        'get_repository_details_from_repo_name',
        AsyncMock(return_value=repo_details),
    )

    make_request = AsyncMock(return_value=('raw-content', {}))
    monkeypatch.setattr(service, '_make_request', make_request)

    expected_content = MicroagentContentResponse(
        content='parsed', path='.openhands/microagents/file.md', triggers=[]
    )
    parse_mock = MagicMock(return_value=expected_content)
    monkeypatch.setattr(service, '_parse_microagent_content', parse_mock)

    result = await service.get_microagent_content(
        'workspace/repo', '.openhands/microagents/file.md'
    )

    call_args = make_request.call_args
    assert (
        call_args.args[0]
        == 'https://api.bitbucket.org/2.0/repositories/workspace/repo/src/main/.openhands/microagents/file.md'
    )
    parse_mock.assert_called_once_with('raw-content', '.openhands/microagents/file.md')
    assert result is expected_content


def test_get_repo_url(bitbucket_handler):
    assert (
        bitbucket_handler.get_repo_url()
        == 'https://bitbucket.org/test-workspace/test-repo'
    )


def test_get_issue_url(bitbucket_handler):
    assert (
        bitbucket_handler.get_issue_url(123)
        == 'https://bitbucket.org/test-workspace/test-repo/issues/123'
    )


def test_get_pr_url(bitbucket_handler):
    assert (
        bitbucket_handler.get_pr_url(123)
        == 'https://bitbucket.org/test-workspace/test-repo/pull-requests/123'
    )


@pytest.mark.asyncio
@patch('httpx.AsyncClient')
async def test_get_issue(mock_client, bitbucket_handler):
    mock_response = MagicMock()
    mock_response.raise_for_status = AsyncMock()
    mock_response.json.return_value = {
        'id': 123,
        'title': 'Test Issue',
        'content': {'raw': 'Test Issue Body'},
        'links': {
            'html': {
                'href': 'https://bitbucket.org/test-workspace/test-repo/issues/123'
            }
        },
        'state': 'open',
        'reporter': {'display_name': 'Test User'},
        'assignee': [{'display_name': 'Assignee User'}],
    }

    mock_client_instance = AsyncMock()
    mock_client_instance.get.return_value = mock_response
    mock_client.return_value.__aenter__.return_value = mock_client_instance

    issue = await bitbucket_handler.get_issue(123)

    assert issue.number == 123
    assert issue.title == 'Test Issue'
    assert issue.body == 'Test Issue Body'
    # We don't test for html_url, state, user, or assignees as they're not part of the Issue model


@patch('httpx.post')
def test_create_pr(mock_post, bitbucket_handler):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        'links': {
            'html': {
                'href': 'https://bitbucket.org/test-workspace/test-repo/pull-requests/123'
            }
        },
    }
    mock_post.return_value = mock_response

    pr_url = bitbucket_handler.create_pr(
        title='Test PR',
        body='Test PR Body',
        head='feature-branch',
        base='main',
    )

    assert pr_url == 'https://bitbucket.org/test-workspace/test-repo/pull-requests/123'

    expected_payload = {
        'title': 'Test PR',
        'description': 'Test PR Body',
        'source': {'branch': {'name': 'feature-branch'}},
        'destination': {'branch': {'name': 'main'}},
        'close_source_branch': False,
    }

    mock_post.assert_called_once_with(
        'https://api.bitbucket.org/2.0/repositories/test-workspace/test-repo/pullrequests',
        headers=bitbucket_handler.headers,
        json=expected_payload,
    )


@patch('httpx.post')
def test_create_pr_server_mode(mock_post):
    handler = BitbucketIssueHandler(
        owner='PROJ',
        repo='test-repo',
        token='user:pass',
        user_id=None,
        base_domain='bitbucket.example.com',
        bitbucket_mode='server',
    )

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        'id': 99,
        'links': {
            'self': [
                {
                    'href': 'https://bitbucket.example.com/projects/PROJ/repos/test-repo/pull-requests/99'
                }
            ]
        },
    }
    mock_post.return_value = mock_response

    pr_url = handler.create_pr(
        title='Server PR',
        body='Server Body',
        head='feature/server',
        base='main',
    )

    assert (
        pr_url
        == 'https://bitbucket.example.com/projects/PROJ/repos/test-repo/pull-requests/99'
    )

    expected_payload = {
        'title': 'Server PR',
        'description': 'Server Body',
        'fromRef': {
            'id': 'refs/heads/feature/server',
            'repository': {'slug': 'test-repo', 'project': {'key': 'PROJ'}},
        },
        'toRef': {
            'id': 'refs/heads/main',
            'repository': {'slug': 'test-repo', 'project': {'key': 'PROJ'}},
        },
    }

    mock_post.assert_called_once_with(
        'https://bitbucket.example.com/rest/api/1.0/projects/PROJ/repos/test-repo/pull-requests',
        headers=handler.headers,
        json=expected_payload,
    )


# Bitbucket Send Pull Request Tests
@patch('openhands.resolver.send_pull_request.ServiceContextIssue')
@patch('openhands.resolver.send_pull_request.BitbucketIssueHandler')
@patch('subprocess.run')
def test_send_pull_request_bitbucket(
    mock_run, mock_bitbucket_handler, mock_service_context
):
    # Mock subprocess.run to avoid actual git operations
    mock_run.return_value = MagicMock(returncode=0)

    # Mock the BitbucketIssueHandler instance
    mock_instance = MagicMock(spec=BitbucketIssueHandler)
    mock_bitbucket_handler.return_value = mock_instance

    # Mock the ServiceContextIssue instance
    mock_service = MagicMock(spec=ServiceContextIssue)
    mock_service.get_branch_name.return_value = 'openhands-fix-123'
    mock_service.branch_exists.return_value = True
    mock_service.get_default_branch_name.return_value = 'main'
    mock_service.get_clone_url.return_value = (
        'https://bitbucket.org/test-workspace/test-repo.git'
    )
    mock_service.create_pull_request.return_value = {
        'html_url': 'https://bitbucket.org/test-workspace/test-repo/pull-requests/123'
    }
    # Add _strategy attribute to mock
    mock_strategy = MagicMock()
    mock_service._strategy = mock_strategy
    mock_service_context.return_value = mock_service

    # Create a mock Issue
    mock_issue = Issue(
        number=123,
        title='Test Issue',
        owner='test-workspace',
        repo='test-repo',
        body='Test body',
        created_at='2023-01-01T00:00:00Z',
        updated_at='2023-01-01T00:00:00Z',
        closed_at=None,
        head_branch='feature-branch',
        thread_ids=None,
    )

    # Call send_pull_request
    result = send_pull_request(
        issue=mock_issue,
        token='test-token',
        username=None,
        platform=ServiceProviderType.BITBUCKET,
        patch_dir='/tmp',  # Use /tmp instead of /tmp/repo to avoid directory not found error
        pr_type='ready',
        pr_title='Test PR',
        target_branch='main',
    )

    # Verify the result
    assert result == 'https://bitbucket.org/test-workspace/test-repo/pull-requests/123'

    # Verify the handler was created correctly
    mock_bitbucket_handler.assert_called_once_with(
        'test-workspace',
        'test-repo',
        'test-token',
        user_id=None,
        base_domain='bitbucket.org',
        bitbucket_mode='cloud',
    )

    # Verify ServiceContextIssue was created correctly
    mock_service_context.assert_called_once()

    # Verify create_pull_request was called with the correct data
    expected_body = f'This pull request fixes #123.\n\n{PR_SIGNATURE}'
    mock_service.create_pull_request.assert_called_once_with(
        {
            'title': 'Test PR',
            'description': expected_body,
            'source_branch': 'openhands-fix-123',
            'target_branch': 'main',
            'draft': False,
        }
    )


# Bitbucket Provider Domain Tests
class TestBitbucketProviderDomain(unittest.TestCase):
    """Test that Bitbucket provider domain is properly handled in Runtime.clone_or_init_repo."""

    @patch('openhands.runtime.base.Runtime.__abstractmethods__', set())
    @patch(
        'openhands.runtime.utils.edit.FileEditRuntimeMixin.__init__', return_value=None
    )
    @patch('openhands.runtime.base.ProviderHandler')
    @pytest.mark.asyncio
    async def test_get_authenticated_git_url_bitbucket(
        self, mock_provider_handler, mock_file_edit_init, *args
    ):
        """Test that _get_authenticated_git_url correctly handles Bitbucket repositories."""
        # Mock the provider handler to return a repository with Bitbucket as the provider
        mock_repository = Repository(
            id='1',
            full_name='workspace/repo',
            git_provider=ServiceProviderType.BITBUCKET,
            is_public=True,
        )

        mock_provider_instance = MagicMock()
        mock_provider_instance.verify_repo_provider.return_value = mock_repository
        mock_provider_handler.return_value = mock_provider_instance

        # Create a minimal runtime instance with abstract methods patched
        config = MagicMock()
        config.get_llm_config.return_value.model = 'test_model'
        runtime = Runtime(config=config, event_stream=MagicMock(), sid='test_sid')

        # Test with no token
        url = await runtime._get_authenticated_git_url('workspace/repo', None)
        self.assertEqual(url, 'https://bitbucket.org/workspace/repo.git')

        # Test with username:password format token
        git_provider_tokens = {
            ProviderType.BITBUCKET: ProviderToken(
                token=SecretStr('username:app_password'), host='bitbucket.org'
            )
        }
        url = await runtime._get_authenticated_git_url(
            'workspace/repo', git_provider_tokens
        )
        # Bitbucket tokens with colon are used directly as username:password
        self.assertEqual(
            url, 'https://username:app_password@bitbucket.org/workspace/repo.git'
        )

        # Test with email:password format token (more realistic)
        git_provider_tokens = {
            ProviderType.BITBUCKET: ProviderToken(
                token=SecretStr('user@example.com:app_password'), host='bitbucket.org'
            )
        }
        url = await runtime._get_authenticated_git_url(
            'workspace/repo', git_provider_tokens
        )
        # Email addresses in tokens are used as-is (no URL encoding in our implementation)
        self.assertEqual(
            url,
            'https://user@example.com:app_password@bitbucket.org/workspace/repo.git',
        )

        # Test with simple token format (access token)
        git_provider_tokens = {
            ProviderType.BITBUCKET: ProviderToken(
                token=SecretStr('simple_token'), host='bitbucket.org'
            )
        }
        url = await runtime._get_authenticated_git_url(
            'workspace/repo', git_provider_tokens
        )
        # Simple tokens use x-token-auth format
        self.assertEqual(
            url, 'https://x-token-auth:simple_token@bitbucket.org/workspace/repo.git'
        )

    @patch('openhands.runtime.base.ProviderHandler')
    @patch.object(Runtime, 'run_action')
    async def test_bitbucket_provider_domain(
        self, mock_run_action, mock_provider_handler
    ):
        # Mock the provider handler to return a repository with Bitbucket as the provider
        mock_repository = Repository(
            id='1',
            full_name='test/repo',
            git_provider=ServiceProviderType.BITBUCKET,
            is_public=True,
        )

        mock_provider_instance = MagicMock()
        mock_provider_instance.verify_repo_provider.return_value = mock_repository
        mock_provider_handler.return_value = mock_provider_instance

        # Create a minimal runtime instance
        runtime = Runtime(config=MagicMock(), event_stream=MagicMock(), sid='test_sid')

        # Mock the workspace_root property to avoid AttributeError
        runtime.workspace_root = '/workspace'

        # Call clone_or_init_repo with a Bitbucket repository
        # This should now succeed with our fix
        await runtime.clone_or_init_repo(
            git_provider_tokens=None,
            selected_repository='test/repo',
            selected_branch=None,
        )

        # Verify that run_action was called at least once (for git clone)
        self.assertTrue(mock_run_action.called)

        # Verify that the domain used was 'bitbucket.org'
        # Extract the command from the first call to run_action
        args, _ = mock_run_action.call_args
        action = args[0]
        self.assertIn('bitbucket.org', action.command)


# Provider Token Validation Tests
@pytest.mark.asyncio
async def test_validate_provider_token_with_bitbucket_token():
    """Test that validate_provider_token correctly identifies a Bitbucket token.

    Ensures GitHub and GitLab validators are not invoked.
    """
    # Mock the service classes to avoid actual API calls
    with (
        patch('openhands.integrations.utils.GitHubService') as mock_github_service,
        patch('openhands.integrations.utils.GitLabService') as mock_gitlab_service,
        patch(
            'openhands.integrations.utils.BitBucketService'
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
async def test_validate_provider_token_with_bitbucket_server_mode():
    """Ensure server mode configuration is passed to Bitbucket service."""

    with (
        patch('openhands.integrations.utils.GitHubService') as mock_github_service,
        patch('openhands.integrations.utils.GitLabService') as mock_gitlab_service,
        patch(
            'openhands.integrations.utils.BitBucketService'
        ) as mock_bitbucket_service,
    ):
        github_instance = AsyncMock()
        github_instance.verify_access.side_effect = Exception('Invalid GitHub token')
        mock_github_service.return_value = github_instance

        gitlab_instance = AsyncMock()
        gitlab_instance.get_user.side_effect = Exception('Invalid GitLab token')
        mock_gitlab_service.return_value = gitlab_instance

        bitbucket_instance = AsyncMock()
        bitbucket_instance.get_user.return_value = {'username': 'server_user'}
        mock_bitbucket_service.return_value = bitbucket_instance

        token = SecretStr('username:password')
        base_domain = 'bitbucket.example.com'
        result = await validate_provider_token(token, base_domain)

        mock_bitbucket_service.assert_called_once_with(
            token=token,
            base_domain=base_domain,
        )
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
        'openhands.server.routes.secrets.validate_provider_token', mock_validate
    ):
        result = await check_provider_tokens(post_model, None)

        # Verify that validate_provider_token was called only once (for Bitbucket)
        assert mock_validate.call_count == 1

        # Verify that the token passed to validate_provider_token was the Bitbucket token
        args, kwargs = mock_validate.call_args
        assert args[0].get_secret_value() == 'username:app_password'
        assert args[1] == 'bitbucket.org'

        # Verify that no error message was returned
        assert result == ''


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
        patch('openhands.integrations.utils.GitHubService') as mock_github_service,
        patch('openhands.integrations.utils.GitLabService') as mock_gitlab_service,
        patch(
            'openhands.integrations.utils.BitBucketService'
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


@pytest.mark.asyncio
async def test_fetch_paginated_data_server_mode():
    """Server pagination should use nextPageStart tokens."""

    service = BitBucketService(
        token=SecretStr('user:pass'),
        base_domain='bitbucket.example.com',
        bitbucket_mode='server',
    )

    first_page = ({'values': [{'id': 1}], 'isLastPage': False, 'nextPageStart': 2}, {})
    second_page = ({'values': [{'id': 2}], 'isLastPage': True}, {})

    with patch.object(
        service,
        '_make_request',
        new=AsyncMock(side_effect=[first_page, second_page]),
    ) as mock_request:
        results = await service._fetch_paginated_data(
            'https://bitbucket.example.com/rest/api/1.0/projects/PROJ/repos/repo',
            {'limit': 50},
            10,
        )

    assert results == [{'id': 1}, {'id': 2}]
    assert mock_request.call_count == 2


@pytest.mark.asyncio
async def test_fetch_paginated_data_cloud_mode():
    """Cloud pagination should follow next links."""

    service = BitBucketService(token=SecretStr('token'))

    first_page = (
        {'values': [{'id': 'a'}], 'next': 'https://api.bitbucket.org/2.0/page/2'},
        {},
    )
    second_page = ({'values': [{'id': 'b'}], 'next': None}, {})

    with patch.object(
        service,
        '_make_request',
        new=AsyncMock(side_effect=[first_page, second_page]),
    ) as mock_request:
        results = await service._fetch_paginated_data(
            'https://api.bitbucket.org/2.0/repositories/workspace/repo',
            {'pagelen': 50},
            5,
        )

    assert results == [{'id': 'a'}, {'id': 'b'}]
    assert mock_request.call_count == 2


def test_parse_repository_server_mode():
    """Server repositories should derive full names from project key and slug."""

    service = BitBucketService(
        token=SecretStr('user:pass'),
        base_domain='bitbucket.example.com',
        bitbucket_mode='server',
    )

    repo_data = {
        'id': 42,
        'slug': 'sample',
        'project': {'key': 'PROJ'},
        'defaultBranch': {'id': 'refs/heads/main'},
        'public': False,
        'updatedDate': '1700000000',
    }

    repository = service._parse_repository(repo_data)

    assert repository.full_name == 'PROJ/sample'
    assert repository.main_branch == 'main'
    assert repository.git_provider == ServiceProviderType.BITBUCKET
# ── Bitbucket email fallback tests ──


@pytest.mark.asyncio
async def test_resolve_primary_email_selects_primary_confirmed():
    """_resolve_primary_email returns the email marked primary and confirmed."""
    from openhands.integrations.bitbucket.service.base import BitBucketMixinBase

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
    from openhands.integrations.bitbucket.service.base import BitBucketMixinBase

    emails = [
        {'email': 'a@example.com', 'is_primary': False, 'is_confirmed': True},
        {'email': 'b@example.com', 'is_primary': False, 'is_confirmed': True},
    ]
    result = BitBucketMixinBase._resolve_primary_email(emails)
    assert result is None


@pytest.mark.asyncio
async def test_resolve_primary_email_returns_none_when_primary_not_confirmed():
    """_resolve_primary_email returns None when primary email is not confirmed."""
    from openhands.integrations.bitbucket.service.base import BitBucketMixinBase

    emails = [
        {'email': 'primary@example.com', 'is_primary': True, 'is_confirmed': False},
        {'email': 'other@example.com', 'is_primary': False, 'is_confirmed': True},
    ]
    result = BitBucketMixinBase._resolve_primary_email(emails)
    assert result is None


@pytest.mark.asyncio
async def test_resolve_primary_email_returns_none_for_empty_list():
    """_resolve_primary_email returns None for an empty list."""
    from openhands.integrations.bitbucket.service.base import BitBucketMixinBase

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


# Setup.py Bitbucket Token Tests
@patch('openhands.core.setup.call_async_from_sync')
@patch('openhands.core.setup.get_file_store')
@patch('openhands.core.setup.EventStream')
def test_initialize_repository_for_runtime_with_bitbucket_token(
    mock_event_stream, mock_get_file_store, mock_call_async_from_sync
):
    """Test that initialize_repository_for_runtime properly handles BITBUCKET_TOKEN."""
    from openhands.core.setup import initialize_repository_for_runtime
    from openhands.integrations.provider import ProviderType

    # Mock runtime
    mock_runtime = MagicMock()
    mock_runtime.clone_or_init_repo = AsyncMock(return_value='test-repo')
    mock_runtime.maybe_run_setup_script = MagicMock()
    mock_runtime.maybe_setup_git_hooks = MagicMock()

    # Mock call_async_from_sync to return the expected result
    mock_call_async_from_sync.return_value = 'test-repo'

    # Set up environment with BITBUCKET_TOKEN
    with patch.dict(os.environ, {'BITBUCKET_TOKEN': 'username:app_password'}):
        result = initialize_repository_for_runtime(
            runtime=mock_runtime, selected_repository='openhands/test-repo'
        )

    # Verify the result
    assert result == 'test-repo'

    # Verify that call_async_from_sync was called with the correct arguments
    mock_call_async_from_sync.assert_called_once()
    args, kwargs = mock_call_async_from_sync.call_args

    # Check that the function called was clone_or_init_repo
    assert args[0] == mock_runtime.clone_or_init_repo

    # Check that provider tokens were passed correctly
    provider_tokens = args[2]  # Third argument is immutable_provider_tokens
    assert provider_tokens is not None
    assert ProviderType.BITBUCKET in provider_tokens
    assert (
        provider_tokens[ProviderType.BITBUCKET].token.get_secret_value()
        == 'username:app_password'
    )

    # Check that the repository was passed correctly
    assert args[3] == 'openhands/test-repo'  # selected_repository
    assert args[4] is None  # selected_branch


@patch('openhands.core.setup.call_async_from_sync')
@patch('openhands.core.setup.get_file_store')
@patch('openhands.core.setup.EventStream')
def test_initialize_repository_for_runtime_with_multiple_tokens(
    mock_event_stream, mock_get_file_store, mock_call_async_from_sync
):
    """Test that initialize_repository_for_runtime handles multiple provider tokens including Bitbucket."""
    from openhands.core.setup import initialize_repository_for_runtime
    from openhands.integrations.provider import ProviderType

    # Mock runtime
    mock_runtime = MagicMock()
    mock_runtime.clone_or_init_repo = AsyncMock(return_value='test-repo')
    mock_runtime.maybe_run_setup_script = MagicMock()
    mock_runtime.maybe_setup_git_hooks = MagicMock()

    # Mock call_async_from_sync to return the expected result
    mock_call_async_from_sync.return_value = 'test-repo'

    # Set up environment with multiple tokens
    with patch.dict(
        os.environ,
        {
            'GITHUB_TOKEN': 'github_token_123',
            'GITLAB_TOKEN': 'gitlab_token_456',
            'BITBUCKET_TOKEN': 'username:bitbucket_app_password',
        },
    ):
        result = initialize_repository_for_runtime(
            runtime=mock_runtime, selected_repository='openhands/test-repo'
        )

    # Verify the result
    assert result == 'test-repo'

    # Verify that call_async_from_sync was called
    mock_call_async_from_sync.assert_called_once()
    args, kwargs = mock_call_async_from_sync.call_args

    # Check that provider tokens were passed correctly
    provider_tokens = args[2]  # Third argument is immutable_provider_tokens
    assert provider_tokens is not None

    # Verify all three provider types are present
    assert ProviderType.GITHUB in provider_tokens
    assert ProviderType.GITLAB in provider_tokens
    assert ProviderType.BITBUCKET in provider_tokens

    # Verify token values
    assert (
        provider_tokens[ProviderType.GITHUB].token.get_secret_value()
        == 'github_token_123'
    )
    assert (
        provider_tokens[ProviderType.GITLAB].token.get_secret_value()
        == 'gitlab_token_456'
    )
    assert (
        provider_tokens[ProviderType.BITBUCKET].token.get_secret_value()
        == 'username:bitbucket_app_password'
    )


@patch('openhands.core.setup.call_async_from_sync')
@patch('openhands.core.setup.get_file_store')
@patch('openhands.core.setup.EventStream')
def test_initialize_repository_for_runtime_without_bitbucket_token(
    mock_event_stream, mock_get_file_store, mock_call_async_from_sync
):
    """Test that initialize_repository_for_runtime works without BITBUCKET_TOKEN."""
    from openhands.core.setup import initialize_repository_for_runtime
    from openhands.integrations.provider import ProviderType

    # Mock runtime
    mock_runtime = MagicMock()
    mock_runtime.clone_or_init_repo = AsyncMock(return_value='test-repo')
    mock_runtime.maybe_run_setup_script = MagicMock()
    mock_runtime.maybe_setup_git_hooks = MagicMock()

    # Mock call_async_from_sync to return the expected result
    mock_call_async_from_sync.return_value = 'test-repo'

    # Set up environment without BITBUCKET_TOKEN but with other tokens
    with patch.dict(
        os.environ,
        {'GITHUB_TOKEN': 'github_token_123', 'GITLAB_TOKEN': 'gitlab_token_456'},
        clear=False,
    ):
        # Ensure BITBUCKET_TOKEN is not in environment
        if 'BITBUCKET_TOKEN' in os.environ:
            del os.environ['BITBUCKET_TOKEN']

        result = initialize_repository_for_runtime(
            runtime=mock_runtime, selected_repository='openhands/test-repo'
        )

    # Verify the result
    assert result == 'test-repo'

    # Verify that call_async_from_sync was called
    mock_call_async_from_sync.assert_called_once()
    args, kwargs = mock_call_async_from_sync.call_args

    # Check that provider tokens were passed correctly
    provider_tokens = args[2]  # Third argument is immutable_provider_tokens
    assert provider_tokens is not None

    # Verify only GitHub and GitLab are present, not Bitbucket
    assert ProviderType.GITHUB in provider_tokens
    assert ProviderType.GITLAB in provider_tokens
    assert ProviderType.BITBUCKET not in provider_tokens
