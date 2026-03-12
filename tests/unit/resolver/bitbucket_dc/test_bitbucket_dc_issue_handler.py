import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from openhands.core.config import LLMConfig
from openhands.integrations.service_types import ProviderType
from openhands.resolver.interfaces.bitbucket_data_center import (
    BitbucketDCIssueHandler,
    BitbucketDCPRHandler,
)
from openhands.resolver.interfaces.issue_definitions import (
    ServiceContextIssue,
    ServiceContextPR,
)
from openhands.resolver.issue_handler_factory import IssueHandlerFactory


@pytest.fixture
def handler():
    return BitbucketDCIssueHandler(
        owner='PROJ',
        repo='my-repo',
        token='user:secret',
        base_domain='bitbucket.example.com',
    )


@pytest.fixture
def llm_config():
    return LLMConfig(model='test-model', api_key=SecretStr('test-key'))


# ---------------------------------------------------------------------------
# URL / attribute tests
# ---------------------------------------------------------------------------


def test_init_sets_correct_urls(handler):
    assert handler.base_url == 'https://bitbucket.example.com/rest/api/1.0'
    assert (
        handler.download_url
        == 'https://bitbucket.example.com/rest/api/latest/projects/PROJ/repos/my-repo/archive?format=zip'
    )
    assert handler.clone_url == 'https://bitbucket.example.com/scm/proj/my-repo.git'


def test_get_headers_returns_basic_auth(handler):
    expected = base64.b64encode(b'user:secret').decode()
    headers = handler.get_headers()
    assert headers['Authorization'] == f'Basic {expected}'
    assert headers['Accept'] == 'application/json'


def test_get_headers_bare_token_with_username():
    h = BitbucketDCIssueHandler(
        owner='PROJ',
        repo='my-repo',
        token='mytoken',
        username='myuser',
        base_domain='dc.example.com',
    )
    expected = base64.b64encode(b'myuser:mytoken').decode()
    assert h.headers['Authorization'] == f'Basic {expected}'


def test_get_repo_url(handler):
    assert (
        handler.get_repo_url()
        == 'https://bitbucket.example.com/projects/PROJ/repos/my-repo'
    )


def test_get_issue_url_returns_pr_url(handler):
    assert (
        handler.get_issue_url(42)
        == 'https://bitbucket.example.com/projects/PROJ/repos/my-repo/pull-requests/42'
    )


def test_get_branch_url(handler):
    assert (
        handler.get_branch_url('feature/x')
        == 'https://bitbucket.example.com/projects/PROJ/repos/my-repo/browse?at=refs/heads/feature/x'
    )


def test_get_authorize_url_with_colon_token(handler):
    url = handler.get_authorize_url()
    assert url == 'https://user:secret@bitbucket.example.com/'


def test_get_authorize_url_with_username_and_bare_token():
    h = BitbucketDCIssueHandler(
        owner='PROJ',
        repo='my-repo',
        token='baretoken',
        username='john',
        base_domain='dc.example.com',
    )
    assert h.get_authorize_url() == 'https://john:baretoken@dc.example.com/'


# ---------------------------------------------------------------------------
# get_compare_url (requires get_default_branch_name)
# ---------------------------------------------------------------------------


def test_get_compare_url(handler):
    with patch.object(handler, 'get_default_branch_name', return_value='main'):
        url = handler.get_compare_url('feature/fix')
    assert url == (
        'https://bitbucket.example.com/projects/PROJ/repos/my-repo/compare/commits'
        '?sourceBranch=refs/heads/feature/fix&targetBranch=refs/heads/main'
    )


# ---------------------------------------------------------------------------
# API methods (mock httpx)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_issue_fetches_pr_endpoint(handler):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        'id': 7,
        'title': 'Fix the thing',
        'description': 'Some body',
        'fromRef': {'displayId': 'feature/fix'},
        'toRef': {'displayId': 'main'},
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch('httpx.AsyncClient', return_value=mock_client):
        issue = await handler.get_issue(7)

    expected_url = (
        'https://bitbucket.example.com/rest/api/1.0'
        '/projects/PROJ/repos/my-repo/pull-requests/7'
    )
    mock_client.get.assert_called_once_with(expected_url, headers=handler.headers)
    assert issue.number == 7
    assert issue.title == 'Fix the thing'
    assert issue.body == 'Some body'
    assert issue.head_branch == 'feature/fix'
    assert issue.base_branch == 'main'


def test_create_pr_uses_from_to_ref(handler):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        'id': 3,
        'links': {
            'self': [
                {
                    'href': 'https://bitbucket.example.com/projects/PROJ/repos/my-repo/pull-requests/3'
                }
            ]
        },
    }
    mock_response.raise_for_status = MagicMock()

    with patch('httpx.post', return_value=mock_response) as mock_post:
        url = handler.create_pr('Title', 'Body', 'feature/src', 'main')

    _, kwargs = mock_post.call_args
    payload = kwargs['json']
    assert payload['fromRef']['id'] == 'refs/heads/feature/src'
    assert payload['toRef']['id'] == 'refs/heads/main'
    assert payload['fromRef']['repository']['slug'] == 'my-repo'
    assert payload['fromRef']['repository']['project']['key'] == 'PROJ'
    assert (
        url
        == 'https://bitbucket.example.com/projects/PROJ/repos/my-repo/pull-requests/3'
    )


def test_create_pull_request_returns_html_url_and_number(handler):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        'id': 5,
        'links': {
            'self': [
                {
                    'href': 'https://bitbucket.example.com/projects/PROJ/repos/my-repo/pull-requests/5'
                }
            ]
        },
    }
    mock_response.raise_for_status = MagicMock()

    with patch('httpx.post', return_value=mock_response):
        result = handler.create_pull_request(
            {
                'title': 'My PR',
                'description': 'desc',
                'source_branch': 'feature/x',
                'target_branch': 'main',
            }
        )

    assert result['number'] == 5
    assert result['html_url'] == (
        'https://bitbucket.example.com/projects/PROJ/repos/my-repo/pull-requests/5'
    )


def test_send_comment_msg_uses_text_field(handler):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch('httpx.post', return_value=mock_response) as mock_post:
        handler.send_comment_msg(7, 'Hello from OpenHands')

    _, kwargs = mock_post.call_args
    assert kwargs['json'] == {'text': 'Hello from OpenHands'}


def test_reply_to_comment_posts_with_parent_id(handler):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch('httpx.post', return_value=mock_response) as mock_post:
        handler.reply_to_comment(7, '42', 'My reply')

    _, kwargs = mock_post.call_args
    assert kwargs['json']['text'] == 'My reply'
    assert kwargs['json']['parent'] == {'id': 42}


def test_branch_exists_true(handler):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        'values': [{'displayId': 'feature/fix', 'id': 'refs/heads/feature/fix'}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch('httpx.get', return_value=mock_response):
        assert handler.branch_exists('feature/fix') is True


def test_branch_exists_false(handler):
    mock_response = MagicMock()
    mock_response.json.return_value = {'values': []}
    mock_response.raise_for_status = MagicMock()

    with patch('httpx.get', return_value=mock_response):
        assert handler.branch_exists('nonexistent') is False


def test_branch_exists_no_match(handler):
    mock_response = MagicMock()
    # filterText returns similar but not exact match
    mock_response.json.return_value = {
        'values': [{'displayId': 'feature/fix-extended'}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch('httpx.get', return_value=mock_response):
        assert handler.branch_exists('feature/fix') is False


def test_get_default_branch_name_reads_display_id(handler):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        'defaultBranch': {'displayId': 'main', 'id': 'refs/heads/main'}
    }
    mock_response.raise_for_status = MagicMock()

    with patch('httpx.get', return_value=mock_response):
        assert handler.get_default_branch_name() == 'main'


def test_get_default_branch_name_strips_refs_heads_prefix(handler):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        'defaultBranch': {'displayId': 'refs/heads/develop'}
    }
    mock_response.raise_for_status = MagicMock()

    with patch('httpx.get', return_value=mock_response):
        assert handler.get_default_branch_name() == 'develop'


def test_get_default_branch_name_fallback_to_master(handler):
    import httpx as httpx_module

    with patch('httpx.get', side_effect=httpx_module.HTTPError('connection refused')):
        assert handler.get_default_branch_name() == 'master'


def test_get_context_returns_early(handler):
    """get_context_from_external_issues_references should return closing_issues without API calls."""
    closing_issues = ['issue body 1']
    with patch('httpx.get') as mock_get:
        result = handler.get_context_from_external_issues_references(
            closing_issues=closing_issues,
            closing_issue_numbers=[1],
            issue_body='some body',
            review_comments=None,
            review_threads=[],
            thread_comments=None,
        )
    mock_get.assert_not_called()
    assert result == ['issue body 1']


def test_download_issues_returns_empty(handler):
    result = handler.download_issues()
    assert result == []


# ---------------------------------------------------------------------------
# Factory integration tests
# ---------------------------------------------------------------------------


def test_factory_creates_dc_issue_handler(llm_config):
    factory = IssueHandlerFactory(
        owner='PROJ',
        repo='my-repo',
        token='user:secret',
        username='user',
        platform=ProviderType.BITBUCKET_DATA_CENTER,
        base_domain='bitbucket.example.com',
        issue_type='issue',
        llm_config=llm_config,
    )
    ctx = factory.create()
    assert isinstance(ctx, ServiceContextIssue)
    assert isinstance(ctx._strategy, BitbucketDCIssueHandler)
    assert ctx._strategy.owner == 'PROJ'
    assert ctx._strategy.repo == 'my-repo'
    assert ctx._strategy.base_domain == 'bitbucket.example.com'


def test_factory_creates_dc_pr_handler(llm_config):
    factory = IssueHandlerFactory(
        owner='PROJ',
        repo='my-repo',
        token='user:secret',
        username='user',
        platform=ProviderType.BITBUCKET_DATA_CENTER,
        base_domain='bitbucket.example.com',
        issue_type='pr',
        llm_config=llm_config,
    )
    ctx = factory.create()
    assert isinstance(ctx, ServiceContextPR)
    assert isinstance(ctx._strategy, BitbucketDCPRHandler)
