"""Tests for BitBucketResolverMixin: get_pr_title_and_body, get_pr_comments,
_process_raw_comments, reply_to_pr_comment, and the MRO contract."""

from unittest.mock import patch

import pytest
from pydantic import SecretStr

from openhands.app_server.integrations.bitbucket.bitbucket_service import (
    BitBucketService,
)
from openhands.app_server.integrations.service_types import Comment, RequestMethod


@pytest.fixture
def svc() -> BitBucketService:
    return BitBucketService(token=SecretStr('user:pass'))


# ── get_pr_title_and_body ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_pr_title_and_body_returns_title_and_description(
    svc: BitBucketService,
) -> None:
    mock_response = {'title': 'Fix the bug', 'description': 'Detailed description'}
    with patch.object(
        svc, '_make_request', return_value=(mock_response, {})
    ) as mock_req:
        title, body = await svc.get_pr_title_and_body('ws', 'repo', 42)

    assert (title, body) == ('Fix the bug', 'Detailed description')
    called_url = mock_req.call_args[0][0]
    assert '/repositories/ws/repo/pullrequests/42' in called_url


@pytest.mark.asyncio
async def test_get_pr_title_and_body_missing_fields_returns_empty_strings(
    svc: BitBucketService,
) -> None:
    with patch.object(svc, '_make_request', return_value=({}, {})):
        title, body = await svc.get_pr_title_and_body('ws', 'repo', 1)

    assert (title, body) == ('', '')


# ── get_pr_comments ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_pr_comments_filters_deleted_and_inline(
    svc: BitBucketService,
) -> None:
    page = {
        'values': [
            {
                'id': 1,
                'content': {'raw': 'Looks good!'},
                'user': {'display_name': 'Alice', 'nickname': 'alice'},
                'created_on': '2024-01-01T10:00:00+00:00',
                'updated_on': '2024-01-01T10:00:00+00:00',
                'deleted': False,
            },
            {
                'id': 2,
                'content': {'raw': 'Should be hidden'},
                'user': {'display_name': 'Bob', 'nickname': 'bob'},
                'created_on': '2024-01-01T11:00:00+00:00',
                'updated_on': '2024-01-01T11:00:00+00:00',
                'deleted': True,
            },
            {
                'id': 3,
                'content': {'raw': 'Inline note'},
                'user': {'display_name': 'Carol', 'nickname': 'carol'},
                'created_on': '2024-01-01T12:00:00+00:00',
                'updated_on': '2024-01-01T12:00:00+00:00',
                'deleted': False,
                'inline': {'path': 'src/x.py', 'to': 5},
            },
        ],
    }
    with patch.object(svc, '_make_request', return_value=(page, {})):
        comments = await svc.get_pr_comments('ws', 'repo', 42, max_comments=10)

    assert [c.id for c in comments] == ['1']


@pytest.mark.asyncio
async def test_get_pr_comments_paginates_via_next(svc: BitBucketService) -> None:
    page1 = {
        'values': [
            {
                'id': 1,
                'content': {'raw': 'first'},
                'user': {'display_name': 'A', 'nickname': 'a'},
                'created_on': '2024-01-01T10:00:00+00:00',
                'updated_on': '2024-01-01T10:00:00+00:00',
                'deleted': False,
            }
        ],
        'next': 'https://api.bitbucket.org/2.0/.../comments?page=2',
    }
    page2 = {
        'values': [
            {
                'id': 2,
                'content': {'raw': 'second'},
                'user': {'display_name': 'B', 'nickname': 'b'},
                'created_on': '2024-01-01T11:00:00+00:00',
                'updated_on': '2024-01-01T11:00:00+00:00',
                'deleted': False,
            }
        ],
    }
    with patch.object(
        svc, '_make_request', side_effect=[(page1, {}), (page2, {})]
    ) as mock_req:
        comments = await svc.get_pr_comments('ws', 'repo', 1, max_comments=10)

    assert mock_req.call_count == 2
    assert [c.id for c in comments] == ['1', '2']


@pytest.mark.asyncio
async def test_get_pr_comments_respects_max_comments(svc: BitBucketService) -> None:
    page = {
        'values': [
            {
                'id': i,
                'content': {'raw': f'c{i}'},
                'user': {'display_name': f'u{i}', 'nickname': f'u{i}'},
                'created_on': f'2024-01-01T10:0{i}:00+00:00',
                'updated_on': f'2024-01-01T10:0{i}:00+00:00',
                'deleted': False,
            }
            for i in range(5)
        ],
    }
    with patch.object(svc, '_make_request', return_value=(page, {})):
        comments = await svc.get_pr_comments('ws', 'repo', 1, max_comments=3)

    assert [c.id for c in comments] == ['2', '3', '4']


# ── _process_raw_comments ─────────────────────────────────────────────────────


def test_process_raw_comments_sorts_by_created_on(svc: BitBucketService) -> None:
    raw = [
        {
            'id': 2,
            'content': {'raw': 'second'},
            'user': {'display_name': 'B', 'nickname': 'b'},
            'created_on': '2024-01-02T00:00:00+00:00',
            'updated_on': '2024-01-02T00:00:00+00:00',
            'deleted': False,
        },
        {
            'id': 1,
            'content': {'raw': 'first'},
            'user': {'display_name': 'A', 'nickname': 'a'},
            'created_on': '2024-01-01T00:00:00+00:00',
            'updated_on': '2024-01-01T00:00:00+00:00',
            'deleted': False,
        },
    ]

    comments = svc._process_raw_comments(raw, max_comments=10)

    assert [c.id for c in comments] == ['1', '2']
    assert all(isinstance(c, Comment) for c in comments)


# ── reply_to_pr_comment ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reply_to_pr_comment_with_parent_id_posts_threaded_reply(
    svc: BitBucketService,
) -> None:
    with patch.object(svc, '_make_request', return_value=({}, {})) as mock_req:
        await svc.reply_to_pr_comment(
            workspace='ws',
            repo_slug='repo',
            pr_id=7,
            body='Working on it',
            parent_comment_id=99,
        )

    args, kwargs = mock_req.call_args
    assert args[0].endswith('/repositories/ws/repo/pullrequests/7/comments')
    assert kwargs['method'] == RequestMethod.POST
    payload = kwargs['params']
    assert payload['content']['raw'] == 'Working on it'
    assert payload['parent'] == {'id': 99}


# ── user_has_write_access_for ─────────────────────────────────────────────────


def _ws_perm(level: str) -> dict:
    return {'values': [{'permission': level}]}


@pytest.mark.asyncio
async def test_user_has_write_access_for_returns_true_for_collaborator(
    svc: BitBucketService,
) -> None:
    with patch.object(
        svc, '_make_request', return_value=(_ws_perm('collaborator'), {})
    ) as mock_req:
        result = await svc.user_has_write_access_for('ws', 'repo', '712020:abc')

    assert result is True
    called_url = mock_req.call_args[0][0]
    called_params = mock_req.call_args[0][1]
    assert called_url.endswith('/workspaces/ws/permissions')
    assert called_params == {'q': 'user.account_id="712020:abc"'}


@pytest.mark.asyncio
async def test_user_has_write_access_for_returns_true_for_owner(
    svc: BitBucketService,
) -> None:
    with patch.object(svc, '_make_request', return_value=(_ws_perm('owner'), {})):
        assert await svc.user_has_write_access_for('ws', 'repo', '712020:abc') is True


@pytest.mark.asyncio
async def test_user_has_write_access_for_returns_false_for_member(
    svc: BitBucketService,
) -> None:
    with patch.object(svc, '_make_request', return_value=(_ws_perm('member'), {})):
        assert await svc.user_has_write_access_for('ws', 'repo', '712020:abc') is False


@pytest.mark.asyncio
async def test_user_has_write_access_for_returns_false_when_no_membership(
    svc: BitBucketService,
) -> None:
    with patch.object(svc, '_make_request', return_value=({'values': []}, {})):
        assert await svc.user_has_write_access_for('ws', 'repo', '712020:abc') is False


@pytest.mark.asyncio
async def test_user_has_write_access_for_returns_false_when_request_fails(
    svc: BitBucketService,
) -> None:
    with patch.object(svc, '_make_request', side_effect=RuntimeError('403')):
        assert await svc.user_has_write_access_for('ws', 'repo', '712020:abc') is False


@pytest.mark.asyncio
async def test_user_has_write_access_for_filters_by_uuid_when_no_account_id(
    svc: BitBucketService,
) -> None:
    with patch.object(
        svc, '_make_request', return_value=(_ws_perm('collaborator'), {})
    ) as mock_req:
        await svc.user_has_write_access_for('ws', 'repo', '9339e48a-a6ce-4791-99f2')

    called_params = mock_req.call_args[0][1]
    assert called_params == {'q': 'user.uuid="{9339e48a-a6ce-4791-99f2}"'}


# ── _get_headers token resolution ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_headers_populates_token_via_get_latest_token_when_empty() -> None:
    """Regression: services constructed with ``external_auth_id`` start with
    an empty ``self.token``; _get_headers must lazy-load via
    ``get_latest_token`` or it ships ``Authorization: Bearer `` (which httpx
    rejects as ``Illegal header value``)."""
    svc = BitBucketService()
    assert svc.token.get_secret_value() == ''

    with patch.object(
        svc, 'get_latest_token', return_value=SecretStr('resolved-token')
    ):
        headers = await svc._get_headers()

    assert headers['Authorization'] == 'Bearer resolved-token'


# ── MRO contract ──────────────────────────────────────────────────────────────


def test_mro_includes_resolver_mixin_before_base_git_service() -> None:
    mro_names = [cls.__name__ for cls in BitBucketService.__mro__]

    assert 'BitBucketResolverMixin' in mro_names
    assert mro_names.index('BitBucketResolverMixin') < mro_names.index('BaseGitService')
