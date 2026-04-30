"""Tests for BitbucketDCResolverMixin: get_pr_title_and_body, get_pr_comments,
_process_raw_comments, reply_to_pr_comment, and user_has_write_access_for."""

from unittest.mock import patch

import pytest
from pydantic import SecretStr

from openhands.app_server.integrations.bitbucket_data_center.bitbucket_dc_service import (
    BitbucketDCService,
)
from openhands.app_server.integrations.service_types import Comment, RequestMethod


@pytest.fixture
def svc():
    return BitbucketDCService(
        token=SecretStr('user:pass'), base_domain='host.example.com'
    )


# ── get_pr_title_and_body ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_pr_title_and_body(svc):
    mock_response = {'title': 'Fix the bug', 'description': 'Detailed description'}
    with patch.object(
        svc, '_make_request', return_value=(mock_response, {})
    ) as mock_req:
        title, body = await svc.get_pr_title_and_body('PROJ', 'myrepo', 42)

    assert title == 'Fix the bug'
    assert body == 'Detailed description'
    called_url = mock_req.call_args[0][0]
    assert '/projects/PROJ/repos/myrepo/pull-requests/42' in called_url


@pytest.mark.asyncio
async def test_get_pr_title_and_body_missing_fields(svc):
    with patch.object(svc, '_make_request', return_value=({}, {})):
        title, body = await svc.get_pr_title_and_body('PROJ', 'myrepo', 1)

    assert title == ''
    assert body == ''


# ── get_pr_comments ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_pr_comments_returns_comments(svc):
    activities = {
        'values': [
            {
                'action': 'COMMENTED',
                'comment': {
                    'id': 10,
                    'text': 'Looks good!',
                    'author': {'slug': 'alice', 'name': 'Alice'},
                    'createdDate': 1_700_000_000_000,
                    'updatedDate': 1_700_000_000_000,
                },
            },
            {
                'action': 'APPROVED',  # should be ignored
                'comment': {},
            },
            {
                'action': 'COMMENTED',
                'comment': {
                    'id': 11,
                    'text': 'Please fix tests',
                    'author': {'slug': 'bob', 'name': 'Bob'},
                    'createdDate': 1_700_000_001_000,
                    'updatedDate': 1_700_000_001_000,
                },
            },
        ],
        'isLastPage': True,
    }

    with patch.object(svc, '_make_request', return_value=(activities, {})):
        comments = await svc.get_pr_comments('PROJ', 'myrepo', 42, max_comments=10)

    assert len(comments) == 2
    assert all(isinstance(c, Comment) for c in comments)
    assert comments[0].author == 'alice'
    assert comments[0].body == 'Looks good!'
    assert comments[1].author == 'bob'


@pytest.mark.asyncio
async def test_get_pr_comments_respects_max(svc):
    activities = {
        'values': [
            {
                'action': 'COMMENTED',
                'comment': {
                    'id': i,
                    'text': f'comment {i}',
                    'author': {'slug': f'user{i}'},
                    'createdDate': 1_700_000_000_000 + i * 1000,
                    'updatedDate': 1_700_000_000_000 + i * 1000,
                },
            }
            for i in range(10)
        ],
        'isLastPage': True,
    }

    with patch.object(svc, '_make_request', return_value=(activities, {})):
        comments = await svc.get_pr_comments('PROJ', 'myrepo', 1, max_comments=3)

    assert len(comments) == 3


@pytest.mark.asyncio
async def test_get_pr_comments_empty(svc):
    with patch.object(
        svc, '_make_request', return_value=({'values': [], 'isLastPage': True}, {})
    ):
        comments = await svc.get_pr_comments('PROJ', 'myrepo', 1)

    assert comments == []


# ── _process_raw_comments ─────────────────────────────────────────────────────


def test_process_raw_comments_sorts_by_date(svc):
    raw = [
        {
            'id': 2,
            'text': 'second',
            'author': {'slug': 'bob'},
            'createdDate': 1_700_000_002_000,
            'updatedDate': 1_700_000_002_000,
        },
        {
            'id': 1,
            'text': 'first',
            'author': {'slug': 'alice'},
            'createdDate': 1_700_000_001_000,
            'updatedDate': 1_700_000_001_000,
        },
    ]
    comments = svc._process_raw_comments(raw, max_comments=10)
    assert comments[0].id == '1'
    assert comments[1].id == '2'


def test_process_raw_comments_missing_timestamps(svc):
    raw = [{'id': 5, 'text': 'no dates', 'author': {'slug': 'eve'}}]
    comments = svc._process_raw_comments(raw)
    assert len(comments) == 1
    assert comments[0].id == '5'


# ── reply_to_pr_comment ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reply_to_pr_comment_posts_to_dc_pr_comments_endpoint_with_text_field(
    svc,
):
    with patch.object(svc, '_make_request', return_value=({}, {})) as mock_req:
        await svc.reply_to_pr_comment(
            owner='PROJ',
            repo_slug='myrepo',
            pr_id=7,
            body='Working on it',
            parent_comment_id=99,
        )

    args, kwargs = mock_req.call_args
    assert args[0].endswith('/projects/PROJ/repos/myrepo/pull-requests/7/comments')
    assert kwargs['method'] == RequestMethod.POST
    payload = kwargs['params']
    assert payload == {'text': 'Working on it', 'parent': {'id': 99}}


@pytest.mark.asyncio
async def test_reply_to_pr_comment_includes_anchor_when_provided(svc):
    anchor = {
        'path': 'src/x.py',
        'line': 12,
        'lineType': 'ADDED',
        'fileType': 'TO',
    }
    with patch.object(svc, '_make_request', return_value=({}, {})) as mock_req:
        await svc.reply_to_pr_comment(
            owner='PROJ',
            repo_slug='myrepo',
            pr_id=7,
            body='Inline note',
            anchor=anchor,
        )

    payload = mock_req.call_args.kwargs['params']
    assert payload == {'text': 'Inline note', 'anchor': anchor}


# ── user_has_write_access_for ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_user_has_write_access_for_returns_true_for_identified_user(svc):
    """Bitbucket Data Center has no OAuth 2.0 endpoint that introspects
    permissions without REPO_ADMIN scope. The resolver downgrades to
    "any identifiable actor is allowed" — webhook signature verification
    plus the actor coming through in the payload is the gate.
    """
    assert await svc.user_has_write_access_for('PROJ', 'myrepo', 'alice') is True


@pytest.mark.asyncio
async def test_user_has_write_access_for_returns_false_for_empty_user(svc):
    """An empty actor identity means we couldn't extract a user from the
    payload — treat as a parse failure and reject.
    """
    assert await svc.user_has_write_access_for('PROJ', 'myrepo', '') is False


@pytest.mark.asyncio
async def test_user_has_write_access_for_does_not_call_admin_endpoint(svc):
    """Regression guard: the previous implementation hit
    ``/permissions/users``, which requires REPO_ADMIN scope and 401s for
    every install that doesn't grant it. The current implementation must
    not make that call.
    """
    with patch.object(svc, '_make_request') as mock_req:
        await svc.user_has_write_access_for('PROJ', 'myrepo', 'alice')

    mock_req.assert_not_called()


# ── MRO check ─────────────────────────────────────────────────────────────────


def test_mro_includes_resolver_mixin_and_base_git_service():
    from openhands.app_server.integrations.bitbucket_data_center.service.resolver import (
        BitbucketDCResolverMixin,
    )
    from openhands.app_server.integrations.service_types import BaseGitService

    mro_names = [cls.__name__ for cls in BitbucketDCService.__mro__]
    assert 'BitbucketDCResolverMixin' in mro_names
    assert 'BaseGitService' in mro_names

    # Resolver mixin should appear before BaseGitService
    assert mro_names.index('BitbucketDCResolverMixin') < mro_names.index(
        'BaseGitService'
    )

    # Verify instances
    assert issubclass(BitbucketDCService, BitbucketDCResolverMixin)
    assert issubclass(BitbucketDCService, BaseGitService)
