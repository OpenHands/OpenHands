"""Tests for BitbucketDCResolverMixin: get_pr_title_and_body, get_pr_comments, _process_raw_comments."""

from unittest.mock import patch

import pytest
from pydantic import SecretStr

from openhands.integrations.bitbucket_data_center.bitbucket_dc_service import (
    BitbucketDCService,
)
from openhands.integrations.service_types import Comment


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


# ── MRO check ─────────────────────────────────────────────────────────────────


def test_mro_includes_resolver_mixin_and_base_git_service():
    from openhands.integrations.bitbucket_data_center.service.resolver import (
        BitbucketDCResolverMixin,
    )
    from openhands.integrations.service_types import BaseGitService

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
