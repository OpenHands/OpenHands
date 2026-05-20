"""Unit tests for ``BitbucketDCWebhooksMixin``.

Mirrors ``tests/unit/integrations/bitbucket/test_bitbucket_webhooks.py``
(Bitbucket Cloud) for the Bitbucket Data Center service mixin added in
PR #14461. Each test stubs ``_make_request`` so we can assert the
exact URL / method / payload the service hits — no network, no DB.
"""

from unittest.mock import AsyncMock

import pytest

from openhands.app_server.integrations.bitbucket_data_center.bitbucket_dc_service import (
    BitbucketDCService,
)
from openhands.app_server.integrations.service_types import RequestMethod


def _make_service() -> BitbucketDCService:
    """Build a service rooted at a known base_domain so URL assertions are stable."""
    return BitbucketDCService(token=None, base_domain='bitbucket.example.com')


@pytest.mark.asyncio
async def test_create_repository_webhook_posts_bbdc_payload():
    service = _make_service()
    service._make_request = AsyncMock(  # type: ignore[method-assign]
        return_value=({'id': 42, 'name': 'OpenHands Resolver'}, {})
    )

    webhook_id = await service.create_repository_webhook(
        owner='PROJ',
        repo_slug='repo-1',
        name='OpenHands Resolver',
        webhook_url='https://app.example.com/integration/bitbucket-dc/events',
        webhook_secret='secret-123',
        events=['pr:comment:added', 'pr:comment:edited'],
    )

    # BBDC returns numeric ids; we normalize to str to match the storage column.
    assert webhook_id == '42'
    service._make_request.assert_awaited_once_with(
        url='https://bitbucket.example.com/rest/api/1.0/projects/PROJ/repos/repo-1/webhooks',
        params={
            'name': 'OpenHands Resolver',
            'url': 'https://app.example.com/integration/bitbucket-dc/events',
            'active': True,
            'events': ['pr:comment:added', 'pr:comment:edited'],
            # The shared secret is nested under ``configuration`` for BBDC —
            # this is the field Cloud puts at the top level.
            'configuration': {'secret': 'secret-123'},
        },
        method=RequestMethod.POST,
    )


@pytest.mark.asyncio
async def test_update_repository_webhook_puts_full_payload():
    service = _make_service()
    service._make_request = AsyncMock(  # type: ignore[method-assign]
        return_value=({'id': 7}, {})
    )

    webhook_id = await service.update_repository_webhook(
        owner='PROJ',
        repo_slug='repo-1',
        webhook_id='7',
        name='OpenHands Resolver',
        webhook_url='https://app.example.com/integration/bitbucket-dc/events',
        webhook_secret='rotated',
        events=['pr:comment:added', 'pr:comment:edited'],
    )

    assert webhook_id == '7'
    service._make_request.assert_awaited_once_with(
        url='https://bitbucket.example.com/rest/api/1.0/projects/PROJ/repos/repo-1/webhooks/7',
        params={
            'name': 'OpenHands Resolver',
            'url': 'https://app.example.com/integration/bitbucket-dc/events',
            'active': True,
            'events': ['pr:comment:added', 'pr:comment:edited'],
            'configuration': {'secret': 'rotated'},
        },
        method=RequestMethod.PUT,
    )


@pytest.mark.asyncio
async def test_update_repository_webhook_falls_back_to_input_id_when_response_lacks_one():
    """BBDC always echoes the id, but defensive code: if it doesn't, we keep
    the id we asked it to update — better than returning ``None``."""
    service = _make_service()
    service._make_request = AsyncMock(  # type: ignore[method-assign]
        return_value=(None, {})
    )

    webhook_id = await service.update_repository_webhook(
        owner='PROJ',
        repo_slug='repo-1',
        webhook_id='9',
        name='OpenHands Resolver',
        webhook_url='https://app.example.com/integration/bitbucket-dc/events',
        webhook_secret='s',
        events=[],
    )

    assert webhook_id == '9'


@pytest.mark.asyncio
async def test_delete_repository_webhook_issues_delete():
    service = _make_service()
    service._make_request = AsyncMock(return_value=({}, {}))  # type: ignore[method-assign]

    await service.delete_repository_webhook('PROJ', 'repo-1', '42')

    service._make_request.assert_awaited_once_with(
        url='https://bitbucket.example.com/rest/api/1.0/projects/PROJ/repos/repo-1/webhooks/42',
        method=RequestMethod.DELETE,
    )


@pytest.mark.asyncio
async def test_check_webhook_exists_on_repository_returns_id_when_found():
    """Matches by URL — the idempotency key Cloud uses too."""
    service = _make_service()
    service._fetch_paginated_data = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            {'id': 1, 'url': 'https://other.example.com/events'},
            {
                'id': 42,
                'url': 'https://app.example.com/integration/bitbucket-dc/events',
            },
        ]
    )

    exists, wid = await service.check_webhook_exists_on_repository(
        'PROJ',
        'repo-1',
        'https://app.example.com/integration/bitbucket-dc/events',
    )

    assert exists is True
    # Numeric ids from BBDC get coerced to str to match the storage column.
    assert wid == '42'


@pytest.mark.asyncio
async def test_check_webhook_exists_on_repository_returns_false_when_absent():
    service = _make_service()
    service._fetch_paginated_data = AsyncMock(  # type: ignore[method-assign]
        return_value=[{'id': 1, 'url': 'https://other.example.com/events'}]
    )

    exists, wid = await service.check_webhook_exists_on_repository(
        'PROJ',
        'repo-1',
        'https://app.example.com/integration/bitbucket-dc/events',
    )

    assert exists is False
    assert wid is None


@pytest.mark.asyncio
async def test_user_has_admin_access_returns_true():
    """Trusts the OAuth scope grant — see the mixin docstring. The webhook
    API call itself is the authoritative auth check; this method is only
    used as a non-strict pre-flight so the route returns a clean 403
    instead of leaking BBDC's error in the *very* unlikely edge case we
    can detect mismatch up-front.
    """
    service = _make_service()
    # Even though there's no real I/O, the method should not raise and
    # should not depend on self.user_id being set.
    assert await service.user_has_admin_access('PROJ', 'repo-1') is True
