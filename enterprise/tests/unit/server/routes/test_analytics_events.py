"""Tests for the generic frontend analytics events router.

Covers ``POST /api/analytics/events``:
- Happy path: returns 200 OK + ``status='ok'`` and forwards the event
  to ``analytics.capture`` with ``event=<event_type>`` and the remaining
  payload fields as properties.
- Silently no-ops when there is no authenticated user.
- Silently no-ops when analytics is disabled (``get_analytics_service``
  returns ``None``).
- Swallows analytics exceptions so a telemetry outage never breaks the
  click.
- Rejects payloads whose ``event_type`` is not allowlisted.
- Rejects payloads whose properties (e.g. ``git_provider``) fall outside
  the typed allow-list - prevents arbitrary client-controlled values from
  reaching PostHog.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError
from server.routes.analytics_events import (
    CreatePrButtonClickedEvent,
    track_frontend_event,
)


@pytest.mark.asyncio
async def test_returns_ok_and_forwards_event_on_happy_path():
    """Endpoint forwards the event to capture() with the right event name
    and properties (event_type is stripped from properties)."""
    mock_analytics = MagicMock()
    mock_ctx = MagicMock(org_id='org-123')

    with (
        patch(
            'server.routes.analytics_events.get_analytics_service',
            return_value=mock_analytics,
        ),
        patch(
            'server.routes.analytics_events.resolve_analytics_context',
            new_callable=AsyncMock,
            return_value=mock_ctx,
        ),
    ):
        result = await track_frontend_event(
            body=CreatePrButtonClickedEvent(
                event_type='create pr button clicked',
                git_provider='github',
            ),
            user_id='user-123',
        )

    assert result.status == 'ok'
    mock_analytics.capture.assert_called_once_with(
        ctx=mock_ctx,
        event='create pr button clicked',
        properties={'git_provider': 'github'},
    )


@pytest.mark.asyncio
async def test_skips_tracking_when_unauthenticated():
    """No user_id means no tracking - but endpoint still returns ok."""
    mock_analytics = MagicMock()

    with (
        patch(
            'server.routes.analytics_events.get_analytics_service',
            return_value=mock_analytics,
        ),
        patch(
            'server.routes.analytics_events.resolve_analytics_context',
            new_callable=AsyncMock,
        ) as mock_resolve,
    ):
        result = await track_frontend_event(
            body=CreatePrButtonClickedEvent(
                event_type='create pr button clicked',
                git_provider='gitlab',
            ),
            user_id=None,
        )

    assert result.status == 'ok'
    mock_analytics.capture.assert_not_called()
    mock_resolve.assert_not_called()


@pytest.mark.asyncio
async def test_skips_tracking_when_analytics_disabled():
    """When get_analytics_service returns None, endpoint still returns ok."""
    with (
        patch(
            'server.routes.analytics_events.get_analytics_service',
            return_value=None,
        ),
        patch(
            'server.routes.analytics_events.resolve_analytics_context',
            new_callable=AsyncMock,
        ) as mock_resolve,
    ):
        result = await track_frontend_event(
            body=CreatePrButtonClickedEvent(
                event_type='create pr button clicked',
                git_provider='github',
            ),
            user_id='user-123',
        )

    assert result.status == 'ok'
    mock_resolve.assert_not_called()


@pytest.mark.asyncio
async def test_swallows_analytics_exceptions():
    """Telemetry failures must not bubble up to the user."""
    mock_analytics = MagicMock()
    mock_analytics.capture.side_effect = RuntimeError('posthog down')

    with (
        patch(
            'server.routes.analytics_events.get_analytics_service',
            return_value=mock_analytics,
        ),
        patch(
            'server.routes.analytics_events.resolve_analytics_context',
            new_callable=AsyncMock,
            return_value=MagicMock(org_id=None),
        ),
    ):
        result = await track_frontend_event(
            body=CreatePrButtonClickedEvent(
                event_type='create pr button clicked',
                git_provider='github',
            ),
            user_id='user-123',
        )

    assert result.status == 'ok'


@pytest.mark.asyncio
async def test_accepts_missing_git_provider():
    """Properties default to None when the payload omits them; the event
    still fires with the default included in properties."""
    mock_analytics = MagicMock()

    with (
        patch(
            'server.routes.analytics_events.get_analytics_service',
            return_value=mock_analytics,
        ),
        patch(
            'server.routes.analytics_events.resolve_analytics_context',
            new_callable=AsyncMock,
            return_value=MagicMock(org_id=None),
        ),
    ):
        result = await track_frontend_event(
            body=CreatePrButtonClickedEvent(
                event_type='create pr button clicked',
            ),
            user_id='user-123',
        )

    assert result.status == 'ok'
    mock_analytics.capture.assert_called_once()
    kwargs = mock_analytics.capture.call_args.kwargs
    assert kwargs['event'] == 'create pr button clicked'
    assert kwargs['properties'] == {'git_provider': None}


def test_payload_rejects_unknown_event_type():
    """Pydantic must reject event_type values not in the allow-list so
    unknown events never reach PostHog."""
    with pytest.raises(ValidationError):
        CreatePrButtonClickedEvent.model_validate(
            {'event_type': 'attacker_event', 'git_provider': 'github'}
        )


def test_payload_rejects_unknown_git_provider():
    """Provider Literal must reject unknown values so arbitrary
    client-controlled strings never become PostHog properties."""
    with pytest.raises(ValidationError):
        CreatePrButtonClickedEvent.model_validate(
            {
                'event_type': 'create pr button clicked',
                'git_provider': 'attacker-provided',
            }
        )


def test_payload_accepts_all_known_git_providers():
    """Sanity check that every Provider value the frontend can produce
    is accepted by the typed payload (kept in sync with ProviderOptions)."""
    for provider in (
        'github',
        'gitlab',
        'bitbucket',
        'bitbucket_data_center',
        'azure_devops',
        'forgejo',
        'enterprise_sso',
    ):
        evt = CreatePrButtonClickedEvent(
            event_type='create pr button clicked', git_provider=provider
        )
        assert evt.git_provider == provider
