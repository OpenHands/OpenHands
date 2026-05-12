"""Tests for the onboarding route.

Covers:
- Successful onboarding submission with analytics tracking
- Analytics tracking with org_id and group_identify
- Handling when analytics service is not available
- Handling when user_id is not available
- Error handling when analytics fails
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from server.routes.onboarding import (
    OnboardingResponse,
    OnboardingSubmission,
    submit_onboarding,
)

from openhands.analytics.analytics_context import AnalyticsContext


@pytest.fixture
def mock_analytics_service():
    """Create a mock analytics service."""
    service = MagicMock()
    service.track_onboarding_completed = MagicMock()
    service.group_identify = MagicMock()
    return service


@pytest.fixture
def mock_analytics_context():
    """Create a mock analytics context with consent and org_id."""
    return AnalyticsContext(
        user_id='test-user-123',
        consented=True,
        org_id='org-abc-456',
        user=None,
    )


@pytest.fixture
def mock_analytics_context_no_org():
    """Create a mock analytics context without org_id."""
    return AnalyticsContext(
        user_id='test-user-123',
        consented=True,
        org_id=None,
        user=None,
    )


@pytest.fixture
def mock_analytics_context_no_consent():
    """Create a mock analytics context without consent."""
    return AnalyticsContext(
        user_id='test-user-123',
        consented=False,
        org_id='org-abc-456',
        user=None,
    )


@pytest.mark.asyncio
async def test_submit_onboarding_returns_ok_response():
    """submit_onboarding returns OnboardingResponse with status ok and redirect_url /."""
    body = OnboardingSubmission(selections={'role': 'developer', 'org_size': '11-50'})

    with patch('openhands.analytics.get_analytics_service', return_value=None):
        result = await submit_onboarding(body=body, user_id='test-user')

    assert isinstance(result, OnboardingResponse)
    assert result.status == 'ok'
    assert result.redirect_url == '/'


@pytest.mark.asyncio
async def test_submit_onboarding_tracks_analytics_event(
    mock_analytics_service, mock_analytics_context
):
    """submit_onboarding calls track_onboarding_completed with correct parameters."""
    body = OnboardingSubmission(
        selections={
            'role': 'software_engineer',
            'org_size': 'solo',
            'use_case': ['new_features', 'fixing_bugs'],
        }
    )

    with (
        patch(
            'openhands.analytics.get_analytics_service',
            return_value=mock_analytics_service,
        ),
        patch(
            'openhands.analytics.resolve_analytics_context',
            new_callable=AsyncMock,
            return_value=mock_analytics_context,
        ),
    ):
        await submit_onboarding(body=body, user_id='test-user-123')

    mock_analytics_service.track_onboarding_completed.assert_called_once()
    call_kwargs = mock_analytics_service.track_onboarding_completed.call_args.kwargs
    assert call_kwargs['ctx'].user_id == 'test-user-123'
    assert call_kwargs['ctx'].org_id == 'org-abc-456'
    assert call_kwargs['ctx'].consented is True
    assert call_kwargs['selections'] == {
        'role': 'software_engineer',
        'org_size': 'solo',
        'use_case': ['new_features', 'fixing_bugs'],
    }


@pytest.mark.asyncio
async def test_submit_onboarding_calls_group_identify_when_org_id_present(
    mock_analytics_service, mock_analytics_context
):
    """submit_onboarding calls group_identify when org_id is present."""
    body = OnboardingSubmission(selections={'role': 'developer'})

    with (
        patch(
            'openhands.analytics.get_analytics_service',
            return_value=mock_analytics_service,
        ),
        patch(
            'openhands.analytics.resolve_analytics_context',
            new_callable=AsyncMock,
            return_value=mock_analytics_context,
        ),
    ):
        await submit_onboarding(body=body, user_id='test-user-123')

    mock_analytics_service.group_identify.assert_called_once()
    call_kwargs = mock_analytics_service.group_identify.call_args.kwargs
    assert call_kwargs['ctx'].user_id == 'test-user-123'
    assert call_kwargs['ctx'].consented is True
    assert call_kwargs['group_type'] == 'org'
    assert call_kwargs['group_key'] == 'org-abc-456'
    assert 'onboarding_completed_at' in call_kwargs['properties']


@pytest.mark.asyncio
async def test_submit_onboarding_skips_group_identify_when_no_org_id(
    mock_analytics_service, mock_analytics_context_no_org
):
    """submit_onboarding skips group_identify when org_id is None."""
    body = OnboardingSubmission(selections={'role': 'developer'})

    with (
        patch(
            'openhands.analytics.get_analytics_service',
            return_value=mock_analytics_service,
        ),
        patch(
            'openhands.analytics.resolve_analytics_context',
            new_callable=AsyncMock,
            return_value=mock_analytics_context_no_org,
        ),
    ):
        await submit_onboarding(body=body, user_id='test-user-123')

    mock_analytics_service.track_onboarding_completed.assert_called_once()
    mock_analytics_service.group_identify.assert_not_called()


@pytest.mark.asyncio
async def test_submit_onboarding_skips_analytics_when_service_is_none():
    """submit_onboarding skips analytics when get_analytics_service returns None."""
    body = OnboardingSubmission(selections={'role': 'developer'})

    with patch('openhands.analytics.get_analytics_service', return_value=None):
        result = await submit_onboarding(body=body, user_id='test-user-123')

    # Should still return success even without analytics
    assert result.status == 'ok'


@pytest.mark.asyncio
async def test_submit_onboarding_skips_analytics_when_user_id_is_none(
    mock_analytics_service,
):
    """submit_onboarding skips analytics when user_id is None."""
    body = OnboardingSubmission(selections={'role': 'developer'})

    with patch(
        'openhands.analytics.get_analytics_service',
        return_value=mock_analytics_service,
    ):
        result = await submit_onboarding(body=body, user_id=None)

    # Should still return success
    assert result.status == 'ok'
    # Analytics should not be called
    mock_analytics_service.track_onboarding_completed.assert_not_called()


@pytest.mark.asyncio
async def test_submit_onboarding_handles_analytics_exception_gracefully(
    mock_analytics_service,
):
    """submit_onboarding catches analytics exceptions and still returns success."""
    body = OnboardingSubmission(selections={'role': 'developer'})
    mock_analytics_service.track_onboarding_completed.side_effect = RuntimeError(
        'PostHog error'
    )

    mock_context = AnalyticsContext(
        user_id='test-user-123',
        consented=True,
        org_id=None,
        user=None,
    )

    with (
        patch(
            'openhands.analytics.get_analytics_service',
            return_value=mock_analytics_service,
        ),
        patch(
            'openhands.analytics.resolve_analytics_context',
            new_callable=AsyncMock,
            return_value=mock_context,
        ),
    ):
        # Should not raise, should return success
        result = await submit_onboarding(body=body, user_id='test-user-123')

    assert result.status == 'ok'


@pytest.mark.asyncio
async def test_submit_onboarding_passes_consent_false_to_analytics(
    mock_analytics_service, mock_analytics_context_no_consent
):
    """submit_onboarding passes consented=False when user has not consented."""
    body = OnboardingSubmission(selections={'role': 'developer'})

    with (
        patch(
            'openhands.analytics.get_analytics_service',
            return_value=mock_analytics_service,
        ),
        patch(
            'openhands.analytics.resolve_analytics_context',
            new_callable=AsyncMock,
            return_value=mock_analytics_context_no_consent,
        ),
    ):
        await submit_onboarding(body=body, user_id='test-user-123')

    call_kwargs = mock_analytics_service.track_onboarding_completed.call_args.kwargs
    assert call_kwargs['ctx'].consented is False
