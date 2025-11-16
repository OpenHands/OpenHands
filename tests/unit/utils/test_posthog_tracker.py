"""Unit tests for PostHog tracking utilities."""

from unittest.mock import MagicMock, patch

import pytest

from openhands.utils.posthog_tracker import (
    alias_user_identities,
    track_agent_task_completed,
    track_credit_limit_reached,
    track_credits_purchased,
    track_user_signup_completed,
)


@pytest.fixture
def mock_posthog():
    """Mock the posthog module."""
    with patch('openhands.utils.posthog_tracker.posthog') as mock_ph:
        mock_ph.capture = MagicMock()
        yield mock_ph


def test_track_agent_task_completed_with_user_id(mock_posthog):
    """Test tracking agent task completion with user ID."""
    # Initialize posthog manually in the test
    import openhands.utils.posthog_tracker as tracker

    tracker.posthog = mock_posthog

    track_agent_task_completed(
        conversation_id='test-conversation-123',
        user_id='user-456',
        app_mode='saas',
    )

    mock_posthog.capture.assert_called_once_with(
        distinct_id='user-456',
        event='agent_task_completed',
        properties={
            'conversation_id': 'test-conversation-123',
            'user_id': 'user-456',
            'app_mode': 'saas',
        },
    )


def test_track_agent_task_completed_without_user_id(mock_posthog):
    """Test tracking agent task completion without user ID (anonymous)."""
    import openhands.utils.posthog_tracker as tracker

    tracker.posthog = mock_posthog

    track_agent_task_completed(
        conversation_id='test-conversation-789',
        user_id=None,
        app_mode='oss',
    )

    mock_posthog.capture.assert_called_once_with(
        distinct_id='conversation_test-conversation-789',
        event='agent_task_completed',
        properties={
            'conversation_id': 'test-conversation-789',
            'user_id': None,
            'app_mode': 'oss',
        },
    )


def test_track_agent_task_completed_default_app_mode(mock_posthog):
    """Test tracking with default app_mode."""
    import openhands.utils.posthog_tracker as tracker

    tracker.posthog = mock_posthog

    track_agent_task_completed(
        conversation_id='test-conversation-999',
        user_id='user-111',
    )

    mock_posthog.capture.assert_called_once_with(
        distinct_id='user-111',
        event='agent_task_completed',
        properties={
            'conversation_id': 'test-conversation-999',
            'user_id': 'user-111',
            'app_mode': 'unknown',
        },
    )


def test_track_agent_task_completed_handles_errors(mock_posthog):
    """Test that tracking errors are handled gracefully."""
    import openhands.utils.posthog_tracker as tracker

    tracker.posthog = mock_posthog
    mock_posthog.capture.side_effect = Exception('PostHog API error')

    # Should not raise an exception
    track_agent_task_completed(
        conversation_id='test-conversation-error',
        user_id='user-error',
        app_mode='saas',
    )


def test_track_agent_task_completed_when_posthog_not_installed():
    """Test tracking when posthog is not installed."""
    import openhands.utils.posthog_tracker as tracker

    # Simulate posthog not being installed
    tracker.posthog = None

    # Should not raise an exception
    track_agent_task_completed(
        conversation_id='test-conversation-no-ph',
        user_id='user-no-ph',
        app_mode='oss',
    )


def test_track_user_signup_completed(mock_posthog):
    """Test tracking user signup completion."""
    import openhands.utils.posthog_tracker as tracker

    tracker.posthog = mock_posthog

    track_user_signup_completed(
        user_id='test-user-123',
        signup_timestamp='2025-01-15T10:30:00Z',
    )

    mock_posthog.capture.assert_called_once_with(
        distinct_id='test-user-123',
        event='user_signup_completed',
        properties={
            'user_id': 'test-user-123',
            'signup_timestamp': '2025-01-15T10:30:00Z',
        },
    )


def test_track_user_signup_completed_handles_errors(mock_posthog):
    """Test that user signup tracking errors are handled gracefully."""
    import openhands.utils.posthog_tracker as tracker

    tracker.posthog = mock_posthog
    mock_posthog.capture.side_effect = Exception('PostHog API error')

    # Should not raise an exception
    track_user_signup_completed(
        user_id='test-user-error',
        signup_timestamp='2025-01-15T12:00:00Z',
    )


def test_track_user_signup_completed_when_posthog_not_installed():
    """Test user signup tracking when posthog is not installed."""
    import openhands.utils.posthog_tracker as tracker

    # Simulate posthog not being installed
    tracker.posthog = None

    # Should not raise an exception
    track_user_signup_completed(
        user_id='test-user-no-ph',
        signup_timestamp='2025-01-15T13:00:00Z',
    )


def test_track_credit_limit_reached_with_user_id(mock_posthog):
    """Test tracking credit limit reached with user ID."""
    import openhands.utils.posthog_tracker as tracker

    tracker.posthog = mock_posthog

    track_credit_limit_reached(
        conversation_id='test-conversation-456',
        user_id='user-789',
        current_budget=10.50,
        max_budget=10.00,
    )

    mock_posthog.capture.assert_called_once_with(
        distinct_id='user-789',
        event='credit_limit_reached',
        properties={
            'conversation_id': 'test-conversation-456',
            'user_id': 'user-789',
            'current_budget': 10.50,
            'max_budget': 10.00,
        },
    )


def test_track_credit_limit_reached_without_user_id(mock_posthog):
    """Test tracking credit limit reached without user ID (anonymous)."""
    import openhands.utils.posthog_tracker as tracker

    tracker.posthog = mock_posthog

    track_credit_limit_reached(
        conversation_id='test-conversation-999',
        user_id=None,
        current_budget=5.25,
        max_budget=5.00,
    )

    mock_posthog.capture.assert_called_once_with(
        distinct_id='conversation_test-conversation-999',
        event='credit_limit_reached',
        properties={
            'conversation_id': 'test-conversation-999',
            'user_id': None,
            'current_budget': 5.25,
            'max_budget': 5.00,
        },
    )


def test_track_credit_limit_reached_handles_errors(mock_posthog):
    """Test that credit limit tracking errors are handled gracefully."""
    import openhands.utils.posthog_tracker as tracker

    tracker.posthog = mock_posthog
    mock_posthog.capture.side_effect = Exception('PostHog API error')

    # Should not raise an exception
    track_credit_limit_reached(
        conversation_id='test-conversation-error',
        user_id='user-error',
        current_budget=15.00,
        max_budget=10.00,
    )


def test_track_credit_limit_reached_when_posthog_not_installed():
    """Test credit limit tracking when posthog is not installed."""
    import openhands.utils.posthog_tracker as tracker

    # Simulate posthog not being installed
    tracker.posthog = None

    # Should not raise an exception
    track_credit_limit_reached(
        conversation_id='test-conversation-no-ph',
        user_id='user-no-ph',
        current_budget=8.00,
        max_budget=5.00,
    )


def test_track_credits_purchased(mock_posthog):
    """Test tracking credits purchased."""
    import openhands.utils.posthog_tracker as tracker

    tracker.posthog = mock_posthog

    track_credits_purchased(
        user_id='test-user-999',
        amount_usd=50.00,
        credits_added=50.00,
        stripe_session_id='cs_test_abc123',
    )

    mock_posthog.capture.assert_called_once_with(
        distinct_id='test-user-999',
        event='credits_purchased',
        properties={
            'user_id': 'test-user-999',
            'amount_usd': 50.00,
            'credits_added': 50.00,
            'stripe_session_id': 'cs_test_abc123',
        },
    )


def test_track_credits_purchased_handles_errors(mock_posthog):
    """Test that credits purchased tracking errors are handled gracefully."""
    import openhands.utils.posthog_tracker as tracker

    tracker.posthog = mock_posthog
    mock_posthog.capture.side_effect = Exception('PostHog API error')

    # Should not raise an exception
    track_credits_purchased(
        user_id='test-user-error',
        amount_usd=100.00,
        credits_added=100.00,
        stripe_session_id='cs_test_error',
    )


def test_track_credits_purchased_when_posthog_not_installed():
    """Test credits purchased tracking when posthog is not installed."""
    import openhands.utils.posthog_tracker as tracker

    # Simulate posthog not being installed
    tracker.posthog = None

    # Should not raise an exception
    track_credits_purchased(
        user_id='test-user-no-ph',
        amount_usd=25.00,
        credits_added=25.00,
        stripe_session_id='cs_test_no_ph',
    )


def test_alias_user_identities(mock_posthog):
    """Test aliasing user identities.

    Verifies that posthog.alias(previous_id, distinct_id) is called correctly
    where git_login is the previous_id and keycloak_user_id is the distinct_id.
    """
    import openhands.utils.posthog_tracker as tracker

    tracker.posthog = mock_posthog
    mock_posthog.alias = MagicMock()

    alias_user_identities(
        keycloak_user_id='keycloak-123',
        git_login='git-user',
    )

    # Verify: posthog.alias(previous_id='git-user', distinct_id='keycloak-123')
    mock_posthog.alias.assert_called_once_with('git-user', 'keycloak-123')


def test_alias_user_identities_handles_errors(mock_posthog):
    """Test that aliasing errors are handled gracefully."""
    import openhands.utils.posthog_tracker as tracker

    tracker.posthog = mock_posthog
    mock_posthog.alias = MagicMock(side_effect=Exception('PostHog API error'))

    # Should not raise an exception
    alias_user_identities(
        keycloak_user_id='keycloak-error',
        git_login='git-error',
    )


def test_alias_user_identities_when_posthog_not_installed():
    """Test aliasing when posthog is not installed."""
    import openhands.utils.posthog_tracker as tracker

    # Simulate posthog not being installed
    tracker.posthog = None

    # Should not raise an exception
    alias_user_identities(
        keycloak_user_id='keycloak-no-ph',
        git_login='git-no-ph',
    )
