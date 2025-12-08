from unittest.mock import MagicMock, patch

import pytest
from integrations.github.github_view import get_user_proactive_conversation_setting
from storage.user_settings import UserSettings

pytestmark = pytest.mark.asyncio


# Mock the call_sync_from_async function to return the result of the function directly
def mock_call_sync_from_async(func, *args, **kwargs):
    return func(*args, **kwargs)


@pytest.fixture
def mock_session():
    session = MagicMock()
    query = MagicMock()
    filter = MagicMock()

    # Mock the context manager behavior
    session.__enter__.return_value = session

    session.query.return_value = query
    query.filter.return_value = filter

    return session, query, filter


async def test_get_user_proactive_conversation_setting_no_user_id():
    """Test that the function returns False when no user ID is provided."""
    with patch(
        'integrations.github.github_view.ENABLE_PROACTIVE_CONVERSATION_STARTERS',
        True,
    ):
        assert await get_user_proactive_conversation_setting(None) is False

    with patch(
        'integrations.github.github_view.ENABLE_PROACTIVE_CONVERSATION_STARTERS',
        False,
    ):
        assert await get_user_proactive_conversation_setting(None) is False


async def test_get_user_proactive_conversation_setting_user_not_found(mock_session):
    """Test that False is returned when the user is not found."""
    session, query, filter = mock_session
    filter.first.return_value = None

    with patch('integrations.github.github_view.session_maker', return_value=session):
        with patch(
            'integrations.github.github_view.ENABLE_PROACTIVE_CONVERSATION_STARTERS',
            True,
        ):
            with patch(
                'integrations.github.github_view.call_sync_from_async',
                side_effect=mock_call_sync_from_async,
            ):
                assert await get_user_proactive_conversation_setting('user-id') is False


async def test_get_user_proactive_conversation_setting_user_setting_none(mock_session):
    """Test that False is returned when the user setting is None."""
    session, query, filter = mock_session
    user_settings = MagicMock(spec=UserSettings)
    user_settings.enable_proactive_conversation_starters = None
    filter.first.return_value = user_settings

    with patch('integrations.github.github_view.session_maker', return_value=session):
        with patch(
            'integrations.github.github_view.ENABLE_PROACTIVE_CONVERSATION_STARTERS',
            True,
        ):
            with patch(
                'integrations.github.github_view.call_sync_from_async',
                side_effect=mock_call_sync_from_async,
            ):
                assert await get_user_proactive_conversation_setting('user-id') is False


async def test_get_user_proactive_conversation_setting_user_setting_true(mock_session):
    """Test that True is returned when the user setting is True and the global setting is True."""
    session, query, filter = mock_session
    user_settings = MagicMock(spec=UserSettings)
    user_settings.enable_proactive_conversation_starters = True
    filter.first.return_value = user_settings

    with patch('integrations.github.github_view.session_maker', return_value=session):
        with patch(
            'integrations.github.github_view.ENABLE_PROACTIVE_CONVERSATION_STARTERS',
            True,
        ):
            with patch(
                'integrations.github.github_view.call_sync_from_async',
                side_effect=mock_call_sync_from_async,
            ):
                assert await get_user_proactive_conversation_setting('user-id') is True


async def test_get_user_proactive_conversation_setting_user_setting_false(mock_session):
    """Test that False is returned when the user setting is False, regardless of global setting."""
    session, query, filter = mock_session
    user_settings = MagicMock(spec=UserSettings)
    user_settings.enable_proactive_conversation_starters = False
    filter.first.return_value = user_settings

    with patch('integrations.github.github_view.session_maker', return_value=session):
        with patch(
            'integrations.github.github_view.ENABLE_PROACTIVE_CONVERSATION_STARTERS',
            True,
        ):
            with patch(
                'integrations.github.github_view.call_sync_from_async',
                side_effect=mock_call_sync_from_async,
            ):
                assert await get_user_proactive_conversation_setting('user-id') is False
