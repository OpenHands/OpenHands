from unittest.mock import MagicMock, patch

import pytest

# Mock the database module before importing
with patch('storage.database.engine'), patch('storage.database.a_engine'):
    from integrations.github.github_view import get_user_proactive_conversation_setting
    from storage.org import Org

pytestmark = pytest.mark.asyncio


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


async def test_get_user_proactive_conversation_setting_user_not_found():
    """Test that False is returned when the user is not found."""
    with patch(
        'integrations.github.github_view.ENABLE_PROACTIVE_CONVERSATION_STARTERS',
        True,
    ):
        with patch(
            'storage.org_store.OrgStore.get_current_org_from_keycloak_user_id',
            return_value=None,
        ):
            assert (
                await get_user_proactive_conversation_setting(
                    '5594c7b6-f959-4b81-92e9-b09c206f5081'
                )
                is False
            )


async def test_get_user_proactive_conversation_setting_user_setting_none():
    """Test that False is returned when the user setting is None."""
    mock_org = MagicMock(spec=Org)
    mock_org.enable_proactive_conversation_starters = None

    with patch(
        'integrations.github.github_view.ENABLE_PROACTIVE_CONVERSATION_STARTERS',
        True,
    ):
        with patch(
            'storage.org_store.OrgStore.get_current_org_from_keycloak_user_id',
            return_value=mock_org,
        ):
            assert (
                await get_user_proactive_conversation_setting(
                    '5594c7b6-f959-4b81-92e9-b09c206f5081'
                )
                is False
            )


async def test_get_user_proactive_conversation_setting_user_setting_true():
    """Test that True is returned when the user setting is True and the global setting is True."""
    mock_org = MagicMock(spec=Org)
    mock_org.enable_proactive_conversation_starters = True

    with patch(
        'integrations.github.github_view.ENABLE_PROACTIVE_CONVERSATION_STARTERS',
        True,
    ):
        with patch(
            'storage.org_store.OrgStore.get_current_org_from_keycloak_user_id',
            return_value=mock_org,
        ):
            assert (
                await get_user_proactive_conversation_setting(
                    '5594c7b6-f959-4b81-92e9-b09c206f5081'
                )
                is True
            )


async def test_get_user_proactive_conversation_setting_user_setting_false():
    """Test that False is returned when the user setting is False, regardless of global setting."""
    mock_org = MagicMock(spec=Org)
    mock_org.enable_proactive_conversation_starters = False

    with patch(
        'integrations.github.github_view.ENABLE_PROACTIVE_CONVERSATION_STARTERS',
        True,
    ):
        with patch(
            'storage.org_store.OrgStore.get_current_org_from_keycloak_user_id',
            return_value=mock_org,
        ):
            assert (
                await get_user_proactive_conversation_setting(
                    '5594c7b6-f959-4b81-92e9-b09c206f5081'
                )
                is False
            )
