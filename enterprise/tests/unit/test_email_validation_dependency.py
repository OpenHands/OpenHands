"""
Unit tests for email validation dependency (get_admin_user_id).

Tests the FastAPI dependency that validates @openhands.dev email domain.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from server.email_validation import get_admin_user_id


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request."""
    return MagicMock(spec=Request)


@pytest.fixture
def mock_user_auth():
    """Create a mock user auth object."""
    mock_auth = AsyncMock()
    mock_auth.get_user_email = AsyncMock()
    return mock_auth


@pytest.mark.asyncio
async def test_get_openhands_user_id_success(mock_request, mock_user_auth):
    """
    GIVEN: Valid user ID and @openhands.dev email
    WHEN: get_admin_user_id is called
    THEN: User ID is returned successfully
    """
    # Arrange
    user_id = 'test-user-123'
    mock_user_auth.get_user_email.return_value = 'test@openhands.dev'

    with patch('server.email_validation.get_user_auth', return_value=mock_user_auth):
        # Act
        result = await get_admin_user_id(mock_request, user_id)

        # Assert
        assert result == user_id
        mock_user_auth.get_user_email.assert_called_once()


@pytest.mark.asyncio
async def test_get_openhands_user_id_no_user_id(mock_request):
    """
    GIVEN: No user ID provided (None)
    WHEN: get_admin_user_id is called
    THEN: 401 Unauthorized is raised
    """
    # Arrange
    user_id = None

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        await get_admin_user_id(mock_request, user_id)

    assert exc_info.value.status_code == 401
    assert 'not authenticated' in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_get_openhands_user_id_no_email(mock_request, mock_user_auth):
    """
    GIVEN: User ID provided but email is None
    WHEN: get_admin_user_id is called
    THEN: 401 Unauthorized is raised
    """
    # Arrange
    user_id = 'test-user-123'
    mock_user_auth.get_user_email.return_value = None

    with patch('server.email_validation.get_user_auth', return_value=mock_user_auth):
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_admin_user_id(mock_request, user_id)

        assert exc_info.value.status_code == 401
        assert 'email not available' in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_get_openhands_user_id_invalid_domain(mock_request, mock_user_auth):
    """
    GIVEN: User ID and email with non-@openhands.dev domain
    WHEN: get_admin_user_id is called
    THEN: 403 Forbidden is raised
    """
    # Arrange
    user_id = 'test-user-123'
    mock_user_auth.get_user_email.return_value = 'test@external.com'

    with patch('server.email_validation.get_user_auth', return_value=mock_user_auth):
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_admin_user_id(mock_request, user_id)

        assert exc_info.value.status_code == 403
        assert 'openhands.dev' in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_get_openhands_user_id_empty_string_user_id(mock_request):
    """
    GIVEN: Empty string user ID
    WHEN: get_admin_user_id is called
    THEN: 401 Unauthorized is raised
    """
    # Arrange
    user_id = ''

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        await get_admin_user_id(mock_request, user_id)

    assert exc_info.value.status_code == 401
    assert 'not authenticated' in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_get_openhands_user_id_case_sensitivity(mock_request, mock_user_auth):
    """
    GIVEN: Email with uppercase @OPENHANDS.DEV domain
    WHEN: get_admin_user_id is called
    THEN: 403 Forbidden is raised (case-sensitive check)
    """
    # Arrange
    user_id = 'test-user-123'
    mock_user_auth.get_user_email.return_value = 'test@OPENHANDS.DEV'

    with patch('server.email_validation.get_user_auth', return_value=mock_user_auth):
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_admin_user_id(mock_request, user_id)

        assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_get_openhands_user_id_subdomain_not_allowed(
    mock_request, mock_user_auth
):
    """
    GIVEN: Email with subdomain like @test.openhands.dev
    WHEN: get_admin_user_id is called
    THEN: 403 Forbidden is raised
    """
    # Arrange
    user_id = 'test-user-123'
    mock_user_auth.get_user_email.return_value = 'test@test.openhands.dev'

    with patch('server.email_validation.get_user_auth', return_value=mock_user_auth):
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_admin_user_id(mock_request, user_id)

        assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_get_openhands_user_id_similar_domain_not_allowed(
    mock_request, mock_user_auth
):
    """
    GIVEN: Email with similar but different domain like @openhands.dev.fake.com
    WHEN: get_admin_user_id is called
    THEN: 403 Forbidden is raised
    """
    # Arrange
    user_id = 'test-user-123'
    mock_user_auth.get_user_email.return_value = 'test@openhands.dev.fake.com'

    with patch('server.email_validation.get_user_auth', return_value=mock_user_auth):
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_admin_user_id(mock_request, user_id)

        assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_get_openhands_user_id_logs_warning_on_invalid_domain(
    mock_request, mock_user_auth
):
    """
    GIVEN: User with invalid email domain
    WHEN: get_admin_user_id is called
    THEN: Warning is logged with user_id and email_domain
    """
    # Arrange
    user_id = 'test-user-123'
    invalid_email = 'test@external.com'
    mock_user_auth.get_user_email.return_value = invalid_email

    with (
        patch('server.email_validation.get_user_auth', return_value=mock_user_auth),
        patch('server.email_validation.logger') as mock_logger,
    ):
        # Act & Assert
        with pytest.raises(HTTPException):
            await get_admin_user_id(mock_request, user_id)

        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert 'Access denied' in call_args[0][0]
        assert call_args[1]['extra']['user_id'] == user_id
        assert call_args[1]['extra']['email_domain'] == 'external.com'


@pytest.mark.asyncio
async def test_get_openhands_user_id_with_plus_addressing(mock_request, mock_user_auth):
    """
    GIVEN: Email with plus addressing (test+tag@openhands.dev)
    WHEN: get_admin_user_id is called
    THEN: User ID is returned successfully
    """
    # Arrange
    user_id = 'test-user-123'
    mock_user_auth.get_user_email.return_value = 'test+tag@openhands.dev'

    with patch('server.email_validation.get_user_auth', return_value=mock_user_auth):
        # Act
        result = await get_admin_user_id(mock_request, user_id)

        # Assert
        assert result == user_id


@pytest.mark.asyncio
async def test_get_openhands_user_id_with_dots_in_local_part(
    mock_request, mock_user_auth
):
    """
    GIVEN: Email with dots in local part (first.last@openhands.dev)
    WHEN: get_admin_user_id is called
    THEN: User ID is returned successfully
    """
    # Arrange
    user_id = 'test-user-123'
    mock_user_auth.get_user_email.return_value = 'first.last@openhands.dev'

    with patch('server.email_validation.get_user_auth', return_value=mock_user_auth):
        # Act
        result = await get_admin_user_id(mock_request, user_id)

        # Assert
        assert result == user_id


@pytest.mark.asyncio
async def test_get_openhands_user_id_empty_email(mock_request, mock_user_auth):
    """
    GIVEN: Empty string email
    WHEN: get_admin_user_id is called
    THEN: 401 Unauthorized is raised
    """
    # Arrange
    user_id = 'test-user-123'
    mock_user_auth.get_user_email.return_value = ''

    with patch('server.email_validation.get_user_auth', return_value=mock_user_auth):
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_admin_user_id(mock_request, user_id)

        assert exc_info.value.status_code == 401
        assert 'email not available' in exc_info.value.detail.lower()
