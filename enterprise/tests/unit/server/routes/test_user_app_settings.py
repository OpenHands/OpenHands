"""
Unit tests for user app settings API routes.

Tests the GET and POST /api/users/app endpoints.
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from server.routes.user_app_settings import user_app_settings_router
from server.routes.user_app_settings_models import (
    UserAppSettingsResponse,
    UserNotFoundError,
)

from openhands.server.user_auth import get_user_id

TEST_USER_ID = str(uuid.uuid4())


@pytest.fixture
def mock_app():
    """Create a test FastAPI app with user app settings routes and mocked auth."""
    app = FastAPI()
    app.include_router(user_app_settings_router)

    def mock_get_user_id():
        return TEST_USER_ID

    app.dependency_overrides[get_user_id] = mock_get_user_id

    return app


@pytest.fixture
def mock_app_unauthenticated():
    """Create a test FastAPI app with no authenticated user."""
    app = FastAPI()
    app.include_router(user_app_settings_router)

    def mock_get_user_id():
        return None

    app.dependency_overrides[get_user_id] = mock_get_user_id

    return app


@pytest.fixture
def mock_settings_response():
    """Create a mock user app settings response."""
    return UserAppSettingsResponse(
        language='en',
        user_consents_to_analytics=True,
        enable_sound_notifications=False,
        git_user_name='testuser',
        git_user_email='test@example.com',
    )


@pytest.mark.asyncio
async def test_get_user_app_settings_success(mock_app, mock_settings_response):
    """
    GIVEN: An authenticated user with app settings
    WHEN: GET /api/users/app is called
    THEN: User's app settings are returned with 200 status
    """
    # Arrange
    with patch(
        'server.routes.user_app_settings.UserAppSettingsService.get_user_app_settings',
        AsyncMock(return_value=mock_settings_response),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.get('/api/users/app')

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data['language'] == 'en'
        assert data['user_consents_to_analytics'] is True
        assert data['enable_sound_notifications'] is False
        assert data['git_user_name'] == 'testuser'
        assert data['git_user_email'] == 'test@example.com'


@pytest.mark.asyncio
async def test_get_user_app_settings_not_authenticated(mock_app_unauthenticated):
    """
    GIVEN: An unauthenticated request
    WHEN: GET /api/users/app is called
    THEN: 401 Unauthorized is returned
    """
    # Arrange
    client = TestClient(mock_app_unauthenticated)

    # Act
    response = client.get('/api/users/app')

    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert 'not authenticated' in response.json()['detail'].lower()


@pytest.mark.asyncio
async def test_get_user_app_settings_user_not_found(mock_app):
    """
    GIVEN: An authenticated user that doesn't exist in the database
    WHEN: GET /api/users/app is called
    THEN: 404 Not Found is returned
    """
    # Arrange
    with patch(
        'server.routes.user_app_settings.UserAppSettingsService.get_user_app_settings',
        AsyncMock(side_effect=UserNotFoundError(TEST_USER_ID)),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.get('/api/users/app')

        # Assert
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert 'not found' in response.json()['detail'].lower()


@pytest.mark.asyncio
async def test_update_user_app_settings_success(mock_app):
    """
    GIVEN: An authenticated user
    WHEN: POST /api/users/app is called with update data
    THEN: Updated settings are returned with 200 status
    """
    # Arrange
    updated_response = UserAppSettingsResponse(
        language='es',
        user_consents_to_analytics=False,
        enable_sound_notifications=True,
        git_user_name='newuser',
        git_user_email='new@example.com',
    )
    request_data = {
        'language': 'es',
        'user_consents_to_analytics': False,
    }

    with patch(
        'server.routes.user_app_settings.UserAppSettingsService.update_user_app_settings',
        AsyncMock(return_value=updated_response),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.post('/api/users/app', json=request_data)

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data['language'] == 'es'
        assert data['user_consents_to_analytics'] is False


@pytest.mark.asyncio
async def test_update_user_app_settings_not_authenticated(mock_app_unauthenticated):
    """
    GIVEN: An unauthenticated request
    WHEN: POST /api/users/app is called
    THEN: 401 Unauthorized is returned
    """
    # Arrange
    request_data = {'language': 'en'}
    client = TestClient(mock_app_unauthenticated)

    # Act
    response = client.post('/api/users/app', json=request_data)

    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert 'not authenticated' in response.json()['detail'].lower()


@pytest.mark.asyncio
async def test_update_user_app_settings_user_not_found(mock_app):
    """
    GIVEN: An authenticated user that doesn't exist in the database
    WHEN: POST /api/users/app is called
    THEN: 404 Not Found is returned
    """
    # Arrange
    request_data = {'language': 'en'}

    with patch(
        'server.routes.user_app_settings.UserAppSettingsService.update_user_app_settings',
        AsyncMock(side_effect=UserNotFoundError(TEST_USER_ID)),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.post('/api/users/app', json=request_data)

        # Assert
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert 'not found' in response.json()['detail'].lower()
