"""
Integration tests for organization API routes.

Tests the POST /api/organizations endpoint with various scenarios.
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient

# Mock database before imports
with patch('storage.database.engine', create=True), patch(
    'storage.database.a_engine', create=True
):
    from server.email_validation import get_admin_user_id
    from server.routes.org_models import (
        LiteLLMIntegrationError,
        OrgDatabaseError,
        OrgNameExistsError,
    )
    from server.routes.orgs import org_router
    from storage.org import Org


@pytest.fixture
def mock_app():
    """Create a test FastAPI app with organization routes and mocked auth."""
    app = FastAPI()
    app.include_router(org_router)

    # Override the auth dependency to return a test user
    def mock_get_openhands_user_id():
        return 'test-user-123'

    app.dependency_overrides[get_admin_user_id] = mock_get_openhands_user_id

    return app


@pytest.mark.asyncio
async def test_create_org_success(mock_app):
    """
    GIVEN: Valid organization creation request
    WHEN: POST /api/organizations is called
    THEN: Organization is created and returned with 201 status
    """
    # Arrange
    org_id = uuid.uuid4()
    mock_org = Org(
        id=org_id,
        name='Test Organization',
        contact_name='John Doe',
        contact_email='john@example.com',
        org_version=5,
        default_llm_model='claude-opus-4-5-20251101',
        enable_default_condenser=True,
        enable_proactive_conversation_starters=True,
    )

    request_data = {
        'name': 'Test Organization',
        'contact_name': 'John Doe',
        'contact_email': 'john@example.com',
    }

    with (
        patch(
            'server.routes.orgs.OrgService.create_org_with_owner',
            AsyncMock(return_value=mock_org),
        ),
        patch(
            'server.routes.orgs.OrgService.get_org_credits',
            AsyncMock(return_value=100.0),
        ),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.post('/api/organizations', json=request_data)

        # Assert
        assert response.status_code == status.HTTP_201_CREATED
        response_data = response.json()
        assert response_data['name'] == 'Test Organization'
        assert response_data['contact_name'] == 'John Doe'
        assert response_data['contact_email'] == 'john@example.com'
        assert response_data['credits'] == 100.0
        assert response_data['org_version'] == 5
        assert response_data['default_llm_model'] == 'claude-opus-4-5-20251101'


@pytest.mark.asyncio
async def test_create_org_invalid_email(mock_app):
    """
    GIVEN: Request with invalid email format
    WHEN: POST /api/organizations is called
    THEN: 422 validation error is returned
    """
    # Arrange
    request_data = {
        'name': 'Test Organization',
        'contact_name': 'John Doe',
        'contact_email': 'invalid-email',  # Missing @
    }

    client = TestClient(mock_app)

    # Act
    response = client.post('/api/organizations', json=request_data)

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_create_org_empty_name(mock_app):
    """
    GIVEN: Request with empty organization name
    WHEN: POST /api/organizations is called
    THEN: 422 validation error is returned
    """
    # Arrange
    request_data = {
        'name': '',  # Empty string (after whitespace stripping)
        'contact_name': 'John Doe',
        'contact_email': 'john@example.com',
    }

    client = TestClient(mock_app)

    # Act
    response = client.post('/api/organizations', json=request_data)

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_create_org_duplicate_name(mock_app):
    """
    GIVEN: Organization name already exists
    WHEN: POST /api/organizations is called
    THEN: 409 Conflict error is returned
    """
    # Arrange
    request_data = {
        'name': 'Existing Organization',
        'contact_name': 'John Doe',
        'contact_email': 'john@example.com',
    }

    with patch(
        'server.routes.orgs.OrgService.create_org_with_owner',
        AsyncMock(side_effect=OrgNameExistsError('Existing Organization')),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.post('/api/organizations', json=request_data)

        # Assert
        assert response.status_code == status.HTTP_409_CONFLICT
        assert 'already exists' in response.json()['detail'].lower()


@pytest.mark.asyncio
async def test_create_org_litellm_failure(mock_app):
    """
    GIVEN: LiteLLM integration fails
    WHEN: POST /api/organizations is called
    THEN: 500 Internal Server Error is returned
    """
    # Arrange
    request_data = {
        'name': 'Test Organization',
        'contact_name': 'John Doe',
        'contact_email': 'john@example.com',
    }

    with patch(
        'server.routes.orgs.OrgService.create_org_with_owner',
        AsyncMock(side_effect=LiteLLMIntegrationError('LiteLLM API unavailable')),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.post('/api/organizations', json=request_data)

        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert 'LiteLLM integration' in response.json()['detail']


@pytest.mark.asyncio
async def test_create_org_database_failure(mock_app):
    """
    GIVEN: Database operation fails
    WHEN: POST /api/organizations is called
    THEN: 500 Internal Server Error is returned
    """
    # Arrange
    request_data = {
        'name': 'Test Organization',
        'contact_name': 'John Doe',
        'contact_email': 'john@example.com',
    }

    with patch(
        'server.routes.orgs.OrgService.create_org_with_owner',
        AsyncMock(side_effect=OrgDatabaseError('Database connection failed')),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.post('/api/organizations', json=request_data)

        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert 'Failed to create organization' in response.json()['detail']


@pytest.mark.asyncio
async def test_create_org_unexpected_error(mock_app):
    """
    GIVEN: Unexpected error occurs
    WHEN: POST /api/organizations is called
    THEN: 500 Internal Server Error is returned with generic message
    """
    # Arrange
    request_data = {
        'name': 'Test Organization',
        'contact_name': 'John Doe',
        'contact_email': 'john@example.com',
    }

    with patch(
        'server.routes.orgs.OrgService.create_org_with_owner',
        AsyncMock(side_effect=RuntimeError('Unexpected system error')),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.post('/api/organizations', json=request_data)

        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert 'unexpected error' in response.json()['detail'].lower()


@pytest.mark.asyncio
async def test_create_org_unauthorized():
    """
    GIVEN: User is not authenticated
    WHEN: POST /api/organizations is called
    THEN: 401 Unauthorized error is returned
    """
    # Arrange
    app = FastAPI()
    app.include_router(org_router)

    # Override to simulate unauthenticated user
    async def mock_unauthenticated():
        raise HTTPException(status_code=401, detail='User not authenticated')

    app.dependency_overrides[get_admin_user_id] = mock_unauthenticated

    request_data = {
        'name': 'Test Organization',
        'contact_name': 'John Doe',
        'contact_email': 'john@example.com',
    }

    client = TestClient(app)

    # Act
    response = client.post('/api/organizations', json=request_data)

    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_create_org_forbidden_non_openhands_email():
    """
    GIVEN: User email is not @openhands.dev
    WHEN: POST /api/organizations is called
    THEN: 403 Forbidden error is returned
    """
    # Arrange
    app = FastAPI()
    app.include_router(org_router)

    # Override to simulate non-@openhands.dev user
    async def mock_forbidden():
        raise HTTPException(
            status_code=403, detail='Access restricted to @openhands.dev users'
        )

    app.dependency_overrides[get_admin_user_id] = mock_forbidden

    request_data = {
        'name': 'Test Organization',
        'contact_name': 'John Doe',
        'contact_email': 'john@example.com',
    }

    client = TestClient(app)

    # Act
    response = client.post('/api/organizations', json=request_data)

    # Assert
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert 'openhands.dev' in response.json()['detail'].lower()


@pytest.mark.asyncio
async def test_create_org_sensitive_fields_not_exposed(mock_app):
    """
    GIVEN: Organization is created successfully
    WHEN: Response is returned
    THEN: Sensitive fields (API keys) are not exposed
    """
    # Arrange
    org_id = uuid.uuid4()
    mock_org = Org(
        id=org_id,
        name='Test Organization',
        contact_name='John Doe',
        contact_email='john@example.com',
        org_version=5,
        default_llm_model='claude-opus-4-5-20251101',
        enable_default_condenser=True,
        enable_proactive_conversation_starters=True,
    )

    request_data = {
        'name': 'Test Organization',
        'contact_name': 'John Doe',
        'contact_email': 'john@example.com',
    }

    with (
        patch(
            'server.routes.orgs.OrgService.create_org_with_owner',
            AsyncMock(return_value=mock_org),
        ),
        patch(
            'server.routes.orgs.OrgService.get_org_credits',
            AsyncMock(return_value=100.0),
        ),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.post('/api/organizations', json=request_data)

        # Assert
        assert response.status_code == status.HTTP_201_CREATED
        response_data = response.json()

        # Verify sensitive fields are not in response or are None
        assert (
            'default_llm_api_key_for_byor' not in response_data
            or response_data.get('default_llm_api_key_for_byor') is None
        )
        assert (
            'search_api_key' not in response_data
            or response_data.get('search_api_key') is None
        )
        assert (
            'sandbox_api_key' not in response_data
            or response_data.get('sandbox_api_key') is None
        )
