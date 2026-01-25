"""
Integration tests for organization API routes.

Tests the POST /api/organizations endpoint with various scenarios.
"""

import uuid
from unittest.mock import AsyncMock, patch

import httpx
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
        OrgAuthorizationError,
        OrgDatabaseError,
        OrgNameExistsError,
        OrgNotFoundError,
    )
    from server.routes.orgs import org_router
    from storage.org import Org

    from openhands.server.user_auth import get_user_id


@pytest.fixture
def mock_app():
    """Create a test FastAPI app with organization routes and mocked auth."""
    app = FastAPI()
    app.include_router(org_router)

    # Override the auth dependency to return a test user
    def mock_get_admin_user_id():
        return 'test-user-123'

    app.dependency_overrides[get_admin_user_id] = mock_get_admin_user_id

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


@pytest.fixture
def mock_app_list():
    """Create a test FastAPI app with organization routes and mocked auth for list endpoint."""
    app = FastAPI()
    app.include_router(org_router)

    # Override the auth dependency to return a test user
    test_user_id = str(uuid.uuid4())

    def mock_get_user_id():
        return test_user_id

    app.dependency_overrides[get_user_id] = mock_get_user_id

    return app


@pytest.mark.asyncio
async def test_list_user_orgs_success(mock_app_list):
    """
    GIVEN: User has organizations
    WHEN: GET /api/organizations is called
    THEN: Paginated list of organizations is returned with 200 status
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
    )

    with patch(
        'server.routes.orgs.OrgService.get_user_orgs_paginated',
        return_value=([mock_org], None),
    ):
        client = TestClient(mock_app_list)

        # Act
        response = client.get('/api/organizations')

        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert 'items' in response_data
        assert 'next_page_id' in response_data
        assert len(response_data['items']) == 1
        assert response_data['items'][0]['name'] == 'Test Organization'
        assert response_data['items'][0]['id'] == str(org_id)
        assert response_data['next_page_id'] is None
        # Credits should be None in list view
        assert response_data['items'][0]['credits'] is None


@pytest.mark.asyncio
async def test_list_user_orgs_with_pagination(mock_app_list):
    """
    GIVEN: User has multiple organizations
    WHEN: GET /api/organizations is called with pagination params
    THEN: Paginated results are returned with next_page_id
    """
    # Arrange
    org1 = Org(
        id=uuid.uuid4(),
        name='Alpha Org',
        contact_name='John Doe',
        contact_email='john@example.com',
    )
    org2 = Org(
        id=uuid.uuid4(),
        name='Beta Org',
        contact_name='Jane Doe',
        contact_email='jane@example.com',
    )

    with patch(
        'server.routes.orgs.OrgService.get_user_orgs_paginated',
        return_value=([org1, org2], '2'),
    ):
        client = TestClient(mock_app_list)

        # Act
        response = client.get('/api/organizations?page_id=0&limit=2')

        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert len(response_data['items']) == 2
        assert response_data['next_page_id'] == '2'
        assert response_data['items'][0]['name'] == 'Alpha Org'
        assert response_data['items'][1]['name'] == 'Beta Org'


@pytest.mark.asyncio
async def test_list_user_orgs_empty(mock_app_list):
    """
    GIVEN: User has no organizations
    WHEN: GET /api/organizations is called
    THEN: Empty list is returned with 200 status
    """
    # Arrange
    with patch(
        'server.routes.orgs.OrgService.get_user_orgs_paginated',
        return_value=([], None),
    ):
        client = TestClient(mock_app_list)

        # Act
        response = client.get('/api/organizations')

        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data['items'] == []
        assert response_data['next_page_id'] is None


@pytest.mark.asyncio
async def test_list_user_orgs_invalid_limit_negative(mock_app_list):
    """
    GIVEN: Invalid limit parameter (negative)
    WHEN: GET /api/organizations is called
    THEN: 422 validation error is returned
    """
    # Arrange
    client = TestClient(mock_app_list)

    # Act - FastAPI should validate and reject limit <= 0
    response = client.get('/api/organizations?limit=-1')

    # Assert - FastAPI should return 422 for validation error
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_list_user_orgs_invalid_limit_zero(mock_app_list):
    """
    GIVEN: Invalid limit parameter (zero or negative)
    WHEN: GET /api/organizations is called
    THEN: 422 validation error is returned
    """
    # Arrange
    client = TestClient(mock_app_list)

    # Act
    response = client.get('/api/organizations?limit=0')

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_list_user_orgs_service_error(mock_app_list):
    """
    GIVEN: Service layer raises an exception
    WHEN: GET /api/organizations is called
    THEN: 500 Internal Server Error is returned
    """
    # Arrange
    with patch(
        'server.routes.orgs.OrgService.get_user_orgs_paginated',
        side_effect=Exception('Database error'),
    ):
        client = TestClient(mock_app_list)

        # Act
        response = client.get('/api/organizations')

        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert 'Failed to retrieve organizations' in response.json()['detail']


@pytest.mark.asyncio
async def test_list_user_orgs_unauthorized():
    """
    GIVEN: User is not authenticated
    WHEN: GET /api/organizations is called
    THEN: 401 Unauthorized error is returned
    """
    # Arrange
    app = FastAPI()
    app.include_router(org_router)

    # Override to simulate unauthenticated user
    async def mock_unauthenticated():
        raise HTTPException(status_code=401, detail='User not authenticated')

    app.dependency_overrides[get_user_id] = mock_unauthenticated

    client = TestClient(app)

    # Act
    response = client.get('/api/organizations')

    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_list_user_orgs_all_fields_present(mock_app_list):
    """
    GIVEN: Organization with all fields populated
    WHEN: GET /api/organizations is called
    THEN: All organization fields are included in response
    """
    # Arrange
    org_id = uuid.uuid4()
    mock_org = Org(
        id=org_id,
        name='Complete Org',
        contact_name='John Doe',
        contact_email='john@example.com',
        conversation_expiration=3600,
        agent='CodeActAgent',
        default_max_iterations=50,
        security_analyzer='enabled',
        confirmation_mode=True,
        default_llm_model='claude-opus-4-5-20251101',
        default_llm_base_url='https://api.example.com',
        remote_runtime_resource_factor=2,
        enable_default_condenser=True,
        billing_margin=0.15,
        enable_proactive_conversation_starters=True,
        sandbox_base_container_image='test-image',
        sandbox_runtime_container_image='test-runtime',
        org_version=5,
        mcp_config={'key': 'value'},
        max_budget_per_task=1000.0,
        enable_solvability_analysis=True,
        v1_enabled=True,
    )

    with patch(
        'server.routes.orgs.OrgService.get_user_orgs_paginated',
        return_value=([mock_org], None),
    ):
        client = TestClient(mock_app_list)

        # Act
        response = client.get('/api/organizations')

        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        org_data = response_data['items'][0]
        assert org_data['name'] == 'Complete Org'
        assert org_data['contact_name'] == 'John Doe'
        assert org_data['contact_email'] == 'john@example.com'
        assert org_data['conversation_expiration'] == 3600
        assert org_data['agent'] == 'CodeActAgent'
        assert org_data['default_max_iterations'] == 50
        assert org_data['security_analyzer'] == 'enabled'
        assert org_data['confirmation_mode'] is True
        assert org_data['default_llm_model'] == 'claude-opus-4-5-20251101'
        assert org_data['default_llm_base_url'] == 'https://api.example.com'
        assert org_data['remote_runtime_resource_factor'] == 2
        assert org_data['enable_default_condenser'] is True
        assert org_data['billing_margin'] == 0.15
        assert org_data['enable_proactive_conversation_starters'] is True
        assert org_data['sandbox_base_container_image'] == 'test-image'
        assert org_data['sandbox_runtime_container_image'] == 'test-runtime'
        assert org_data['org_version'] == 5
        assert org_data['mcp_config'] == {'key': 'value'}
        assert org_data['max_budget_per_task'] == 1000.0
        assert org_data['enable_solvability_analysis'] is True
        assert org_data['v1_enabled'] is True
        assert org_data['credits'] is None


@pytest.fixture
def mock_app_with_get_user_id():
    """Create a test FastAPI app with organization routes and mocked get_user_id auth."""
    app = FastAPI()
    app.include_router(org_router)

    # Override the auth dependency to return a test user
    def mock_get_user_id():
        return 'test-user-123'

    app.dependency_overrides[get_user_id] = mock_get_user_id

    return app


@pytest.mark.asyncio
async def test_get_org_success(mock_app_with_get_user_id):
    """
    GIVEN: Valid org_id and authenticated user who is a member
    WHEN: GET /api/organizations/{org_id} is called
    THEN: Organization details are returned with 200 status
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

    with (
        patch(
            'server.routes.orgs.OrgService.get_org_by_id',
            AsyncMock(return_value=mock_org),
        ),
        patch(
            'server.routes.orgs.OrgService.get_org_credits',
            AsyncMock(return_value=75.5),
        ),
    ):
        client = TestClient(mock_app_with_get_user_id)

        # Act
        response = client.get(f'/api/organizations/{org_id}')

        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data['id'] == str(org_id)
        assert response_data['name'] == 'Test Organization'
        assert response_data['contact_name'] == 'John Doe'
        assert response_data['contact_email'] == 'john@example.com'
        assert response_data['credits'] == 75.5
        assert response_data['org_version'] == 5


@pytest.mark.asyncio
async def test_get_org_user_not_member(mock_app_with_get_user_id):
    """
    GIVEN: User is not a member of the organization
    WHEN: GET /api/organizations/{org_id} is called
    THEN: 404 Not Found error is returned
    """
    # Arrange
    org_id = uuid.uuid4()

    with patch(
        'server.routes.orgs.OrgService.get_org_by_id',
        AsyncMock(side_effect=OrgNotFoundError(str(org_id))),
    ):
        client = TestClient(mock_app_with_get_user_id)

        # Act
        response = client.get(f'/api/organizations/{org_id}')

        # Assert
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert 'not found' in response.json()['detail'].lower()


@pytest.mark.asyncio
async def test_get_org_not_found(mock_app_with_get_user_id):
    """
    GIVEN: Organization does not exist
    WHEN: GET /api/organizations/{org_id} is called
    THEN: 404 Not Found error is returned
    """
    # Arrange
    org_id = uuid.uuid4()

    with patch(
        'server.routes.orgs.OrgService.get_org_by_id',
        AsyncMock(side_effect=OrgNotFoundError(str(org_id))),
    ):
        client = TestClient(mock_app_with_get_user_id)

        # Act
        response = client.get(f'/api/organizations/{org_id}')

        # Assert
        assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_get_org_invalid_uuid(mock_app_with_get_user_id):
    """
    GIVEN: Invalid UUID format for org_id
    WHEN: GET /api/organizations/{org_id} is called
    THEN: 422 Unprocessable Entity error is returned
    """
    # Arrange
    invalid_org_id = 'not-a-valid-uuid'

    client = TestClient(mock_app_with_get_user_id)

    # Act
    response = client.get(f'/api/organizations/{invalid_org_id}')

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_get_org_unauthorized():
    """
    GIVEN: User is not authenticated
    WHEN: GET /api/organizations/{org_id} is called
    THEN: 401 Unauthorized error is returned
    """
    # Arrange
    app = FastAPI()
    app.include_router(org_router)

    # Override to simulate unauthenticated user
    async def mock_unauthenticated():
        raise HTTPException(status_code=401, detail='User not authenticated')

    app.dependency_overrides[get_user_id] = mock_unauthenticated

    org_id = uuid.uuid4()
    client = TestClient(app)

    # Act
    response = client.get(f'/api/organizations/{org_id}')

    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_get_org_unexpected_error(mock_app_with_get_user_id):
    """
    GIVEN: Unexpected error occurs during retrieval
    WHEN: GET /api/organizations/{org_id} is called
    THEN: 500 Internal Server Error is returned
    """
    # Arrange
    org_id = uuid.uuid4()

    with patch(
        'server.routes.orgs.OrgService.get_org_by_id',
        AsyncMock(side_effect=RuntimeError('Unexpected database error')),
    ):
        client = TestClient(mock_app_with_get_user_id)

        # Act
        response = client.get(f'/api/organizations/{org_id}')

        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert 'unexpected error' in response.json()['detail'].lower()


@pytest.mark.asyncio
async def test_get_org_with_credits_none(mock_app_with_get_user_id):
    """
    GIVEN: Organization exists but credits retrieval returns None
    WHEN: GET /api/organizations/{org_id} is called
    THEN: Organization is returned with credits as None
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

    with (
        patch(
            'server.routes.orgs.OrgService.get_org_by_id',
            AsyncMock(return_value=mock_org),
        ),
        patch(
            'server.routes.orgs.OrgService.get_org_credits',
            AsyncMock(return_value=None),
        ),
    ):
        client = TestClient(mock_app_with_get_user_id)

        # Act
        response = client.get(f'/api/organizations/{org_id}')

        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data['credits'] is None


@pytest.mark.asyncio
async def test_get_org_sensitive_fields_not_exposed(mock_app_with_get_user_id):
    """
    GIVEN: Organization is retrieved successfully
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
        search_api_key='secret-search-key-123',  # Should not be exposed
        sandbox_api_key='secret-sandbox-key-123',  # Should not be exposed
        enable_default_condenser=True,
        enable_proactive_conversation_starters=True,
    )

    with (
        patch(
            'server.routes.orgs.OrgService.get_org_by_id',
            AsyncMock(return_value=mock_org),
        ),
        patch(
            'server.routes.orgs.OrgService.get_org_credits',
            AsyncMock(return_value=100.0),
        ),
    ):
        client = TestClient(mock_app_with_get_user_id)

        # Act
        response = client.get(f'/api/organizations/{org_id}')

        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()

        # Verify sensitive fields are not in response or are None
        assert (
            'search_api_key' not in response_data
            or response_data.get('search_api_key') is None
        )
        assert (
            'sandbox_api_key' not in response_data
            or response_data.get('sandbox_api_key') is None
        )


@pytest.mark.asyncio
async def test_delete_org_success(mock_app):
    """
    GIVEN: Valid organization deletion request by owner
    WHEN: DELETE /api/organizations/{org_id} is called
    THEN: Organization is deleted and 200 status with confirmation is returned
    """
    # Arrange
    org_id = uuid.uuid4()
    mock_deleted_org = Org(
        id=org_id,
        name='Deleted Organization',
        contact_name='John Doe',
        contact_email='john@example.com',
    )

    with patch(
        'server.routes.orgs.OrgService.delete_org_with_cleanup',
        AsyncMock(return_value=mock_deleted_org),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.delete(f'/api/organizations/{org_id}')

        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data['message'] == 'Organization deleted successfully'
        assert response_data['organization']['id'] == str(org_id)
        assert response_data['organization']['name'] == 'Deleted Organization'
        assert response_data['organization']['contact_name'] == 'John Doe'
        assert response_data['organization']['contact_email'] == 'john@example.com'


@pytest.mark.asyncio
async def test_delete_org_not_found(mock_app):
    """
    GIVEN: Organization does not exist
    WHEN: DELETE /api/organizations/{org_id} is called
    THEN: 404 Not Found error is returned
    """
    # Arrange
    org_id = uuid.uuid4()

    with patch(
        'server.routes.orgs.OrgService.delete_org_with_cleanup',
        AsyncMock(side_effect=OrgNotFoundError(str(org_id))),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.delete(f'/api/organizations/{org_id}')

        # Assert
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert str(org_id) in response.json()['detail']


@pytest.mark.asyncio
async def test_delete_org_not_owner(mock_app):
    """
    GIVEN: User is not the organization owner
    WHEN: DELETE /api/organizations/{org_id} is called
    THEN: 403 Forbidden error is returned
    """
    # Arrange
    org_id = uuid.uuid4()

    with patch(
        'server.routes.orgs.OrgService.delete_org_with_cleanup',
        AsyncMock(
            side_effect=OrgAuthorizationError(
                'Only organization owners can delete organizations'
            )
        ),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.delete(f'/api/organizations/{org_id}')

        # Assert
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert 'organization owners' in response.json()['detail']


@pytest.mark.asyncio
async def test_delete_org_not_member(mock_app):
    """
    GIVEN: User is not a member of the organization
    WHEN: DELETE /api/organizations/{org_id} is called
    THEN: 403 Forbidden error is returned
    """
    # Arrange
    org_id = uuid.uuid4()

    with patch(
        'server.routes.orgs.OrgService.delete_org_with_cleanup',
        AsyncMock(
            side_effect=OrgAuthorizationError(
                'User is not a member of this organization'
            )
        ),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.delete(f'/api/organizations/{org_id}')

        # Assert
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert 'not a member' in response.json()['detail']


@pytest.mark.asyncio
async def test_delete_org_database_failure(mock_app):
    """
    GIVEN: Database operation fails during deletion
    WHEN: DELETE /api/organizations/{org_id} is called
    THEN: 500 Internal Server Error is returned
    """
    # Arrange
    org_id = uuid.uuid4()

    with patch(
        'server.routes.orgs.OrgService.delete_org_with_cleanup',
        AsyncMock(side_effect=OrgDatabaseError('Database connection failed')),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.delete(f'/api/organizations/{org_id}')

        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert response.json()['detail'] == 'Failed to delete organization'


@pytest.mark.asyncio
async def test_delete_org_unexpected_error(mock_app):
    """
    GIVEN: Unexpected error occurs during deletion
    WHEN: DELETE /api/organizations/{org_id} is called
    THEN: 500 Internal Server Error is returned with generic message
    """
    # Arrange
    org_id = uuid.uuid4()

    with patch(
        'server.routes.orgs.OrgService.delete_org_with_cleanup',
        AsyncMock(side_effect=RuntimeError('Unexpected system error')),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.delete(f'/api/organizations/{org_id}')

        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert 'unexpected error' in response.json()['detail'].lower()


@pytest.mark.asyncio
async def test_delete_org_invalid_uuid(mock_app):
    """
    GIVEN: Invalid UUID format in URL
    WHEN: DELETE /api/organizations/{invalid_uuid} is called
    THEN: 422 validation error is returned
    """
    # Arrange
    invalid_uuid = 'not-a-valid-uuid'
    client = TestClient(mock_app)

    # Act
    response = client.delete(f'/api/organizations/{invalid_uuid}')

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_delete_org_unauthorized():
    """
    GIVEN: User is not authenticated
    WHEN: DELETE /api/organizations/{org_id} is called
    THEN: 401 Unauthorized error is returned
    """
    # Arrange
    app = FastAPI()
    app.include_router(org_router)

    # Override to simulate unauthenticated user
    async def mock_unauthenticated():
        raise HTTPException(status_code=401, detail='User not authenticated')

    app.dependency_overrides[get_admin_user_id] = mock_unauthenticated

    org_id = uuid.uuid4()
    client = TestClient(app)

    # Act
    response = client.delete(f'/api/organizations/{org_id}')

    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.fixture
def mock_update_app():
    """Create a test FastAPI app with organization routes and mocked auth for update endpoint."""
    app = FastAPI()
    app.include_router(org_router)

    # Override the auth dependency to return a test user
    async def mock_user_id():
        return 'test-user-123'

    app.dependency_overrides[get_user_id] = mock_user_id

    return app


# Note: Success cases for update endpoint are tested in test_org_service.py
# Route handler tests focus on error handling and validation


@pytest.mark.asyncio
async def test_update_org_not_found(mock_update_app):
    """
    GIVEN: Organization ID does not exist
    WHEN: PATCH /api/organizations/{org_id} is called
    THEN: 404 Not Found error is returned
    """
    # Arrange
    org_id = uuid.uuid4()
    update_data = {'contact_name': 'Jane Doe'}

    with patch(
        'server.routes.orgs.OrgService.update_org_with_permissions',
        AsyncMock(side_effect=ValueError(f'Organization with ID {org_id} not found')),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_update_app), base_url='http://test'
        ) as client:
            # Act
            response = await client.patch(
                f'/api/organizations/{org_id}', json=update_data
            )

            # Assert
            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert 'not found' in response.json()['detail'].lower()


@pytest.mark.asyncio
async def test_update_org_permission_denied_non_member(mock_update_app):
    """
    GIVEN: User is not a member of the organization
    WHEN: PATCH /api/organizations/{org_id} is called
    THEN: 403 Forbidden error is returned
    """
    # Arrange
    org_id = uuid.uuid4()
    update_data = {'contact_name': 'Jane Doe'}

    with patch(
        'server.routes.orgs.OrgService.update_org_with_permissions',
        AsyncMock(
            side_effect=PermissionError(
                'User must be a member of the organization to update it'
            )
        ),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_update_app), base_url='http://test'
        ) as client:
            # Act
            response = await client.patch(
                f'/api/organizations/{org_id}', json=update_data
            )

            # Assert
            assert response.status_code == status.HTTP_403_FORBIDDEN
            assert 'member' in response.json()['detail'].lower()


@pytest.mark.asyncio
async def test_update_org_permission_denied_llm_settings(mock_update_app):
    """
    GIVEN: User lacks admin/owner role but tries to update LLM settings
    WHEN: PATCH /api/organizations/{org_id} is called
    THEN: 403 Forbidden error is returned
    """
    # Arrange
    org_id = uuid.uuid4()
    update_data = {'default_llm_model': 'claude-opus-4-5-20251101'}

    with patch(
        'server.routes.orgs.OrgService.update_org_with_permissions',
        AsyncMock(
            side_effect=PermissionError(
                'Admin or owner role required to update LLM settings'
            )
        ),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_update_app), base_url='http://test'
        ) as client:
            # Act
            response = await client.patch(
                f'/api/organizations/{org_id}', json=update_data
            )

            # Assert
            assert response.status_code == status.HTTP_403_FORBIDDEN
            assert (
                'admin' in response.json()['detail'].lower()
                or 'owner' in response.json()['detail'].lower()
            )


@pytest.mark.asyncio
async def test_update_org_database_error(mock_update_app):
    """
    GIVEN: Database operation fails during update
    WHEN: PATCH /api/organizations/{org_id} is called
    THEN: 500 Internal Server Error is returned
    """
    # Arrange
    org_id = uuid.uuid4()
    update_data = {'contact_name': 'Jane Doe'}

    with patch(
        'server.routes.orgs.OrgService.update_org_with_permissions',
        AsyncMock(side_effect=OrgDatabaseError('Database connection failed')),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_update_app), base_url='http://test'
        ) as client:
            # Act
            response = await client.patch(
                f'/api/organizations/{org_id}', json=update_data
            )

            # Assert
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert 'Failed to update organization' in response.json()['detail']


@pytest.mark.asyncio
async def test_update_org_unexpected_error(mock_update_app):
    """
    GIVEN: Unexpected error occurs during update
    WHEN: PATCH /api/organizations/{org_id} is called
    THEN: 500 Internal Server Error is returned with generic message
    """
    # Arrange
    org_id = uuid.uuid4()
    update_data = {'contact_name': 'Jane Doe'}

    with patch(
        'server.routes.orgs.OrgService.update_org_with_permissions',
        AsyncMock(side_effect=RuntimeError('Unexpected system error')),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_update_app), base_url='http://test'
        ) as client:
            # Act
            response = await client.patch(
                f'/api/organizations/{org_id}', json=update_data
            )

            # Assert
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert 'unexpected error' in response.json()['detail'].lower()


@pytest.mark.asyncio
async def test_update_org_invalid_uuid_format(mock_update_app):
    """
    GIVEN: Invalid UUID format in org_id path parameter
    WHEN: PATCH /api/organizations/{org_id} is called
    THEN: 422 validation error is returned (handled by FastAPI)
    """
    # Arrange
    invalid_org_id = 'not-a-valid-uuid'
    update_data = {'contact_name': 'Jane Doe'}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_update_app), base_url='http://test'
    ) as client:
        # Act
        response = await client.patch(
            f'/api/organizations/{invalid_org_id}', json=update_data
        )

        # Assert
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_update_org_invalid_field_values(mock_update_app):
    """
    GIVEN: Update request with invalid field values (e.g., negative max_iterations)
    WHEN: PATCH /api/organizations/{org_id} is called
    THEN: 422 validation error is returned
    """
    # Arrange
    org_id = uuid.uuid4()
    update_data = {'default_max_iterations': -1}  # Invalid: must be > 0

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_update_app), base_url='http://test'
    ) as client:
        # Act
        response = await client.patch(f'/api/organizations/{org_id}', json=update_data)

        # Assert
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_update_org_invalid_email_format(mock_update_app):
    """
    GIVEN: Update request with invalid email format
    WHEN: PATCH /api/organizations/{org_id} is called
    THEN: 422 validation error is returned
    """
    # Arrange
    org_id = uuid.uuid4()
    update_data = {'contact_email': 'invalid-email'}  # Missing @

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_update_app), base_url='http://test'
    ) as client:
        # Act
        response = await client.patch(f'/api/organizations/{org_id}', json=update_data)

        # Assert
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
