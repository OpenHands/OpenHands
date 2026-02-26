"""
Integration tests for organization API routes.

Tests the POST /api/organizations endpoint with various scenarios.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.testclient import TestClient

# Mock database before imports
with patch('storage.database.engine', create=True), patch(
    'storage.database.a_engine', create=True
):
    from server.email_validation import get_admin_user_id
    from server.routes.org_models import (
        CannotModifySelfError,
        InsufficientPermissionError,
        InvalidRoleError,
        LastOwnerError,
        LiteLLMIntegrationError,
        MeResponse,
        OrgAuthorizationError,
        OrgDatabaseError,
        OrgMemberNotFoundError,
        OrgMemberPage,
        OrgMemberResponse,
        OrgMemberUpdate,
        OrgNameExistsError,
        OrgNotFoundError,
        OrphanedUserError,
        RoleNotFoundError,
    )
    from server.routes.orgs import (
        get_me,
        get_org_members,
        org_router,
        remove_org_member,
        update_org_member,
    )
    from storage.org import Org

    from openhands.server.user_auth import get_user_id


# Test user ID constant (must be a valid UUID string)
TEST_USER_ID = str(uuid.uuid4())


@pytest.fixture
def mock_app():
    """Create a test FastAPI app with organization routes and mocked auth."""
    app = FastAPI()
    app.include_router(org_router)

    # Override the auth dependency to return a test user
    def mock_get_admin_user_id():
        return TEST_USER_ID

    def mock_get_user_id():
        return TEST_USER_ID

    app.dependency_overrides[get_admin_user_id] = mock_get_admin_user_id
    app.dependency_overrides[get_user_id] = mock_get_user_id

    return app


@pytest.fixture
def mock_request():
    """Create a mock request object."""
    request = MagicMock(spec=Request)
    return request


@pytest.fixture
def org_id():
    """Create a test organization ID."""
    return str(uuid.uuid4())


@pytest.fixture
def current_user_id():
    """Create a test current user ID."""
    return str(uuid.uuid4())


@pytest.fixture
def target_user_id():
    """Create a test target user ID."""
    return str(uuid.uuid4())


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
async def test_create_org_is_not_personal(mock_app):
    """
    GIVEN: Admin creates a new team organization
    WHEN: POST /api/organizations is called
    THEN: is_personal field is False (team orgs have different ID than creator)
    """
    # Arrange
    org_id = uuid.uuid4()  # Different from user_id
    mock_org = Org(
        id=org_id,
        name='New Team Organization',
        contact_name='John Doe',
        contact_email='john@example.com',
        org_version=5,
    )

    request_data = {
        'name': 'New Team Organization',
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
        assert response_data['is_personal'] is False


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

    # Store test_user_id for test access
    app.state.test_user_id = test_user_id

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
    mock_user = MagicMock()
    mock_user.current_org_id = org_id

    with (
        patch(
            'server.routes.orgs.UserStore.get_user_by_id_async',
            AsyncMock(return_value=mock_user),
        ),
        patch(
            'server.routes.orgs.OrgService.get_user_orgs_paginated',
            return_value=([mock_org], None),
        ),
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
async def test_list_user_orgs_returns_current_org_id(mock_app_list):
    """
    GIVEN: User has a current organization set
    WHEN: GET /api/organizations is called
    THEN: Response includes current_org_id matching the user's current org
    """
    # Arrange
    current_org_id = uuid.uuid4()
    other_org_id = uuid.uuid4()

    current_org = Org(
        id=current_org_id,
        name='Current Organization',
        contact_name='John Doe',
        contact_email='john@example.com',
    )
    other_org = Org(
        id=other_org_id,
        name='Other Organization',
        contact_name='Jane Doe',
        contact_email='jane@example.com',
    )
    mock_user = MagicMock()
    mock_user.current_org_id = current_org_id

    with (
        patch(
            'server.routes.orgs.UserStore.get_user_by_id_async',
            AsyncMock(return_value=mock_user),
        ),
        patch(
            'server.routes.orgs.OrgService.get_user_orgs_paginated',
            return_value=([current_org, other_org], None),
        ),
    ):
        client = TestClient(mock_app_list)

        # Act
        response = client.get('/api/organizations')

        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert 'current_org_id' in response_data
        assert response_data['current_org_id'] == str(current_org_id)


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
    mock_user = MagicMock()
    mock_user.current_org_id = org1.id

    with (
        patch(
            'server.routes.orgs.UserStore.get_user_by_id_async',
            AsyncMock(return_value=mock_user),
        ),
        patch(
            'server.routes.orgs.OrgService.get_user_orgs_paginated',
            return_value=([org1, org2], '2'),
        ),
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
    mock_user = MagicMock()
    mock_user.current_org_id = uuid.uuid4()

    with (
        patch(
            'server.routes.orgs.UserStore.get_user_by_id_async',
            AsyncMock(return_value=mock_user),
        ),
        patch(
            'server.routes.orgs.OrgService.get_user_orgs_paginated',
            return_value=([], None),
        ),
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
    mock_user = MagicMock()
    mock_user.current_org_id = uuid.uuid4()

    with (
        patch(
            'server.routes.orgs.UserStore.get_user_by_id_async',
            AsyncMock(return_value=mock_user),
        ),
        patch(
            'server.routes.orgs.OrgService.get_user_orgs_paginated',
            side_effect=Exception('Database error'),
        ),
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
async def test_list_user_orgs_personal_org_identified(mock_app_list):
    """
    GIVEN: User has a personal organization (org.id == user_id)
    WHEN: GET /api/organizations is called
    THEN: is_personal field is True for personal org
    """
    # Arrange
    user_id = mock_app_list.state.test_user_id
    personal_org_id = uuid.UUID(user_id)

    personal_org = Org(
        id=personal_org_id,
        name=f'user_{user_id}_org',
        contact_name='John Doe',
        contact_email='john@example.com',
    )
    mock_user = MagicMock()
    mock_user.current_org_id = personal_org_id

    with (
        patch(
            'server.routes.orgs.UserStore.get_user_by_id_async',
            AsyncMock(return_value=mock_user),
        ),
        patch(
            'server.routes.orgs.OrgService.get_user_orgs_paginated',
            return_value=([personal_org], None),
        ),
    ):
        client = TestClient(mock_app_list)

        # Act
        response = client.get('/api/organizations')

        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert len(response_data['items']) == 1
        assert response_data['items'][0]['is_personal'] is True


@pytest.mark.asyncio
async def test_list_user_orgs_team_org_identified(mock_app_list):
    """
    GIVEN: User has a team organization (org.id != user_id)
    WHEN: GET /api/organizations is called
    THEN: is_personal field is False for team org
    """
    # Arrange
    team_org = Org(
        id=uuid.uuid4(),  # Different from user_id
        name='Team Organization',
        contact_name='John Doe',
        contact_email='john@example.com',
    )
    mock_user = MagicMock()
    mock_user.current_org_id = team_org.id

    with (
        patch(
            'server.routes.orgs.UserStore.get_user_by_id_async',
            AsyncMock(return_value=mock_user),
        ),
        patch(
            'server.routes.orgs.OrgService.get_user_orgs_paginated',
            return_value=([team_org], None),
        ),
    ):
        client = TestClient(mock_app_list)

        # Act
        response = client.get('/api/organizations')

        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert len(response_data['items']) == 1
        assert response_data['items'][0]['is_personal'] is False


@pytest.mark.asyncio
async def test_list_user_orgs_mixed_personal_and_team(mock_app_list):
    """
    GIVEN: User has both personal and team organizations
    WHEN: GET /api/organizations is called
    THEN: is_personal field correctly identifies each org type
    """
    # Arrange
    user_id = mock_app_list.state.test_user_id
    personal_org_id = uuid.UUID(user_id)

    personal_org = Org(
        id=personal_org_id,
        name=f'user_{user_id}_org',
        contact_name='John Doe',
        contact_email='john@example.com',
    )

    team_org = Org(
        id=uuid.uuid4(),
        name='Team Organization',
        contact_name='Jane Doe',
        contact_email='jane@example.com',
    )
    mock_user = MagicMock()
    mock_user.current_org_id = personal_org_id

    with (
        patch(
            'server.routes.orgs.UserStore.get_user_by_id_async',
            AsyncMock(return_value=mock_user),
        ),
        patch(
            'server.routes.orgs.OrgService.get_user_orgs_paginated',
            return_value=([personal_org, team_org], None),
        ),
    ):
        client = TestClient(mock_app_list)

        # Act
        response = client.get('/api/organizations')

        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert len(response_data['items']) == 2

        # Find personal and team orgs in response
        personal_org_response = next(
            item
            for item in response_data['items']
            if item['id'] == str(personal_org_id)
        )
        team_org_response = next(
            item
            for item in response_data['items']
            if item['id'] != str(personal_org_id)
        )

        assert personal_org_response['is_personal'] is True
        assert team_org_response['is_personal'] is False


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
    mock_user = MagicMock()
    mock_user.current_org_id = org_id

    with (
        patch(
            'server.routes.orgs.UserStore.get_user_by_id_async',
            AsyncMock(return_value=mock_user),
        ),
        patch(
            'server.routes.orgs.OrgService.get_user_orgs_paginated',
            return_value=([mock_org], None),
        ),
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

    # Override the auth dependency to return a valid UUID test user
    def mock_get_user_id():
        return TEST_USER_ID

    app.dependency_overrides[get_user_id] = mock_get_user_id

    return app


@pytest.fixture
def mock_owner_role():
    """Create a mock owner role for authorization tests."""
    mock_role = MagicMock()
    mock_role.name = 'owner'
    return mock_role


@pytest.mark.asyncio
async def test_get_org_success(mock_app_with_get_user_id, mock_owner_role):
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
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_owner_role),
        ),
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
    THEN: 403 Forbidden error is returned (permission check fails first)
    """
    # Arrange
    org_id = uuid.uuid4()

    # When user is not a member, get_user_org_role returns None
    with patch(
        'server.auth.authorization.get_user_org_role_async',
        AsyncMock(return_value=None),
    ):
        client = TestClient(mock_app_with_get_user_id)

        # Act
        response = client.get(f'/api/organizations/{org_id}')

        # Assert - Permission check now returns 403 before org lookup
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert 'not a member' in response.json()['detail'].lower()


@pytest.mark.asyncio
async def test_get_org_not_found(mock_app_with_get_user_id, mock_owner_role):
    """
    GIVEN: Organization does not exist
    WHEN: GET /api/organizations/{org_id} is called
    THEN: 404 Not Found error is returned
    """
    # Arrange
    org_id = uuid.uuid4()

    with (
        patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_owner_role),
        ),
        patch(
            'server.routes.orgs.OrgService.get_org_by_id',
            AsyncMock(side_effect=OrgNotFoundError(str(org_id))),
        ),
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
async def test_get_org_unexpected_error(mock_app_with_get_user_id, mock_owner_role):
    """
    GIVEN: Unexpected error occurs during retrieval
    WHEN: GET /api/organizations/{org_id} is called
    THEN: 500 Internal Server Error is returned
    """
    # Arrange
    org_id = uuid.uuid4()

    with (
        patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_owner_role),
        ),
        patch(
            'server.routes.orgs.OrgService.get_org_by_id',
            AsyncMock(side_effect=RuntimeError('Unexpected database error')),
        ),
    ):
        client = TestClient(mock_app_with_get_user_id)

        # Act
        response = client.get(f'/api/organizations/{org_id}')

        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert 'unexpected error' in response.json()['detail'].lower()


@pytest.mark.asyncio
async def test_get_org_personal_workspace():
    """
    GIVEN: User retrieves their personal organization (org.id == user_id)
    WHEN: GET /api/organizations/{org_id} is called
    THEN: is_personal field is True
    """
    # Arrange
    app = FastAPI()
    app.include_router(org_router)

    # Use a valid UUID for user_id
    user_id = str(uuid.uuid4())
    org_id = uuid.UUID(user_id)  # Personal org has same ID as user

    def mock_get_user_id():
        return user_id

    app.dependency_overrides[get_user_id] = mock_get_user_id

    mock_org = Org(
        id=org_id,
        name=f'user_{user_id}_org',
        contact_name='John Doe',
        contact_email='john@example.com',
        org_version=5,
    )

    mock_role = MagicMock()
    mock_role.name = 'owner'

    with (
        patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_role),
        ),
        patch(
            'server.routes.orgs.OrgService.get_org_by_id',
            AsyncMock(return_value=mock_org),
        ),
        patch(
            'server.routes.orgs.OrgService.get_org_credits',
            AsyncMock(return_value=50.0),
        ),
    ):
        client = TestClient(app)

        # Act
        response = client.get(f'/api/organizations/{org_id}')

        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data['is_personal'] is True


@pytest.mark.asyncio
async def test_get_org_team_workspace(mock_app_with_get_user_id, mock_owner_role):
    """
    GIVEN: User retrieves a team organization (org.id != user_id)
    WHEN: GET /api/organizations/{org_id} is called
    THEN: is_personal field is False
    """
    # Arrange
    org_id = uuid.uuid4()  # Different from user_id
    mock_org = Org(
        id=org_id,
        name='Team Organization',
        contact_name='John Doe',
        contact_email='john@example.com',
        org_version=5,
    )

    with (
        patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_owner_role),
        ),
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
        assert response_data['is_personal'] is False


@pytest.mark.asyncio
async def test_get_org_with_credits_none(mock_app_with_get_user_id, mock_owner_role):
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
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_owner_role),
        ),
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
async def test_get_org_sensitive_fields_not_exposed(
    mock_app_with_get_user_id, mock_owner_role
):
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
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_owner_role),
        ),
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
async def test_delete_org_success(mock_app, mock_owner_role):
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

    with (
        patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_owner_role),
        ),
        patch(
            'server.routes.orgs.OrgService.delete_org_with_cleanup',
            AsyncMock(return_value=mock_deleted_org),
        ),
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
async def test_delete_org_not_found(mock_app, mock_owner_role):
    """
    GIVEN: Organization does not exist
    WHEN: DELETE /api/organizations/{org_id} is called
    THEN: 404 Not Found error is returned
    """
    # Arrange
    org_id = uuid.uuid4()

    with (
        patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_owner_role),
        ),
        patch(
            'server.routes.orgs.OrgService.delete_org_with_cleanup',
            AsyncMock(side_effect=OrgNotFoundError(str(org_id))),
        ),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.delete(f'/api/organizations/{org_id}')

        # Assert
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert str(org_id) in response.json()['detail']


@pytest.mark.asyncio
async def test_delete_org_not_owner(mock_app, mock_owner_role):
    """
    GIVEN: User is not the organization owner
    WHEN: DELETE /api/organizations/{org_id} is called
    THEN: 403 Forbidden error is returned
    """
    # Arrange
    org_id = uuid.uuid4()

    with (
        patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_owner_role),
        ),
        patch(
            'server.routes.orgs.OrgService.delete_org_with_cleanup',
            AsyncMock(
                side_effect=OrgAuthorizationError(
                    'Only organization owners can delete organizations'
                )
            ),
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

    # When user is not a member, get_user_org_role returns None
    with patch(
        'server.auth.authorization.get_user_org_role_async',
        AsyncMock(return_value=None),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.delete(f'/api/organizations/{org_id}')

        # Assert
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert 'not a member' in response.json()['detail']


@pytest.mark.asyncio
async def test_delete_org_database_failure(mock_app, mock_owner_role):
    """
    GIVEN: Database operation fails during deletion
    WHEN: DELETE /api/organizations/{org_id} is called
    THEN: 500 Internal Server Error is returned
    """
    # Arrange
    org_id = uuid.uuid4()

    with (
        patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_owner_role),
        ),
        patch(
            'server.routes.orgs.OrgService.delete_org_with_cleanup',
            AsyncMock(side_effect=OrgDatabaseError('Database connection failed')),
        ),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.delete(f'/api/organizations/{org_id}')

        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert response.json()['detail'] == 'Failed to delete organization'


@pytest.mark.asyncio
async def test_delete_org_unexpected_error(mock_app, mock_owner_role):
    """
    GIVEN: Unexpected error occurs during deletion
    WHEN: DELETE /api/organizations/{org_id} is called
    THEN: 500 Internal Server Error is returned with generic message
    """
    # Arrange
    org_id = uuid.uuid4()

    with (
        patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_owner_role),
        ),
        patch(
            'server.routes.orgs.OrgService.delete_org_with_cleanup',
            AsyncMock(side_effect=RuntimeError('Unexpected system error')),
        ),
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
async def test_delete_org_unauthorized(mock_app, mock_owner_role):
    """
    GIVEN: User is not authenticated
    WHEN: DELETE /api/organizations/{org_id} is called
    THEN: 403 Forbidden error is returned (user not authorized)
    """
    # Arrange
    org_id = uuid.uuid4()

    with (
        patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_owner_role),
        ),
        patch(
            'server.routes.orgs.OrgService.delete_org_with_cleanup',
            AsyncMock(side_effect=OrgAuthorizationError('User not authorized')),
        ),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.delete(f'/api/organizations/{org_id}')

        # Assert
        assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_delete_org_orphaned_users(mock_app, mock_owner_role):
    """
    GIVEN: Deleting org would leave users without any organization
    WHEN: DELETE /api/organizations/{org_id} is called
    THEN: 400 Bad Request error is returned with user count in message
    """
    # Arrange
    org_id = uuid.uuid4()
    orphaned_user_ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    with (
        patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_owner_role),
        ),
        patch(
            'server.routes.orgs.OrgService.delete_org_with_cleanup',
            AsyncMock(side_effect=OrphanedUserError(orphaned_user_ids)),
        ),
    ):
        client = TestClient(mock_app)

        # Act
        response = client.delete(f'/api/organizations/{org_id}')

        # Assert
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert '2 user(s)' in response.json()['detail']
        assert 'no remaining organization' in response.json()['detail']


@pytest.fixture
def mock_update_app():
    """Create a test FastAPI app with organization routes and mocked auth for update endpoint."""
    app = FastAPI()
    app.include_router(org_router)

    # Override the auth dependency to return a test user
    async def mock_user_id():
        return TEST_USER_ID

    app.dependency_overrides[get_user_id] = mock_user_id

    return app


# Note: Success cases for update endpoint are tested in test_org_service.py
# Route handler tests focus on error handling and validation


@pytest.mark.asyncio
async def test_update_org_personal_workspace_preserved():
    """
    GIVEN: User updates their personal organization
    WHEN: PATCH /api/organizations/{org_id} is called
    THEN: is_personal field remains True in response
    """
    # Arrange
    app = FastAPI()
    app.include_router(org_router)

    user_id = str(uuid.uuid4())
    org_id = uuid.UUID(user_id)  # Personal org

    async def mock_user_id():
        return user_id

    app.dependency_overrides[get_user_id] = mock_user_id

    updated_org = Org(
        id=org_id,
        name=f'user_{user_id}_org',
        contact_name='Updated Name',
        contact_email='john@example.com',
        org_version=5,
    )

    update_data = {'contact_name': 'Updated Name'}

    mock_role = MagicMock()
    mock_role.name = 'owner'

    with (
        patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_role),
        ),
        patch(
            'server.routes.orgs.OrgService.update_org_with_permissions',
            AsyncMock(return_value=updated_org),
        ),
        patch(
            'server.routes.orgs.OrgService.get_org_credits',
            AsyncMock(return_value=75.0),
        ),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            # Act
            response = await client.patch(
                f'/api/organizations/{org_id}', json=update_data
            )

            # Assert
            assert response.status_code == status.HTTP_200_OK
            response_data = response.json()
            assert response_data['is_personal'] is True
            assert response_data['contact_name'] == 'Updated Name'


@pytest.mark.asyncio
async def test_update_org_team_workspace_preserved():
    """
    GIVEN: User updates a team organization
    WHEN: PATCH /api/organizations/{org_id} is called
    THEN: is_personal field remains False in response
    """
    # Arrange
    app = FastAPI()
    app.include_router(org_router)

    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()  # Team org (different from user_id)

    async def mock_user_id():
        return user_id

    app.dependency_overrides[get_user_id] = mock_user_id

    updated_org = Org(
        id=org_id,
        name='Updated Team Org',
        contact_name='Jane Doe',
        contact_email='jane@example.com',
        org_version=5,
    )

    update_data = {'name': 'Updated Team Org'}

    mock_role = MagicMock()
    mock_role.name = 'owner'

    with (
        patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_role),
        ),
        patch(
            'server.routes.orgs.OrgService.update_org_with_permissions',
            AsyncMock(return_value=updated_org),
        ),
        patch(
            'server.routes.orgs.OrgService.get_org_credits',
            AsyncMock(return_value=150.0),
        ),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            # Act
            response = await client.patch(
                f'/api/organizations/{org_id}', json=update_data
            )

            # Assert
            assert response.status_code == status.HTTP_200_OK
            response_data = response.json()
            assert response_data['is_personal'] is False
            assert response_data['name'] == 'Updated Team Org'


@pytest.mark.asyncio
async def test_update_org_not_found(mock_update_app, mock_owner_role):
    """
    GIVEN: Organization ID does not exist
    WHEN: PATCH /api/organizations/{org_id} is called
    THEN: 404 Not Found error is returned
    """
    # Arrange
    org_id = uuid.uuid4()
    update_data = {'contact_name': 'Jane Doe'}

    with (
        patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_owner_role),
        ),
        patch(
            'server.routes.orgs.OrgService.update_org_with_permissions',
            AsyncMock(
                side_effect=ValueError(f'Organization with ID {org_id} not found')
            ),
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

    # When user is not a member, get_user_org_role returns None
    with patch(
        'server.auth.authorization.get_user_org_role_async',
        AsyncMock(return_value=None),
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
async def test_update_org_permission_denied_llm_settings(
    mock_update_app, mock_owner_role
):
    """
    GIVEN: User lacks admin/owner role but tries to update LLM settings
    WHEN: PATCH /api/organizations/{org_id} is called
    THEN: 403 Forbidden error is returned
    """
    # Arrange
    org_id = uuid.uuid4()
    update_data = {'default_llm_model': 'claude-opus-4-5-20251101'}

    with (
        patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_owner_role),
        ),
        patch(
            'server.routes.orgs.OrgService.update_org_with_permissions',
            AsyncMock(
                side_effect=PermissionError(
                    'Admin or owner role required to update LLM settings'
                )
            ),
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
async def test_update_org_duplicate_name_returns_409(mock_update_app, mock_owner_role):
    """
    GIVEN: User updates organization name to one already used by another org
    WHEN: PATCH /api/organizations/{org_id} is called with that name
    THEN: 409 Conflict is returned with message about name already existing
    """
    # Arrange
    org_id = uuid.uuid4()
    update_data = {'name': 'Existing Organization'}

    with (
        patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_owner_role),
        ),
        patch(
            'server.routes.orgs.OrgService.update_org_with_permissions',
            AsyncMock(side_effect=OrgNameExistsError('Existing Organization')),
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
            assert response.status_code == status.HTTP_409_CONFLICT
            assert 'already exists' in response.json()['detail'].lower()


@pytest.mark.asyncio
async def test_update_org_database_error(mock_update_app, mock_owner_role):
    """
    GIVEN: Database operation fails during update
    WHEN: PATCH /api/organizations/{org_id} is called
    THEN: 500 Internal Server Error is returned
    """
    # Arrange
    org_id = uuid.uuid4()
    update_data = {'contact_name': 'Jane Doe'}

    with (
        patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_owner_role),
        ),
        patch(
            'server.routes.orgs.OrgService.update_org_with_permissions',
            AsyncMock(side_effect=OrgDatabaseError('Database connection failed')),
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
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert 'Failed to update organization' in response.json()['detail']


@pytest.mark.asyncio
async def test_update_org_unexpected_error(mock_update_app, mock_owner_role):
    """
    GIVEN: Unexpected error occurs during update
    WHEN: PATCH /api/organizations/{org_id} is called
    THEN: 500 Internal Server Error is returned with generic message
    """
    # Arrange
    org_id = uuid.uuid4()
    update_data = {'contact_name': 'Jane Doe'}

    with (
        patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_owner_role),
        ),
        patch(
            'server.routes.orgs.OrgService.update_org_with_permissions',
            AsyncMock(side_effect=RuntimeError('Unexpected system error')),
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
async def test_update_org_invalid_field_values(mock_update_app, mock_owner_role):
    """
    GIVEN: Update request with invalid field values (e.g., negative max_iterations)
    WHEN: PATCH /api/organizations/{org_id} is called
    THEN: 422 validation error is returned
    """
    # Arrange
    org_id = uuid.uuid4()
    update_data = {'default_max_iterations': -1}  # Invalid: must be > 0

    with patch(
        'server.auth.authorization.get_user_org_role_async',
        AsyncMock(return_value=mock_owner_role),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_update_app), base_url='http://test'
        ) as client:
            # Act
            response = await client.patch(
                f'/api/organizations/{org_id}', json=update_data
            )

            # Assert
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_update_org_empty_name_returns_422(mock_update_app, mock_owner_role):
    """
    GIVEN: Update request with empty organization name (after strip)
    WHEN: PATCH /api/organizations/{org_id} is called
    THEN: 422 validation error is returned (OrgUpdate name min_length=1)
    """
    # Arrange
    org_id = uuid.uuid4()
    update_data = {'name': '   '}

    with patch(
        'server.auth.authorization.get_user_org_role_async',
        AsyncMock(return_value=mock_owner_role),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_update_app), base_url='http://test'
        ) as client:
            # Act
            response = await client.patch(
                f'/api/organizations/{org_id}', json=update_data
            )

            # Assert
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_update_org_invalid_email_format(mock_update_app, mock_owner_role):
    """
    GIVEN: Update request with invalid email format
    WHEN: PATCH /api/organizations/{org_id} is called
    THEN: 422 validation error is returned
    """
    # Arrange
    org_id = uuid.uuid4()
    update_data = {'contact_email': 'invalid-email'}  # Missing @

    with patch(
        'server.auth.authorization.get_user_org_role_async',
        AsyncMock(return_value=mock_owner_role),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_update_app), base_url='http://test'
        ) as client:
            # Act
            response = await client.patch(
                f'/api/organizations/{org_id}', json=update_data
            )

            # Assert
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestGetOrgMembersEndpoint:
    """Test cases for GET /api/organizations/{org_id}/members endpoint."""

    @pytest.mark.asyncio
    async def test_get_members_succeeds_returns_200(self, org_id, current_user_id):
        """Test that successful retrieval returns 200 with paginated members."""
        # Arrange
        mock_page = OrgMemberPage(
            items=[
                OrgMemberResponse(
                    user_id=str(uuid.uuid4()),
                    email='user1@example.com',
                    role_id=1,
                    role='owner',
                    role_rank=10,
                    status='active',
                )
            ],
            current_page=1,
            per_page=100,
        )

        with patch(
            'server.routes.orgs.OrgMemberService.get_org_members',
            AsyncMock(return_value=(True, None, mock_page)),
        ) as mock_get:
            # Act
            result = await get_org_members(
                org_id=uuid.UUID(org_id),
                page_id=None,
                limit=100,
                user_id=current_user_id,
            )

            # Assert
            assert isinstance(result, OrgMemberPage)
            assert len(result.items) == 1
            assert result.current_page == 1
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_not_a_member_returns_403(self, org_id, current_user_id):
        """Test that not being a member returns 403 Forbidden."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.get_org_members',
            AsyncMock(return_value=(False, 'not_a_member', None)),
        ):
            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await get_org_members(
                    org_id=uuid.UUID(org_id),
                    page_id=None,
                    limit=100,
                    user_id=current_user_id,
                )

            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
            assert 'not a member of this organization' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_invalid_page_id_returns_400(self, org_id, current_user_id):
        """Test that invalid page_id format returns 400 Bad Request."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.get_org_members',
            AsyncMock(return_value=(False, 'invalid_page_id', None)),
        ):
            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await get_org_members(
                    org_id=uuid.UUID(org_id),
                    page_id='invalid',
                    limit=100,
                    user_id=current_user_id,
                )

            assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
            assert 'Invalid page_id format' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_invalid_org_id_format_returns_422(self, mock_app, current_user_id):
        """Test that invalid org_id UUID format returns 422 Unprocessable Entity.

        Note: FastAPI validates UUID path parameters at the routing level,
        so this test uses the HTTP test client to verify the validation.
        """
        # Arrange
        invalid_org_id = 'not-a-uuid'

        # Act & Assert - Use test client to test route-level validation
        with TestClient(mock_app) as client:
            response = client.get(f'/api/organizations/{invalid_org_id}/members')
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_invalid_current_user_id_format_returns_400(self, org_id):
        """Test that invalid current_user_id UUID format returns 400 Bad Request."""
        # Arrange
        invalid_current_user_id = 'not-a-uuid'

        # Act & Assert - endpoint now expects UUID type for org_id
        with pytest.raises(HTTPException) as exc_info:
            await get_org_members(
                org_id=uuid.UUID(org_id),
                page_id=None,
                limit=100,
                user_id=invalid_current_user_id,
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'Invalid organization ID format' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_data_is_none_returns_500(self, org_id, current_user_id):
        """Test that None data returns 500 Internal Server Error."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.get_org_members',
            AsyncMock(return_value=(True, None, None)),
        ):
            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await get_org_members(
                    org_id=uuid.UUID(org_id),
                    page_id=None,
                    limit=100,
                    user_id=current_user_id,
                )

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert 'Failed to retrieve members' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_unknown_error_returns_500(self, org_id, current_user_id):
        """Test that unknown error returns 500 Internal Server Error."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.get_org_members',
            AsyncMock(return_value=(False, 'unknown_error', None)),
        ):
            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await get_org_members(
                    org_id=uuid.UUID(org_id),
                    page_id=None,
                    limit=100,
                    user_id=current_user_id,
                )

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert 'An error occurred' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_service_exception_returns_500(self, org_id, current_user_id):
        """Test that service exception returns 500 Internal Server Error."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.get_org_members',
            AsyncMock(side_effect=Exception('Database connection failed')),
        ):
            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await get_org_members(
                    org_id=uuid.UUID(org_id),
                    page_id=None,
                    limit=100,
                    user_id=current_user_id,
                )

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert 'Failed to retrieve members' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_http_exception_is_re_raised(self, org_id, current_user_id):
        """Test that HTTPException from service is re-raised."""
        # Arrange
        original_exception = HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Service temporarily unavailable',
        )

        with patch(
            'server.routes.orgs.OrgMemberService.get_org_members',
            AsyncMock(side_effect=original_exception),
        ):
            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await get_org_members(
                    org_id=uuid.UUID(org_id),
                    page_id=None,
                    limit=100,
                    user_id=current_user_id,
                )

            assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            assert exc_info.value.detail == 'Service temporarily unavailable'

    @pytest.mark.asyncio
    async def test_pagination_with_page_id(self, org_id, current_user_id):
        """Test that pagination works with page_id parameter."""
        # Arrange
        mock_page = OrgMemberPage(
            items=[
                OrgMemberResponse(
                    user_id=str(uuid.uuid4()),
                    email='user2@example.com',
                    role_id=2,
                    role='admin',
                    role_rank=20,
                    status='active',
                )
            ],
            current_page=2,
            per_page=100,
        )

        with patch(
            'server.routes.orgs.OrgMemberService.get_org_members',
            AsyncMock(return_value=(True, None, mock_page)),
        ) as mock_get:
            # Act - endpoint now expects UUID type for org_id
            result = await get_org_members(
                org_id=uuid.UUID(org_id),
                page_id='100',
                limit=100,
                user_id=current_user_id,
            )

            # Assert
            assert isinstance(result, OrgMemberPage)
            assert result.current_page == 2
            mock_get.assert_called_once_with(
                org_id=uuid.UUID(org_id),
                current_user_id=uuid.UUID(current_user_id),
                page_id='100',
                limit=100,
                email_filter=None,
            )


class TestGetOrgMembersCountEndpoint:
    """Test cases for GET /api/organizations/{org_id}/members/count endpoint."""

    @pytest.mark.asyncio
    async def test_count_succeeds_returns_int(self, org_id, current_user_id):
        """Test that successful count returns an integer."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.get_org_members_count',
            AsyncMock(return_value=42),
        ) as mock_get_count:
            # Import here to avoid circular import issues
            from server.routes.orgs import get_org_members_count

            # Act
            result = await get_org_members_count(
                org_id=uuid.UUID(org_id),
                email=None,
                user_id=current_user_id,
            )

            # Assert
            assert result == 42
            mock_get_count.assert_called_once_with(
                org_id=uuid.UUID(org_id),
                current_user_id=uuid.UUID(current_user_id),
                email_filter=None,
            )

    @pytest.mark.asyncio
    async def test_count_with_email_filter(self, org_id, current_user_id):
        """Test that email filter is passed to service."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.get_org_members_count',
            AsyncMock(return_value=5),
        ) as mock_get_count:
            from server.routes.orgs import get_org_members_count

            # Act
            result = await get_org_members_count(
                org_id=uuid.UUID(org_id),
                email='alice',
                user_id=current_user_id,
            )

            # Assert
            assert result == 5
            mock_get_count.assert_called_once_with(
                org_id=uuid.UUID(org_id),
                current_user_id=uuid.UUID(current_user_id),
                email_filter='alice',
            )

    @pytest.mark.asyncio
    async def test_not_a_member_returns_403(self, org_id, current_user_id):
        """Test that OrgMemberNotFoundError returns 403 Forbidden."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.get_org_members_count',
            AsyncMock(side_effect=OrgMemberNotFoundError(org_id, current_user_id)),
        ):
            from server.routes.orgs import get_org_members_count

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await get_org_members_count(
                    org_id=uuid.UUID(org_id),
                    email=None,
                    user_id=current_user_id,
                )

            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
            assert 'not a member of this organization' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_invalid_uuid_returns_400(self, org_id):
        """Test that invalid user_id UUID format returns 400 Bad Request."""
        # Arrange
        invalid_user_id = 'not-a-uuid'

        from server.routes.orgs import get_org_members_count

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_org_members_count(
                org_id=uuid.UUID(org_id),
                email=None,
                user_id=invalid_user_id,
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'Invalid organization ID format' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_service_exception_returns_500(self, org_id, current_user_id):
        """Test that generic exception returns 500 Internal Server Error."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.get_org_members_count',
            AsyncMock(side_effect=Exception('Database error')),
        ):
            from server.routes.orgs import get_org_members_count

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await get_org_members_count(
                    org_id=uuid.UUID(org_id),
                    email=None,
                    user_id=current_user_id,
                )

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert 'Failed to retrieve member count' in exc_info.value.detail


class TestRemoveOrgMemberEndpoint:
    """Test cases for DELETE /api/organizations/{org_id}/members/{user_id} endpoint."""

    @pytest.mark.asyncio
    async def test_remove_member_succeeds_returns_200(
        self, mock_request, org_id, current_user_id, target_user_id
    ):
        """Test that successful removal returns 200 with success message."""
        # Arrange
        with (
            patch(
                'server.routes.orgs.get_user_id', return_value=current_user_id
            ) as mock_get_user_id,
            patch(
                'server.routes.orgs.OrgMemberService.remove_org_member'
            ) as mock_remove,
        ):
            mock_remove.return_value = (True, None)

            # Act
            result = await remove_org_member(
                org_id=uuid.UUID(org_id),
                user_id=target_user_id,
                current_user_id=current_user_id,
            )

            # Assert
            assert result == {'message': 'Member removed successfully'}
            mock_get_user_id.assert_not_called()  # current_user_id is passed directly
            mock_remove.assert_called_once()

    @pytest.mark.asyncio
    async def test_not_a_member_returns_403(
        self, mock_request, org_id, current_user_id, target_user_id
    ):
        """Test that not being a member returns 403 Forbidden."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.remove_org_member'
        ) as mock_remove:
            mock_remove.return_value = (False, 'not_a_member')

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await remove_org_member(
                    org_id=uuid.UUID(org_id),
                    user_id=target_user_id,
                    current_user_id=current_user_id,
                )

            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
            assert 'not a member of this organization' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_cannot_remove_self_returns_403(
        self, mock_request, org_id, current_user_id
    ):
        """Test that trying to remove oneself returns 403 Forbidden."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.remove_org_member'
        ) as mock_remove:
            mock_remove.return_value = (False, 'cannot_remove_self')

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await remove_org_member(
                    org_id=uuid.UUID(org_id),
                    user_id=current_user_id,
                    current_user_id=current_user_id,
                )

            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
            assert 'Cannot remove yourself' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_member_not_found_returns_404(
        self, mock_request, org_id, current_user_id, target_user_id
    ):
        """Test that member not found returns 404 Not Found."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.remove_org_member'
        ) as mock_remove:
            mock_remove.return_value = (False, 'member_not_found')

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await remove_org_member(
                    org_id=uuid.UUID(org_id),
                    user_id=target_user_id,
                    current_user_id=current_user_id,
                )

            assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
            assert 'Member not found' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_insufficient_permission_returns_403(
        self, mock_request, org_id, current_user_id, target_user_id
    ):
        """Test that insufficient permission returns 403 Forbidden."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.remove_org_member'
        ) as mock_remove:
            mock_remove.return_value = (False, 'insufficient_permission')

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await remove_org_member(
                    org_id=uuid.UUID(org_id),
                    user_id=target_user_id,
                    current_user_id=current_user_id,
                )

            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
            assert 'do not have permission' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_cannot_remove_last_owner_returns_400(
        self, mock_request, org_id, current_user_id, target_user_id
    ):
        """Test that removing last owner returns 400 Bad Request."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.remove_org_member'
        ) as mock_remove:
            mock_remove.return_value = (False, 'cannot_remove_last_owner')

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await remove_org_member(
                    org_id=uuid.UUID(org_id),
                    user_id=target_user_id,
                    current_user_id=current_user_id,
                )

            assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
            assert 'Cannot remove the last owner' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_removal_failed_returns_500(
        self, mock_request, org_id, current_user_id, target_user_id
    ):
        """Test that removal failure returns 500 Internal Server Error."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.remove_org_member'
        ) as mock_remove:
            mock_remove.return_value = (False, 'removal_failed')

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await remove_org_member(
                    org_id=uuid.UUID(org_id),
                    user_id=target_user_id,
                    current_user_id=current_user_id,
                )

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert 'Failed to remove member' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_unknown_error_returns_500(
        self, mock_request, org_id, current_user_id, target_user_id
    ):
        """Test that unknown error returns 500 Internal Server Error."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.remove_org_member'
        ) as mock_remove:
            mock_remove.return_value = (False, 'unknown_error')

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await remove_org_member(
                    org_id=uuid.UUID(org_id),
                    user_id=target_user_id,
                    current_user_id=current_user_id,
                )

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert 'An error occurred' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_invalid_org_id_format_returns_422(
        self, mock_app, current_user_id, target_user_id
    ):
        """Test that invalid org_id UUID format returns 422 Unprocessable Entity.

        Note: FastAPI validates UUID path parameters at the routing level,
        so this test uses the HTTP test client to verify the validation.
        """
        # Arrange
        invalid_org_id = 'not-a-uuid'

        # Act & Assert - Use test client to test route-level validation
        with TestClient(mock_app) as client:
            response = client.delete(
                f'/api/organizations/{invalid_org_id}/members/{target_user_id}'
            )
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_invalid_user_id_format_returns_400(
        self, mock_request, org_id, current_user_id
    ):
        """Test that invalid user_id UUID format returns 400 Bad Request."""
        # Arrange
        invalid_user_id = 'not-a-uuid'

        # Act & Assert - endpoint now expects UUID type for org_id
        with pytest.raises(HTTPException) as exc_info:
            await remove_org_member(
                org_id=uuid.UUID(org_id),
                user_id=invalid_user_id,
                current_user_id=current_user_id,
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'Invalid organization or user ID format' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_invalid_current_user_id_format_returns_400(
        self, mock_request, org_id, target_user_id
    ):
        """Test that invalid current_user_id UUID format returns 400 Bad Request."""
        # Arrange
        invalid_current_user_id = 'not-a-uuid'

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await remove_org_member(
                org_id=uuid.UUID(org_id),
                user_id=target_user_id,
                current_user_id=invalid_current_user_id,
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'Invalid organization or user ID format' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_service_exception_returns_500(
        self, mock_request, org_id, current_user_id, target_user_id
    ):
        """Test that service exception returns 500 Internal Server Error."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.remove_org_member'
        ) as mock_remove:
            mock_remove.side_effect = Exception('Database connection failed')

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await remove_org_member(
                    org_id=uuid.UUID(org_id),
                    user_id=target_user_id,
                    current_user_id=current_user_id,
                )

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert 'Failed to remove member' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_http_exception_is_re_raised(
        self, mock_request, org_id, current_user_id, target_user_id
    ):
        """Test that HTTPException from service is re-raised."""
        # Arrange
        original_exception = HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Service temporarily unavailable',
        )

        with patch(
            'server.routes.orgs.OrgMemberService.remove_org_member'
        ) as mock_remove:
            mock_remove.side_effect = original_exception

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await remove_org_member(
                    org_id=uuid.UUID(org_id),
                    user_id=target_user_id,
                    current_user_id=current_user_id,
                )

            assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            assert exc_info.value.detail == 'Service temporarily unavailable'


class TestUpdateOrgMemberEndpoint:
    """Test cases for PATCH /api/organizations/{org_id}/members/{user_id} endpoint."""

    @pytest.mark.asyncio
    async def test_update_member_role_succeeds_returns_member_response(
        self, org_id, current_user_id, target_user_id
    ):
        """GIVEN valid role update request WHEN PATCH is called THEN returns 200 with updated OrgMemberResponse."""
        # Arrange
        updated = OrgMemberResponse(
            user_id=target_user_id,
            email='user@example.com',
            role_id=2,
            role='admin',
            role_rank=20,
            status='active',
        )
        with patch(
            'server.routes.orgs.OrgMemberService.update_org_member'
        ) as mock_update:
            mock_update.return_value = updated

            # Act - endpoint now expects UUID type for org_id
            result = await update_org_member(
                org_id=uuid.UUID(org_id),
                user_id=target_user_id,
                update_data=OrgMemberUpdate(role='admin'),
                current_user_id=current_user_id,
            )

            # Assert
            assert result == updated
            mock_update.assert_called_once_with(
                org_id=uuid.UUID(org_id),
                target_user_id=uuid.UUID(target_user_id),
                current_user_id=uuid.UUID(current_user_id),
                update_data=OrgMemberUpdate(role='admin'),
            )

    @pytest.mark.asyncio
    async def test_not_a_member_returns_403(
        self, org_id, current_user_id, target_user_id
    ):
        """GIVEN requester is not a member WHEN PATCH is called THEN returns 403."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.update_org_member'
        ) as mock_update:
            mock_update.side_effect = OrgMemberNotFoundError(org_id, current_user_id)

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await update_org_member(
                    org_id=uuid.UUID(org_id),
                    user_id=target_user_id,
                    update_data=OrgMemberUpdate(role='user'),
                    current_user_id=current_user_id,
                )
            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
            assert 'not a member' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_cannot_modify_self_returns_403(self, org_id, current_user_id):
        """GIVEN target user is self WHEN PATCH is called THEN returns 403."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.update_org_member'
        ) as mock_update:
            mock_update.side_effect = CannotModifySelfError('modify')

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await update_org_member(
                    org_id=uuid.UUID(org_id),
                    user_id=current_user_id,
                    update_data=OrgMemberUpdate(role='admin'),
                    current_user_id=current_user_id,
                )
            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
            assert 'Cannot modify your own role' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_member_not_found_returns_404(
        self, org_id, current_user_id, target_user_id
    ):
        """GIVEN target member does not exist WHEN PATCH is called THEN returns 404."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.update_org_member'
        ) as mock_update:
            mock_update.side_effect = OrgMemberNotFoundError(org_id, target_user_id)

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await update_org_member(
                    org_id=uuid.UUID(org_id),
                    user_id=target_user_id,
                    update_data=OrgMemberUpdate(role='user'),
                    current_user_id=current_user_id,
                )
            assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
            assert 'Member not found' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_invalid_role_returns_400(
        self, org_id, current_user_id, target_user_id
    ):
        """GIVEN invalid role name WHEN PATCH is called THEN returns 400."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.update_org_member'
        ) as mock_update:
            mock_update.side_effect = InvalidRoleError('superuser')

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await update_org_member(
                    org_id=uuid.UUID(org_id),
                    user_id=target_user_id,
                    update_data=OrgMemberUpdate(role='superuser'),
                    current_user_id=current_user_id,
                )
            assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
            assert 'Invalid role' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_insufficient_permission_returns_403(
        self, org_id, current_user_id, target_user_id
    ):
        """GIVEN requester lacks permission to change target WHEN PATCH is called THEN returns 403."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.update_org_member'
        ) as mock_update:
            mock_update.side_effect = InsufficientPermissionError(
                'You do not have permission to modify this member'
            )

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await update_org_member(
                    org_id=uuid.UUID(org_id),
                    user_id=target_user_id,
                    update_data=OrgMemberUpdate(role='admin'),
                    current_user_id=current_user_id,
                )
            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
            assert 'do not have permission' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_cannot_demote_last_owner_returns_400(
        self, org_id, current_user_id, target_user_id
    ):
        """GIVEN demoting last owner WHEN PATCH is called THEN returns 400."""
        # Arrange
        with patch(
            'server.routes.orgs.OrgMemberService.update_org_member'
        ) as mock_update:
            mock_update.side_effect = LastOwnerError('demote')

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await update_org_member(
                    org_id=uuid.UUID(org_id),
                    user_id=target_user_id,
                    update_data=OrgMemberUpdate(role='admin'),
                    current_user_id=current_user_id,
                )
            assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
            assert 'Cannot demote the last owner' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_invalid_org_id_returns_422(
        self, mock_app, current_user_id, target_user_id
    ):
        """GIVEN invalid org_id UUID WHEN PATCH is called THEN returns 422.

        Note: FastAPI validates UUID path parameters at the routing level,
        so this test uses the HTTP test client to verify the validation.
        """
        # Arrange
        invalid_org_id = 'not-a-uuid'

        # Act & Assert - Use test client to test route-level validation
        with TestClient(mock_app) as client:
            response = client.patch(
                f'/api/organizations/{invalid_org_id}/members/{target_user_id}',
                json={'role': 'user'},
            )
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestGetMeEndpoint:
    """Tests for GET /api/organizations/{org_id}/me endpoint.

    This endpoint returns the current authenticated user's membership record
    for the specified organization, including role, status, email, and LLM
    override fields (with masked API key).

    Why: The frontend useMe() hook calls this endpoint to determine the user's
    role in the org, which gates read-only mode on settings pages. Without it,
    all role-based access control on settings pages is broken (returns 404).
    """

    @pytest.fixture
    def test_user_id(self):
        """Create a test user ID."""
        return str(uuid.uuid4())

    @pytest.fixture
    def test_org_id(self):
        """Create a test organization ID."""
        return uuid.uuid4()

    @pytest.fixture
    def mock_me_app(self, test_user_id):
        """Create a test FastAPI app with org routes and mocked auth."""
        app = FastAPI()
        app.include_router(org_router)

        def mock_get_user_id():
            return test_user_id

        app.dependency_overrides[get_user_id] = mock_get_user_id
        return app

    def _make_me_response(
        self,
        org_id,
        user_id,
        email='test@example.com',
        role='owner',
        llm_api_key='****2345',
        llm_model='gpt-4',
        llm_base_url='https://api.example.com',
        max_iterations=50,
        llm_api_key_for_byor=None,
        status_val='active',
    ):
        """Create a MeResponse for testing."""
        return MeResponse(
            org_id=str(org_id),
            user_id=str(user_id),
            email=email,
            role=role,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            max_iterations=max_iterations,
            llm_api_key_for_byor=llm_api_key_for_byor,
            status=status_val,
        )

    @pytest.mark.asyncio
    async def test_get_me_success(self, mock_me_app, test_user_id, test_org_id):
        """GIVEN: Authenticated user who is a member of the organization
        WHEN: GET /api/organizations/{org_id}/me is called
        THEN: Returns 200 with the user's membership data including role name and email
        """
        me_response = self._make_me_response(
            org_id=test_org_id,
            user_id=test_user_id,
            email='owner@example.com',
            role='owner',
            llm_model='gpt-4',
            llm_base_url='https://api.example.com',
            max_iterations=50,
            status_val='active',
        )

        with patch(
            'server.routes.orgs.OrgMemberService.get_me',
            return_value=me_response,
        ):
            client = TestClient(mock_me_app)
            response = client.get(f'/api/organizations/{test_org_id}/me')

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data['org_id'] == str(test_org_id)
        assert data['user_id'] == test_user_id
        assert data['email'] == 'owner@example.com'
        assert data['role'] == 'owner'
        assert data['llm_model'] == 'gpt-4'
        assert data['llm_base_url'] == 'https://api.example.com'
        assert data['max_iterations'] == 50
        assert data['status'] == 'active'

    @pytest.mark.asyncio
    async def test_get_me_masks_llm_api_key(
        self, mock_me_app, test_user_id, test_org_id
    ):
        """GIVEN: User is a member with an LLM API key set
        WHEN: GET /api/organizations/{org_id}/me is called
        THEN: The llm_api_key field is masked (not the raw secret value)

        Why: API keys must never be returned in plaintext in API responses.
        The frontend only needs to know if a key is set, not its value.
        """
        me_response = self._make_me_response(
            org_id=test_org_id,
            user_id=test_user_id,
            llm_api_key='****cdef',  # Masked key
        )

        with patch(
            'server.routes.orgs.OrgMemberService.get_me',
            return_value=me_response,
        ):
            client = TestClient(mock_me_app)
            response = client.get(f'/api/organizations/{test_org_id}/me')

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # The raw key must NOT appear in the response
        assert data['llm_api_key'] != 'sk-secret-real-key-abcdef'
        # Should be masked with stars
        assert '**' in data['llm_api_key']

    @pytest.mark.asyncio
    async def test_get_me_not_a_member(self, mock_me_app, test_org_id):
        """GIVEN: Authenticated user who is NOT a member of the organization
        WHEN: GET /api/organizations/{org_id}/me is called
        THEN: Returns 404 (to avoid leaking org existence per spec)
        """
        with patch(
            'server.routes.orgs.OrgMemberService.get_me',
            side_effect=OrgMemberNotFoundError(str(test_org_id), 'user-id'),
        ):
            client = TestClient(mock_me_app)
            response = client.get(f'/api/organizations/{test_org_id}/me')

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_get_me_invalid_uuid(self, mock_me_app):
        """GIVEN: Invalid UUID format for org_id
        WHEN: GET /api/organizations/{org_id}/me is called
        THEN: Returns 422 (FastAPI validates UUID path parameter)
        """
        client = TestClient(mock_me_app)
        response = client.get('/api/organizations/not-a-valid-uuid/me')

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_get_me_unauthenticated(self, test_org_id):
        """GIVEN: User is not authenticated
        WHEN: GET /api/organizations/{org_id}/me is called
        THEN: Returns 401
        """
        app = FastAPI()
        app.include_router(org_router)

        async def mock_unauthenticated():
            raise HTTPException(status_code=401, detail='User not authenticated')

        app.dependency_overrides[get_user_id] = mock_unauthenticated

        client = TestClient(app)
        response = client.get(f'/api/organizations/{test_org_id}/me')

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_get_me_unexpected_error(self, mock_me_app, test_org_id):
        """GIVEN: An unexpected error occurs during membership lookup
        WHEN: GET /api/organizations/{org_id}/me is called
        THEN: Returns 500
        """
        with patch(
            'server.routes.orgs.OrgMemberService.get_me',
            side_effect=RuntimeError('Database connection failed'),
        ):
            client = TestClient(mock_me_app)
            response = client.get(f'/api/organizations/{test_org_id}/me')

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @pytest.mark.asyncio
    async def test_get_me_with_null_optional_fields(
        self, mock_me_app, test_user_id, test_org_id
    ):
        """GIVEN: User is a member with null optional fields (llm_model, llm_base_url, etc.)
        WHEN: GET /api/organizations/{org_id}/me is called
        THEN: Returns 200 with null values for optional fields
        """
        me_response = self._make_me_response(
            org_id=test_org_id,
            user_id=test_user_id,
            llm_model=None,
            llm_base_url=None,
            max_iterations=None,
            llm_api_key='',
        )

        with patch(
            'server.routes.orgs.OrgMemberService.get_me',
            return_value=me_response,
        ):
            client = TestClient(mock_me_app)
            response = client.get(f'/api/organizations/{test_org_id}/me')

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data['llm_model'] is None
        assert data['llm_base_url'] is None
        assert data['max_iterations'] is None

    @pytest.mark.asyncio
    async def test_get_me_with_admin_role(self, mock_me_app, test_user_id, test_org_id):
        """GIVEN: User is an admin member of the organization
        WHEN: GET /api/organizations/{org_id}/me is called
        THEN: Returns correct role name 'admin'

        Why: The frontend uses the role to determine if settings are read-only.
        Admins and owners can edit; members see read-only.
        """
        me_response = self._make_me_response(
            org_id=test_org_id,
            user_id=test_user_id,
            role='admin',
        )

        with patch(
            'server.routes.orgs.OrgMemberService.get_me',
            return_value=me_response,
        ):
            client = TestClient(mock_me_app)
            response = client.get(f'/api/organizations/{test_org_id}/me')

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data['role'] == 'admin'

    @pytest.mark.asyncio
    async def test_get_me_masks_byor_api_key(
        self, mock_me_app, test_user_id, test_org_id
    ):
        """GIVEN: User has an llm_api_key_for_byor set
        WHEN: GET /api/organizations/{org_id}/me is called
        THEN: The llm_api_key_for_byor field is also masked
        """
        me_response = self._make_me_response(
            org_id=test_org_id,
            user_id=test_user_id,
            llm_api_key_for_byor='****-key',  # Masked key
        )

        with patch(
            'server.routes.orgs.OrgMemberService.get_me',
            return_value=me_response,
        ):
            client = TestClient(mock_me_app)
            response = client.get(f'/api/organizations/{test_org_id}/me')

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data['llm_api_key_for_byor'] != 'sk-byor-secret-key'
        assert (
            data['llm_api_key_for_byor'] is None or '**' in data['llm_api_key_for_byor']
        )

    @pytest.mark.asyncio
    async def test_get_me_role_not_found_returns_500(self, mock_me_app, test_org_id):
        """GIVEN: Role lookup fails (data integrity issue)
        WHEN: GET /api/organizations/{org_id}/me is called
        THEN: Returns 500 Internal Server Error
        """
        with patch(
            'server.routes.orgs.OrgMemberService.get_me',
            side_effect=RoleNotFoundError(role_id=999),
        ):
            client = TestClient(mock_me_app)
            response = client.get(f'/api/organizations/{test_org_id}/me')

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert 'unexpected error' in response.json()['detail'].lower()

    @pytest.mark.asyncio
    async def test_get_me_direct_function_call_success(self, test_user_id, test_org_id):
        """Test direct function call to get_me returns MeResponse."""
        me_response = self._make_me_response(
            org_id=test_org_id,
            user_id=test_user_id,
            email='test@example.com',
            role='owner',
        )

        with patch(
            'server.routes.orgs.OrgMemberService.get_me',
            return_value=me_response,
        ):
            result = await get_me(org_id=test_org_id, user_id=test_user_id)

        assert isinstance(result, MeResponse)
        assert result.org_id == str(test_org_id)
        assert result.user_id == test_user_id
        assert result.role == 'owner'

    @pytest.mark.asyncio
    async def test_get_me_direct_function_call_member_not_found(
        self, test_user_id, test_org_id
    ):
        """Test direct function call to get_me raises HTTPException on member not found."""
        with patch(
            'server.routes.orgs.OrgMemberService.get_me',
            side_effect=OrgMemberNotFoundError(str(test_org_id), test_user_id),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_me(org_id=test_org_id, user_id=test_user_id)

        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert str(test_org_id) in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_me_direct_function_call_role_not_found(
        self, test_user_id, test_org_id
    ):
        """Test direct function call to get_me raises HTTPException on role not found."""
        with patch(
            'server.routes.orgs.OrgMemberService.get_me',
            side_effect=RoleNotFoundError(role_id=999),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_me(org_id=test_org_id, user_id=test_user_id)

        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
async def test_switch_org_success(mock_app_with_get_user_id):
    """
    GIVEN: Valid org_id and authenticated user who is a member
    WHEN: POST /api/organizations/{org_id}/switch is called
    THEN: User's current org is switched and org details returned with 200 status
    """
    # Arrange
    org_id = uuid.uuid4()
    mock_org = Org(
        id=org_id,
        name='Target Organization',
        contact_name='John Doe',
        contact_email='john@example.com',
        org_version=5,
        default_llm_model='claude-opus-4-5-20251101',
    )

    with (
        patch(
            'server.routes.orgs.OrgService.switch_org',
            AsyncMock(return_value=mock_org),
        ),
        patch(
            'server.routes.orgs.OrgService.get_org_credits',
            AsyncMock(return_value=100.0),
        ),
    ):
        client = TestClient(mock_app_with_get_user_id)

        # Act
        response = client.post(f'/api/organizations/{org_id}/switch')

        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data['id'] == str(org_id)
        assert response_data['name'] == 'Target Organization'
        assert response_data['credits'] == 100.0


@pytest.mark.asyncio
async def test_switch_org_not_member(mock_app_with_get_user_id):
    """
    GIVEN: User is not a member of the target organization
    WHEN: POST /api/organizations/{org_id}/switch is called
    THEN: 403 Forbidden error is returned
    """
    # Arrange
    org_id = uuid.uuid4()

    with patch(
        'server.routes.orgs.OrgService.switch_org',
        AsyncMock(
            side_effect=OrgAuthorizationError(
                'User must be a member of the organization to switch to it'
            )
        ),
    ):
        client = TestClient(mock_app_with_get_user_id)

        # Act
        response = client.post(f'/api/organizations/{org_id}/switch')

        # Assert
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert 'member' in response.json()['detail'].lower()


@pytest.mark.asyncio
async def test_switch_org_not_found(mock_app_with_get_user_id):
    """
    GIVEN: Organization does not exist
    WHEN: POST /api/organizations/{org_id}/switch is called
    THEN: 404 Not Found error is returned
    """
    # Arrange
    org_id = uuid.uuid4()

    with patch(
        'server.routes.orgs.OrgService.switch_org',
        AsyncMock(side_effect=OrgNotFoundError(str(org_id))),
    ):
        client = TestClient(mock_app_with_get_user_id)

        # Act
        response = client.post(f'/api/organizations/{org_id}/switch')

        # Assert
        assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_switch_org_invalid_uuid(mock_app_with_get_user_id):
    """
    GIVEN: Invalid UUID format for org_id
    WHEN: POST /api/organizations/{org_id}/switch is called
    THEN: 422 Unprocessable Entity error is returned
    """
    # Arrange
    client = TestClient(mock_app_with_get_user_id)

    # Act
    response = client.post('/api/organizations/not-a-valid-uuid/switch')

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_switch_org_database_error(mock_app_with_get_user_id):
    """
    GIVEN: Database operation fails during switch
    WHEN: POST /api/organizations/{org_id}/switch is called
    THEN: 500 Internal Server Error is returned
    """
    # Arrange
    org_id = uuid.uuid4()

    with patch(
        'server.routes.orgs.OrgService.switch_org',
        AsyncMock(side_effect=OrgDatabaseError('Database connection failed')),
    ):
        client = TestClient(mock_app_with_get_user_id)

        # Act
        response = client.post(f'/api/organizations/{org_id}/switch')

        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert 'Failed to switch organization' in response.json()['detail']
