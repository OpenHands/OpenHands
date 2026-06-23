import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import HTTPException, Request, status
from fastapi.responses import RedirectResponse
from integrations.jira_dc.jira_dc_service_account import JiraDcServiceAccount
from integrations.jira_dc.jira_dc_user_token import JiraDcUserToken
from pydantic import SecretStr, ValidationError
from server.auth.saas_user_auth import SaasUserAuth
from server.routes.integration.jira_dc import (
    JiraDcLinkCreate,
    JiraDcWorkspaceCreate,
    _handle_workspace_link_creation,
    _validate_workspace_update_permissions,
    create_jira_dc_workspace,
    create_workspace_link,
    get_current_workspace_link,
    get_jira_dc_secret_token,
    jira_dc_callback,
    jira_dc_connection_events,
    unlink_workspace,
    validate_workspace_integration,
)


def create_httpx_mock(post_response=None, get_response=None, get_responses=None):
    """Create a mock for httpx.AsyncClient.

    Args:
        post_response: Response for POST requests. Can be a dict (JSON body) or
            a MagicMock for full control.
        get_response: Default response for GET requests. Can be a dict (JSON body) or
            a MagicMock for full control.
        get_responses: Dict of URL pattern -> response for conditional GET responses.
            e.g. {'accessible-resources': {...}, '/myself': {...}}

    Returns:
        AsyncMock that can be used as httpx.AsyncClient context manager.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'access_token': 'token'}

    if isinstance(post_response, dict):
        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = post_response
    else:
        mock_post_response = post_response or mock_response

    # Create mock client with post/get methods
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_post_response)
    mock_client.get = AsyncMock(return_value=mock_response)

    # Set up conditional GET responses if provided
    if get_responses:

        async def mock_get(url, **kwargs):
            for pattern, resp in get_responses.items():
                if pattern in url:
                    mock_resp = MagicMock()
                    mock_resp.status_code = 200
                    if isinstance(resp, dict):
                        mock_resp.json.return_value = resp
                    else:
                        mock_resp = resp
                    return mock_resp
            # Default response for unmatched URLs
            default_resp = MagicMock()
            default_resp.status_code = 404
            default_resp.json.return_value = {}
            return default_resp

        mock_client.get = AsyncMock(side_effect=mock_get)
    elif isinstance(get_response, dict):
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = get_response
        mock_client.get = AsyncMock(return_value=mock_get_response)

    # Create the AsyncClient context manager mock
    @asynccontextmanager
    async def mock_async_client():
        yield mock_client

    mock_httpx = AsyncMock()
    mock_httpx.__aenter__ = AsyncMock(return_value=mock_client)
    mock_httpx.__aexit__ = AsyncMock(return_value=None)
    return mock_httpx


@pytest.fixture
def mock_httpx_client():
    """Fixture to provide httpx.AsyncClient mock."""
    return create_httpx_mock


@pytest.fixture
def mock_request():
    req = MagicMock(spec=Request)
    req.headers = {}
    req.cookies = {}
    req.app.state.redis = MagicMock()
    return req


@pytest.fixture
def mock_jira_dc_manager():
    manager = MagicMock()
    manager.integration_store = AsyncMock()
    manager.validate_request = AsyncMock()
    manager.validate_request_context = AsyncMock()
    return manager


@pytest.fixture
def mock_token_manager():
    return MagicMock()


@pytest.fixture
def mock_redis_client():
    client = MagicMock()
    client.exists.return_value = False
    client.setex.return_value = True
    return client


@pytest.fixture
def mock_user_auth():
    auth = AsyncMock(spec=SaasUserAuth)
    auth.get_user_id.return_value = 'test_user_id'
    auth.get_user_email.return_value = 'test@example.com'
    auth.get_effective_org_id.return_value = uuid.UUID(
        '00000000-0000-0000-0000-000000000123'
    )
    return auth


@pytest.mark.asyncio
@patch(
    'server.routes.integration.jira_dc.get_user_jira_dc_token', new_callable=AsyncMock
)
@patch('server.routes.integration.jira_dc.jira_dc_manager')
async def test_get_jira_dc_secret_token_success(mock_manager, mock_get_token):
    store = AsyncMock()
    mock_manager.integration_store = store
    store.get_workspace_by_id.return_value = SimpleNamespace(
        id=7,
        name='jira.example.com',
        status='active',
    )
    store.get_active_user_by_keycloak_id_and_workspace.return_value = MagicMock()
    mock_get_token.return_value = JiraDcUserToken(
        access_token=SecretStr('plain-token'), expires_at=0
    )
    jwt_service = MagicMock()
    jwt_service.verify_jws_token.return_value = {
        'integration': 'jira_dc',
        'secret_name': 'JIRA_DC_TOKEN',
        'user_id': 'kc-user',
        'workspace_id': 7,
    }

    with (
        patch('server.routes.integration.jira_dc.JIRA_DC_ENABLE_OAUTH', True),
        patch(
            'server.routes.integration.jira_dc.JIRA_DC_BASE_URL',
            'https://jira.example.com',
        ),
    ):
        response = await get_jira_dc_secret_token(
            access_token='signed-token',
            jwt_service=jwt_service,
        )

    assert response.body == b'plain-token'
    store.get_workspace_by_id.assert_awaited_once_with(7)
    store.get_active_user_by_keycloak_id_and_workspace.assert_awaited_once_with(
        'kc-user',
        7,
    )
    mock_get_token.assert_awaited_once()


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch('server.routes.integration.jira_dc.redis_client', new_callable=MagicMock)
async def test_jira_dc_events_invalid_signature(mock_redis, mock_manager, mock_request):
    with patch('server.routes.integration.jira_dc.JIRA_DC_WEBHOOKS_ENABLED', True):
        mock_manager.validate_request_context.return_value = (False, None, None, None)
        with pytest.raises(HTTPException) as exc_info:
            await jira_dc_connection_events(10, mock_request, MagicMock())
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == 'Invalid webhook signature!'


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch('server.routes.integration.jira_dc.redis_client')
async def test_jira_dc_events_duplicate_request(mock_redis, mock_manager, mock_request):
    with patch('server.routes.integration.jira_dc.JIRA_DC_WEBHOOKS_ENABLED', True):
        mock_workspace = MagicMock()
        mock_workspace.id = 10
        mock_manager.validate_request_context.return_value = (
            True,
            'sig123',
            'payload',
            mock_workspace,
        )
        mock_redis.exists.return_value = True
        response = await jira_dc_connection_events(10, mock_request, MagicMock())
        assert response.status_code == 200
        body = json.loads(response.body)
        assert body['success'] is True


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.redis_client')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch('server.routes.integration.jira_dc.JIRA_DC_ENABLE_OAUTH', True)
async def test_create_jira_dc_workspace_oauth_success(
    mock_manager, mock_redis, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    mock_manager.integration_store.get_workspace_by_name.return_value = None
    mock_redis.setex.return_value = True
    workspace_data = JiraDcWorkspaceCreate(
        workspace_name='test-workspace',
        webhook_secret='secret',
        svc_acc_email='svc@test.com',
        svc_acc_api_key='key',
        is_active=True,
    )

    response = await create_jira_dc_workspace(mock_request, workspace_data)
    content = json.loads(response.body)

    assert response.status_code == 200
    assert content['success'] is True
    assert content['redirect'] is True
    assert 'authorizationUrl' in content
    mock_redis.setex.assert_called_once()


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.redis_client')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch('server.routes.integration.jira_dc.JIRA_DC_ENABLE_OAUTH', True)
async def test_create_jira_dc_workspace_oauth_url_uses_jira_dc_write_scope(
    mock_manager, mock_redis, mock_get_auth, mock_request, mock_user_auth
):
    """OAuth authorization URL must request the Jira DC `WRITE` scope.

    Jira DC OAuth 2.0 uses coarse scopes (READ/WRITE/ADMIN/SYSTEM_ADMIN). Sending
    Atlassian Cloud-style scopes such as `read:me read:jira-user read:jira-work`
    is rejected with `invalid_scope`, which breaks the consent flow.
    """
    # Arrange
    mock_get_auth.return_value = mock_user_auth
    mock_manager.integration_store.get_workspace_by_name.return_value = None
    mock_redis.setex.return_value = True
    workspace_data = JiraDcWorkspaceCreate(
        workspace_name='test-workspace',
        webhook_secret='secret',
        svc_acc_email='svc@test.com',
        svc_acc_api_key='key',
        is_active=True,
    )

    # Act
    response = await create_jira_dc_workspace(mock_request, workspace_data)
    content = json.loads(response.body)

    # Assert
    query = parse_qs(urlparse(content['authorizationUrl']).query)
    assert query['scope'] == ['WRITE']


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.redis_client')
@patch('server.routes.integration.jira_dc.JIRA_DC_ENABLE_OAUTH', True)
async def test_create_workspace_link_oauth_success(
    mock_redis, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    mock_redis.setex.return_value = True
    link_data = JiraDcLinkCreate(workspace_name='test-workspace')

    response = await create_workspace_link(mock_request, link_data)
    content = json.loads(response.body)

    assert response.status_code == 200
    assert content['success'] is True
    assert content['redirect'] is True
    assert 'authorizationUrl' in content
    mock_redis.setex.assert_called_once()


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.redis_client')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch(
    'server.routes.integration.jira_dc._handle_workspace_link_creation',
    new_callable=AsyncMock,
)
async def test_jira_dc_callback_workspace_integration_new_workspace(
    mock_handle_link, mock_manager, mock_redis, mock_request
):
    state = 'test_state'
    code = 'test_code'
    session_data = {
        'operation_type': 'workspace_integration',
        'keycloak_user_id': 'user1',
        'target_workspace': 'test.atlassian.net',
        'webhook_secret': 'secret',
        'svc_acc_email': 'email@test.com',
        'svc_acc_api_key': 'apikey',
        'is_active': True,
        'state': state,
    }
    mock_redis.get.return_value = json.dumps(session_data)

    # Set up httpx mock with conditional GET responses
    httpx_mock = create_httpx_mock(
        post_response={'access_token': 'token'},
        get_responses={
            'accessible-resources': [{'url': 'https://test.atlassian.net'}],
            '/myself': {'key': 'jira_user_123'},
        },
    )

    mock_manager.integration_store.get_workspace_by_name.return_value = None
    mock_workspace = MagicMock(id=1)
    mock_manager.integration_store.create_workspace.return_value = mock_workspace

    with patch('server.routes.integration.jira_dc.token_manager') as mock_token_manager:
        with patch(
            'server.routes.integration.jira_dc.JIRA_DC_BASE_URL',
            'https://test.atlassian.net',
        ):
            with patch('httpx.AsyncClient', return_value=httpx_mock):
                mock_token_manager.encrypt_text.side_effect = lambda x: f'enc_{x}'
                response = await jira_dc_callback(mock_request, code, state)

                assert isinstance(response, RedirectResponse)
                assert response.status_code == status.HTTP_302_FOUND
                mock_manager.integration_store.create_workspace.assert_called_once()
                mock_handle_link.assert_called_once_with(
                    'user1',
                    'jira_user_123',
                    'test.atlassian.net',
                    replace_stale_active_link=True,
                )


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.redis_client')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch(
    'server.routes.integration.jira_dc._handle_workspace_link_creation',
    new_callable=AsyncMock,
)
@patch(
    'server.routes.integration.jira_dc._maybe_register_webhook',
    new_callable=AsyncMock,
)
async def test_jira_dc_callback_redirects_with_webhook_install_failure(
    mock_register_webhook,
    mock_handle_link,
    mock_manager,
    mock_redis,
    mock_request,
):
    state = 'test_state'
    session_data = {
        'operation_type': 'workspace_integration',
        'keycloak_user_id': 'user1',
        'target_workspace': 'test.atlassian.net',
        'webhook_secret': 'secret',
        'svc_acc_email': 'email@test.com',
        'svc_acc_api_key': 'apikey',
        'admin_api_key': 'bad-admin-pat',
        'is_active': True,
        'state': state,
    }
    mock_redis.get.return_value = json.dumps(session_data)

    # Set up httpx mock
    httpx_mock = create_httpx_mock(
        post_response={'access_token': 'token'},
        get_response={'key': 'jira_user_123'},
    )

    mock_manager.integration_store.get_workspace_by_name.return_value = None
    mock_workspace = MagicMock(id=1)
    mock_manager.integration_store.create_workspace.return_value = mock_workspace
    mock_register_webhook.return_value = False

    with patch('server.routes.integration.jira_dc.token_manager') as mock_token_manager:
        with patch(
            'server.routes.integration.jira_dc.JIRA_DC_BASE_URL',
            'https://test.atlassian.net',
        ):
            with patch('httpx.AsyncClient', return_value=httpx_mock):
                mock_token_manager.encrypt_text.side_effect = lambda x: f'enc_{x}'
                response = await jira_dc_callback(mock_request, 'code', state)

    assert isinstance(response, RedirectResponse)
    assert response.status_code == status.HTTP_302_FOUND
    assert response.headers['location'] == (
        '/settings/integrations?jira_dc_webhook=install_failed'
    )
    mock_register_webhook.assert_awaited_once_with(
        'bad-admin-pat', 'https://test.atlassian.net', 'secret', 1
    )
    mock_handle_link.assert_called_once_with(
        'user1',
        'jira_user_123',
        'test.atlassian.net',
        replace_stale_active_link=True,
    )


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.redis_client')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch(
    'server.routes.integration.jira_dc._handle_workspace_link_creation',
    new_callable=AsyncMock,
)
async def test_jira_dc_callback_token_exchange_uses_form_encoding(
    mock_handle_link, mock_manager, mock_redis, mock_request
):
    """Token exchange must POST `application/x-www-form-urlencoded`, not JSON.

    Jira DC's `/rest/oauth2/latest/token` endpoint is JAX-RS with @FormParam, which
    only populates parameters when the body is form-encoded. Sending JSON returns
    a 500 ("@FormParam is utilized when the content type ... is not
    application/x-www-form-urlencoded").
    """
    # Arrange
    state = 'test_state'
    code = 'test_code'
    session_data = {
        'operation_type': 'workspace_integration',
        'keycloak_user_id': 'user1',
        'target_workspace': 'test.atlassian.net',
        'webhook_secret': 'secret',
        'svc_acc_email': 'email@test.com',
        'svc_acc_api_key': 'apikey',
        'is_active': True,
        'state': state,
    }
    mock_redis.get.return_value = json.dumps(session_data)
    mock_manager.integration_store.get_workspace_by_name.return_value = None

    # Create a mock that captures POST call arguments
    mock_post_response = MagicMock()
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = {'access_token': 'token'}
    mock_post_response.text = ''

    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = {'key': 'jira_user_123'}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_post_response)
    mock_client.get = AsyncMock(return_value=mock_get_response)

    mock_httpx = AsyncMock()
    mock_httpx.__aenter__ = AsyncMock(return_value=mock_client)
    mock_httpx.__aexit__ = AsyncMock(return_value=None)

    with patch('server.routes.integration.jira_dc.token_manager') as mock_token_manager:
        with patch(
            'server.routes.integration.jira_dc.JIRA_DC_BASE_URL',
            'https://test.atlassian.net',
        ):
            with patch('httpx.AsyncClient', return_value=mock_httpx):
                mock_token_manager.encrypt_text.side_effect = lambda x: f'enc_{x}'

                # Act
                await jira_dc_callback(mock_request, code, state)

    # Assert: token POST is form-encoded (data=), not JSON (json=).
    kwargs = mock_client.post.call_args.kwargs
    assert kwargs.get('data', {}).get('grant_type') == 'authorization_code'
    assert 'json' not in kwargs


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_get_current_workspace_link_found(
    mock_manager, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    user_id = 'test_user_id'

    mock_user_created_at = datetime.now()
    mock_user_updated_at = datetime.now()
    mock_user = MagicMock(
        id=1,
        keycloak_user_id=user_id,
        jira_dc_workspace_id=10,
        status='active',
    )
    mock_user.created_at = mock_user_created_at
    mock_user.updated_at = mock_user_updated_at

    mock_workspace_created_at = datetime.now()
    mock_workspace_updated_at = datetime.now()
    mock_workspace = MagicMock(
        id=10,
        status='active',
        admin_user_id=user_id,
        svc_acc_email='svc@test.com',
    )
    mock_workspace.name = 'test-space'
    mock_workspace.svc_acc_email = 'svc@test.com'
    mock_workspace.created_at = mock_workspace_created_at
    mock_workspace.updated_at = mock_workspace_updated_at

    mock_manager.integration_store.get_user_by_active_workspace.return_value = mock_user
    mock_manager.integration_store.get_workspace_by_id.return_value = mock_workspace

    response = await get_current_workspace_link(mock_request)
    assert response.workspace.name == 'test-space'
    assert response.workspace.editable is True
    assert response.workspace.events_url.endswith(
        '/integration/jira-dc/connections/10/events'
    )


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch('server.routes.integration.jira_dc.JIRA_DC_ENABLE_OAUTH', True)
@patch(
    'server.routes.integration.jira_dc.JIRA_DC_BASE_URL', 'https://current-jira.test'
)
async def test_get_current_workspace_link_ignores_stale_configured_host_link(
    mock_manager, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    mock_user = MagicMock(
        id=1,
        keycloak_user_id='test_user_id',
        jira_dc_workspace_id=10,
        status='active',
    )
    mock_user.created_at = datetime.now()
    mock_user.updated_at = datetime.now()

    mock_workspace = MagicMock(
        id=10,
        status='active',
        admin_user_id='test_user_id',
        svc_acc_email='svc@test.com',
    )
    mock_workspace.name = 'old-jira.test'
    mock_workspace.created_at = datetime.now()
    mock_workspace.updated_at = datetime.now()

    mock_manager.integration_store.get_user_by_active_workspace.return_value = mock_user
    mock_manager.integration_store.get_workspace_by_id.return_value = mock_workspace

    with pytest.raises(HTTPException) as exc_info:
        await get_current_workspace_link(mock_request)

    assert exc_info.value.status_code == 404
    assert 'configured Jira DC integration' in exc_info.value.detail


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_unlink_workspace_admin(
    mock_manager, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    user_id = 'test_user_id'
    mock_user = MagicMock(jira_dc_workspace_id=10)
    mock_workspace = MagicMock(id=10, admin_user_id=user_id)
    mock_manager.integration_store.get_user_by_active_workspace.return_value = mock_user
    mock_manager.integration_store.get_workspace_by_id.return_value = mock_workspace

    response = await unlink_workspace(mock_request)
    content = json.loads(response.body)
    assert content['success'] is True
    mock_manager.integration_store.deactivate_workspace.assert_called_once_with(
        workspace_id=10
    )


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch(
    'server.routes.integration.jira_dc._maybe_delete_webhook',
    new_callable=AsyncMock,
)
@patch('server.routes.integration.jira_dc.JIRA_DC_ENABLE_OAUTH', False)
async def test_unlink_workspace_admin_reports_webhook_delete_failure(
    mock_delete_webhook, mock_manager, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    user_id = 'test_user_id'
    mock_request.json = AsyncMock(return_value={'admin_api_key': 'bad-admin-pat'})
    mock_user = MagicMock(jira_dc_workspace_id=10)
    mock_workspace = MagicMock(id=10, admin_user_id=user_id)
    mock_workspace.name = 'test-workspace'
    mock_manager.integration_store.get_user_by_active_workspace.return_value = mock_user
    mock_manager.integration_store.get_workspace_by_id.return_value = mock_workspace
    mock_delete_webhook.return_value = False

    response = await unlink_workspace(mock_request)
    content = json.loads(response.body)

    assert content == {'success': True, 'webhookRemoved': False}
    mock_delete_webhook.assert_awaited_once_with(
        'bad-admin-pat', 'https://test-workspace', 10
    )
    mock_manager.integration_store.deactivate_workspace.assert_called_once_with(
        workspace_id=10
    )


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_validate_workspace_integration_success(
    mock_manager, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    workspace_name = 'active-workspace'
    mock_workspace = MagicMock(status='active')
    mock_workspace.name = workspace_name
    mock_manager.integration_store.get_workspace_by_name.return_value = mock_workspace

    response = await validate_workspace_integration(mock_request, workspace_name)
    assert response.name == workspace_name
    assert response.status == 'active'
    assert response.message == 'Workspace integration is active'


# Additional comprehensive tests for better coverage


# Test Pydantic Model Validations
class TestJiraDcWorkspaceCreateValidation:
    def test_valid_workspace_create(self):
        data = JiraDcWorkspaceCreate(
            workspace_name='test-workspace',
            webhook_secret='secret123',
            svc_acc_email='test@example.com',
            svc_acc_api_key='api_key_123',
            is_active=True,
        )
        assert data.workspace_name == 'test-workspace'
        assert data.svc_acc_email == 'test@example.com'

    def test_invalid_workspace_name(self):
        with pytest.raises(ValidationError) as exc_info:
            JiraDcWorkspaceCreate(
                workspace_name='test workspace!',  # Contains space and special char
                webhook_secret='secret123',
                svc_acc_email='test@example.com',
                svc_acc_api_key='api_key_123',
            )
        assert 'workspace_name can only contain alphanumeric characters' in str(
            exc_info.value
        )

    def test_invalid_email(self):
        with pytest.raises(ValidationError) as exc_info:
            JiraDcWorkspaceCreate(
                workspace_name='test-workspace',
                webhook_secret='secret123',
                svc_acc_email='invalid-email',
                svc_acc_api_key='api_key_123',
            )
        assert 'svc_acc_email must be a valid email address' in str(exc_info.value)

    def test_webhook_secret_with_spaces(self):
        with pytest.raises(ValidationError) as exc_info:
            JiraDcWorkspaceCreate(
                workspace_name='test-workspace',
                webhook_secret='secret with spaces',
                svc_acc_email='test@example.com',
                svc_acc_api_key='api_key_123',
            )
        assert 'webhook_secret cannot contain spaces' in str(exc_info.value)

    def test_api_key_with_spaces(self):
        with pytest.raises(ValidationError) as exc_info:
            JiraDcWorkspaceCreate(
                workspace_name='test-workspace',
                webhook_secret='secret123',
                svc_acc_email='test@example.com',
                svc_acc_api_key='api key with spaces',
            )
        assert 'svc_acc_api_key cannot contain spaces' in str(exc_info.value)


class TestJiraDcLinkCreateValidation:
    def test_valid_link_create(self):
        data = JiraDcLinkCreate(workspace_name='test-workspace')
        assert data.workspace_name == 'test-workspace'

    def test_invalid_workspace_name(self):
        with pytest.raises(ValidationError) as exc_info:
            JiraDcLinkCreate(workspace_name='invalid workspace!')
        assert 'workspace can only contain alphanumeric characters' in str(
            exc_info.value
        )


# Test jira_dc_events error scenarios
@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch('server.routes.integration.jira_dc.redis_client', new_callable=MagicMock)
async def test_jira_dc_events_processing_success(
    mock_redis, mock_manager, mock_request
):
    with patch('server.routes.integration.jira_dc.JIRA_DC_WEBHOOKS_ENABLED', True):
        mock_workspace = MagicMock()
        mock_workspace.id = 10
        mock_workspace.org_id = uuid.UUID('00000000-0000-0000-0000-000000000123')
        mock_workspace.name = 'jira.company.com'
        mock_manager.validate_request_context.return_value = (
            True,
            'sig123',
            {'test': 'payload'},
            mock_workspace,
        )
        mock_redis.exists.return_value = False

        background_tasks = MagicMock()
        response = await jira_dc_connection_events(10, mock_request, background_tasks)

        assert response.status_code == 200
        body = json.loads(response.body)
        assert body['success'] is True
        mock_redis.setex.assert_called_once_with('jira_dc:10:sig123', 120, 1)
        background_tasks.add_task.assert_called_once()


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch('server.routes.integration.jira_dc.redis_client', new_callable=MagicMock)
async def test_jira_dc_connection_events_validates_workspace_id(
    mock_redis, mock_manager, mock_request
):
    with patch('server.routes.integration.jira_dc.JIRA_DC_WEBHOOKS_ENABLED', True):
        mock_workspace = MagicMock()
        mock_workspace.id = 42
        mock_workspace.org_id = None
        mock_workspace.name = 'jira.company.com'
        mock_manager.validate_request_context.return_value = (
            True,
            'sig123',
            {'test': 'payload'},
            mock_workspace,
        )
        mock_redis.exists.return_value = False

        response = await jira_dc_connection_events(42, mock_request, MagicMock())

        assert response.status_code == 200
        body = json.loads(response.body)
        assert body['success'] is True
        mock_manager.validate_request_context.assert_awaited_once_with(
            mock_request,
            workspace_id=42,
        )
        mock_redis.setex.assert_called_once_with('jira_dc:42:sig123', 120, 1)


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.automation_event_service')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch('server.routes.integration.jira_dc.redis_client', new_callable=MagicMock)
async def test_jira_dc_events_forwards_to_automations(
    mock_redis, mock_manager, mock_automation_service, mock_request
):
    org_id = uuid.UUID('00000000-0000-0000-0000-000000000123')
    payload = {'webhookEvent': 'comment_created'}
    mock_workspace = MagicMock()
    mock_workspace.id = 10
    mock_workspace.org_id = org_id
    mock_workspace.name = 'jira.company.com'
    mock_manager.validate_request_context.return_value = (
        True,
        'sig123',
        payload,
        mock_workspace,
    )
    mock_redis.exists.return_value = False

    with (
        patch('server.routes.integration.jira_dc.JIRA_DC_WEBHOOKS_ENABLED', True),
        patch(
            'server.routes.integration.jira_dc.AUTOMATION_EVENT_FORWARDING_ENABLED',
            True,
        ),
    ):
        background_tasks = MagicMock()
        response = await jira_dc_connection_events(10, mock_request, background_tasks)

    assert response.status_code == 200
    background_tasks.add_task.assert_any_call(
        mock_automation_service.forward_jira_dc_event,
        org_id=org_id,
        payload=payload,
        workspace_name='jira.company.com',
        connection_id=10,
        delivery_id='sig123',
    )
    assert background_tasks.add_task.call_count == 2


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.automation_event_service')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch('server.routes.integration.jira_dc.redis_client', new_callable=MagicMock)
async def test_jira_dc_events_forwards_issue_created_to_automations(
    mock_redis, mock_manager, mock_automation_service, mock_request
):
    org_id = uuid.UUID('00000000-0000-0000-0000-000000000123')
    payload = {
        'webhookEvent': 'jira:issue_created',
        'issue': {'key': 'PROJ-123'},
    }
    mock_workspace = MagicMock()
    mock_workspace.id = 10
    mock_workspace.org_id = org_id
    mock_workspace.name = 'jira.company.com'
    mock_manager.validate_request_context.return_value = (
        True,
        'sig123',
        payload,
        mock_workspace,
    )
    mock_redis.exists.return_value = False

    with (
        patch('server.routes.integration.jira_dc.JIRA_DC_WEBHOOKS_ENABLED', True),
        patch(
            'server.routes.integration.jira_dc.AUTOMATION_EVENT_FORWARDING_ENABLED',
            True,
        ),
    ):
        background_tasks = MagicMock()
        response = await jira_dc_connection_events(10, mock_request, background_tasks)

    assert response.status_code == 200
    background_tasks.add_task.assert_any_call(
        mock_automation_service.forward_jira_dc_event,
        org_id=org_id,
        payload=payload,
        workspace_name='jira.company.com',
        connection_id=10,
        delivery_id='sig123',
    )
    assert background_tasks.add_task.call_count == 2


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch('server.routes.integration.jira_dc.redis_client', new_callable=MagicMock)
async def test_jira_dc_events_general_exception(mock_redis, mock_manager, mock_request):
    with patch('server.routes.integration.jira_dc.JIRA_DC_WEBHOOKS_ENABLED', True):
        mock_manager.validate_request_context.side_effect = Exception(
            'Unexpected error'
        )

        response = await jira_dc_connection_events(10, mock_request, MagicMock())

        assert response.status_code == 500
        body = json.loads(response.body)
        assert 'Internal server error processing webhook' in body['error']


# Test create_jira_dc_workspace with OAuth disabled
@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch('server.routes.integration.jira_dc.JIRA_DC_ENABLE_OAUTH', False)
@patch(
    'server.routes.integration.jira_dc._handle_workspace_link_creation',
    new_callable=AsyncMock,
)
async def test_create_jira_dc_workspace_oauth_disabled_new_workspace(
    mock_handle_link, mock_manager, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    mock_manager.integration_store.get_workspace_by_name.return_value = None
    mock_workspace = MagicMock(name='test-workspace')
    mock_workspace.id = 10
    mock_workspace.name = 'test-workspace'
    mock_manager.integration_store.create_workspace.return_value = mock_workspace

    workspace_data = JiraDcWorkspaceCreate(
        workspace_name='test-workspace',
        webhook_secret='secret',
        svc_acc_email='svc@test.com',
        svc_acc_api_key='key',
        is_active=True,
    )

    with patch('server.routes.integration.jira_dc.token_manager') as mock_token_manager:
        mock_token_manager.encrypt_text.side_effect = lambda x: f'enc_{x}'

        response = await create_jira_dc_workspace(mock_request, workspace_data)
        content = json.loads(response.body)

        assert response.status_code == 200
        assert content['success'] is True
        assert content['redirect'] is False
        assert content['authorizationUrl'] == ''
        assert content['eventsUrl'].endswith(
            '/integration/jira-dc/connections/10/events'
        )
        mock_manager.integration_store.create_workspace.assert_called_once()
        mock_handle_link.assert_called_once()


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch('server.routes.integration.jira_dc.JIRA_DC_ENABLE_OAUTH', False)
@patch(
    'server.routes.integration.jira_dc._handle_workspace_link_creation',
    new_callable=AsyncMock,
)
async def test_create_jira_dc_workspace_manual_setup_starts_inactive(
    mock_handle_link, mock_manager, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    mock_manager.integration_store.get_workspace_by_name.return_value = None
    mock_workspace = MagicMock(name='test-workspace')
    mock_workspace.id = 10
    mock_workspace.name = 'test-workspace'
    mock_manager.integration_store.create_workspace.return_value = mock_workspace

    workspace_data = JiraDcWorkspaceCreate(
        workspace_name='test-workspace',
        webhook_secret='secret',
        svc_acc_email='svc@test.com',
        svc_acc_api_key='key',
        is_active=False,
    )

    with patch('server.routes.integration.jira_dc.token_manager') as mock_token_manager:
        mock_token_manager.encrypt_text.side_effect = lambda x: f'enc_{x}'

        response = await create_jira_dc_workspace(mock_request, workspace_data)

    content = json.loads(response.body)
    assert response.status_code == 200
    assert content['success'] is True
    assert content['webhookEnrolled'] is False
    assert content['eventsUrl'].endswith('/integration/jira-dc/connections/10/events')

    create_kwargs = mock_manager.integration_store.create_workspace.call_args.kwargs
    assert create_kwargs['status'] == 'inactive'
    mock_handle_link.assert_awaited_once_with(
        'test_user_id',
        'unavailable',
        'test-workspace',
        require_active_workspace=False,
    )


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch('server.routes.integration.jira_dc.JIRA_DC_ENABLE_OAUTH', False)
@patch(
    'server.routes.integration.jira_dc._handle_workspace_link_creation',
    new_callable=AsyncMock,
)
@patch(
    'server.routes.integration.jira_dc._maybe_register_webhook',
    new_callable=AsyncMock,
)
async def test_create_jira_dc_workspace_reports_webhook_install_failure(
    mock_register_webhook,
    mock_handle_link,
    mock_manager,
    mock_get_auth,
    mock_request,
    mock_user_auth,
):
    mock_get_auth.return_value = mock_user_auth
    mock_manager.integration_store.get_workspace_by_name.return_value = None
    mock_workspace = MagicMock(id=10)
    mock_workspace.name = 'test-workspace'
    mock_manager.integration_store.create_workspace.return_value = mock_workspace
    mock_register_webhook.return_value = False

    workspace_data = JiraDcWorkspaceCreate(
        workspace_name='test-workspace',
        webhook_secret='secret',
        svc_acc_email='svc@test.com',
        svc_acc_api_key='key',
        admin_api_key='bad-admin-pat',
        is_active=True,
    )

    with patch('server.routes.integration.jira_dc.token_manager') as mock_token_manager:
        mock_token_manager.encrypt_text.side_effect = lambda x: f'enc_{x}'

        response = await create_jira_dc_workspace(mock_request, workspace_data)

    content = json.loads(response.body)
    assert content['success'] is True
    assert content['webhookEnrolled'] is False
    assert content['eventsUrl'].endswith('/integration/jira-dc/connections/10/events')
    mock_register_webhook.assert_awaited_once_with(
        'bad-admin-pat', 'https://test-workspace', 'secret', 10
    )
    mock_handle_link.assert_called_once()
    mock_manager.integration_store.update_workspace.assert_awaited_once_with(
        id=10,
        status='inactive',
    )


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch('server.routes.integration.jira_dc.JIRA_DC_ENABLE_OAUTH', False)
@patch(
    'server.routes.integration.jira_dc._handle_workspace_link_creation',
    new_callable=AsyncMock,
)
@patch('server.routes.integration.jira_dc.get_jira_dc_service_account_config_error')
@patch('server.routes.integration.jira_dc.get_jira_dc_managed_service_account')
async def test_create_jira_dc_workspace_uses_managed_service_account(
    mock_get_managed_service_account,
    mock_get_service_account_error,
    mock_handle_link,
    mock_manager,
    mock_get_auth,
    mock_request,
    mock_user_auth,
):
    mock_get_auth.return_value = mock_user_auth
    mock_get_service_account_error.return_value = None
    mock_get_managed_service_account.return_value = JiraDcServiceAccount(
        email='managed@test.com',
        api_key='managed-pat',
        managed_by_env=True,
    )
    mock_manager.integration_store.get_workspace_by_name.return_value = None
    mock_workspace = MagicMock()
    mock_workspace.id = 10
    mock_workspace.name = 'test-workspace'
    mock_manager.integration_store.create_workspace.return_value = mock_workspace

    workspace_data = JiraDcWorkspaceCreate(
        workspace_name='test-workspace',
        webhook_secret='secret',
        is_active=True,
    )

    with patch('server.routes.integration.jira_dc.token_manager') as mock_token_manager:
        mock_token_manager.encrypt_text.side_effect = lambda x: f'enc_{x}'

        response = await create_jira_dc_workspace(mock_request, workspace_data)
        content = json.loads(response.body)

    assert response.status_code == 200
    assert content['success'] is True
    mock_manager.integration_store.create_workspace.assert_called_once()
    create_kwargs = mock_manager.integration_store.create_workspace.call_args.kwargs
    assert create_kwargs['svc_acc_email'] == 'managed@test.com'
    assert create_kwargs['encrypted_svc_acc_api_key'] == 'enc_managed-pat'
    mock_handle_link.assert_called_once()


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch('server.routes.integration.jira_dc.JIRA_DC_ENABLE_OAUTH', False)
@patch(
    'server.routes.integration.jira_dc._validate_workspace_update_permissions',
    new_callable=AsyncMock,
)
@patch(
    'server.routes.integration.jira_dc._handle_workspace_link_creation',
    new_callable=AsyncMock,
)
async def test_create_jira_dc_workspace_oauth_disabled_existing_workspace(
    mock_handle_link,
    mock_validate,
    mock_manager,
    mock_get_auth,
    mock_request,
    mock_user_auth,
):
    mock_get_auth.return_value = mock_user_auth
    mock_workspace = MagicMock(id=1, name='test-workspace')
    mock_workspace.id = 1
    mock_workspace.name = 'test-workspace'
    mock_manager.integration_store.get_workspace_by_name.return_value = mock_workspace
    mock_validate.return_value = mock_workspace
    mock_manager.integration_store.update_workspace.return_value = mock_workspace

    workspace_data = JiraDcWorkspaceCreate(
        workspace_name='test-workspace',
        webhook_secret='secret',
        svc_acc_email='svc@test.com',
        svc_acc_api_key='key',
        is_active=True,
    )

    with patch('server.routes.integration.jira_dc.token_manager') as mock_token_manager:
        mock_token_manager.encrypt_text.side_effect = lambda x: f'enc_{x}'

        response = await create_jira_dc_workspace(mock_request, workspace_data)
        content = json.loads(response.body)

        assert response.status_code == 200
        assert content['success'] is True
        assert content['redirect'] is False
        mock_manager.integration_store.update_workspace.assert_called_once()
        mock_handle_link.assert_called_once()


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch('server.routes.integration.jira_dc.JIRA_DC_ENABLE_OAUTH', False)
@patch(
    'server.routes.integration.jira_dc._validate_workspace_update_permissions',
    new_callable=AsyncMock,
)
@patch(
    'server.routes.integration.jira_dc._handle_workspace_link_creation',
    new_callable=AsyncMock,
)
async def test_create_jira_dc_workspace_preserves_secret_when_omitted_on_update(
    mock_handle_link,
    mock_validate,
    mock_manager,
    mock_get_auth,
    mock_request,
    mock_user_auth,
):
    mock_get_auth.return_value = mock_user_auth
    mock_workspace = MagicMock(
        id=1, name='test-workspace', webhook_secret='encrypted_old_secret'
    )
    mock_manager.integration_store.get_workspace_by_name.return_value = mock_workspace
    mock_validate.return_value = mock_workspace
    mock_manager.integration_store.update_workspace.return_value = mock_workspace

    workspace_data = JiraDcWorkspaceCreate(
        workspace_name='test-workspace',
        svc_acc_email='svc@test.com',
        svc_acc_api_key='key',
        is_active=True,
    )

    with patch('server.routes.integration.jira_dc.token_manager') as mock_token_manager:
        mock_token_manager.decrypt_text.return_value = 'old-secret'
        mock_token_manager.encrypt_text.side_effect = lambda x: f'enc_{x}'

        response = await create_jira_dc_workspace(mock_request, workspace_data)
        content = json.loads(response.body)

        assert response.status_code == 200
        assert content['success'] is True
        mock_token_manager.decrypt_text.assert_called_once_with('encrypted_old_secret')
        mock_manager.integration_store.update_workspace.assert_called_once()
        assert (
            mock_manager.integration_store.update_workspace.call_args.kwargs[
                'encrypted_webhook_secret'
            ]
            == 'enc_old-secret'
        )
        mock_handle_link.assert_called_once()


# Test create_workspace_link with OAuth disabled
@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.JIRA_DC_ENABLE_OAUTH', False)
@patch(
    'server.routes.integration.jira_dc._handle_workspace_link_creation',
    new_callable=AsyncMock,
)
async def test_create_workspace_link_oauth_disabled(
    mock_handle_link, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    link_data = JiraDcLinkCreate(workspace_name='test-workspace')

    response = await create_workspace_link(mock_request, link_data)
    content = json.loads(response.body)

    assert response.status_code == 200
    assert content['success'] is True
    assert content['redirect'] is False
    assert content['authorizationUrl'] == ''
    mock_handle_link.assert_called_once_with(
        'test_user_id', 'unavailable', 'test-workspace'
    )


# Test create_jira_dc_workspace error scenarios
@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
async def test_create_jira_dc_workspace_auth_failure(mock_get_auth, mock_request):
    mock_get_auth.side_effect = HTTPException(status_code=401, detail='Unauthorized')

    workspace_data = JiraDcWorkspaceCreate(
        workspace_name='test-workspace',
        webhook_secret='secret',
        svc_acc_email='svc@test.com',
        svc_acc_api_key='key',
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_jira_dc_workspace(mock_request, workspace_data)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.redis_client')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch('server.routes.integration.jira_dc.JIRA_DC_ENABLE_OAUTH', True)
async def test_create_jira_dc_workspace_redis_failure(
    mock_manager, mock_redis, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    mock_manager.integration_store.get_workspace_by_name.return_value = None
    mock_redis.setex.return_value = False  # Redis operation failed

    workspace_data = JiraDcWorkspaceCreate(
        workspace_name='test-workspace',
        webhook_secret='secret',
        svc_acc_email='svc@test.com',
        svc_acc_api_key='key',
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_jira_dc_workspace(mock_request, workspace_data)
    assert exc_info.value.status_code == 500
    assert 'Failed to create integration session' in exc_info.value.detail


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
async def test_create_jira_dc_workspace_unexpected_error(mock_get_auth, mock_request):
    mock_get_auth.side_effect = Exception('Unexpected error')

    workspace_data = JiraDcWorkspaceCreate(
        workspace_name='test-workspace',
        webhook_secret='secret',
        svc_acc_email='svc@test.com',
        svc_acc_api_key='key',
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_jira_dc_workspace(mock_request, workspace_data)
    assert exc_info.value.status_code == 500
    assert 'Failed to create workspace' in exc_info.value.detail


# Test create_workspace_link error scenarios
@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.redis_client')
@patch('server.routes.integration.jira_dc.JIRA_DC_ENABLE_OAUTH', True)
async def test_create_workspace_link_redis_failure(
    mock_redis, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    mock_redis.setex.return_value = False

    link_data = JiraDcLinkCreate(workspace_name='test-workspace')

    with pytest.raises(HTTPException) as exc_info:
        await create_workspace_link(mock_request, link_data)
    assert exc_info.value.status_code == 500
    assert 'Failed to create integration session' in exc_info.value.detail


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
async def test_create_workspace_link_unexpected_error(mock_get_auth, mock_request):
    mock_get_auth.side_effect = Exception('Unexpected error')

    link_data = JiraDcLinkCreate(workspace_name='test-workspace')

    with pytest.raises(HTTPException) as exc_info:
        await create_workspace_link(mock_request, link_data)
    assert exc_info.value.status_code == 500
    assert 'Failed to register user' in exc_info.value.detail


# Test jira_dc_callback error scenarios
@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.redis_client')
async def test_jira_dc_callback_no_session(mock_redis, mock_request):
    mock_redis.get.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await jira_dc_callback(mock_request, 'code', 'state')
    assert exc_info.value.status_code == 400
    assert 'No active integration session found' in exc_info.value.detail


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.redis_client')
async def test_jira_dc_callback_state_mismatch(mock_redis, mock_request):
    session_data = {'state': 'different_state'}
    mock_redis.get.return_value = json.dumps(session_data)

    with pytest.raises(HTTPException) as exc_info:
        await jira_dc_callback(mock_request, 'code', 'wrong_state')
    assert exc_info.value.status_code == 400
    assert 'State mismatch. Possible CSRF attack' in exc_info.value.detail


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.redis_client')
async def test_jira_dc_callback_token_fetch_failure(mock_redis, mock_request):
    session_data = {'state': 'test_state'}
    mock_redis.get.return_value = json.dumps(session_data)

    # Create httpx mock with failing POST response
    mock_post_response = MagicMock()
    mock_post_response.status_code = 400
    mock_post_response.text = 'Token error'

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_post_response)

    mock_httpx = AsyncMock()
    mock_httpx.__aenter__ = AsyncMock(return_value=mock_client)
    mock_httpx.__aexit__ = AsyncMock(return_value=None)

    with patch('httpx.AsyncClient', return_value=mock_httpx):
        with pytest.raises(HTTPException) as exc_info:
            await jira_dc_callback(mock_request, 'code', 'test_state')
        assert exc_info.value.status_code == 400
        assert 'Error fetching token' in exc_info.value.detail


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.redis_client')
async def test_jira_dc_callback_resources_fetch_failure(mock_redis, mock_request):
    session_data = {'state': 'test_state'}
    mock_redis.get.return_value = json.dumps(session_data)

    # Create httpx mock with successful POST but failing GET
    mock_post_response = MagicMock()
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = {'access_token': 'token'}

    mock_get_response = MagicMock()
    mock_get_response.status_code = 400
    mock_get_response.text = 'Resources error'

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_post_response)
    mock_client.get = AsyncMock(return_value=mock_get_response)

    mock_httpx = AsyncMock()
    mock_httpx.__aenter__ = AsyncMock(return_value=mock_client)
    mock_httpx.__aexit__ = AsyncMock(return_value=None)

    with patch('httpx.AsyncClient', return_value=mock_httpx):
        with pytest.raises(HTTPException) as exc_info:
            await jira_dc_callback(mock_request, 'code', 'test_state')
        assert exc_info.value.status_code == 400
        assert 'Error fetching user info: Resources error' in exc_info.value.detail


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.redis_client')
async def test_jira_dc_callback_unauthorized_workspace(mock_redis, mock_request):
    session_data = {
        'state': 'test_state',
        'target_workspace': 'target.atlassian.net',
        'keycloak_user_id': 'user1',
    }
    mock_redis.get.return_value = json.dumps(session_data)

    # Create httpx mock with different URL for accessible-resources
    mock_post_response = MagicMock()
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = {'access_token': 'token'}

    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = [{'url': 'https://different.atlassian.net'}]

    mock_get_user_response = MagicMock()
    mock_get_user_response.status_code = 200
    mock_get_user_response.json.return_value = {'key': 'jira_user_123'}

    async def mock_get(url, **kwargs):
        if 'accessible-resources' in url:
            return mock_get_response
        return mock_get_user_response

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_post_response)
    mock_client.get = AsyncMock(side_effect=mock_get)

    mock_httpx = AsyncMock()
    mock_httpx.__aenter__ = AsyncMock(return_value=mock_client)
    mock_httpx.__aexit__ = AsyncMock(return_value=None)

    with patch(
        'server.routes.integration.jira_dc.JIRA_DC_BASE_URL',
        'https://target.atlassian.net',
    ):
        with patch('httpx.AsyncClient', return_value=mock_httpx):
            with pytest.raises(HTTPException) as exc_info:
                await jira_dc_callback(mock_request, 'code', 'test_state')
            assert exc_info.value.status_code == 400
            assert 'Invalid operation type' in exc_info.value.detail


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.redis_client')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch(
    'server.routes.integration.jira_dc._handle_workspace_link_creation',
    new_callable=AsyncMock,
)
async def test_jira_dc_callback_workspace_integration_existing_workspace(
    mock_handle_link, mock_manager, mock_redis, mock_request
):
    state = 'test_state'
    session_data = {
        'operation_type': 'workspace_integration',
        'keycloak_user_id': 'user1',
        'target_workspace': 'existing.atlassian.net',
        'webhook_secret': 'secret',
        'svc_acc_email': 'email@test.com',
        'svc_acc_api_key': 'apikey',
        'is_active': True,
        'state': state,
    }
    mock_redis.get.return_value = json.dumps(session_data)

    # Set up httpx mock with conditional GET responses
    httpx_mock = create_httpx_mock(
        post_response={'access_token': 'token'},
        get_responses={
            'accessible-resources': [{'url': 'https://existing.atlassian.net'}],
            '/myself': {'key': 'jira_user_123'},
        },
    )

    # Mock existing workspace
    mock_workspace = MagicMock(id=1)
    mock_manager.integration_store.get_workspace_by_name.return_value = mock_workspace
    mock_manager.integration_store.update_workspace.return_value = mock_workspace

    with patch('server.routes.integration.jira_dc.token_manager') as mock_token_manager:
        with patch(
            'server.routes.integration.jira_dc.JIRA_DC_BASE_URL',
            'https://existing.atlassian.net',
        ):
            with patch('httpx.AsyncClient', return_value=httpx_mock):
                with patch(
                    'server.routes.integration.jira_dc._validate_workspace_update_permissions'
                ) as mock_validate:
                    mock_validate.return_value = mock_workspace
                    mock_token_manager.encrypt_text.side_effect = lambda x: f'enc_{x}'

                    response = await jira_dc_callback(mock_request, 'code', state)

                    assert isinstance(response, RedirectResponse)
                    assert response.status_code == status.HTTP_302_FOUND
                    mock_manager.integration_store.update_workspace.assert_called_once()
                    mock_handle_link.assert_called_once_with(
                        'user1',
                        'jira_user_123',
                        'existing.atlassian.net',
                        replace_stale_active_link=True,
                    )


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.redis_client')
async def test_jira_dc_callback_invalid_operation_type(mock_redis, mock_request):
    session_data = {
        'operation_type': 'invalid_operation',
        'target_workspace': 'test.atlassian.net',
        'keycloak_user_id': 'user1',
        'state': 'test_state',
    }
    mock_redis.get.return_value = json.dumps(session_data)

    # Set up httpx mock (though it won't be called due to invalid operation type check)
    httpx_mock = create_httpx_mock(
        post_response={'access_token': 'token'},
        get_responses={
            'accessible-resources': [{'url': 'https://test.atlassian.net'}],
            '/myself': {'key': 'jira_user_123'},
        },
    )

    with patch(
        'server.routes.integration.jira_dc.JIRA_DC_BASE_URL',
        'https://test.atlassian.net',
    ):
        with patch('httpx.AsyncClient', return_value=httpx_mock):
            with pytest.raises(HTTPException) as exc_info:
                await jira_dc_callback(mock_request, 'code', 'test_state')
            assert exc_info.value.status_code == 400
            assert 'Invalid operation type' in exc_info.value.detail


# Test get_current_workspace_link error scenarios
@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_get_current_workspace_link_user_not_found(
    mock_manager, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    mock_manager.integration_store.get_user_by_active_workspace.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await get_current_workspace_link(mock_request)
    assert exc_info.value.status_code == 404
    assert 'User is not registered for Jira DC integration' in exc_info.value.detail


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_get_current_workspace_link_workspace_not_found(
    mock_manager, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    mock_user = MagicMock(jira_dc_workspace_id=10)
    mock_manager.integration_store.get_user_by_active_workspace.return_value = mock_user
    mock_manager.integration_store.get_workspace_by_id.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await get_current_workspace_link(mock_request)
    assert exc_info.value.status_code == 404
    assert 'Workspace not found for the user' in exc_info.value.detail


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_get_current_workspace_link_not_editable(
    mock_manager, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    user_id = 'test_user_id'
    different_admin = 'different_admin'

    mock_user = MagicMock(
        id=1,
        keycloak_user_id=user_id,
        jira_dc_workspace_id=10,
        status='active',
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    mock_workspace = MagicMock(
        id=10,
        status='active',
        admin_user_id=different_admin,
        svc_acc_email='svc@test.com',
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    # Fix the name attribute to be a string instead of MagicMock
    mock_workspace.name = 'test-space'
    mock_workspace.svc_acc_email = 'svc@test.com'

    mock_manager.integration_store.get_user_by_active_workspace.return_value = mock_user
    mock_manager.integration_store.get_workspace_by_id.return_value = mock_workspace

    response = await get_current_workspace_link(mock_request)
    assert response.workspace.editable is False


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_get_current_workspace_link_unexpected_error(
    mock_manager, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    mock_manager.integration_store.get_user_by_active_workspace.side_effect = Exception(
        'DB error'
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_current_workspace_link(mock_request)
    assert exc_info.value.status_code == 500
    assert 'Failed to retrieve user' in exc_info.value.detail


# Test unlink_workspace error scenarios
@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_unlink_workspace_user_not_found(
    mock_manager, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    mock_manager.integration_store.get_user_by_active_workspace.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await unlink_workspace(mock_request)
    assert exc_info.value.status_code == 404
    assert 'User is not registered for Jira DC integration' in exc_info.value.detail


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_unlink_workspace_workspace_not_found(
    mock_manager, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    mock_user = MagicMock(jira_dc_workspace_id=10)
    mock_manager.integration_store.get_user_by_active_workspace.return_value = mock_user
    mock_manager.integration_store.get_workspace_by_id.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await unlink_workspace(mock_request)
    assert exc_info.value.status_code == 404
    assert 'Workspace not found for the user' in exc_info.value.detail


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_unlink_workspace_non_admin(
    mock_manager, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    user_id = 'test_user_id'
    mock_user = MagicMock(jira_dc_workspace_id=10)
    mock_workspace = MagicMock(id=10, admin_user_id='different_admin')
    mock_manager.integration_store.get_user_by_active_workspace.return_value = mock_user
    mock_manager.integration_store.get_workspace_by_id.return_value = mock_workspace

    response = await unlink_workspace(mock_request)
    content = json.loads(response.body)
    assert content['success'] is True
    mock_manager.integration_store.update_user_integration_status.assert_called_once_with(
        user_id, 10, 'inactive'
    )


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_unlink_workspace_unexpected_error(
    mock_manager, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    mock_manager.integration_store.get_user_by_active_workspace.side_effect = Exception(
        'DB error'
    )

    with pytest.raises(HTTPException) as exc_info:
        await unlink_workspace(mock_request)
    assert exc_info.value.status_code == 500
    assert 'Failed to unlink user' in exc_info.value.detail


# Test validate_workspace_integration error scenarios
@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
async def test_validate_workspace_integration_invalid_name(
    mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth

    with pytest.raises(HTTPException) as exc_info:
        await validate_workspace_integration(mock_request, 'invalid workspace!')
    assert exc_info.value.status_code == 400
    assert (
        'workspace_name can only contain alphanumeric characters'
        in exc_info.value.detail
    )


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_validate_workspace_integration_workspace_not_found(
    mock_manager, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    mock_manager.integration_store.get_workspace_by_name.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await validate_workspace_integration(mock_request, 'nonexistent-workspace')
    assert exc_info.value.status_code == 404
    assert (
        "Workspace with name 'nonexistent-workspace' not found" in exc_info.value.detail
    )


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_validate_workspace_integration_inactive_workspace(
    mock_manager, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    mock_workspace = MagicMock(status='inactive')
    # Fix the name attribute to be a string instead of MagicMock
    mock_workspace.name = 'test-workspace'
    mock_manager.integration_store.get_workspace_by_name.return_value = mock_workspace

    with pytest.raises(HTTPException) as exc_info:
        await validate_workspace_integration(mock_request, 'test-workspace')
    assert exc_info.value.status_code == 404
    assert "Workspace 'test-workspace' is not active" in exc_info.value.detail


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.get_user_auth')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_validate_workspace_integration_unexpected_error(
    mock_manager, mock_get_auth, mock_request, mock_user_auth
):
    mock_get_auth.return_value = mock_user_auth
    mock_manager.integration_store.get_workspace_by_name.side_effect = Exception(
        'DB error'
    )

    with pytest.raises(HTTPException) as exc_info:
        await validate_workspace_integration(mock_request, 'test-workspace')
    assert exc_info.value.status_code == 500
    assert 'Failed to validate workspace' in exc_info.value.detail


# Test helper functions
@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_handle_workspace_link_creation_workspace_not_found(mock_manager):
    mock_manager.integration_store.get_workspace_by_name.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await _handle_workspace_link_creation(
            'user1', 'jira_user_123', 'nonexistent-workspace'
        )
    assert exc_info.value.status_code == 404
    assert 'Workspace "nonexistent-workspace" not found' in exc_info.value.detail


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_handle_workspace_link_creation_inactive_workspace(mock_manager):
    mock_workspace = MagicMock(status='inactive')
    mock_manager.integration_store.get_workspace_by_name.return_value = mock_workspace

    with pytest.raises(HTTPException) as exc_info:
        await _handle_workspace_link_creation(
            'user1', 'jira_user_123', 'inactive-workspace'
        )
    assert exc_info.value.status_code == 400
    assert 'Workspace "inactive-workspace" is not active' in exc_info.value.detail


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_handle_workspace_link_creation_already_linked_same_workspace(
    mock_manager,
):
    mock_workspace = MagicMock(id=1, status='active')
    mock_existing_user = MagicMock(jira_dc_workspace_id=1)

    mock_manager.integration_store.get_workspace_by_name.return_value = mock_workspace
    mock_manager.integration_store.get_user_by_active_workspace.return_value = (
        mock_existing_user
    )

    # Should not raise exception and should not create new link
    await _handle_workspace_link_creation('user1', 'jira_user_123', 'test-workspace')

    mock_manager.integration_store.create_workspace_link.assert_not_called()
    mock_manager.integration_store.update_user_integration_status.assert_not_called()


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_handle_workspace_link_creation_already_linked_different_workspace(
    mock_manager,
):
    mock_workspace = MagicMock(id=2, status='active')
    mock_existing_user = MagicMock(jira_dc_workspace_id=1)  # Different workspace

    mock_manager.integration_store.get_workspace_by_name.return_value = mock_workspace
    mock_manager.integration_store.get_user_by_active_workspace.return_value = (
        mock_existing_user
    )

    with pytest.raises(HTTPException) as exc_info:
        await _handle_workspace_link_creation(
            'user1', 'jira_user_123', 'test-workspace'
        )
    assert exc_info.value.status_code == 400
    assert 'You already have an active workspace link' in exc_info.value.detail


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_handle_workspace_link_creation_replaces_stale_active_link(
    mock_manager,
):
    mock_workspace = MagicMock(id=2, status='active')

    mock_manager.integration_store.get_workspace_by_name.return_value = mock_workspace
    mock_manager.integration_store.get_user_by_active_workspace.return_value = None
    mock_manager.integration_store.get_user_by_keycloak_id_and_workspace.return_value = None

    await _handle_workspace_link_creation(
        'user1',
        'jira_user_123',
        'current-workspace',
        replace_stale_active_link=True,
    )

    mock_manager.integration_store.deactivate_user_links_except_workspace.assert_called_once_with(
        'user1', 2
    )
    mock_manager.integration_store.create_workspace_link.assert_called_once_with(
        keycloak_user_id='user1',
        jira_dc_user_id='jira_user_123',
        jira_dc_workspace_id=2,
    )


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_handle_workspace_link_creation_reactivate_existing_link(mock_manager):
    mock_workspace = MagicMock(id=1, status='active')
    mock_existing_link = MagicMock()

    mock_manager.integration_store.get_workspace_by_name.return_value = mock_workspace
    mock_manager.integration_store.get_user_by_active_workspace.return_value = None
    mock_manager.integration_store.get_user_by_keycloak_id_and_workspace.return_value = mock_existing_link

    await _handle_workspace_link_creation('user1', 'jira_user_123', 'test-workspace')

    mock_manager.integration_store.update_user_integration_status.assert_called_once_with(
        'user1', 1, 'active'
    )
    mock_manager.integration_store.create_workspace_link.assert_not_called()


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_handle_workspace_link_creation_create_new_link(mock_manager):
    mock_workspace = MagicMock(id=1, status='active')

    mock_manager.integration_store.get_workspace_by_name.return_value = mock_workspace
    mock_manager.integration_store.get_user_by_active_workspace.return_value = None
    mock_manager.integration_store.get_user_by_keycloak_id_and_workspace.return_value = None

    await _handle_workspace_link_creation('user1', 'jira_user_123', 'test-workspace')

    mock_manager.integration_store.create_workspace_link.assert_called_once_with(
        keycloak_user_id='user1',
        jira_dc_user_id='jira_user_123',
        jira_dc_workspace_id=1,
    )
    mock_manager.integration_store.update_user_integration_status.assert_not_called()


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_validate_workspace_update_permissions_workspace_not_found(mock_manager):
    mock_manager.integration_store.get_workspace_by_name.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await _validate_workspace_update_permissions('user1', 'nonexistent-workspace')
    assert exc_info.value.status_code == 404
    assert 'Workspace "nonexistent-workspace" not found' in exc_info.value.detail


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_validate_workspace_update_permissions_not_admin(mock_manager):
    mock_workspace = MagicMock(admin_user_id='different_user')
    mock_manager.integration_store.get_workspace_by_name.return_value = mock_workspace

    with pytest.raises(HTTPException) as exc_info:
        await _validate_workspace_update_permissions('user1', 'test-workspace')
    assert exc_info.value.status_code == 403
    assert (
        'You do not have permission to update this workspace' in exc_info.value.detail
    )


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_validate_workspace_update_permissions_wrong_linked_workspace(
    mock_manager,
):
    mock_workspace = MagicMock(id=1, admin_user_id='user1')
    mock_user_link = MagicMock(jira_dc_workspace_id=2)  # Different workspace

    mock_manager.integration_store.get_workspace_by_name.return_value = mock_workspace
    mock_manager.integration_store.get_user_by_active_workspace.return_value = (
        mock_user_link
    )

    with pytest.raises(HTTPException) as exc_info:
        await _validate_workspace_update_permissions('user1', 'test-workspace')
    assert exc_info.value.status_code == 403
    assert (
        'You can only update the workspace you are currently linked to'
        in exc_info.value.detail
    )


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_validate_workspace_update_permissions_success(mock_manager):
    mock_workspace = MagicMock(id=1, admin_user_id='user1')
    mock_user_link = MagicMock(jira_dc_workspace_id=1)

    mock_manager.integration_store.get_workspace_by_name.return_value = mock_workspace
    mock_manager.integration_store.get_user_by_active_workspace.return_value = (
        mock_user_link
    )

    result = await _validate_workspace_update_permissions('user1', 'test-workspace')
    assert result == mock_workspace


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
async def test_validate_workspace_update_permissions_no_current_link(mock_manager):
    mock_workspace = MagicMock(id=1, admin_user_id='user1')

    mock_manager.integration_store.get_workspace_by_name.return_value = mock_workspace
    mock_manager.integration_store.get_user_by_active_workspace.return_value = None

    result = await _validate_workspace_update_permissions('user1', 'test-workspace')
    assert result == mock_workspace


# Tests for OAuth URL encoding
class TestJiraDcOAuthUrlEncoding:
    """Tests to verify OAuth authorization URLs are properly URL-encoded."""

    @pytest.mark.asyncio
    @patch('server.routes.integration.jira_dc.get_user_auth')
    @patch('server.routes.integration.jira_dc.redis_client')
    @patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
    @patch('server.routes.integration.jira_dc.JIRA_DC_ENABLE_OAUTH', True)
    async def test_create_jira_dc_workspace_url_encoding(
        self, mock_manager, mock_redis, mock_get_auth, mock_request, mock_user_auth
    ):
        """Test that create_jira_dc_workspace properly URL-encodes the authorization URL."""
        mock_get_auth.return_value = mock_user_auth
        mock_manager.integration_store.get_workspace_by_name.return_value = None
        mock_redis.setex.return_value = True
        workspace_data = JiraDcWorkspaceCreate(
            workspace_name='test-workspace',
            webhook_secret='secret',
            svc_acc_email='svc@test.com',
            svc_acc_api_key='key',
            is_active=True,
        )

        response = await create_jira_dc_workspace(mock_request, workspace_data)
        content = json.loads(response.body)

        auth_url = content['authorizationUrl']
        # Verify no raw spaces in the URL (spaces should be encoded as + or %20)
        assert ' ' not in auth_url
        # Verify redirect_uri is properly encoded
        assert 'redirect_uri=https%3A%2F%2F' in auth_url

    @pytest.mark.asyncio
    @patch('server.routes.integration.jira_dc.get_user_auth')
    @patch('server.routes.integration.jira_dc.redis_client')
    @patch('server.routes.integration.jira_dc.JIRA_DC_ENABLE_OAUTH', True)
    async def test_create_workspace_link_url_encoding(
        self, mock_redis, mock_get_auth, mock_request, mock_user_auth
    ):
        """Test that create_workspace_link properly URL-encodes the authorization URL."""
        mock_get_auth.return_value = mock_user_auth
        mock_redis.setex.return_value = True
        link_data = JiraDcLinkCreate(workspace_name='test-workspace')

        response = await create_workspace_link(mock_request, link_data)
        content = json.loads(response.body)

        auth_url = content['authorizationUrl']
        # Verify no raw spaces in the URL (spaces should be encoded as + or %20)
        assert ' ' not in auth_url
        # Verify redirect_uri is properly encoded
        assert 'redirect_uri=https%3A%2F%2F' in auth_url


# ---------------------------------------------------------------------------
# Token persistence in the OAuth callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.redis_client')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch(
    'server.routes.integration.jira_dc._handle_workspace_link_creation',
    new_callable=AsyncMock,
)
async def test_callback_workspace_integration_persists_tokens(
    mock_handle_link, mock_manager, mock_redis, mock_request
):
    """workspace_integration callback must encrypt and store access + refresh tokens."""
    state = 'test_state'
    token_data = {
        'access_token': 'at',
        'refresh_token': 'rt',
        'expires_in': 3600,
        'refresh_token_expires_in': 86400,
    }
    session_data = {
        'operation_type': 'workspace_integration',
        'keycloak_user_id': 'user1',
        'target_workspace': 'test.atlassian.net',
        'webhook_secret': 'secret',
        'svc_acc_email': 'svc@test.com',
        'svc_acc_api_key': 'apikey',
        'is_active': True,
        'state': state,
    }
    mock_redis.get.return_value = json.dumps(session_data)
    httpx_mock = create_httpx_mock(
        post_response=token_data,
        get_responses={
            'accessible-resources': [{'url': 'https://test.atlassian.net'}],
            '/myself': {'key': 'jira_user_1'},
        },
    )
    mock_workspace = MagicMock(id=5)
    mock_manager.integration_store.get_workspace_by_name.return_value = None
    mock_manager.integration_store.create_workspace.return_value = mock_workspace

    with patch('server.routes.integration.jira_dc.token_manager') as mock_tm:
        with patch(
            'server.routes.integration.jira_dc.JIRA_DC_BASE_URL',
            'https://test.atlassian.net',
        ):
            mock_tm.encrypt_text.side_effect = lambda v: f'enc({v})'
            with patch('httpx.AsyncClient', return_value=httpx_mock):
                await jira_dc_callback(mock_request, 'code', state)

    mock_manager.integration_store.update_user_oauth_tokens.assert_awaited_once()
    kwargs = mock_manager.integration_store.update_user_oauth_tokens.call_args.kwargs
    assert kwargs['keycloak_user_id'] == 'user1'
    assert kwargs['workspace_id'] == 5
    assert kwargs['encrypted_access_token'] == 'enc(at)'
    assert kwargs['encrypted_refresh_token'] == 'enc(rt)'
    assert kwargs['access_token_expires_at'] > 0
    assert kwargs['refresh_token_expires_at'] > 0


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.redis_client')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch(
    'server.routes.integration.jira_dc._handle_workspace_link_creation',
    new_callable=AsyncMock,
)
async def test_callback_workspace_integration_null_refresh_token_tolerated(
    mock_handle_link, mock_manager, mock_redis, mock_request
):
    """Missing refresh_token in IdP response must be stored as NULL, not raise."""
    state = 'test_state'
    token_data = {'access_token': 'at'}  # no refresh_token, no expires_in
    session_data = {
        'operation_type': 'workspace_integration',
        'keycloak_user_id': 'user1',
        'target_workspace': 'test.atlassian.net',
        'webhook_secret': 'secret',
        'svc_acc_email': 'svc@test.com',
        'svc_acc_api_key': 'apikey',
        'is_active': True,
        'state': state,
    }
    mock_redis.get.return_value = json.dumps(session_data)
    httpx_mock = create_httpx_mock(
        post_response=token_data,
        get_responses={
            'accessible-resources': [{'url': 'https://test.atlassian.net'}],
            '/myself': {'key': 'jira_user_1'},
        },
    )
    mock_workspace = MagicMock(id=7)
    mock_manager.integration_store.get_workspace_by_name.return_value = None
    mock_manager.integration_store.create_workspace.return_value = mock_workspace

    with patch('server.routes.integration.jira_dc.token_manager') as mock_tm:
        with patch(
            'server.routes.integration.jira_dc.JIRA_DC_BASE_URL',
            'https://test.atlassian.net',
        ):
            mock_tm.encrypt_text.side_effect = lambda v: f'enc({v})'
            with patch('httpx.AsyncClient', return_value=httpx_mock):
                await jira_dc_callback(mock_request, 'code', state)

    kwargs = mock_manager.integration_store.update_user_oauth_tokens.call_args.kwargs
    assert kwargs['encrypted_refresh_token'] is None
    assert kwargs['access_token_expires_at'] == 0
    assert kwargs['refresh_token_expires_at'] == 0


@pytest.mark.asyncio
@patch('server.routes.integration.jira_dc.redis_client')
@patch('server.routes.integration.jira_dc.jira_dc_manager', new_callable=AsyncMock)
@patch(
    'server.routes.integration.jira_dc._handle_workspace_link_creation',
    new_callable=AsyncMock,
)
async def test_callback_workspace_link_persists_tokens(
    mock_handle_link, mock_manager, mock_redis, mock_request
):
    """workspace_link callback must also persist tokens."""
    state = 'test_state'
    token_data = {'access_token': 'at', 'refresh_token': 'rt', 'expires_in': 1800}
    session_data = {
        'operation_type': 'workspace_link',
        'keycloak_user_id': 'user2',
        'target_workspace': 'test.atlassian.net',
        'state': state,
    }
    mock_redis.get.return_value = json.dumps(session_data)
    httpx_mock = create_httpx_mock(
        post_response=token_data,
        get_responses={
            'accessible-resources': [{'url': 'https://test.atlassian.net'}],
            '/myself': {'key': 'jira_user_2'},
        },
    )
    mock_workspace = MagicMock(id=9)
    mock_manager.integration_store.get_workspace_by_name.return_value = mock_workspace

    with patch('server.routes.integration.jira_dc.token_manager') as mock_tm:
        with patch(
            'server.routes.integration.jira_dc.JIRA_DC_BASE_URL',
            'https://test.atlassian.net',
        ):
            mock_tm.encrypt_text.side_effect = lambda v: f'enc({v})'
            with patch('httpx.AsyncClient', return_value=httpx_mock):
                await jira_dc_callback(mock_request, 'code', state)

    mock_manager.integration_store.update_user_oauth_tokens.assert_awaited_once()
    kwargs = mock_manager.integration_store.update_user_oauth_tokens.call_args.kwargs
    assert kwargs['keycloak_user_id'] == 'user2'
    assert kwargs['workspace_id'] == 9
    assert kwargs['encrypted_access_token'] == 'enc(at)'
