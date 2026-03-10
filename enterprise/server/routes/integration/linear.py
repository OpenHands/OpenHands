import json
import os
import re
import uuid
from typing import cast

import requests
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from integrations.linear.linear_manager import LinearManager
from integrations.models import Message, SourceType
from pydantic import BaseModel, Field, field_validator
from server.auth.constants import LINEAR_CLIENT_ID, LINEAR_CLIENT_SECRET
from server.auth.saas_user_auth import SaasUserAuth
from server.auth.token_manager import TokenManager
from server.constants import WEB_HOST
from storage.redis import create_redis_client

from openhands.core.logger import openhands_logger as logger
from openhands.server.user_auth.user_auth import get_user_auth

# Environment variable to disable Linear webhooks
LINEAR_WEBHOOKS_ENABLED = os.environ.get('LINEAR_WEBHOOKS_ENABLED', '0') in (
    '1',
    'true',
)
LINEAR_REDIRECT_URI = f'https://{WEB_HOST}/integration/linear/callback'
LINEAR_SCOPES = 'read'
LINEAR_AUTH_URL = 'https://linear.app/oauth/authorize'
LINEAR_TOKEN_URL = 'https://api.linear.app/oauth/token'
LINEAR_GRAPHQL_URL = 'https://api.linear.app/graphql'


# Request/Response models
class LinearWorkspaceCreate(BaseModel):
    workspace_name: str = Field(..., description='Workspace display name')
    webhook_secret: str = Field(..., description='Webhook secret for verification')
    svc_acc_email: str = Field(..., description='Service account email')
    svc_acc_api_key: str = Field(..., description='Service account API key')
    is_active: bool = Field(
        default=False,
        description='Indicates if the workspace integration is active',
    )

    @field_validator('workspace_name')
    @classmethod
    def validate_workspace_name(cls, v):
        if not re.match(r'^[a-zA-Z0-9_.-]+$', v):
            raise ValueError(
                'workspace_name can only contain alphanumeric characters, hyphens, underscores, and periods'
            )
        return v

    @field_validator('svc_acc_email')
    @classmethod
    def validate_svc_acc_email(cls, v):
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError('svc_acc_email must be a valid email address')
        return v

    @field_validator('webhook_secret')
    @classmethod
    def validate_webhook_secret(cls, v):
        if ' ' in v:
            raise ValueError('webhook_secret cannot contain spaces')
        return v

    @field_validator('svc_acc_api_key')
    @classmethod
    def validate_svc_acc_api_key(cls, v):
        if ' ' in v:
            raise ValueError('svc_acc_api_key cannot contain spaces')
        return v


class LinearLinkCreate(BaseModel):
    workspace_name: str = Field(
        ..., description='Name of the Linear workspace to link to'
    )

    @field_validator('workspace_name')
    @classmethod
    def validate_workspace(cls, v):
        if not re.match(r'^[a-zA-Z0-9_.-]+$', v):
            raise ValueError(
                'workspace can only contain alphanumeric characters, hyphens, underscores, and periods'
            )
        return v


class LinearWorkspaceResponse(BaseModel):
    id: int
    name: str
    linear_org_id: str
    status: str
    editable: bool
    created_at: str
    updated_at: str


class LinearUserResponse(BaseModel):
    id: int
    keycloak_user_id: str
    linear_workspace_id: int
    status: str
    created_at: str
    updated_at: str
    workspace: LinearWorkspaceResponse


class LinearValidateWorkspaceResponse(BaseModel):
    name: str
    status: str
    message: str


linear_integration_router = APIRouter(prefix='/integration/linear')
token_manager = TokenManager()
linear_manager = LinearManager(token_manager)
redis_client = create_redis_client()


async def _handle_workspace_link_creation(
    user_id: str, linear_user_id: str, target_workspace: str
):
    """Handle the creation or reactivation of a workspace link for a user."""
    # Verify workspace exists and is active
    workspace = await linear_manager.integration_store.get_workspace_by_name(
        target_workspace
    )
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Workspace "{target_workspace}" not found',
        )

    if workspace.status.lower() != 'active':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Workspace "{target_workspace}" is not active',
        )

    # Check if user currently has an active workspace link
    existing_user = await linear_manager.integration_store.get_user_by_active_workspace(
        user_id
    )

    if existing_user:
        # User has an active link - check if it's to the same workspace
        if existing_user.linear_workspace_id == workspace.id:
            # Already linked to this workspace, nothing to do
            return
        else:
            # User is trying to link to a different workspace while having an active link
            # This is not allowed - they must unlink first
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='You already have an active workspace link. Please unlink from your current workspace before linking to a different one.',
            )

    # Check if user had a previous link to this specific workspace
    existing_link = (
        await linear_manager.integration_store.get_user_by_keycloak_id_and_workspace(
            user_id, workspace.id
        )
    )

    if existing_link:
        # Reactivate previous link to this workspace
        await linear_manager.integration_store.update_user_integration_status(
            user_id, 'active'
        )
    else:
        # Create new workspace link
        await linear_manager.integration_store.create_workspace_link(
            keycloak_user_id=user_id,
            linear_user_id=linear_user_id,
            linear_workspace_id=workspace.id,
        )


async def _validate_workspace_update_permissions(user_id: str, target_workspace: str):
    """Validate that user can update the target workspace."""
    workspace = await linear_manager.integration_store.get_workspace_by_name(
        target_workspace
    )
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Workspace "{target_workspace}" not found',
        )

    # Check if user is the admin of the workspace
    if workspace.admin_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='You do not have permission to update this workspace',
        )

    # Check if user's current link matches the workspace
    current_user_link = (
        await linear_manager.integration_store.get_user_by_active_workspace(user_id)
    )
    if current_user_link and current_user_link.linear_workspace_id != workspace.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='You can only update the workspace you are currently linked to',
        )

    return workspace


@linear_integration_router.post('/events')
async def linear_events(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Handle Linear webhook events."""
    # Check if Linear webhooks are enabled
    if not LINEAR_WEBHOOKS_ENABLED:
        logger.info(
            'Linear webhooks are disabled by LINEAR_WEBHOOKS_ENABLED environment variable'
        )
        return JSONResponse(
            status_code=200,
            content={'message': 'Linear webhooks are currently disabled.'},
        )

    try:
        signature_valid, signature, payload = await linear_manager.validate_request(
            request
        )

        if not signature_valid:
            logger.warning('[Linear] Invalid webhook signature')
            raise HTTPException(status_code=403, detail='Invalid webhook signature!')

        # Check for duplicate requests using Redis
        key = f'linear:{signature}'
        keyExists = redis_client.exists(key)
        if keyExists:
            logger.info(f'Received duplicate Linear webhook event: {signature}')
            return JSONResponse({'success': True})
        else:
            redis_client.setex(key, 60, 1)

        # Process the webhook
        message_payload = {'payload': payload}
        message = Message(source=SourceType.LINEAR, message=message_payload)

        background_tasks.add_task(linear_manager.receive_message, message)

        return JSONResponse({'success': True})

    except HTTPException:
        # Re-raise HTTP exceptions (like signature verification failures)
        raise
    except Exception as e:
        logger.exception(f'Error processing Linear webhook: {e}')
        return JSONResponse(
            status_code=500,
            content={'error': 'Internal server error processing webhook.'},
        )


@linear_integration_router.post('/workspaces')
async def create_linear_workspace(
    request: Request, workspace_data: LinearWorkspaceCreate
):
    """Create a new Linear workspace registration."""
    try:
        user_auth = cast(SaasUserAuth, await get_user_auth(request))
        user_id = await user_auth.get_user_id()
        user_email = await user_auth.get_user_email()

        state = str(uuid.uuid4())

        integration_session = {
            'operation_type': 'workspace_integration',
            'keycloak_user_id': user_id,
            'user_email': user_email,
            'target_workspace': workspace_data.workspace_name,
            'webhook_secret': workspace_data.webhook_secret,
            'svc_acc_email': workspace_data.svc_acc_email,
            'svc_acc_api_key': workspace_data.svc_acc_api_key,
            'is_active': workspace_data.is_active,
            'state': state,
        }

        created = redis_client.setex(
            state,
            60,
            json.dumps(integration_session),
        )

        if not created:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='Failed to create integration session',
            )

        auth_params = {
            'client_id': LINEAR_CLIENT_ID,
            'redirect_uri': LINEAR_REDIRECT_URI,
            'scope': LINEAR_SCOPES,
            'state': state,
            'response_type': 'code',
        }

        auth_url = f"{LINEAR_AUTH_URL}?{'&'.join([f'{k}={v}' for k, v in auth_params.items()])}"

        return JSONResponse(
            content={
                'success': True,
                'redirect': True,
                'authorizationUrl': auth_url,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error creating Linear workspace: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to create workspace',
        )


@linear_integration_router.post('/workspaces/link')
async def create_workspace_link(request: Request, link_data: LinearLinkCreate):
    """Register a user mapping to a Linear workspace."""
    try:
        user_auth = cast(SaasUserAuth, await get_user_auth(request))
        user_id = await user_auth.get_user_id()
        user_email = await user_auth.get_user_email()

        state = str(uuid.uuid4())

        integration_session = {
            'operation_type': 'workspace_link',
            'keycloak_user_id': user_id,
            'user_email': user_email,
            'target_workspace': link_data.workspace_name,
            'state': state,
        }

        created = redis_client.setex(
            state,
            60,
            json.dumps(integration_session),
        )

        if not created:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='Failed to create integration session',
            )

        auth_params = {
            'client_id': LINEAR_CLIENT_ID,
            'redirect_uri': LINEAR_REDIRECT_URI,
            'scope': LINEAR_SCOPES,
            'state': state,
            'response_type': 'code',
        }

        auth_url = f"{LINEAR_AUTH_URL}?{'&'.join([f'{k}={v}' for k, v in auth_params.items()])}"

        return JSONResponse(
            content={
                'success': True,
                'redirect': True,
                'authorizationUrl': auth_url,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error registering Linear user: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to register user',
        )


@linear_integration_router.get('/callback')
async def linear_callback(request: Request, code: str, state: str):
    integration_session_json = redis_client.get(state)
    if not integration_session_json:
        raise HTTPException(
            status_code=400, detail='No active integration session found.'
        )

    integration_session = json.loads(integration_session_json)

    # Security check: verify the state parameter
    if integration_session.get('state') != state:
        raise HTTPException(
            status_code=400, detail='State mismatch. Possible CSRF attack.'
        )

    token_payload = {
        'grant_type': 'authorization_code',
        'client_id': LINEAR_CLIENT_ID,
        'client_secret': LINEAR_CLIENT_SECRET,
        'code': code,
        'redirect_uri': LINEAR_REDIRECT_URI,
    }
    response = requests.post(LINEAR_TOKEN_URL, data=token_payload)
    if response.status_code != 200:
        raise HTTPException(
            status_code=400, detail=f'Error fetching token: {response.text}'
        )

    token_data = response.json()
    access_token = token_data['access_token']

    # Query Linear API to get workspace information
    headers = {'Authorization': f'Bearer {access_token}'}
    graphql_query = {
        'query': '{ viewer { id name email organization { id name urlKey } } }'
    }
    response = requests.post(LINEAR_GRAPHQL_URL, json=graphql_query, headers=headers)

    if response.status_code != 200:
        raise HTTPException(
            status_code=400, detail=f'Error fetching workspace: {response.text}'
        )

    workspace_data = response.json()
    workspace_info = (
        workspace_data.get('data', {}).get('viewer', {}).get('organization', {})
    )
    workspace_name = workspace_info.get('urlKey', '').lower()
    linear_org_id = workspace_info.get('id', '')

    target_workspace = integration_session.get('target_workspace')

    # Verify user has access to the target workspace
    if workspace_name != target_workspace.lower():
        raise HTTPException(
            status_code=401,
            detail=f'User is not authorized to access workspace: {target_workspace}',
        )

    user_id = integration_session['keycloak_user_id']
    linear_user_id = workspace_data.get('data', {}).get('viewer', {}).get('id')

    if integration_session.get('operation_type') == 'workspace_integration':
        workspace = await linear_manager.integration_store.get_workspace_by_name(
            target_workspace
        )
        if not workspace:
            # Create new workspace if it doesn't exist
            encrypted_webhook_secret = token_manager.encrypt_text(
                integration_session['webhook_secret']
            )
            encrypted_svc_acc_api_key = token_manager.encrypt_text(
                integration_session['svc_acc_api_key']
            )

            await linear_manager.integration_store.create_workspace(
                name=target_workspace,
                linear_org_id=linear_org_id,
                admin_user_id=integration_session['keycloak_user_id'],
                encrypted_webhook_secret=encrypted_webhook_secret,
                svc_acc_email=integration_session['svc_acc_email'],
                encrypted_svc_acc_api_key=encrypted_svc_acc_api_key,
                status='active' if integration_session['is_active'] else 'inactive',
            )

            # Create a workspace link for the user (admin automatically gets linked)
            await _handle_workspace_link_creation(
                user_id, linear_user_id, target_workspace
            )
        else:
            # Workspace exists - validate user can update it
            await _validate_workspace_update_permissions(user_id, target_workspace)

            encrypted_webhook_secret = token_manager.encrypt_text(
                integration_session['webhook_secret']
            )
            encrypted_svc_acc_api_key = token_manager.encrypt_text(
                integration_session['svc_acc_api_key']
            )

            # Update workspace details
            await linear_manager.integration_store.update_workspace(
                id=workspace.id,
                linear_org_id=linear_org_id,
                encrypted_webhook_secret=encrypted_webhook_secret,
                svc_acc_email=integration_session['svc_acc_email'],
                encrypted_svc_acc_api_key=encrypted_svc_acc_api_key,
                status='active' if integration_session['is_active'] else 'inactive',
            )

            await _handle_workspace_link_creation(
                user_id, linear_user_id, target_workspace
            )

        return RedirectResponse(
            url='/settings/integrations',
            status_code=status.HTTP_302_FOUND,
        )
    elif integration_session.get('operation_type') == 'workspace_link':
        await _handle_workspace_link_creation(user_id, linear_user_id, target_workspace)
        return RedirectResponse(
            url='/settings/integrations', status_code=status.HTTP_302_FOUND
        )
    else:
        raise HTTPException(status_code=400, detail='Invalid operation type')


@linear_integration_router.get(
    '/workspaces/link',
    response_model=LinearUserResponse,
)
async def get_current_workspace_link(request: Request):
    """Get current user's Linear integration details."""
    try:
        user_auth = cast(SaasUserAuth, await get_user_auth(request))
        user_id = await user_auth.get_user_id()
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='User not authenticated',
            )

        user = await linear_manager.integration_store.get_user_by_active_workspace(
            user_id
        )
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='User is not registered for Linear integration',
            )

        workspace = await linear_manager.integration_store.get_workspace_by_id(
            user.linear_workspace_id
        )
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Workspace not found for the user',
            )

        return LinearUserResponse(
            id=user.id,
            keycloak_user_id=user.keycloak_user_id,
            linear_workspace_id=user.linear_workspace_id,
            status=user.status,
            created_at=user.created_at.isoformat(),
            updated_at=user.updated_at.isoformat(),
            workspace=LinearWorkspaceResponse(
                id=workspace.id,
                name=workspace.name,
                linear_org_id=workspace.linear_org_id,
                status=workspace.status,
                editable=workspace.admin_user_id == user.keycloak_user_id,
                created_at=workspace.created_at.isoformat(),
                updated_at=workspace.updated_at.isoformat(),
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error retrieving Linear user: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve user',
        )


@linear_integration_router.post('/workspaces/unlink')
async def unlink_workspace(request: Request):
    """Unlink user from Linear integration by setting status to inactive."""
    try:
        user_auth = cast(SaasUserAuth, await get_user_auth(request))
        user_id = await user_auth.get_user_id()
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='User not authenticated',
            )

        user = await linear_manager.integration_store.get_user_by_active_workspace(
            user_id
        )
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='User is not registered for Linear integration',
            )

        workspace = await linear_manager.integration_store.get_workspace_by_id(
            user.linear_workspace_id
        )
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Workspace not found for the user',
            )

        if workspace.admin_user_id == user_id:
            await linear_manager.integration_store.deactivate_workspace(
                workspace_id=workspace.id,
            )
        else:
            await linear_manager.integration_store.update_user_integration_status(
                user_id, 'inactive'
            )

        return JSONResponse({'success': True})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error unlinking Linear user: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to unlink user',
        )


@linear_integration_router.get(
    '/workspaces/validate/{workspace_name}',
    response_model=LinearValidateWorkspaceResponse,
)
async def validate_workspace_integration(request: Request, workspace_name: str):
    """Validate if the workspace has an active Linear integration."""
    try:
        # Validate workspace_name format
        if not re.match(r'^[a-zA-Z0-9_.-]+$', workspace_name):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='workspace_name can only contain alphanumeric characters, hyphens, underscores, and periods',
            )

        user_auth = cast(SaasUserAuth, await get_user_auth(request))
        user_email = await user_auth.get_user_email()
        if not user_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Unable to retrieve user email',
            )

        # Check if workspace exists
        workspace = await linear_manager.integration_store.get_workspace_by_name(
            workspace_name
        )
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace with name '{workspace_name}' not found",
            )

        # Check if workspace is active
        if workspace.status.lower() != 'active':
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace '{workspace.name}' is not active",
            )

        return LinearValidateWorkspaceResponse(
            name=workspace.name,
            status=workspace.status,
            message='Workspace integration is active',
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error validating Linear workspace: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to validate workspace',
        )
