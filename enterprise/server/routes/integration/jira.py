import hashlib
import hmac
import json
import os
import re
import uuid
from urllib.parse import urlparse

import requests
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from integrations.jira.jira_manager import JiraManager
from integrations.models import Message, SourceType
from integrations.utils import HOST_URL
from pydantic import BaseModel, Field, field_validator
from server.auth.constants import JIRA_CLIENT_ID, JIRA_CLIENT_SECRET
from server.auth.saas_user_auth import SaasUserAuth
from server.auth.token_manager import TokenManager
from storage.jira_workspace import JiraWorkspace
from storage.redis import create_redis_client

from openhands.core.logger import openhands_logger as logger
from openhands.server.user_auth.user_auth import get_user_auth

# Environment variable to disable Jira webhooks
JIRA_WEBHOOKS_ENABLED = os.environ.get('JIRA_WEBHOOKS_ENABLED', '0') in (
    '1',
    'true',
)
JIRA_REDIRECT_URI = f'{HOST_URL}/integration/jira/callback'
JIRA_SCOPES = 'read:me read:jira-user read:jira-work'
JIRA_AUTH_URL = 'https://auth.atlassian.com/authorize'
JIRA_TOKEN_URL = 'https://auth.atlassian.com/oauth/token'
JIRA_RESOURCES_URL = 'https://api.atlassian.com/oauth/token/accessible-resources'
JIRA_USER_INFO_URL = 'https://api.atlassian.com/me'


# Request/Response models
class JiraWorkspaceCreate(BaseModel):
    workspace_name: str = Field(..., description='Workspace display name')
    webhook_secret: str = Field(..., description='Webhook secret for verification')
    svc_acc_email: str = Field(..., description='Service account email')
    svc_acc_api_key: str = Field(..., description='Service account API token')
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


class JiraLinkCreate(BaseModel):
    workspace_name: str = Field(
        ..., description='Name of the Jira workspace to link to'
    )

    @field_validator('workspace_name')
    @classmethod
    def validate_workspace(cls, v):
        if not re.match(r'^[a-zA-Z0-9_.-]+$', v):
            raise ValueError(
                'workspace can only contain alphanumeric characters, hyphens, underscores, and periods'
            )
        return v


class JiraWorkspaceResponse(BaseModel):
    id: int
    name: str
    jira_cloud_id: str
    status: str
    editable: bool
    created_at: str
    updated_at: str


class JiraUserResponse(BaseModel):
    id: int
    keycloak_user_id: str
    jira_workspace_id: int
    status: str
    created_at: str
    updated_at: str
    workspace: JiraWorkspaceResponse


class JiraValidateWorkspaceResponse(BaseModel):
    name: str
    status: str
    message: str


jira_integration_router = APIRouter(prefix='/integration/jira')
token_manager = TokenManager()
jira_manager = JiraManager(token_manager)
redis_client = create_redis_client()


async def verify_jira_signature(body: bytes, signature: str, payload: dict):
    """
    Verify Jira webhook signature.

    Args:
        body: Raw request body bytes
        signature: Signature from x-hub-signature header (format: "sha256=<hash>")
        payload: Parsed JSON payload from webhook

    Raises:
        HTTPException: 403 if signature verification fails or workspace is invalid

    Returns:
        None (raises exception on failure)
    """

    if not signature:
        raise HTTPException(
            status_code=403, detail='x-hub-signature header is missing!'
        )

    workspace_name = jira_manager.get_workspace_name_from_payload(payload)
    if workspace_name is None:
        logger.warning('[Jira] No workspace name found in webhook payload')
        raise HTTPException(
            status_code=403, detail='Workspace name not found in payload'
        )

    workspace: (
        JiraWorkspace | None
    ) = await jira_manager.integration_store.get_workspace_by_name(workspace_name)

    if workspace is None:
        logger.warning(f'[Jira] Could not identify workspace {workspace_name}')
        raise HTTPException(status_code=403, detail='Unidentified workspace')

    if workspace.status != 'active':
        logger.warning(
            '[Jira] Workspace is inactive',
            extra={
                'jira_workspace_id': workspace.id,
                'parsed_workspace_name': workspace.name,
                'status': workspace.status,
            },
        )

        raise HTTPException(status_code=403, detail='Workspace is inactive')

    webhook_secret = token_manager.decrypt_text(workspace.webhook_secret)
    expected_signature = hmac.new(
        webhook_secret.encode(), body, hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected_signature, signature):
        raise HTTPException(status_code=403, detail="Request signatures didn't match!")


async def _handle_workspace_link_creation(
    user_id: str, jira_user_id: str, target_workspace: str
):
    """Handle the creation or reactivation of a workspace link for a user."""
    # Verify workspace exists and is active
    workspace = await jira_manager.integration_store.get_workspace_by_name(
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
    existing_user = await jira_manager.integration_store.get_user_by_active_workspace(
        user_id
    )

    if existing_user:
        # User has an active link - check if it's to the same workspace
        if existing_user.jira_workspace_id == workspace.id:
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
        await jira_manager.integration_store.get_user_by_keycloak_id_and_workspace(
            user_id, workspace.id
        )
    )

    if existing_link:
        # Reactivate previous link to this workspace
        await jira_manager.integration_store.update_user_integration_status(
            user_id, 'active'
        )
    else:
        # Create new workspace link
        await jira_manager.integration_store.create_workspace_link(
            keycloak_user_id=user_id,
            jira_user_id=jira_user_id,
            jira_workspace_id=workspace.id,
        )


async def _validate_workspace_update_permissions(user_id: str, target_workspace: str):
    """Validate that user can update the target workspace."""
    workspace = await jira_manager.integration_store.get_workspace_by_name(
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
        await jira_manager.integration_store.get_user_by_active_workspace(user_id)
    )
    if current_user_link and current_user_link.jira_workspace_id != workspace.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='You can only update the workspace you are currently linked to',
        )

    return workspace


@jira_integration_router.post('/events')
async def jira_events(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature: str = Header(None),
):
    """Handle Jira webhook events."""
    # Check if Jira webhooks are enabled
    if not JIRA_WEBHOOKS_ENABLED:
        logger.info('[Jira] Webhooks are disabled')
        return JSONResponse(
            status_code=200,
            content={'message': 'Jira webhooks are disabled'},
        )

    try:
        parts = x_hub_signature.split('=', 1)
        if not (len(parts) == 2 and parts[1]):
            raise HTTPException(status_code=403, detail='Malformed x-hub-signature!')

        signature = parts[1]
        body = await request.body()
        payload = await request.json()

        await verify_jira_signature(body, signature, payload)

        # Check for duplicate requests using Redis
        key = f'jira:{signature}'
        keyExists = redis_client.exists(key)
        if keyExists:
            logger.info(f'Received duplicate Jira webhook event: {signature}')
            return JSONResponse({'success': True})
        else:
            logger.info(f'Processing new Jira webhook event: {signature}')
            redis_client.setex(key, 300, '1')

        # Process the webhook
        message_payload = {'payload': payload}
        message = Message(source=SourceType.JIRA, message=message_payload)

        background_tasks.add_task(jira_manager.receive_message, message)

        return JSONResponse({'success': True})

    except HTTPException:
        # Re-raise HTTP exceptions (like signature verification failures)
        raise
    except Exception as e:
        logger.exception(f'Error processing Jira webhook: {e}')
        return JSONResponse(
            status_code=500,
            content={'error': 'Internal server error processing webhook.'},
        )


@jira_integration_router.post('/workspaces')
async def create_jira_workspace(request: Request, workspace_data: JiraWorkspaceCreate):
    """Create a new Jira workspace registration."""
    try:
        user_auth: SaasUserAuth = await get_user_auth(request)
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
            'audience': 'api.atlassian.com',
            'client_id': JIRA_CLIENT_ID,
            'scope': JIRA_SCOPES,
            'redirect_uri': JIRA_REDIRECT_URI,
            'state': state,
            'response_type': 'code',
            'prompt': 'consent',
        }

        auth_url = (
            f"{JIRA_AUTH_URL}?{'&'.join([f'{k}={v}' for k, v in auth_params.items()])}"
        )

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
        logger.exception(f'Error creating Jira workspace: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to create workspace',
        )


@jira_integration_router.post('/workspaces/link')
async def create_workspace_link(request: Request, link_data: JiraLinkCreate):
    """Register a user mapping to a Jira workspace."""
    try:
        user_auth: SaasUserAuth = await get_user_auth(request)
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
            'audience': 'api.atlassian.com',
            'client_id': JIRA_CLIENT_ID,
            'scope': JIRA_SCOPES,
            'redirect_uri': JIRA_REDIRECT_URI,
            'state': state,
            'response_type': 'code',
            'prompt': 'consent',
        }
        auth_url = (
            f"{JIRA_AUTH_URL}?{'&'.join([f'{k}={v}' for k, v in auth_params.items()])}"
        )

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
        logger.exception(f'Error registering Jira user: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to register user',
        )


@jira_integration_router.get('/callback')
async def jira_callback(request: Request, code: str, state: str):
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
        'client_id': JIRA_CLIENT_ID,
        'client_secret': JIRA_CLIENT_SECRET,
        'code': code,
        'redirect_uri': JIRA_REDIRECT_URI,
    }
    response = requests.post(JIRA_TOKEN_URL, json=token_payload)
    if response.status_code != 200:
        raise HTTPException(
            status_code=400, detail=f'Error fetching token: {response.text}'
        )

    token_data = response.json()
    access_token = token_data['access_token']

    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(JIRA_RESOURCES_URL, headers=headers)

    if response.status_code != 200:
        raise HTTPException(
            status_code=400, detail=f'Error fetching resources: {response.text}'
        )

    workspaces = response.json()

    logger.info(f'Jira workspaces: {workspaces}')

    target_workspace = integration_session.get('target_workspace')

    # Filter workspaces based on the target workspace
    target_workspace_data = next(
        (
            ws
            for ws in workspaces
            if urlparse(ws.get('url', '')).hostname == target_workspace
        ),
        None,
    )
    if not target_workspace_data:
        raise HTTPException(
            status_code=401,
            detail=f'User is not authorized to access workspace: {target_workspace}',
        )

    jira_cloud_id = target_workspace_data.get('id', '')

    jira_user_response = requests.get(JIRA_USER_INFO_URL, headers=headers)
    if jira_user_response.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail=f'Error fetching user info: {jira_user_response.text}',
        )

    jira_user_info = jira_user_response.json()
    jira_user_id = jira_user_info.get('account_id')

    user_id = integration_session['keycloak_user_id']

    if integration_session.get('operation_type') == 'workspace_integration':
        workspace = await jira_manager.integration_store.get_workspace_by_name(
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

            await jira_manager.integration_store.create_workspace(
                name=target_workspace,
                jira_cloud_id=jira_cloud_id,
                admin_user_id=integration_session['keycloak_user_id'],
                encrypted_webhook_secret=encrypted_webhook_secret,
                svc_acc_email=integration_session['svc_acc_email'],
                encrypted_svc_acc_api_key=encrypted_svc_acc_api_key,
                status='active' if integration_session['is_active'] else 'inactive',
            )

            # Create a workspace link for the user (admin automatically gets linked)
            await _handle_workspace_link_creation(
                user_id, jira_user_id, target_workspace
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
            await jira_manager.integration_store.update_workspace(
                id=workspace.id,
                jira_cloud_id=jira_cloud_id,
                encrypted_webhook_secret=encrypted_webhook_secret,
                svc_acc_email=integration_session['svc_acc_email'],
                encrypted_svc_acc_api_key=encrypted_svc_acc_api_key,
                status='active' if integration_session['is_active'] else 'inactive',
            )

            await _handle_workspace_link_creation(
                user_id, jira_user_id, target_workspace
            )

        return RedirectResponse(
            url='/settings/integrations', status_code=status.HTTP_302_FOUND
        )
    elif integration_session.get('operation_type') == 'workspace_link':
        await _handle_workspace_link_creation(user_id, jira_user_id, target_workspace)
        return RedirectResponse(
            url='/settings/integrations', status_code=status.HTTP_302_FOUND
        )
    else:
        raise HTTPException(status_code=400, detail='Invalid operation type')


@jira_integration_router.get(
    '/workspaces/link',
    response_model=JiraUserResponse,
)
async def get_current_workspace_link(request: Request):
    """Get current user's Jira integration details."""
    try:
        user_auth: SaasUserAuth = await get_user_auth(request)
        user_id = await user_auth.get_user_id()

        user = await jira_manager.integration_store.get_user_by_active_workspace(
            user_id
        )
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='User is not registered for Jira integration',
            )

        workspace = await jira_manager.integration_store.get_workspace_by_id(
            user.jira_workspace_id
        )
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Workspace not found for the user',
            )

        return JiraUserResponse(
            id=user.id,
            keycloak_user_id=user.keycloak_user_id,
            jira_workspace_id=user.jira_workspace_id,
            status=user.status,
            created_at=user.created_at.isoformat(),
            updated_at=user.updated_at.isoformat(),
            workspace=JiraWorkspaceResponse(
                id=workspace.id,
                name=workspace.name,
                jira_cloud_id=workspace.jira_cloud_id,
                status=workspace.status,
                editable=workspace.admin_user_id == user.keycloak_user_id,
                created_at=workspace.created_at.isoformat(),
                updated_at=workspace.updated_at.isoformat(),
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error retrieving Jira user: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve user',
        )


@jira_integration_router.post('/workspaces/unlink')
async def unlink_workspace(request: Request):
    """Unlink user from Jira integration by setting status to inactive."""
    try:
        user_auth: SaasUserAuth = await get_user_auth(request)
        user_id = await user_auth.get_user_id()

        user = await jira_manager.integration_store.get_user_by_active_workspace(
            user_id
        )
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='User is not registered for Jira integration',
            )

        workspace = await jira_manager.integration_store.get_workspace_by_id(
            user.jira_workspace_id
        )
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Workspace not found for the user',
            )

        if workspace.admin_user_id == user_id:
            await jira_manager.integration_store.deactivate_workspace(
                workspace_id=workspace.id,
            )
        else:
            await jira_manager.integration_store.update_user_integration_status(
                user_id, 'inactive'
            )

        return JSONResponse({'success': True})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error unlinking Jira user: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to unlink user',
        )


@jira_integration_router.get(
    '/workspaces/validate/{workspace_name}',
    response_model=JiraValidateWorkspaceResponse,
)
async def validate_workspace_integration(request: Request, workspace_name: str):
    """Validate if the user's organization has an active Jira integration."""
    try:
        # Validate workspace_name format
        if not re.match(r'^[a-zA-Z0-9_.-]+$', workspace_name):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='workspace_name can only contain alphanumeric characters, hyphens, underscores, and periods',
            )

        user_auth: SaasUserAuth = await get_user_auth(request)
        user_email = await user_auth.get_user_email()
        if not user_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Unable to retrieve user email',
            )

        # Check if workspace exists
        workspace = await jira_manager.integration_store.get_workspace_by_name(
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

        return JiraValidateWorkspaceResponse(
            name=workspace.name,
            status=workspace.status,
            message='Workspace integration is active',
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error validating Jira organization: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to validate organization',
        )
