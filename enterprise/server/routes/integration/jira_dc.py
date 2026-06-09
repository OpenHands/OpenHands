import json
import os
import re
import secrets
import uuid
from typing import cast
from urllib.parse import urlencode, urlparse
from uuid import UUID

import httpx
from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import JSONResponse, RedirectResponse
from integrations.jira_dc.jira_dc_manager import JiraDcManager
from integrations.jira_dc.jira_dc_service_account import (
    get_jira_dc_managed_service_account,
    get_jira_dc_service_account_config_error,
)
from integrations.models import Message, SourceType
from pydantic import BaseModel, Field, field_validator
from server.auth.constants import (
    AUTOMATION_EVENT_FORWARDING_ENABLED,
    JIRA_DC_BASE_URL,
    JIRA_DC_CLIENT_ID,
    JIRA_DC_CLIENT_SECRET,
    JIRA_DC_ENABLE_OAUTH,
)
from server.auth.saas_user_auth import SaasUserAuth
from server.auth.token_manager import TokenManager
from server.constants import WEB_HOST
from server.services.automation_event_service import AutomationEventService
from storage.redis import get_redis_client

from openhands.app_server.user_auth.user_auth import get_user_auth
from openhands.app_server.utils.logger import openhands_logger as logger

# Environment variable to disable Jira DC webhooks
JIRA_DC_WEBHOOKS_ENABLED = os.environ.get('JIRA_DC_WEBHOOKS_ENABLED', '0') in (
    '1',
    'true',
)
JIRA_DC_REDIRECT_URI = f'https://{WEB_HOST}/integration/jira-dc/callback'
JIRA_DC_SCOPES = 'WRITE'
JIRA_DC_AUTH_URL = f'{JIRA_DC_BASE_URL}/rest/oauth2/latest/authorize'
JIRA_DC_TOKEN_URL = f'{JIRA_DC_BASE_URL}/rest/oauth2/latest/token'
JIRA_DC_USER_INFO_URL = f'{JIRA_DC_BASE_URL}/rest/api/2/myself'
JIRA_DC_OAUTH_STATE_TTL_SECONDS = 600


# Request/Response models
class JiraDcWorkspaceCreate(BaseModel):
    workspace_name: str = Field(..., description='Workspace display name')
    webhook_secret: str | None = Field(
        default=None,
        description=(
            'Webhook secret used to verify inbound signatures. Optional: when '
            'omitted the server generates a random one. The frontend supplies it '
            'only in manual mode (so the admin can copy it into Jira); in '
            'auto-enroll mode it is left blank and generated here.'
        ),
    )
    svc_acc_email: str | None = Field(default=None, description='Service account email')
    svc_acc_api_key: str | None = Field(
        default=None,
        description=(
            'Service account API token/PAT. Required when creating a new '
            'workspace; optional on update — omit/leave blank to keep the '
            'stored value (so admins never have to re-paste it to edit).'
        ),
    )
    admin_api_key: str | None = Field(
        default=None,
        description=(
            'Optional Jira admin PAT used once to auto-register the webhook. '
            'Used transiently for the enrollment call and never stored.'
        ),
    )
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
        if v is None or v == '':
            return v
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError('svc_acc_email must be a valid email address')
        return v

    @field_validator('webhook_secret')
    @classmethod
    def validate_webhook_secret(cls, v):
        if v is not None and ' ' in v:
            raise ValueError('webhook_secret cannot contain spaces')
        return v

    @field_validator('svc_acc_api_key')
    @classmethod
    def validate_svc_acc_api_key(cls, v):
        if v is not None and ' ' in v:
            raise ValueError('svc_acc_api_key cannot contain spaces')
        return v


class JiraDcLinkCreate(BaseModel):
    workspace_name: str = Field(
        ..., description='Name of the Jira DC workspace to link to'
    )

    @field_validator('workspace_name')
    @classmethod
    def validate_workspace(cls, v):
        if not re.match(r'^[a-zA-Z0-9_.-]+$', v):
            raise ValueError(
                'workspace can only contain alphanumeric characters, hyphens, underscores, and periods'
            )
        return v


class JiraDcWorkspaceStatusUpdate(BaseModel):
    workspace_name: str = Field(..., description='Workspace display name')
    is_active: bool = Field(
        ...,
        description='Indicates if the workspace integration should be active',
    )

    @field_validator('workspace_name')
    @classmethod
    def validate_workspace_name(cls, v):
        if not re.match(r'^[a-zA-Z0-9_.-]+$', v):
            raise ValueError(
                'workspace_name can only contain alphanumeric characters, hyphens, underscores, and periods'
            )
        return v


class JiraDcWorkspaceResponse(BaseModel):
    id: int
    name: str
    status: str
    editable: bool
    events_url: str
    # Service-account email is non-secret and is returned so the configure form
    # can pre-fill it when editing. The service-account PAT is never returned.
    svc_acc_email: str | None = None
    created_at: str
    updated_at: str


class JiraDcUserResponse(BaseModel):
    id: int
    keycloak_user_id: str
    jira_dc_workspace_id: int
    status: str
    created_at: str
    updated_at: str
    workspace: JiraDcWorkspaceResponse


class JiraDcValidateWorkspaceResponse(BaseModel):
    name: str
    status: str
    message: str


jira_dc_integration_router = APIRouter(prefix='/integration/jira-dc')
token_manager = TokenManager()
jira_dc_manager = JiraDcManager(token_manager)
automation_event_service = AutomationEventService(token_manager)
redis_client = get_redis_client()


def _jira_dc_events_url(workspace_id: int) -> str:
    return f'https://{WEB_HOST}/integration/jira-dc/connections/{workspace_id}/events'


async def _handle_workspace_link_creation(
    user_id: str,
    jira_dc_user_id: str,
    target_workspace: str,
    require_active_workspace: bool = True,
):
    """Handle the creation or reactivation of a workspace link for a user."""
    # Verify workspace exists and is active
    workspace = await jira_dc_manager.integration_store.get_workspace_by_name(
        target_workspace
    )
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Workspace "{target_workspace}" not found',
        )

    if require_active_workspace and workspace.status.lower() != 'active':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Workspace "{target_workspace}" is not active',
        )

    # Check if user currently has an active workspace link
    existing_user = (
        await jira_dc_manager.integration_store.get_user_by_active_workspace(user_id)
    )

    if existing_user:
        # User has an active link - check if it's to the same workspace
        if existing_user.jira_dc_workspace_id == workspace.id:
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
        await jira_dc_manager.integration_store.get_user_by_keycloak_id_and_workspace(
            user_id, workspace.id
        )
    )

    if existing_link:
        # Reactivate previous link to this workspace
        await jira_dc_manager.integration_store.update_user_integration_status(
            user_id, workspace.id, 'active'
        )
    else:
        # Create new workspace link
        await jira_dc_manager.integration_store.create_workspace_link(
            keycloak_user_id=user_id,
            jira_dc_user_id=jira_dc_user_id,
            jira_dc_workspace_id=workspace.id,
        )


async def _validate_workspace_update_permissions(user_id: str, target_workspace: str):
    """Validate that user can update the target workspace."""
    workspace = await jira_dc_manager.integration_store.get_workspace_by_name(
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
        await jira_dc_manager.integration_store.get_user_by_active_workspace(user_id)
    )
    if current_user_link and current_user_link.jira_dc_workspace_id != workspace.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='You can only update the workspace you are currently linked to',
        )

    return workspace


async def _process_jira_dc_event(
    request: Request,
    background_tasks: BackgroundTasks,
    workspace_id: int | None = None,
):
    """Handle Jira DC webhook events."""
    # Check if Jira DC webhooks are enabled
    if not JIRA_DC_WEBHOOKS_ENABLED:
        return JSONResponse(
            status_code=200,
            content={'message': 'Jira DC webhooks are disabled.'},
        )

    try:
        (
            signature_valid,
            signature,
            payload,
            workspace,
        ) = await jira_dc_manager.validate_request_context(
            request,
            workspace_id=workspace_id,
        )

        if not signature_valid:
            logger.warning('[Jira DC] Invalid webhook signature')
            raise HTTPException(status_code=403, detail='Invalid webhook signature!')

        # Check for duplicate requests using Redis
        key_workspace_id = workspace.id if workspace else workspace_id
        key = (
            f'jira_dc:{key_workspace_id}:{signature}'
            if key_workspace_id
            else f'jira_dc:{signature}'
        )
        keyExists = redis_client.exists(key)
        if keyExists:
            logger.info(f'Received duplicate Jira DC webhook event: {signature}')
            return JSONResponse({'success': True})
        else:
            redis_client.setex(key, 120, 1)

        if AUTOMATION_EVENT_FORWARDING_ENABLED and payload and workspace:
            if workspace.org_id:
                background_tasks.add_task(
                    automation_event_service.forward_jira_dc_event,
                    org_id=workspace.org_id,
                    payload=payload,
                    workspace_name=workspace.name,
                    connection_id=workspace.id,
                    delivery_id=signature,
                )
            else:
                logger.warning(
                    '[Jira DC] Workspace %s has no org_id; skipping automation forwarding',
                    workspace.id,
                )

        # Process the webhook
        message_payload = {'payload': payload}
        message = Message(source=SourceType.JIRA_DC, message=message_payload)

        background_tasks.add_task(jira_dc_manager.receive_message, message)

        return JSONResponse({'success': True})
    except HTTPException:
        # Re-raise HTTP exceptions (like signature verification failures)
        raise
    except Exception as e:
        logger.exception(f'Error processing Jira DC webhook: {e}')
        return JSONResponse(
            status_code=500,
            content={'error': 'Internal server error processing webhook.'},
        )


@jira_dc_integration_router.post('/connections/{workspace_id}/events')
async def jira_dc_connection_events(
    workspace_id: int,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Handle Jira DC webhook events for a specific workspace connection."""
    return await _process_jira_dc_event(
        request,
        background_tasks,
        workspace_id=workspace_id,
    )


async def _maybe_register_webhook(
    admin_api_key: str | None,
    base_api_url: str,
    webhook_secret: str,
    workspace_id: int,
) -> bool:
    """Best-effort auto-enrollment of the OpenHands webhook in Jira.

    When an admin PAT is supplied, register (or update) the webhook via
    JiraDcManager.register_webhook. The PAT is used only for this call and never
    stored. Returns True on success; False when skipped or failed -- workspace
    creation must never fail because enrollment did.
    """
    if not admin_api_key:
        return False
    try:
        await jira_dc_manager.register_webhook(
            base_api_url=base_api_url,
            admin_api_key=admin_api_key,
            events_url=_jira_dc_events_url(workspace_id),
            secret=webhook_secret,
        )
        logger.info('[Jira DC] Auto-enrolled webhook during workspace configure')
        return True
    except Exception as e:
        logger.warning(f'[Jira DC] Webhook auto-enrollment failed: {e}')
        return False


async def _maybe_delete_webhook(
    admin_api_key: str | None, base_api_url: str, workspace_id: int
) -> bool:
    """Best-effort removal of the OpenHands webhook from Jira during teardown.

    Symmetric with :func:`_maybe_register_webhook`. When an admin PAT is
    supplied, delete the webhook via JiraDcManager.delete_webhook. The PAT is
    used only for this call and never stored. Returns True on success; False
    when skipped or failed -- teardown (workspace deactivation) must never fail
    because the Jira-side cleanup did.
    """
    if not admin_api_key:
        return False
    try:
        return await jira_dc_manager.delete_webhook(
            base_api_url=base_api_url,
            admin_api_key=admin_api_key,
            events_url=_jira_dc_events_url(workspace_id),
        )
    except Exception as e:
        logger.warning(f'[Jira DC] Webhook removal failed: {e}')
        return False


def _resolve_webhook_secret(
    submitted_secret: str | None, encrypted_existing_secret: str | None = None
) -> str:
    """Resolve the secret to persist and use for optional webhook enrollment.

    First-time auto-enroll omits ``submitted_secret`` so we generate a new
    random value. Existing-workspace updates that omit it must preserve the
    stored secret; otherwise an ordinary service-account update would silently
    break the already-installed Jira webhook.
    """
    if submitted_secret:
        return submitted_secret
    if encrypted_existing_secret:
        return token_manager.decrypt_text(encrypted_existing_secret)
    return secrets.token_urlsafe(32)


def _resolve_submitted_service_account(
    workspace_data: JiraDcWorkspaceCreate,
) -> tuple[str, str]:
    """Resolve service-account values to persist for this workspace request."""
    managed_service_account = get_jira_dc_managed_service_account()
    if managed_service_account:
        return managed_service_account.email, managed_service_account.api_key

    svc_acc_email = (workspace_data.svc_acc_email or '').strip()
    if not svc_acc_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='A service account email is required when configuring Jira DC in OpenHands.',
        )

    return svc_acc_email, (workspace_data.svc_acc_api_key or '').strip()


@jira_dc_integration_router.post('/workspaces')
async def create_jira_dc_workspace(
    request: Request, workspace_data: JiraDcWorkspaceCreate
):
    """Create a new Jira DC workspace registration."""
    try:
        service_account_config_error = get_jira_dc_service_account_config_error()
        if service_account_config_error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=service_account_config_error,
            )

        user_auth = cast(SaasUserAuth, await get_user_auth(request))
        user_id = await user_auth.get_user_id()
        user_email = await user_auth.get_user_email()
        effective_org_id = await user_auth.get_effective_org_id()

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='User ID not found',
            )

        # Look up the workspace once; reused by both the OAuth and email paths.
        existing_workspace = (
            await jira_dc_manager.integration_store.get_workspace_by_name(
                workspace_data.workspace_name
            )
        )
        svc_acc_email, provided_api_key = _resolve_submitted_service_account(
            workspace_data
        )
        # The service-account PAT is required to create a NEW workspace, but is
        # optional when editing one (blank = keep the stored token), so admins
        # never have to re-paste it just to change other fields.
        if existing_workspace is None and not provided_api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='A service account PAT is required when configuring a new workspace.',
            )

        if JIRA_DC_ENABLE_OAUTH:
            if existing_workspace:
                await _validate_workspace_update_permissions(
                    user_id, workspace_data.workspace_name
                )
            resolved_webhook_secret = _resolve_webhook_secret(
                workspace_data.webhook_secret,
                existing_workspace.webhook_secret if existing_workspace else None,
            )

            # OAuth flow enabled - create session and redirect to OAuth
            state = str(uuid.uuid4())

            integration_session = {
                'operation_type': 'workspace_integration',
                'keycloak_user_id': user_id,
                'org_id': str(effective_org_id) if effective_org_id else None,
                'user_email': user_email,
                'target_workspace': workspace_data.workspace_name,
                'webhook_secret': resolved_webhook_secret,
                'svc_acc_email': svc_acc_email,
                # Empty when editing without changing the PAT; the callback then
                # keeps the workspace's stored token instead of overwriting it.
                'svc_acc_api_key': provided_api_key,
                'admin_api_key': workspace_data.admin_api_key,
                'is_active': workspace_data.is_active,
                'state': state,
            }

            created = redis_client.setex(
                state,
                JIRA_DC_OAUTH_STATE_TTL_SECONDS,
                json.dumps(integration_session),
            )

            if not created:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail='Failed to create integration session',
                )

            auth_params = {
                'client_id': JIRA_DC_CLIENT_ID,
                'scope': JIRA_DC_SCOPES,
                'redirect_uri': JIRA_DC_REDIRECT_URI,
                'state': state,
                'response_type': 'code',
            }

            auth_url = f'{JIRA_DC_AUTH_URL}?{urlencode(auth_params)}'

            return JSONResponse(
                content={
                    'success': True,
                    'redirect': True,
                    'authorizationUrl': auth_url,
                }
            )
        else:
            # OAuth flow disabled - directly create workspace
            workspace = existing_workspace
            if not workspace:
                resolved_webhook_secret = _resolve_webhook_secret(
                    workspace_data.webhook_secret
                )
                # Create new workspace if it doesn't exist
                encrypted_webhook_secret = token_manager.encrypt_text(
                    resolved_webhook_secret
                )
                encrypted_new_svc_acc_api_key = token_manager.encrypt_text(
                    provided_api_key
                )

                workspace = await jira_dc_manager.integration_store.create_workspace(
                    name=workspace_data.workspace_name,
                    admin_user_id=user_id,
                    org_id=effective_org_id,
                    encrypted_webhook_secret=encrypted_webhook_secret,
                    svc_acc_email=svc_acc_email,
                    encrypted_svc_acc_api_key=encrypted_new_svc_acc_api_key,
                    status='active' if workspace_data.is_active else 'inactive',
                )

                # Create a workspace link for the user (admin automatically gets linked)
                if workspace_data.is_active:
                    await _handle_workspace_link_creation(
                        user_id, 'unavailable', workspace.name
                    )
                else:
                    await _handle_workspace_link_creation(
                        user_id,
                        'unavailable',
                        workspace.name,
                        require_active_workspace=False,
                    )
            else:
                # Workspace exists - validate user can update it
                await _validate_workspace_update_permissions(
                    user_id, workspace_data.workspace_name
                )
                resolved_webhook_secret = _resolve_webhook_secret(
                    workspace_data.webhook_secret, workspace.webhook_secret
                )

                encrypted_webhook_secret = token_manager.encrypt_text(
                    resolved_webhook_secret
                )
                # None when the admin left the PAT blank on edit → the store
                # preserves the existing encrypted token.
                updated_encrypted_svc_acc_api_key = (
                    token_manager.encrypt_text(provided_api_key)
                    if provided_api_key
                    else None
                )

                # Update workspace details
                workspace = await jira_dc_manager.integration_store.update_workspace(
                    id=workspace.id,
                    org_id=effective_org_id,
                    encrypted_webhook_secret=encrypted_webhook_secret,
                    svc_acc_email=svc_acc_email,
                    encrypted_svc_acc_api_key=updated_encrypted_svc_acc_api_key,
                    status='active' if workspace_data.is_active else 'inactive',
                )

                if workspace_data.is_active:
                    await _handle_workspace_link_creation(
                        user_id, 'unavailable', workspace.name
                    )
                else:
                    await _handle_workspace_link_creation(
                        user_id,
                        'unavailable',
                        workspace.name,
                        require_active_workspace=False,
                    )

            webhook_enrolled = await _maybe_register_webhook(
                workspace_data.admin_api_key,
                f'https://{workspace_data.workspace_name}',
                resolved_webhook_secret,
                workspace.id,
            )
            if workspace_data.admin_api_key and not webhook_enrolled:
                await jira_dc_manager.integration_store.update_workspace(
                    id=workspace.id,
                    status='inactive',
                )

            return JSONResponse(
                content={
                    'success': True,
                    'redirect': False,
                    'authorizationUrl': '',
                    'webhookEnrolled': webhook_enrolled,
                    'eventsUrl': _jira_dc_events_url(workspace.id),
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error creating Jira DC workspace: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to create workspace',
        )


@jira_dc_integration_router.post('/workspaces/status')
async def update_jira_dc_workspace_status(
    request: Request, workspace_data: JiraDcWorkspaceStatusUpdate
):
    """Update Jira DC workspace active state without starting OAuth."""
    try:
        user_auth = cast(SaasUserAuth, await get_user_auth(request))
        user_id = await user_auth.get_user_id()

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='User ID not found',
            )

        workspace = await _validate_workspace_update_permissions(
            user_id, workspace_data.workspace_name
        )
        workspace_status = 'active' if workspace_data.is_active else 'inactive'
        await jira_dc_manager.integration_store.update_workspace(
            id=workspace.id,
            status=workspace_status,
        )

        return JSONResponse({'success': True, 'status': workspace_status})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error updating Jira DC workspace status: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to update workspace status',
        )


@jira_dc_integration_router.post('/workspaces/link')
async def create_workspace_link(request: Request, link_data: JiraDcLinkCreate):
    """Register a user mapping to a Jira DC workspace."""
    try:
        user_auth = cast(SaasUserAuth, await get_user_auth(request))
        user_id = await user_auth.get_user_id()
        user_email = await user_auth.get_user_email()

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='User ID not found',
            )

        target_workspace = link_data.workspace_name

        if JIRA_DC_ENABLE_OAUTH:
            # OAuth flow enabled
            state = str(uuid.uuid4())

            integration_session = {
                'operation_type': 'workspace_link',
                'keycloak_user_id': user_id,
                'user_email': user_email,
                'target_workspace': target_workspace,
                'state': state,
            }

            created = redis_client.setex(
                state,
                JIRA_DC_OAUTH_STATE_TTL_SECONDS,
                json.dumps(integration_session),
            )

            if not created:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail='Failed to create integration session',
                )

            auth_params = {
                'client_id': JIRA_DC_CLIENT_ID,
                'scope': JIRA_DC_SCOPES,
                'redirect_uri': JIRA_DC_REDIRECT_URI,
                'state': state,
                'response_type': 'code',
            }
            auth_url = f'{JIRA_DC_AUTH_URL}?{urlencode(auth_params)}'

            return JSONResponse(
                content={
                    'success': True,
                    'redirect': True,
                    'authorizationUrl': auth_url,
                }
            )
        else:
            # OAuth flow disabled - directly link user
            await _handle_workspace_link_creation(
                user_id, 'unavailable', target_workspace
            )
            return JSONResponse(
                content={
                    'success': True,
                    'redirect': False,
                    'authorizationUrl': '',
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error registering Jira DC user: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to register user',
        )


@jira_dc_integration_router.get('/callback')
async def jira_dc_callback(request: Request, code: str, state: str):
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
        'client_id': JIRA_DC_CLIENT_ID,
        'client_secret': JIRA_DC_CLIENT_SECRET,
        'code': code,
        'redirect_uri': JIRA_DC_REDIRECT_URI,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(JIRA_DC_TOKEN_URL, data=token_payload)
    if response.status_code != 200:
        raise HTTPException(
            status_code=400, detail=f'Error fetching token: {response.text}'
        )

    token_data = response.json()
    access_token = token_data['access_token']
    headers = {'Authorization': f'Bearer {access_token}'}
    target_workspace = integration_session.get('target_workspace')

    if target_workspace != urlparse(JIRA_DC_BASE_URL).hostname:
        raise HTTPException(status_code=400, detail='Target workspace mismatch.')

    async with httpx.AsyncClient() as client:
        jira_dc_user_response = await client.get(JIRA_DC_USER_INFO_URL, headers=headers)
    if jira_dc_user_response.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail=f'Error fetching user info: {jira_dc_user_response.text}',
        )

    jira_user_info = jira_dc_user_response.json()
    jira_dc_user_id = jira_user_info.get('key')

    user_id = integration_session['keycloak_user_id']

    if integration_session.get('operation_type') == 'workspace_integration':
        workspace = await jira_dc_manager.integration_store.get_workspace_by_name(
            target_workspace
        )
        session_org_id = integration_session.get('org_id')
        org_id = UUID(session_org_id) if session_org_id else None
        if not workspace:
            # Create new workspace if it doesn't exist
            encrypted_webhook_secret = token_manager.encrypt_text(
                integration_session['webhook_secret']
            )
            encrypted_new_svc_acc_api_key = token_manager.encrypt_text(
                integration_session['svc_acc_api_key']
            )

            workspace = await jira_dc_manager.integration_store.create_workspace(
                name=target_workspace,
                admin_user_id=integration_session['keycloak_user_id'],
                org_id=org_id,
                encrypted_webhook_secret=encrypted_webhook_secret,
                svc_acc_email=integration_session['svc_acc_email'],
                encrypted_svc_acc_api_key=encrypted_new_svc_acc_api_key,
                status='active' if integration_session['is_active'] else 'inactive',
            )

            # Create a workspace link for the user (admin automatically gets linked)
            if integration_session['is_active']:
                await _handle_workspace_link_creation(
                    user_id, jira_dc_user_id, target_workspace
                )
            else:
                await _handle_workspace_link_creation(
                    user_id,
                    jira_dc_user_id,
                    target_workspace,
                    require_active_workspace=False,
                )
        else:
            # Workspace exists - validate user can update it
            await _validate_workspace_update_permissions(user_id, target_workspace)

            encrypted_webhook_secret = token_manager.encrypt_text(
                integration_session['webhook_secret']
            )
            # Empty session PAT (admin edited without changing it) → None so the
            # store preserves the existing encrypted token.
            session_api_key = integration_session.get('svc_acc_api_key')
            updated_encrypted_svc_acc_api_key = (
                token_manager.encrypt_text(session_api_key) if session_api_key else None
            )

            # Update workspace details
            workspace = await jira_dc_manager.integration_store.update_workspace(
                id=workspace.id,
                org_id=org_id,
                encrypted_webhook_secret=encrypted_webhook_secret,
                svc_acc_email=integration_session['svc_acc_email'],
                encrypted_svc_acc_api_key=updated_encrypted_svc_acc_api_key,
                status='active' if integration_session['is_active'] else 'inactive',
            )

            if integration_session['is_active']:
                await _handle_workspace_link_creation(
                    user_id, jira_dc_user_id, target_workspace
                )
            else:
                await _handle_workspace_link_creation(
                    user_id,
                    jira_dc_user_id,
                    target_workspace,
                    require_active_workspace=False,
                )

        webhook_enrolled = await _maybe_register_webhook(
            integration_session.get('admin_api_key'),
            JIRA_DC_BASE_URL,
            integration_session['webhook_secret'],
            workspace.id,
        )
        if integration_session.get('admin_api_key') and not webhook_enrolled:
            await jira_dc_manager.integration_store.update_workspace(
                id=workspace.id,
                status='inactive',
            )

        redirect_url = '/settings/integrations'
        if integration_session.get('admin_api_key') and not webhook_enrolled:
            redirect_url += '?jira_dc_webhook=install_failed'
        elif integration_session.get('admin_api_key') and webhook_enrolled:
            redirect_url += '?jira_dc_webhook=installed'

        return RedirectResponse(
            url=redirect_url,
            status_code=status.HTTP_302_FOUND,
        )
    elif integration_session.get('operation_type') == 'workspace_link':
        await _handle_workspace_link_creation(
            user_id, jira_dc_user_id, target_workspace
        )
        return RedirectResponse(
            url='/settings/integrations', status_code=status.HTTP_302_FOUND
        )
    else:
        raise HTTPException(status_code=400, detail='Invalid operation type')


@jira_dc_integration_router.get(
    '/workspaces/link',
    response_model=JiraDcUserResponse,
)
async def get_current_workspace_link(request: Request):
    """Get current user's Jira DC integration details."""
    try:
        user_auth = cast(SaasUserAuth, await get_user_auth(request))
        user_id = await user_auth.get_user_id()

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='User ID not found',
            )

        user = await jira_dc_manager.integration_store.get_user_by_active_workspace(
            user_id
        )
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='User is not registered for Jira DC integration',
            )

        workspace = await jira_dc_manager.integration_store.get_workspace_by_id(
            user.jira_dc_workspace_id
        )
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Workspace not found for the user',
            )

        return JiraDcUserResponse(
            id=user.id,
            keycloak_user_id=user.keycloak_user_id,
            jira_dc_workspace_id=user.jira_dc_workspace_id,
            status=user.status,
            created_at=user.created_at.isoformat(),
            updated_at=user.updated_at.isoformat(),
            workspace=JiraDcWorkspaceResponse(
                id=workspace.id,
                name=workspace.name,
                status=workspace.status,
                editable=workspace.admin_user_id == user.keycloak_user_id,
                events_url=_jira_dc_events_url(workspace.id),
                svc_acc_email=workspace.svc_acc_email,
                created_at=workspace.created_at.isoformat(),
                updated_at=workspace.updated_at.isoformat(),
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error retrieving Jira DC user: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve user',
        )


@jira_dc_integration_router.post('/workspaces/unlink')
async def unlink_workspace(request: Request):
    """Unlink from Jira DC and, for integration owners, optionally revoke the hook.

    A non-owner user is only detached from the workspace (their personal link
    goes inactive) -- never touching Jira. The integration owner instead tears
    the whole integration down: the workspace is deactivated and, when a Jira
    admin PAT is supplied in the request body, the Jira webhook is deleted too
    (best effort, mirroring auto-enrollment). Deactivation never fails if the
    Jira-side cleanup did; the owner can always remove the webhook by hand.
    """
    try:
        user_auth = cast(SaasUserAuth, await get_user_auth(request))
        user_id = await user_auth.get_user_id()

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='User ID not found',
            )

        # The body is optional: the non-admin "disconnect" path sends nothing,
        # while the admin "remove integration" path may include an admin PAT.
        admin_api_key: str | None = None
        try:
            body = await request.json()
            if isinstance(body, dict):
                admin_api_key = body.get('admin_api_key')
        except Exception:
            admin_api_key = None

        user = await jira_dc_manager.integration_store.get_user_by_active_workspace(
            user_id
        )
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='User is not registered for Jira DC integration',
            )

        workspace = await jira_dc_manager.integration_store.get_workspace_by_id(
            user.jira_dc_workspace_id
        )
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Workspace not found for the user',
            )

        webhook_removed = False
        if workspace.admin_user_id == user_id:
            base_api_url = (
                JIRA_DC_BASE_URL
                if JIRA_DC_ENABLE_OAUTH
                else f'https://{workspace.name}'
            )
            webhook_removed = await _maybe_delete_webhook(
                admin_api_key, base_api_url, workspace.id
            )
            await jira_dc_manager.integration_store.deactivate_workspace(
                workspace_id=workspace.id,
            )
        else:
            await jira_dc_manager.integration_store.update_user_integration_status(
                user_id, user.jira_dc_workspace_id, 'inactive'
            )

        return JSONResponse({'success': True, 'webhookRemoved': webhook_removed})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error unlinking Jira DC user: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to unlink user',
        )


@jira_dc_integration_router.get(
    '/workspaces/validate/{workspace_name}',
    response_model=JiraDcValidateWorkspaceResponse,
)
async def validate_workspace_integration(request: Request, workspace_name: str):
    """Validate if the workspace has an active Jira DC integration."""
    try:
        await get_user_auth(request)

        # Validate workspace_name format
        if not re.match(r'^[a-zA-Z0-9_.-]+$', workspace_name):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='workspace_name can only contain alphanumeric characters, hyphens, underscores, and periods',
            )

        # Check if workspace exists
        workspace = await jira_dc_manager.integration_store.get_workspace_by_name(
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

        return JiraDcValidateWorkspaceResponse(
            name=workspace.name,
            status=workspace.status,
            message='Workspace integration is active',
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error validating Jira DC workspace: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to validate workspace',
        )
