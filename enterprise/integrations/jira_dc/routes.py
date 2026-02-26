import json
import re
import uuid
from urllib.parse import urlencode, urlparse

import requests
from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import JSONResponse, RedirectResponse
from openhands.core.logger import openhands_logger as logger
from openhands.server.user_auth.user_auth import get_user_auth
from pydantic import BaseModel, Field, field_validator

from integrations.jira_dc.jira_dc_manager import JiraDcManager
from integrations.models import Message, SourceType
from server.auth.saas_user_auth import SaasUserAuth
from server.auth.token_manager import TokenManager
from storage.redis import create_redis_client

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class JiraDcWorkspaceCreate(BaseModel):
    workspace_name: str = Field(..., description='Workspace display name')
    webhook_secret: str = Field(..., description='Webhook secret for verification')
    svc_acc_email: str = Field(..., description='Service account email')
    svc_acc_api_key: str = Field(..., description='Service account API token/PAT')
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


class JiraDcWorkspaceResponse(BaseModel):
    id: int
    name: str
    status: str
    editable: bool
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


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_jira_dc_router(plugin_config) -> APIRouter:
    """Create a self-contained FastAPI router for Jira DC integration.

    Args:
        plugin_config: A JiraDcPluginConfig instance with all settings.

    Returns:
        Configured APIRouter with all Jira DC endpoints.
    """
    from integrations.jira_dc.plugin import JiraDcPluginConfig

    cfg: JiraDcPluginConfig = plugin_config

    router = APIRouter(prefix='/integration/jira-dc')
    token_manager = TokenManager()
    jira_dc_manager = JiraDcManager(token_manager)
    redis_client = create_redis_client()

    # Derive URLs from config
    jira_dc_redirect_uri = f'https://{cfg.web_host}/integration/jira-dc/callback'
    jira_dc_scopes = 'read:me read:jira-user read:jira-work'
    jira_dc_auth_url = f'{cfg.base_url}/rest/oauth2/latest/authorize'
    jira_dc_token_url = f'{cfg.base_url}/rest/oauth2/latest/token'
    jira_dc_user_info_url = f'{cfg.base_url}/rest/api/2/myself'

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    async def _handle_workspace_link_creation(
        user_id: str, jira_dc_user_id: str, target_workspace: str
    ):
        """Handle the creation or reactivation of a workspace link for a user."""
        workspace = await jira_dc_manager.integration_store.get_workspace_by_name(
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

        existing_user = (
            await jira_dc_manager.integration_store.get_user_by_active_workspace(
                user_id
            )
        )

        if existing_user:
            if existing_user.jira_dc_workspace_id == workspace.id:
                return
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail='You already have an active workspace link. Please unlink from your current workspace before linking to a different one.',
                )

        existing_link = await jira_dc_manager.integration_store.get_user_by_keycloak_id_and_workspace(
            user_id, workspace.id
        )

        if existing_link:
            await jira_dc_manager.integration_store.update_user_integration_status(
                user_id, 'active'
            )
        else:
            await jira_dc_manager.integration_store.create_workspace_link(
                keycloak_user_id=user_id,
                jira_dc_user_id=jira_dc_user_id,
                jira_dc_workspace_id=workspace.id,
            )

    async def _validate_workspace_update_permissions(
        user_id: str, target_workspace: str
    ):
        """Validate that user can update the target workspace."""
        workspace = await jira_dc_manager.integration_store.get_workspace_by_name(
            target_workspace
        )
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f'Workspace "{target_workspace}" not found',
            )

        if workspace.admin_user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='You do not have permission to update this workspace',
            )

        current_user_link = (
            await jira_dc_manager.integration_store.get_user_by_active_workspace(
                user_id
            )
        )
        if current_user_link and current_user_link.jira_dc_workspace_id != workspace.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='You can only update the workspace you are currently linked to',
            )

        return workspace

    # -------------------------------------------------------------------
    # Endpoints
    # -------------------------------------------------------------------

    @router.post('/events')
    async def jira_dc_events(
        request: Request,
        background_tasks: BackgroundTasks,
    ):
        """Handle Jira DC webhook events."""
        if not cfg.webhooks_enabled:
            return JSONResponse(
                status_code=200,
                content={'message': 'Jira DC webhooks are disabled.'},
            )

        try:
            (
                signature_valid,
                signature,
                payload,
            ) = await jira_dc_manager.validate_request(request)

            if not signature_valid:
                logger.warning('[Jira DC] Invalid webhook signature')
                raise HTTPException(
                    status_code=403, detail='Invalid webhook signature!'
                )

            key = f'jira_dc:{signature}'
            keyExists = redis_client.exists(key)
            if keyExists:
                logger.info(f'Received duplicate Jira DC webhook event: {signature}')
                return JSONResponse({'success': True})
            else:
                redis_client.setex(key, 120, 1)

            message_payload = {'payload': payload}
            message = Message(source=SourceType.JIRA_DC, message=message_payload)

            background_tasks.add_task(jira_dc_manager.receive_message, message)

            return JSONResponse({'success': True})
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f'Error processing Jira DC webhook: {e}')
            return JSONResponse(
                status_code=500,
                content={'error': 'Internal server error processing webhook.'},
            )

    @router.post('/workspaces')
    async def create_jira_dc_workspace(
        request: Request, workspace_data: JiraDcWorkspaceCreate
    ):
        """Create a new Jira DC workspace registration."""
        try:
            user_auth: SaasUserAuth = await get_user_auth(request)
            user_id = await user_auth.get_user_id()
            user_email = await user_auth.get_user_email()

            if cfg.enable_oauth:
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
                    'client_id': cfg.client_id,
                    'scope': jira_dc_scopes,
                    'redirect_uri': jira_dc_redirect_uri,
                    'state': state,
                    'response_type': 'code',
                }

                auth_url = f'{jira_dc_auth_url}?{urlencode(auth_params)}'

                return JSONResponse(
                    content={
                        'success': True,
                        'redirect': True,
                        'authorizationUrl': auth_url,
                    }
                )
            else:
                workspace = (
                    await jira_dc_manager.integration_store.get_workspace_by_name(
                        workspace_data.workspace_name
                    )
                )
                if not workspace:
                    encrypted_webhook_secret = token_manager.encrypt_text(
                        workspace_data.webhook_secret
                    )
                    encrypted_svc_acc_api_key = token_manager.encrypt_text(
                        workspace_data.svc_acc_api_key
                    )

                    workspace = (
                        await jira_dc_manager.integration_store.create_workspace(
                            name=workspace_data.workspace_name,
                            admin_user_id=user_id,
                            encrypted_webhook_secret=encrypted_webhook_secret,
                            svc_acc_email=workspace_data.svc_acc_email,
                            encrypted_svc_acc_api_key=encrypted_svc_acc_api_key,
                            status='active' if workspace_data.is_active else 'inactive',
                        )
                    )

                    await _handle_workspace_link_creation(
                        user_id, 'unavailable', workspace.name
                    )
                else:
                    await _validate_workspace_update_permissions(
                        user_id, workspace_data.workspace_name
                    )

                    encrypted_webhook_secret = token_manager.encrypt_text(
                        workspace_data.webhook_secret
                    )
                    encrypted_svc_acc_api_key = token_manager.encrypt_text(
                        workspace_data.svc_acc_api_key
                    )

                    await jira_dc_manager.integration_store.update_workspace(
                        id=workspace.id,
                        encrypted_webhook_secret=encrypted_webhook_secret,
                        svc_acc_email=workspace_data.svc_acc_email,
                        encrypted_svc_acc_api_key=encrypted_svc_acc_api_key,
                        status='active' if workspace_data.is_active else 'inactive',
                    )

                    await _handle_workspace_link_creation(
                        user_id, 'unavailable', workspace.name
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
            logger.exception(f'Error creating Jira DC workspace: {e}')
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='Failed to create workspace',
            )

    @router.post('/workspaces/link')
    async def create_workspace_link(request: Request, link_data: JiraDcLinkCreate):
        """Register a user mapping to a Jira DC workspace."""
        try:
            user_auth: SaasUserAuth = await get_user_auth(request)
            user_id = await user_auth.get_user_id()
            user_email = await user_auth.get_user_email()

            target_workspace = link_data.workspace_name

            if cfg.enable_oauth:
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
                    60,
                    json.dumps(integration_session),
                )

                if not created:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail='Failed to create integration session',
                    )

                auth_params = {
                    'client_id': cfg.client_id,
                    'scope': jira_dc_scopes,
                    'redirect_uri': jira_dc_redirect_uri,
                    'state': state,
                    'response_type': 'code',
                }
                auth_url = f'{jira_dc_auth_url}?{urlencode(auth_params)}'

                return JSONResponse(
                    content={
                        'success': True,
                        'redirect': True,
                        'authorizationUrl': auth_url,
                    }
                )
            else:
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

    @router.get('/callback')
    async def jira_dc_callback(request: Request, code: str, state: str):
        integration_session_json = redis_client.get(state)
        if not integration_session_json:
            raise HTTPException(
                status_code=400, detail='No active integration session found.'
            )

        integration_session = json.loads(integration_session_json)

        if integration_session.get('state') != state:
            raise HTTPException(
                status_code=400, detail='State mismatch. Possible CSRF attack.'
            )

        token_payload = {
            'grant_type': 'authorization_code',
            'client_id': cfg.client_id,
            'client_secret': cfg.client_secret,
            'code': code,
            'redirect_uri': jira_dc_redirect_uri,
        }
        response = requests.post(jira_dc_token_url, json=token_payload)
        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f'Error fetching token: {response.text}',
            )

        token_data = response.json()
        access_token = token_data['access_token']
        headers = {'Authorization': f'Bearer {access_token}'}
        target_workspace = integration_session.get('target_workspace')

        if target_workspace != urlparse(cfg.base_url).hostname:
            raise HTTPException(status_code=400, detail='Target workspace mismatch.')

        jira_dc_user_response = requests.get(jira_dc_user_info_url, headers=headers)
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
            if not workspace:
                encrypted_webhook_secret = token_manager.encrypt_text(
                    integration_session['webhook_secret']
                )
                encrypted_svc_acc_api_key = token_manager.encrypt_text(
                    integration_session['svc_acc_api_key']
                )

                await jira_dc_manager.integration_store.create_workspace(
                    name=target_workspace,
                    admin_user_id=integration_session['keycloak_user_id'],
                    encrypted_webhook_secret=encrypted_webhook_secret,
                    svc_acc_email=integration_session['svc_acc_email'],
                    encrypted_svc_acc_api_key=encrypted_svc_acc_api_key,
                    status='active' if integration_session['is_active'] else 'inactive',
                )

                await _handle_workspace_link_creation(
                    user_id, jira_dc_user_id, target_workspace
                )
            else:
                await _validate_workspace_update_permissions(user_id, target_workspace)

                encrypted_webhook_secret = token_manager.encrypt_text(
                    integration_session['webhook_secret']
                )
                encrypted_svc_acc_api_key = token_manager.encrypt_text(
                    integration_session['svc_acc_api_key']
                )

                await jira_dc_manager.integration_store.update_workspace(
                    id=workspace.id,
                    encrypted_webhook_secret=encrypted_webhook_secret,
                    svc_acc_email=integration_session['svc_acc_email'],
                    encrypted_svc_acc_api_key=encrypted_svc_acc_api_key,
                    status='active' if integration_session['is_active'] else 'inactive',
                )

                await _handle_workspace_link_creation(
                    user_id, jira_dc_user_id, target_workspace
                )

            return RedirectResponse(
                url='/settings/integrations',
                status_code=status.HTTP_302_FOUND,
            )
        elif integration_session.get('operation_type') == 'workspace_link':
            await _handle_workspace_link_creation(
                user_id, jira_dc_user_id, target_workspace
            )
            return RedirectResponse(
                url='/settings/integrations',
                status_code=status.HTTP_302_FOUND,
            )
        else:
            raise HTTPException(status_code=400, detail='Invalid operation type')

    @router.get(
        '/workspaces/link',
        response_model=JiraDcUserResponse,
    )
    async def get_current_workspace_link(request: Request):
        """Get current user's Jira DC integration details."""
        try:
            user_auth: SaasUserAuth = await get_user_auth(request)
            user_id = await user_auth.get_user_id()

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

    @router.post('/workspaces/unlink')
    async def unlink_workspace(request: Request):
        """Unlink user from Jira DC integration by setting status to inactive."""
        try:
            user_auth: SaasUserAuth = await get_user_auth(request)
            user_id = await user_auth.get_user_id()

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

            if workspace.admin_user_id == user_id:
                await jira_dc_manager.integration_store.deactivate_workspace(
                    workspace_id=workspace.id,
                )
            else:
                await jira_dc_manager.integration_store.update_user_integration_status(
                    user_id, 'inactive'
                )

            return JSONResponse({'success': True})

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f'Error unlinking Jira DC user: {e}')
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='Failed to unlink user',
            )

    @router.get(
        '/workspaces/validate/{workspace_name}',
        response_model=JiraDcValidateWorkspaceResponse,
    )
    async def validate_workspace_integration(request: Request, workspace_name: str):
        """Validate if the workspace has an active Jira DC integration."""
        try:
            await get_user_auth(request)

            if not re.match(r'^[a-zA-Z0-9_.-]+$', workspace_name):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail='workspace_name can only contain alphanumeric characters, hyphens, underscores, and periods',
                )

            workspace = await jira_dc_manager.integration_store.get_workspace_by_name(
                workspace_name
            )
            if not workspace:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Workspace with name '{workspace_name}' not found",
                )

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

    return router
