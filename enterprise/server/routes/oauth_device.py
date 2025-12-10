"""OAuth 2.0 Device Flow endpoints for CLI authentication."""

import asyncio
import html
from typing import Optional
from urllib.parse import quote

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from integrations.utils import (
    HOST_URL,
)
from pydantic import BaseModel, SecretStr
from server.auth.constants import (
    KEYCLOAK_CLIENT_ID,
    KEYCLOAK_REALM_NAME,
    KEYCLOAK_SERVER_URL_EXT,
)
from server.auth.saas_user_auth import SaasUserAuth
from server.auth.token_manager import TokenManager
from server.config import get_config
from storage.api_key_store import ApiKeyStore
from storage.database import session_maker
from storage.device_code_store import DeviceCodeStore
from storage.saas_settings_store import SaasSettingsStore

from openhands.core.logger import openhands_logger as logger
from openhands.server.shared import config
from openhands.server.user_auth import get_user_id
from openhands.server.user_auth.user_auth import get_user_auth


# OAuth Device Flow models
class DeviceAuthorizationRequest(BaseModel):
    pass  # No fields needed for device authorization request


class DeviceAuthorizationResponse(BaseModel):
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int


class DeviceTokenRequest(BaseModel):
    device_code: str


class DeviceTokenResponse(BaseModel):
    access_token: str  # This will be the user's API key
    token_type: str = "Bearer"
    expires_in: Optional[int] = None  # API keys may not have expiration


class DeviceTokenErrorResponse(BaseModel):
    error: str
    error_description: Optional[str] = None


class DeviceVerificationRequest(BaseModel):
    user_code: str
    action: str  # "authorize" or "deny"


# Initialize router and store
oauth_device_router = APIRouter(prefix='/oauth')
device_code_store = DeviceCodeStore(session_maker)
token_manager = TokenManager()


@oauth_device_router.post('/device/authorize', response_model=DeviceAuthorizationResponse)
async def device_authorization(
    request: DeviceAuthorizationRequest, 
    http_request: Request
):
    """Initiate OAuth 2.0 Device Flow authorization.
    
    This endpoint starts the device flow by generating device and user codes.
    The client will poll the token endpoint while the user authorizes on another device.
    """
    try:
        # Create device code entry (no user authentication required at this stage)
        device_code_entry = device_code_store.create_device_code(
            expires_in=600  # 10 minutes
        )
        
        # Build verification URIs
        base_url = str(http_request.base_url).rstrip('/')
        verification_uri = f"{base_url}/oauth/verify"
        verification_uri_complete = f"{verification_uri}?user_code={device_code_entry.user_code}"
        
        logger.info(
            f"Device authorization initiated: user_code={device_code_entry.user_code}"
        )
        
        return DeviceAuthorizationResponse(
            device_code=device_code_entry.device_code,
            user_code=device_code_entry.user_code,
            verification_uri=verification_uri,
            verification_uri_complete=verification_uri_complete,
            expires_in=600,  # 10 minutes
            interval=5  # Poll every 5 seconds
        )
        
    except Exception as e:
        logger.exception(f"Error in device authorization: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@oauth_device_router.post('/device/token')
async def device_token(request: DeviceTokenRequest):
    """Poll for OAuth 2.0 Device Flow token.
    
    The client polls this endpoint until the user completes authorization
    or the device code expires.
    """
    try:
        # Get device code entry
        device_code_entry = device_code_store.get_by_device_code(request.device_code)
        
        if not device_code_entry:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=DeviceTokenErrorResponse(
                    error="invalid_grant",
                    error_description="Invalid device code"
                ).model_dump()
            )
        
        # Check if expired
        if device_code_entry.is_expired():
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=DeviceTokenErrorResponse(
                    error="expired_token",
                    error_description="Device code has expired"
                ).model_dump()
            )
        
        # Check status
        if device_code_entry.status == "denied":
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=DeviceTokenErrorResponse(
                    error="access_denied",
                    error_description="User denied the authorization request"
                ).model_dump()
            )
        
        if device_code_entry.status == "pending":
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=DeviceTokenErrorResponse(
                    error="authorization_pending",
                    error_description="User has not yet completed authorization"
                ).model_dump()
            )
        
        if device_code_entry.status == "authorized":
            # Return the API key as access_token
            return DeviceTokenResponse(
                access_token=device_code_entry.access_token  # This is the API key
            )
        
        # Unknown status
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=DeviceTokenErrorResponse(
                error="server_error",
                error_description="Unknown device code status"
            ).model_dump()
        )
        
    except Exception as e:
        logger.exception(f"Error in device token: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=DeviceTokenErrorResponse(
                error="server_error",
                error_description="Internal server error"
            ).model_dump()
        )


@oauth_device_router.get('/verify')
async def device_verification_page(request: Request, user_code: Optional[str] = None):
    """Device verification page - redirects to Keycloak for authentication.
    
    This endpoint initiates the OAuth device authorization flow by redirecting
    the user to Keycloak for authentication. After successful authentication,
    the user will be redirected back to complete device authorization.
    """
    # If no user_code provided, show a form to enter it
    if not user_code:
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Device Verification</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
                .form-group { margin: 20px 0; }
                input[type="text"] { padding: 10px; font-size: 16px; width: 200px; }
                button { padding: 10px 20px; font-size: 16px; background-color: #4CAF50; color: white; border: none; }
            </style>
        </head>
        <body>
            <h1>Device Authorization</h1>
            <p>Enter the code displayed on your device:</p>
            
            <form method="get" action="/oauth/verify">
                <div class="form-group">
                    <label for="user_code">Device Code:</label><br>
                    <input type="text" id="user_code" name="user_code" required>
                </div>
                
                <div class="form-group">
                    <button type="submit">Continue</button>
                </div>
            </form>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    # Validate the user_code exists
    device_code_entry = device_code_store.get_by_user_code(user_code)
    if not device_code_entry:
        return HTMLResponse(
            content="<h1>Error</h1><p>Invalid or expired device code.</p>",
            status_code=400
        )
    
    # Create JWT state with user_code
    jwt_secret: SecretStr = config.jwt_secret  # type: ignore[assignment]
    payload = {'user_code': user_code}
    state = jwt.encode(payload, jwt_secret.get_secret_value(), algorithm='HS256')
    
    # Redirect to Keycloak for authentication
    scope = quote('openid email profile offline_access')
    redirect_uri = quote(f'{HOST_URL}/oauth/keycloak-callback')
    auth_url = (
        f'{KEYCLOAK_SERVER_URL_EXT}/realms/{KEYCLOAK_REALM_NAME}/protocol/openid-connect/auth'
        f'?client_id={KEYCLOAK_CLIENT_ID}&response_type=code'
        f'&redirect_uri={redirect_uri}'
        f'&scope={scope}'
        f'&state={state}'
    )
    
    return RedirectResponse(auth_url)


def _html_response(title: str, description: str, status_code: int = 200) -> HTMLResponse:
    """Helper function to create HTML responses."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{html.escape(title)}</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }}
            .container {{ text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{html.escape(title)}</h1>
            <p>{description}</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=status_code)


@oauth_device_router.get('/keycloak-callback')
async def keycloak_callback(
    request: Request,
    code: str = '',
    state: str = '',
    error: str = '',
):
    """Handle Keycloak authentication callback and complete device authorization."""
    if not code or error:
        logger.warning(
            'keycloak_callback_error',
            extra={
                'code': code,
                'state': state,
                'error': error,
            },
        )
        return _html_response(
            title='Authentication Error',
            description=html.escape(error or 'No authorization code provided'),
            status_code=400,
        )

    try:
        # Decode the JWT state to get user_code
        jwt_secret: SecretStr = config.jwt_secret  # type: ignore[assignment]
        payload: dict[str, str] = jwt.decode(
            state, jwt_secret.get_secret_value(), algorithms=['HS256']
        )
        user_code = payload['user_code']

        # Get Keycloak tokens
        redirect_uri = f'{HOST_URL}/oauth/keycloak-callback'
        (
            keycloak_access_token,
            keycloak_refresh_token,
        ) = await token_manager.get_keycloak_tokens(code, redirect_uri)
        
        if not keycloak_access_token or not keycloak_refresh_token:
            logger.warning(
                'failed_to_get_keycloak_tokens',
                extra={
                    'code': code,
                    'state': state,
                    'error': error,
                },
            )
            return _html_response(
                title='Failed to authenticate.',
                description=f'Please re-login into <a href="{HOST_URL}" style="color:#ecedee;text-decoration:underline;">OpenHands Cloud</a>. Then try the device authorization again.',
                status_code=400,
            )

        # Get user info from Keycloak token
        user_info = await token_manager.get_user_info(keycloak_access_token)
        if not user_info or not user_info.get('sub'):
            logger.warning('failed_to_get_user_info_from_keycloak')
            return _html_response(
                title='Failed to authenticate.',
                description=f'Please re-login into <a href="{HOST_URL}" style="color:#ecedee;text-decoration:underline;">OpenHands Cloud</a>. Then try the device authorization again.',
                status_code=400,
            )

        user_id = user_info['sub']

        # Validate the device code still exists and is pending
        device_code_entry = device_code_store.get_by_user_code(user_code)
        if not device_code_entry:
            return _html_response(
                title='Invalid Code',
                description='The device code is invalid or has expired.',
                status_code=400,
            )

        if not device_code_entry.is_pending():
            return _html_response(
                title='Code Already Used',
                description='This device code has already been processed.',
                status_code=400,
            )

        # Create API key for the user
        api_key_store = ApiKeyStore.get_instance()
        try:
            cli_api_key = api_key_store.create_api_key(
                user_id, 
                name="CLI Authentication",
                expires_at=None  # No expiration for CLI keys
            )
            logger.info(f"Created new CLI API key for user: {user_id}")
        except Exception as e:
            logger.exception(f"Failed to create CLI API key: {str(e)}")
            return _html_response(
                title='Error',
                description='Failed to create API key for CLI access.',
                status_code=500,
            )

        # Authorize the device
        success = device_code_store.authorize_device_code(
            user_code,
            user_id,
            cli_api_key
        )

        if success:
            logger.info(f"Device code authorized: user_code={user_code}, user_id={user_id}")
            return _html_response(
                title='Success!',
                description='Device authorized successfully! You can now return to your CLI and close this window.',
            )
        else:
            return _html_response(
                title='Authorization Failed',
                description='Failed to authorize the device. Please try again.',
                status_code=500,
            )

    except jwt.InvalidTokenError:
        logger.warning('invalid_jwt_state_token')
        return _html_response(
            title='Invalid Request',
            description='Invalid authentication state. Please try again.',
            status_code=400,
        )
    except Exception as e:
        logger.exception(f"Error in keycloak callback: {str(e)}")
        return _html_response(
            title='Internal Error',
            description='An unexpected error occurred. Please try again.',
            status_code=500,
        )


# Cleanup task (should be run periodically)
async def cleanup_expired_device_codes():
    """Background task to clean up expired device codes."""
    try:
        count = device_code_store.cleanup_expired_codes()
        if count > 0:
            logger.info(f"Cleaned up {count} expired device codes")
    except Exception as e:
        logger.exception(f"Error cleaning up expired device codes: {str(e)}")


# Add cleanup task to run every 5 minutes
async def start_cleanup_task():
    """Start the periodic cleanup task."""
    while True:
        await asyncio.sleep(300)  # 5 minutes
        await cleanup_expired_device_codes()


