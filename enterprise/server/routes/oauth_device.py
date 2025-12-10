"""OAuth 2.0 Device Flow endpoints for CLI authentication."""

from datetime import UTC, datetime, timedelta
import html
from typing import Optional
from urllib.parse import quote

import jwt
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from integrations.utils import HOST_URL
from pydantic import BaseModel, SecretStr
from server.auth.constants import (
    KEYCLOAK_CLIENT_ID,
    KEYCLOAK_REALM_NAME,
    KEYCLOAK_SERVER_URL_EXT,
)
from server.auth.token_manager import TokenManager
from storage.api_key_store import ApiKeyStore
from storage.database import session_maker
from storage.device_code_store import DeviceCodeStore

from openhands.core.logger import openhands_logger as logger
from openhands.server.shared import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE_CODE_EXPIRES_IN = 600  # 10 minutes
DEVICE_TOKEN_POLL_INTERVAL = 5  # seconds

FAILED_AUTH_DESCRIPTION = (
    f'Please re-login into '
    f'<a href="{HOST_URL}" style="color:#ecedee;text-decoration:underline;">'
    f'OpenHands Cloud</a>. Then try the device authorization again.'
)

API_KEY_NAME = 'CLI Authentication'
KEY_EXPIRATION_TIME = timedelta(days=1) # Key expires in 24 hours

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

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
    token_type: str = 'Bearer'
    expires_in: Optional[int] = None  # API keys may not have expiration


class DeviceTokenErrorResponse(BaseModel):
    error: str
    error_description: Optional[str] = None


# ---------------------------------------------------------------------------
# Router + stores
# ---------------------------------------------------------------------------

oauth_device_router = APIRouter(prefix='/oauth/device')
device_code_store = DeviceCodeStore(session_maker)
token_manager = TokenManager()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _oauth_error(
    status_code: int,
    error: str,
    description: str,
) -> JSONResponse:
    """Return a JSON OAuth-style error response."""
    return JSONResponse(
        status_code=status_code,
        content=DeviceTokenErrorResponse(
            error=error,
            error_description=description,
        ).model_dump(),
    )


def _html_response(
    title: str, description: str, status_code: int = 200
) -> HTMLResponse:
    """Helper to build a simple HTML page."""
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@oauth_device_router.post('/authorize', response_model=DeviceAuthorizationResponse)
async def device_authorization(
    http_request: Request,
) -> DeviceAuthorizationResponse:
    """Start device flow by generating device and user codes."""
    try:
        device_code_entry = device_code_store.create_device_code(
            expires_in=DEVICE_CODE_EXPIRES_IN,
        )

        base_url = str(http_request.base_url).rstrip('/')
        verification_uri = f'{base_url}/oauth/device/verify'
        verification_uri_complete = (
            f'{verification_uri}?user_code={device_code_entry.user_code}'
        )

        logger.info(
            'Device authorization initiated',
            extra={'user_code': device_code_entry.user_code},
        )

        return DeviceAuthorizationResponse(
            device_code=device_code_entry.device_code,
            user_code=device_code_entry.user_code,
            verification_uri=verification_uri,
            verification_uri_complete=verification_uri_complete,
            expires_in=DEVICE_CODE_EXPIRES_IN,
            interval=DEVICE_TOKEN_POLL_INTERVAL,
        )
    except Exception as e:
        logger.exception('Error in device authorization: %s', str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Internal server error',
        ) from e


@oauth_device_router.post('/token')
async def device_token(request: DeviceTokenRequest):
    """Poll for a token until the user authorizes or the code expires."""
    try:
        device_code_entry = device_code_store.get_by_device_code(request.device_code)

        if not device_code_entry:
            return _oauth_error(
                status.HTTP_400_BAD_REQUEST,
                'invalid_grant',
                'Invalid device code',
            )

        if device_code_entry.is_expired():
            return _oauth_error(
                status.HTTP_400_BAD_REQUEST,
                'expired_token',
                'Device code has expired',
            )

        if device_code_entry.status == 'denied':
            return _oauth_error(
                status.HTTP_400_BAD_REQUEST,
                'access_denied',
                'User denied the authorization request',
            )

        if device_code_entry.status == 'pending':
            return _oauth_error(
                status.HTTP_400_BAD_REQUEST,
                'authorization_pending',
                'User has not yet completed authorization',
            )

        if device_code_entry.status == 'authorized':
            # Return the API key as access_token
            return DeviceTokenResponse(
                access_token=device_code_entry.access_token,
            )

        # Fallback for unexpected status values
        logger.error(
            'Unknown device code status',
            extra={'status': device_code_entry.status},
        )
        return _oauth_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            'server_error',
            'Unknown device code status',
        )

    except Exception as e:
        logger.exception('Error in device token: %s', str(e))
        return _oauth_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            'server_error',
            'Internal server error',
        )


@oauth_device_router.get('/verify')
async def device_verification_page(
    user_code: Optional[str] = None,
):
    """Show device code form, or redirect to Keycloak for authentication."""
    # If no user_code provided, show a simple HTML form
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

            <form method="get" action="/oauth/device/verify">
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

    # Validate the user_code
    device_code_entry = device_code_store.get_by_user_code(user_code)
    if not device_code_entry:
        return _html_response(
            title='Error',
            description='Invalid or expired device code.',
            status_code=400,
        )

    # Encode user_code into JWT state
    jwt_secret: SecretStr = config.jwt_secret  # type: ignore[assignment]
    state = jwt.encode(
        {'user_code': user_code},
        jwt_secret.get_secret_value(),
        algorithm='HS256',
    )

    # Redirect to Keycloak
    scope = quote('openid email profile offline_access')
    redirect_uri = quote(f'{HOST_URL}/oauth/device/keycloak-callback')
    auth_url = (
        f'{KEYCLOAK_SERVER_URL_EXT}/realms/{KEYCLOAK_REALM_NAME}'
        f'/protocol/openid-connect/auth'
        f'?client_id={KEYCLOAK_CLIENT_ID}'
        f'&response_type=code'
        f'&redirect_uri={redirect_uri}'
        f'&scope={scope}'
        f'&state={state}'
    )

    return RedirectResponse(auth_url)


@oauth_device_router.get('/keycloak-callback')
async def keycloak_callback(
    request: Request,
    code: str = '',
    state: str = '',
    error: str = '',
):
    """Handle Keycloak callback and complete device authorization."""
    if not code or error:
        logger.warning(
            'keycloak_callback_error',
            extra={'code': code, 'state': state, 'error': error},
        )
        return _html_response(
            title='Authentication Error',
            description=html.escape(error or 'No authorization code provided'),
            status_code=400,
        )

    try:
        # Decode state to get user_code
        jwt_secret: SecretStr = config.jwt_secret  # type: ignore[assignment]
        payload: dict[str, str] = jwt.decode(
            state,
            jwt_secret.get_secret_value(),
            algorithms=['HS256'],
        )
        user_code = payload['user_code']

        # Exchange code for Keycloak tokens
        redirect_uri = f'{HOST_URL}/oauth/device/keycloak-callback'
        (
            keycloak_access_token,
            keycloak_refresh_token,
        ) = await token_manager.get_keycloak_tokens(code, redirect_uri)

        if not keycloak_access_token or not keycloak_refresh_token:
            logger.warning(
                'failed_to_get_keycloak_tokens',
                extra={'code': code, 'state': state, 'error': error},
            )
            return _html_response(
                title='Failed to authenticate.',
                description=FAILED_AUTH_DESCRIPTION,
                status_code=400,
            )

        # Get user info
        user_info = await token_manager.get_user_info(keycloak_access_token)
        if not user_info or not user_info.get('sub'):
            logger.warning('failed_to_get_user_info_from_keycloak')
            return _html_response(
                title='Failed to authenticate.',
                description=FAILED_AUTH_DESCRIPTION,
                status_code=400,
            )

        user_id = user_info['sub']

        # Validate device code
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

        # Create API key for CLI
        api_key_store = ApiKeyStore.get_instance()
        try:
            api_key_store.delete_api_key(API_KEY_NAME)
            cli_api_key = api_key_store.create_api_key(
                user_id,
                name=API_KEY_NAME,
                expires_at=datetime.now(UTC) + KEY_EXPIRATION_TIME,
            )
            logger.info('Created new CLI API key for user', extra={'user_id': user_id})
        except Exception as e:
            logger.exception('Failed to create CLI API key: %s', str(e))
            return _html_response(
                title='Error',
                description='Failed to create API key for CLI access.',
                status_code=500,
            )

        # Mark device as authorized
        success = device_code_store.authorize_device_code(
            user_code=user_code,
            user_id=user_id,
            api_key=cli_api_key,
        )

        if success:
            logger.info(
                'Device code authorized',
                extra={'user_code': user_code, 'user_id': user_id},
            )
            return _html_response(
                title='Success!',
                description='Device authorized successfully! You can now return to your CLI and close this window.',
            )

        logger.error(
            'Failed to authorize device code',
            extra={'user_code': user_code, 'user_id': user_id},
        )
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
        logger.exception('Error in keycloak callback: %s', str(e))
        return _html_response(
            title='Internal Error',
            description='An unexpected error occurred. Please try again.',
            status_code=500,
        )
