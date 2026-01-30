"""OAuth 2.0 Device Flow endpoints for CLI authentication."""

from datetime import UTC, datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from storage.api_key_store import ApiKeyStore
from storage.database import session_maker
from storage.device_code_store import DeviceCodeStore

from openhands.core.logger import openhands_logger as logger
from openhands.server.user_auth import get_user_id

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE_CODE_EXPIRES_IN = 600  # 10 minutes
DEVICE_TOKEN_POLL_INTERVAL = 5  # seconds

API_KEY_NAME = 'Device Link Access Key'
KEY_EXPIRATION_TIME = timedelta(days=7)  # Key expires in a week

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


class DeviceTokenResponse(BaseModel):
    access_token: str  # This will be the user's API key
    token_type: str = 'Bearer'
    expires_in: Optional[int] = None  # API keys may not have expiration


class DeviceTokenErrorResponse(BaseModel):
    error: str
    error_description: Optional[str] = None
    interval: Optional[int] = None  # Required for slow_down error


# ---------------------------------------------------------------------------
# Router + stores
# ---------------------------------------------------------------------------

oauth_device_router = APIRouter(prefix='/oauth/device')
device_code_store = DeviceCodeStore(session_maker)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _oauth_error(
    status_code: int,
    error: str,
    description: str,
    interval: Optional[int] = None,
) -> JSONResponse:
    """Return a JSON OAuth-style error response."""
    return JSONResponse(
        status_code=status_code,
        content=DeviceTokenErrorResponse(
            error=error,
            error_description=description,
            interval=interval,
        ).model_dump(),
    )


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
            interval=device_code_entry.current_interval,
        )
    except Exception as e:
        logger.exception('Error in device authorization: %s', str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Internal server error',
        ) from e


@oauth_device_router.post('/token')
async def device_token(device_code: str = Form(...)):
    """Poll for a token until the user authorizes or the code expires."""
    try:
        device_code_entry = device_code_store.get_by_device_code(device_code)

        if not device_code_entry:
            return _oauth_error(
                status.HTTP_400_BAD_REQUEST,
                'invalid_grant',
                'Invalid device code',
            )

        # Check rate limiting (RFC 8628 section 3.5)
        is_too_fast, current_interval = device_code_entry.check_rate_limit()
        if is_too_fast:
            # Update poll time and increase interval
            device_code_store.update_poll_time(device_code, increase_interval=True)
            logger.warning(
                'Client polling too fast, returning slow_down error',
                extra={
                    'device_code': device_code[:8] + '...',  # Log partial for privacy
                    'new_interval': current_interval,
                },
            )
            return _oauth_error(
                status.HTTP_400_BAD_REQUEST,
                'slow_down',
                f'Polling too frequently. Wait at least {current_interval} seconds between requests.',
                interval=current_interval,
            )

        # Update poll time for successful rate limit check
        device_code_store.update_poll_time(device_code, increase_interval=False)

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
            # Retrieve the specific API key for this device using the user_code
            api_key_store = ApiKeyStore.get_instance()
            device_key_name = f'{API_KEY_NAME} ({device_code_entry.user_code})'
            device_api_key = api_key_store.retrieve_api_key_by_name(
                device_code_entry.keycloak_user_id, device_key_name
            )

            if not device_api_key:
                logger.error(
                    'No device API key found for authorized device',
                    extra={
                        'user_id': device_code_entry.keycloak_user_id,
                        'user_code': device_code_entry.user_code,
                    },
                )
                return _oauth_error(
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    'server_error',
                    'API key not found',
                )

            # Return the API key as access_token
            return DeviceTokenResponse(
                access_token=device_api_key,
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


@oauth_device_router.post('/verify-authenticated')
async def device_verification_authenticated(
    user_code: str = Form(...),
    user_id: str = Depends(get_user_id),
):
    """Process device verification for authenticated users (called by frontend)."""
    try:
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='Authentication required',
            )

        # Validate device code
        device_code_entry = device_code_store.get_by_user_code(user_code)
        if not device_code_entry:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='The device code is invalid or has expired.',
            )

        if not device_code_entry.is_pending():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='This device code has already been processed.',
            )

        # First, authorize the device code
        success = device_code_store.authorize_device_code(
            user_code=user_code,
            user_id=user_id,
        )

        if not success:
            logger.error(
                'Failed to authorize device code',
                extra={'user_code': user_code, 'user_id': user_id},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='Failed to authorize the device. Please try again.',
            )

        # Only create API key AFTER successful authorization
        api_key_store = ApiKeyStore.get_instance()
        try:
            # Create a unique API key for this device using user_code in the name
            device_key_name = f'{API_KEY_NAME} ({user_code})'
            await api_key_store.create_api_key(
                user_id,
                name=device_key_name,
                expires_at=datetime.now(UTC) + KEY_EXPIRATION_TIME,
            )
            logger.info(
                'Created new device API key for user after successful authorization',
                extra={'user_id': user_id, 'user_code': user_code},
            )
        except Exception as e:
            logger.exception(
                'Failed to create device API key after authorization: %s', str(e)
            )

            # Clean up: revert the device authorization since API key creation failed
            # This prevents the device from being in an authorized state without an API key
            try:
                device_code_store.deny_device_code(user_code)
                logger.info(
                    'Reverted device authorization due to API key creation failure',
                    extra={'user_code': user_code, 'user_id': user_id},
                )
            except Exception as cleanup_error:
                logger.exception(
                    'Failed to revert device authorization during cleanup: %s',
                    str(cleanup_error),
                )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='Failed to create API key for device access.',
            )

        logger.info(
            'Device code authorized with API key successfully',
            extra={'user_code': user_code, 'user_id': user_id},
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={'message': 'Device authorized successfully!'},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception('Error in device verification: %s', str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='An unexpected error occurred. Please try again.',
        )
