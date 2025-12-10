"""OAuth 2.0 Device Flow endpoints for CLI authentication."""

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from server.auth.saas_user_auth import SaasUserAuth
from server.config import get_config
from storage.api_key_store import ApiKeyStore
from storage.database import session_maker
from storage.device_code_store import DeviceCodeStore
from storage.saas_settings_store import SaasSettingsStore

from openhands.core.logger import openhands_logger as logger
from openhands.server.user_auth.user_auth import get_user_auth


# OAuth Device Flow models
class DeviceAuthorizationRequest(BaseModel):
    client_id: str
    scope: Optional[str] = None


class DeviceAuthorizationResponse(BaseModel):
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int


class DeviceTokenRequest(BaseModel):
    grant_type: str
    device_code: str
    client_id: str


class DeviceTokenResponse(BaseModel):
    access_token: str  # This will be the user's API key
    token_type: str = "Bearer"
    expires_in: Optional[int] = None  # API keys may not have expiration
    scope: Optional[str] = None


class DeviceTokenErrorResponse(BaseModel):
    error: str
    error_description: Optional[str] = None


class DeviceVerificationRequest(BaseModel):
    user_code: str
    action: str  # "authorize" or "deny"


# Initialize router and store
oauth_device_router = APIRouter(prefix='/oauth/device')
device_code_store = DeviceCodeStore(session_maker)


@oauth_device_router.post('/authorize', response_model=DeviceAuthorizationResponse)
async def device_authorization(request: DeviceAuthorizationRequest, http_request: Request):
    """Initiate OAuth 2.0 Device Flow authorization.
    
    This endpoint starts the device flow by generating device and user codes.
    The client will poll the token endpoint while the user authorizes on another device.
    """
    try:
        # Validate client_id (for now, accept any non-empty client_id)
        if not request.client_id or not request.client_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid client_id"
            )
        
        # Create device code entry
        device_code_entry = device_code_store.create_device_code(
            client_id=request.client_id,
            scope=request.scope,
            expires_in=600  # 10 minutes
        )
        
        # Build verification URIs
        base_url = str(http_request.base_url).rstrip('/')
        verification_uri = f"{base_url}/oauth/device/verify"
        verification_uri_complete = f"{verification_uri}?user_code={device_code_entry.user_code}"
        
        logger.info(
            f"Device authorization initiated: user_code={device_code_entry.user_code}, "
            f"client_id={request.client_id}"
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


@oauth_device_router.post('/token')
async def device_token(request: DeviceTokenRequest):
    """Poll for OAuth 2.0 Device Flow token.
    
    The client polls this endpoint until the user completes authorization
    or the device code expires.
    """
    try:
        # Validate grant type
        if request.grant_type != "urn:ietf:params:oauth:grant-type:device_code":
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=DeviceTokenErrorResponse(
                    error="unsupported_grant_type",
                    error_description="Grant type must be 'urn:ietf:params:oauth:grant-type:device_code'"
                ).dict()
            )
        
        # Get device code entry
        device_code_entry = device_code_store.get_by_device_code(request.device_code)
        
        if not device_code_entry:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=DeviceTokenErrorResponse(
                    error="invalid_grant",
                    error_description="Invalid device code"
                ).dict()
            )
        
        # Check if expired
        if device_code_entry.is_expired():
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=DeviceTokenErrorResponse(
                    error="expired_token",
                    error_description="Device code has expired"
                ).dict()
            )
        
        # Check status
        if device_code_entry.status == "denied":
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=DeviceTokenErrorResponse(
                    error="access_denied",
                    error_description="User denied the authorization request"
                ).dict()
            )
        
        if device_code_entry.status == "pending":
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=DeviceTokenErrorResponse(
                    error="authorization_pending",
                    error_description="User has not yet completed authorization"
                ).dict()
            )
        
        if device_code_entry.status == "authorized":
            # Return the API key as access_token
            return DeviceTokenResponse(
                access_token=device_code_entry.access_token,  # This is the API key
                scope=device_code_entry.scope
            )
        
        # Unknown status
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=DeviceTokenErrorResponse(
                error="server_error",
                error_description="Unknown device code status"
            ).dict()
        )
        
    except Exception as e:
        logger.exception(f"Error in device token: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=DeviceTokenErrorResponse(
                error="server_error",
                error_description="Internal server error"
            ).dict()
        )


@oauth_device_router.get('/verify')
async def device_verification_page(user_code: Optional[str] = None):
    """Device verification page where users enter their user code.
    
    This would typically render an HTML page, but for now we'll return
    a simple JSON response indicating what should be displayed.
    """
    from fastapi.responses import HTMLResponse
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>OpenHands Device Verification</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }}
            .container {{ text-align: center; }}
            .code-input {{ font-size: 24px; padding: 10px; margin: 20px; text-align: center; letter-spacing: 2px; }}
            .button {{ background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 10px; }}
            .button:hover {{ background-color: #0056b3; }}
            .deny-button {{ background-color: #dc3545; }}
            .deny-button:hover {{ background-color: #c82333; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>OpenHands Device Verification</h1>
            <p>Enter the code displayed on your device to authorize access:</p>
            
            <form method="post" action="/oauth/device/verify">
                <input type="text" name="user_code" class="code-input" placeholder="Enter code" 
                       value="{user_code or ''}" maxlength="8" required>
                <br>
                <button type="submit" name="action" value="authorize" class="button">Authorize</button>
                <button type="submit" name="action" value="deny" class="button deny-button">Deny</button>
            </form>
            
            <p><small>This will grant access to your OpenHands account from the requesting device.</small></p>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)


@oauth_device_router.post('/verify')
async def device_verification(http_request: Request):
    """Handle device verification form submission.
    
    This endpoint processes the user's authorization decision.
    """
    from fastapi.responses import HTMLResponse
    
    try:
        # Parse form data
        form_data = await http_request.form()
        user_code = form_data.get("user_code")
        action = form_data.get("action")
        
        if not user_code or not action:
            return HTMLResponse(
                content="<h1>Error</h1><p>Missing user code or action.</p>",
                status_code=400
            )
        
        # Get the authenticated user
        user_auth: SaasUserAuth = await get_user_auth(http_request)
        access_token = await user_auth.get_access_token()
        refresh_token = user_auth.refresh_token
        user_id = await user_auth.get_user_id()
        
        if not access_token or not refresh_token or not user_id:
            return HTMLResponse(
                content="<h1>Authentication Required</h1><p>Please log in to continue.</p>",
                status_code=401
            )
        
        # Get device code entry
        device_code_entry = device_code_store.get_by_user_code(user_code)
        
        if not device_code_entry:
            return HTMLResponse(
                content="<h1>Error</h1><p>Invalid user code.</p>",
                status_code=400
            )
        
        if not device_code_entry.is_pending():
            return HTMLResponse(
                content="<h1>Error</h1><p>User code is no longer valid.</p>",
                status_code=400
            )
        
        # Process the user's decision
        if action == "authorize":
            # Get or create an API key for the user
            api_key_store = ApiKeyStore.get_instance()
            
            # Check if user already has a CLI API key
            existing_keys = api_key_store.list_api_keys(user_id)
            cli_api_key = None
            
            for key_info in existing_keys:
                if key_info.get('name') == 'CLI Authentication':
                    # Use existing CLI key - we need to get the actual key value
                    # Note: This is a limitation - we can't retrieve the key value from storage
                    # So we'll create a new one
                    break
            
            # Create a new API key for CLI authentication
            try:
                cli_api_key = api_key_store.create_api_key(
                    user_id, 
                    name="CLI Authentication",
                    expires_at=None  # No expiration for CLI keys
                )
                logger.info(f"Created new CLI API key for user: {user_id}")
            except Exception as e:
                logger.exception(f"Failed to create CLI API key: {str(e)}")
                return HTMLResponse(
                    content="<h1>Error</h1><p>Failed to create API key for CLI access.</p>",
                    status_code=500
                )
            
            success = device_code_store.authorize_device_code(
                user_code,
                user_id,
                cli_api_key
            )
            
            if success:
                logger.info(f"Device code authorized: user_code={user_code}, user_id={user_id}")
                return HTMLResponse(
                    content="""
                    <h1>Success!</h1>
                    <p>Device authorized successfully! You can now return to your CLI.</p>
                    <p>You may close this window.</p>
                    """
                )
            else:
                return HTMLResponse(
                    content="<h1>Error</h1><p>Failed to authorize device.</p>",
                    status_code=500
                )
        
        elif action == "deny":
            success = device_code_store.deny_device_code(user_code)
            
            if success:
                logger.info(f"Device code denied: user_code={user_code}, user_id={user_id}")
                return HTMLResponse(
                    content="""
                    <h1>Authorization Denied</h1>
                    <p>Device authorization has been denied.</p>
                    <p>You may close this window.</p>
                    """
                )
            else:
                return HTMLResponse(
                    content="<h1>Error</h1><p>Failed to deny device.</p>",
                    status_code=500
                )
        
        else:
            return HTMLResponse(
                content="<h1>Error</h1><p>Invalid action.</p>",
                status_code=400
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in device verification: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
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


