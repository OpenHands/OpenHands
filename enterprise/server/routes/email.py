import re

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, field_validator
from server.auth.constants import KEYCLOAK_CLIENT_ID
from server.auth.keycloak_manager import get_keycloak_admin
from server.auth.saas_user_auth import SaasUserAuth
from server.routes.auth import set_response_cookie
from server.utils.rate_limit_utils import check_rate_limit_by_user_id

from openhands.core.logger import openhands_logger as logger
from openhands.server.user_auth import get_user_id
from openhands.server.user_auth.user_auth import get_user_auth

# Email validation regex pattern
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

api_router = APIRouter(prefix='/api/email')


class EmailUpdate(BaseModel):
    email: str

    @field_validator('email')
    def validate_email(cls, v):
        if not EMAIL_REGEX.match(v):
            raise ValueError('Invalid email format')
        return v


class ResendEmailVerificationRequest(BaseModel):
    user_id: str | None = None
    is_auth_flow: bool = False


@api_router.post('')
async def update_email(
    email_data: EmailUpdate, request: Request, user_id: str = Depends(get_user_id)
):
    # Email validation is now handled by the Pydantic model
    # If we get here, the email has already passed validation

    try:
        keycloak_admin = get_keycloak_admin()
        user = keycloak_admin.get_user(user_id)
        email = email_data.email

        # Additional validation check just to be safe
        if not EMAIL_REGEX.match(email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid email format'
            )

        await keycloak_admin.a_update_user(
            user_id=user_id,
            payload={
                'email': email,
                'emailVerified': False,
                'enabled': user['enabled'],  # Retain existing values
                'username': user['username'],  # Required field
            },
        )

        user_auth: SaasUserAuth = await get_user_auth(request)
        await user_auth.refresh()  # refresh so access token has updated email
        user_auth.email = email
        user_auth.email_verified = False
        response = JSONResponse(
            status_code=status.HTTP_200_OK, content={'message': 'Email changed'}
        )

        # need to set auth cookie to the new tokens
        set_response_cookie(
            request=request,
            response=response,
            keycloak_access_token=user_auth.access_token.get_secret_value(),
            keycloak_refresh_token=user_auth.refresh_token.get_secret_value(),
            secure=False if request.url.hostname == 'localhost' else True,
            accepted_tos=user_auth.accepted_tos,
        )

        await verify_email(request=request, user_id=user_id)

        logger.info(f'Updating email address for {user_id} to {email}')
        return response

    except ValueError as e:
        # Handle validation errors from Pydantic
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f'Error updating email: {str(e)}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='An error occurred while updating the email',
        )


@api_router.put('/resend')
async def resend_email_verification(
    request: Request,
    body: ResendEmailVerificationRequest | None = None,
):
    # Get user_id from body if provided, otherwise from auth
    user_id: str | None = None
    if body and body.user_id:
        user_id = body.user_id
    else:
        try:
            user_id = await get_user_id(request)
        except Exception:
            pass

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='user_id is required in request body or user must be authenticated',
        )

    # Check rate limit (uses user_id if available, otherwise falls back to IP)
    # Use 30 seconds for user-based rate limiting to match frontend cooldown
    await check_rate_limit_by_user_id(
        request=request,
        key_prefix='email_resend',
        user_id=user_id,
        user_rate_limit_seconds=30,
        ip_rate_limit_seconds=60,  # 1 minute for IP-based limiting (more lenient)
    )

    # Get is_auth_flow from body if provided, default to False
    is_auth_flow = body.is_auth_flow if body else False

    await verify_email(request=request, user_id=user_id, is_auth_flow=is_auth_flow)

    logger.info(f'Resending verification email for {user_id}')
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={'message': 'Email verification message sent'},
    )


@api_router.get('/verified')
async def verified_email(request: Request):
    user_auth: SaasUserAuth = await get_user_auth(request)
    await user_auth.refresh()  # refresh so access token has updated email
    user_auth.email_verified = True
    scheme = 'http' if request.url.hostname == 'localhost' else 'https'
    redirect_uri = f'{scheme}://{request.url.netloc}/settings/user'
    response = RedirectResponse(redirect_uri, status_code=302)

    # need to set auth cookie to the new tokens
    set_response_cookie(
        request=request,
        response=response,
        keycloak_access_token=user_auth.access_token.get_secret_value(),
        keycloak_refresh_token=user_auth.refresh_token.get_secret_value(),
        secure=False if request.url.hostname == 'localhost' else True,
        accepted_tos=user_auth.accepted_tos,
    )

    logger.info(f'Email {user_auth.email} verified.')
    return response


async def verify_email(request: Request, user_id: str, is_auth_flow: bool = False):
    keycloak_admin = get_keycloak_admin()
    scheme = 'http' if request.url.hostname == 'localhost' else 'https'
    if is_auth_flow:
        redirect_uri = f'{scheme}://{request.url.netloc}/login?email_verified=true'
    else:
        redirect_uri = f'{scheme}://{request.url.netloc}/api/email/verified'
    logger.info(f'Redirect URI: {redirect_uri}')
    await keycloak_admin.a_send_verify_email(
        user_id=user_id,
        redirect_uri=redirect_uri,
        client_id=KEYCLOAK_CLIENT_ID,
    )
