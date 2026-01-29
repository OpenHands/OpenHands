import base64
import json
import uuid
import warnings
from datetime import datetime, timezone
from typing import Annotated, Literal, Optional
from urllib.parse import quote

import posthog
from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import SecretStr
from server.auth.auth_utils import user_verifier
from server.auth.constants import (
    KEYCLOAK_CLIENT_ID,
    KEYCLOAK_REALM_NAME,
    KEYCLOAK_SERVER_URL_EXT,
    RECAPTCHA_SITE_KEY,
    ROLE_CHECK_ENABLED,
)
from server.auth.domain_blocker import domain_blocker
from server.auth.gitlab_sync import schedule_gitlab_repo_sync
from server.auth.recaptcha_service import recaptcha_service
from server.auth.saas_user_auth import SaasUserAuth
from server.auth.token_manager import TokenManager
from server.config import sign_token
from server.constants import IS_FEATURE_ENV
from server.routes.event_webhook import _get_session_api_key, _get_user_id
from storage.database import session_maker
from storage.user import User
from storage.user_store import UserStore

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.provider import ProviderHandler
from openhands.integrations.service_types import ProviderType, TokenResponse
from openhands.server.services.conversation_service import create_provider_tokens_object
from openhands.server.shared import config
from openhands.server.user_auth import get_access_token
from openhands.server.user_auth.user_auth import get_user_auth

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

api_router = APIRouter(prefix='/api')
oauth_router = APIRouter(prefix='/oauth')

token_manager = TokenManager()


def set_response_cookie(
    request: Request,
    response: Response,
    keycloak_access_token: str,
    keycloak_refresh_token: str,
    secure: bool = True,
    accepted_tos: bool = False,
):
    # Create a signed JWT token
    cookie_data = {
        'access_token': keycloak_access_token,
        'refresh_token': keycloak_refresh_token,
        'accepted_tos': accepted_tos,
    }
    signed_token = sign_token(cookie_data, config.jwt_secret.get_secret_value())  # type: ignore

    # Set secure cookie with signed token
    domain = get_cookie_domain(request)
    if domain:
        response.set_cookie(
            key='keycloak_auth',
            value=signed_token,
            domain=domain,
            httponly=True,
            secure=secure,
            samesite=get_cookie_samesite(request),
        )
    else:
        response.set_cookie(
            key='keycloak_auth',
            value=signed_token,
            httponly=True,
            secure=secure,
            samesite=get_cookie_samesite(request),
        )


def get_cookie_domain(request: Request) -> str | None:
    # for now just use the full hostname except for staging stacks.
    return (
        None
        if not request.url.hostname
        or request.url.hostname.endswith('staging.all-hands.dev')
        else request.url.hostname
    )


def get_cookie_samesite(request: Request) -> Literal['lax', 'strict']:
    # for localhost and feature/staging stacks we set it to 'lax' as the cookie domain won't allow 'strict'
    return (
        'lax'
        if request.url.hostname == 'localhost'
        or (request.url.hostname or '').endswith('staging.all-hands.dev')
        else 'strict'
    )


def _extract_recaptcha_state(state: str | None) -> tuple[str, str | None]:
    """Extract redirect URL and reCAPTCHA token from OAuth state.

    Returns:
        Tuple of (redirect_url, recaptcha_token). Token may be None.
    """
    if not state:
        return '', None

    try:
        # Try to decode as JSON (new format with reCAPTCHA)
        state_data = json.loads(base64.urlsafe_b64decode(state.encode()).decode())
        return state_data.get('redirect_url', ''), state_data.get('recaptcha_token')
    except Exception:
        # Old format - state is just the redirect URL
        return state, None


@oauth_router.get('/keycloak/callback')
async def keycloak_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
):
    # Extract redirect URL and reCAPTCHA token from state
    redirect_url, recaptcha_token = _extract_recaptcha_state(state)
    if not redirect_url:
        redirect_url = str(request.base_url)

    if not code:
        # check if this is a forward from the account linking page
        if (
            error == 'temporarily_unavailable'
            and error_description == 'authentication_expired'
        ):
            return RedirectResponse(redirect_url, status_code=302)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={'error': 'Missing code in request params'},
        )
    scheme = 'http' if request.url.hostname == 'localhost' else 'https'
    redirect_uri = f'{scheme}://{request.url.netloc}{request.url.path}'
    logger.debug(f'code: {code}, redirect_uri: {redirect_uri}')

    (
        keycloak_access_token,
        keycloak_refresh_token,
    ) = await token_manager.get_keycloak_tokens(code, redirect_uri)
    if not keycloak_access_token or not keycloak_refresh_token:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={'error': 'Problem retrieving Keycloak tokens'},
        )

    user_info = await token_manager.get_user_info(keycloak_access_token)
    logger.debug(f'user_info: {user_info}')
    if ROLE_CHECK_ENABLED and 'roles' not in user_info:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={'error': 'Missing required role'},
        )

    if 'sub' not in user_info or 'preferred_username' not in user_info:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={'error': 'Missing user ID or username in response'},
        )

    email = user_info.get('email')
    user_id = user_info['sub']
    user = await UserStore.get_user_by_id_async(user_id)
    if not user:
        user = await UserStore.create_user(user_id, user_info)

    if not user:
        logger.error(f'Failed to authenticate user {user_info["preferred_username"]}')
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                'error': f'Failed to authenticate user {user_info["preferred_username"]}'
            },
        )

    logger.info(f'Logging in user {str(user.id)} in org {user.current_org_id}')

    # reCAPTCHA verification with Account Defender
    if RECAPTCHA_SITE_KEY:
        if not recaptcha_token:
            logger.warning(
                'recaptcha_token_missing',
                extra={
                    'user_id': user_id,
                    'email': email,
                },
            )
            error_url = f'{request.base_url}login?recaptcha_blocked=true'
            return RedirectResponse(error_url, status_code=302)

        user_ip = request.client.host if request.client else 'unknown'
        user_agent = request.headers.get('User-Agent', '')

        # Handle X-Forwarded-For for proxied requests
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            user_ip = forwarded_for.split(',')[0].strip()

        try:
            result = recaptcha_service.create_assessment(
                token=recaptcha_token,
                action='LOGIN',
                user_ip=user_ip,
                user_agent=user_agent,
                email=email,
            )

            if not result.allowed:
                logger.warning(
                    'recaptcha_blocked_at_callback',
                    extra={
                        'user_ip': user_ip,
                        'score': result.score,
                        'user_id': user_id,
                    },
                )
                # Redirect to home with error parameter
                error_url = f'{request.base_url}login?recaptcha_blocked=true'
                return RedirectResponse(error_url, status_code=302)

        except Exception as e:
            logger.exception(f'reCAPTCHA verification error at callback: {e}')
            # Fail open - continue with login if reCAPTCHA service unavailable

    # Check if email domain is blocked
    if email and domain_blocker.is_domain_blocked(email):
        logger.warning(
            f'Blocked authentication attempt for email: {email}, user_id: {user_id}'
        )

        # Disable the Keycloak account
        await token_manager.disable_keycloak_user(user_id, email)

        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                'error': 'Access denied: Your email domain is not allowed to access this service'
            },
        )

    # Check for duplicate email with + modifier
    if email:
        try:
            has_duplicate = await token_manager.check_duplicate_base_email(
                email, user_id
            )
            if has_duplicate:
                logger.warning(
                    f'Blocked signup attempt for email {email} - duplicate base email found',
                    extra={'user_id': user_id, 'email': email},
                )

                # Delete the Keycloak user that was automatically created during OAuth
                # This prevents orphaned accounts in Keycloak
                # The delete_keycloak_user method already handles all errors internally
                deletion_success = await token_manager.delete_keycloak_user(user_id)
                if deletion_success:
                    logger.info(
                        f'Deleted Keycloak user {user_id} after detecting duplicate email {email}'
                    )
                else:
                    logger.warning(
                        f'Failed to delete Keycloak user {user_id} after detecting duplicate email {email}. '
                        f'User may need to be manually cleaned up.'
                    )

                # Redirect to home page with query parameter indicating the issue
                home_url = f'{request.base_url}/login?duplicated_email=true'
                return RedirectResponse(home_url, status_code=302)
        except Exception as e:
            # Log error but allow signup to proceed (fail open)
            logger.error(
                f'Error checking duplicate email for {email}: {e}',
                extra={'user_id': user_id, 'email': email},
            )

    # Check email verification status
    email_verified = user_info.get('email_verified', False)
    if not email_verified:
        # Send verification email
        # Import locally to avoid circular import with email.py
        from server.routes.email import verify_email

        await verify_email(request=request, user_id=user_id, is_auth_flow=True)
        redirect_url = f'{request.base_url}login?email_verification_required=true&user_id={user_id}'
        response = RedirectResponse(redirect_url, status_code=302)
        return response

    # default to github IDP for now.
    # TODO: remove default once Keycloak is updated universally with the new attribute.
    idp: str = user_info.get('identity_provider', ProviderType.GITHUB.value)
    logger.info(f'Full IDP is {idp}')
    idp_type = 'oidc'
    if ':' in idp:
        idp, idp_type = idp.rsplit(':', 1)
        idp_type = idp_type.lower()

    await token_manager.store_idp_tokens(
        ProviderType(idp), user_id, keycloak_access_token
    )

    username = user_info['preferred_username']
    if user_verifier.is_active() and not user_verifier.is_user_allowed(username):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={'error': 'Not authorized via waitlist'},
        )

    valid_offline_token = (
        await token_manager.validate_offline_token(user_id=user_info['sub'])
        if idp_type != 'saml'
        else True
    )

    logger.debug(
        f'keycloakAccessToken: {keycloak_access_token}, keycloakUserId: {user_id}'
    )

    # adding in posthog tracking

    # If this is a feature environment, add "FEATURE_" prefix to user_id for PostHog
    posthog_user_id = f'FEATURE_{user_id}' if IS_FEATURE_ENV else user_id

    try:
        posthog.set(
            distinct_id=posthog_user_id,
            properties={
                'user_id': posthog_user_id,
                'original_user_id': user_id,
                'is_feature_env': IS_FEATURE_ENV,
            },
        )
    except Exception as e:
        logger.error(
            'auth:posthog_set:failed',
            extra={
                'user_id': user_id,
                'error': str(e),
            },
        )
        # Continue execution as this is not critical

    logger.info(
        'user_logged_in',
        extra={
            'idp': idp,
            'idp_type': idp_type,
            'posthog_user_id': posthog_user_id,
            'is_feature_env': IS_FEATURE_ENV,
        },
    )

    if not valid_offline_token:
        redirect_url = (
            f'{KEYCLOAK_SERVER_URL_EXT}/realms/{KEYCLOAK_REALM_NAME}/protocol/openid-connect/auth'
            f'?client_id={KEYCLOAK_CLIENT_ID}&response_type=code'
            f'&kc_idp_hint={idp}'
            f'&redirect_uri={scheme}%3A%2F%2F{request.url.netloc}%2Foauth%2Fkeycloak%2Foffline%2Fcallback'
            f'&scope=openid%20email%20profile%20offline_access'
            f'&state={state}'
        )

    has_accepted_tos = user.accepted_tos is not None
    # If the user hasn't accepted the TOS, redirect to the TOS page
    if not has_accepted_tos:
        encoded_redirect_url = quote(redirect_url, safe='')
        tos_redirect_url = (
            f'{request.base_url}accept-tos?redirect_url={encoded_redirect_url}'
        )
        response = RedirectResponse(tos_redirect_url, status_code=302)
    else:
        response = RedirectResponse(redirect_url, status_code=302)

    set_response_cookie(
        request=request,
        response=response,
        keycloak_access_token=keycloak_access_token,
        keycloak_refresh_token=keycloak_refresh_token,
        secure=True if scheme == 'https' else False,
        accepted_tos=has_accepted_tos,
    )

    # Sync GitLab repos & set up webhooks
    # Use Keycloak access token (first-time users lack offline token at this stage)
    # Normally, offline token is used to fetch GitLab token via user_id
    schedule_gitlab_repo_sync(user_id, SecretStr(keycloak_access_token))
    return response


@oauth_router.get('/keycloak/offline/callback')
async def keycloak_offline_callback(code: str, state: str, request: Request):
    if not code:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={'error': 'Missing code in request params'},
        )
    scheme = 'https'
    if request.url.hostname == 'localhost':
        scheme = 'http'
    redirect_uri = f'{scheme}://{request.url.netloc}{request.url.path}'
    logger.debug(f'code: {code}, redirect_uri: {redirect_uri}')

    (
        keycloak_access_token,
        keycloak_refresh_token,
    ) = await token_manager.get_keycloak_tokens(code, redirect_uri)
    if not keycloak_access_token or not keycloak_refresh_token:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={'error': 'Problem retrieving Keycloak tokens'},
        )

    user_info = await token_manager.get_user_info(keycloak_access_token)
    logger.debug(f'user_info: {user_info}')
    if 'sub' not in user_info:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={'error': 'Missing Keycloak ID in response'},
        )

    await token_manager.store_offline_token(
        user_id=user_info['sub'], offline_token=keycloak_refresh_token
    )

    return RedirectResponse(state if state else request.base_url, status_code=302)


@oauth_router.get('/github/callback')
async def github_dummy_callback(request: Request):
    """Callback for GitHub that just forwards the user to the app base URL."""
    return RedirectResponse(request.base_url, status_code=302)


@api_router.post('/authenticate')
async def authenticate(request: Request):
    try:
        await get_access_token(request)
        return JSONResponse(
            status_code=status.HTTP_200_OK, content={'message': 'User authenticated'}
        )
    except Exception:
        # For any error during authentication, clear the auth cookie and return 401
        response = JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={'error': 'User is not authenticated'},
        )

        # Delete the auth cookie if it exists
        keycloak_auth_cookie = request.cookies.get('keycloak_auth')
        if keycloak_auth_cookie:
            response.delete_cookie(
                key='keycloak_auth',
                domain=get_cookie_domain(request),
                samesite=get_cookie_samesite(request),
            )

        return response


@api_router.post('/accept_tos')
async def accept_tos(request: Request):
    user_auth: SaasUserAuth = await get_user_auth(request)
    access_token = await user_auth.get_access_token()
    refresh_token = user_auth.refresh_token
    user_id = await user_auth.get_user_id()

    if not access_token or not refresh_token or not user_id:
        logger.warning(
            f'accept_tos: One or more is None: access_token {access_token}, refresh_token {refresh_token}, user_id {user_id}'
        )
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={'error': 'User is not authenticated'},
        )

    # Get redirect URL from request body
    body = await request.json()
    redirect_url = body.get('redirect_url', str(request.base_url))

    # Update user settings with TOS acceptance
    accepted_tos: datetime = datetime.now(timezone.utc)
    with session_maker() as session:
        user = session.query(User).filter(User.id == uuid.UUID(user_id)).first()
        if not user:
            session.rollback()
            logger.error('User for {user_id} not found.')
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={'error': 'User does not exist'},
            )
        user.accepted_tos = accepted_tos
        session.commit()

        logger.info(f'User {user_id} accepted TOS')

    response = JSONResponse(
        status_code=status.HTTP_200_OK, content={'redirect_url': redirect_url}
    )

    set_response_cookie(
        request=request,
        response=response,
        keycloak_access_token=access_token.get_secret_value(),
        keycloak_refresh_token=refresh_token.get_secret_value(),
        secure=False if request.url.hostname == 'localhost' else True,
        accepted_tos=True,
    )
    return response


@api_router.post('/logout')
async def logout(request: Request):
    # Always create the response object first to ensure we can return it even if errors occur
    response = JSONResponse(
        status_code=status.HTTP_200_OK,
        content={'message': 'User logged out'},
    )

    # Always delete the cookie regardless of what happens
    response.delete_cookie(
        key='keycloak_auth',
        domain=get_cookie_domain(request),
        samesite=get_cookie_samesite(request),
    )

    # Try to properly logout from Keycloak, but don't fail if it doesn't work
    try:
        user_auth: SaasUserAuth = await get_user_auth(request)
        if user_auth and user_auth.refresh_token:
            refresh_token = user_auth.refresh_token.get_secret_value()
            await token_manager.logout(refresh_token)
    except Exception as e:
        # Log any errors but don't fail the request
        logger.debug(f'Error during logout: {str(e)}')
        # We still want to clear the cookie and return success

    return response


@api_router.get('/refresh-tokens', response_model=TokenResponse)
async def refresh_tokens(
    request: Request,
    provider: ProviderType,
    sid: str,
    x_session_api_key: Annotated[str | None, Header(alias='X-Session-API-Key')],
) -> TokenResponse:
    """Return the latest token for a given provider."""
    user_id = _get_user_id(sid)
    session_api_key = await _get_session_api_key(user_id, sid)
    if session_api_key != x_session_api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Forbidden')

    logger.info(f'Refreshing token for conversation {sid}')
    provider_handler = ProviderHandler(
        create_provider_tokens_object([provider]), external_auth_id=user_id
    )
    service = provider_handler.get_service(provider)
    token = await service.get_latest_token()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No token found for provider '{provider}'",
        )

    return TokenResponse(token=token.get_secret_value())
