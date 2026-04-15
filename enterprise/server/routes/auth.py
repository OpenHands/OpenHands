import base64
import json
import uuid
import warnings
from datetime import datetime, timezone
from typing import Annotated, Optional, cast
from urllib.parse import quote, urlencode
from uuid import UUID as parse_uuid

import posthog
from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import SecretStr
from server.auth.constants import (
    KEYCLOAK_CLIENT_ID,
    KEYCLOAK_REALM_NAME,
    KEYCLOAK_SERVER_URL_EXT,
    RECAPTCHA_SITE_KEY,
    ROLE_CHECK_ENABLED,
)
from server.auth.gitlab_sync import schedule_gitlab_repo_sync
from server.auth.recaptcha_service import recaptcha_service
from server.auth.saas_user_auth import SaasUserAuth
from server.auth.token_manager import TokenManager
from server.auth.user.user_authorizer import (
    UserAuthorizer,
    depends_user_authorizer,
)
from server.config import sign_token
from server.constants import IS_FEATURE_ENV, IS_LOCAL_ENV
from server.routes.event_webhook import _get_session_api_key, _get_user_id
from server.services.org_invitation_service import (
    EmailMismatchError,
    InvitationExpiredError,
    InvitationInvalidError,
    OrgInvitationService,
    UserAlreadyMemberError,
)
from server.utils.rate_limit_utils import check_rate_limit_by_user_id
from server.utils.url_utils import get_cookie_domain, get_cookie_samesite, get_web_url
from sqlalchemy import select
from storage.database import a_session_maker
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
    domain = get_cookie_domain()
    if domain:
        response.set_cookie(
            key='keycloak_auth',
            value=signed_token,
            domain=domain,
            httponly=True,
            secure=secure,
            samesite=get_cookie_samesite(),
        )
    else:
        response.set_cookie(
            key='keycloak_auth',
            value=signed_token,
            httponly=True,
            secure=secure,
            samesite=get_cookie_samesite(),
        )


def _extract_oauth_state(state: str | None) -> tuple[str, str | None, str | None]:
    """Extract redirect URL, reCAPTCHA token, and invitation token from OAuth state.

    Returns:
        Tuple of (redirect_url, recaptcha_token, invitation_token).
        Tokens may be None.
    """
    if not state:
        return '', None, None

    try:
        # Try to decode as JSON (new format with reCAPTCHA and/or invitation)
        state_data = json.loads(base64.urlsafe_b64decode(state.encode()).decode())
        return (
            state_data.get('redirect_url', ''),
            state_data.get('recaptcha_token'),
            state_data.get('invitation_token'),
        )
    except Exception:
        # Old format - state is just the redirect URL
        return state, None, None


@oauth_router.get('/keycloak/callback')
async def keycloak_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
    user_authorizer: UserAuthorizer = depends_user_authorizer(),
):
    # Extract redirect URL, reCAPTCHA token, and invitation token from state
    redirect_url, recaptcha_token, invitation_token = _extract_oauth_state(state)

    if redirect_url is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Missing state in request params',
        )

    if not code:
        # check if this is a forward from the account linking page
        if (
            error == 'temporarily_unavailable'
            and error_description == 'authentication_expired'
        ):
            return RedirectResponse(redirect_url, status_code=302)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Missing code in request params',
        )

    web_url = get_web_url(request)
    redirect_uri = web_url + request.url.path

    (
        keycloak_access_token,
        keycloak_refresh_token,
    ) = await token_manager.get_keycloak_tokens(code, redirect_uri)
    if not keycloak_access_token or not keycloak_refresh_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Problem retrieving Keycloak tokens',
        )

    user_info = await token_manager.get_user_info(keycloak_access_token)
    logger.debug(f'user_info: {user_info}')
    if ROLE_CHECK_ENABLED and user_info.roles is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail='Missing required role'
        )

    authorization = await user_authorizer.authorize_user(user_info)
    if not authorization.success:
        # For duplicate_email errors, clean up the newly created Keycloak user
        # (only if they're not already in our UserStore, i.e., they're a new user)
        if authorization.error_detail == 'duplicate_email':
            try:
                existing_user = await UserStore.get_user_by_id(user_info.sub)
                if not existing_user:
                    # New user created during OAuth should be deleted from Keycloak
                    await token_manager.delete_keycloak_user(user_info.sub)
                    logger.info(
                        f'Deleted orphaned Keycloak user {user_info.sub} '
                        'after duplicate_email rejection'
                    )
            except Exception as e:
                # Log but don't fail - user should still get 401 response
                logger.warning(
                    f'Failed to clean up orphaned Keycloak user {user_info.sub}: {e}'
                )
        # Return unauthorized
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=authorization.error_detail,
        )

    email = user_info.email
    user_id = user_info.sub
    user_info_dict = user_info.model_dump(exclude_none=True)
    user = await UserStore.get_user_by_id(user_id)
    if not user:
        user = await UserStore.create_user(user_id, user_info_dict)
    else:
        # Existing user — gradually backfill contact_name if it still has a username-style value
        await UserStore.backfill_contact_name(user_id, user_info_dict)
        await UserStore.backfill_user_email(user_id, user_info_dict)

    if not user:
        logger.error(f'Failed to authenticate user {user_info.email}')
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f'Failed to authenticate user {user_info.email}',
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
            error_url = f'{web_url}/login?recaptcha_blocked=true'
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
                user_id=user_id,
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
                error_url = f'{web_url}/login?recaptcha_blocked=true'
                return RedirectResponse(error_url, status_code=302)

        except Exception as e:
            logger.exception(f'reCAPTCHA verification error at callback: {e}')
            # Fail open - continue with login if reCAPTCHA service unavailable

    # Check email verification status
    email_verified = user_info.email_verified or False
    if not email_verified:
        # Send verification email with rate limiting to prevent abuse
        # Users who repeatedly login without verifying would otherwise trigger
        # unlimited verification emails
        # Import locally to avoid circular import with email.py
        from server.routes.email import verify_email

        # Rate limit verification emails during auth flow (60 seconds per user)
        # This is separate from the manual resend rate limit which uses 30 seconds
        rate_limited = False
        try:
            await check_rate_limit_by_user_id(
                request=request,
                key_prefix='auth_verify_email',
                user_id=user_id,
                user_rate_limit_seconds=60,
                ip_rate_limit_seconds=120,
            )
            await verify_email(request=request, user_id=user_id, is_auth_flow=True)
        except HTTPException as e:
            if e.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                # Rate limited - still redirect to verification page but don't send email
                rate_limited = True
                logger.info(
                    f'Rate limited verification email for user {user_id} during auth flow'
                )
            else:
                raise

        verification_redirect_url = (
            f'{web_url}/login?email_verification_required=true&user_id={user_id}'
        )
        if rate_limited:
            verification_redirect_url = f'{verification_redirect_url}&rate_limited=true'

        # Preserve invitation token so it can be included in OAuth state after verification
        if invitation_token:
            verification_redirect_url = (
                f'{verification_redirect_url}&invitation_token={invitation_token}'
            )
        response = RedirectResponse(verification_redirect_url, status_code=302)
        return response

    # default to github IDP for now.
    # TODO: remove default once Keycloak is updated universally with the new attribute.
    idp: str = user_info.identity_provider or ProviderType.GITHUB.value
    logger.info(f'Full IDP is {idp}')
    idp_type = 'oidc'
    if ':' in idp:
        idp, idp_type = idp.rsplit(':', 1)
        idp_type = idp_type.lower()

    await token_manager.store_idp_tokens(
        ProviderType(idp), user_id, keycloak_access_token
    )

    valid_offline_token = (
        await token_manager.validate_offline_token(user_id=user_info.sub)
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
        param_str = urlencode(
            {
                'client_id': KEYCLOAK_CLIENT_ID,
                'response_type': 'code',
                'kc_idp_hint': idp,
                'redirect_uri': f'{web_url}/oauth/keycloak/offline/callback',
                'scope': 'openid email profile offline_access',
                'state': state,
            }
        )
        redirect_url = (
            f'{KEYCLOAK_SERVER_URL_EXT}/realms/{KEYCLOAK_REALM_NAME}/protocol/openid-connect/auth'
            f'?{param_str}'
        )

    has_accepted_tos = user.accepted_tos is not None

    # Process invitation token if present (after email verification but before TOS)
    if invitation_token:
        try:
            logger.info(
                'Processing invitation token during auth callback',
                extra={
                    'user_id': user_id,
                    'invitation_token_prefix': invitation_token[:10] + '...',
                },
            )

            await OrgInvitationService.accept_invitation(
                invitation_token, parse_uuid(user_id)
            )
            logger.info(
                'Invitation accepted during auth callback',
                extra={'user_id': user_id},
            )

        except InvitationExpiredError:
            logger.warning(
                'Invitation expired during auth callback',
                extra={'user_id': user_id},
            )
            # Add query param to redirect URL
            if '?' in redirect_url:
                redirect_url = f'{redirect_url}&invitation_expired=true'
            else:
                redirect_url = f'{redirect_url}?invitation_expired=true'

        except InvitationInvalidError as e:
            logger.warning(
                'Invalid invitation during auth callback',
                extra={'user_id': user_id, 'error': str(e)},
            )
            if '?' in redirect_url:
                redirect_url = f'{redirect_url}&invitation_invalid=true'
            else:
                redirect_url = f'{redirect_url}?invitation_invalid=true'

        except UserAlreadyMemberError:
            logger.info(
                'User already member during invitation acceptance',
                extra={'user_id': user_id},
            )
            if '?' in redirect_url:
                redirect_url = f'{redirect_url}&already_member=true'
            else:
                redirect_url = f'{redirect_url}?already_member=true'

        except EmailMismatchError as e:
            logger.warning(
                'Email mismatch during auth callback invitation acceptance',
                extra={'user_id': user_id, 'error': str(e)},
            )
            if '?' in redirect_url:
                redirect_url = f'{redirect_url}&email_mismatch=true'
            else:
                redirect_url = f'{redirect_url}?email_mismatch=true'

        except Exception as e:
            logger.exception(
                'Unexpected error processing invitation during auth callback',
                extra={'user_id': user_id, 'error': str(e)},
            )
            # Don't fail the login if invitation processing fails
            if '?' in redirect_url:
                redirect_url = f'{redirect_url}&invitation_error=true'
            else:
                redirect_url = f'{redirect_url}?invitation_error=true'

    # If the user hasn't accepted the TOS, redirect to the TOS page
    if not has_accepted_tos:
        encoded_redirect_url = quote(redirect_url, safe='')
        tos_redirect_url = f'{web_url}/accept-tos?redirect_url={encoded_redirect_url}'
        if invitation_token:
            tos_redirect_url = f'{tos_redirect_url}&invitation_success=true'
        response = RedirectResponse(tos_redirect_url, status_code=302)
    else:
        if invitation_token:
            redirect_url = f'{redirect_url}&invitation_success=true'
        response = RedirectResponse(redirect_url, status_code=302)

    set_response_cookie(
        request=request,
        response=response,
        keycloak_access_token=keycloak_access_token,
        keycloak_refresh_token=keycloak_refresh_token,
        secure=True if redirect_url.startswith('https') else False,
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

    web_url = get_web_url(request)
    redirect_uri = web_url + request.url.path
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
    # sub is a required field in KeycloakUserInfo, validation happens in get_user_info

    await token_manager.store_offline_token(
        user_id=user_info.sub, offline_token=keycloak_refresh_token
    )

    redirect_url, _, _ = _extract_oauth_state(state)
    return RedirectResponse(redirect_url if redirect_url else web_url, status_code=302)


@oauth_router.get('/github/callback')
async def github_dummy_callback(request: Request):
    """Callback for GitHub that just forwards the user to the app base URL."""
    web_url = get_web_url(request)
    return RedirectResponse(web_url, status_code=302)


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
                domain=get_cookie_domain(),
                samesite=get_cookie_samesite(),
            )

        return response


@api_router.post('/accept_tos')
async def accept_tos(request: Request):
    user_auth = cast(SaasUserAuth, await get_user_auth(request))
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
    web_url = get_web_url(request)
    redirect_url = body.get('redirect_url', str(web_url))

    # Update user settings with TOS acceptance
    accepted_tos: datetime = datetime.now(timezone.utc).replace(tzinfo=None)
    async with a_session_maker() as session:
        result = await session.execute(
            select(User).where(User.id == uuid.UUID(user_id))
        )
        user = result.scalar_one_or_none()
        if not user:
            await session.rollback()
            logger.error('User for {user_id} not found.')
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={'error': 'User does not exist'},
            )
        user.accepted_tos = accepted_tos
        await session.commit()

        logger.info(f'User {user_id} accepted TOS')

    response = JSONResponse(
        status_code=status.HTTP_200_OK, content={'redirect_url': redirect_url}
    )

    set_response_cookie(
        request=request,
        response=response,
        keycloak_access_token=access_token.get_secret_value(),
        keycloak_refresh_token=refresh_token.get_secret_value(),
        secure=not IS_LOCAL_ENV,
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
        domain=get_cookie_domain(),
        samesite=get_cookie_samesite(),
    )

    # Try to properly logout from Keycloak, but don't fail if it doesn't work
    try:
        user_auth = cast(SaasUserAuth, await get_user_auth(request))
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
