from typing import Callable

import jwt
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import SecretStr
from server.auth.auth_error import (
    AuthError,
    EmailNotVerifiedError,
    NoCredentialsError,
    TosNotAcceptedError,
)
from server.auth.gitlab_sync import schedule_gitlab_repo_sync
from server.auth.saas_user_auth import SaasUserAuth, token_manager
from server.routes.auth import (
    get_cookie_domain,
    get_cookie_samesite,
    set_response_cookie,
)

from openhands.core.logger import openhands_logger as logger
from openhands.server.user_auth.user_auth import AuthType, get_user_auth
from openhands.server.utils import config


class SetAuthCookieMiddleware:
    """
    Update the auth cookie with the current authentication state if it was refreshed before sending response to user.
    Deleting invalid cookies is handled by CookieError using FastAPIs standard error handling mechanism
    """

    async def __call__(self, request: Request, call_next: Callable):
        keycloak_auth_cookie = request.cookies.get('keycloak_auth')
        logger.debug('request_with_cookie', extra={'cookie': keycloak_auth_cookie})
        try:
            if self._should_attach(request):
                self._check_tos(request)

            response: Response = await call_next(request)
            if not keycloak_auth_cookie:
                return response
            user_auth = self._get_user_auth(request)
            if not user_auth or user_auth.auth_type != AuthType.COOKIE:
                return response
            if user_auth.refreshed:
                set_response_cookie(
                    request=request,
                    response=response,
                    keycloak_access_token=user_auth.access_token.get_secret_value(),
                    keycloak_refresh_token=user_auth.refresh_token.get_secret_value(),
                    secure=False if request.url.hostname == 'localhost' else True,
                    accepted_tos=user_auth.accepted_tos,
                )

                # On re-authentication (token refresh), kick off background sync for GitLab repos
                schedule_gitlab_repo_sync(
                    await user_auth.get_user_id(),
                )

            if (
                self._should_attach(request)
                and not request.url.path.startswith('/api/email')
                and request.url.path
                not in ('/api/settings', '/api/logout', '/api/authenticate')
                and not user_auth.email_verified
            ):
                raise EmailNotVerifiedError

            return response
        except EmailNotVerifiedError as e:
            return JSONResponse(
                {'error': str(e) or e.__class__.__name__}, status.HTTP_403_FORBIDDEN
            )
        except NoCredentialsError as e:
            logger.info(e.__class__.__name__)
            # The user is trying to use an expired token or has not logged in. No special event handling is required
            return JSONResponse(
                {'error': str(e) or e.__class__.__name__}, status.HTTP_401_UNAUTHORIZED
            )
        except AuthError as e:
            logger.warning('auth_error', exc_info=True)
            try:
                await self._logout(request)
            except Exception as logout_error:
                logger.debug(str(logout_error))

            # Send a response that deletes the auth cookie if needed
            response = JSONResponse(
                {'error': str(e) or e.__class__.__name__}, status.HTTP_401_UNAUTHORIZED
            )
            if keycloak_auth_cookie:
                response.delete_cookie(
                    key='keycloak_auth',
                    domain=get_cookie_domain(request),
                    samesite=get_cookie_samesite(request),
                )
            return response

    def _get_user_auth(self, request: Request) -> SaasUserAuth | None:
        return getattr(request.state, 'user_auth', None)

    def _check_tos(self, request: Request):
        keycloak_auth_cookie = request.cookies.get('keycloak_auth')
        auth_header = request.headers.get('Authorization')
        mcp_auth_header = request.headers.get('X-Session-API-Key')
        accepted_tos = False
        if (
            keycloak_auth_cookie is None
            and (auth_header is None or not auth_header.startswith('Bearer '))
            and mcp_auth_header is None
        ):
            raise NoCredentialsError

        jwt_secret: SecretStr = config.jwt_secret  # type: ignore[assignment]
        if keycloak_auth_cookie:
            try:
                decoded = jwt.decode(
                    keycloak_auth_cookie,
                    jwt_secret.get_secret_value(),
                    algorithms=['HS256'],
                )
                accepted_tos = decoded.get('accepted_tos')
            except jwt.exceptions.InvalidSignatureError:
                # If we can't decode the token, treat it as an auth error
                logger.warning('Invalid JWT signature detected')
                raise AuthError('Invalid authentication token')
            except Exception as e:
                # Handle any other JWT decoding errors
                logger.warning(f'JWT decode error: {str(e)}')
                raise AuthError('Invalid authentication token')
        else:
            # Don't fail an API call if the TOS has not been accepted.
            # The user will accept the TOS the next time they login.
            accepted_tos = True

        # TODO: This explicitly checks for "False" so it doesn't logout anyone
        # that has logged in prior to this change:
        # accepted_tos is "None" means the user has not re-logged in since this TOS change.
        # accepted_tos is "False" means the user was shown the TOS but has not accepted.
        # accepted_tos is "True" means the user has accepted the TOS
        #
        # Once the initial deploy is complete and every user has been logged out
        # after this change (12 hrs max), this should be changed to check
        # "if accepted_tos is not None" as there should not be any users with
        # accepted_tos equal to "None"
        if accepted_tos is False and request.url.path != '/api/accept_tos':
            logger.warning('User has not accepted the terms of service')
            raise TosNotAcceptedError

    def _should_attach(self, request: Request) -> bool:
        if request.method == 'OPTIONS':
            return False
        path = request.url.path

        ignore_paths = (
            '/api/options/config',
            '/api/keycloak/callback',
            '/api/billing/success',
            '/api/billing/cancel',
            '/api/billing/customer-setup-success',
            '/api/billing/stripe-webhook',
            '/api/email/resend',
            '/oauth/device/authorize',
            '/oauth/device/token',
            '/api/v1/web-client/config',
            '/api/v1/webhooks/secrets',
        )
        if path in ignore_paths:
            return False

        # Allow public access to shared conversations and events
        if path.startswith('/api/shared-conversations') or path.startswith(
            '/api/shared-events'
        ):
            return False

        is_mcp = path.startswith('/mcp')
        is_api_route = path.startswith('/api')
        return is_api_route or is_mcp

    async def _logout(self, request: Request):
        # Log out of keycloak - this prevents issues where you did not log in with the idp you believe you used
        try:
            user_auth: SaasUserAuth = await get_user_auth(request)
            if user_auth and user_auth.refresh_token:
                await token_manager.logout(user_auth.refresh_token.get_secret_value())
        except Exception:
            logger.debug('Error logging out')
