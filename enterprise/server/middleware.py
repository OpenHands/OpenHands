from typing import Callable, cast

import jwt
from fastapi import Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from server.auth.auth_error import (
    AuthError,
    EmailNotVerifiedError,
    NoCredentialsError,
    TosNotAcceptedError,
)
from server.auth.cookie_chunking import delete_chunked_cookie, read_chunked_cookie
from server.auth.gitlab_sync import schedule_gitlab_repo_sync
from server.auth.saas_user_auth import SaasUserAuth, token_manager
from server.routes.auth import set_response_cookie
from server.utils.url_utils import get_cookie_domain, get_cookie_samesite
from starlette.types import ASGIApp, Receive, Scope, Send

from openhands.app_server.user_auth.user_auth import AuthType, UserAuth, get_user_auth
from openhands.app_server.utils.logger import openhands_logger as logger


class SetAuthCookieMiddleware:
    """
    Update the auth cookie with the current authentication state if it was refreshed before sending response to user.
    Deleting invalid cookies is handled by CookieError using FastAPIs standard error handling mechanism
    """

    async def __call__(self, request: Request, call_next: Callable):
        keycloak_auth_cookie = read_chunked_cookie(request, 'keycloak_auth')
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
                if user_auth.access_token is None:
                    return response
                set_response_cookie(
                    request=request,
                    response=response,
                    keycloak_access_token=user_auth.access_token.get_secret_value(),
                    keycloak_refresh_token=user_auth.refresh_token.get_secret_value(),
                    secure=False if request.url.hostname == 'localhost' else True,
                    accepted_tos=user_auth.accepted_tos or False,
                )

                # On re-authentication (token refresh), kick off background sync for GitLab repos
                user_id = await user_auth.get_user_id()
                if user_id:
                    schedule_gitlab_repo_sync(user_id)

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
            # Only attempt a Keycloak logout when this looked like a cookie
            # session going bad. Bearer-token auth failures (e.g., a
            # ``BearerTokenError`` from a transient Keycloak refresh
            # failure) must NOT revoke the user's offline session — that
            # would brick every subsequent API-key call until the user
            # logs back in through the browser. The API key's lifecycle is
            # managed via key mint/delete, not via per-request refresh
            # outcomes. See ``_logout`` for the defense-in-depth check.
            if keycloak_auth_cookie:
                try:
                    await self._logout(request)
                except Exception as logout_error:
                    logger.debug(str(logout_error))

            # Send a response that deletes the auth cookie if needed
            response = JSONResponse(
                {'error': str(e) or e.__class__.__name__}, status.HTTP_401_UNAUTHORIZED
            )
            if keycloak_auth_cookie:
                delete_chunked_cookie(
                    response,
                    'keycloak_auth',
                    domain=get_cookie_domain(),
                    samesite=get_cookie_samesite(),
                )
            return response

    def _get_user_auth(self, request: Request) -> SaasUserAuth | None:
        user_auth: UserAuth | None = getattr(request.state, 'user_auth', None)
        if user_auth is None:
            return None
        return cast(SaasUserAuth, user_auth)

    def _check_tos(self, request: Request):
        keycloak_auth_cookie = read_chunked_cookie(request, 'keycloak_auth')
        auth_header = request.headers.get('Authorization')
        mcp_auth_header = request.headers.get('X-Session-API-Key')
        api_auth_header = request.headers.get('X-Access-Token')
        api_key_cookie = request.cookies.get('api_key')
        accepted_tos: bool | None = False
        if (
            keycloak_auth_cookie is None
            and (auth_header is None or not auth_header.startswith('Bearer '))
            and mcp_auth_header is None
            and api_auth_header is None
            and api_key_cookie is None
        ):
            raise NoCredentialsError

        if keycloak_auth_cookie:
            try:
                from storage.encrypt_utils import get_jwt_service

                decoded = get_jwt_service().verify_jws_token(keycloak_auth_cookie)
                accepted_tos = decoded.get('accepted_tos')
            except (jwt.InvalidTokenError, ValueError):
                logger.warning('Invalid JWT signature detected')
                raise AuthError('Invalid authentication token')
            except Exception as e:
                logger.warning(f'JWT decode error: {str(e)}')
                raise AuthError('Invalid authentication token') from e
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
            '/api/organizations/members/invite/accept',
            '/oauth/device/authorize',
            '/oauth/device/token',
            '/api/v1/web-client/config',
        )
        if path in ignore_paths:
            return False

        # Allow public access to shared conversations and events
        if path.startswith('/api/shared-conversations') or path.startswith(
            '/api/shared-events'
        ):
            return False

        # Webhooks access is controlled using separate API keys
        if path.startswith('/api/v1/webhooks/'):
            return False

        # Service API uses its own authentication (X-Service-API-Key header)
        if path.startswith('/api/service/'):
            return False

        is_mcp = path.startswith('/mcp')
        is_api_route = path.startswith('/api')
        return is_api_route or is_mcp

    async def _logout(self, request: Request):
        # Log out of keycloak - this prevents issues where you did not log in with the idp you believe you used.
        #
        # IMPORTANT: only terminate the Keycloak session when the request
        # carried a *cookie* (browser session). For bearer-token (API
        # key) requests, ``user_auth.refresh_token`` is the user's stored
        # *offline_token* loaded from ``OfflineTokenStore``. Calling
        # ``token_manager.logout`` with that value asks Keycloak to
        # revoke the offline session, which permanently breaks every API
        # key minted for the user until they re-authenticate through the
        # browser (``/keycloak/callback`` rewrites the offline_token).
        # A single transient Keycloak hiccup that surfaces as
        # ``BearerTokenError`` must not be allowed to cause this damage.
        try:
            user_auth = cast(SaasUserAuth, await get_user_auth(request))
            if (
                user_auth
                and user_auth.refresh_token
                and user_auth.auth_type == AuthType.COOKIE
            ):
                await token_manager.logout(user_auth.refresh_token.get_secret_value())
        except Exception:
            logger.debug('Error logging out')


_CREDENTIALLESS_PATH_PREFIXES = (
    # RFC 8628 device authorization endpoints — unauthenticated by design,
    # called cross-origin from clients that are exchanging device codes for
    # API keys.
    '/oauth/device/authorize',
    '/oauth/device/token',
)


class _OriginStrippingApp:
    """Hide Origin from inner middleware after outer CORS classifies the request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        inner_scope = dict(scope)
        inner_scope['headers'] = tuple(
            (name, value)
            for name, value in scope['headers']
            if name.lower() != b'origin'
        )
        await self.app(inner_scope, receive, send)


class ApiKeyAwareCORSMiddleware:
    """CORS dispatcher that loosens the policy for credential-less requests.

    Requests that authenticate via API key (``Authorization: Bearer …``,
    ``X-Session-API-Key``, or ``X-Access-Token``) or that target a known
    unauthenticated cross-origin endpoint (RFC 8628 device flow) get
    ``Access-Control-Allow-Origin: *`` with credentials disabled — the
    wildcard is safe because the browser cannot attach cookies when
    credentials are off, so the only way to authenticate is the explicit
    key (or no auth, for public endpoints).

    Cookie/session requests keep the strict origin allowlist with
    credentials enabled.
    """

    def __init__(self, app: ASGIApp, allow_origins: list[str]) -> None:
        self._permissive = CORSMiddleware(
            _OriginStrippingApp(app),
            allow_origins=['*'],
            allow_credentials=False,
            allow_methods=['*'],
            allow_headers=['*'],
        )
        self._strict = CORSMiddleware(
            app,
            allow_origins=allow_origins,
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*'],
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] == 'http' and self._is_credentialless(scope):
            await self._permissive(scope, receive, send)
        else:
            await self._strict(scope, receive, send)

    @staticmethod
    def _is_credentialless(scope: Scope) -> bool:
        path = scope.get('path', '')
        if any(path.startswith(prefix) for prefix in _CREDENTIALLESS_PATH_PREFIXES):
            return True
        if scope['method'] == 'OPTIONS':
            # Preflight: the auth header hasn't been sent yet, so look at the
            # headers the browser is asking permission to send. Parse the
            # comma-separated list into a set so we match whole header names
            # only — otherwise something like ``x-my-authorization-token``
            # would substring-match ``authorization``.
            for name, value in scope['headers']:
                if name == b'access-control-request-headers':
                    requested_headers = {
                        h.strip() for h in value.decode('latin-1').lower().split(',')
                    }
                    return bool(
                        requested_headers
                        & {'authorization', 'x-session-api-key', 'x-access-token'}
                    )
            return False
        for name, value in scope['headers']:
            if name == b'authorization' and value[:7].lower() == b'bearer ':
                return True
            if name in (b'x-session-api-key', b'x-access-token'):
                return True
        return False


class PostHogSessionMiddleware:
    """Extract the PostHog session ID from the incoming request header.

    Stores the value on ``request.state.posthog_session_id`` so that
    subsequent event-capture call sites can link server-side events to the
    corresponding frontend session-replay recording.

    When the ``X-POSTHOG-SESSION-ID`` header is absent the attribute is set
    to ``None`` — never raises, never blocks.
    """

    async def __call__(self, request: Request, call_next: Callable):
        request.state.posthog_session_id = request.headers.get('X-POSTHOG-SESSION-ID')
        return await call_next(request)
