# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
import asyncio
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from urllib.parse import urlencode, urlparse

import httpx
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request as StarletteRequest
from starlette.responses import RedirectResponse, Response
from starlette.types import ASGIApp


class LocalhostCORSMiddleware(CORSMiddleware):
    """Custom CORS middleware that allows any request from localhost/127.0.0.1 domains,
    while using standard CORS rules for other origins.
    """

    def __init__(self, app: ASGIApp) -> None:
        allow_origins_str = os.getenv('PERMITTED_CORS_ORIGINS')
        if allow_origins_str:
            allow_origins = tuple(
                origin.strip() for origin in allow_origins_str.split(',')
            )
        else:
            allow_origins = ()
        super().__init__(
            app,
            allow_origins=allow_origins,
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*'],
        )

    def is_allowed_origin(self, origin: str) -> bool:
        if origin and not self.allow_origins and not self.allow_origin_regex:
            parsed = urlparse(origin)
            hostname = parsed.hostname or ''

            # Allow any localhost/127.0.0.1 origin regardless of port
            if hostname in ['localhost', '127.0.0.1']:
                return True

        # For missing origin or other origins, use the parent class's logic
        result: bool = super().is_allowed_origin(origin)
        return result


class CacheControlMiddleware(BaseHTTPMiddleware):
    """Middleware to disable caching for all routes by adding appropriate headers"""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        if request.url.path.startswith('/assets'):
            # The content of the assets directory has fingerprinted file names so we cache aggressively
            response.headers['Cache-Control'] = 'public, max-age=2592000, immutable'
        else:
            response.headers['Cache-Control'] = (
                'no-cache, no-store, must-revalidate, max-age=0'
            )
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        return response


class InMemoryRateLimiter:
    history: dict[str, list[datetime]]
    requests: int
    seconds: int
    sleep_seconds: int

    def __init__(self, requests: int = 2, seconds: int = 1, sleep_seconds: int = 1):
        self.requests = requests
        self.seconds = seconds
        self.sleep_seconds = sleep_seconds
        self.history = defaultdict(list)
        self.sleep_seconds = sleep_seconds

    def _clean_old_requests(self, key: str) -> None:
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.seconds)
        self.history[key] = [ts for ts in self.history[key] if ts > cutoff]

    async def __call__(self, request: Request) -> bool:
        key = request.client.host
        now = datetime.now()

        self._clean_old_requests(key)

        self.history[key].append(now)

        if len(self.history[key]) > self.requests * 2:
            return False
        elif len(self.history[key]) > self.requests:
            if self.sleep_seconds > 0:
                await asyncio.sleep(self.sleep_seconds)
                return True
            else:
                return False

        return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, rate_limiter: InMemoryRateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if not self.is_rate_limited_request(request):
            return await call_next(request)
        ok = await self.rate_limiter(request)
        if not ok:
            return JSONResponse(
                status_code=429,
                content={'message': 'Too many requests'},
                headers={'Retry-After': '1'},
            )
        return await call_next(request)

    def is_rate_limited_request(self, request: StarletteRequest) -> bool:
        if request.url.path.startswith('/assets'):
            return False
        # Put Other non rate limited checks here
        return True


class BetterAuthMiddleware(BaseHTTPMiddleware):
    """Middleware that validates sessions via a remote Better Auth server."""

    SKIP_PATHS = {
        '/api/auth/sign-in',
        '/api/auth/sign-up',
        '/api/auth/sign-in/social',
        '/api/auth/oauth-proxy-callback',
        '/api/auth/handoff/redeem',
        '/api/auth/providers',
        '/api/authenticate',
        '/api/login',
        '/api/logout',
        '/api/options/config',
        '/api/health',
        '/api/v1/web-client/config',
    }

    SESSION_COOKIES = ('__Secure-b1.session_token', 'b1.session_token')
    CACHE_TTL_SECONDS = 60

    def __init__(self, app: ASGIApp, better_auth_url: str):
        super().__init__(app)
        self.better_auth_url = better_auth_url.rstrip('/')
        # In-memory session cache: token -> (expiry_time, user_data)
        self._session_cache: dict[str, tuple[float, dict]] = {}

    def _get_cached_user(self, token: str) -> dict | None:
        entry = self._session_cache.get(token)
        if entry and entry[0] > time.monotonic():
            return entry[1]
        if entry:
            del self._session_cache[token]
        return None

    def _cache_user(self, token: str, user_data: dict) -> None:
        self._session_cache[token] = (
            time.monotonic() + self.CACHE_TTL_SECONDS,
            user_data,
        )

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        logger = logging.getLogger(__name__)
        path = request.url.path

        # Only gate /api/ paths
        if not path.startswith('/api/'):
            return await call_next(request)

        # Skip public endpoints
        if path in self.SKIP_PATHS:
            return await call_next(request)

        # Skip webhook callbacks from agent containers
        # (authenticated via X-Session-API-Key header in the endpoint)
        if path.startswith('/api/v1/webhooks/'):
            return await call_next(request)

        # Extract session cookie (check both __Secure- and plain variants)
        cookie_name = None
        session_token = None
        for name in self.SESSION_COOKIES:
            session_token = request.cookies.get(name)
            if session_token:
                cookie_name = name
                break

        if not session_token or not cookie_name:
            logger.info(
                'BetterAuth: no session cookie for %s (cookies: %s)',
                path,
                list(request.cookies.keys()),
            )
            return JSONResponse(
                status_code=401,
                content={'error': 'Not authenticated'},
            )

        # Check cache first
        cached_user = self._get_cached_user(session_token)
        if cached_user:
            request.state.better_auth_user = cached_user
            return await call_next(request)

        # Validate session with remote Better Auth server.
        # Forward host/proto so Better Auth resolves baseURL to our origin.
        forwarded_host = request.headers.get('x-forwarded-host') or request.headers.get(
            'host', ''
        )
        forwarded_proto = request.headers.get('x-forwarded-proto') or request.url.scheme
        forwarded_headers = (
            {
                'x-forwarded-host': forwarded_host,
                'x-forwarded-proto': forwarded_proto,
            }
            if forwarded_host
            else {}
        )
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f'{self.better_auth_url}/api/auth/get-session',
                    cookies={cookie_name: session_token},
                    headers=forwarded_headers,
                    timeout=10.0,
                )
        except Exception:
            return JSONResponse(
                status_code=401,
                content={'error': 'Auth service unavailable'},
            )

        if resp.status_code == 200:
            data = resp.json()
            user = data.get('user')
            if user:
                self._cache_user(session_token, user)
                request.state.better_auth_user = user
                return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={'error': 'Invalid or expired session'},
        )


def is_request_origin_https(request: Request) -> bool:
    """Determine whether the public origin facing the browser is HTTPS.

    APP_URL takes precedence (the configured public origin). X-Forwarded-Proto
    is a fallback for proxies that don't expose APP_URL. request.url.scheme
    alone is unreliable behind a TLS-terminating reverse proxy.
    """
    app_url = os.environ.get('APP_URL', '')
    if app_url:
        return app_url.lower().startswith('https://')
    proto = request.headers.get('x-forwarded-proto') or request.url.scheme
    return proto.lower() == 'https'


def _adapt_cookie_for_iframe(cookie_header: str, *, is_https: bool) -> str:
    """Rewrite a Set-Cookie header from the auth server for our origin's iframe context.

    HTTPS (production): the iframe is cross-site relative to its parent. Add the
    Partitioned attribute (CHIPS) so Chrome's third-party-cookie blocking does
    not silently drop the cookie. Keep Secure and SameSite=None.

    HTTP (local dev): strip the __Secure- name prefix, drop Secure, and rewrite
    SameSite=None to SameSite=Lax. Browsers refuse Secure on HTTP and refuse
    SameSite=None without Secure.
    """
    parts = [p.strip() for p in cookie_header.split(';') if p.strip()]
    if not parts:
        return cookie_header

    name_value = parts[0]
    attrs = parts[1:]

    if is_https:
        if not any(a.lower() == 'partitioned' for a in attrs):
            attrs.append('Partitioned')
        return '; '.join([name_value, *attrs])

    # HTTP: rewrite for local dev.
    if name_value.startswith('__Secure-'):
        name_value = name_value[len('__Secure-') :]

    rewritten: list[str] = []
    for attr in attrs:
        lower = attr.lower()
        if lower == 'secure':
            continue
        if lower == 'partitioned':
            # Partitioned is only meaningful with Secure, which we just dropped.
            continue
        if lower.startswith('samesite='):
            value = attr.split('=', 1)[1].strip().lower()
            if value == 'none':
                rewritten.append('SameSite=Lax')
                continue
        rewritten.append(attr)

    return '; '.join([name_value, *rewritten])


class HandoffRedemptionMiddleware(BaseHTTPMiddleware):
    """Redeems a single-use auth handoff code for a session cookie at our origin.

    The embedding parent app (B1/Vanguard) loads our SPA in an iframe and appends
    ?handoff_code=<code> to the URL. We exchange that code for session cookies at
    the Better Auth server (server-to-server), set them on the browser response,
    and 302 to the same URL with the param stripped so a refresh can't replay
    the (now-consumed) code and the code doesn't sit in the address bar.

    Any failure path — already authenticated, redeem non-200, network error —
    falls through to the existing sign-in flow by stripping the param and
    redirecting to the clean URL.
    """

    SESSION_COOKIES = ('__Secure-b1.session_token', 'b1.session_token')
    HANDOFF_PARAM = 'handoff_code'

    def __init__(self, app: ASGIApp, better_auth_url: str):
        super().__init__(app)
        self.better_auth_url = better_auth_url.rstrip('/')

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.method != 'GET':
            return await call_next(request)
        if self.HANDOFF_PARAM not in request.query_params:
            return await call_next(request)

        path = request.url.path
        # Don't try to redeem on API or static-asset requests; the handoff is
        # only meaningful for SPA HTML loads (e.g. /conversations/:id).
        if path.startswith('/api/') or path.startswith('/assets/'):
            return await call_next(request)

        clean_url = self._strip_handoff_param(request)
        logger = logging.getLogger(__name__)

        # Already signed in — no need to redeem, just clean the URL.
        if any(request.cookies.get(name) for name in self.SESSION_COOKIES):
            return RedirectResponse(url=clean_url, status_code=302)

        code = request.query_params[self.HANDOFF_PARAM]
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f'{self.better_auth_url}/api/auth/mcp/handoff/redeem',
                    json={'code': code},
                    timeout=10.0,
                )
        except Exception as e:
            logger.warning('Handoff redemption network error: %s', e)
            return RedirectResponse(url=clean_url, status_code=302)

        if resp.status_code != 200:
            logger.info(
                'Handoff redemption rejected (%s): %s',
                resp.status_code,
                resp.text[:200],
            )
            return RedirectResponse(url=clean_url, status_code=302)

        is_https = is_request_origin_https(request)
        redirect = RedirectResponse(url=clean_url, status_code=302)
        for raw_cookie in resp.headers.get_list('set-cookie'):
            adapted = _adapt_cookie_for_iframe(raw_cookie, is_https=is_https)
            redirect.raw_headers.append((b'set-cookie', adapted.encode('latin-1')))
        return redirect

    @classmethod
    def _strip_handoff_param(cls, request: Request) -> str:
        remaining = [
            (k, v)
            for k, v in request.query_params.multi_items()
            if k != cls.HANDOFF_PARAM
        ]
        query = urlencode(remaining)
        return request.url.path + (f'?{query}' if query else '')
