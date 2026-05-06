import base64
import hashlib
import json
import logging
import os
import re
import time
from urllib.parse import parse_qs, quote, unquote, urlparse

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse
from nacl.bindings import crypto_aead_xchacha20poly1305_ietf_decrypt

from openhands.server.middleware import (
    _adapt_cookie_for_iframe,
    is_request_origin_https,
)

logger = logging.getLogger(__name__)

app = APIRouter(prefix='/api')

BETTER_AUTH_URL = os.environ.get('BETTER_AUTH_URL', '').rstrip('/')
BETTER_AUTH_SECRET = os.environ.get('BETTER_AUTH_SECRET', '')
_SESSION_COOKIES = ('__Secure-b1.session_token', 'b1.session_token')

# Regexes to strip Domain and Path attributes from proxied Set-Cookie headers
_DOMAIN_ATTR_RE = re.compile(r';\s*domain=[^;]*', re.IGNORECASE)
_PATH_ATTR_RE = re.compile(r';\s*path=[^;]*', re.IGNORECASE)


def _auth_url(path: str) -> str:
    return f'{BETTER_AUTH_URL}{path}'


def _get_session_token(request: Request) -> tuple[str | None, str | None]:
    """Return (cookie_name, token_value) for the first matching session cookie."""
    for name in _SESSION_COOKIES:
        token = request.cookies.get(name)
        if token:
            return name, token
    return None, None


def _request_origin(request: Request) -> str:
    """Extract the client origin from a request (e.g. 'https://host:port')."""
    origin = request.headers.get('origin', '')
    if origin:
        return origin
    referer = request.headers.get('referer', '')
    if referer:
        parsed = urlparse(referer)
        return f'{parsed.scheme}://{parsed.netloc}'
    return ''


def _build_proxy_headers(origin: str) -> dict[str, str]:
    """Build headers that tell Better Auth we're a same-origin proxy.

    Newer Better Auth resolves its per-request baseURL from x-forwarded-host
    / x-forwarded-proto, so include those too.
    """
    if not origin:
        return {}
    parsed = urlparse(origin)
    return {
        'x-better-auth-proxy-mode': 'enabled',
        'x-better-auth-url': origin,
        'x-forwarded-host': parsed.netloc,
        'x-forwarded-proto': parsed.scheme or 'https',
    }


def _proxy_set_cookie_headers(
    proxy_response: httpx.Response, response: JSONResponse
) -> None:
    """Forward Set-Cookie headers from the remote Better Auth response.

    Strips the Domain attribute so cookies are set for the current
    request origin instead of the remote auth server's domain.
    Replaces the Path attribute with Path=/ so cookies are sent
    for all paths (not just /api/auth/*).
    """
    for value in proxy_response.headers.get_list('set-cookie'):
        # Remove Domain=... so the cookie defaults to the current host
        cleaned = _DOMAIN_ATTR_RE.sub('', value)
        # Remove any Path=... and replace with Path=/
        cleaned = _PATH_ATTR_RE.sub('', cleaned)
        cleaned += '; Path=/'
        logger.debug('Proxied Set-Cookie: %s', cleaned.split('=', 1)[0])
        response.headers.append('set-cookie', cleaned)


@app.post('/auth/sign-in')
async def sign_in(request: Request):
    """Proxy email/password sign-in to Better Auth server."""
    if not BETTER_AUTH_URL:
        return JSONResponse(
            status_code=501,
            content={'error': 'Better Auth is not configured'},
        )

    body = await request.json()

    # Forward proxy headers so Better Auth knows the real client origin
    origin = _request_origin(request)
    proxy_headers = _build_proxy_headers(origin)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _auth_url('/api/auth/sign-in/email'),
            json=body,
            headers=proxy_headers,
        )

    try:
        content = resp.json()
    except Exception:
        content = {'error': resp.text or 'Sign-in failed'}
    response = JSONResponse(status_code=resp.status_code, content=content)
    _proxy_set_cookie_headers(resp, response)
    return response


@app.post('/auth/sign-up')
async def sign_up(request: Request):
    """Proxy email/password sign-up to Better Auth server."""
    if not BETTER_AUTH_URL:
        return JSONResponse(
            status_code=501,
            content={'error': 'Better Auth is not configured'},
        )

    body = await request.json()

    # Forward proxy headers so Better Auth knows the real client origin
    origin = _request_origin(request)
    proxy_headers = _build_proxy_headers(origin)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _auth_url('/api/auth/sign-up/email'),
            json=body,
            headers=proxy_headers,
        )

    try:
        content = resp.json()
    except Exception:
        content = {'error': resp.text or 'Sign-up failed'}
    response = JSONResponse(status_code=resp.status_code, content=content)
    _proxy_set_cookie_headers(resp, response)
    return response


@app.post('/auth/sign-in/social')
async def sign_in_social(request: Request):
    """Proxy OAuth sign-in to Better Auth server. Returns redirect URL."""
    if not BETTER_AUTH_URL:
        return JSONResponse(
            status_code=501,
            content={'error': 'Better Auth is not configured'},
        )

    body = await request.json()
    provider = body.get('provider', '')
    callback_url = body.get('callbackURL', '/')

    # Derive the proxy callback URL from the callbackURL origin so we
    # can use it as the callbackURL for Better Auth. This ensures that
    # after the OAuth flow, Better Auth redirects to our proxy callback
    # endpoint — regardless of whether Better Auth has the redirect proxy
    # feature configured.
    parsed = urlparse(callback_url)
    origin = f'{parsed.scheme}://{parsed.netloc}' if parsed.scheme else ''

    # Build a callbackURL that points to our proxy callback endpoint.
    # The actual final destination is encoded as a query parameter.
    if origin:
        ba_callback_url = (
            f'{origin}/api/auth/oauth-proxy-callback'
            f'?_destination={quote(callback_url, safe="")}'
        )
    else:
        ba_callback_url = callback_url

    if not BETTER_AUTH_SECRET:
        logger.warning(
            'Social sign-in: BETTER_AUTH_SECRET is not set! '
            'OAuth proxy callback will not be able to decrypt cookies.'
        )

    # Build proxy headers so Better Auth uses the OAuth proxy flow:
    # encrypts cookies and redirects via oauth-proxy-callback on our domain.
    proxy_headers = _build_proxy_headers(origin)

    async with httpx.AsyncClient(follow_redirects=False) as client:
        resp = await client.post(
            _auth_url('/api/auth/sign-in/social'),
            json={'provider': provider, 'callbackURL': ba_callback_url},
            headers=proxy_headers,
        )

    try:
        content = resp.json()
    except Exception:
        content = {'error': resp.text or 'OAuth request failed'}

    # Better Auth returns 200 with {"url": "...", "redirect": true}
    url = content.get('url', '')
    if resp.status_code == 200 and url:
        response = JSONResponse(
            status_code=200,
            content={'url': url},
        )
        _proxy_set_cookie_headers(resp, response)
        return response

    response = JSONResponse(
        status_code=resp.status_code if resp.status_code != 200 else 400,
        content=content,
    )
    _proxy_set_cookie_headers(resp, response)
    return response


@app.get('/auth/providers')
async def get_providers():
    """Return available OAuth providers."""
    if not BETTER_AUTH_URL:
        return JSONResponse(status_code=200, content={'providers': []})

    return JSONResponse(
        status_code=200,
        content={'providers': ['microsoft', 'github', 'google']},
    )


@app.post('/authenticate')
async def authenticate(request: Request):
    """Validate the current session by forwarding the cookie to Better Auth."""
    if not BETTER_AUTH_URL:
        return JSONResponse(content={'status': 'ok'})

    cookie_name, session_token = _get_session_token(request)
    if not session_token or not cookie_name:
        return JSONResponse(
            status_code=401,
            content={'error': 'Not authenticated'},
        )

    origin = _request_origin(request)
    proxy_headers = _build_proxy_headers(origin)

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            _auth_url('/api/auth/get-session'),
            cookies={cookie_name: session_token},
            headers=proxy_headers,
        )

    if resp.status_code == 200:
        data = resp.json()
        if data.get('user'):
            return JSONResponse(content={'status': 'ok', 'user': data['user']})

    return JSONResponse(
        status_code=401,
        content={'error': 'Invalid or expired session'},
    )


@app.post('/logout')
async def logout(request: Request):
    """Proxy sign-out to Better Auth server and clear the session cookie."""
    cookie_name, session_token = _get_session_token(request)

    if BETTER_AUTH_URL and session_token and cookie_name:
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    _auth_url('/api/auth/sign-out'),
                    cookies={cookie_name: session_token},
                )
        except Exception:
            pass

    response = JSONResponse(content={'status': 'ok'})
    is_https = is_request_origin_https(request)

    # Partitioned cookies (CHIPS) live in a separate jar keyed by top-level
    # site. Deleting them requires Set-Cookie with the Partitioned attribute
    # — Starlette's delete_cookie can't express this, so craft the header
    # manually. Also issue a non-Partitioned deletion so we cover legacy
    # cookies that were set before the handoff flow added Partitioned.
    if is_https:
        response.headers.append(
            'set-cookie',
            '__Secure-b1.session_token=; Path=/; HttpOnly; Secure; '
            'SameSite=None; Partitioned; Max-Age=0',
        )
        response.headers.append(
            'set-cookie',
            '__Secure-b1.session_token=; Path=/; HttpOnly; Secure; '
            'SameSite=None; Max-Age=0',
        )

    response.headers.append(
        'set-cookie',
        'b1.session_token=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0',
    )
    return response


def _symmetric_decrypt(secret: str, data_encoded: str) -> str:
    """Decrypt data encrypted with Better Auth's symmetricEncrypt (XChaCha20-Poly1305).

    Format: nonce_24_bytes || ciphertext || tag_16_bytes
    The payload may be encoded as hex or base64url depending on the Better Auth version.
    Key: SHA-256 hash of the secret string.
    """
    key = hashlib.sha256(secret.encode()).digest()

    # Detect encoding: if the string is all hex chars, decode as hex; otherwise base64url
    is_hex = all(c in '0123456789abcdefABCDEF' for c in data_encoded)
    if is_hex and len(data_encoded) >= 80:  # min 40 bytes hex = 24 nonce + 16 tag
        raw = bytes.fromhex(data_encoded)
    else:
        padded = (
            data_encoded + '=' * (4 - len(data_encoded) % 4)
            if len(data_encoded) % 4
            else data_encoded
        )
        raw = base64.urlsafe_b64decode(padded)

    nonce = raw[:24]
    ciphertext_with_tag = raw[24:]
    plaintext = crypto_aead_xchacha20poly1305_ietf_decrypt(
        ciphertext_with_tag, None, nonce, key
    )
    return plaintext.decode('utf-8')


def _parse_set_cookie(cookie_str: str, is_secure: bool) -> dict | None:
    """Parse a Set-Cookie string into components."""
    parts = [p.strip() for p in cookie_str.split(';')]
    if not parts:
        return None
    eq_idx = parts[0].find('=')
    if eq_idx == -1:
        return None
    name = parts[0][:eq_idx].strip()
    value = parts[0][eq_idx + 1 :].strip()
    attrs: dict = {'name': name, 'value': value}
    for attr in parts[1:]:
        kv = attr.split('=', 1)
        k = kv[0].strip().lower()
        v = kv[1].strip() if len(kv) > 1 else None
        if k == 'path' and v:
            attrs['path'] = v
        elif k == 'expires' and v:
            attrs['expires'] = v
        elif k == 'max-age' and v:
            attrs['max_age'] = int(v)
        elif k == 'httponly':
            attrs['httponly'] = True
        elif k == 'samesite' and v:
            attrs['samesite'] = v.lower()
        elif k == 'secure':
            attrs['secure'] = True
    # Enforce Secure for HTTPS origins
    if is_secure:
        attrs['secure'] = True
    return attrs


@app.get('/auth/oauth-proxy-callback')
async def oauth_proxy_callback(request: Request):
    """Handle the OAuth redirect from Better Auth with encrypted cookies.

    This endpoint is called in two scenarios:

    1. **Direct redirect from Better Auth** (no redirect-proxy feature):
       Better Auth redirects here because we set the callbackURL to this
       endpoint. The `_destination` query param holds the actual frontend URL.
       No `cookies` param is present — we fall back to checking the request
       for session cookies that Better Auth may have set on a prior redirect.

    2. **Redirect-proxy redirect from Better Auth**:
       Better Auth redirects here with encrypted session cookies in the
       `cookies` query param, and the full callbackURL in `callbackURL`.
       We decrypt and set cookies, then redirect to the final destination.
    """
    # Newer Better Auth (>=1.x) stores cookies server-side and sends a
    # short reference key as `p`. Older versions sent the actual encrypted
    # payload as `cookies`.
    proxy_ref = request.query_params.get('p', '')
    encrypted_cookies = request.query_params.get('cookies', '')

    # Determine the final destination URL.
    # When Better Auth's redirect proxy is active, it sends `callbackURL`
    # (which is our proxy endpoint + `_destination`). When the redirect
    # proxy is NOT active, Better Auth redirects directly to our endpoint
    # and we use `_destination` from the query string.
    ba_callback_url = request.query_params.get('callbackURL', '')
    destination = request.query_params.get('_destination', '')
    if destination:
        destination = unquote(destination)

    if (proxy_ref or encrypted_cookies) and ba_callback_url:
        # Redirect-proxy scenario: the `callbackURL` in the request is our
        # proxy endpoint URL (with our `_destination` param). Pull the real
        # destination out of it.
        parsed_ba = urlparse(ba_callback_url)
        ba_qs = parse_qs(parsed_ba.query)
        final_destination = ba_qs.get('_destination', ['/'])[0]
    elif destination:
        # Direct redirect from Better Auth (no proxy).
        final_destination = destination
    else:
        # Fallback
        final_destination = ba_callback_url or '/'

    # --- New Better Auth (>=1.x): server-side stored cookies, fetched by ref ---
    if proxy_ref and BETTER_AUTH_URL:
        # Forward the request to the auth server's own oauth-proxy-callback,
        # which returns a redirect with Set-Cookie headers carrying the
        # session cookies. Proxy those Set-Cookie headers back to our
        # response so the cookies are set on our origin.
        proxy_headers = _build_proxy_headers(_request_origin(request))
        async with httpx.AsyncClient(follow_redirects=False) as client:
            ba_resp = await client.get(
                _auth_url('/api/auth/oauth-proxy-callback'),
                params={'p': proxy_ref, 'callbackURL': ba_callback_url},
                headers=proxy_headers,
            )

        set_cookies = ba_resp.headers.get_list('set-cookie')

        if not set_cookies:
            logger.error(
                'OAuth proxy callback: auth server returned no Set-Cookie '
                '(status=%s, body_prefix=%r)',
                ba_resp.status_code,
                ba_resp.text[:200],
            )
            parsed_dest = urlparse(final_destination)
            login_url = (
                f'{parsed_dest.scheme}://{parsed_dest.netloc}/login'
                if parsed_dest.scheme
                else '/login'
            )
            return RedirectResponse(url=login_url, status_code=302)

        response = RedirectResponse(url=final_destination, status_code=302)
        _proxy_set_cookie_headers(ba_resp, response)
        return response

    # --- Handle encrypted cookies (redirect-proxy scenario) ---
    if encrypted_cookies:
        if not BETTER_AUTH_SECRET:
            logger.error('OAuth proxy callback: BETTER_AUTH_SECRET not configured')
            return JSONResponse(
                status_code=500,
                content={'error': 'Auth secret not configured'},
            )

        # Decrypt the cookies payload
        try:
            decrypted = _symmetric_decrypt(BETTER_AUTH_SECRET, encrypted_cookies)
        except Exception:
            logger.exception('OAuth proxy callback: failed to decrypt cookies')
            return JSONResponse(
                status_code=400,
                content={'error': 'Failed to decrypt cookies'},
            )

        try:
            payload = json.loads(decrypted)
        except Exception:
            logger.error('OAuth proxy callback: invalid payload format')
            return JSONResponse(
                status_code=400,
                content={'error': 'Invalid payload format'},
            )

        # Check timestamp (max 60 seconds old)
        timestamp = payload.get('timestamp', 0)
        age = (time.time() * 1000 - timestamp) / 1000
        if age > 60 or age < -10:
            logger.error('OAuth proxy callback: payload expired (age=%.1fs)', age)
            return JSONResponse(
                status_code=400,
                content={'error': 'Payload expired'},
            )

        # Parse and set cookies from the decrypted payload
        cookies_str = payload.get('cookies', '')
        cookie_strings = re.split(r',(?=\s*[\w.-]+=)', cookies_str)
        is_secure = final_destination.startswith('https://')

        response = RedirectResponse(url=final_destination, status_code=302)

        for cookie_str in cookie_strings:
            cookie_str = cookie_str.strip()
            if not cookie_str:
                continue
            cookie = _parse_set_cookie(cookie_str, is_secure)
            if not cookie:
                continue

            name = cookie['name']
            value = cookie['value']
            response.set_cookie(
                key=name,
                value=value,
                path=cookie.get('path', '/'),
                max_age=cookie.get('max_age'),
                httponly=cookie.get('httponly', False),
                samesite=cookie.get('samesite', 'lax'),
                secure=cookie.get('secure', False),
            )

        return response

    # --- No encrypted cookies (direct redirect scenario) ---
    # Better Auth redirected here directly without the redirect-proxy.
    # Check if Better Auth set session cookies on the request itself
    # (they travel with the redirect if same domain, which won't happen
    # cross-origin, but check anyway).
    cookie_name, session_token = _get_session_token(request)
    if session_token and cookie_name:
        is_secure = final_destination.startswith('https://')
        response = RedirectResponse(url=final_destination, status_code=302)
        response.set_cookie(
            key=cookie_name,
            value=session_token,
            path='/',
            httponly=True,
            samesite='lax',
            secure=is_secure,
        )
        return response

    logger.error(
        'OAuth proxy callback: no encrypted cookies and no session cookie '
        'found. This likely means Better Auth redirect proxy is not '
        'configured. Ensure BETTER_AUTH_SECRET is set and matches the '
        'Better Auth server. params=%s',
        list(request.query_params.keys()),
    )
    # Redirect to the login page so the user can try again
    parsed_dest = urlparse(final_destination)
    login_url = (
        f'{parsed_dest.scheme}://{parsed_dest.netloc}/login'
        if parsed_dest.scheme
        else '/login'
    )
    return RedirectResponse(url=login_url, status_code=302)


@app.post('/auth/handoff/redeem')
async def handoff_redeem(request: Request):
    """Redeem a single-use handoff code for a session cookie at our origin.

    The embedding parent app appends ?handoff_code=<code> to the iframe URL.
    In production the request-time middleware redeems it before the SPA loads.
    In dev (Vite serves the SPA HTML) the middleware never sees the request,
    so the SPA reads the param on first render and POSTs it here.

    Cookies returned by the auth server are adapted for cross-site iframe use:
    HTTPS adds Partitioned (CHIPS); HTTP strips __Secure- name prefix, drops
    Secure, and rewrites SameSite=None to SameSite=Lax.

    On any non-200 from the auth server we pass through the status so the
    client can decide whether to fall through to the existing sign-in flow.
    """
    if not BETTER_AUTH_URL:
        return JSONResponse(
            status_code=501,
            content={'error': 'Better Auth is not configured'},
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={'error': 'Invalid body'})

    code = body.get('code')
    if not code or not isinstance(code, str):
        return JSONResponse(status_code=400, content={'error': 'Missing code'})

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                _auth_url('/api/auth/mcp/handoff/redeem'),
                json={'code': code},
                timeout=10.0,
            )
    except Exception as e:
        logger.warning('Handoff redeem network error: %s', e)
        return JSONResponse(
            status_code=502,
            content={'error': 'Auth service unavailable'},
        )

    if resp.status_code != 200:
        logger.info(
            'Handoff redeem rejected (%s): %s',
            resp.status_code,
            resp.text[:200],
        )
        try:
            content = resp.json()
        except Exception:
            content = {'error': resp.text or 'Redeem failed'}
        return JSONResponse(status_code=resp.status_code, content=content)

    is_https = is_request_origin_https(request)
    response = JSONResponse(content={'status': 'ok'})
    for raw_cookie in resp.headers.get_list('set-cookie'):
        adapted = _adapt_cookie_for_iframe(raw_cookie, is_https=is_https)
        response.headers.append('set-cookie', adapted)
    return response


@app.post('/login')
async def login_deprecated():
    """Old password login endpoint — removed."""
    return JSONResponse(
        status_code=410,
        content={
            'error': 'Password login has been removed. Use /api/auth/sign-in instead.'
        },
    )
