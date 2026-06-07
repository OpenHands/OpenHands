"""FastAPI routes for Databricks U2M OAuth (PKCE).

Env vars (read lazily per request — P0-5):
  DATABRICKS_HOST           — workspace URL (https://…)
  DATABRICKS_U2M_CLIENT_ID — OAuth app client id (NOT the M2M service principal id)
  DATABRICKS_REDIRECT_URI  — full callback URL registered with the OAuth app

Token storage design
--------------------
Starlette's ``SessionMiddleware`` **signs but does not encrypt** the session
cookie, so it must never carry secret material. Only a short opaque
``databricks_u2m_session_id`` is stored in the cookie. All secret material —
the OAuth access/refresh tokens and any confidential-app ``client_secret`` —
lives server-side in :data:`u2m_session_store`, keyed by that session id.

The server-side store is in-memory and single-process (TTL + LRU bounded); see
:mod:`openhands.app_server.auth.databricks_token_store` for the single-worker
caveat. Tokens are lost on server restart — users must re-authenticate. For
long-lived production deployments backed by M2M / PAT auth this is not a
concern; U2M is primarily used for local interactive sessions.

The local-dev OAuth "bridge server" and the root ``/callback`` alias are gated
behind ``RUNTIME=local`` so production exposes only the single stable
``/auth/databricks/callback`` redirect URI.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from openhands.app_server.auth.databricks_oauth import (
    async_exchange_code_for_tokens,
    build_authorize_url,
    generate_pkce,
)
from openhands.app_server.auth.databricks_token_store import u2m_session_store
from openhands.app_server.config import depends_user_context
from openhands.app_server.user.user_context import UserContext

_logger = logging.getLogger(__name__)

databricks_router = APIRouter(prefix='/auth/databricks', tags=['databricks-auth'])

_user_dependency = depends_user_context()

# The cookie carries ONLY this opaque session id. All secret material (tokens,
# confidential-app client secret) lives server-side in u2m_session_store keyed
# by this id — never in the signed-but-unencrypted session cookie.
_SESSION_ID_KEY = 'databricks_u2m_session_id'
# Record keys inside the server-side store.
_STORE_TOKENS_KEY = 'tokens'
_STORE_CLIENT_SECRET_KEY = 'oauth_client_secret'


def _is_local_runtime() -> bool:
    """True when running in local dev (``RUNTIME=local``).

    Local-dev-only conveniences (the OAuth port bridge server and the root
    ``/callback`` alias) are gated behind this so production never exposes them.
    Mirrors the ``OPENHANDS_SESSION_SECRET`` guard in ``app.py``.
    """
    return os.environ.get('RUNTIME', '').lower() == 'local'


# Bridge servers: port -> asyncio.Server
# When the user's registered redirect URI is on a port other than the main
# server port (e.g. the CLI-default http://localhost:8080/callback), we spin
# up a lightweight TCP server on that port — exactly like the CLI's
# databricks_pkce.py does.  The bridge server just 302-redirects the browser
# back to the main server's /auth/databricks/callback so the full token
# exchange happens there with the correct session cookie.
_BRIDGE_SERVERS: dict[int, asyncio.AbstractServer] = {}


async def _start_bridge_server(port: int, main_callback_url: str) -> None:
    """Start a TCP bridge server that forwards OAuth callbacks to the main server.

    Listens on ``http://127.0.0.1:<port>`` for any request (regardless of
    path) and immediately returns a ``302`` to ``main_callback_url`` with the
    original query string preserved.  This mirrors the CLI's
    ``databricks_pkce.run_browser_pkce_flow`` approach of starting a local
    server on the configured callback port.

    The bridge is kept alive for the lifetime of the backend process; it
    handles concurrent sign-ins and requires no cleanup per request.

    Args:
        port: Local port to listen on (extracted from the user's redirect URI).
        main_callback_url: Full URL to redirect to, e.g.
            ``http://localhost:3002/auth/databricks/callback``.

    Raises:
        OSError: If the port is already bound by another process.
    """
    if port in _BRIDGE_SERVERS:
        return  # already running for this port

    async def _handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            raw = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            first_line = raw.decode(errors='replace').split('\r\n')[0]
            # e.g. "GET /callback?code=X&state=Y HTTP/1.1"
            req_path = first_line.split(' ')[1] if ' ' in first_line else '/'
            qs = urlparse(req_path).query
            target = f'{main_callback_url}{"?" + qs if qs else ""}'
            response = (
                b'HTTP/1.1 302 Found\r\n'
                + f'Location: {target}\r\n'.encode()
                + b'Content-Length: 0\r\nConnection: close\r\n\r\n'
            )
            writer.write(response)
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(_handle, '127.0.0.1', port)
    _BRIDGE_SERVERS[port] = server
    _logger.info('databricks_u2m_bridge_started', extra={'port': port})


def _ensure_session_id(request: Request) -> str:
    """Return the opaque session id from the cookie, minting one if absent."""
    sid = request.session.get(_SESSION_ID_KEY)
    if not sid:
        sid = secrets.token_urlsafe(32)
        request.session[_SESSION_ID_KEY] = sid
    return sid


def _get_u2m_tokens(request: Request) -> Optional[dict]:
    """Look up U2M tokens from the server-side store via the cookie session id."""
    try:
        sid = request.session.get(_SESSION_ID_KEY)
    except (AssertionError, AttributeError):
        return None
    record = u2m_session_store.get(sid)
    if not record:
        return None
    tokens = record.get(_STORE_TOKENS_KEY)
    if isinstance(tokens, dict) and tokens.get('access_token'):
        return tokens
    return None


def read_u2m_tokens(request: Request) -> Optional[dict]:
    """Public accessor for the U2M token payload (server-side store only).

    Used by ``LiveStatusAppConversationService`` to pick up the browser-login
    tokens when starting a conversation. Reads exclusively from the server-side
    store — the signed cookie never carries tokens.
    """
    return _get_u2m_tokens(request)


def _store_u2m_tokens(request: Request, token_payload: dict) -> None:
    """Persist tokens server-side, keyed by the opaque cookie session id.

    Secret material (access + refresh tokens) is never written to the
    signed-but-unencrypted session cookie — only the opaque session id is.
    """
    sid = _ensure_session_id(request)
    u2m_session_store.put(sid, {_STORE_TOKENS_KEY: token_payload})


def _clear_u2m_tokens(request: Request) -> None:
    """Remove the server-side record and drop the session id from the cookie."""
    sid = request.session.pop(_SESSION_ID_KEY, None)
    u2m_session_store.delete(sid)
    request.session.pop('databricks_authenticated_host', None)


# Session keys for non-secret OAuth app config set via /prepare. The
# confidential-app client_secret is NOT among these — it lives server-side in
# u2m_session_store under _STORE_CLIENT_SECRET_KEY.
_SESSION_CLIENT_ID_KEY = 'databricks_u2m_client_id'
_SESSION_REDIRECT_URI_KEY = 'databricks_u2m_redirect_uri'


class _PrepareRequest(BaseModel):
    client_id: str
    client_secret: Optional[str] = None
    host: Optional[str] = None
    redirect_uri: Optional[str] = None
    # Browser-facing origin (e.g. "http://localhost:3002") sent by the frontend
    # so the bridge server knows where to redirect after capturing the callback.
    # Required when redirect_uri is on a different port than the main server.
    origin: Optional[str] = None


def _is_masked_secret(value: Optional[str]) -> bool:
    """Return True if *value* looks like a Pydantic SecretStr mask (e.g. '**********').

    The GET /settings response serialises SecretStr fields as a string of
    asterisks when ``expose_secrets`` is False.  We must never use such a
    placeholder as a real credential.
    """
    if not value:
        return False
    return all(c == '*' for c in value)


async def _resolve_stored_u2m_secret(user_context: UserContext) -> Optional[str]:
    """Read the stored databricks_u2m_client_secret from the user's settings.

    Secrets are never echoed back to the frontend (only _set flags are returned),
    so the backend must look them up directly from storage rather than relying on
    the client to re-supply them.
    """
    try:
        user = await user_context.get_user_info()
        raw = getattr(user, 'databricks_u2m_client_secret', None)
        if raw is None:
            return None
        # May be a SecretStr (Pydantic secret type) or a plain string.
        try:
            return raw.get_secret_value() or None
        except AttributeError:
            return str(raw) or None
    except Exception:
        return None


def _get_u2m_config(request: Request) -> tuple[str, str, str, Optional[str]]:
    """Resolve OAuth app config: (host, client_id, redirect_uri, client_secret).

    Priority for each field:
      1. Session (set by POST /prepare before the browser redirect)
      2. Environment variable fallback

    This allows users to supply their own custom OAuth app credentials through
    the UI without requiring server-side env vars.
    """
    try:
        session_client_id = request.session.get(_SESSION_CLIENT_ID_KEY)
        session_host = request.session.get('databricks_u2m_host')
        session_redirect_uri = request.session.get(_SESSION_REDIRECT_URI_KEY)
        sid = request.session.get(_SESSION_ID_KEY)
    except (AssertionError, AttributeError):
        session_client_id = session_host = None
        session_redirect_uri = sid = None
    # The confidential-app client secret lives server-side (never in the cookie).
    record = u2m_session_store.get(sid)
    session_client_secret = record.get(_STORE_CLIENT_SECRET_KEY) if record else None

    host = (session_host or os.environ.get('DATABRICKS_HOST', '')).strip().rstrip('/')
    client_id = (
        session_client_id or os.environ.get('DATABRICKS_U2M_CLIENT_ID', '')
    ).strip()
    client_secret = (
        session_client_secret
        or os.environ.get('DATABRICKS_U2M_CLIENT_SECRET', '')
        or None
    )

    # Priority: user-supplied redirect_uri (stored in session via /prepare)
    # > DATABRICKS_REDIRECT_URI env var > auto-constructed from PORT.
    # The user-supplied value must exactly match what was registered in the
    # Databricks OAuth app — this is the most common source of the
    # "redirect_uri not registered" error.
    redirect_uri = (
        session_redirect_uri or os.environ.get('DATABRICKS_REDIRECT_URI', '').strip()
    )
    if not redirect_uri:
        port = os.environ.get('PORT', '3000')
        redirect_uri = f'http://localhost:{port}/auth/databricks/callback'

    if not host or not client_id:
        raise HTTPException(
            status_code=501,
            detail=(
                'Databricks U2M OAuth is not configured. '
                'Call POST /auth/databricks/prepare with {client_id, host} first, '
                'or set DATABRICKS_HOST and DATABRICKS_U2M_CLIENT_ID env vars. '
                'Create an OAuth app at: https://accounts.cloud.databricks.com/settings/app-connections'
            ),
        )
    if not host.startswith('https://'):
        raise HTTPException(
            status_code=400,
            detail='Databricks host must use https://',
        )
    return host, client_id, redirect_uri, client_secret


@databricks_router.post('/prepare')
async def prepare_u2m(
    body: _PrepareRequest,
    request: Request,
    user_context: UserContext = _user_dependency,
) -> JSONResponse:
    """Store OAuth app credentials in session before the browser redirect.

    The frontend calls this endpoint with the user-supplied ``client_id``
    (and optional ``client_secret`` for confidential apps) so that
    ``/initiate`` and ``/callback`` can use them without requiring server-side
    env vars.  Credentials are stored only in the signed server-side session
    and are never echoed back.

    ``client_secret`` from the request body takes priority; if omitted (the
    frontend never echoes secrets back), we fall back to the value stored in
    the user's settings (SecretStr).  This way confidential-app token exchange
    always sends the correct secret even though the UI cannot supply it.

    ``host`` is optional; when provided it overrides ``DATABRICKS_HOST`` for
    this session's OAuth flow.
    """
    if not body.client_id.strip():
        raise HTTPException(status_code=400, detail='client_id must not be empty')
    request.session[_SESSION_CLIENT_ID_KEY] = body.client_id.strip()

    # The client secret is sensitive — keep it OUT of the signed cookie and put
    # it in the server-side store, keyed by the opaque session id.
    sid = _ensure_session_id(request)

    # Resolve client_secret: request body → stored settings → clear.
    # GET /settings masks secrets as "**********", so the frontend may
    # inadvertently send that placeholder.  Treat masked values as absent and
    # fall through to the stored-settings lookup.
    raw_secret = (
        body.client_secret if not _is_masked_secret(body.client_secret) else None
    )
    resolved_secret = raw_secret or await _resolve_stored_u2m_secret(user_context)
    if resolved_secret:
        u2m_session_store.put(sid, {_STORE_CLIENT_SECRET_KEY: resolved_secret})
    else:
        # Clear any previously stored secret without dropping the token record.
        record = u2m_session_store.get(sid) or {}
        if _STORE_CLIENT_SECRET_KEY in record:
            record.pop(_STORE_CLIENT_SECRET_KEY, None)
            u2m_session_store.delete(sid)
            if record:
                u2m_session_store.put(sid, record)

    if body.host:
        new_host = body.host.strip().rstrip('/')
        request.session['databricks_u2m_host'] = new_host
        # If the user is switching to a different workspace, clear any stale
        # token so they are prompted to sign in to the new workspace.
        authenticated_host = (
            request.session.get('databricks_authenticated_host', '').strip().rstrip('/')
        )
        env_host = os.environ.get('DATABRICKS_HOST', '').strip().rstrip('/')
        stale_host = authenticated_host or env_host
        if stale_host and stale_host != new_host:
            _clear_u2m_tokens(request)
            # Restore the new host (clear may have been a no-op but be explicit).
            request.session['databricks_u2m_host'] = new_host
    if body.redirect_uri:
        request.session[_SESSION_REDIRECT_URI_KEY] = body.redirect_uri.strip()
    elif _SESSION_REDIRECT_URI_KEY in request.session:
        del request.session[_SESSION_REDIRECT_URI_KEY]

    # If the configured redirect URI is on a port other than the main server
    # port, start a bridge server on that port — exactly as the CLI's
    # databricks_pkce.py does.  The bridge forwards the browser's callback
    # (code + state) to the main server's /auth/databricks/callback endpoint.
    #
    # This raw-socket listener is a local-dev convenience only; in production we
    # require a single stable registered redirect URI, so gate it on RUNTIME=local.
    if body.redirect_uri and _is_local_runtime():
        parsed_uri = urlparse(body.redirect_uri.strip())
        redirect_port = parsed_uri.port
        redirect_host = (parsed_uri.hostname or '').lower()
        main_port = int(os.environ.get('PORT', '3000'))

        # Determine the browser-facing origin for the bridge redirect.
        # Prefer the explicit origin from the frontend (window.location.origin)
        # because Vite rewrites the Host header when proxying, so request.base_url
        # may reflect the internal backend port rather than the browser-facing port.
        if body.origin:
            origin = body.origin.rstrip('/')
        else:
            bu = request.base_url
            origin = (
                f'{bu.scheme}://{bu.hostname}{":" + str(bu.port) if bu.port else ""}'
            )
        # Parse the browser-facing port so we can compare against redirect_port.
        browser_port = urlparse(origin).port or (
            443 if origin.startswith('https') else 80
        )

        if redirect_port and redirect_host in ('localhost', '127.0.0.1'):
            main_callback_url = f'{origin}/auth/databricks/callback'

            # Start a bridge only when:
            #   • redirect port differs from both the backend port and the
            #     browser-facing port (already reachable via existing routes), AND
            #   • we haven't already started one on this port.
            needs_bridge = (
                redirect_port != main_port
                and redirect_port != browser_port
                and redirect_port not in _BRIDGE_SERVERS
            )
            if needs_bridge:
                try:
                    await _start_bridge_server(redirect_port, main_callback_url)
                except OSError as exc:
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f'Cannot start OAuth callback bridge on port {redirect_port}: {exc}. '
                            'Either free that port or register a different redirect URI '
                            f'(e.g. http://localhost:{main_port}/callback).'
                        ),
                    )

    return JSONResponse({'redirect_url': '/auth/databricks/initiate'})


@databricks_router.get('/initiate')
async def initiate_u2m(request: Request) -> RedirectResponse:
    """Start PKCE browser flow; store state and verifier in session."""
    host, client_id, redirect_uri, _client_secret = _get_u2m_config(request)
    state = secrets.token_urlsafe(32)
    verifier, challenge = generate_pkce()
    request.session['databricks_oauth_state'] = state
    request.session['databricks_pkce_verifier'] = verifier
    url = build_authorize_url(host, client_id, redirect_uri, state, challenge)
    return RedirectResponse(url)


@databricks_router.get('/callback')
async def handle_u2m_callback(request: Request, code: str, state: str) -> HTMLResponse:
    """OAuth callback: validate state, exchange code, store tokens in session."""
    stored_state = request.session.pop('databricks_oauth_state', None)
    verifier = request.session.pop('databricks_pkce_verifier', None)
    if not stored_state or state != stored_state:
        raise HTTPException(
            status_code=400,
            detail='Invalid OAuth state — possible CSRF attempt.',
        )
    if not verifier:
        raise HTTPException(
            status_code=400,
            detail='Missing PKCE verifier — restart login at /auth/databricks/initiate.',
        )

    host, client_id, redirect_uri, client_secret = _get_u2m_config(request)
    try:
        # Async exchange so the token round-trip does not block the event loop.
        token_payload = await async_exchange_code_for_tokens(
            host,
            client_id,
            redirect_uri,
            code,
            verifier,
            client_secret=client_secret,
        )
    except Exception as exc:
        # Surface the Databricks error body so the user sees the actual reason
        # (e.g. redirect_uri mismatch, invalid_client, expired code) instead of
        # a generic 500.
        import httpx as _httpx  # local import — keep top-level imports clean

        detail = str(exc)
        if isinstance(exc, _httpx.HTTPStatusError):
            try:
                err_body = exc.response.json()
                detail = (
                    err_body.get('error_description') or err_body.get('error') or detail
                )
            except Exception:
                detail = exc.response.text[:400] or detail
        _logger.warning(
            'databricks_u2m_token_exchange_failed',
            extra={'host': host, 'detail': detail},
        )
        safe_detail = detail.replace('<', '&lt;').replace('>', '&gt;')
        return HTMLResponse(
            content=f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Databricks Sign-in Failed</title>
<style>
  body {{ font-family: system-ui, sans-serif; display: flex; align-items: center;
         justify-content: center; height: 100vh; margin: 0; background: #1a1a2e; color: #e0e0e0; }}
  .card {{ text-align: center; padding: 2rem; border-radius: 12px;
           background: #16213e; border: 1px solid #7f1d1d; max-width: 480px; }}
  .x {{ font-size: 3rem; color: #f87171; }}
  h1 {{ margin: .5rem 0; font-size: 1.25rem; color: #fca5a5; }}
  .reason {{ font-family: monospace; font-size: .8rem; color: #fca5a5;
             background: #450a0a; border-radius: 6px; padding: .75rem;
             margin-top: .75rem; text-align: left; word-break: break-word; }}
  p {{ color: #aaa; font-size: .875rem; margin: .75rem 0 0; }}
</style>
</head>
<body>
<div class="card">
  <div class="x">&#10007;</div>
  <h1>Sign-in Failed</h1>
  <div class="reason">{safe_detail}</div>
  <p>Close this tab, check your OAuth app settings, and try again.</p>
</div>
</body>
</html>""",
            status_code=400,
        )
    # Record which workspace the user authenticated against so /status can show
    # the correct host even when DATABRICKS_HOST env var points elsewhere.
    request.session['databricks_authenticated_host'] = host
    _store_u2m_tokens(request, token_payload)
    _logger.info('databricks_u2m_authenticated', extra={'host': host})

    # Return an HTML page rather than JSON so the popup/new-tab flow works
    # cleanly: the page tries to close itself, and shows a human-readable
    # success message as a fallback (browsers may block window.close() if the
    # tab wasn't opened by a script).
    safe_host = host.replace('<', '&lt;').replace('>', '&gt;')
    return HTMLResponse(
        content=f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Databricks Sign-in</title>
<style>
  body {{ font-family: system-ui, sans-serif; display: flex; align-items: center;
         justify-content: center; height: 100vh; margin: 0; background: #1a1a2e; color: #e0e0e0; }}
  .card {{ text-align: center; padding: 2rem; border-radius: 12px;
           background: #16213e; border: 1px solid #0f3460; max-width: 400px; }}
  .check {{ font-size: 3rem; color: #4caf50; }}
  h1 {{ margin: .5rem 0; font-size: 1.25rem; }}
  p  {{ color: #aaa; font-size: .875rem; margin: .5rem 0; }}
  .host {{ font-family: monospace; color: #81d4fa; font-size: .8rem; }}
</style>
</head>
<body>
<div class="card">
  <div class="check">&#10003;</div>
  <h1>Signed in to Databricks</h1>
  <p class="host">{safe_host}</p>
  <p>You can close this tab and return to OpenHands.</p>
</div>
<script>
  // Try to close the popup. This only works if the tab was opened by window.open().
  // If the user opened this URL directly it will silently fail — the message above
  // serves as the fallback.
  try {{ window.close(); }} catch (_) {{}}
</script>
</body>
</html>""",
        status_code=200,
    )


@databricks_router.post('/logout')
async def logout_u2m(request: Request) -> JSONResponse:
    """Clear U2M tokens from server-side store and PKCE state from session."""
    _clear_u2m_tokens(request)
    request.session.pop('databricks_oauth_state', None)
    request.session.pop('databricks_pkce_verifier', None)
    return JSONResponse({'status': 'logged_out'})


@databricks_router.get('/status')
async def u2m_status(request: Request) -> JSONResponse:
    """Return whether the current session has an active U2M login.

    Intended for the frontend "Sign in with Databricks" affordance — lets it
    render "Sign in" vs "Signed in as <host>" without leaking the token. The
    response also advertises whether U2M is configured at all, so the button
    can be hidden when the deployment didn't set up an OAuth app.
    """
    env_host = os.environ.get('DATABRICKS_HOST', '').strip().rstrip('/')
    client_id = os.environ.get('DATABRICKS_U2M_CLIENT_ID', '').strip()
    # Also configured if the session has credentials from a /prepare call
    try:
        session_client_id = request.session.get(_SESSION_CLIENT_ID_KEY, '')
        session_host = request.session.get('databricks_u2m_host', '')
        authenticated_host = (
            request.session.get('databricks_authenticated_host', '').strip().rstrip('/')
        )
    except (AssertionError, AttributeError):
        session_client_id = session_host = authenticated_host = ''
    # For "configured" check, any known host counts.
    effective_host = session_host or env_host
    effective_client_id = client_id or session_client_id
    configured = bool(effective_host and effective_client_id)

    tokens = _get_u2m_tokens(request)
    authenticated = bool(
        tokens and isinstance(tokens, dict) and tokens.get('access_token')
    )

    # When showing "Signed in to <host>", prefer the host that was actually
    # used during OAuth (stored in 'databricks_authenticated_host') over the
    # env var.  This ensures the UI reflects the workspace the user chose,
    # not a server-side default.
    if authenticated:
        display_host = authenticated_host or session_host or env_host
    else:
        display_host = None

    return JSONResponse(
        {
            'configured': configured,
            'authenticated': authenticated,
            # Host is public (it's the workspace URL) — safe to echo so the
            # UI can show "Signed in to adb-xxx.azuredatabricks.net".
            'host': display_host,
        }
    )
