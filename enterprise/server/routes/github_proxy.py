import hashlib
import json
import os
import zlib
from base64 import b64decode, b64encode
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from cryptography.fernet import Fernet
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from server.logger import logger

from openhands.server.shared import config
from openhands.utils.http_session import httpx_verify_option

GITHUB_PROXY_ENDPOINTS = bool(os.environ.get('GITHUB_PROXY_ENDPOINTS'))


def _is_safe_redirect_uri(redirect_uri: str, request_netloc: str) -> bool:
    """Validate that redirect_uri points to the same host as the proxy server.

    This prevents open-redirect attacks where an attacker crafts a redirect_uri
    pointing to an external domain (e.g., https://evil.com/steal) that would
    receive the OAuth authorization code after the callback.

    The redirect_uri must:
    - Use the https scheme (or http for localhost)
    - Have a netloc (hostname:port) that matches the request's own netloc
    """
    parsed = urlparse(redirect_uri)

    # Reject non-HTTP(S) schemes (e.g., javascript:, data:, etc.)
    if parsed.scheme not in ('https', 'http'):
        return False

    # Reject if no hostname
    if not parsed.hostname:
        return False

    # The redirect_uri must target the same host as the proxy server itself
    if parsed.netloc != request_netloc:
        return False

    return True


def add_github_proxy_routes(app: FastAPI):
    """
    Authentication endpoints for feature branches.

    # Requirements
    * This should never be enabled in prod!
    * Authentication on staging should be EXACTLY the same as prod - this only applies
    to feature branches!
    * We are only allowed 10 callback uris in github - so this does not scale.

    # How this works
    * It sits between keycloak and github.
    * For outgoing logins, it uses the OAuth state parameter to encode
    the subdomain of the actual redirect_uri ad well as the existing state
    * For incoming callbacks the state is decoded and the system redirects accordingly

    """
    # If the environment variable is not set, don't add these endpoints. (Typically only staging has this set.)
    if not GITHUB_PROXY_ENDPOINTS:
        return

    def _fernet():
        if not config.jwt_secret:
            raise ValueError('jwt_secret must be defined on config')
        jwt_secret = config.jwt_secret.get_secret_value()
        fernet_key = b64encode(hashlib.sha256(jwt_secret.encode()).digest())
        return Fernet(fernet_key)

    @app.get('/github-proxy/{subdomain}/login/oauth/authorize')
    def github_proxy_start(request: Request):
        parsed_url = urlparse(str(request.url))
        query_params = parse_qs(parsed_url.query)
        redirect_uri = query_params['redirect_uri'][0]

        # Validate redirect_uri before encrypting it into the state
        request_netloc = str(request.url.netloc)
        if not _is_safe_redirect_uri(redirect_uri, request_netloc):
            return JSONResponse(
                status_code=400,
                content={
                    'error': 'invalid_redirect_uri',
                    'message': 'redirect_uri must target the same host as the proxy server',
                },
            )

        state_payload = json.dumps(
            [query_params['state'][0], redirect_uri, request_netloc]
        )
        # Compress before encrypting to reduce URL length
        # This is critical for feature deployments where reCAPTCHA tokens in state
        # can cause "URL too long" errors from GitHub
        compressed_payload = zlib.compress(state_payload.encode())
        state = b64encode(_fernet().encrypt(compressed_payload)).decode()
        query_params['state'] = [state]
        query_params['redirect_uri'] = [
            f'https://{request.url.netloc}/github-proxy/callback'
        ]
        query_string = urlencode(query_params, doseq=True)
        return RedirectResponse(
            f'https://github.com/login/oauth/authorize?{query_string}'
        )

    @app.get('/github-proxy/callback')
    def github_proxy_callback(request: Request):
        # Decode state
        parsed_url = urlparse(str(request.url))
        query_params = parse_qs(parsed_url.query)
        state = query_params['state'][0]
        # Decrypt and decompress (reverse of github_proxy_start)
        decrypted_payload = _fernet().decrypt(b64decode(state.encode()))
        decrypted_state = zlib.decompress(decrypted_payload).decode()

        # Build query Params
        payload = json.loads(decrypted_state)
        # Support both old format [state, redirect_uri] and new [state, redirect_uri, netloc]
        if len(payload) == 3:
            state, redirect_uri, origin_netloc = payload
        else:
            state, redirect_uri = payload
            origin_netloc = str(request.url.netloc)

        # Validate redirect_uri before redirecting (defense in depth)
        if not _is_safe_redirect_uri(redirect_uri, origin_netloc):
            return JSONResponse(
                status_code=400,
                content={
                    'error': 'invalid_redirect_uri',
                    'message': 'redirect_uri must target the same host as the proxy server',
                },
            )

        query_params['state'] = [state]
        query_string = urlencode(query_params, doseq=True)

        # Redirect
        return RedirectResponse(f'{redirect_uri}?{query_string}')

    @app.post('/github-proxy/{subdomain}/login/oauth/access_token')
    async def access_token(request: Request, subdomain: str):
        body_bytes = await request.body()
        query_params = parse_qs(body_bytes.decode())
        body: bytes | str = body_bytes
        if query_params.get('redirect_uri'):
            query_params['redirect_uri'] = [
                f'https://{request.url.netloc}/github-proxy/callback'
            ]
            body = urlencode(query_params, doseq=True)
        url = 'https://github.com/login/oauth/access_token'
        async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
            response = await client.post(url, content=body)
            return Response(
                response.content,
                response.status_code,
                response.headers,
                media_type='application/x-www-form-urlencoded',
            )

    @app.post('/github-proxy/{subdomain}/{path:path}')
    async def post_proxy(request: Request, subdomain: str, path: str):
        logger.info(f'github_proxy_post:1:{path}')
        body = await request.body()
        url = f'https://github.com/{path}'
        async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
            response = await client.post(url, content=body, headers=request.headers)
            return Response(
                response.content,
                response.status_code,
                response.headers,
                media_type='application/x-www-form-urlencoded',
            )
