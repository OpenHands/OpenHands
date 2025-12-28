import hashlib
import json
import os
from base64 import b64decode, b64encode
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from cryptography.fernet import Fernet
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from server.logger import logger

from openhands.server.shared import config
from openhands.utils.http_session import httpx_verify_option

GITHUB_PROXY_ENDPOINTS = bool(os.environ.get('GITHUB_PROXY_ENDPOINTS'))


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
        state_payload = json.dumps(
            [query_params['state'][0], query_params['redirect_uri'][0]]
        )
        state = b64encode(_fernet().encrypt(state_payload.encode())).decode()
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
        decrypted_state = _fernet().decrypt(b64decode(state.encode())).decode()

        # Build query Params
        state, redirect_uri = json.loads(decrypted_state)
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
