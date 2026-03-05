import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from server.auth.constants import BITBUCKET_DATA_CENTER_HOST

from openhands.utils.http_session import httpx_verify_option

router = APIRouter(prefix='/bitbucket-dc-proxy')

BITBUCKET_DC_TIMEOUT = 10  # seconds


# Bitbucket Data Center is not an OIDC provider, so keycloak
# can't retrieve user info from it directly.
# This endpoint proxies requests to bitbucket data center to get user info
# given a Bitbucket Data Center access token. Keycloak
# is configured to use this endpoint as the User Info Endpoint
# for the Bitbucket Data Center OIDC provider.
@router.get('/oauth2/userinfo')
async def userinfo(request: Request):
    if not BITBUCKET_DATA_CENTER_HOST:
        raise ValueError('BITBUCKET_DATA_CENTER_HOST must be configured')
    bitbucket_base_url = f'https://{BITBUCKET_DATA_CENTER_HOST}'

    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return JSONResponse({'error': 'missing_token'}, status_code=401)

    headers = {'Authorization': auth_header}
    async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
        # Step 1: get username
        whoami_resp = await client.get(
            f'{bitbucket_base_url}/plugins/servlet/applinks/whoami',
            headers=headers,
            timeout=BITBUCKET_DC_TIMEOUT,
        )
        if whoami_resp.status_code != 200:
            return JSONResponse({'error': 'not_authenticated'}, status_code=401)
        username = whoami_resp.text.strip()
        if not username:
            return JSONResponse({'error': 'not_authenticated'}, status_code=401)

        # Step 2: get user details
        user_resp = await client.get(
            f'{bitbucket_base_url}/rest/api/latest/users/{username}',
            headers=headers,
            timeout=BITBUCKET_DC_TIMEOUT,
        )
        if user_resp.status_code != 200:
            return JSONResponse(
                {'error': f'bitbucket_error: {user_resp.status_code}'},
                status_code=user_resp.status_code,
            )
        user_data = user_resp.json()

    return JSONResponse(
        {
            'sub': str(user_data.get('id', username)),
            'preferred_username': user_data.get('name', username),
            'name': user_data.get('displayName', username),
            'email': user_data.get('emailAddress', ''),
        }
    )
