"""Per-user Jira DC OAuth token resolution and refresh."""

import time
from dataclasses import dataclass
from typing import Final

import httpx
from pydantic import SecretStr
from server.auth.constants import (
    JIRA_DC_BASE_URL,
    JIRA_DC_CLIENT_ID,
    JIRA_DC_CLIENT_SECRET,
    JIRA_DC_HTTP_TIMEOUT,
)
from server.auth.token_manager import TokenManager
from storage.jira_dc_integration_store import JiraDcIntegrationStore

from openhands.app_server.utils.http_session import httpx_verify_option
from openhands.app_server.utils.logger import openhands_logger as logger

# Refresh access tokens this many seconds before wire expiry.
_ACCESS_REFRESH_BUFFER_SECONDS: Final = 300

JIRA_DC_TOKEN_URL = f'{JIRA_DC_BASE_URL}/rest/oauth2/latest/token'


class JiraDcUserTokenError(Exception):
    """Per-user Jira DC OAuth token is missing or unrefreshable."""


@dataclass(frozen=True)
class JiraDcUserToken:
    access_token: SecretStr
    expires_at: int  # 0 if unknown


async def get_user_jira_dc_token(
    *,
    keycloak_user_id: str,
    workspace_id: int,
    token_manager: TokenManager,
    store: JiraDcIntegrationStore,
) -> JiraDcUserToken:
    """Resolve a valid per-user Jira DC access token, refreshing if necessary.

    Raises JiraDcUserTokenError when no usable token is available.
    """
    # TODO(jira-dc-refresh-race): single-flight refreshes with SELECT FOR UPDATE
    # or an advisory lock before exposing this token through shared account-level
    # integrations. Some IdPs rotate refresh tokens on use, so concurrent
    # refreshes can otherwise write a stale refresh token back to the row.
    row = await store.get_user_oauth_tokens(
        keycloak_user_id=keycloak_user_id,
        workspace_id=workspace_id,
    )
    if row is None:
        raise JiraDcUserTokenError(
            'No stored Jira DC OAuth token for this user/workspace. '
            'Please re-link via OpenHands Settings → Integrations.'
        )

    enc_access, enc_refresh, access_exp, refresh_exp = row
    now = int(time.time())

    fresh = access_exp == 0 or access_exp > now + _ACCESS_REFRESH_BUFFER_SECONDS
    if fresh:
        return JiraDcUserToken(
            SecretStr(token_manager.decrypt_text(enc_access)), access_exp
        )

    if not enc_refresh:
        raise JiraDcUserTokenError(
            'Jira DC access token expired and no refresh token is stored. '
            'Please re-link via OpenHands Settings → Integrations.'
        )
    if refresh_exp and refresh_exp <= now:
        raise JiraDcUserTokenError(
            'Jira DC refresh token has expired. '
            'Please re-link via OpenHands Settings → Integrations.'
        )

    refresh_token = token_manager.decrypt_text(enc_refresh)
    try:
        async with httpx.AsyncClient(
            verify=httpx_verify_option(), timeout=JIRA_DC_HTTP_TIMEOUT
        ) as client:
            response = await client.post(
                JIRA_DC_TOKEN_URL,
                data={
                    'grant_type': 'refresh_token',
                    'client_id': JIRA_DC_CLIENT_ID,
                    'client_secret': JIRA_DC_CLIENT_SECRET,
                    'refresh_token': refresh_token,
                },
            )
    except httpx.RequestError as e:
        # Unreachable token endpoint -- usually wrong scheme (http vs https) or
        # blocked egress; surface that instead of a bare connection error.
        logger.warning(
            '[Jira DC] Token refresh could not reach %s: %s',
            JIRA_DC_TOKEN_URL,
            type(e).__name__,
        )
        raise JiraDcUserTokenError(
            f'Could not reach the Jira Data Center token endpoint '
            f'({JIRA_DC_TOKEN_URL}): {type(e).__name__}. Check that the Jira Base '
            f'URL uses https and that OpenHands can reach the Jira server.'
        ) from e

    if response.status_code != 200:
        logger.warning(
            '[Jira DC] Token refresh failed: %s %s',
            response.status_code,
            response.text[:200],
        )
        raise JiraDcUserTokenError(
            'Jira DC token refresh failed. '
            'Please re-link via OpenHands Settings → Integrations.'
        )
    data = response.json()

    new_access = data.get('access_token')
    if not new_access:
        raise JiraDcUserTokenError(
            'Jira DC token refresh returned no access_token. '
            'Please re-link via OpenHands Settings → Integrations.'
        )
    # Some IdPs rotate the refresh token on each use; fall back to the existing one.
    new_refresh = data.get('refresh_token') or refresh_token
    new_expires_in = int(data.get('expires_in') or 0)
    new_refresh_expires_in = int(
        data.get('refresh_token_expires_in') or data.get('refresh_expires_in') or 0
    )
    ts = int(time.time())
    new_access_exp = (ts + new_expires_in) if new_expires_in else 0
    new_refresh_exp = (ts + new_refresh_expires_in) if new_refresh_expires_in else 0

    updated_count = await store.update_user_oauth_tokens(
        keycloak_user_id=keycloak_user_id,
        workspace_id=workspace_id,
        encrypted_access_token=token_manager.encrypt_text(new_access),
        encrypted_refresh_token=token_manager.encrypt_text(new_refresh),
        access_token_expires_at=new_access_exp,
        refresh_token_expires_at=new_refresh_exp,
    )
    if updated_count == 0:
        raise JiraDcUserTokenError(
            'Stored Jira DC OAuth link was not found while refreshing tokens. '
            'Please re-link via OpenHands Settings → Integrations.'
        )
    return JiraDcUserToken(SecretStr(new_access), new_access_exp)
