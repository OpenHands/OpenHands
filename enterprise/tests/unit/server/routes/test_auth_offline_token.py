"""Tests for the keycloak offline-token callback flow.

Regression coverage for the bug fixed in
https://github.com/OpenHands/OpenHands/pull/14387:

The keycloak *offline* callback previously wrote the offline refresh token
into the ``keycloak_auth`` cookie. The regular ``/logout`` endpoint reads
that cookie to call Keycloak's ``/logout`` endpoint with the refresh token,
which terminates the associated session. When the cookie contained the
*offline* token, logging out killed the offline session as well, which in
turn invalidated any API keys that depended on the offline token (only
visible for new users, since pre-existing users had an unrelated offline
token stored).

The flow exercised here mirrors the manual test in the PR description:

  1. A new user completes the keycloak offline-token authentication flow.
  2. The user logs out, which ends the *online* keycloak session.
  3. The offline token (used to authenticate API key requests) must still
     work because the offline callback no longer overwrites the auth
     cookie with the offline refresh token.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from server.routes.auth import keycloak_offline_callback, logout


@pytest.fixture
def mock_request():
    """Mock FastAPI Request used by the offline callback."""
    request = MagicMock(spec=Request)
    request.url = MagicMock()
    request.url.hostname = 'localhost'
    request.url.netloc = 'localhost:8000'
    request.url.path = '/oauth/keycloak/offline/callback'
    request.base_url = 'http://localhost:8000/'
    request.headers = {}
    request.cookies = {}
    return request


class TestOfflineCallbackPreservesAuthCookie:
    """The offline callback must not overwrite the online ``keycloak_auth`` cookie."""

    @pytest.mark.asyncio
    async def test_offline_callback_does_not_set_keycloak_auth_cookie(
        self, mock_request, create_keycloak_user_info
    ):
        """Regression test: the offline callback must not call ``set_response_cookie``.

        Putting the offline refresh token in the ``keycloak_auth`` cookie causes
        the subsequent ``/logout`` call to terminate the offline session along
        with the online one, which invalidates the user's API keys.
        """
        with (
            patch('server.routes.auth.token_manager') as mock_token_manager,
            patch('server.routes.auth.UserStore') as mock_user_store,
            patch('server.routes.auth.set_response_cookie') as mock_set_cookie,
            patch(
                'server.routes.auth._get_post_auth_redirect',
                new_callable=AsyncMock,
                return_value='http://localhost:8000/',
            ),
        ):
            mock_user_store.get_user_by_id = AsyncMock(return_value=None)
            mock_token_manager.get_keycloak_tokens = AsyncMock(
                return_value=('online_access_token', 'offline_refresh_token')
            )
            mock_token_manager.get_user_info = AsyncMock(
                return_value=create_keycloak_user_info(sub='new_user_id')
            )
            mock_token_manager.store_offline_token = AsyncMock()

            result = await keycloak_offline_callback(
                'test_code', 'test_state', mock_request
            )

            # The offline token is persisted in the dedicated offline-token store...
            mock_token_manager.store_offline_token.assert_awaited_once_with(
                user_id='new_user_id', offline_token='offline_refresh_token'
            )
            # ...but it must NOT be written into the keycloak_auth cookie.
            mock_set_cookie.assert_not_called()

            assert isinstance(result, RedirectResponse)
            assert result.status_code == 302
            # And as a belt-and-braces check, the redirect response has no
            # Set-Cookie header overwriting the auth cookie.
            cookie_headers = [
                v for k, v in result.raw_headers if k.lower() == b'set-cookie'
            ]
            assert not any(b'keycloak_auth' in h for h in cookie_headers)


class TestOfflineTokenSurvivesLogout:
    """End-to-end style test mirroring the PR's manual repro steps."""

    @pytest.mark.asyncio
    async def test_offline_token_survives_user_logout(
        self, mock_request, create_keycloak_user_info
    ):
        """Simulate: new user -> offline auth -> logout -> offline token still valid.

        After PR #14387 the offline callback only persists the offline
        refresh token via ``token_manager.store_offline_token``; it does not
        touch the auth cookie. The online ``/logout`` endpoint therefore
        cannot accidentally pass the offline token to Keycloak.
        """
        user_id = 'new_user_id'
        online_refresh_token = 'online_refresh_token'
        offline_refresh_token = 'offline_refresh_token_for_api_keys'

        with (
            patch('server.routes.auth.token_manager') as mock_token_manager,
            patch('server.routes.auth.UserStore') as mock_user_store,
            patch('server.routes.auth.set_response_cookie') as mock_set_cookie,
            patch(
                'server.routes.auth._get_post_auth_redirect',
                new_callable=AsyncMock,
                return_value='http://localhost:8000/',
            ),
        ):
            mock_user_store.get_user_by_id = AsyncMock(return_value=None)
            mock_token_manager.get_keycloak_tokens = AsyncMock(
                return_value=('online_access_token', offline_refresh_token)
            )
            mock_token_manager.get_user_info = AsyncMock(
                return_value=create_keycloak_user_info(sub=user_id)
            )
            mock_token_manager.store_offline_token = AsyncMock()
            mock_token_manager.logout = AsyncMock()
            mock_token_manager.validate_offline_token = AsyncMock(return_value=True)

            # 1. New user completes keycloak offline-token authentication.
            await keycloak_offline_callback('test_code', 'test_state', mock_request)
            mock_token_manager.store_offline_token.assert_awaited_once_with(
                user_id=user_id, offline_token=offline_refresh_token
            )
            # The offline callback must not write the offline token into the
            # online auth cookie (which would later be passed to keycloak logout).
            mock_set_cookie.assert_not_called()

            # 2. The user logs out. The logout endpoint pulls the online
            # refresh token from the request (via SaasUserAuth) and only ever
            # passes that to keycloak -- never the offline token.
            logout_request = MagicMock(spec=Request)
            logout_request.cookies = {}
            logout_request.headers = {}

            mock_user_auth = MagicMock()
            mock_user_auth.refresh_token = MagicMock()
            mock_user_auth.refresh_token.get_secret_value.return_value = (
                online_refresh_token
            )

            with patch(
                'server.routes.auth.get_user_auth',
                new_callable=AsyncMock,
                return_value=mock_user_auth,
            ):
                response = await logout(logout_request)

            assert isinstance(response, JSONResponse)
            assert response.status_code == status.HTTP_200_OK
            # Keycloak logout is called with the *online* refresh token only.
            mock_token_manager.logout.assert_awaited_once_with(online_refresh_token)
            logout_call_args = mock_token_manager.logout.await_args.args
            assert offline_refresh_token not in logout_call_args

            # 3. The offline token (used by API keys) is still valid after
            # logout because it was never touched by the online logout flow.
            assert await mock_token_manager.validate_offline_token(user_id=user_id)
