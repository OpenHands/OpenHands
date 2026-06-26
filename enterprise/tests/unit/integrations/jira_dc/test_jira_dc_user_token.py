"""Unit tests for jira_dc_user_token — token resolution and refresh logic."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from integrations.jira_dc.jira_dc_user_token import (
    JiraDcUserToken,
    JiraDcUserTokenError,
    get_user_jira_dc_token,
)
from pydantic import SecretStr


def _make_store(row):
    store = MagicMock()
    store.get_user_oauth_tokens = AsyncMock(return_value=row)
    store.update_user_oauth_tokens = AsyncMock(return_value=1)
    return store


def _make_token_manager(decrypt_side_effect=None, decrypt_return=None):
    tm = MagicMock()
    if decrypt_side_effect:
        tm.decrypt_text.side_effect = decrypt_side_effect
    else:
        tm.decrypt_text.return_value = decrypt_return or 'decrypted'
    tm.encrypt_text.side_effect = lambda v: f'enc({v})'
    return tm


# ---------------------------------------------------------------------------
# Fast path — access token is still fresh
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fresh_access_token_returned_without_refresh():
    future_exp = int(time.time()) + 3600
    store = _make_store(('enc_access', 'enc_refresh', future_exp, 0))
    tm = _make_token_manager(decrypt_return='raw_access')

    result = await get_user_jira_dc_token(
        keycloak_user_id='kc1',
        workspace_id=1,
        token_manager=tm,
        store=store,
    )

    assert isinstance(result, JiraDcUserToken)
    assert isinstance(result.access_token, SecretStr)
    assert result.access_token.get_secret_value() == 'raw_access'
    assert result.expires_at == future_exp
    store.update_user_oauth_tokens.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_expiry_treated_as_fresh():
    """expires_at == 0 means we don't know; treat as fresh."""
    store = _make_store(('enc_access', None, 0, 0))
    tm = _make_token_manager(decrypt_return='raw_access')

    result = await get_user_jira_dc_token(
        keycloak_user_id='kc1',
        workspace_id=1,
        token_manager=tm,
        store=store,
    )

    assert result.access_token.get_secret_value() == 'raw_access'
    store.update_user_oauth_tokens.assert_not_called()


# ---------------------------------------------------------------------------
# No stored tokens → error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raises_when_no_stored_tokens():
    store = _make_store(None)
    tm = _make_token_manager()

    with pytest.raises(JiraDcUserTokenError, match='No stored'):
        await get_user_jira_dc_token(
            keycloak_user_id='kc1',
            workspace_id=1,
            token_manager=tm,
            store=store,
        )


# ---------------------------------------------------------------------------
# Expired access token, no refresh token → error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raises_when_expired_and_no_refresh_token():
    expired = int(time.time()) - 100
    store = _make_store(('enc_access', None, expired, 0))
    tm = _make_token_manager()

    with pytest.raises(JiraDcUserTokenError, match='no refresh token'):
        await get_user_jira_dc_token(
            keycloak_user_id='kc1',
            workspace_id=1,
            token_manager=tm,
            store=store,
        )


# ---------------------------------------------------------------------------
# Expired access token, refresh token also expired → error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raises_when_refresh_token_expired():
    expired = int(time.time()) - 100
    store = _make_store(('enc_access', 'enc_refresh', expired, expired))
    tm = _make_token_manager()

    with pytest.raises(JiraDcUserTokenError, match='refresh token has expired'):
        await get_user_jira_dc_token(
            keycloak_user_id='kc1',
            workspace_id=1,
            token_manager=tm,
            store=store,
        )


# ---------------------------------------------------------------------------
# Successful refresh
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_path_updates_store_and_returns_new_token():
    expired = int(time.time()) - 100
    future_refresh = int(time.time()) + 86400
    store = _make_store(('enc_access', 'enc_refresh', expired, future_refresh))

    decrypt_map = {'enc_access': 'old_access', 'enc_refresh': 'old_refresh'}
    tm = _make_token_manager(decrypt_side_effect=lambda v: decrypt_map[v])

    new_token_data = {
        'access_token': 'new_access',
        'refresh_token': 'new_refresh',
        'expires_in': 3600,
        'refresh_token_expires_in': 86400,
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = new_token_data

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch(
        'integrations.jira_dc.jira_dc_user_token.httpx.AsyncClient',
        return_value=mock_client,
    ):
        result = await get_user_jira_dc_token(
            keycloak_user_id='kc1',
            workspace_id=1,
            token_manager=tm,
            store=store,
        )

    assert result.access_token.get_secret_value() == 'new_access'
    store.update_user_oauth_tokens.assert_awaited_once()
    call_kwargs = store.update_user_oauth_tokens.call_args.kwargs
    assert call_kwargs['encrypted_access_token'] == 'enc(new_access)'
    assert call_kwargs['encrypted_refresh_token'] == 'enc(new_refresh)'
    assert call_kwargs['access_token_expires_at'] > 0
    assert call_kwargs['refresh_token_expires_at'] > 0


@pytest.mark.asyncio
async def test_refresh_uses_existing_refresh_token_when_idp_does_not_rotate():
    """When the IdP doesn't return a new refresh_token, keep the old one."""
    expired = int(time.time()) - 100
    future_refresh = int(time.time()) + 86400
    store = _make_store(('enc_access', 'enc_refresh', expired, future_refresh))

    decrypt_map = {'enc_access': 'old_access', 'enc_refresh': 'old_refresh'}
    tm = _make_token_manager(decrypt_side_effect=lambda v: decrypt_map[v])

    # IdP omits refresh_token in the response
    new_token_data = {'access_token': 'new_access', 'expires_in': 3600}

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = new_token_data

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch(
        'integrations.jira_dc.jira_dc_user_token.httpx.AsyncClient',
        return_value=mock_client,
    ):
        result = await get_user_jira_dc_token(
            keycloak_user_id='kc1',
            workspace_id=1,
            token_manager=tm,
            store=store,
        )

    assert result.access_token.get_secret_value() == 'new_access'
    call_kwargs = store.update_user_oauth_tokens.call_args.kwargs
    # Refresh token should be the original one re-encrypted
    assert call_kwargs['encrypted_refresh_token'] == 'enc(old_refresh)'


# ---------------------------------------------------------------------------
# Refresh HTTP failure → error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raises_when_refresh_request_fails():
    expired = int(time.time()) - 100
    future_refresh = int(time.time()) + 86400
    store = _make_store(('enc_access', 'enc_refresh', expired, future_refresh))
    tm = _make_token_manager(decrypt_return='tok')

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = 'Unauthorized'

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch(
        'integrations.jira_dc.jira_dc_user_token.httpx.AsyncClient',
        return_value=mock_client,
    ):
        with pytest.raises(JiraDcUserTokenError, match='refresh failed'):
            await get_user_jira_dc_token(
                keycloak_user_id='kc1',
                workspace_id=1,
                token_manager=tm,
                store=store,
            )


@pytest.mark.asyncio
async def test_raises_actionable_error_when_token_endpoint_unreachable():
    """A connection error (wrong scheme / blocked egress) surfaces an actionable
    JiraDcUserTokenError, not a bare httpx exception."""
    expired = int(time.time()) - 100
    future_refresh = int(time.time()) + 86400
    store = _make_store(('enc_access', 'enc_refresh', expired, future_refresh))
    tm = _make_token_manager(decrypt_return='tok')

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(side_effect=httpx.ConnectTimeout('timed out'))

    with patch(
        'integrations.jira_dc.jira_dc_user_token.httpx.AsyncClient',
        return_value=mock_client,
    ):
        with pytest.raises(JiraDcUserTokenError, match='Could not reach'):
            await get_user_jira_dc_token(
                keycloak_user_id='kc1',
                workspace_id=1,
                token_manager=tm,
                store=store,
            )

    store.update_user_oauth_tokens.assert_not_called()


@pytest.mark.asyncio
async def test_raises_when_refresh_response_has_no_access_token():
    expired = int(time.time()) - 100
    future_refresh = int(time.time()) + 86400
    store = _make_store(('enc_access', 'enc_refresh', expired, future_refresh))

    decrypt_map = {'enc_access': 'old_access', 'enc_refresh': 'old_refresh'}
    tm = _make_token_manager(decrypt_side_effect=lambda v: decrypt_map[v])

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'expires_in': 3600}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch(
        'integrations.jira_dc.jira_dc_user_token.httpx.AsyncClient',
        return_value=mock_client,
    ):
        with pytest.raises(JiraDcUserTokenError, match='returned no access_token'):
            await get_user_jira_dc_token(
                keycloak_user_id='kc1',
                workspace_id=1,
                token_manager=tm,
                store=store,
            )

    store.update_user_oauth_tokens.assert_not_called()


@pytest.mark.asyncio
async def test_raises_when_refreshed_token_update_matches_no_rows():
    expired = int(time.time()) - 100
    future_refresh = int(time.time()) + 86400
    store = _make_store(('enc_access', 'enc_refresh', expired, future_refresh))
    store.update_user_oauth_tokens.return_value = 0

    decrypt_map = {'enc_access': 'old_access', 'enc_refresh': 'old_refresh'}
    tm = _make_token_manager(decrypt_side_effect=lambda v: decrypt_map[v])

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'access_token': 'new_access', 'expires_in': 3600}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch(
        'integrations.jira_dc.jira_dc_user_token.httpx.AsyncClient',
        return_value=mock_client,
    ):
        with pytest.raises(JiraDcUserTokenError, match='link was not found'):
            await get_user_jira_dc_token(
                keycloak_user_id='kc1',
                workspace_id=1,
                token_manager=tm,
                store=store,
            )
