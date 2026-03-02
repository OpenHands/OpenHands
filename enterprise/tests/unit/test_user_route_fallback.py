"""Tests for the fallback User path in the /api/user/info endpoint.

When a user authenticates via Keycloak without provider tokens (e.g., SAML, enterprise SSO),
the endpoint constructs a User from OIDC claims. These tests verify that name and company
fields are correctly populated from Keycloak claims in this fallback path.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from openhands.integrations.service_types import User


@pytest.fixture
def mock_token_manager():
    """Mock the token_manager used by user.py routes."""
    with patch('server.routes.user.token_manager') as mock_tm:
        yield mock_tm


@pytest.fixture
def mock_check_idp():
    """Mock _check_idp to pass through the default_value (the fallback User).

    This isolates the test to just the User construction logic in saas_get_user,
    without needing to set up Keycloak IDP token checks.
    """
    with patch('server.routes.user._check_idp') as mock_fn:
        # Return the default_value argument that was passed to _check_idp
        mock_fn.side_effect = lambda **kwargs: kwargs.get('default_value')
        yield mock_fn


@pytest.fixture
def mock_user_store():
    """Mock UserStore.get_user_by_id_async to return None by default."""
    with patch(
        'server.routes.user.UserStore.get_user_by_id_async',
        new_callable=AsyncMock,
        return_value=None,
    ) as mock_fn:
        yield mock_fn


@pytest.mark.asyncio
async def test_fallback_user_includes_name_from_name_claim(
    mock_token_manager, mock_check_idp, mock_user_store
):
    """When Keycloak provides a 'name' claim, the fallback User should include it."""
    from server.routes.user import saas_get_user

    mock_token_manager.get_user_info = AsyncMock(
        return_value={
            'sub': '248289761001',
            'name': 'Jane Doe',
            'preferred_username': 'j.doe',
            'email': 'jane@example.com',
        }
    )

    result = await saas_get_user(
        provider_tokens=None,
        access_token=SecretStr('test-token'),
        user_id='248289761001',
    )

    assert isinstance(result, User)
    assert result.name == 'Jane Doe'
    assert result.email == 'jane@example.com'


@pytest.mark.asyncio
async def test_fallback_user_combines_given_and_family_name(
    mock_token_manager, mock_check_idp, mock_user_store
):
    """When 'name' is absent, combine given_name + family_name."""
    from server.routes.user import saas_get_user

    mock_token_manager.get_user_info = AsyncMock(
        return_value={
            'sub': '248289761001',
            'given_name': 'Jane',
            'family_name': 'Doe',
            'preferred_username': 'j.doe',
            'email': 'jane@example.com',
        }
    )

    result = await saas_get_user(
        provider_tokens=None,
        access_token=SecretStr('test-token'),
        user_id='248289761001',
    )

    assert isinstance(result, User)
    assert result.name == 'Jane Doe'


@pytest.mark.asyncio
async def test_fallback_user_name_is_none_when_no_name_claims(
    mock_token_manager, mock_check_idp, mock_user_store
):
    """When no name claims exist, name should be None."""
    from server.routes.user import saas_get_user

    mock_token_manager.get_user_info = AsyncMock(
        return_value={
            'sub': '248289761001',
            'preferred_username': 'j.doe',
            'email': 'jane@example.com',
        }
    )

    result = await saas_get_user(
        provider_tokens=None,
        access_token=SecretStr('test-token'),
        user_id='248289761001',
    )

    assert isinstance(result, User)
    assert result.name is None


@pytest.mark.asyncio
async def test_fallback_user_includes_company_claim(
    mock_token_manager, mock_check_idp, mock_user_store
):
    """When Keycloak provides a 'company' claim, include it in the User."""
    from server.routes.user import saas_get_user

    mock_token_manager.get_user_info = AsyncMock(
        return_value={
            'sub': '248289761001',
            'name': 'Jane Doe',
            'preferred_username': 'j.doe',
            'email': 'jane@example.com',
            'company': 'Acme Corp',
        }
    )

    result = await saas_get_user(
        provider_tokens=None,
        access_token=SecretStr('test-token'),
        user_id='248289761001',
    )

    assert isinstance(result, User)
    assert result.company == 'Acme Corp'


@pytest.mark.asyncio
async def test_fallback_user_company_is_none_when_absent(
    mock_token_manager, mock_check_idp, mock_user_store
):
    """When 'company' is not in Keycloak claims, company should be None."""
    from server.routes.user import saas_get_user

    mock_token_manager.get_user_info = AsyncMock(
        return_value={
            'sub': '248289761001',
            'name': 'Jane Doe',
            'preferred_username': 'j.doe',
            'email': 'jane@example.com',
        }
    )

    result = await saas_get_user(
        provider_tokens=None,
        access_token=SecretStr('test-token'),
        user_id='248289761001',
    )

    assert isinstance(result, User)
    assert result.company is None


@pytest.mark.asyncio
async def test_fallback_user_email_from_db_when_available(
    mock_token_manager, mock_check_idp, mock_user_store
):
    """When User.email is stored in DB, use it instead of Keycloak's live email."""
    from server.routes.user import saas_get_user

    mock_token_manager.get_user_info = AsyncMock(
        return_value={
            'sub': '248289761001',
            'preferred_username': 'j.doe',
            'email': 'keycloak@example.com',
        }
    )

    mock_db_user = MagicMock()
    mock_db_user.email = 'db@example.com'
    mock_user_store.return_value = mock_db_user

    result = await saas_get_user(
        provider_tokens=None,
        access_token=SecretStr('test-token'),
        user_id='248289761001',
    )

    assert isinstance(result, User)
    assert result.email == 'db@example.com'


@pytest.mark.asyncio
async def test_fallback_user_email_falls_back_to_keycloak_when_db_null(
    mock_token_manager, mock_check_idp, mock_user_store
):
    """When User.email is NULL in DB, fall back to Keycloak's email."""
    from server.routes.user import saas_get_user

    mock_token_manager.get_user_info = AsyncMock(
        return_value={
            'sub': '248289761001',
            'preferred_username': 'j.doe',
            'email': 'keycloak@example.com',
        }
    )

    mock_db_user = MagicMock()
    mock_db_user.email = None
    mock_user_store.return_value = mock_db_user

    result = await saas_get_user(
        provider_tokens=None,
        access_token=SecretStr('test-token'),
        user_id='248289761001',
    )

    assert isinstance(result, User)
    assert result.email == 'keycloak@example.com'


@pytest.mark.asyncio
async def test_fallback_user_email_falls_back_to_keycloak_when_no_db_user(
    mock_token_manager, mock_check_idp, mock_user_store
):
    """When DB user doesn't exist, fall back to Keycloak's email."""
    from server.routes.user import saas_get_user

    mock_token_manager.get_user_info = AsyncMock(
        return_value={
            'sub': '248289761001',
            'preferred_username': 'j.doe',
            'email': 'keycloak@example.com',
        }
    )

    # mock_user_store already returns None by default

    result = await saas_get_user(
        provider_tokens=None,
        access_token=SecretStr('test-token'),
        user_id='248289761001',
    )

    assert isinstance(result, User)
    assert result.email == 'keycloak@example.com'
