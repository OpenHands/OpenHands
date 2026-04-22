"""Tests for BitbucketDCService core: init, headers, get_user, pagination, email."""

import base64
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from openhands.integrations.bitbucket_data_center.bitbucket_dc_service import (
    BitbucketDCService,
)
from openhands.integrations.service_types import AuthenticationError, User
from openhands.server.types import AppMode

# ── init / BASE_URL ───────────────────────────────────────────────────────────


def test_init_plain_domain():
    svc = BitbucketDCService(token=SecretStr('tok'), base_domain='host.example.com')
    assert svc.BASE_URL == 'https://host.example.com/rest/api/1.0'


def test_init_no_domain():
    svc = BitbucketDCService(token=SecretStr('tok'), base_domain=None)
    assert svc.BASE_URL == ''


# ── token wrapping ────────────────────────────────────────────────────────────


def test_token_wraps_simple_token():
    svc = BitbucketDCService(token=SecretStr('mytoken'))
    assert svc.token.get_secret_value() == 'x-token-auth:mytoken'


def test_token_preserves_colon_token():
    svc = BitbucketDCService(token=SecretStr('alice:secret'))
    assert svc.token.get_secret_value() == 'alice:secret'


# ── user_id derivation ────────────────────────────────────────────────────────


def test_user_id_derived_from_username_password_token():
    svc = BitbucketDCService(token=SecretStr('alice:secret'))
    assert svc.user_id == 'alice'


def test_user_id_not_derived_from_xtoken_auth_token():
    svc = BitbucketDCService(token=SecretStr('x-token-auth:mytoken'))
    assert svc.user_id is None


def test_explicit_user_id_not_overridden():
    svc = BitbucketDCService(token=SecretStr('alice:secret'), user_id='bob')
    assert svc.user_id == 'bob'


# ── _get_headers ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_headers_basic_auth():
    svc = BitbucketDCService(
        token=SecretStr('user:pass'), base_domain='host.example.com'
    )
    headers = await svc._get_headers()
    expected = 'Basic ' + base64.b64encode(b'user:pass').decode()
    assert headers['Authorization'] == expected


@pytest.mark.asyncio
async def test_get_headers_xtoken_auth():
    svc = BitbucketDCService(
        token=SecretStr('plaintoken'), base_domain='host.example.com'
    )
    # plaintoken has no ':' so it gets wrapped as x-token-auth:plaintoken
    headers = await svc._get_headers()
    expected = 'Basic ' + base64.b64encode(b'x-token-auth:plaintoken').decode()
    assert headers['Authorization'] == expected


# ── get_user ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_user_with_user_id():
    svc = BitbucketDCService(
        token=SecretStr('tok'),
        base_domain='host.example.com',
        user_id='jdoe',
    )
    mock_response = {
        'values': [
            {
                'id': 5,
                'slug': 'jdoe',
                'name': 'jdoe',
                'displayName': 'J Doe',
                'emailAddress': 'j@example.com',
                'avatarUrl': '',
            }
        ]
    }
    with patch.object(svc, '_make_request', return_value=(mock_response, {})):
        user = await svc.get_user()

    assert user.id == '5'
    assert user.login == 'jdoe'
    assert user.name == 'J Doe'
    assert user.email == 'j@example.com'


@pytest.mark.asyncio
async def test_get_user_without_user_id():
    # x-token-auth tokens don't have a derivable username, so user_id stays None
    svc = BitbucketDCService(
        token=SecretStr('x-token-auth:mytoken'), base_domain='host.example.com'
    )
    with patch.object(svc, '_make_request') as mock_req:
        user = await svc.get_user()
        mock_req.assert_not_called()

    assert isinstance(user, User)
    assert user.id == ''
    assert user.login == ''


@pytest.mark.asyncio
async def test_get_user_raises_when_not_found():
    svc = BitbucketDCService(
        token=SecretStr('tok'),
        base_domain='host.example.com',
        user_id='jdoe',
    )
    mock_response = {'values': []}
    with patch.object(svc, '_make_request', return_value=(mock_response, {})):
        with pytest.raises(AuthenticationError):
            await svc.get_user()


# ── _resolve_primary_email ────────────────────────────────────────────────────


def test_resolve_primary_email_selects_primary_confirmed():
    from openhands.integrations.bitbucket_data_center.service.base import (
        BitbucketDCMixinBase,
    )

    emails = [
        {'email': 'secondary@example.com', 'is_primary': False, 'is_confirmed': True},
        {'email': 'primary@example.com', 'is_primary': True, 'is_confirmed': True},
        {
            'email': 'unconfirmed@example.com',
            'is_primary': False,
            'is_confirmed': False,
        },
    ]
    result = BitbucketDCMixinBase._resolve_primary_email(emails)
    assert result == 'primary@example.com'


def test_resolve_primary_email_returns_none_when_no_primary():
    from openhands.integrations.bitbucket_data_center.service.base import (
        BitbucketDCMixinBase,
    )

    emails = [
        {'email': 'a@example.com', 'is_primary': False, 'is_confirmed': True},
        {'email': 'b@example.com', 'is_primary': False, 'is_confirmed': True},
    ]
    result = BitbucketDCMixinBase._resolve_primary_email(emails)
    assert result is None


def test_resolve_primary_email_returns_none_when_primary_not_confirmed():
    from openhands.integrations.bitbucket_data_center.service.base import (
        BitbucketDCMixinBase,
    )

    emails = [
        {'email': 'primary@example.com', 'is_primary': True, 'is_confirmed': False},
        {'email': 'other@example.com', 'is_primary': False, 'is_confirmed': True},
    ]
    result = BitbucketDCMixinBase._resolve_primary_email(emails)
    assert result is None


def test_resolve_primary_email_returns_none_for_empty_list():
    from openhands.integrations.bitbucket_data_center.service.base import (
        BitbucketDCMixinBase,
    )

    result = BitbucketDCMixinBase._resolve_primary_email([])
    assert result is None


# ── get_user_emails ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_user_emails():
    svc = BitbucketDCService(token=SecretStr('tok'), base_domain='host.example.com')
    mock_response = {
        'values': [
            {'email': 'primary@example.com', 'is_primary': True, 'is_confirmed': True},
            {
                'email': 'secondary@example.com',
                'is_primary': False,
                'is_confirmed': True,
            },
        ]
    }
    with patch.object(svc, '_make_request', return_value=(mock_response, {})):
        emails = await svc.get_user_emails()

    assert emails == mock_response['values']


# ── pagination (get_all_repositories iterates projects) ──────────────────────


@pytest.mark.asyncio
async def test_pagination_iterates_projects():
    svc = BitbucketDCService(token=SecretStr('tok'), base_domain='host.example.com')

    def _repo_dict(key='PROJ', slug='myrepo'):
        return {'id': 1, 'slug': slug, 'project': {'key': key}, 'public': False}

    async def fake_fetch(url, params, max_items):
        if '/projects' in url and '/repos' not in url:
            return [{'key': 'PROJ1'}, {'key': 'PROJ2'}]
        if 'PROJ1' in url:
            return [_repo_dict('PROJ1', 'repo1')]
        if 'PROJ2' in url:
            return [_repo_dict('PROJ2', 'repo2')]
        return []

    with patch.object(svc, '_fetch_paginated_data', side_effect=fake_fetch):
        repos = await svc.get_all_repositories('name', AppMode.SAAS)

    full_names = {r.full_name for r in repos}
    assert 'PROJ1/repo1' in full_names
    assert 'PROJ2/repo2' in full_names


# ── verify_access ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_access_makes_request():
    svc = BitbucketDCService(token=SecretStr('tok'), base_domain='host.example.com')
    with patch.object(svc, '_make_request', return_value=({}, {})) as mock_req:
        await svc.verify_access()

    mock_req.assert_called_once()
    call_url = mock_req.call_args[0][0]
    assert call_url.endswith('/repos')
