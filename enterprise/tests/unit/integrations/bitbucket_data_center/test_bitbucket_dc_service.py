"""Unit tests for SaaSBitbucketDCService."""

from unittest.mock import AsyncMock, patch

import pytest
from integrations.bitbucket_data_center.bitbucket_dc_service import (
    SaaSBitbucketDCService,
)
from pydantic import SecretStr
from server.auth.token_manager import TokenManager

from openhands.app_server.integrations.service_types import RequestMethod


@pytest.fixture
def service():
    return SaaSBitbucketDCService()


@pytest.fixture
def service_with_external_auth_token():
    return SaaSBitbucketDCService(external_auth_token=SecretStr('test_keycloak_token'))


@pytest.fixture
def service_with_external_auth_id():
    return SaaSBitbucketDCService(external_auth_id='test_user_id')


@pytest.fixture
def service_with_user_id():
    return SaaSBitbucketDCService(user_id='test_user_id')


class TestSaaSBitbucketDCServiceInit:
    def test_refresh_flag_is_true(self):
        # self.refresh = True is required so the base class BitbucketDCService
        # retries the request with a refreshed token on 401 responses.
        # See openhands/app_server/integrations/bitbucket_data_center/service/base.py,
        # which checks `if self.refresh` before attempting the retry.
        service = SaaSBitbucketDCService()
        assert service.refresh is True

    def test_token_manager_is_created(self):
        service = SaaSBitbucketDCService()
        assert isinstance(service.token_manager, TokenManager)

    def test_external_token_manager_flag_passed(self):
        service = SaaSBitbucketDCService(external_token_manager=True)
        assert service.token_manager.external is True


class TestGetLatestToken:
    @pytest.mark.asyncio
    async def test_get_latest_token_with_external_auth_token(
        self, service_with_external_auth_token
    ):
        expected_token = 'test_bitbucket_dc_token'
        with patch.object(
            service_with_external_auth_token.token_manager,
            'get_idp_token',
            new_callable=AsyncMock,
            return_value=expected_token,
        ):
            token = await service_with_external_auth_token.get_latest_token()

        assert token is not None
        assert token.get_secret_value() == expected_token

    @pytest.mark.asyncio
    async def test_get_latest_token_with_external_auth_id(
        self, service_with_external_auth_id
    ):
        offline_token = 'test_offline_token'
        expected_token = 'test_bitbucket_dc_token'
        with (
            patch.object(
                service_with_external_auth_id.token_manager,
                'load_offline_token',
                new_callable=AsyncMock,
                return_value=offline_token,
            ),
            patch.object(
                service_with_external_auth_id.token_manager,
                'get_idp_token_from_offline_token',
                new_callable=AsyncMock,
                return_value=expected_token,
            ),
        ):
            token = await service_with_external_auth_id.get_latest_token()

        assert token is not None
        assert token.get_secret_value() == expected_token

    @pytest.mark.asyncio
    async def test_get_latest_token_with_user_id(self, service_with_user_id):
        expected_token = 'test_bitbucket_dc_token'
        with patch.object(
            service_with_user_id.token_manager,
            'get_idp_token_from_idp_user_id',
            new_callable=AsyncMock,
            return_value=expected_token,
        ):
            token = await service_with_user_id.get_latest_token()

        assert token is not None
        assert token.get_secret_value() == expected_token

    @pytest.mark.asyncio
    async def test_get_latest_token_no_auth_returns_none(self, service):
        token = await service.get_latest_token()
        assert token is None

    @pytest.mark.asyncio
    async def test_get_latest_token_external_auth_token_priority(self):
        """external_auth_token takes priority over external_auth_id."""
        expected_token = 'test_bitbucket_dc_token'
        service = SaaSBitbucketDCService(
            external_auth_token=SecretStr('test_keycloak_token'),
            external_auth_id='test_user_id',
        )
        with (
            patch.object(
                service.token_manager,
                'get_idp_token',
                new_callable=AsyncMock,
                return_value=expected_token,
            ) as mock_get_idp_token,
            patch.object(
                service.token_manager,
                'load_offline_token',
                new_callable=AsyncMock,
            ) as mock_load_offline,
        ):
            token = await service.get_latest_token()

        assert token is not None
        assert token.get_secret_value() == expected_token
        mock_get_idp_token.assert_called_once()
        mock_load_offline.assert_not_called()


@pytest.mark.asyncio
async def test_add_comment_reaction_uses_comment_likes_put():
    """BBDC reactions live in the comment-likes plugin, not core /rest/api/1.0.

    Regression guard: the call must be a PUT to
    ``/rest/comment-likes/latest/.../comments/{id}/reactions/{emoticon}`` with
    the bare emoticon name (``eyes``). The original endpoint
    (POST /rest/api/1.0/.../reactions with ``:eyes:``) returns 404/400.
    """
    service = SaaSBitbucketDCService()
    service.BASE_URL = 'https://bb.example.com/rest/api/1.0'

    with patch.object(service, '_make_request', new_callable=AsyncMock) as mock_request:
        await service.add_comment_reaction(
            owner='PROJ',
            repo_slug='myrepo',
            pr_id=7,
            comment_id=99,
            emoticon='eyes',
        )

    mock_request.assert_awaited_once()
    args, kwargs = mock_request.await_args
    assert args[0] == (
        'https://bb.example.com/rest/comment-likes/latest/projects/PROJ'
        '/repos/myrepo/pull-requests/7/comments/99/reactions/eyes'
    )
    assert kwargs['method'] == RequestMethod.PUT
