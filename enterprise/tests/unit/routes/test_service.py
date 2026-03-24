"""Unit tests for service API routes."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from server.routes.service import (
    CreateUserApiKeyRequest,
    delete_user_api_key,
    get_or_create_api_key_for_user,
    validate_service_api_key,
)


class TestValidateServiceApiKey:
    """Test cases for validate_service_api_key."""

    @pytest.mark.asyncio
    async def test_valid_service_key(self):
        """Test validation with valid service API key."""
        with patch('server.routes.service.AUTOMATIONS_SERVICE_KEY', 'test-service-key'):
            result = await validate_service_api_key('test-service-key')
        assert result == 'automations-service'

    @pytest.mark.asyncio
    async def test_missing_service_key(self):
        """Test validation with missing service API key header."""
        with patch('server.routes.service.AUTOMATIONS_SERVICE_KEY', 'test-service-key'):
            with pytest.raises(HTTPException) as exc_info:
                await validate_service_api_key(None)
        assert exc_info.value.status_code == 401
        assert 'X-Service-API-Key header is required' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_invalid_service_key(self):
        """Test validation with invalid service API key."""
        with patch('server.routes.service.AUTOMATIONS_SERVICE_KEY', 'test-service-key'):
            with pytest.raises(HTTPException) as exc_info:
                await validate_service_api_key('wrong-key')
        assert exc_info.value.status_code == 401
        assert 'Invalid service API key' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_service_auth_not_configured(self):
        """Test validation when service auth is not configured."""
        with patch('server.routes.service.AUTOMATIONS_SERVICE_KEY', ''):
            with pytest.raises(HTTPException) as exc_info:
                await validate_service_api_key('any-key')
        assert exc_info.value.status_code == 503
        assert 'Service authentication not configured' in exc_info.value.detail


class TestCreateUserApiKeyRequest:
    """Test cases for CreateUserApiKeyRequest validation."""

    def test_valid_request(self):
        """Test valid request with all fields."""
        request = CreateUserApiKeyRequest(
            name='automation',
        )
        assert request.name == 'automation'

    def test_name_is_required(self):
        """Test that name field is required."""
        with pytest.raises(ValueError):
            CreateUserApiKeyRequest(
                name='',  # Empty name should fail
            )

    def test_name_is_stripped(self):
        """Test that name field is stripped of whitespace."""
        request = CreateUserApiKeyRequest(
            name='  automation  ',
        )
        assert request.name == 'automation'

    def test_whitespace_only_name_fails(self):
        """Test that whitespace-only name fails validation."""
        with pytest.raises(ValueError):
            CreateUserApiKeyRequest(
                name='   ',
            )


class TestGetOrCreateApiKeyForUser:
    """Test cases for get_or_create_api_key_for_user endpoint."""

    @pytest.fixture
    def valid_user_id(self):
        """Return a valid user ID."""
        return '5594c7b6-f959-4b81-92e9-b09c206f5081'

    @pytest.fixture
    def valid_org_id(self):
        """Return a valid org ID."""
        return uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5081')

    @pytest.fixture
    def valid_request(self):
        """Create a valid request object."""
        return CreateUserApiKeyRequest(
            name='automation',
        )

    @pytest.mark.asyncio
    async def test_user_not_found(self, valid_user_id, valid_org_id, valid_request):
        """Test error when user doesn't exist."""
        with patch('server.routes.service.AUTOMATIONS_SERVICE_KEY', 'test-key'):
            with patch(
                'server.routes.service.UserStore.get_user_by_id', new_callable=AsyncMock
            ) as mock_get_user:
                mock_get_user.return_value = None
                with pytest.raises(HTTPException) as exc_info:
                    await get_or_create_api_key_for_user(
                        user_id=valid_user_id,
                        org_id=valid_org_id,
                        request=valid_request,
                        x_service_api_key='test-key',
                    )
        assert exc_info.value.status_code == 404
        assert 'not found' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_user_not_in_org(self, valid_user_id, valid_org_id, valid_request):
        """Test error when user is not a member of the org."""
        mock_user = MagicMock()

        with patch('server.routes.service.AUTOMATIONS_SERVICE_KEY', 'test-key'):
            with patch(
                'server.routes.service.UserStore.get_user_by_id', new_callable=AsyncMock
            ) as mock_get_user:
                with patch(
                    'server.routes.service.OrgMemberStore.get_org_member',
                    new_callable=AsyncMock,
                ) as mock_get_member:
                    mock_get_user.return_value = mock_user
                    mock_get_member.return_value = None
                    with pytest.raises(HTTPException) as exc_info:
                        await get_or_create_api_key_for_user(
                            user_id=valid_user_id,
                            org_id=valid_org_id,
                            request=valid_request,
                            x_service_api_key='test-key',
                        )
        assert exc_info.value.status_code == 403
        assert 'not a member of org' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_successful_key_creation(
        self, valid_user_id, valid_org_id, valid_request
    ):
        """Test successful API key creation."""
        mock_user = MagicMock()
        mock_org_member = MagicMock()
        mock_api_key_store = MagicMock()
        mock_api_key_store.get_or_create_system_api_key = AsyncMock(
            return_value='sk-oh-test-key-12345678901234567890'
        )

        with patch('server.routes.service.AUTOMATIONS_SERVICE_KEY', 'test-key'):
            with patch(
                'server.routes.service.UserStore.get_user_by_id', new_callable=AsyncMock
            ) as mock_get_user:
                with patch(
                    'server.routes.service.OrgMemberStore.get_org_member',
                    new_callable=AsyncMock,
                ) as mock_get_member:
                    with patch(
                        'server.routes.service.ApiKeyStore.get_instance'
                    ) as mock_get_store:
                        mock_get_user.return_value = mock_user
                        mock_get_member.return_value = mock_org_member
                        mock_get_store.return_value = mock_api_key_store

                        response = await get_or_create_api_key_for_user(
                            user_id=valid_user_id,
                            org_id=valid_org_id,
                            request=valid_request,
                            x_service_api_key='test-key',
                        )

        assert response.key == 'sk-oh-test-key-12345678901234567890'
        assert response.user_id == valid_user_id
        assert response.org_id == str(valid_org_id)
        assert response.name == 'automation'

        # Verify the store was called with correct arguments
        mock_api_key_store.get_or_create_system_api_key.assert_called_once_with(
            user_id=valid_user_id,
            org_id=valid_org_id,
            name='automation',
        )

    @pytest.mark.asyncio
    async def test_store_exception_handling(
        self, valid_user_id, valid_org_id, valid_request
    ):
        """Test error handling when store raises exception."""
        mock_user = MagicMock()
        mock_org_member = MagicMock()
        mock_api_key_store = MagicMock()
        mock_api_key_store.get_or_create_system_api_key = AsyncMock(
            side_effect=Exception('Database error')
        )

        with patch('server.routes.service.AUTOMATIONS_SERVICE_KEY', 'test-key'):
            with patch(
                'server.routes.service.UserStore.get_user_by_id', new_callable=AsyncMock
            ) as mock_get_user:
                with patch(
                    'server.routes.service.OrgMemberStore.get_org_member',
                    new_callable=AsyncMock,
                ) as mock_get_member:
                    with patch(
                        'server.routes.service.ApiKeyStore.get_instance'
                    ) as mock_get_store:
                        mock_get_user.return_value = mock_user
                        mock_get_member.return_value = mock_org_member
                        mock_get_store.return_value = mock_api_key_store

                        with pytest.raises(HTTPException) as exc_info:
                            await get_or_create_api_key_for_user(
                                user_id=valid_user_id,
                                org_id=valid_org_id,
                                request=valid_request,
                                x_service_api_key='test-key',
                            )

        assert exc_info.value.status_code == 500
        assert 'Failed to get or create API key' in exc_info.value.detail


class TestDeleteUserApiKey:
    """Test cases for delete_user_api_key endpoint."""

    @pytest.fixture
    def valid_org_id(self):
        """Return a valid org ID."""
        return uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5081')

    @pytest.mark.asyncio
    async def test_successful_delete(self, valid_org_id):
        """Test successful deletion of a system API key."""
        mock_api_key_store = MagicMock()
        mock_api_key_store.make_system_key_name.return_value = '__SYSTEM__:automation'
        mock_api_key_store.delete_api_key_by_name = AsyncMock(return_value=True)

        with patch('server.routes.service.AUTOMATIONS_SERVICE_KEY', 'test-key'):
            with patch(
                'server.routes.service.ApiKeyStore.get_instance'
            ) as mock_get_store:
                mock_get_store.return_value = mock_api_key_store

                response = await delete_user_api_key(
                    user_id='user-123',
                    org_id=valid_org_id,
                    key_name='automation',
                    x_service_api_key='test-key',
                )

        assert response == {'message': 'API key deleted successfully'}

        # Verify the store was called with correct arguments
        mock_api_key_store.make_system_key_name.assert_called_once_with('automation')
        mock_api_key_store.delete_api_key_by_name.assert_called_once_with(
            user_id='user-123',
            org_id=valid_org_id,
            name='__SYSTEM__:automation',
            allow_system=True,
        )

    @pytest.mark.asyncio
    async def test_delete_key_not_found(self, valid_org_id):
        """Test error when key to delete is not found."""
        mock_api_key_store = MagicMock()
        mock_api_key_store.make_system_key_name.return_value = '__SYSTEM__:nonexistent'
        mock_api_key_store.delete_api_key_by_name = AsyncMock(return_value=False)

        with patch('server.routes.service.AUTOMATIONS_SERVICE_KEY', 'test-key'):
            with patch(
                'server.routes.service.ApiKeyStore.get_instance'
            ) as mock_get_store:
                mock_get_store.return_value = mock_api_key_store

                with pytest.raises(HTTPException) as exc_info:
                    await delete_user_api_key(
                        user_id='user-123',
                        org_id=valid_org_id,
                        key_name='nonexistent',
                        x_service_api_key='test-key',
                    )

        assert exc_info.value.status_code == 404
        assert 'not found' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_delete_invalid_service_key(self, valid_org_id):
        """Test error when service API key is invalid."""
        with patch('server.routes.service.AUTOMATIONS_SERVICE_KEY', 'test-key'):
            with pytest.raises(HTTPException) as exc_info:
                await delete_user_api_key(
                    user_id='user-123',
                    org_id=valid_org_id,
                    key_name='automation',
                    x_service_api_key='wrong-key',
                )

        assert exc_info.value.status_code == 401
        assert 'Invalid service API key' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_delete_missing_service_key(self, valid_org_id):
        """Test error when service API key header is missing."""
        with patch('server.routes.service.AUTOMATIONS_SERVICE_KEY', 'test-key'):
            with pytest.raises(HTTPException) as exc_info:
                await delete_user_api_key(
                    user_id='user-123',
                    org_id=valid_org_id,
                    key_name='automation',
                    x_service_api_key=None,
                )

        assert exc_info.value.status_code == 401
        assert 'X-Service-API-Key header is required' in exc_info.value.detail
