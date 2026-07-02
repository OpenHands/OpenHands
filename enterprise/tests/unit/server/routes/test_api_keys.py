"""Unit tests for API keys routes, focusing on BYOR key validation and retrieval."""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from pydantic import SecretStr
from server.auth.saas_user_auth import SaasUserAuth
from server.routes.api_keys import (
    ByorPermittedResponse,
    CurrentApiKeyResponse,
    LlmApiKeyResponse,
    check_byor_permitted,
    delete_byor_key_from_litellm,
    get_current_api_key,
    get_llm_api_key_for_byor,
)
from storage.lite_llm_manager import LiteLlmManager

from openhands.app_server.user_auth.user_auth import AuthType


class TestVerifyByorKeyInLitellm:
    """Test the verify_byor_key_in_litellm function."""

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'https://litellm.example.com')
    @patch('storage.lite_llm_manager.httpx.AsyncClient')
    async def test_verify_valid_key_returns_true(self, mock_client_class):
        """Test that a valid key (200 response) returns True."""
        # Arrange
        byor_key = 'sk-valid-key-123'
        user_id = 'user-123'
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Act
        result = await LiteLlmManager.verify_key(byor_key, user_id)

        # Assert
        assert result is True
        mock_client.get.assert_called_once_with(
            'https://litellm.example.com/v1/models',
            headers={'Authorization': f'Bearer {byor_key}'},
        )

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'https://litellm.example.com')
    @patch('storage.lite_llm_manager.httpx.AsyncClient')
    async def test_verify_invalid_key_401_returns_false(self, mock_client_class):
        """Test that an invalid key (401 response) returns False."""
        # Arrange
        byor_key = 'sk-invalid-key-123'
        user_id = 'user-123'
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Act
        result = await LiteLlmManager.verify_key(byor_key, user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'https://litellm.example.com')
    @patch('storage.lite_llm_manager.httpx.AsyncClient')
    async def test_verify_invalid_key_403_returns_false(self, mock_client_class):
        """Test that an invalid key (403 response) returns False."""
        # Arrange
        byor_key = 'sk-forbidden-key-123'
        user_id = 'user-123'
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Act
        result = await LiteLlmManager.verify_key(byor_key, user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'https://litellm.example.com')
    @patch('storage.lite_llm_manager.httpx.AsyncClient')
    async def test_verify_server_error_returns_false(self, mock_client_class):
        """Test that a server error (500) returns False to ensure key validity."""
        # Arrange
        byor_key = 'sk-key-123'
        user_id = 'user-123'
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.is_success = False
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Act
        result = await LiteLlmManager.verify_key(byor_key, user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'https://litellm.example.com')
    @patch('storage.lite_llm_manager.httpx.AsyncClient')
    async def test_verify_timeout_returns_false(self, mock_client_class):
        """Test that a timeout returns False to ensure key validity."""
        # Arrange
        byor_key = 'sk-key-123'
        user_id = 'user-123'
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get.side_effect = httpx.TimeoutException('Request timed out')
        mock_client_class.return_value = mock_client

        # Act
        result = await LiteLlmManager.verify_key(byor_key, user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'https://litellm.example.com')
    @patch('storage.lite_llm_manager.httpx.AsyncClient')
    async def test_verify_network_error_returns_false(self, mock_client_class):
        """Test that a network error returns False to ensure key validity."""
        # Arrange
        byor_key = 'sk-key-123'
        user_id = 'user-123'
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get.side_effect = httpx.NetworkError('Network error')
        mock_client_class.return_value = mock_client

        # Act
        result = await LiteLlmManager.verify_key(byor_key, user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', None)
    async def test_verify_missing_api_url_returns_false(self):
        """Test that missing LITE_LLM_API_URL returns False."""
        # Arrange
        byor_key = 'sk-key-123'
        user_id = 'user-123'

        # Act
        result = await LiteLlmManager.verify_key(byor_key, user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'https://litellm.example.com')
    async def test_verify_empty_key_returns_false(self):
        """Test that empty key returns False."""
        # Arrange
        byor_key = ''
        user_id = 'user-123'

        # Act
        result = await LiteLlmManager.verify_key(byor_key, user_id)

        # Assert
        assert result is False


class TestGetLlmApiKeyForByor:
    """Test the get_llm_api_key_for_byor endpoint."""

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    @patch('server.routes.api_keys.store_byor_key_in_db')
    @patch('server.routes.api_keys.generate_byor_key')
    @patch('server.routes.api_keys.get_byor_key_from_db')
    async def test_no_key_in_database_generates_new(
        self, mock_get_key, mock_generate_key, mock_store_key, mock_check_enabled
    ):
        """Test that when no key exists in database, a new one is generated."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        new_key = 'sk-new-generated-key'
        mock_check_enabled.return_value = True
        mock_get_key.return_value = None
        mock_generate_key.return_value = new_key
        mock_store_key.return_value = None

        # Act
        result = await get_llm_api_key_for_byor(
            user_id=user_id, effective_org_id=org_id
        )

        # Assert
        assert result == LlmApiKeyResponse(key=new_key)
        mock_check_enabled.assert_called_once_with(user_id, org_id=org_id)
        mock_get_key.assert_called_once_with(user_id, org_id)
        mock_generate_key.assert_called_once_with(user_id, org_id)
        mock_store_key.assert_called_once_with(user_id, org_id, new_key)

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    @patch('storage.lite_llm_manager.LiteLlmManager.verify_key')
    @patch('server.routes.api_keys.get_byor_key_from_db')
    async def test_valid_key_in_database_returns_key(
        self, mock_get_key, mock_verify_key, mock_check_enabled
    ):
        """Test that when a valid key exists in database, it is returned."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        existing_key = 'sk-existing-valid-key'
        mock_check_enabled.return_value = True
        mock_get_key.return_value = existing_key
        mock_verify_key.return_value = True

        # Act
        result = await get_llm_api_key_for_byor(
            user_id=user_id, effective_org_id=org_id
        )

        # Assert
        assert result == LlmApiKeyResponse(key=existing_key)
        mock_check_enabled.assert_called_once_with(user_id, org_id=org_id)
        mock_get_key.assert_called_once_with(user_id, org_id)
        mock_verify_key.assert_called_once_with(existing_key, user_id)

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    @patch('server.routes.api_keys.store_byor_key_in_db')
    @patch('server.routes.api_keys.generate_byor_key')
    @patch('server.routes.api_keys.delete_byor_key_from_litellm')
    @patch('storage.lite_llm_manager.LiteLlmManager.verify_key')
    @patch('server.routes.api_keys.get_byor_key_from_db')
    async def test_invalid_key_in_database_regenerates(
        self,
        mock_get_key,
        mock_verify_key,
        mock_delete_key,
        mock_generate_key,
        mock_store_key,
        mock_check_enabled,
    ):
        """Test that when an invalid key exists in database, it is regenerated."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        invalid_key = 'sk-invalid-key'
        new_key = 'sk-new-generated-key'
        mock_check_enabled.return_value = True
        mock_get_key.return_value = invalid_key
        mock_verify_key.return_value = False
        mock_delete_key.return_value = True
        mock_generate_key.return_value = new_key
        mock_store_key.return_value = None

        # Act
        result = await get_llm_api_key_for_byor(
            user_id=user_id, effective_org_id=org_id
        )

        # Assert
        assert result == LlmApiKeyResponse(key=new_key)
        mock_check_enabled.assert_called_once_with(user_id, org_id=org_id)
        mock_get_key.assert_called_once_with(user_id, org_id)
        mock_verify_key.assert_called_once_with(invalid_key, user_id)
        mock_delete_key.assert_called_once_with(user_id, org_id, invalid_key)
        mock_generate_key.assert_called_once_with(user_id, org_id)
        mock_store_key.assert_called_once_with(user_id, org_id, new_key)

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    @patch('server.routes.api_keys.store_byor_key_in_db')
    @patch('server.routes.api_keys.generate_byor_key')
    @patch('server.routes.api_keys.delete_byor_key_from_litellm')
    @patch('storage.lite_llm_manager.LiteLlmManager.verify_key')
    @patch('server.routes.api_keys.get_byor_key_from_db')
    async def test_invalid_key_deletion_failure_still_regenerates(
        self,
        mock_get_key,
        mock_verify_key,
        mock_delete_key,
        mock_generate_key,
        mock_store_key,
        mock_check_enabled,
    ):
        """Test that even if deletion fails, regeneration still proceeds."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        invalid_key = 'sk-invalid-key'
        new_key = 'sk-new-generated-key'
        mock_check_enabled.return_value = True
        mock_get_key.return_value = invalid_key
        mock_verify_key.return_value = False
        mock_delete_key.return_value = False  # Deletion fails
        mock_generate_key.return_value = new_key
        mock_store_key.return_value = None

        # Act
        result = await get_llm_api_key_for_byor(
            user_id=user_id, effective_org_id=org_id
        )

        # Assert
        assert result == LlmApiKeyResponse(key=new_key)
        mock_check_enabled.assert_called_once_with(user_id, org_id=org_id)
        mock_delete_key.assert_called_once_with(user_id, org_id, invalid_key)
        mock_generate_key.assert_called_once_with(user_id, org_id)
        mock_store_key.assert_called_once_with(user_id, org_id, new_key)

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    @patch('server.routes.api_keys.generate_byor_key')
    @patch('server.routes.api_keys.get_byor_key_from_db')
    async def test_key_generation_failure_raises_exception(
        self, mock_get_key, mock_generate_key, mock_check_enabled
    ):
        """Test that when key generation fails, an HTTPException is raised."""
        # Arrange
        user_id = 'user-123'
        mock_check_enabled.return_value = True
        mock_get_key.return_value = None
        mock_generate_key.return_value = None

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_llm_api_key_for_byor(user_id=user_id)

        assert exc_info.value.status_code == 500
        assert 'Failed to generate new BYOR LLM API key' in exc_info.value.detail

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    @patch('server.routes.api_keys.get_byor_key_from_db')
    async def test_database_error_raises_exception(
        self, mock_get_key, mock_check_enabled
    ):
        """Test that database errors are properly handled."""
        # Arrange
        user_id = 'user-123'
        mock_check_enabled.return_value = True
        mock_get_key.side_effect = Exception('Database connection error')

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_llm_api_key_for_byor(user_id=user_id)

        assert exc_info.value.status_code == 500
        assert 'Failed to retrieve BYOR LLM API key' in exc_info.value.detail

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    async def test_byor_export_disabled_returns_402(self, mock_check_enabled):
        """Test that when BYOR export is disabled, 402 is returned."""
        # Arrange
        user_id = 'user-123'
        mock_check_enabled.return_value = False

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_llm_api_key_for_byor(user_id=user_id)

        assert exc_info.value.status_code == 402
        assert 'BYOR key export is not enabled' in exc_info.value.detail


class TestDeleteByorKeyFromLitellm:
    """Test the delete_byor_key_from_litellm function with alias cleanup."""

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LiteLlmManager.delete_key')
    async def test_delete_constructs_alias_from_org(self, mock_delete_key):
        """Test that delete_byor_key_from_litellm builds the key alias from the effective org."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        byor_key = 'sk-byor-key-to-delete'
        expected_alias = f'BYOR Key - user {user_id}, org {org_id}'
        mock_delete_key.return_value = None

        # Act
        result = await delete_byor_key_from_litellm(user_id, org_id, byor_key)

        # Assert
        assert result is True
        mock_delete_key.assert_called_once_with(byor_key, key_alias=expected_alias)

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LiteLlmManager.delete_key')
    async def test_delete_returns_false_on_exception(self, mock_delete_key):
        """Test that exceptions during deletion return False."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        byor_key = 'sk-byor-key-to-delete'
        mock_delete_key.side_effect = Exception('LiteLLM API error')

        # Act
        result = await delete_byor_key_from_litellm(user_id, org_id, byor_key)

        # Assert
        assert result is False


class TestCheckByorPermitted:
    """Test the check_byor_permitted endpoint."""

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    async def test_permitted_when_enabled(self, mock_check_enabled):
        """Test that permitted=True is returned when BYOR export is enabled."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        mock_check_enabled.return_value = True

        # Act
        result = await check_byor_permitted(user_id=user_id, effective_org_id=org_id)

        # Assert
        assert result == ByorPermittedResponse(permitted=True)
        mock_check_enabled.assert_called_once_with(user_id, org_id=org_id)

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    async def test_not_permitted_when_disabled(self, mock_check_enabled):
        """Test that permitted=False is returned when BYOR export is disabled."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        mock_check_enabled.return_value = False

        # Act
        result = await check_byor_permitted(user_id=user_id, effective_org_id=org_id)

        # Assert
        assert result == ByorPermittedResponse(permitted=False)
        mock_check_enabled.assert_called_once_with(user_id, org_id=org_id)

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    async def test_error_raises_500(self, mock_check_enabled):
        """Test that an exception raises 500 error."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        mock_check_enabled.side_effect = Exception('Database error')

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await check_byor_permitted(user_id=user_id, effective_org_id=org_id)

        assert exc_info.value.status_code == 500
        assert 'Failed to check BYOR export permission' in exc_info.value.detail


class TestGetCurrentApiKey:
    """Test the get_current_api_key endpoint."""

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.get_user_auth')
    async def test_returns_api_key_info_for_bearer_auth(self, mock_get_user_auth):
        """Org-bound API key reports its bound org in both fields."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        mock_request = MagicMock()

        user_auth = SaasUserAuth(
            refresh_token=SecretStr('mock-token'),
            user_id=user_id,
            auth_type=AuthType.BEARER,
            api_key_org_id=org_id,
            api_key_id=42,
            api_key_name='My Production Key',
        )
        # Bound keys resolve to their bound org in all contexts.
        user_auth.get_effective_org_id = AsyncMock(return_value=org_id)
        mock_get_user_auth.return_value = user_auth

        # Act
        result = await get_current_api_key(request=mock_request, user_id=user_id)

        # Assert
        assert isinstance(result, CurrentApiKeyResponse)
        assert result.org_id == str(org_id)
        assert result.bound_org_id == str(org_id)
        assert result.id == 42
        assert result.name == 'My Production Key'
        assert result.user_id == user_id
        assert result.auth_type == 'bearer'

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.get_user_auth')
    async def test_returns_400_for_cookie_auth(self, mock_get_user_auth):
        """Test that 400 Bad Request is returned when using cookie authentication."""
        # Arrange
        user_id = 'user-123'
        mock_request = MagicMock()

        mock_user_auth = MagicMock()
        mock_user_auth.get_auth_type.return_value = AuthType.COOKIE
        mock_get_user_auth.return_value = mock_user_auth

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_current_api_key(request=mock_request, user_id=user_id)

        assert exc_info.value.status_code == 400
        assert 'API key authentication' in exc_info.value.detail

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.get_user_auth')
    async def test_unbound_key_reports_effective_org(self, mock_get_user_auth):
        """An unbound API key reports the resolved effective org and None bound."""
        # Arrange
        user_id = 'user-123'
        effective_org = uuid.uuid4()
        mock_request = MagicMock()

        user_auth = SaasUserAuth(
            refresh_token=SecretStr('mock-token'),
            user_id=user_id,
            auth_type=AuthType.BEARER,
            api_key_org_id=None,  # Unbound key
            api_key_id=42,
            api_key_name='Multi-org Key',
        )
        # Unbound keys resolve to the request's effective org (X-Org-Id or
        # user.current_org_id).
        user_auth.get_effective_org_id = AsyncMock(return_value=effective_org)
        mock_get_user_auth.return_value = user_auth

        # Act
        result = await get_current_api_key(request=mock_request, user_id=user_id)

        # Assert
        assert isinstance(result, CurrentApiKeyResponse)
        assert result.org_id == str(effective_org)
        assert result.bound_org_id is None
        assert result.id == 42
        assert result.name == 'Multi-org Key'


class TestApiKeyCreateValidation:
    """Test the ApiKeyCreate Pydantic model's validators."""

    def test_accepts_no_window(self):
        """A request with neither bound set is valid."""
        from server.routes.api_keys import ApiKeyCreate

        model = ApiKeyCreate(name='unbound-style')
        assert model.not_before is None
        assert model.expires_at is None
        assert model.org_id is None

    def test_accepts_explicit_unbound_org_id(self):
        """An explicit ``org_id=None`` is preserved as a model-field-set signal."""
        from server.routes.api_keys import ApiKeyCreate

        model = ApiKeyCreate(name='unbound-style', org_id=None)
        assert model.org_id is None
        # ``org_id`` is in ``model_fields_set`` even when set to ``None``;
        # the route uses this to distinguish "explicit unbound" from
        # "field omitted -> fall back to effective org".
        assert 'org_id' in model.model_fields_set

    def test_accepts_specific_org_id(self):
        """An explicit ``org_id=<UUID>`` binds the new key to that org."""
        from server.routes.api_keys import ApiKeyCreate

        bound_org = uuid.uuid4()
        model = ApiKeyCreate(name='bound-style', org_id=bound_org)
        assert model.org_id == bound_org

    def test_accepts_not_before_only(self):
        """A request with only not_before is valid."""
        from server.routes.api_keys import ApiKeyCreate

        future = datetime.now(UTC) + timedelta(days=1)
        model = ApiKeyCreate(name='future-key', not_before=future)
        assert model.not_before == future
        assert model.expires_at is None

    def test_accepts_expires_at_only(self):
        """A request with only expires_at is valid."""
        from server.routes.api_keys import ApiKeyCreate

        future = datetime.now(UTC) + timedelta(days=1)
        model = ApiKeyCreate(name='expiring-key', expires_at=future)
        assert model.not_before is None
        assert model.expires_at == future

    def test_accepts_valid_window(self):
        """A request with not_before < expires_at is valid."""
        from server.routes.api_keys import ApiKeyCreate

        not_before = datetime.now(UTC) + timedelta(days=1)
        expires_at = not_before + timedelta(days=30)
        model = ApiKeyCreate(
            name='window-key', not_before=not_before, expires_at=expires_at
        )
        assert model.not_before == not_before
        assert model.expires_at == expires_at

    def test_rejects_expires_at_in_past(self):
        from pydantic import ValidationError
        from server.routes.api_keys import ApiKeyCreate

        with pytest.raises(ValidationError) as exc_info:
            ApiKeyCreate(
                name='past-key',
                expires_at=datetime.now(UTC) - timedelta(days=1),
            )
        assert 'Expiration' in str(exc_info.value)

    def test_rejects_inverted_window(self):
        from pydantic import ValidationError
        from server.routes.api_keys import ApiKeyCreate

        not_before = datetime.now(UTC) + timedelta(days=10)
        expires_at = datetime.now(UTC) + timedelta(days=1)
        with pytest.raises(ValidationError) as exc_info:
            ApiKeyCreate(
                name='inverted-key',
                not_before=not_before,
                expires_at=expires_at,
            )
        assert 'not_before must be earlier than expires_at' in str(exc_info.value)

    def test_rejects_equal_window(self):
        """not_before == expires_at is rejected (degenerate window)."""
        from pydantic import ValidationError
        from server.routes.api_keys import ApiKeyCreate

        same = datetime.now(UTC) + timedelta(days=1)
        with pytest.raises(ValidationError) as exc_info:
            ApiKeyCreate(
                name='equal-key',
                not_before=same,
                expires_at=same,
            )
        assert 'not_before must be earlier than expires_at' in str(exc_info.value)


class TestCreateApiKeyRoute:
    """End-to-end tests for the ``POST /api/keys`` route."""

    @pytest.mark.asyncio
    async def test_unbound_org_id_creates_unbound_key(self):
        """Regression: ``org_id=None`` must produce an unbound (org_id NULL) row.

        The route previously delegated to ``ApiKeyStore.create_api_key``
        with ``org_id=None``; the store then silently rebound to
        ``user.current_org_id``, so the row inserted never matched the
        route's ``name + org_id is None`` lookup and the route returned
        500. The store now accepts ``use_current_org_fallback=False`` and
        the route passes that flag.
        """
        from server.routes.api_keys import ApiKeyCreate, create_api_key

        # A row the route will "find" after insert: an unbound key with
        # the matching name. This is what the (now fixed) route expects
        # ``list_api_keys`` to surface.
        matching_key = MagicMock()
        matching_key.id = 1
        matching_key.name = 'AllOrgs'
        matching_key.org_id = None
        matching_key.created_at = datetime.now(UTC)
        matching_key.last_used_at = None
        matching_key.not_before = None
        matching_key.expires_at = None

        captured: dict = {}

        async def fake_create_api_key(
            user_id,
            name,
            expires_at=None,
            not_before=None,
            org_id=...,
            **kwargs,
        ):
            # The store should be called with the route's explicit
            # ``None`` and the fallback disabled.
            captured['org_id'] = org_id
            captured['use_current_org_fallback'] = kwargs.get(
                'use_current_org_fallback'
            )
            return 'sk-oh-fake'

        with (
            patch(
                'server.routes.api_keys.api_key_store.create_api_key',
                AsyncMock(side_effect=fake_create_api_key),
            ),
            patch(
                'server.routes.api_keys.api_key_store.list_api_keys',
                AsyncMock(return_value=[matching_key]),
            ),
        ):
            result = await create_api_key(
                key_data=ApiKeyCreate(name='AllOrgs', org_id=None),
                user_id=str(uuid.uuid4()),
                effective_org_id=uuid.uuid4(),
            )

        assert captured['org_id'] is None
        assert captured['use_current_org_fallback'] is False
        assert result.org_id is None
        assert result.name == 'AllOrgs'

    @pytest.mark.asyncio
    async def test_omitted_org_id_falls_back_to_effective_org(self):
        """An *omitted* ``org_id`` binds the new key to the effective org."""
        from server.routes.api_keys import ApiKeyCreate, create_api_key

        effective_org = uuid.uuid4()
        bound_key = MagicMock()
        bound_key.id = 2
        bound_key.name = 'Bound'
        bound_key.org_id = effective_org
        bound_key.created_at = datetime.now(UTC)
        bound_key.last_used_at = None
        bound_key.not_before = None
        bound_key.expires_at = None

        captured: dict = {}

        async def fake_create_api_key(
            user_id,
            name,
            expires_at=None,
            not_before=None,
            org_id=...,
            **kwargs,
        ):
            captured['org_id'] = org_id
            captured['use_current_org_fallback'] = kwargs.get(
                'use_current_org_fallback'
            )
            return 'sk-oh-fake'

        with (
            patch(
                'server.routes.api_keys.api_key_store.create_api_key',
                AsyncMock(side_effect=fake_create_api_key),
            ),
            patch(
                'server.routes.api_keys.api_key_store.list_api_keys',
                AsyncMock(return_value=[bound_key]),
            ),
            patch(
                # Membership check on the effective org -- return a non-None
                # member so the route passes the check.
                'server.routes.api_keys.OrgMemberStore.get_org_member',
                AsyncMock(return_value=MagicMock()),
            ),
        ):
            result = await create_api_key(
                key_data=ApiKeyCreate(name='Bound'),  # no org_id
                user_id=str(uuid.uuid4()),
                effective_org_id=effective_org,
            )

        # Route forwards the effective org to the store, and disables the
        # fallback so the store doesn't double-rebind.
        assert captured['org_id'] == effective_org
        assert captured['use_current_org_fallback'] is False
        assert result.org_id == effective_org
