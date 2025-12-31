"""Unit tests for API keys routes, focusing on BYOR key validation and retrieval."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from server.routes.api_keys import (
    get_llm_api_key_for_byor,
    verify_byor_key_in_litellm,
)


class TestVerifyByorKeyInLitellm:
    """Test the verify_byor_key_in_litellm function."""

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.LITE_LLM_API_URL', 'https://litellm.example.com')
    @patch('server.routes.api_keys.httpx.AsyncClient')
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
        result = await verify_byor_key_in_litellm(byor_key, user_id)

        # Assert
        assert result is True
        mock_client.get.assert_called_once_with(
            'https://litellm.example.com/v1/models',
            headers={'Authorization': f'Bearer {byor_key}'},
        )

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.LITE_LLM_API_URL', 'https://litellm.example.com')
    @patch('server.routes.api_keys.httpx.AsyncClient')
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
        result = await verify_byor_key_in_litellm(byor_key, user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.LITE_LLM_API_URL', 'https://litellm.example.com')
    @patch('server.routes.api_keys.httpx.AsyncClient')
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
        result = await verify_byor_key_in_litellm(byor_key, user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.LITE_LLM_API_URL', 'https://litellm.example.com')
    @patch('server.routes.api_keys.httpx.AsyncClient')
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
        result = await verify_byor_key_in_litellm(byor_key, user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.LITE_LLM_API_URL', 'https://litellm.example.com')
    @patch('server.routes.api_keys.httpx.AsyncClient')
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
        result = await verify_byor_key_in_litellm(byor_key, user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.LITE_LLM_API_URL', 'https://litellm.example.com')
    @patch('server.routes.api_keys.httpx.AsyncClient')
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
        result = await verify_byor_key_in_litellm(byor_key, user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.LITE_LLM_API_URL', None)
    async def test_verify_missing_api_url_returns_false(self):
        """Test that missing LITE_LLM_API_URL returns False."""
        # Arrange
        byor_key = 'sk-key-123'
        user_id = 'user-123'

        # Act
        result = await verify_byor_key_in_litellm(byor_key, user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.LITE_LLM_API_URL', 'https://litellm.example.com')
    async def test_verify_empty_key_returns_false(self):
        """Test that empty key returns False."""
        # Arrange
        byor_key = ''
        user_id = 'user-123'

        # Act
        result = await verify_byor_key_in_litellm(byor_key, user_id)

        # Assert
        assert result is False


class TestGetLlmApiKeyForByor:
    """Test the get_llm_api_key_for_byor endpoint."""

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.store_byor_key_in_db')
    @patch('server.routes.api_keys.generate_byor_key')
    @patch('server.routes.api_keys.get_byor_key_from_db')
    async def test_no_key_in_database_generates_new(
        self, mock_get_key, mock_generate_key, mock_store_key
    ):
        """Test that when no key exists in database, a new one is generated."""
        # Arrange
        user_id = 'user-123'
        new_key = 'sk-new-generated-key'
        mock_get_key.return_value = None
        mock_generate_key.return_value = new_key
        mock_store_key.return_value = None

        # Act
        result = await get_llm_api_key_for_byor(user_id=user_id)

        # Assert
        assert result == {'key': new_key}
        mock_get_key.assert_called_once_with(user_id)
        mock_generate_key.assert_called_once_with(user_id)
        mock_store_key.assert_called_once_with(user_id, new_key)

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.verify_byor_key_in_litellm')
    @patch('server.routes.api_keys.get_byor_key_from_db')
    async def test_valid_key_in_database_returns_key(
        self, mock_get_key, mock_verify_key
    ):
        """Test that when a valid key exists in database, it is returned."""
        # Arrange
        user_id = 'user-123'
        existing_key = 'sk-existing-valid-key'
        mock_get_key.return_value = existing_key
        mock_verify_key.return_value = True

        # Act
        result = await get_llm_api_key_for_byor(user_id=user_id)

        # Assert
        assert result == {'key': existing_key}
        mock_get_key.assert_called_once_with(user_id)
        mock_verify_key.assert_called_once_with(existing_key, user_id)

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.store_byor_key_in_db')
    @patch('server.routes.api_keys.generate_byor_key')
    @patch('server.routes.api_keys.delete_byor_key_from_litellm')
    @patch('server.routes.api_keys.verify_byor_key_in_litellm')
    @patch('server.routes.api_keys.get_byor_key_from_db')
    async def test_invalid_key_in_database_regenerates(
        self,
        mock_get_key,
        mock_verify_key,
        mock_delete_key,
        mock_generate_key,
        mock_store_key,
    ):
        """Test that when an invalid key exists in database, it is regenerated."""
        # Arrange
        user_id = 'user-123'
        invalid_key = 'sk-invalid-key'
        new_key = 'sk-new-generated-key'
        mock_get_key.return_value = invalid_key
        mock_verify_key.return_value = False
        mock_delete_key.return_value = True
        mock_generate_key.return_value = new_key
        mock_store_key.return_value = None

        # Act
        result = await get_llm_api_key_for_byor(user_id=user_id)

        # Assert
        assert result == {'key': new_key}
        mock_get_key.assert_called_once_with(user_id)
        mock_verify_key.assert_called_once_with(invalid_key, user_id)
        mock_delete_key.assert_called_once_with(user_id, invalid_key)
        mock_generate_key.assert_called_once_with(user_id)
        mock_store_key.assert_called_once_with(user_id, new_key)

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.store_byor_key_in_db')
    @patch('server.routes.api_keys.generate_byor_key')
    @patch('server.routes.api_keys.delete_byor_key_from_litellm')
    @patch('server.routes.api_keys.verify_byor_key_in_litellm')
    @patch('server.routes.api_keys.get_byor_key_from_db')
    async def test_invalid_key_deletion_failure_still_regenerates(
        self,
        mock_get_key,
        mock_verify_key,
        mock_delete_key,
        mock_generate_key,
        mock_store_key,
    ):
        """Test that even if deletion fails, regeneration still proceeds."""
        # Arrange
        user_id = 'user-123'
        invalid_key = 'sk-invalid-key'
        new_key = 'sk-new-generated-key'
        mock_get_key.return_value = invalid_key
        mock_verify_key.return_value = False
        mock_delete_key.return_value = False  # Deletion fails
        mock_generate_key.return_value = new_key
        mock_store_key.return_value = None

        # Act
        result = await get_llm_api_key_for_byor(user_id=user_id)

        # Assert
        assert result == {'key': new_key}
        mock_delete_key.assert_called_once_with(user_id, invalid_key)
        mock_generate_key.assert_called_once_with(user_id)
        mock_store_key.assert_called_once_with(user_id, new_key)

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.generate_byor_key')
    @patch('server.routes.api_keys.get_byor_key_from_db')
    async def test_key_generation_failure_raises_exception(
        self, mock_get_key, mock_generate_key
    ):
        """Test that when key generation fails, an HTTPException is raised."""
        # Arrange
        user_id = 'user-123'
        mock_get_key.return_value = None
        mock_generate_key.return_value = None

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_llm_api_key_for_byor(user_id=user_id)

        assert exc_info.value.status_code == 500
        assert 'Failed to generate new BYOR LLM API key' in exc_info.value.detail

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.get_byor_key_from_db')
    async def test_database_error_raises_exception(self, mock_get_key):
        """Test that database errors are properly handled."""
        # Arrange
        user_id = 'user-123'
        mock_get_key.side_effect = Exception('Database connection error')

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_llm_api_key_for_byor(user_id=user_id)

        assert exc_info.value.status_code == 500
        assert 'Failed to retrieve BYOR LLM API key' in exc_info.value.detail
