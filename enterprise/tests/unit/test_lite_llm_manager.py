"""
Unit tests for LiteLlmManager class.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr
from server.constants import (
    get_default_litellm_model,
)
from storage.lite_llm_manager import LiteLlmManager
from storage.user_settings import UserSettings

from openhands.server.settings import Settings


class TestLiteLlmManager:
    """Test cases for LiteLlmManager class."""

    @pytest.fixture
    def mock_settings(self):
        """Create a mock Settings object."""
        settings = Settings()
        settings.agent = 'TestAgent'
        settings.llm_model = 'test-model'
        settings.llm_api_key = SecretStr('test-key')
        settings.llm_base_url = 'http://test.com'
        return settings

    @pytest.fixture
    def mock_user_settings(self):
        """Create a mock UserSettings object."""
        user_settings = UserSettings()
        user_settings.agent = 'TestAgent'
        user_settings.llm_model = 'test-model'
        user_settings.llm_api_key = SecretStr('test-key')
        user_settings.llm_base_url = 'http://test.com'
        user_settings.user_version = 4  # Set version to avoid None comparison
        return user_settings

    @pytest.fixture
    def mock_http_client(self):
        """Create a mock HTTP client."""
        client = AsyncMock(spec=httpx.AsyncClient)
        return client

    @pytest.fixture
    def mock_response(self):
        """Create a mock HTTP response."""
        response = MagicMock()
        response.is_success = True
        response.status_code = 200
        response.text = 'Success'
        response.json.return_value = {'key': 'test-api-key'}
        response.raise_for_status = MagicMock()
        return response

    @pytest.fixture
    def mock_team_response(self):
        """Create a mock team response."""
        response = MagicMock()
        response.is_success = True
        response.status_code = 200
        response.json.return_value = {
            'team_memberships': [
                {
                    'user_id': 'test-user-id',
                    'team_id': 'test-org-id',
                    'max_budget': 100.0,
                }
            ]
        }
        response.raise_for_status = MagicMock()
        return response

    @pytest.fixture
    def mock_user_response(self):
        """Create a mock user response."""
        response = MagicMock()
        response.is_success = True
        response.status_code = 200
        response.json.return_value = {
            'user_info': {
                'max_budget': 50.0,
                'spend': 10.0,
            }
        }
        response.raise_for_status = MagicMock()
        return response

    @pytest.fixture
    def mock_key_info_response(self):
        """Create a mock key info response."""
        response = MagicMock()
        response.is_success = True
        response.status_code = 200
        response.json.return_value = {
            'info': {
                'max_budget': 100.0,
                'spend': 25.0,
            }
        }
        response.raise_for_status = MagicMock()
        return response

    @pytest.mark.asyncio
    async def test_create_entries_missing_config(self, mock_settings):
        """Test create_entries when LiteLLM config is missing."""
        with patch.dict(os.environ, {'LITE_LLM_API_KEY': '', 'LITE_LLM_API_URL': ''}):
            with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', None):
                with patch('storage.lite_llm_manager.LITE_LLM_API_URL', None):
                    result = await LiteLlmManager.create_entries(
                        'test-org-id', 'test-user-id', mock_settings
                    )
                    assert result is None

    @pytest.mark.asyncio
    async def test_create_entries_local_deployment(self, mock_settings):
        """Test create_entries in local deployment mode."""
        with patch.dict(os.environ, {'LOCAL_DEPLOYMENT': '1'}):
            with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
                with patch(
                    'storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'
                ):
                    result = await LiteLlmManager.create_entries(
                        'test-org-id', 'test-user-id', mock_settings
                    )

                    assert result is not None
                    assert result.agent == 'CodeActAgent'
                    assert result.llm_model == get_default_litellm_model()
                    assert result.llm_api_key.get_secret_value() == 'test-key'
                    assert result.llm_base_url == 'http://test.com'

    @pytest.mark.asyncio
    async def test_create_entries_cloud_deployment(self, mock_settings, mock_response):
        """Test create_entries in cloud deployment mode."""
        with patch.dict(os.environ, {'LOCAL_DEPLOYMENT': ''}):
            with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
                with patch(
                    'storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'
                ):
                    with patch(
                        'storage.lite_llm_manager.TokenManager'
                    ) as mock_token_manager:
                        mock_token_manager.return_value.get_user_info_from_user_id = (
                            AsyncMock(return_value={'email': 'test@example.com'})
                        )

                        with patch('httpx.AsyncClient') as mock_client_class:
                            mock_client = AsyncMock()
                            mock_client_class.return_value.__aenter__.return_value = (
                                mock_client
                            )
                            mock_client.post.return_value = mock_response

                            result = await LiteLlmManager.create_entries(
                                'test-org-id', 'test-user-id', mock_settings
                            )

                            assert result is not None
                            assert result.agent == 'CodeActAgent'
                            assert result.llm_model == get_default_litellm_model()
                            assert (
                                result.llm_api_key.get_secret_value() == 'test-api-key'
                            )
                            assert result.llm_base_url == 'http://test.com'

                            # Verify API calls were made
                            assert (
                                mock_client.post.call_count == 4
                            )  # create_team, create_user, add_user_to_team, generate_key

    @pytest.mark.asyncio
    async def test_migrate_entries_missing_config(self, mock_user_settings):
        """Test migrate_entries when LiteLLM config is missing."""
        with patch.dict(os.environ, {'LITE_LLM_API_KEY': '', 'LITE_LLM_API_URL': ''}):
            with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', None):
                with patch('storage.lite_llm_manager.LITE_LLM_API_URL', None):
                    result = await LiteLlmManager.migrate_entries(
                        'test-org-id',
                        'test-user-id',
                        mock_user_settings,
                    )
                    assert result is None

    @pytest.mark.asyncio
    async def test_migrate_entries_local_deployment(self, mock_user_settings):
        """Test migrate_entries in local deployment mode."""
        with patch.dict(os.environ, {'LOCAL_DEPLOYMENT': '1'}):
            with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
                with patch(
                    'storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'
                ):
                    result = await LiteLlmManager.migrate_entries(
                        'test-org-id',
                        'test-user-id',
                        mock_user_settings,
                    )

                    # migrate_entries returns the user_settings unchanged
                    assert result is not None
                    assert result.agent == 'TestAgent'
                    assert result.llm_model == 'test-model'
                    assert result.llm_api_key.get_secret_value() == 'test-key'
                    assert result.llm_base_url == 'http://test.com'

    @pytest.mark.asyncio
    async def test_migrate_entries_no_user_found(self, mock_user_settings):
        """Test migrate_entries when user is not found."""
        with patch.dict(os.environ, {'LOCAL_DEPLOYMENT': ''}):
            with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
                with patch(
                    'storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'
                ):
                    with patch(
                        'storage.lite_llm_manager.TokenManager'
                    ) as mock_token_manager:
                        mock_token_manager.return_value.get_user_info_from_user_id = (
                            AsyncMock(return_value={'email': 'test@example.com'})
                        )

                        # Mock the _get_user method directly to return None
                        with patch.object(
                            LiteLlmManager, '_get_user', new_callable=AsyncMock
                        ) as mock_get_user:
                            mock_get_user.return_value = None

                            result = await LiteLlmManager.migrate_entries(
                                'test-org-id',
                                'test-user-id',
                                mock_user_settings,
                            )

                            assert result is None

    @pytest.mark.asyncio
    async def test_migrate_entries_already_migrated(
        self, mock_user_settings, mock_user_response
    ):
        """Test migrate_entries when user is already migrated (no max_budget)."""
        mock_user_response.json.return_value = {
            'user_info': {
                'max_budget': None,  # Already migrated
                'spend': 10.0,
            }
        }

        with patch.dict(os.environ, {'LOCAL_DEPLOYMENT': ''}):
            with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
                with patch(
                    'storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'
                ):
                    with patch(
                        'storage.lite_llm_manager.TokenManager'
                    ) as mock_token_manager:
                        mock_token_manager.return_value.get_user_info_from_user_id = (
                            AsyncMock(return_value={'email': 'test@example.com'})
                        )

                        with patch('httpx.AsyncClient') as mock_client_class:
                            mock_client = AsyncMock()
                            mock_client_class.return_value.__aenter__.return_value = (
                                mock_client
                            )
                            mock_client.get.return_value = mock_user_response

                            result = await LiteLlmManager.migrate_entries(
                                'test-org-id',
                                'test-user-id',
                                mock_user_settings,
                            )

                            assert result is None

    @pytest.mark.asyncio
    async def test_migrate_entries_successful_migration(
        self, mock_user_settings, mock_user_response, mock_response
    ):
        """Test successful migrate_entries operation."""
        with patch.dict(os.environ, {'LOCAL_DEPLOYMENT': ''}):
            with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
                with patch(
                    'storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'
                ):
                    with patch(
                        'storage.lite_llm_manager.TokenManager'
                    ) as mock_token_manager:
                        mock_token_manager.return_value.get_user_info_from_user_id = (
                            AsyncMock(return_value={'email': 'test@example.com'})
                        )

                        with patch('httpx.AsyncClient') as mock_client_class:
                            mock_client = AsyncMock()
                            mock_client_class.return_value.__aenter__.return_value = (
                                mock_client
                            )
                            mock_client.get.return_value = mock_user_response
                            mock_client.post.return_value = mock_response

                            result = await LiteLlmManager.migrate_entries(
                                'test-org-id',
                                'test-user-id',
                                mock_user_settings,
                            )

                            # migrate_entries returns the user_settings unchanged
                            assert result is not None
                            assert result.agent == 'TestAgent'
                            assert result.llm_model == 'test-model'
                            assert result.llm_api_key.get_secret_value() == 'test-key'
                            assert result.llm_base_url == 'http://test.com'

                            # Verify migration steps were called
                            assert (
                                mock_client.post.call_count == 4
                            )  # create_team, update_user, add_user_to_team, update_key

    @pytest.mark.asyncio
    async def test_update_team_and_users_budget_missing_config(self):
        """Test update_team_and_users_budget when LiteLLM config is missing."""
        with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', None):
            with patch('storage.lite_llm_manager.LITE_LLM_API_URL', None):
                # Should not raise an exception, just return early
                await LiteLlmManager.update_team_and_users_budget('test-team-id', 100.0)

    @pytest.mark.asyncio
    async def test_update_team_and_users_budget_successful(
        self, mock_team_response, mock_response
    ):
        """Test successful update_team_and_users_budget operation."""
        with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
            with patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'):
                with patch('httpx.AsyncClient') as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client_class.return_value.__aenter__.return_value = mock_client
                    mock_client.post.return_value = mock_response
                    mock_client.get.return_value = mock_team_response

                    await LiteLlmManager.update_team_and_users_budget(
                        'test-team-id', 100.0
                    )

                    # Verify update_team and update_user_in_team were called
                    assert (
                        mock_client.post.call_count == 2
                    )  # update_team, update_user_in_team

    @pytest.mark.asyncio
    async def test_create_team_success(self, mock_http_client, mock_response):
        """Test successful _create_team operation."""
        mock_http_client.post.return_value = mock_response

        with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
            with patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'):
                await LiteLlmManager._create_team(
                    mock_http_client, 'test-alias', 'test-team-id', 100.0
                )

                mock_http_client.post.assert_called_once()
                call_args = mock_http_client.post.call_args
                assert 'http://test.com/team/new' in call_args[0]
                assert call_args[1]['json']['team_id'] == 'test-team-id'
                assert call_args[1]['json']['team_alias'] == 'test-alias'
                assert call_args[1]['json']['max_budget'] == 100.0

    @pytest.mark.asyncio
    async def test_create_team_already_exists(self, mock_http_client):
        """Test _create_team when team already exists."""
        error_response = MagicMock()
        error_response.is_success = False
        error_response.status_code = 400
        error_response.text = 'Team already exists. Please use a different team id'
        mock_http_client.post.return_value = error_response

        with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
            with patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'):
                with patch.object(
                    LiteLlmManager, '_update_team', new_callable=AsyncMock
                ) as mock_update:
                    await LiteLlmManager._create_team(
                        mock_http_client, 'test-alias', 'test-team-id', 100.0
                    )

                    mock_update.assert_called_once_with(
                        mock_http_client, 'test-team-id', 'test-alias', 100.0
                    )

    @pytest.mark.asyncio
    async def test_create_team_error(self, mock_http_client):
        """Test _create_team with unexpected error."""
        error_response = MagicMock()
        error_response.is_success = False
        error_response.status_code = 500
        error_response.text = 'Internal server error'
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            'Server error', request=MagicMock(), response=error_response
        )
        mock_http_client.post.return_value = error_response

        with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
            with patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'):
                with pytest.raises(httpx.HTTPStatusError):
                    await LiteLlmManager._create_team(
                        mock_http_client, 'test-alias', 'test-team-id', 100.0
                    )

    @pytest.mark.asyncio
    async def test_get_team_success(self, mock_http_client, mock_team_response):
        """Test successful _get_team operation."""
        mock_http_client.get.return_value = mock_team_response

        with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
            with patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'):
                result = await LiteLlmManager._get_team(
                    mock_http_client, 'test-team-id'
                )

                assert result is not None
                assert 'team_memberships' in result
                mock_http_client.get.assert_called_once_with(
                    'http://test.com/team/info?team_id=test-team-id'
                )

    @pytest.mark.asyncio
    async def test_create_user_success(self, mock_http_client, mock_response):
        """Test successful _create_user operation."""
        mock_http_client.post.return_value = mock_response

        with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
            with patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'):
                await LiteLlmManager._create_user(
                    mock_http_client, 'test@example.com', 'test-user-id'
                )

                mock_http_client.post.assert_called_once()
                call_args = mock_http_client.post.call_args
                assert 'http://test.com/user/new' in call_args[0]
                assert call_args[1]['json']['user_email'] == 'test@example.com'
                assert call_args[1]['json']['user_id'] == 'test-user-id'

    @pytest.mark.asyncio
    async def test_create_user_duplicate_email(self, mock_http_client, mock_response):
        """Test _create_user with duplicate email handling."""
        # First call fails with duplicate email
        error_response = MagicMock()
        error_response.is_success = False
        error_response.status_code = 400
        error_response.text = 'duplicate email'

        # Second call succeeds
        mock_http_client.post.side_effect = [error_response, mock_response]

        with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
            with patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'):
                await LiteLlmManager._create_user(
                    mock_http_client, 'test@example.com', 'test-user-id'
                )

                assert mock_http_client.post.call_count == 2
                # Second call should have None email
                second_call_args = mock_http_client.post.call_args_list[1]
                assert second_call_args[1]['json']['user_email'] is None

    @pytest.mark.asyncio
    async def test_generate_key_success(self, mock_http_client, mock_response):
        """Test successful _generate_key operation."""
        mock_http_client.post.return_value = mock_response

        with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
            with patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'):
                result = await LiteLlmManager._generate_key(
                    mock_http_client,
                    'test-user-id',
                    'test-team-id',
                    'test-alias',
                    {'test': 'metadata'},
                )

                assert result == 'test-api-key'
                mock_http_client.post.assert_called_once()
                call_args = mock_http_client.post.call_args
                assert 'http://test.com/key/generate' in call_args[0]
                assert call_args[1]['json']['user_id'] == 'test-user-id'
                assert call_args[1]['json']['team_id'] == 'test-team-id'
                assert call_args[1]['json']['key_alias'] == 'test-alias'
                assert call_args[1]['json']['metadata'] == {'test': 'metadata'}

    @pytest.mark.asyncio
    async def test_get_key_info_success(self, mock_http_client, mock_key_info_response):
        """Test successful _get_key_info operation."""
        mock_http_client.get.return_value = mock_key_info_response

        with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
            with patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'):
                with patch('storage.user_store.UserStore') as mock_user_store:
                    # Mock user with org member
                    mock_user = MagicMock()
                    mock_org_member = MagicMock()
                    mock_org_member.org_id = 'test-ord-id'
                    mock_org_member.llm_api_key = 'test-api-key'
                    mock_user.org_members = [mock_org_member]
                    mock_user_store.get_user_by_id.return_value = mock_user

                    result = await LiteLlmManager._get_key_info(
                        mock_http_client, 'test-ord-id', 'test-user-id'
                    )

                    assert result is not None
                    assert result['key_max_budget'] == 100.0
                    assert result['key_spend'] == 25.0

    @pytest.mark.asyncio
    async def test_get_key_info_no_user(self, mock_http_client):
        """Test _get_key_info when user is not found."""
        with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
            with patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'):
                with patch('storage.user_store.UserStore') as mock_user_store:
                    mock_user_store.get_user_by_id.return_value = None

                    result = await LiteLlmManager._get_key_info(
                        mock_http_client, 'test-ord-id', 'test-user-id'
                    )

                    assert result == {}

    @pytest.mark.asyncio
    async def test_delete_key_success(self, mock_http_client, mock_response):
        """Test successful _delete_key operation."""
        mock_http_client.post.return_value = mock_response

        with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
            with patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'):
                await LiteLlmManager._delete_key(mock_http_client, 'test-key-id')

                mock_http_client.post.assert_called_once()
                call_args = mock_http_client.post.call_args
                assert 'http://test.com/key/delete' in call_args[0]
                assert call_args[1]['json']['keys'] == ['test-key-id']

    @pytest.mark.asyncio
    async def test_delete_key_not_found(self, mock_http_client):
        """Test _delete_key when key is not found (404 error)."""
        error_response = MagicMock()
        error_response.is_success = False
        error_response.status_code = 404
        error_response.text = 'Key not found'
        mock_http_client.post.return_value = error_response

        with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
            with patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'):
                # Should not raise an exception for 404
                await LiteLlmManager._delete_key(mock_http_client, 'test-key-id')

    @pytest.mark.asyncio
    async def test_with_http_client_decorator(self):
        """Test the with_http_client decorator functionality."""

        # Create a mock internal function
        async def mock_internal_fn(client, arg1, arg2, kwarg1=None):
            return f'client={type(client).__name__}, arg1={arg1}, arg2={arg2}, kwarg1={kwarg1}'

        # Apply the decorator
        decorated_fn = LiteLlmManager.with_http_client(mock_internal_fn)

        with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value.__aenter__.return_value = mock_client

                result = await decorated_fn('test1', 'test2', kwarg1='test3')

                # Verify the client was injected as the first argument
                assert 'client=AsyncMock' in result
                assert 'arg1=test1' in result
                assert 'arg2=test2' in result
                assert 'kwarg1=test3' in result

    def test_public_methods_exist(self):
        """Test that all public wrapper methods exist and are properly decorated."""
        public_methods = [
            'create_team',
            'get_team',
            'update_team',
            'create_user',
            'get_user',
            'update_user',
            'delete_user',
            'add_user_to_team',
            'get_user_team_info',
            'update_user_in_team',
            'generate_key',
            'get_key_info',
            'delete_key',
        ]

        for method_name in public_methods:
            assert hasattr(LiteLlmManager, method_name)
            method = getattr(LiteLlmManager, method_name)
            assert callable(method)
            # The methods are created by the with_http_client decorator, so they're functions
            # We can verify they exist and are callable, which is the important part

    @pytest.mark.asyncio
    async def test_error_handling_missing_config_all_methods(self):
        """Test that all methods handle missing configuration gracefully."""
        with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', None):
            with patch('storage.lite_llm_manager.LITE_LLM_API_URL', None):
                mock_client = AsyncMock()

                # Test all private methods that check for config
                await LiteLlmManager._create_team(
                    mock_client, 'alias', 'team_id', 100.0
                )
                await LiteLlmManager._update_team(
                    mock_client, 'team_id', 'alias', 100.0
                )
                await LiteLlmManager._create_user(mock_client, 'email', 'user_id')
                await LiteLlmManager._update_user(mock_client, 'user_id')
                await LiteLlmManager._delete_user(mock_client, 'user_id')
                await LiteLlmManager._add_user_to_team(
                    mock_client, 'user_id', 'team_id', 100.0
                )
                await LiteLlmManager._update_user_in_team(
                    mock_client, 'user_id', 'team_id', 100.0
                )
                await LiteLlmManager._delete_key(mock_client, 'key_id')

                result1 = await LiteLlmManager._get_team(mock_client, 'team_id')
                result2 = await LiteLlmManager._get_user(mock_client, 'user_id')
                result3 = await LiteLlmManager._generate_key(
                    mock_client, 'user_id', 'team_id', 'alias', {}
                )
                result4 = await LiteLlmManager._get_user_team_info(
                    mock_client, 'user_id', 'team_id'
                )
                result5 = await LiteLlmManager._get_key_info(
                    mock_client, 'test-ord-id', 'user_id'
                )

                # Methods that return None when config is missing
                assert result1 is None
                assert result2 is None
                assert result3 is None
                assert result4 is None
                assert result5 is None

                # Verify no HTTP calls were made
                mock_client.get.assert_not_called()
                mock_client.post.assert_not_called()
