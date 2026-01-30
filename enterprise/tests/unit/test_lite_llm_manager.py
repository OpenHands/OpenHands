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
                        'test-org-id', 'test-user-id', mock_settings, create_user=True
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
                        'test-org-id', 'test-user-id', mock_settings, create_user=True
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
                                'test-org-id',
                                'test-user-id',
                                mock_settings,
                                create_user=False,
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
                                mock_client.post.call_count == 3
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
        # Mock response for key list
        mock_key_list_response = MagicMock()
        mock_key_list_response.is_success = True
        mock_key_list_response.status_code = 200
        mock_key_list_response.json.return_value = {
            'keys': ['test-key-1', 'test-key-2'],
            'total_count': 2,
        }
        mock_key_list_response.raise_for_status = MagicMock()

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
                            # First GET is for _get_user, second GET is for _get_user_keys
                            mock_client.get.side_effect = [
                                mock_user_response,
                                mock_key_list_response,
                            ]
                            mock_client.post.return_value = mock_response

                            # Mock verify_key to return True (key exists in LiteLLM)
                            with patch.object(
                                LiteLlmManager, 'verify_key', return_value=True
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

                            # Verify migration steps were called:
                            # - 2 GET requests: _get_user, _get_user_keys
                            # - POST requests: create_team, update_user, add_user_to_team,
                            #   and update_key for each key (2 keys)
                            assert mock_client.get.call_count == 2
                            assert (
                                mock_client.post.call_count == 5
                            )  # create_team, update_user, add_user_to_team, 2x update_key

    @pytest.mark.asyncio
    async def test_migrate_entries_generates_key_when_db_key_not_in_litellm(
        self, mock_user_settings, mock_user_response, mock_response
    ):
        """Test migrate_entries generates a new key when the DB key doesn't exist in LiteLLM."""
        # Mock response for key list
        mock_key_list_response = MagicMock()
        mock_key_list_response.is_success = True
        mock_key_list_response.status_code = 200
        mock_key_list_response.json.return_value = {
            'keys': ['test-key-1', 'test-key-2'],
            'total_count': 2,
        }
        mock_key_list_response.raise_for_status = MagicMock()

        # Mock response for key generation
        mock_generate_response = MagicMock()
        mock_generate_response.is_success = True
        mock_generate_response.status_code = 200
        mock_generate_response.json.return_value = {'key': 'new-generated-key'}
        mock_generate_response.raise_for_status = MagicMock()

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
                            # First GET is for _get_user, second GET is for _get_user_keys
                            mock_client.get.side_effect = [
                                mock_user_response,
                                mock_key_list_response,
                            ]
                            # POST responses: create_team, update_user, add_user_to_team,
                            # 2x update_key, and 1x generate_key
                            mock_client.post.side_effect = [
                                mock_response,  # create_team
                                mock_response,  # update_user
                                mock_response,  # add_user_to_team
                                mock_response,  # update_key 1
                                mock_response,  # update_key 2
                                mock_generate_response,  # generate_key
                            ]

                            # Mock verify_key to return False (key doesn't exist in LiteLLM)
                            with patch.object(
                                LiteLlmManager, 'verify_key', return_value=False
                            ):
                                result = await LiteLlmManager.migrate_entries(
                                    'test-org-id',
                                    'test-user-id',
                                    mock_user_settings,
                                )

                            # migrate_entries should update user_settings with the new key
                            assert result is not None
                            assert (
                                result.llm_api_key.get_secret_value()
                                == 'new-generated-key'
                            )
                            assert (
                                result.llm_api_key_for_byor.get_secret_value()
                                == 'new-generated-key'
                            )

                            # Verify migration steps were called including key generation:
                            # - 2 GET requests: _get_user, _get_user_keys
                            # - 6 POST requests: create_team, update_user, add_user_to_team,
                            #   2x update_key, 1x generate_key
                            assert mock_client.get.call_count == 2
                            assert mock_client.post.call_count == 6

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
    @patch('storage.lite_llm_manager.logger')
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com')
    @patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key')
    async def test_create_user_already_exists_with_409_status_code(
        self, mock_logger, mock_http_client
    ):
        """Test _create_user handles 409 Conflict when user already exists."""
        # Arrange
        first_response = MagicMock()
        first_response.is_success = False
        first_response.status_code = 400
        first_response.text = 'duplicate email'

        second_response = MagicMock()
        second_response.is_success = False
        second_response.status_code = 409
        second_response.text = 'User with id test-user-id already exists'

        mock_http_client.post.side_effect = [first_response, second_response]

        # Act
        await LiteLlmManager._create_user(
            mock_http_client, 'test@example.com', 'test-user-id'
        )

        # Assert
        mock_logger.warning.assert_any_call(
            'litellm_user_already_exists',
            extra={'user_id': 'test-user-id'},
        )

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.logger')
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com')
    @patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key')
    async def test_create_user_already_exists_with_400_status_code(
        self, mock_logger, mock_http_client
    ):
        """Test _create_user handles 400 Bad Request when user already exists."""
        # Arrange
        first_response = MagicMock()
        first_response.is_success = False
        first_response.status_code = 400
        first_response.text = 'duplicate email'

        second_response = MagicMock()
        second_response.is_success = False
        second_response.status_code = 400
        second_response.text = 'User already exists'

        mock_http_client.post.side_effect = [first_response, second_response]

        # Act
        await LiteLlmManager._create_user(
            mock_http_client, 'test@example.com', 'test-user-id'
        )

        # Assert
        mock_logger.warning.assert_any_call(
            'litellm_user_already_exists',
            extra={'user_id': 'test-user-id'},
        )

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com')
    @patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key')
    async def test_add_user_to_team_success(self, mock_http_client, mock_response):
        """Test successful _add_user_to_team operation."""
        # Arrange
        mock_http_client.post.return_value = mock_response

        # Act
        await LiteLlmManager._add_user_to_team(
            mock_http_client, 'test-user-id', 'test-team-id', 100.0
        )

        # Assert
        mock_http_client.post.assert_called_once()
        call_args = mock_http_client.post.call_args
        assert 'http://test.com/team/member_add' in call_args[0]
        assert call_args[1]['json']['team_id'] == 'test-team-id'
        assert call_args[1]['json']['member'] == {
            'user_id': 'test-user-id',
            'role': 'user',
        }
        assert call_args[1]['json']['max_budget_in_team'] == 100.0

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.logger')
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com')
    @patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key')
    async def test_add_user_to_team_already_in_team(
        self, mock_logger, mock_http_client
    ):
        """Test _add_user_to_team handles 'already in team' error gracefully."""
        # Arrange
        error_response = MagicMock()
        error_response.is_success = False
        error_response.status_code = 400
        error_response.text = (
            '{"error":{"message":"User already in team. Member: '
            'user_id=test-user-id","type":"team_member_already_in_team"}}'
        )
        mock_http_client.post.return_value = error_response

        # Act
        await LiteLlmManager._add_user_to_team(
            mock_http_client, 'test-user-id', 'test-team-id', 100.0
        )

        # Assert
        mock_logger.warning.assert_called_once_with(
            'user_already_in_team',
            extra={
                'user_id': 'test-user-id',
                'team_id': 'test-team-id',
            },
        )

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com')
    @patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key')
    async def test_add_user_to_team_other_error_raises_exception(
        self, mock_http_client
    ):
        """Test _add_user_to_team raises exception for non-'already in team' errors."""
        # Arrange
        error_response = MagicMock()
        error_response.is_success = False
        error_response.status_code = 500
        error_response.text = 'Internal server error'
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            'Server error', request=MagicMock(), response=error_response
        )
        mock_http_client.post.return_value = error_response

        # Act & Assert
        with pytest.raises(httpx.HTTPStatusError):
            await LiteLlmManager._add_user_to_team(
                mock_http_client, 'test-user-id', 'test-team-id', 100.0
            )

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com')
    @patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key')
    async def test_update_key_success(self, mock_http_client, mock_response):
        """Test successful _update_key operation."""
        # Arrange
        mock_http_client.post.return_value = mock_response

        # Act
        await LiteLlmManager._update_key(
            mock_http_client, 'test-user-id', 'test-api-key', team_id='test-team-id'
        )

        # Assert
        mock_http_client.post.assert_called_once()
        call_args = mock_http_client.post.call_args
        assert 'http://test.com/key/update' in call_args[0]
        assert call_args[1]['json']['key'] == 'test-api-key'
        assert call_args[1]['json']['team_id'] == 'test-team-id'

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.logger')
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com')
    @patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key')
    async def test_update_key_invalid_key_returns_gracefully(
        self, mock_logger, mock_http_client
    ):
        """Test _update_key handles 401 Unauthorized for invalid keys gracefully."""
        # Arrange
        error_response = MagicMock()
        error_response.is_success = False
        error_response.status_code = 401
        error_response.text = 'Unauthorized'
        mock_http_client.post.return_value = error_response

        # Act
        await LiteLlmManager._update_key(
            mock_http_client, 'test-user-id', 'invalid-api-key', team_id='test-team-id'
        )

        # Assert
        mock_logger.warning.assert_called_once_with(
            'invalid_litellm_key_during_update',
            extra={'user_id': 'test-user-id', 'text': 'Unauthorized'},
        )

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com')
    @patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key')
    async def test_update_key_other_error_raises_exception(self, mock_http_client):
        """Test _update_key raises exception for non-401 errors."""
        # Arrange
        error_response = MagicMock()
        error_response.is_success = False
        error_response.status_code = 500
        error_response.text = 'Internal server error'
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            'Server error', request=MagicMock(), response=error_response
        )
        mock_http_client.post.return_value = error_response

        # Act & Assert
        with pytest.raises(httpx.HTTPStatusError):
            await LiteLlmManager._update_key(
                mock_http_client, 'test-user-id', 'test-api-key', team_id='test-team-id'
            )

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com')
    @patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key')
    async def test_get_user_keys_success(self, mock_http_client):
        """Test successful _get_user_keys operation."""
        # Arrange
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'keys': ['key-1', 'key-2', 'key-3'],
            'total_count': 3,
        }
        mock_http_client.get.return_value = mock_response

        # Act
        keys = await LiteLlmManager._get_user_keys(mock_http_client, 'test-user-id')

        # Assert
        assert keys == ['key-1', 'key-2', 'key-3']
        mock_http_client.get.assert_called_once()
        call_args = mock_http_client.get.call_args
        assert 'http://test.com/key/list' in call_args[0]
        assert call_args[1]['params'] == {'user_id': 'test-user-id'}

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com')
    @patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key')
    async def test_get_user_keys_empty_list(self, mock_http_client):
        """Test _get_user_keys returns empty list when user has no keys."""
        # Arrange
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'keys': [],
            'total_count': 0,
        }
        mock_http_client.get.return_value = mock_response

        # Act
        keys = await LiteLlmManager._get_user_keys(mock_http_client, 'test-user-id')

        # Assert
        assert keys == []

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com')
    @patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key')
    async def test_get_user_keys_error_returns_empty_list(self, mock_http_client):
        """Test _get_user_keys returns empty list on error."""
        # Arrange
        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.status_code = 500
        mock_response.text = 'Internal server error'
        mock_http_client.get.return_value = mock_response

        # Act
        keys = await LiteLlmManager._get_user_keys(mock_http_client, 'test-user-id')

        # Assert
        assert keys == []

    @pytest.mark.asyncio
    async def test_get_user_keys_missing_config(self, mock_http_client):
        """Test _get_user_keys returns empty list when config is missing."""
        with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', None):
            with patch('storage.lite_llm_manager.LITE_LLM_API_URL', None):
                keys = await LiteLlmManager._get_user_keys(
                    mock_http_client, 'test-user-id'
                )
                assert keys == []

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com')
    @patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key')
    async def test_update_user_keys_success(self, mock_http_client, mock_response):
        """Test successful _update_user_keys operation."""
        # Arrange
        mock_key_list_response = MagicMock()
        mock_key_list_response.is_success = True
        mock_key_list_response.status_code = 200
        mock_key_list_response.json.return_value = {
            'keys': ['key-1', 'key-2'],
            'total_count': 2,
        }
        mock_http_client.get.return_value = mock_key_list_response
        mock_http_client.post.return_value = mock_response

        # Act
        await LiteLlmManager._update_user_keys(
            mock_http_client, 'test-user-id', team_id='test-team-id'
        )

        # Assert
        # Should call GET once for key list
        assert mock_http_client.get.call_count == 1
        # Should call POST twice (once for each key)
        assert mock_http_client.post.call_count == 2

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com')
    @patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key')
    async def test_update_user_keys_no_keys(self, mock_http_client, mock_response):
        """Test _update_user_keys when user has no keys."""
        # Arrange
        mock_key_list_response = MagicMock()
        mock_key_list_response.is_success = True
        mock_key_list_response.status_code = 200
        mock_key_list_response.json.return_value = {
            'keys': [],
            'total_count': 0,
        }
        mock_http_client.get.return_value = mock_key_list_response

        # Act
        await LiteLlmManager._update_user_keys(
            mock_http_client, 'test-user-id', team_id='test-team-id'
        )

        # Assert
        # Should call GET once for key list
        assert mock_http_client.get.call_count == 1
        # Should not call POST since there are no keys
        assert mock_http_client.post.call_count == 0

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
                    mock_user_store.get_user_by_id_async = AsyncMock(
                        return_value=mock_user
                    )

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
                    mock_user_store.get_user_by_id_async = AsyncMock(return_value=None)

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
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com')
    @patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key')
    async def test_delete_key_not_found_with_alias_triggers_alias_deletion(
        self, mock_http_client
    ):
        """Test _delete_key falls back to alias deletion when key_id returns 404."""
        # Arrange
        not_found_response = MagicMock()
        not_found_response.is_success = False
        not_found_response.status_code = 404
        not_found_response.text = 'Key not found'

        alias_success_response = MagicMock()
        alias_success_response.is_success = True
        alias_success_response.status_code = 200

        mock_http_client.post.side_effect = [not_found_response, alias_success_response]

        # Act
        await LiteLlmManager._delete_key(
            mock_http_client, 'test-key-id', key_alias='BYOR Key - user 123, org 456'
        )

        # Assert
        assert mock_http_client.post.call_count == 2
        first_call = mock_http_client.post.call_args_list[0]
        assert first_call[1]['json']['keys'] == ['test-key-id']
        second_call = mock_http_client.post.call_args_list[1]
        assert second_call[1]['json']['key_aliases'] == ['BYOR Key - user 123, org 456']

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com')
    @patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key')
    async def test_delete_key_not_found_without_alias_no_fallback(
        self, mock_http_client
    ):
        """Test _delete_key without alias does not attempt alias deletion on 404."""
        # Arrange
        not_found_response = MagicMock()
        not_found_response.is_success = False
        not_found_response.status_code = 404
        not_found_response.text = 'Key not found'
        mock_http_client.post.return_value = not_found_response

        # Act
        await LiteLlmManager._delete_key(mock_http_client, 'test-key-id')

        # Assert
        assert mock_http_client.post.call_count == 1

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com')
    @patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key')
    async def test_delete_key_by_alias_success(self, mock_http_client, mock_response):
        """Test successful _delete_key_by_alias operation."""
        # Arrange
        mock_http_client.post.return_value = mock_response

        # Act
        await LiteLlmManager._delete_key_by_alias(
            mock_http_client, 'BYOR Key - user 123, org 456'
        )

        # Assert
        mock_http_client.post.assert_called_once()
        call_args = mock_http_client.post.call_args
        assert 'http://test.com/key/delete' in call_args[0]
        assert call_args[1]['json']['key_aliases'] == ['BYOR Key - user 123, org 456']

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com')
    @patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key')
    async def test_delete_key_by_alias_not_found(self, mock_http_client):
        """Test _delete_key_by_alias when alias is not found (404)."""
        # Arrange
        not_found_response = MagicMock()
        not_found_response.is_success = False
        not_found_response.status_code = 404
        not_found_response.text = 'Key alias not found'
        mock_http_client.post.return_value = not_found_response

        # Act & Assert - should not raise exception for 404
        await LiteLlmManager._delete_key_by_alias(
            mock_http_client, 'BYOR Key - user 123, org 456'
        )

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.logger')
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com')
    @patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key')
    async def test_delete_key_by_alias_server_error_logs_warning(
        self, mock_logger, mock_http_client
    ):
        """Test _delete_key_by_alias logs warning for non-404 errors."""
        # Arrange
        error_response = MagicMock()
        error_response.is_success = False
        error_response.status_code = 500
        error_response.text = 'Internal server error'
        mock_http_client.post.return_value = error_response

        # Act
        await LiteLlmManager._delete_key_by_alias(
            mock_http_client, 'BYOR Key - user 123, org 456'
        )

        # Assert
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert call_args[0][0] == 'error_deleting_key_by_alias'

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', None)
    @patch('storage.lite_llm_manager.LITE_LLM_API_KEY', None)
    async def test_delete_key_by_alias_missing_config(self, mock_http_client):
        """Test _delete_key_by_alias returns early when config is missing."""
        # Act
        await LiteLlmManager._delete_key_by_alias(
            mock_http_client, 'BYOR Key - user 123, org 456'
        )

        # Assert
        mock_http_client.post.assert_not_called()

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

    @pytest.mark.asyncio
    async def test_delete_team_success(self, mock_http_client, mock_response):
        """
        GIVEN: Valid team_id and configured LiteLLM API
        WHEN: delete_team is called
        THEN: Team is deleted successfully via POST /team/delete
        """
        # Arrange
        team_id = 'test-team-123'
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_http_client.post.return_value = mock_response

        with (
            patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'),
            patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.url'),
            patch('storage.lite_llm_manager.LITE_LLM_TEAM_ID', 'test-team'),
        ):
            # Act
            await LiteLlmManager._delete_team(mock_http_client, team_id)

            # Assert
            mock_http_client.post.assert_called_once_with(
                'http://test.url/team/delete',
                json={'team_ids': [team_id]},
            )

    @pytest.mark.asyncio
    async def test_delete_team_not_found_is_idempotent(
        self, mock_http_client, mock_response
    ):
        """
        GIVEN: Team does not exist (404 response)
        WHEN: delete_team is called
        THEN: Operation succeeds without raising exception (idempotent)
        """
        # Arrange
        team_id = 'non-existent-team'
        mock_response.is_success = False
        mock_response.status_code = 404
        mock_http_client.post.return_value = mock_response

        with (
            patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'),
            patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.url'),
            patch('storage.lite_llm_manager.LITE_LLM_TEAM_ID', 'test-team'),
        ):
            # Act - should not raise
            await LiteLlmManager._delete_team(mock_http_client, team_id)

            # Assert
            mock_http_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_team_api_error_raises_exception(
        self, mock_http_client, mock_response
    ):
        """
        GIVEN: LiteLLM API returns error (non-404)
        WHEN: delete_team is called
        THEN: HTTPStatusError is raised
        """
        # Arrange
        team_id = 'test-team-123'
        mock_response.is_success = False
        mock_response.status_code = 500
        mock_response.text = 'Internal Server Error'
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                'Server error', request=MagicMock(), response=mock_response
            )
        )
        mock_http_client.post.return_value = mock_response

        with (
            patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'),
            patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.url'),
            patch('storage.lite_llm_manager.LITE_LLM_TEAM_ID', 'test-team'),
        ):
            # Act & Assert
            with pytest.raises(httpx.HTTPStatusError):
                await LiteLlmManager._delete_team(mock_http_client, team_id)

    @pytest.mark.asyncio
    async def test_delete_team_no_config_returns_early(self, mock_http_client):
        """
        GIVEN: LiteLLM API is not configured
        WHEN: delete_team is called
        THEN: Function returns early without making API call
        """
        # Arrange
        team_id = 'test-team-123'

        with (
            patch('storage.lite_llm_manager.LITE_LLM_API_KEY', None),
            patch('storage.lite_llm_manager.LITE_LLM_API_URL', None),
        ):
            # Act
            await LiteLlmManager._delete_team(mock_http_client, team_id)

            # Assert
            mock_http_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_team_public_method(self):
        """
        GIVEN: Valid team_id
        WHEN: Public delete_team method is called
        THEN: HTTP client is created and team is deleted
        """
        # Arrange
        team_id = 'test-team-123'
        mock_response = AsyncMock()
        mock_response.is_success = True
        mock_response.status_code = 200

        with (
            patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'),
            patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.url'),
            patch('storage.lite_llm_manager.LITE_LLM_TEAM_ID', 'test-team'),
            patch('httpx.AsyncClient') as mock_client_class,
        ):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Act
            await LiteLlmManager.delete_team(team_id)

            # Assert
            mock_client.post.assert_called_once_with(
                'http://test.url/team/delete',
                json={'team_ids': [team_id]},
            )

    @pytest.mark.asyncio
    async def test_remove_user_from_team_successful(self):
        """
        GIVEN: Valid user_id and team_id
        WHEN: _remove_user_from_team is called
        THEN: HTTP POST is made to remove user from team
        """
        mock_response = AsyncMock()
        mock_response.is_success = True
        mock_response.status_code = 200

        with (
            patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'),
            patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.url'),
        ):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response

            await LiteLlmManager._remove_user_from_team(
                mock_client, 'test-user-id', 'test-team-id'
            )

            mock_client.post.assert_called_once_with(
                'http://test.url/team/member_delete',
                json={
                    'team_id': 'test-team-id',
                    'user_id': 'test-user-id',
                },
            )

    @pytest.mark.asyncio
    async def test_remove_user_from_team_not_found(self):
        """
        GIVEN: User not in team
        WHEN: _remove_user_from_team is called
        THEN: 404 response is handled gracefully without raising
        """
        mock_response = AsyncMock()
        mock_response.is_success = False
        mock_response.status_code = 404
        mock_response.text = 'User not found in team'
        mock_response.raise_for_status = MagicMock()

        with (
            patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'),
            patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.url'),
        ):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response

            # Should not raise an exception
            await LiteLlmManager._remove_user_from_team(
                mock_client, 'test-user-id', 'test-team-id'
            )

    @pytest.mark.asyncio
    async def test_downgrade_entries_missing_config(self, mock_user_settings):
        """Test downgrade_entries when LiteLLM config is missing."""
        with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', None):
            with patch('storage.lite_llm_manager.LITE_LLM_API_URL', None):
                result = await LiteLlmManager.downgrade_entries(
                    'test-org-id',
                    'test-user-id',
                    mock_user_settings,
                )
                assert result is None

    @pytest.mark.asyncio
    async def test_downgrade_entries_team_not_found(self, mock_user_settings):
        """Test downgrade_entries when team is not found."""
        with patch.dict(os.environ, {'LOCAL_DEPLOYMENT': ''}):
            with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
                with patch(
                    'storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'
                ):
                    with patch.object(
                        LiteLlmManager, '_get_team', new_callable=AsyncMock
                    ) as mock_get_team:
                        mock_get_team.return_value = None

                        result = await LiteLlmManager.downgrade_entries(
                            'test-org-id',
                            'test-user-id',
                            mock_user_settings,
                        )

                        assert result is None

    @pytest.mark.asyncio
    async def test_downgrade_entries_successful(self, mock_user_settings):
        """Test successful downgrade_entries operation."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_team_info_response = MagicMock()
        mock_team_info_response.is_success = True
        mock_team_info_response.status_code = 200
        mock_team_info_response.json.return_value = {
            'team_info': {
                'max_budget': 100.0,
                'spend': 20.0,
            },
            'team_memberships': [
                {
                    'user_id': 'test-user-id',
                    'team_id': 'test-org-id',
                    'max_budget_in_team': 100.0,
                    'spend': 20.0,
                }
            ],
        }
        mock_team_info_response.raise_for_status = MagicMock()

        mock_key_list_response = MagicMock()
        mock_key_list_response.is_success = True
        mock_key_list_response.status_code = 200
        mock_key_list_response.json.return_value = {
            'keys': ['test-key-1', 'test-key-2'],
            'total_count': 2,
        }
        mock_key_list_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {'LOCAL_DEPLOYMENT': ''}):
            with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
                with patch(
                    'storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'
                ):
                    with patch(
                        'storage.lite_llm_manager.LITE_LLM_TEAM_ID', 'default-team'
                    ):
                        with patch('httpx.AsyncClient') as mock_client_class:
                            mock_client = AsyncMock()
                            mock_client_class.return_value.__aenter__.return_value = (
                                mock_client
                            )
                            # GET requests: get_team (x2 for team info), get_user_keys
                            mock_client.get.side_effect = [
                                mock_team_info_response,
                                mock_team_info_response,
                                mock_key_list_response,
                            ]
                            mock_client.post.return_value = mock_response

                            result = await LiteLlmManager.downgrade_entries(
                                'test-org-id',
                                'test-user-id',
                                mock_user_settings,
                            )

                            # downgrade_entries returns the user_settings
                            assert result is not None
                            assert result.agent == 'TestAgent'

                            # Verify downgrade steps were called:
                            # GET requests:
                            # 1. get_team (GET)
                            # 2. get_user_team_info (GET via _get_team)
                            # 3. get_user_keys (GET)
                            # POST requests:
                            # 1. update_user (POST)
                            # 2. add_user_to_team (POST)
                            # 3. update_key for key 1 (POST)
                            # 4. update_key for key 2 (POST)
                            # 5. remove_user_from_team (POST)
                            # 6. delete_team (POST)
                            assert mock_client.get.call_count == 3
                            assert mock_client.post.call_count == 6

    @pytest.mark.asyncio
    async def test_downgrade_entries_local_deployment(self, mock_user_settings):
        """Test downgrade_entries in local deployment mode (skips LiteLLM calls)."""
        with patch.dict(os.environ, {'LOCAL_DEPLOYMENT': 'true'}):
            with patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test-key'):
                with patch(
                    'storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.com'
                ):
                    result = await LiteLlmManager.downgrade_entries(
                        'test-org-id',
                        'test-user-id',
                        mock_user_settings,
                    )

                    # In local deployment, should return user_settings without
                    # making any LiteLLM calls
                    assert result is not None
                    assert result.agent == 'TestAgent'
