"""Unit tests for the methods in LiveStatusAppConversationService."""

import io
import json
import os
import zipfile
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr

from openhands.agent_server.models import (
    SendMessageRequest,
    StartConversationRequest,
)
from openhands.app_server.app_conversation.app_conversation_models import (
    AgentType,
    AppConversationInfo,
    AppConversationStartRequest,
)
from openhands.app_server.app_conversation.live_status_app_conversation_service import (
    LiveStatusAppConversationService,
)
from openhands.app_server.sandbox.sandbox_models import (
    AGENT_SERVER,
    ExposedUrl,
    SandboxInfo,
    SandboxStatus,
)
from openhands.app_server.sandbox.sandbox_spec_models import SandboxSpecInfo
from openhands.app_server.user.user_context import UserContext
from openhands.integrations.provider import ProviderToken, ProviderType
from openhands.sdk import Agent, Event
from openhands.sdk.llm import LLM
from openhands.sdk.secret import LookupSecret, StaticSecret
from openhands.sdk.workspace import LocalWorkspace
from openhands.sdk.workspace.remote.async_remote_workspace import AsyncRemoteWorkspace
from openhands.server.types import AppMode

# Env var used by openhands SDK LLM to skip context-window validation (e.g. for gpt-4 in tests)
_ALLOW_SHORT_CONTEXT_WINDOWS = 'ALLOW_SHORT_CONTEXT_WINDOWS'


@pytest.fixture(autouse=True)
def allow_short_context_windows():
    """Allow small context windows so unit tests can create LLM with gpt-4 etc."""
    old = os.environ.pop(_ALLOW_SHORT_CONTEXT_WINDOWS, None)
    os.environ[_ALLOW_SHORT_CONTEXT_WINDOWS] = 'true'
    try:
        yield
    finally:
        if old is not None:
            os.environ[_ALLOW_SHORT_CONTEXT_WINDOWS] = old
        else:
            os.environ.pop(_ALLOW_SHORT_CONTEXT_WINDOWS, None)


class TestLiveStatusAppConversationService:
    """Test cases for the methods in LiveStatusAppConversationService."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_user_context = Mock(spec=UserContext)
        self.mock_user_auth = Mock()
        self.mock_user_context.user_auth = self.mock_user_auth
        self.mock_jwt_service = Mock()
        self.mock_sandbox_service = Mock()
        self.mock_sandbox_spec_service = Mock()
        self.mock_app_conversation_info_service = Mock()
        self.mock_app_conversation_start_task_service = Mock()
        self.mock_event_callback_service = Mock()
        self.mock_event_service = Mock()
        self.mock_httpx_client = Mock()

        # Create service instance
        self.service = LiveStatusAppConversationService(
            init_git_in_empty_workspace=True,
            user_context=self.mock_user_context,
            app_conversation_info_service=self.mock_app_conversation_info_service,
            app_conversation_start_task_service=self.mock_app_conversation_start_task_service,
            event_callback_service=self.mock_event_callback_service,
            event_service=self.mock_event_service,
            sandbox_service=self.mock_sandbox_service,
            sandbox_spec_service=self.mock_sandbox_spec_service,
            jwt_service=self.mock_jwt_service,
            sandbox_startup_timeout=30,
            sandbox_startup_poll_frequency=1,
            httpx_client=self.mock_httpx_client,
            web_url='https://test.example.com',
            openhands_provider_base_url='https://provider.example.com',
            access_token_hard_timeout=None,
            app_mode='test',
        )

        # Mock user info
        self.mock_user = Mock()
        self.mock_user.id = 'test_user_123'
        self.mock_user.llm_model = 'gpt-4'
        self.mock_user.llm_base_url = 'https://api.openai.com/v1'
        self.mock_user.llm_api_key = 'test_api_key'
        self.mock_user.confirmation_mode = False
        self.mock_user.search_api_key = None  # Default to None
        self.mock_user.condenser_max_size = None  # Default to None
        self.mock_user.llm_base_url = 'https://api.openai.com/v1'
        self.mock_user.mcp_config = None  # Default to None to avoid error handling path

        # Mock sandbox
        self.mock_sandbox = Mock(spec=SandboxInfo)
        self.mock_sandbox.id = uuid4()
        self.mock_sandbox.status = SandboxStatus.RUNNING

    @pytest.mark.asyncio
    async def test_setup_secrets_for_git_providers_no_provider_tokens(self):
        """Test _setup_secrets_for_git_providers with no provider tokens."""
        # Arrange
        base_secrets = {'existing': 'secret'}
        self.mock_user_context.get_secrets.return_value = base_secrets
        self.mock_user_context.get_provider_tokens = AsyncMock(return_value=None)

        # Act
        result = await self.service._setup_secrets_for_git_providers(self.mock_user)

        # Assert
        assert result == base_secrets
        self.mock_user_context.get_secrets.assert_called_once()
        self.mock_user_context.get_provider_tokens.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_secrets_for_git_providers_with_web_url(self):
        """Test _setup_secrets_for_git_providers with web URL (creates access token)."""
        # Arrange
        base_secrets = {}
        self.mock_user_context.get_secrets.return_value = base_secrets
        self.mock_jwt_service.create_jws_token.return_value = 'test_access_token'

        # Mock provider tokens
        provider_tokens = {
            ProviderType.GITHUB: ProviderToken(token=SecretStr('github_token')),
            ProviderType.GITLAB: ProviderToken(token=SecretStr('gitlab_token')),
        }
        self.mock_user_context.get_provider_tokens = AsyncMock(
            return_value=provider_tokens
        )

        # Act
        result = await self.service._setup_secrets_for_git_providers(self.mock_user)

        # Assert
        assert 'GITHUB_TOKEN' in result
        assert 'GITLAB_TOKEN' in result
        assert isinstance(result['GITHUB_TOKEN'], LookupSecret)
        assert isinstance(result['GITLAB_TOKEN'], LookupSecret)
        assert (
            result['GITHUB_TOKEN'].url
            == 'https://test.example.com/api/v1/webhooks/secrets'
        )
        assert result['GITHUB_TOKEN'].headers['X-Access-Token'] == 'test_access_token'
        # Verify descriptions are included
        assert result['GITHUB_TOKEN'].description == 'GITHUB authentication token'
        assert result['GITLAB_TOKEN'].description == 'GITLAB authentication token'

        # Should be called twice, once for each provider
        assert self.mock_jwt_service.create_jws_token.call_count == 2

    @pytest.mark.asyncio
    async def test_setup_secrets_for_git_providers_with_saas_mode(self):
        """Test _setup_secrets_for_git_providers with SaaS mode uses LookupSecret with X-Access-Token."""
        # Arrange
        self.service.app_mode = 'saas'
        base_secrets = {}
        self.mock_user_context.get_secrets.return_value = base_secrets
        self.mock_jwt_service.create_jws_token.return_value = 'test_access_token'

        # Mock provider tokens
        provider_tokens = {
            ProviderType.GITLAB: ProviderToken(token=SecretStr('gitlab_token')),
        }
        self.mock_user_context.get_provider_tokens = AsyncMock(
            return_value=provider_tokens
        )

        # Act
        result = await self.service._setup_secrets_for_git_providers(self.mock_user)

        # Assert
        assert 'GITLAB_TOKEN' in result
        lookup_secret = result['GITLAB_TOKEN']
        assert isinstance(lookup_secret, LookupSecret)
        assert 'X-Access-Token' in lookup_secret.headers
        assert lookup_secret.headers['X-Access-Token'] == 'test_access_token'
        # Verify no cookie is included (authentication is via X-Access-Token only)
        assert 'Cookie' not in lookup_secret.headers
        # Verify description is included
        assert lookup_secret.description == 'GITLAB authentication token'

    @pytest.mark.asyncio
    async def test_setup_secrets_for_git_providers_without_web_url(self):
        """Test _setup_secrets_for_git_providers without web URL (uses static token)."""
        # Arrange
        self.service.web_url = None
        base_secrets = {}
        self.mock_user_context.get_secrets.return_value = base_secrets
        self.mock_user_context.get_latest_token.return_value = 'static_token_value'

        # Mock provider tokens
        provider_tokens = {
            ProviderType.GITHUB: ProviderToken(token=SecretStr('github_token')),
        }
        self.mock_user_context.get_provider_tokens = AsyncMock(
            return_value=provider_tokens
        )

        # Act
        result = await self.service._setup_secrets_for_git_providers(self.mock_user)

        # Assert
        assert 'GITHUB_TOKEN' in result
        assert isinstance(result['GITHUB_TOKEN'], StaticSecret)
        assert result['GITHUB_TOKEN'].value.get_secret_value() == 'static_token_value'
        # Verify description is included
        assert result['GITHUB_TOKEN'].description == 'GITHUB authentication token'
        self.mock_user_context.get_latest_token.assert_called_once_with(
            ProviderType.GITHUB
        )

    @pytest.mark.asyncio
    async def test_setup_secrets_for_git_providers_no_static_token(self):
        """Test _setup_secrets_for_git_providers when no static token is available."""
        # Arrange
        self.service.web_url = None
        base_secrets = {}
        self.mock_user_context.get_secrets.return_value = base_secrets
        self.mock_user_context.get_latest_token.return_value = None

        # Mock provider tokens
        provider_tokens = {
            ProviderType.GITHUB: ProviderToken(token=SecretStr('github_token')),
        }
        self.mock_user_context.get_provider_tokens = AsyncMock(
            return_value=provider_tokens
        )

        # Act
        result = await self.service._setup_secrets_for_git_providers(self.mock_user)

        # Assert
        assert 'GITHUB_TOKEN' not in result
        assert result == base_secrets

    @pytest.mark.asyncio
    async def test_setup_secrets_for_git_providers_descriptions_included(self):
        """Test _setup_secrets_for_git_providers includes descriptions for all provider types."""
        # Arrange
        base_secrets = {}
        self.mock_user_context.get_secrets.return_value = base_secrets
        self.mock_jwt_service.create_jws_token.return_value = 'test_access_token'

        # Mock provider tokens for multiple providers
        provider_tokens = {
            ProviderType.GITHUB: ProviderToken(token=SecretStr('github_token')),
            ProviderType.GITLAB: ProviderToken(token=SecretStr('gitlab_token')),
            ProviderType.BITBUCKET: ProviderToken(token=SecretStr('bitbucket_token')),
        }
        self.mock_user_context.get_provider_tokens = AsyncMock(
            return_value=provider_tokens
        )

        # Act
        result = await self.service._setup_secrets_for_git_providers(self.mock_user)

        # Assert - verify all secrets have correct descriptions
        assert 'GITHUB_TOKEN' in result
        assert isinstance(result['GITHUB_TOKEN'], LookupSecret)
        assert result['GITHUB_TOKEN'].description == 'GITHUB authentication token'

        assert 'GITLAB_TOKEN' in result
        assert isinstance(result['GITLAB_TOKEN'], LookupSecret)
        assert result['GITLAB_TOKEN'].description == 'GITLAB authentication token'

        assert 'BITBUCKET_TOKEN' in result
        assert isinstance(result['BITBUCKET_TOKEN'], LookupSecret)
        assert result['BITBUCKET_TOKEN'].description == 'BITBUCKET authentication token'

    @pytest.mark.asyncio
    async def test_setup_secrets_for_git_providers_static_secret_description(self):
        """Test _setup_secrets_for_git_providers includes description for StaticSecret."""
        # Arrange
        self.service.web_url = None
        base_secrets = {}
        self.mock_user_context.get_secrets.return_value = base_secrets
        self.mock_user_context.get_latest_token.return_value = 'static_token_value'

        # Mock provider tokens for multiple providers
        provider_tokens = {
            ProviderType.GITHUB: ProviderToken(token=SecretStr('github_token')),
            ProviderType.GITLAB: ProviderToken(token=SecretStr('gitlab_token')),
        }
        self.mock_user_context.get_provider_tokens = AsyncMock(
            return_value=provider_tokens
        )

        # Act
        result = await self.service._setup_secrets_for_git_providers(self.mock_user)

        # Assert - verify StaticSecret objects have descriptions
        assert 'GITHUB_TOKEN' in result
        assert isinstance(result['GITHUB_TOKEN'], StaticSecret)
        assert result['GITHUB_TOKEN'].description == 'GITHUB authentication token'

        assert 'GITLAB_TOKEN' in result
        assert isinstance(result['GITLAB_TOKEN'], StaticSecret)
        assert result['GITLAB_TOKEN'].description == 'GITLAB authentication token'

    @pytest.mark.asyncio
    async def test_setup_secrets_for_git_providers_preserves_custom_secret_descriptions(
        self,
    ):
        """Test _setup_secrets_for_git_providers preserves descriptions from custom secrets."""
        # Arrange
        # Mock custom secrets with descriptions
        custom_secret_with_desc = StaticSecret(
            value=SecretStr('custom_secret_value'),
            description='Custom API key for external service',
        )
        custom_secret_no_desc = StaticSecret(
            value=SecretStr('another_secret_value'),
            description=None,
        )
        base_secrets = {
            'CUSTOM_API_KEY': custom_secret_with_desc,
            'ANOTHER_SECRET': custom_secret_no_desc,
        }
        self.mock_user_context.get_secrets.return_value = base_secrets
        self.mock_jwt_service.create_jws_token.return_value = 'test_access_token'

        # Mock provider tokens
        provider_tokens = {
            ProviderType.GITHUB: ProviderToken(token=SecretStr('github_token')),
        }
        self.mock_user_context.get_provider_tokens = AsyncMock(
            return_value=provider_tokens
        )

        # Act
        result = await self.service._setup_secrets_for_git_providers(self.mock_user)

        # Assert - verify custom secrets are preserved with their descriptions
        assert 'CUSTOM_API_KEY' in result
        assert isinstance(result['CUSTOM_API_KEY'], StaticSecret)
        assert (
            result['CUSTOM_API_KEY'].description
            == 'Custom API key for external service'
        )
        assert (
            result['CUSTOM_API_KEY'].value.get_secret_value() == 'custom_secret_value'
        )

        assert 'ANOTHER_SECRET' in result
        assert isinstance(result['ANOTHER_SECRET'], StaticSecret)
        assert result['ANOTHER_SECRET'].description is None
        assert (
            result['ANOTHER_SECRET'].value.get_secret_value() == 'another_secret_value'
        )

        # Verify git provider token is also included
        assert 'GITHUB_TOKEN' in result
        assert result['GITHUB_TOKEN'].description == 'GITHUB authentication token'

    @pytest.mark.asyncio
    async def test_setup_secrets_for_git_providers_custom_secret_empty_description(
        self,
    ):
        """Test _setup_secrets_for_git_providers handles custom secrets with empty descriptions."""
        # Arrange
        custom_secret_empty_desc = StaticSecret(
            value=SecretStr('secret_value'),
            description='',  # Empty string description
        )
        base_secrets = {'MY_SECRET': custom_secret_empty_desc}
        self.mock_user_context.get_secrets.return_value = base_secrets
        self.mock_user_context.get_provider_tokens = AsyncMock(return_value=None)

        # Act
        result = await self.service._setup_secrets_for_git_providers(self.mock_user)

        # Assert - empty description should be preserved as-is
        assert 'MY_SECRET' in result
        assert isinstance(result['MY_SECRET'], StaticSecret)
        # Empty string description is preserved
        assert result['MY_SECRET'].description == ''

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_with_custom_model(self):
        """Test _configure_llm_and_mcp with custom LLM model."""
        # Arrange
        custom_model = 'gpt-3.5-turbo'
        self.mock_user_context.get_mcp_api_key.return_value = 'mcp_api_key'

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, custom_model
        )

        # Assert
        assert isinstance(llm, LLM)
        assert llm.model == custom_model
        assert llm.base_url == self.mock_user.llm_base_url
        assert llm.api_key.get_secret_value() == self.mock_user.llm_api_key
        assert llm.usage_id == 'agent'

        assert 'mcpServers' in mcp_config
        assert 'default' in mcp_config['mcpServers']
        assert (
            mcp_config['mcpServers']['default']['url']
            == 'https://test.example.com/mcp/mcp'
        )
        assert (
            mcp_config['mcpServers']['default']['headers']['X-Session-API-Key']
            == 'mcp_api_key'
        )

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_openhands_model_prefers_user_base_url(self):
        """openhands/* model uses user.llm_base_url when provided."""
        # Arrange
        self.mock_user.llm_model = 'openhands/special'
        self.mock_user.llm_base_url = 'https://user-llm.example.com'
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, _ = await self.service._configure_llm_and_mcp(
            self.mock_user, self.mock_user.llm_model
        )

        # Assert
        assert llm.base_url == 'https://user-llm.example.com'

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_openhands_model_uses_provider_default(self):
        """openhands/* model falls back to configured provider base URL."""
        # Arrange
        self.mock_user.llm_model = 'openhands/default'
        self.mock_user.llm_base_url = None
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, _ = await self.service._configure_llm_and_mcp(
            self.mock_user, self.mock_user.llm_model
        )

        # Assert
        assert llm.base_url == 'https://provider.example.com'

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_openhands_model_no_base_urls(self):
        """openhands/* model sets base_url to None when no sources available."""
        # Arrange
        self.mock_user.llm_model = 'openhands/default'
        self.mock_user.llm_base_url = None
        self.service.openhands_provider_base_url = None
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, _ = await self.service._configure_llm_and_mcp(
            self.mock_user, self.mock_user.llm_model
        )

        # Assert
        assert llm.base_url == 'https://llm-proxy.app.all-hands.dev/'

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_non_openhands_model_ignores_provider(self):
        """Non-openhands model ignores provider base URL and uses user base URL."""
        # Arrange
        self.mock_user.llm_model = 'gpt-4'
        self.mock_user.llm_base_url = 'https://user-llm.example.com'
        self.service.openhands_provider_base_url = 'https://provider.example.com'
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, _ = await self.service._configure_llm_and_mcp(self.mock_user, None)

        # Assert
        assert llm.base_url == 'https://user-llm.example.com'

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_with_user_default_model(self):
        """Test _configure_llm_and_mcp using user's default model."""
        # Arrange
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        assert llm.model == self.mock_user.llm_model
        assert 'mcpServers' in mcp_config
        assert 'default' in mcp_config['mcpServers']
        assert 'headers' not in mcp_config['mcpServers']['default']

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_without_web_url(self):
        """Test _configure_llm_and_mcp without web URL (no MCP config)."""
        # Arrange
        self.service.web_url = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        assert isinstance(llm, LLM)
        assert mcp_config == {}

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_tavily_with_user_search_api_key(self):
        """Test _configure_llm_and_mcp adds tavily when user has search_api_key."""
        # Arrange
        self.mock_user.search_api_key = SecretStr('user_search_key')
        self.mock_user_context.get_mcp_api_key.return_value = 'mcp_api_key'

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        assert isinstance(llm, LLM)
        assert 'mcpServers' in mcp_config
        assert 'default' in mcp_config['mcpServers']
        assert 'tavily' in mcp_config['mcpServers']
        assert (
            mcp_config['mcpServers']['tavily']['url']
            == 'https://mcp.tavily.com/mcp/?tavilyApiKey=user_search_key'
        )

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_tavily_with_env_tavily_key(self):
        """Test _configure_llm_and_mcp adds tavily when service has tavily_api_key."""
        # Arrange
        self.service.tavily_api_key = 'env_tavily_key'
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        assert isinstance(llm, LLM)
        assert 'mcpServers' in mcp_config
        assert 'default' in mcp_config['mcpServers']
        assert 'tavily' in mcp_config['mcpServers']
        assert (
            mcp_config['mcpServers']['tavily']['url']
            == 'https://mcp.tavily.com/mcp/?tavilyApiKey=env_tavily_key'
        )

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_tavily_user_key_takes_precedence(self):
        """Test _configure_llm_and_mcp user search_api_key takes precedence over env key."""
        # Arrange
        self.mock_user.search_api_key = SecretStr('user_search_key')
        self.service.tavily_api_key = 'env_tavily_key'
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        assert isinstance(llm, LLM)
        assert 'mcpServers' in mcp_config
        assert 'tavily' in mcp_config['mcpServers']
        assert (
            mcp_config['mcpServers']['tavily']['url']
            == 'https://mcp.tavily.com/mcp/?tavilyApiKey=user_search_key'
        )

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_no_tavily_without_keys(self):
        """Test _configure_llm_and_mcp does not add tavily when no keys are available."""
        # Arrange
        self.mock_user.search_api_key = None
        self.service.tavily_api_key = None
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        assert isinstance(llm, LLM)
        assert 'mcpServers' in mcp_config
        assert 'default' in mcp_config['mcpServers']
        assert 'tavily' not in mcp_config['mcpServers']

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_saas_mode_no_tavily_without_user_key(self):
        """Test _configure_llm_and_mcp does not add tavily in SAAS mode without user search_api_key.

        In SAAS mode, the global tavily_api_key should not be passed to the service instance,
        so tavily should only be added if the user has their own search_api_key.
        """
        # Arrange - simulate SAAS mode where no global tavily key is available
        self.service.app_mode = AppMode.SAAS.value
        self.service.tavily_api_key = None  # In SAAS mode, this should be None
        self.mock_user.search_api_key = None
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        assert isinstance(llm, LLM)
        assert 'mcpServers' in mcp_config
        assert 'default' in mcp_config['mcpServers']
        assert 'tavily' not in mcp_config['mcpServers']

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_saas_mode_with_user_search_key(self):
        """Test _configure_llm_and_mcp adds tavily in SAAS mode when user has search_api_key.

        Even in SAAS mode, if the user has their own search_api_key, tavily should be added.
        """
        # Arrange - simulate SAAS mode with user having their own search key
        self.service.app_mode = AppMode.SAAS.value
        self.service.tavily_api_key = None  # In SAAS mode, this should be None
        self.mock_user.search_api_key = SecretStr('user_search_key')
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        assert isinstance(llm, LLM)
        assert 'mcpServers' in mcp_config
        assert 'default' in mcp_config['mcpServers']
        assert 'tavily' in mcp_config['mcpServers']
        assert (
            mcp_config['mcpServers']['tavily']['url']
            == 'https://mcp.tavily.com/mcp/?tavilyApiKey=user_search_key'
        )

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_tavily_with_empty_user_search_key(self):
        """Test _configure_llm_and_mcp handles empty user search_api_key correctly."""
        # Arrange
        self.mock_user.search_api_key = SecretStr('')  # Empty string
        self.service.tavily_api_key = 'env_tavily_key'
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        assert isinstance(llm, LLM)
        assert 'mcpServers' in mcp_config
        assert 'tavily' in mcp_config['mcpServers']
        # Should fall back to env key since user key is empty
        assert (
            mcp_config['mcpServers']['tavily']['url']
            == 'https://mcp.tavily.com/mcp/?tavilyApiKey=env_tavily_key'
        )

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_tavily_with_whitespace_user_search_key(self):
        """Test _configure_llm_and_mcp handles whitespace-only user search_api_key correctly."""
        # Arrange
        self.mock_user.search_api_key = SecretStr('   ')  # Whitespace only
        self.service.tavily_api_key = 'env_tavily_key'
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        assert isinstance(llm, LLM)
        assert 'mcpServers' in mcp_config
        assert 'tavily' in mcp_config['mcpServers']
        # Should fall back to env key since user key is whitespace only
        assert (
            mcp_config['mcpServers']['tavily']['url']
            == 'https://mcp.tavily.com/mcp/?tavilyApiKey=env_tavily_key'
        )

    def test_compute_plan_path_default_uses_agents_tmp(self):
        """Test _compute_plan_path returns .agents_tmp/PLAN.md for default/GitHub."""
        # Arrange
        working_dir = '/workspace/project'

        # Act
        path_none = self.service._compute_plan_path(working_dir, None)
        path_github = self.service._compute_plan_path(working_dir, ProviderType.GITHUB)

        # Assert
        assert path_none == '/workspace/project/.agents_tmp/PLAN.md'
        assert path_github == '/workspace/project/.agents_tmp/PLAN.md'

    def test_compute_plan_path_gitlab_uses_agents_tmp_config(self):
        """Test _compute_plan_path returns agents-tmp-config/PLAN.md for GitLab."""
        # Arrange
        working_dir = '/workspace/project'

        # Act
        path = self.service._compute_plan_path(working_dir, ProviderType.GITLAB)

        # Assert
        assert path == '/workspace/project/agents-tmp-config/PLAN.md'

    def test_compute_plan_path_azure_uses_agents_tmp_config(self):
        """Test _compute_plan_path returns agents-tmp-config/PLAN.md for Azure."""
        # Arrange
        working_dir = '/workspace/project'

        # Act
        path = self.service._compute_plan_path(working_dir, ProviderType.AZURE_DEVOPS)

        # Assert
        assert path == '/workspace/project/agents-tmp-config/PLAN.md'

    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.get_planning_tools'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.AppConversationServiceBase._create_condenser'
    )
    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.format_plan_structure'
    )
    def test_create_agent_with_context_planning_agent(
        self, mock_format_plan, mock_create_condenser, mock_get_tools
    ):
        """Test _create_agent_with_context for planning agent type."""
        # Arrange
        mock_llm = Mock(spec=LLM)
        mock_llm.model_copy.return_value = mock_llm
        mock_get_tools.return_value = []
        mock_condenser = Mock()
        mock_create_condenser.return_value = mock_condenser
        mock_format_plan.return_value = 'test_plan_structure'
        mcp_config = {'default': {'url': 'test'}}
        system_message_suffix = 'Test suffix'
        working_dir = '/workspace/project'
        git_provider = ProviderType.GITHUB

        # Act
        with patch(
            'openhands.app_server.app_conversation.live_status_app_conversation_service.Agent'
        ) as mock_agent_class:
            mock_agent_instance = Mock()
            mock_agent_instance.model_copy.return_value = mock_agent_instance
            mock_agent_class.return_value = mock_agent_instance

            self.service._create_agent_with_context(
                mock_llm,
                AgentType.PLAN,
                system_message_suffix,
                mcp_config,
                self.mock_user.condenser_max_size,
                git_provider=git_provider,
                working_dir=working_dir,
            )

            # Assert
            mock_get_tools.assert_called_once_with(
                plan_path='/workspace/project/.agents_tmp/PLAN.md'
            )
            mock_agent_class.assert_called_once()
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs['llm'] == mock_llm
            assert call_kwargs['system_prompt_filename'] == 'system_prompt_planning.j2'
            assert (
                call_kwargs['system_prompt_kwargs']['plan_structure']
                == 'test_plan_structure'
            )
            assert call_kwargs['mcp_config'] == mcp_config
            assert call_kwargs['security_analyzer'] is None
            assert call_kwargs['condenser'] == mock_condenser
            mock_create_condenser.assert_called_once_with(
                mock_llm, AgentType.PLAN, self.mock_user.condenser_max_size
            )

    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.get_default_tools'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.AppConversationServiceBase._create_condenser'
    )
    def test_create_agent_with_context_default_agent(
        self, mock_create_condenser, mock_get_tools
    ):
        """Test _create_agent_with_context for default agent type."""
        # Arrange
        mock_llm = Mock(spec=LLM)
        mock_llm.model_copy.return_value = mock_llm
        mock_get_tools.return_value = []
        mock_condenser = Mock()
        mock_create_condenser.return_value = mock_condenser
        mcp_config = {'default': {'url': 'test'}}

        # Act
        with patch(
            'openhands.app_server.app_conversation.live_status_app_conversation_service.Agent'
        ) as mock_agent_class:
            mock_agent_instance = Mock()
            mock_agent_instance.model_copy.return_value = mock_agent_instance
            mock_agent_class.return_value = mock_agent_instance

            self.service._create_agent_with_context(
                mock_llm,
                AgentType.DEFAULT,
                None,
                mcp_config,
                self.mock_user.condenser_max_size,
            )

            # Assert
            mock_agent_class.assert_called_once()
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs['llm'] == mock_llm
            assert call_kwargs['system_prompt_kwargs']['cli_mode'] is False
            assert call_kwargs['mcp_config'] == mcp_config
            assert call_kwargs['condenser'] == mock_condenser
            mock_get_tools.assert_called_once_with(enable_browser=True)
            mock_create_condenser.assert_called_once_with(
                mock_llm, AgentType.DEFAULT, self.mock_user.condenser_max_size
            )

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.ExperimentManagerImpl'
    )
    async def test_finalize_conversation_request_with_skills(
        self, mock_experiment_manager
    ):
        """Test _finalize_conversation_request with skills loading."""
        # Arrange
        mock_agent = Mock(spec=Agent)

        # Create mock LLM with required attributes for _update_agent_with_llm_metadata
        mock_llm = Mock(spec=LLM)
        mock_llm.model = 'gpt-4'  # Non-openhands model, so no metadata update
        mock_llm.usage_id = 'agent'

        mock_updated_agent = Mock(spec=Agent)
        mock_updated_agent.llm = mock_llm
        mock_updated_agent.condenser = None  # No condenser
        mock_experiment_manager.run_agent_variant_tests__v1.return_value = (
            mock_updated_agent
        )

        conversation_id = uuid4()
        workspace = LocalWorkspace(working_dir='/test')
        initial_message = Mock(spec=SendMessageRequest)
        secrets = {'test': StaticSecret(value='secret')}
        remote_workspace = Mock(spec=AsyncRemoteWorkspace)

        # Mock the skills loading method
        self.service._load_skills_and_update_agent = AsyncMock(
            return_value=mock_updated_agent
        )

        # Act
        result = await self.service._finalize_conversation_request(
            mock_agent,
            conversation_id,
            self.mock_user,
            workspace,
            initial_message,
            secrets,
            self.mock_sandbox,
            remote_workspace,
            'test_repo',
            '/test/dir',
        )

        # Assert
        assert isinstance(result, StartConversationRequest)
        assert result.conversation_id == conversation_id
        assert result.agent == mock_updated_agent
        assert result.workspace == workspace
        assert result.initial_message == initial_message
        assert result.secrets == secrets

        mock_experiment_manager.run_agent_variant_tests__v1.assert_called_once_with(
            self.mock_user.id, conversation_id, mock_agent
        )
        self.service._load_skills_and_update_agent.assert_called_once_with(
            self.mock_sandbox,
            mock_updated_agent,
            remote_workspace,
            'test_repo',
            '/test/dir',
        )

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.ExperimentManagerImpl'
    )
    async def test_finalize_conversation_request_without_skills(
        self, mock_experiment_manager
    ):
        """Test _finalize_conversation_request without remote workspace (no skills)."""
        # Arrange
        mock_agent = Mock(spec=Agent)

        # Create mock LLM with required attributes for _update_agent_with_llm_metadata
        mock_llm = Mock(spec=LLM)
        mock_llm.model = 'gpt-4'  # Non-openhands model, so no metadata update
        mock_llm.usage_id = 'agent'

        mock_updated_agent = Mock(spec=Agent)
        mock_updated_agent.llm = mock_llm
        mock_updated_agent.condenser = None  # No condenser
        mock_experiment_manager.run_agent_variant_tests__v1.return_value = (
            mock_updated_agent
        )

        workspace = LocalWorkspace(working_dir='/test')
        secrets = {'test': StaticSecret(value='secret')}

        # Act
        result = await self.service._finalize_conversation_request(
            mock_agent,
            None,
            self.mock_user,
            workspace,
            None,
            secrets,
            self.mock_sandbox,
            None,
            None,
            '/test/dir',
        )

        # Assert
        assert isinstance(result, StartConversationRequest)
        assert isinstance(result.conversation_id, UUID)
        assert result.agent == mock_updated_agent
        mock_experiment_manager.run_agent_variant_tests__v1.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.ExperimentManagerImpl'
    )
    async def test_finalize_conversation_request_skills_loading_fails(
        self, mock_experiment_manager
    ):
        """Test _finalize_conversation_request when skills loading fails."""
        # Arrange
        mock_agent = Mock(spec=Agent)

        # Create mock LLM with required attributes for _update_agent_with_llm_metadata
        mock_llm = Mock(spec=LLM)
        mock_llm.model = 'gpt-4'  # Non-openhands model, so no metadata update
        mock_llm.usage_id = 'agent'

        mock_updated_agent = Mock(spec=Agent)
        mock_updated_agent.llm = mock_llm
        mock_updated_agent.condenser = None  # No condenser
        mock_experiment_manager.run_agent_variant_tests__v1.return_value = (
            mock_updated_agent
        )

        workspace = LocalWorkspace(working_dir='/test')
        secrets = {'test': StaticSecret(value='secret')}
        remote_workspace = Mock(spec=AsyncRemoteWorkspace)

        # Mock skills loading to raise an exception
        self.service._load_skills_and_update_agent = AsyncMock(
            side_effect=Exception('Skills loading failed')
        )

        # Act
        with patch(
            'openhands.app_server.app_conversation.live_status_app_conversation_service._logger'
        ) as mock_logger:
            result = await self.service._finalize_conversation_request(
                mock_agent,
                None,
                self.mock_user,
                workspace,
                None,
                secrets,
                self.mock_sandbox,
                remote_workspace,
                'test_repo',
                '/test/dir',
            )

            # Assert
            assert isinstance(result, StartConversationRequest)
            assert (
                result.agent == mock_updated_agent
            )  # Should still use the experiment-modified agent
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_start_conversation_request_for_user_integration(self):
        """Test the main _build_start_conversation_request_for_user method integration."""
        # Arrange
        self.mock_user_context.get_user_info.return_value = self.mock_user

        # Mock all the helper methods
        mock_secrets = {'GITHUB_TOKEN': Mock()}
        mock_llm = Mock(spec=LLM)
        mock_mcp_config = {'default': {'url': 'test'}}
        mock_agent = Mock(spec=Agent)
        mock_final_request = Mock(spec=StartConversationRequest)

        self.service._setup_secrets_for_git_providers = AsyncMock(
            return_value=mock_secrets
        )
        self.service._configure_llm_and_mcp = AsyncMock(
            return_value=(mock_llm, mock_mcp_config)
        )
        self.service._create_agent_with_context = Mock(return_value=mock_agent)
        self.service._finalize_conversation_request = AsyncMock(
            return_value=mock_final_request
        )

        # Act
        result = await self.service._build_start_conversation_request_for_user(
            sandbox=self.mock_sandbox,
            initial_message=None,
            system_message_suffix='Test suffix',
            git_provider=ProviderType.GITHUB,
            working_dir='/test/dir',
            agent_type=AgentType.DEFAULT,
            llm_model='gpt-4',
            conversation_id=None,
            remote_workspace=None,
            selected_repository='test/repo',
        )

        # Assert
        assert result == mock_final_request

        self.service._setup_secrets_for_git_providers.assert_called_once_with(
            self.mock_user
        )
        self.service._configure_llm_and_mcp.assert_called_once_with(
            self.mock_user, 'gpt-4'
        )
        self.service._create_agent_with_context.assert_called_once_with(
            mock_llm,
            AgentType.DEFAULT,
            'Test suffix',
            mock_mcp_config,
            self.mock_user.condenser_max_size,
            secrets=mock_secrets,
            git_provider=ProviderType.GITHUB,
            working_dir='/test/dir',
        )
        self.service._finalize_conversation_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_conversation_success(self):
        """Test successful download of conversation trajectory."""
        # Arrange
        conversation_id = uuid4()

        # Mock conversation info
        mock_conversation_info = Mock(spec=AppConversationInfo)
        mock_conversation_info.id = conversation_id
        mock_conversation_info.title = 'Test Conversation'
        mock_conversation_info.created_at = datetime(2024, 1, 1, 12, 0, 0)
        mock_conversation_info.updated_at = datetime(2024, 1, 1, 13, 0, 0)
        mock_conversation_info.selected_repository = 'test/repo'
        mock_conversation_info.git_provider = 'github'
        mock_conversation_info.selected_branch = 'main'
        mock_conversation_info.model_dump_json = Mock(
            return_value='{"id": "test", "title": "Test Conversation"}'
        )

        self.mock_app_conversation_info_service.get_app_conversation_info = AsyncMock(
            return_value=mock_conversation_info
        )

        # Mock events
        mock_event1 = Mock(spec=Event)
        mock_event1.id = uuid4()
        mock_event1.model_dump = Mock(
            return_value={'id': str(mock_event1.id), 'type': 'action'}
        )

        mock_event2 = Mock(spec=Event)
        mock_event2.id = uuid4()
        mock_event2.model_dump = Mock(
            return_value={'id': str(mock_event2.id), 'type': 'observation'}
        )

        # Mock event service search_events to return paginated results
        mock_event_page1 = Mock()
        mock_event_page1.items = [mock_event1]
        mock_event_page1.next_page_id = 'page2'

        mock_event_page2 = Mock()
        mock_event_page2.items = [mock_event2]
        mock_event_page2.next_page_id = None

        self.mock_event_service.search_events = AsyncMock(
            side_effect=[mock_event_page1, mock_event_page2]
        )

        # Act
        result = await self.service.export_conversation(conversation_id)

        # Assert
        assert result is not None
        assert isinstance(result, bytes)  # Should be bytes

        # Verify the zip file contents
        with zipfile.ZipFile(io.BytesIO(result), 'r') as zipf:
            file_list = zipf.namelist()

            # Should contain meta.json and event files
            assert 'meta.json' in file_list
            assert any(
                f.startswith('event_') and f.endswith('.json') for f in file_list
            )

            # Check meta.json content
            with zipf.open('meta.json') as meta_file:
                meta_content = meta_file.read().decode('utf-8')
                assert '"id": "test"' in meta_content
                assert '"title": "Test Conversation"' in meta_content

            # Check event files
            event_files = [f for f in file_list if f.startswith('event_')]
            assert len(event_files) == 2  # Should have 2 event files

            # Verify event file content
            with zipf.open(event_files[0]) as event_file:
                event_content = json.loads(event_file.read().decode('utf-8'))
                assert 'id' in event_content
                assert 'type' in event_content

        # Verify service calls
        self.mock_app_conversation_info_service.get_app_conversation_info.assert_called_once_with(
            conversation_id
        )
        assert self.mock_event_service.search_events.call_count == 2
        mock_conversation_info.model_dump_json.assert_called_once_with(indent=2)

    @pytest.mark.asyncio
    async def test_export_conversation_conversation_not_found(self):
        """Test download when conversation is not found."""
        # Arrange
        conversation_id = uuid4()
        self.mock_app_conversation_info_service.get_app_conversation_info = AsyncMock(
            return_value=None
        )

        # Act & Assert
        with pytest.raises(
            ValueError, match=f'Conversation not found: {conversation_id}'
        ):
            await self.service.export_conversation(conversation_id)

        # Verify service calls
        self.mock_app_conversation_info_service.get_app_conversation_info.assert_called_once_with(
            conversation_id
        )
        self.mock_event_service.search_events.assert_not_called()

    @pytest.mark.asyncio
    async def test_export_conversation_empty_events(self):
        """Test download with conversation that has no events."""
        # Arrange
        conversation_id = uuid4()

        # Mock conversation info
        mock_conversation_info = Mock(spec=AppConversationInfo)
        mock_conversation_info.id = conversation_id
        mock_conversation_info.title = 'Empty Conversation'
        mock_conversation_info.model_dump_json = Mock(
            return_value='{"id": "test", "title": "Empty Conversation"}'
        )

        self.mock_app_conversation_info_service.get_app_conversation_info = AsyncMock(
            return_value=mock_conversation_info
        )

        # Mock empty event page
        mock_event_page = Mock()
        mock_event_page.items = []
        mock_event_page.next_page_id = None

        self.mock_event_service.search_events = AsyncMock(return_value=mock_event_page)

        # Act
        result = await self.service.export_conversation(conversation_id)

        # Assert
        assert result is not None
        assert isinstance(result, bytes)  # Should be bytes

        # Verify the zip file contents
        with zipfile.ZipFile(io.BytesIO(result), 'r') as zipf:
            file_list = zipf.namelist()

            # Should only contain meta.json (no event files)
            assert 'meta.json' in file_list
            assert len([f for f in file_list if f.startswith('event_')]) == 0

        # Verify service calls
        self.mock_app_conversation_info_service.get_app_conversation_info.assert_called_once_with(
            conversation_id
        )
        self.mock_event_service.search_events.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_conversation_calls_search_events_with_correct_parameter_name(
        self,
    ):
        """Test that export_conversation calls search_events with 'conversation_id' parameter, not 'conversation_id__eq'.

        This test verifies the fix for a bug where page_iterator was called with
        conversation_id__eq instead of conversation_id, causing a TypeError since
        the search_events method expects conversation_id as its parameter name.
        """
        # Arrange
        conversation_id = uuid4()

        # Mock conversation info
        mock_conversation_info = Mock(spec=AppConversationInfo)
        mock_conversation_info.id = conversation_id
        mock_conversation_info.model_dump_json = Mock(return_value='{}')

        self.mock_app_conversation_info_service.get_app_conversation_info = AsyncMock(
            return_value=mock_conversation_info
        )

        # Mock empty event page to simplify test
        mock_event_page = Mock()
        mock_event_page.items = []
        mock_event_page.next_page_id = None

        self.mock_event_service.search_events = AsyncMock(return_value=mock_event_page)

        # Act
        await self.service.export_conversation(conversation_id)

        # Assert - Verify search_events was called with 'conversation_id', not 'conversation_id__eq'
        self.mock_event_service.search_events.assert_called()
        call_kwargs = self.mock_event_service.search_events.call_args[1]

        assert 'conversation_id' in call_kwargs, (
            "search_events should be called with 'conversation_id' parameter"
        )
        assert 'conversation_id__eq' not in call_kwargs, (
            "search_events should NOT be called with 'conversation_id__eq' parameter"
        )
        assert call_kwargs['conversation_id'] == conversation_id

    @pytest.mark.asyncio
    async def test_export_conversation_large_pagination(self):
        """Test download with multiple pages of events."""
        # Arrange
        conversation_id = uuid4()

        # Mock conversation info
        mock_conversation_info = Mock(spec=AppConversationInfo)
        mock_conversation_info.id = conversation_id
        mock_conversation_info.title = 'Large Conversation'
        mock_conversation_info.model_dump_json = Mock(
            return_value='{"id": "test", "title": "Large Conversation"}'
        )

        self.mock_app_conversation_info_service.get_app_conversation_info = AsyncMock(
            return_value=mock_conversation_info
        )

        # Create multiple pages of events
        events_per_page = 3
        total_pages = 4
        all_events = []

        for page_num in range(total_pages):
            page_events = []
            for i in range(events_per_page):
                mock_event = Mock(spec=Event)
                mock_event.id = uuid4()
                mock_event.model_dump = Mock(
                    return_value={
                        'id': str(mock_event.id),
                        'type': f'event_page_{page_num}_item_{i}',
                    }
                )
                page_events.append(mock_event)
                all_events.append(mock_event)

            mock_event_page = Mock()
            mock_event_page.items = page_events
            mock_event_page.next_page_id = (
                f'page{page_num + 1}' if page_num < total_pages - 1 else None
            )

            if page_num == 0:
                first_page = mock_event_page
            elif page_num == 1:
                second_page = mock_event_page
            elif page_num == 2:
                third_page = mock_event_page
            else:
                fourth_page = mock_event_page

        self.mock_event_service.search_events = AsyncMock(
            side_effect=[first_page, second_page, third_page, fourth_page]
        )

        # Act
        result = await self.service.export_conversation(conversation_id)

        # Assert
        assert result is not None
        assert isinstance(result, bytes)  # Should be bytes

        # Verify the zip file contents
        with zipfile.ZipFile(io.BytesIO(result), 'r') as zipf:
            file_list = zipf.namelist()

            # Should contain meta.json and all event files
            assert 'meta.json' in file_list
            event_files = [f for f in file_list if f.startswith('event_')]
            assert (
                len(event_files) == total_pages * events_per_page
            )  # Should have all events

        # Verify service calls - should call search_events for each page
        assert self.mock_event_service.search_events.call_count == total_pages

    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.AsyncRemoteWorkspace'
    )
    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.ConversationInfo'
    )
    async def test_start_app_conversation_default_title_uses_first_five_characters(
        self, mock_conversation_info_class, mock_remote_workspace_class
    ):
        """Test that v1 conversations use first 5 characters of conversation ID for default title."""
        # Arrange
        conversation_id = uuid4()
        conversation_id_hex = conversation_id.hex
        expected_title = f'Conversation {conversation_id_hex[:5]}'

        # Mock user context
        self.mock_user_context.get_user_id = AsyncMock(return_value='test_user_123')
        self.mock_user_context.get_user_info = AsyncMock(return_value=self.mock_user)

        # Mock sandbox and sandbox spec
        mock_sandbox_spec = Mock(spec=SandboxSpecInfo)
        mock_sandbox_spec.working_dir = '/test/workspace'
        self.mock_sandbox.sandbox_spec_id = str(uuid4())
        self.mock_sandbox.id = str(uuid4())  # Ensure sandbox.id is a string
        self.mock_sandbox.session_api_key = 'test_session_key'
        exposed_url = ExposedUrl(
            name=AGENT_SERVER, url='http://agent-server:8000', port=60000
        )
        self.mock_sandbox.exposed_urls = [exposed_url]

        self.mock_sandbox_service.get_sandbox = AsyncMock(
            return_value=self.mock_sandbox
        )
        self.mock_sandbox_spec_service.get_sandbox_spec = AsyncMock(
            return_value=mock_sandbox_spec
        )

        # Mock remote workspace
        mock_remote_workspace = Mock()
        mock_remote_workspace_class.return_value = mock_remote_workspace

        # Mock the wait for sandbox and setup scripts
        async def mock_wait_for_sandbox(task):
            task.sandbox_id = self.mock_sandbox.id
            yield task

        async def mock_run_setup_scripts(task, sandbox, workspace, agent_server_url):
            yield task

        self.service._wait_for_sandbox_start = mock_wait_for_sandbox
        self.service.run_setup_scripts = mock_run_setup_scripts

        # Mock build start conversation request
        mock_agent = Mock(spec=Agent)
        mock_agent.llm = Mock(spec=LLM)
        mock_agent.llm.model = 'gpt-4'
        mock_start_request = Mock(spec=StartConversationRequest)
        mock_start_request.agent = mock_agent
        mock_start_request.model_dump.return_value = {'test': 'data'}

        self.service._build_start_conversation_request_for_user = AsyncMock(
            return_value=mock_start_request
        )

        # Mock ConversationInfo returned from agent server
        mock_conversation_info = Mock()
        mock_conversation_info.id = conversation_id
        mock_conversation_info_class.model_validate.return_value = (
            mock_conversation_info
        )

        # Mock HTTP response from agent server
        mock_response = Mock()
        mock_response.json.return_value = {'id': str(conversation_id)}
        mock_response.raise_for_status = Mock()
        self.mock_httpx_client.post = AsyncMock(return_value=mock_response)

        # Mock event callback service
        self.mock_event_callback_service.save_event_callback = AsyncMock()

        # Create request
        request = AppConversationStartRequest()

        # Act
        async for task in self.service._start_app_conversation(request):
            # Consume all tasks to reach the point where title is set
            pass

        # Assert
        # Verify that save_app_conversation_info was called with the correct title format
        self.mock_app_conversation_info_service.save_app_conversation_info.assert_called_once()
        call_args = (
            self.mock_app_conversation_info_service.save_app_conversation_info.call_args
        )
        saved_info = call_args[0][0]  # First positional argument

        assert saved_info.title == expected_title, (
            f'Expected title to be "{expected_title}" (first 5 chars), '
            f'but got "{saved_info.title}"'
        )
        assert saved_info.id == conversation_id

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_with_custom_sse_servers(self):
        """Test _configure_llm_and_mcp merges custom SSE servers with UUID-based names."""
        # Arrange

        from openhands.core.config.mcp_config import MCPConfig, MCPSSEServerConfig

        self.mock_user.mcp_config = MCPConfig(
            sse_servers=[
                MCPSSEServerConfig(url='https://linear.app/sse', api_key='linear_key'),
                MCPSSEServerConfig(url='https://notion.com/sse'),
            ]
        )
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        assert isinstance(llm, LLM)
        assert 'mcpServers' in mcp_config

        # Should have default server + 2 custom SSE servers
        mcp_servers = mcp_config['mcpServers']
        assert 'default' in mcp_servers

        # Find SSE servers (they have sse_ prefix)
        sse_servers = {k: v for k, v in mcp_servers.items() if k.startswith('sse_')}
        assert len(sse_servers) == 2

        # Verify SSE server configurations
        for server_name, server_config in sse_servers.items():
            assert server_name.startswith('sse_')
            assert len(server_name) > 4  # Has UUID suffix
            assert 'url' in server_config
            assert 'transport' in server_config
            assert server_config['transport'] == 'sse'

            # Check if this is the Linear server (has headers)
            if 'headers' in server_config:
                assert server_config['headers']['Authorization'] == 'Bearer linear_key'

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_with_custom_shttp_servers(self):
        """Test _configure_llm_and_mcp merges custom SHTTP servers with timeout."""
        # Arrange
        from openhands.core.config.mcp_config import MCPConfig, MCPSHTTPServerConfig

        self.mock_user.mcp_config = MCPConfig(
            shttp_servers=[
                MCPSHTTPServerConfig(
                    url='https://example.com/mcp',
                    api_key='test_key',
                    timeout=120,
                )
            ]
        )
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        assert isinstance(llm, LLM)
        mcp_servers = mcp_config['mcpServers']

        # Find SHTTP servers
        shttp_servers = {k: v for k, v in mcp_servers.items() if k.startswith('shttp_')}
        assert len(shttp_servers) == 1

        server_config = list(shttp_servers.values())[0]
        assert server_config['url'] == 'https://example.com/mcp'
        assert server_config['transport'] == 'streamable-http'
        assert server_config['headers']['Authorization'] == 'Bearer test_key'
        assert server_config['timeout'] == 120

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_with_custom_stdio_servers(self):
        """Test _configure_llm_and_mcp merges custom STDIO servers with explicit names."""
        # Arrange
        from openhands.core.config.mcp_config import MCPConfig, MCPStdioServerConfig

        self.mock_user.mcp_config = MCPConfig(
            stdio_servers=[
                MCPStdioServerConfig(
                    name='my-custom-server',
                    command='npx',
                    args=['-y', 'my-package'],
                    env={'API_KEY': 'secret'},
                )
            ]
        )
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        assert isinstance(llm, LLM)
        mcp_servers = mcp_config['mcpServers']

        # STDIO server should use its explicit name
        assert 'my-custom-server' in mcp_servers
        server_config = mcp_servers['my-custom-server']
        assert server_config['command'] == 'npx'
        assert server_config['args'] == ['-y', 'my-package']
        assert server_config['env'] == {'API_KEY': 'secret'}

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_merges_system_and_custom_servers(self):
        """Test _configure_llm_and_mcp merges both system and custom MCP servers."""
        # Arrange
        from openhands.core.config.mcp_config import (
            MCPConfig,
            MCPSSEServerConfig,
            MCPStdioServerConfig,
        )

        self.mock_user.search_api_key = SecretStr('tavily_key')
        self.mock_user.mcp_config = MCPConfig(
            sse_servers=[MCPSSEServerConfig(url='https://custom.com/sse')],
            stdio_servers=[
                MCPStdioServerConfig(
                    name='custom-stdio', command='node', args=['app.js']
                )
            ],
        )
        self.mock_user_context.get_mcp_api_key.return_value = 'mcp_api_key'

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        mcp_servers = mcp_config['mcpServers']

        # Should have system servers
        assert 'default' in mcp_servers
        assert 'tavily' in mcp_servers

        # Should have custom SSE server with UUID name
        sse_servers = [k for k in mcp_servers if k.startswith('sse_')]
        assert len(sse_servers) == 1

        # Should have custom STDIO server with explicit name
        assert 'custom-stdio' in mcp_servers

        # Total: default + tavily + 1 SSE + 1 STDIO = 4 servers
        assert len(mcp_servers) == 4

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_custom_config_error_handling(self):
        """Test _configure_llm_and_mcp handles errors in custom MCP config gracefully."""
        # Arrange
        self.mock_user.mcp_config = Mock()
        # Simulate error when accessing sse_servers
        self.mock_user.mcp_config.sse_servers = property(
            lambda self: (_ for _ in ()).throw(Exception('Config error'))
        )
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert - should still return valid config with system servers only
        assert isinstance(llm, LLM)
        mcp_servers = mcp_config['mcpServers']
        assert 'default' in mcp_servers
        # Custom servers should not be added due to error

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_sdk_format_with_mcpservers_wrapper(self):
        """Test _configure_llm_and_mcp returns SDK-required format with mcpServers key."""
        # Arrange
        self.mock_user_context.get_mcp_api_key.return_value = 'mcp_key'

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert - SDK expects {'mcpServers': {...}} format
        assert 'mcpServers' in mcp_config
        assert isinstance(mcp_config['mcpServers'], dict)

        # Verify structure matches SDK expectations
        for server_name, server_config in mcp_config['mcpServers'].items():
            assert isinstance(server_name, str)
            assert isinstance(server_config, dict)

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_empty_custom_config(self):
        """Test _configure_llm_and_mcp handles empty custom MCP config."""
        # Arrange
        from openhands.core.config.mcp_config import MCPConfig

        self.mock_user.mcp_config = MCPConfig(
            sse_servers=[], stdio_servers=[], shttp_servers=[]
        )
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        mcp_servers = mcp_config['mcpServers']
        # Should only have system default server
        assert 'default' in mcp_servers
        assert len(mcp_servers) == 1

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_sse_server_without_api_key(self):
        """Test _configure_llm_and_mcp handles SSE servers without API keys."""
        # Arrange
        from openhands.core.config.mcp_config import MCPConfig, MCPSSEServerConfig

        self.mock_user.mcp_config = MCPConfig(
            sse_servers=[MCPSSEServerConfig(url='https://public.com/sse')]
        )
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        mcp_servers = mcp_config['mcpServers']
        sse_servers = {k: v for k, v in mcp_servers.items() if k.startswith('sse_')}

        # Server should exist but without headers
        assert len(sse_servers) == 1
        server_config = list(sse_servers.values())[0]
        assert 'headers' not in server_config
        assert server_config['url'] == 'https://public.com/sse'
        assert server_config['transport'] == 'sse'

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_shttp_server_without_timeout(self):
        """Test _configure_llm_and_mcp handles SHTTP servers without timeout."""
        # Arrange
        from openhands.core.config.mcp_config import MCPConfig, MCPSHTTPServerConfig

        self.mock_user.mcp_config = MCPConfig(
            shttp_servers=[MCPSHTTPServerConfig(url='https://example.com/mcp')]
        )
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        mcp_servers = mcp_config['mcpServers']
        shttp_servers = {k: v for k, v in mcp_servers.items() if k.startswith('shttp_')}

        assert len(shttp_servers) == 1
        server_config = list(shttp_servers.values())[0]
        # Timeout should be included even if None (defaults to 60)
        assert 'timeout' in server_config

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_stdio_server_without_env(self):
        """Test _configure_llm_and_mcp handles STDIO servers without environment variables."""
        # Arrange
        from openhands.core.config.mcp_config import MCPConfig, MCPStdioServerConfig

        self.mock_user.mcp_config = MCPConfig(
            stdio_servers=[
                MCPStdioServerConfig(
                    name='simple-server', command='node', args=['app.js']
                )
            ]
        )
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        mcp_servers = mcp_config['mcpServers']
        assert 'simple-server' in mcp_servers
        server_config = mcp_servers['simple-server']

        # Should not have env key if not provided
        assert 'env' not in server_config
        assert server_config['command'] == 'node'
        assert server_config['args'] == ['app.js']

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_multiple_servers_same_type(self):
        """Test _configure_llm_and_mcp handles multiple custom servers of the same type."""
        # Arrange
        from openhands.core.config.mcp_config import MCPConfig, MCPSSEServerConfig

        self.mock_user.mcp_config = MCPConfig(
            sse_servers=[
                MCPSSEServerConfig(url='https://server1.com/sse'),
                MCPSSEServerConfig(url='https://server2.com/sse'),
                MCPSSEServerConfig(url='https://server3.com/sse'),
            ]
        )
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        mcp_servers = mcp_config['mcpServers']
        sse_servers = {k: v for k, v in mcp_servers.items() if k.startswith('sse_')}

        # All 3 servers should be present with unique UUID-based names
        assert len(sse_servers) == 3

        # Verify all have unique names
        server_names = list(sse_servers.keys())
        assert len(set(server_names)) == 3  # All names are unique

        # Verify all URLs are preserved
        urls = [v['url'] for v in sse_servers.values()]
        assert 'https://server1.com/sse' in urls
        assert 'https://server2.com/sse' in urls
        assert 'https://server3.com/sse' in urls

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_mixed_server_types(self):
        """Test _configure_llm_and_mcp handles all three server types together."""
        # Arrange
        from openhands.core.config.mcp_config import (
            MCPConfig,
            MCPSHTTPServerConfig,
            MCPSSEServerConfig,
            MCPStdioServerConfig,
        )

        self.mock_user.mcp_config = MCPConfig(
            sse_servers=[
                MCPSSEServerConfig(url='https://sse.example.com/sse', api_key='sse_key')
            ],
            shttp_servers=[
                MCPSHTTPServerConfig(url='https://shttp.example.com/mcp', timeout=90)
            ],
            stdio_servers=[
                MCPStdioServerConfig(
                    name='stdio-server',
                    command='npx',
                    args=['mcp-server'],
                    env={'TOKEN': 'value'},
                )
            ],
        )
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        mcp_servers = mcp_config['mcpServers']

        # Check all server types are present
        sse_count = len([k for k in mcp_servers if k.startswith('sse_')])
        shttp_count = len([k for k in mcp_servers if k.startswith('shttp_')])
        stdio_count = 1 if 'stdio-server' in mcp_servers else 0

        assert sse_count == 1
        assert shttp_count == 1
        assert stdio_count == 1

        # Verify each type has correct configuration
        sse_server = next(v for k, v in mcp_servers.items() if k.startswith('sse_'))
        assert sse_server['transport'] == 'sse'
        assert sse_server['headers']['Authorization'] == 'Bearer sse_key'

        shttp_server = next(v for k, v in mcp_servers.items() if k.startswith('shttp_'))
        assert shttp_server['transport'] == 'streamable-http'
        assert shttp_server['timeout'] == 90

        stdio_server = mcp_servers['stdio-server']
        assert stdio_server['command'] == 'npx'
        assert stdio_server['env'] == {'TOKEN': 'value'}


class TestPluginHandling:
    """Test cases for plugin-related functionality in LiveStatusAppConversationService."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_user_context = Mock(spec=UserContext)
        self.mock_user_auth = Mock()
        self.mock_user_context.user_auth = self.mock_user_auth
        self.mock_jwt_service = Mock()
        self.mock_sandbox_service = Mock()
        self.mock_sandbox_spec_service = Mock()
        self.mock_app_conversation_info_service = Mock()
        self.mock_app_conversation_start_task_service = Mock()
        self.mock_event_callback_service = Mock()
        self.mock_event_service = Mock()
        self.mock_httpx_client = Mock()

        # Create service instance
        self.service = LiveStatusAppConversationService(
            init_git_in_empty_workspace=True,
            user_context=self.mock_user_context,
            app_conversation_info_service=self.mock_app_conversation_info_service,
            app_conversation_start_task_service=self.mock_app_conversation_start_task_service,
            event_callback_service=self.mock_event_callback_service,
            event_service=self.mock_event_service,
            sandbox_service=self.mock_sandbox_service,
            sandbox_spec_service=self.mock_sandbox_spec_service,
            jwt_service=self.mock_jwt_service,
            sandbox_startup_timeout=30,
            sandbox_startup_poll_frequency=1,
            httpx_client=self.mock_httpx_client,
            web_url='https://test.example.com',
            openhands_provider_base_url='https://provider.example.com',
            access_token_hard_timeout=None,
            app_mode='test',
        )

        # Mock user info
        self.mock_user = Mock()
        self.mock_user.id = 'test_user_123'
        self.mock_user.llm_model = 'gpt-4'
        self.mock_user.llm_base_url = 'https://api.openai.com/v1'
        self.mock_user.llm_api_key = 'test_api_key'
        self.mock_user.confirmation_mode = False
        self.mock_user.search_api_key = None
        self.mock_user.condenser_max_size = None
        self.mock_user.mcp_config = None
        self.mock_user.security_analyzer = None

        # Mock sandbox
        self.mock_sandbox = Mock(spec=SandboxInfo)
        self.mock_sandbox.id = uuid4()
        self.mock_sandbox.status = SandboxStatus.RUNNING

    def test_construct_initial_message_with_plugin_params_no_plugins(self):
        """Test _construct_initial_message_with_plugin_params with no plugins returns original message."""
        from openhands.agent_server.models import SendMessageRequest, TextContent

        # Test with None initial message and None plugins
        result = self.service._construct_initial_message_with_plugin_params(None, None)
        assert result is None

        # Test with None initial message and empty plugins list
        result = self.service._construct_initial_message_with_plugin_params(None, [])
        assert result is None

        # Test with initial message but None plugins
        initial_msg = SendMessageRequest(content=[TextContent(text='Hello world')])
        result = self.service._construct_initial_message_with_plugin_params(
            initial_msg, None
        )
        assert result is initial_msg

    def test_construct_initial_message_with_plugin_params_no_params(self):
        """Test _construct_initial_message_with_plugin_params with plugins but no parameters."""
        from openhands.agent_server.models import SendMessageRequest, TextContent
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        # Plugin with no parameters
        plugins = [PluginSpec(source='github:owner/repo')]

        # Test with None initial message
        result = self.service._construct_initial_message_with_plugin_params(
            None, plugins
        )
        assert result is None

        # Test with initial message
        initial_msg = SendMessageRequest(content=[TextContent(text='Hello world')])
        result = self.service._construct_initial_message_with_plugin_params(
            initial_msg, plugins
        )
        assert result is initial_msg

    def test_construct_initial_message_with_plugin_params_creates_new_message(self):
        """Test _construct_initial_message_with_plugin_params creates message when no initial message."""
        from openhands.agent_server.models import TextContent
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        plugins = [
            PluginSpec(
                source='github:owner/repo',
                parameters={'api_key': 'test123', 'debug': True},
            )
        ]

        result = self.service._construct_initial_message_with_plugin_params(
            None, plugins
        )

        assert result is not None
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert 'Plugin Configuration Parameters:' in result.content[0].text
        assert '- api_key: test123' in result.content[0].text
        assert '- debug: True' in result.content[0].text
        assert result.run is True

    def test_construct_initial_message_with_plugin_params_appends_to_message(self):
        """Test _construct_initial_message_with_plugin_params appends to existing message."""
        from openhands.agent_server.models import SendMessageRequest, TextContent
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        initial_msg = SendMessageRequest(
            content=[TextContent(text='Please analyze this codebase')],
            run=False,
        )
        plugins = [
            PluginSpec(
                source='github:owner/repo',
                ref='v1.0.0',
                parameters={'target_dir': '/src', 'verbose': True},
            )
        ]

        result = self.service._construct_initial_message_with_plugin_params(
            initial_msg, plugins
        )

        assert result is not None
        assert len(result.content) == 1
        text = result.content[0].text
        assert text.startswith('Please analyze this codebase')
        assert 'Plugin Configuration Parameters:' in text
        assert '- target_dir: /src' in text
        assert '- verbose: True' in text
        assert result.run is False

    def test_construct_initial_message_with_plugin_params_preserves_role(self):
        """Test _construct_initial_message_with_plugin_params preserves message role."""
        from openhands.agent_server.models import SendMessageRequest, TextContent
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        initial_msg = SendMessageRequest(
            role='system',
            content=[TextContent(text='System message')],
        )
        plugins = [PluginSpec(source='github:owner/repo', parameters={'key': 'value'})]

        result = self.service._construct_initial_message_with_plugin_params(
            initial_msg, plugins
        )

        assert result is not None
        assert result.role == 'system'

    def test_construct_initial_message_with_plugin_params_empty_content(self):
        """Test _construct_initial_message_with_plugin_params handles empty content list."""
        from openhands.agent_server.models import SendMessageRequest, TextContent
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        initial_msg = SendMessageRequest(content=[])
        plugins = [PluginSpec(source='github:owner/repo', parameters={'key': 'value'})]

        result = self.service._construct_initial_message_with_plugin_params(
            initial_msg, plugins
        )

        assert result is not None
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert 'Plugin Configuration Parameters:' in result.content[0].text

    def test_construct_initial_message_with_multiple_plugins(self):
        """Test _construct_initial_message_with_plugin_params handles multiple plugins."""
        from openhands.agent_server.models import TextContent
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        plugins = [
            PluginSpec(
                source='github:owner/plugin1',
                parameters={'key1': 'value1'},
            ),
            PluginSpec(
                source='github:owner/plugin2',
                parameters={'key2': 'value2'},
            ),
        ]

        result = self.service._construct_initial_message_with_plugin_params(
            None, plugins
        )

        assert result is not None
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        text = result.content[0].text
        assert 'Plugin Configuration Parameters:' in text
        # Multiple plugins should show grouped by plugin name
        assert 'plugin1' in text
        assert 'plugin2' in text
        assert 'key1: value1' in text
        assert 'key2: value2' in text

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.ExperimentManagerImpl'
    )
    async def test_finalize_conversation_request_with_plugins(
        self, mock_experiment_manager
    ):
        """Test _finalize_conversation_request passes plugins list to StartConversationRequest."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        # Arrange
        mock_agent = Mock(spec=Agent)
        mock_llm = Mock(spec=LLM)
        mock_llm.model = 'gpt-4'
        mock_llm.usage_id = 'agent'

        mock_updated_agent = Mock(spec=Agent)
        mock_updated_agent.llm = mock_llm
        mock_updated_agent.condenser = None
        mock_experiment_manager.run_agent_variant_tests__v1.return_value = (
            mock_updated_agent
        )

        workspace = LocalWorkspace(working_dir='/test')
        secrets = {'test': StaticSecret(value='secret')}

        plugins = [
            PluginSpec(
                source='github:owner/my-plugin',
                ref='v1.0.0',
                parameters={'api_key': 'test123'},
            )
        ]

        # Act
        result = await self.service._finalize_conversation_request(
            mock_agent,
            None,
            self.mock_user,
            workspace,
            None,
            secrets,
            self.mock_sandbox,
            None,
            None,
            '/test/dir',
            plugins=plugins,
        )

        # Assert
        assert isinstance(result, StartConversationRequest)
        assert result.plugins is not None
        assert len(result.plugins) == 1
        assert result.plugins[0].source == 'github:owner/my-plugin'
        assert result.plugins[0].ref == 'v1.0.0'
        # Also verify initial message contains plugin params
        assert result.initial_message is not None
        assert (
            'Plugin Configuration Parameters:' in result.initial_message.content[0].text
        )
        assert '- api_key: test123' in result.initial_message.content[0].text

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.ExperimentManagerImpl'
    )
    async def test_finalize_conversation_request_without_plugins(
        self, mock_experiment_manager
    ):
        """Test _finalize_conversation_request without plugins sets plugins to None."""
        # Arrange
        mock_agent = Mock(spec=Agent)
        mock_llm = Mock(spec=LLM)
        mock_llm.model = 'gpt-4'
        mock_llm.usage_id = 'agent'

        mock_updated_agent = Mock(spec=Agent)
        mock_updated_agent.llm = mock_llm
        mock_updated_agent.condenser = None
        mock_experiment_manager.run_agent_variant_tests__v1.return_value = (
            mock_updated_agent
        )

        workspace = LocalWorkspace(working_dir='/test')
        secrets = {}

        # Act
        result = await self.service._finalize_conversation_request(
            mock_agent,
            None,
            self.mock_user,
            workspace,
            None,
            secrets,
            self.mock_sandbox,
            None,
            None,
            '/test/dir',
            plugins=None,
        )

        # Assert
        assert isinstance(result, StartConversationRequest)
        assert result.plugins is None

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.ExperimentManagerImpl'
    )
    async def test_finalize_conversation_request_plugin_without_ref(
        self, mock_experiment_manager
    ):
        """Test _finalize_conversation_request with plugin that has no ref."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        # Arrange
        mock_agent = Mock(spec=Agent)
        mock_llm = Mock(spec=LLM)
        mock_llm.model = 'gpt-4'
        mock_llm.usage_id = 'agent'

        mock_updated_agent = Mock(spec=Agent)
        mock_updated_agent.llm = mock_llm
        mock_updated_agent.condenser = None
        mock_experiment_manager.run_agent_variant_tests__v1.return_value = (
            mock_updated_agent
        )

        workspace = LocalWorkspace(working_dir='/test')
        secrets = {}

        # Plugin without ref or parameters
        plugins = [PluginSpec(source='github:owner/my-plugin')]

        # Act
        result = await self.service._finalize_conversation_request(
            mock_agent,
            None,
            self.mock_user,
            workspace,
            None,
            secrets,
            self.mock_sandbox,
            None,
            None,
            '/test/dir',
            plugins=plugins,
        )

        # Assert
        assert isinstance(result, StartConversationRequest)
        assert result.plugins is not None
        assert len(result.plugins) == 1
        assert result.plugins[0].source == 'github:owner/my-plugin'
        assert result.plugins[0].ref is None
        # No parameters, so initial message should be None
        assert result.initial_message is None

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.ExperimentManagerImpl'
    )
    async def test_finalize_conversation_request_plugin_with_repo_path(
        self, mock_experiment_manager
    ):
        """Test _finalize_conversation_request passes repo_path to PluginSource."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        # Arrange
        mock_agent = Mock(spec=Agent)
        mock_llm = Mock(spec=LLM)
        mock_llm.model = 'gpt-4'
        mock_llm.usage_id = 'agent'

        mock_updated_agent = Mock(spec=Agent)
        mock_updated_agent.llm = mock_llm
        mock_updated_agent.condenser = None
        mock_experiment_manager.run_agent_variant_tests__v1.return_value = (
            mock_updated_agent
        )

        workspace = LocalWorkspace(working_dir='/test')
        secrets = {}

        # Plugin with repo_path (for marketplace repos containing multiple plugins)
        plugins = [
            PluginSpec(
                source='github:owner/marketplace-repo',
                ref='main',
                repo_path='plugins/city-weather',
            )
        ]

        # Act
        result = await self.service._finalize_conversation_request(
            mock_agent,
            None,
            self.mock_user,
            workspace,
            None,
            secrets,
            self.mock_sandbox,
            None,
            None,
            '/test/dir',
            plugins=plugins,
        )

        # Assert
        assert isinstance(result, StartConversationRequest)
        assert result.plugins is not None
        assert len(result.plugins) == 1
        assert result.plugins[0].source == 'github:owner/marketplace-repo'
        assert result.plugins[0].ref == 'main'
        assert result.plugins[0].repo_path == 'plugins/city-weather'

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.ExperimentManagerImpl'
    )
    async def test_finalize_conversation_request_multiple_plugins(
        self, mock_experiment_manager
    ):
        """Test _finalize_conversation_request with multiple plugins."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        # Arrange
        mock_agent = Mock(spec=Agent)
        mock_llm = Mock(spec=LLM)
        mock_llm.model = 'gpt-4'
        mock_llm.usage_id = 'agent'

        mock_updated_agent = Mock(spec=Agent)
        mock_updated_agent.llm = mock_llm
        mock_updated_agent.condenser = None
        mock_experiment_manager.run_agent_variant_tests__v1.return_value = (
            mock_updated_agent
        )

        workspace = LocalWorkspace(working_dir='/test')
        secrets = {}

        # Multiple plugins
        plugins = [
            PluginSpec(source='github:owner/security-plugin', ref='v2.0.0'),
            PluginSpec(
                source='github:owner/monorepo',
                repo_path='plugins/logging',
            ),
            PluginSpec(source='/local/path/to/plugin'),
        ]

        # Act
        result = await self.service._finalize_conversation_request(
            mock_agent,
            None,
            self.mock_user,
            workspace,
            None,
            secrets,
            self.mock_sandbox,
            None,
            None,
            '/test/dir',
            plugins=plugins,
        )

        # Assert
        assert isinstance(result, StartConversationRequest)
        assert result.plugins is not None
        assert len(result.plugins) == 3
        assert result.plugins[0].source == 'github:owner/security-plugin'
        assert result.plugins[0].ref == 'v2.0.0'
        assert result.plugins[1].source == 'github:owner/monorepo'
        assert result.plugins[1].repo_path == 'plugins/logging'
        assert result.plugins[2].source == '/local/path/to/plugin'

    @pytest.mark.asyncio
    async def test_build_start_conversation_request_for_user_with_plugins(self):
        """Test _build_start_conversation_request_for_user passes plugins to finalize method."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        # Arrange
        self.mock_user_context.get_user_info.return_value = self.mock_user
        self.mock_user_context.get_secrets.return_value = {}
        self.mock_user_context.get_provider_tokens = AsyncMock(return_value=None)
        self.mock_user_context.get_mcp_api_key.return_value = None

        plugins = [
            PluginSpec(
                source='https://github.com/org/plugin.git',
                ref='main',
                parameters={'config_file': 'custom.yaml'},
            )
        ]

        # Mock _finalize_conversation_request to capture the call
        mock_finalize = AsyncMock(return_value=Mock(spec=StartConversationRequest))
        self.service._finalize_conversation_request = mock_finalize

        # Act
        await self.service._build_start_conversation_request_for_user(
            self.mock_sandbox,
            None,
            None,
            None,
            '/workspace',
            plugins=plugins,
        )

        # Assert
        mock_finalize.assert_called_once()
        call_kwargs = mock_finalize.call_args.kwargs
        assert call_kwargs['plugins'] == plugins

    @pytest.mark.asyncio
    async def test_build_start_conversation_request_for_user_without_plugins(self):
        """Test _build_start_conversation_request_for_user works without plugins."""
        # Arrange
        self.mock_user_context.get_user_info.return_value = self.mock_user
        self.mock_user_context.get_secrets.return_value = {}
        self.mock_user_context.get_provider_tokens = AsyncMock(return_value=None)
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Mock _finalize_conversation_request
        mock_finalize = AsyncMock(return_value=Mock(spec=StartConversationRequest))
        self.service._finalize_conversation_request = mock_finalize

        # Act
        await self.service._build_start_conversation_request_for_user(
            self.mock_sandbox,
            None,
            None,
            None,
            '/workspace',
        )

        # Assert
        mock_finalize.assert_called_once()
        call_kwargs = mock_finalize.call_args.kwargs
        assert call_kwargs.get('plugins') is None


class TestPluginSpecModel:
    """Test cases for the PluginSpec model."""

    def test_plugin_spec_with_all_fields(self):
        """Test PluginSpec with all fields provided."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        plugin = PluginSpec(
            source='github:owner/repo',
            ref='v1.0.0',
            repo_path='plugins/my-plugin',
            parameters={'key1': 'value1', 'key2': 123, 'key3': True},
        )

        assert plugin.source == 'github:owner/repo'
        assert plugin.ref == 'v1.0.0'
        assert plugin.repo_path == 'plugins/my-plugin'
        assert plugin.parameters == {'key1': 'value1', 'key2': 123, 'key3': True}

    def test_plugin_spec_with_only_source(self):
        """Test PluginSpec with only source provided."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        plugin = PluginSpec(source='https://github.com/owner/repo.git')

        assert plugin.source == 'https://github.com/owner/repo.git'
        assert plugin.ref is None
        assert plugin.repo_path is None
        assert plugin.parameters is None

    def test_plugin_spec_serialization(self):
        """Test PluginSpec serialization to JSON."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        plugin = PluginSpec(
            source='github:owner/repo',
            ref='main',
            repo_path='plugins/my-plugin',
            parameters={'debug': True},
        )

        json_data = plugin.model_dump()
        assert json_data == {
            'source': 'github:owner/repo',
            'ref': 'main',
            'repo_path': 'plugins/my-plugin',
            'parameters': {'debug': True},
        }

    def test_plugin_spec_deserialization(self):
        """Test PluginSpec deserialization from dict."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        data = {
            'source': 'github:owner/repo',
            'ref': 'v2.0.0',
            'repo_path': 'plugins/weather',
            'parameters': {'timeout': 30},
        }

        plugin = PluginSpec.model_validate(data)

        assert plugin.source == 'github:owner/repo'
        assert plugin.ref == 'v2.0.0'
        assert plugin.repo_path == 'plugins/weather'
        assert plugin.parameters == {'timeout': 30}

    def test_plugin_spec_display_name_github_format(self):
        """Test display_name extracts repo name from github:owner/repo format."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        plugin = PluginSpec(source='github:owner/my-plugin')
        assert plugin.display_name == 'my-plugin'

    def test_plugin_spec_display_name_git_url(self):
        """Test display_name extracts repo name from git URL."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        plugin = PluginSpec(source='https://github.com/owner/repo.git')
        assert plugin.display_name == 'repo.git'

    def test_plugin_spec_display_name_local_path(self):
        """Test display_name extracts directory name from local path."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        plugin = PluginSpec(source='/local/path/to/plugin')
        assert plugin.display_name == 'plugin'

    def test_plugin_spec_display_name_no_slash(self):
        """Test display_name returns source as-is when no slash present."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        plugin = PluginSpec(source='local-plugin')
        assert plugin.display_name == 'local-plugin'

    def test_plugin_spec_format_params_as_text(self):
        """Test format_params_as_text formats parameters as text."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        plugin = PluginSpec(
            source='github:owner/repo',
            parameters={'key1': 'value1', 'key2': 123},
        )

        result = plugin.format_params_as_text()
        assert result == '- key1: value1\n- key2: 123'

    def test_plugin_spec_format_params_as_text_with_indent(self):
        """Test format_params_as_text with custom indent."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        plugin = PluginSpec(
            source='github:owner/repo',
            parameters={'debug': True},
        )

        result = plugin.format_params_as_text(indent='  ')
        assert result == '  - debug: True'

    def test_plugin_spec_format_params_as_text_no_params(self):
        """Test format_params_as_text returns None when no parameters."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        plugin = PluginSpec(source='github:owner/repo')
        assert plugin.format_params_as_text() is None

    def test_plugin_spec_inherits_repo_path_validation(self):
        """Test PluginSpec inherits validation from SDK's PluginSource."""
        import pytest

        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        # Should reject absolute paths
        with pytest.raises(ValueError, match='must be relative'):
            PluginSpec(source='github:owner/repo', repo_path='/absolute/path')

        # Should reject parent traversal
        with pytest.raises(ValueError, match="cannot contain '..'"):
            PluginSpec(source='github:owner/repo', repo_path='../parent/path')


class TestAppConversationStartRequestWithPlugins:
    """Test cases for AppConversationStartRequest with plugins field."""

    def test_start_request_with_plugins(self):
        """Test AppConversationStartRequest with plugins field."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            AppConversationStartRequest,
            PluginSpec,
        )

        plugins = [
            PluginSpec(
                source='github:owner/my-plugin',
                ref='v1.0.0',
                parameters={'api_key': 'test'},
            )
        ]

        request = AppConversationStartRequest(
            title='Test conversation',
            plugins=plugins,
        )

        assert request.plugins is not None
        assert len(request.plugins) == 1
        assert request.plugins[0].source == 'github:owner/my-plugin'
        assert request.plugins[0].ref == 'v1.0.0'
        assert request.plugins[0].parameters == {'api_key': 'test'}

    def test_start_request_without_plugins(self):
        """Test AppConversationStartRequest without plugins field (backwards compatible)."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            AppConversationStartRequest,
        )

        request = AppConversationStartRequest(
            title='Test conversation',
        )

        assert request.plugins is None

    def test_start_request_serialization_with_plugins(self):
        """Test AppConversationStartRequest serialization includes plugins."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            AppConversationStartRequest,
            PluginSpec,
        )

        plugins = [PluginSpec(source='github:owner/repo')]
        request = AppConversationStartRequest(plugins=plugins)

        json_data = request.model_dump()

        assert 'plugins' in json_data
        assert len(json_data['plugins']) == 1
        assert json_data['plugins'][0]['source'] == 'github:owner/repo'

    def test_start_request_deserialization_with_plugins(self):
        """Test AppConversationStartRequest deserialization from JSON with plugins."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            AppConversationStartRequest,
        )

        data = {
            'title': 'Test',
            'plugins': [
                {
                    'source': 'github:owner/plugin',
                    'ref': 'main',
                    'parameters': {'key': 'value'},
                },
            ],
        }

        request = AppConversationStartRequest.model_validate(data)

        assert request.plugins is not None
        assert len(request.plugins) == 1
        assert request.plugins[0].source == 'github:owner/plugin'
        assert request.plugins[0].ref == 'main'
        assert request.plugins[0].parameters == {'key': 'value'}

    def test_start_request_with_multiple_plugins(self):
        """Test AppConversationStartRequest with multiple plugins."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            AppConversationStartRequest,
            PluginSpec,
        )

        plugins = [
            PluginSpec(source='github:owner/plugin1', ref='v1.0.0'),
            PluginSpec(source='github:owner/plugin2', repo_path='plugins/sub'),
            PluginSpec(source='/local/path'),
        ]

        request = AppConversationStartRequest(
            title='Test conversation',
            plugins=plugins,
        )

        assert request.plugins is not None
        assert len(request.plugins) == 3
        assert request.plugins[0].source == 'github:owner/plugin1'
        assert request.plugins[1].repo_path == 'plugins/sub'
        assert request.plugins[2].source == '/local/path'
