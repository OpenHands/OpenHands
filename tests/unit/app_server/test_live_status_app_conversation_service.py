"""Unit tests for the methods in LiveStatusAppConversationService."""

import io
import json
import os
import zipfile
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from pydantic import SecretStr

from openhands.agent_server.models import (
    SendMessageRequest,
    StartConversationRequest,
    TextContent,
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
    SandboxPage,
    SandboxStatus,
)
from openhands.app_server.sandbox.sandbox_spec_models import SandboxSpecInfo
from openhands.app_server.user.user_context import UserContext
from openhands.integrations.provider import ProviderToken, ProviderType
from openhands.integrations.service_types import SuggestedTask, TaskType
from openhands.sdk import Agent, Event
from openhands.sdk.llm import LLM
from openhands.sdk.secret import LookupSecret, StaticSecret
from openhands.sdk.settings import AgentSettings, ConversationSettings
from openhands.sdk.workspace.remote.async_remote_workspace import AsyncRemoteWorkspace
from openhands.server.types import AppMode
from openhands.storage.data_models.conversation_metadata import ConversationTrigger
from openhands.storage.data_models.settings import SandboxGroupingStrategy, Settings


def _build_test_user_agent_settings(user: SimpleNamespace) -> AgentSettings:
    llm_vals: dict = {}
    model = getattr(user, 'llm_model', '') or ''
    llm_vals['model'] = model

    llm_api_key = getattr(user, 'llm_api_key', None)
    if llm_api_key:
        llm_vals['api_key'] = llm_api_key

    llm_base_url = getattr(user, 'llm_base_url', None)
    if llm_base_url:
        llm_vals['base_url'] = llm_base_url

    agent_vals: dict = {'llm': llm_vals}

    mcp_config = getattr(user, '_mcp_config', None) or getattr(user, 'mcp_config', None)
    if mcp_config:
        agent_vals['mcp_config'] = mcp_config.model_dump(mode='python')

    return Settings(agent_settings=agent_vals).agent_settings


class _TestUserInfo(SimpleNamespace):
    @property
    def agent_settings(self) -> AgentSettings:
        override = getattr(self, '_agent_settings_override', None)
        if override is not None:
            return override
        return _build_test_user_agent_settings(self)

    @agent_settings.setter
    def agent_settings(self, value):
        object.__setattr__(self, '_agent_settings_override', value)

    @property
    def conversation_settings(self) -> ConversationSettings:
        kwargs: dict = {
            'confirmation_mode': getattr(self, 'confirmation_mode', False),
            'security_analyzer': getattr(self, 'security_analyzer', None),
        }
        max_iter = getattr(self, 'max_iterations', None)
        if max_iter is not None:
            kwargs['max_iterations'] = max_iter
        return ConversationSettings(**kwargs)

    def to_agent_settings(self) -> AgentSettings:
        return self.agent_settings


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
        self.mock_pending_message_service = Mock()

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
            pending_message_service=self.mock_pending_message_service,
            sandbox_startup_timeout=30,
            sandbox_startup_poll_frequency=1,
            max_num_conversations_per_sandbox=20,
            httpx_client=self.mock_httpx_client,
            web_url='https://test.example.com',
            openhands_provider_base_url='https://provider.example.com',
            access_token_hard_timeout=None,
            app_mode='test',
        )

        # Mock user info
        self.mock_user = _TestUserInfo(
            id='test_user_123',
            llm_model='gpt-4',
            llm_base_url='https://api.openai.com/v1',
            llm_api_key='test_api_key',
            sandbox_grouping_strategy=SandboxGroupingStrategy.ADD_TO_ANY,
            confirmation_mode=False,
            security_analyzer='llm',
            search_api_key=None,
            mcp_config=None,
            disabled_skills=[],
        )

        # Mock sandbox
        self.mock_sandbox = Mock(spec=SandboxInfo)
        self.mock_sandbox.id = uuid4()
        self.mock_sandbox.status = SandboxStatus.RUNNING

        # Stable conversation ID for tests that call _configure_llm_and_mcp directly
        self.conversation_id = uuid4()

        # Default mock for hooks loading - returns None (no hooks found)
        # Tests that specifically test hooks loading can override this mock
        self.service._load_hooks_from_workspace = AsyncMock(return_value=None)

    def test_apply_suggested_task_sets_prompt_and_trigger(self):
        """Test suggested task prompts populate initial message and trigger."""
        suggested_task = SuggestedTask(
            git_provider=ProviderType.GITHUB,
            task_type=TaskType.UNRESOLVED_COMMENTS,
            repo='owner/repo',
            issue_number=42,
            title='Handle review comments',
        )
        request = AppConversationStartRequest(suggested_task=suggested_task)

        self.service._apply_suggested_task(request)

        assert request.initial_message is not None
        assert (
            request.initial_message.content[0].text
            == suggested_task.get_prompt_for_task()
        )
        assert request.trigger == ConversationTrigger.SUGGESTED_TASK
        assert request.selected_repository == suggested_task.repo
        assert request.git_provider == suggested_task.git_provider

    def test_apply_suggested_task_raises_if_initial_message_present(self):
        suggested_task = SuggestedTask(
            repo='foo/bar',
            git_provider=ProviderType.GITHUB,
            title='Some title',
            task_type=TaskType.OPEN_ISSUE,
            issue_number=123,
        )

        request = AppConversationStartRequest(
            suggested_task=suggested_task,
            initial_message=SendMessageRequest(
                role='user',
                content=[TextContent(text='User provided message')],
            ),
        )

        with pytest.raises(ValueError, match='initial_message cannot be provided'):
            self.service._apply_suggested_task(request)

    def test_apply_suggested_task_raises_if_prompt_empty(self):
        suggested_task = SuggestedTask(
            repo='foo/bar',
            git_provider=ProviderType.GITHUB,
            title='Some title',
            task_type=TaskType.OPEN_ISSUE,
            issue_number=123,
        )
        request = AppConversationStartRequest(suggested_task=suggested_task)

        with patch.object(SuggestedTask, 'get_prompt_for_task', return_value=''):
            with pytest.raises(ValueError, match='empty prompt'):
                self.service._apply_suggested_task(request)

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
            self.mock_user, custom_model, self.conversation_id
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
        assert mcp_config['mcpServers']['default']['headers'][
            'X-OpenHands-ServerConversation-ID'
        ] == str(self.conversation_id)
        assert (
            mcp_config['mcpServers']['default']['headers']['X-Session-API-Key']
            == 'mcp_api_key'
        )

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_uses_user_llm_settings(self):
        """User LLM fields should drive the configured LLM."""
        self.mock_user.llm_model = 'sdk-model'
        self.mock_user.llm_base_url = 'https://sdk-llm.example.com'
        self.mock_user.llm_api_key = 'test-key'
        self.mock_user_context.get_mcp_api_key.return_value = None

        llm, _ = await self.service._configure_llm_and_mcp(
            self.mock_user, None, self.conversation_id
        )

        assert llm.model == 'sdk-model'
        assert llm.base_url == 'https://sdk-llm.example.com'

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_openhands_model_uses_user_base_url(
        self,
    ):
        """openhands/* model uses user's base_url when set."""
        # Arrange
        self.mock_user.llm_model = 'openhands/special'
        self.mock_user.llm_base_url = 'https://user-llm.example.com'
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, _ = await self.service._configure_llm_and_mcp(
            self.mock_user, self.mock_user.llm_model, self.conversation_id
        )

        # Assert — user base_url takes precedence for openhands/ models
        assert llm.base_url == 'https://user-llm.example.com'

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_openhands_model_falls_back_to_provider_url(
        self,
    ):
        """openhands/* model falls back to provider base URL when user has no base_url."""
        # Arrange
        self.mock_user.llm_model = 'openhands/default'
        self.mock_user.llm_base_url = None
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, _ = await self.service._configure_llm_and_mcp(
            self.mock_user, self.mock_user.llm_model, self.conversation_id
        )

        # Assert — falls back to service-level openhands_provider_base_url
        assert llm.base_url == 'https://provider.example.com'

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_openhands_model_no_base_urls(self):
        """openhands/* model still uses the SDK proxy when no other URLs exist."""
        # Arrange
        self.mock_user.llm_model = 'openhands/default'
        self.mock_user.llm_base_url = None
        self.service.openhands_provider_base_url = None
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, _ = await self.service._configure_llm_and_mcp(
            self.mock_user, self.mock_user.llm_model, self.conversation_id
        )

        # Assert
        assert llm.base_url == 'https://llm-proxy.app.all-hands.dev/'

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_litellm_proxy_model_uses_provider_default(
        self,
    ):
        """litellm_proxy/* model (inherited by sub-conversations) falls back to provider base URL."""
        # Arrange — simulates a sub-conversation inheriting the SDK-transformed model name
        self.mock_user.llm_base_url = None
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, _ = await self.service._configure_llm_and_mcp(
            self.mock_user, 'litellm_proxy/minimax-2.5', self.conversation_id
        )

        # Assert
        assert llm.base_url == 'https://provider.example.com'

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_litellm_proxy_model_prefers_user_base_url(
        self,
    ):
        """litellm_proxy/* model uses user.llm_base_url when provided."""
        # Arrange
        self.mock_user.llm_base_url = 'https://user-llm.example.com'
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, _ = await self.service._configure_llm_and_mcp(
            self.mock_user, 'litellm_proxy/minimax-2.5', self.conversation_id
        )

        # Assert
        assert llm.base_url == 'https://user-llm.example.com'

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_non_openhands_model_ignores_provider(self):
        """Non-openhands model ignores provider base URL and uses user base URL."""
        # Arrange
        self.mock_user.llm_model = 'gpt-4'
        self.mock_user.llm_base_url = 'https://user-llm.example.com'
        self.service.openhands_provider_base_url = 'https://provider.example.com'
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, _ = await self.service._configure_llm_and_mcp(
            self.mock_user, None, self.conversation_id
        )

        # Assert
        assert llm.base_url == 'https://user-llm.example.com'

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_with_user_default_model(self):
        """Test _configure_llm_and_mcp using user's default model."""
        # Arrange
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None, self.conversation_id
        )

        # Assert
        assert llm.model == self.mock_user.llm_model
        assert 'mcpServers' in mcp_config
        assert 'default' in mcp_config['mcpServers']

        headers = mcp_config['mcpServers']['default']['headers']
        assert headers['X-OpenHands-ServerConversation-ID'] == str(self.conversation_id)
        assert 'X-Session-API-Key' not in headers

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_without_web_url(self):
        """Test _configure_llm_and_mcp without web URL (no MCP config)."""
        # Arrange
        self.service.web_url = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None, self.conversation_id
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
            self.mock_user, None, self.conversation_id
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
            self.mock_user, None, self.conversation_id
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
            self.mock_user, None, self.conversation_id
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
            self.mock_user, None, self.conversation_id
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
            self.mock_user, None, self.conversation_id
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
            self.mock_user, None, self.conversation_id
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
            self.mock_user, None, self.conversation_id
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
            self.mock_user, None, self.conversation_id
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
        'openhands.app_server.app_conversation.live_status_app_conversation_service.get_default_tools',
        return_value=[],
    )
    @pytest.mark.asyncio
    async def test_build_request_with_skills(self, _mock_tools):
        """Skills are loaded when a remote_workspace is provided."""
        self.mock_user_context.get_user_info.return_value = self.mock_user

        real_llm = LLM(model='gpt-4', api_key=SecretStr('test-key'))
        mock_agent = Mock(spec=Agent)
        mock_agent.llm = real_llm
        mock_agent.condenser = None

        self.service._setup_secrets_for_git_providers = AsyncMock(return_value={})
        self.service._configure_llm_and_mcp = AsyncMock(return_value=(real_llm, {}))
        self.service._load_skills_and_update_agent = AsyncMock(return_value=mock_agent)

        remote_workspace = Mock(spec=AsyncRemoteWorkspace)
        conversation_id = uuid4()

        result = await self.service._build_start_conversation_request_for_user(
            sandbox=self.mock_sandbox,
            conversation_id=conversation_id,
            initial_message=None,
            system_message_suffix=None,
            git_provider=None,
            working_dir='/test/dir',
            remote_workspace=remote_workspace,
            selected_repository='test_repo',
        )

        assert isinstance(result, StartConversationRequest)
        assert result.conversation_id == conversation_id
        self.service._load_skills_and_update_agent.assert_called_once()

    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.get_default_tools',
        return_value=[],
    )
    @pytest.mark.asyncio
    async def test_build_request_without_remote_workspace(self, _mock_tools):
        """Skills loading is skipped when no remote_workspace is provided."""
        self.mock_user_context.get_user_info.return_value = self.mock_user

        real_llm = LLM(model='gpt-4', api_key=SecretStr('test-key'))

        self.service._setup_secrets_for_git_providers = AsyncMock(return_value={})
        self.service._configure_llm_and_mcp = AsyncMock(return_value=(real_llm, {}))

        conversation_id = uuid4()

        result = await self.service._build_start_conversation_request_for_user(
            sandbox=self.mock_sandbox,
            conversation_id=conversation_id,
            initial_message=None,
            system_message_suffix=None,
            git_provider=None,
            working_dir='/test/dir',
            remote_workspace=None,
        )

        assert isinstance(result, StartConversationRequest)
        assert result.conversation_id == conversation_id

    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.get_default_tools',
        return_value=[],
    )
    @pytest.mark.asyncio
    async def test_build_request_skills_loading_fails_gracefully(self, _mock_tools):
        """Conversation still starts when skills loading raises."""
        self.mock_user_context.get_user_info.return_value = self.mock_user

        real_llm = LLM(model='gpt-4', api_key=SecretStr('test-key'))

        self.service._setup_secrets_for_git_providers = AsyncMock(return_value={})
        self.service._configure_llm_and_mcp = AsyncMock(return_value=(real_llm, {}))
        self.service._load_skills_and_update_agent = AsyncMock(
            side_effect=Exception('Skills loading failed')
        )

        remote_workspace = Mock(spec=AsyncRemoteWorkspace)
        conversation_id = uuid4()

        with patch(
            'openhands.app_server.app_conversation.live_status_app_conversation_service._logger'
        ) as mock_logger:
            result = await self.service._build_start_conversation_request_for_user(
                sandbox=self.mock_sandbox,
                conversation_id=conversation_id,
                initial_message=None,
                system_message_suffix=None,
                git_provider=None,
                working_dir='/test/dir',
                remote_workspace=remote_workspace,
                selected_repository='test_repo',
            )

            assert isinstance(result, StartConversationRequest)
            mock_logger.warning.assert_called_once()

    def test_apply_server_overrides_sets_condenser_usage_id(self):
        """Condenser LLM must get usage_id='condenser' even when it inherits 'agent'."""
        from openhands.sdk.context.condenser import LLMSummarizingCondenser

        llm = LLM(model='openhands/gpt-4', api_key='k', usage_id='agent')
        condenser = LLMSummarizingCondenser(llm=llm)
        agent = Agent(llm=llm, tools=[], condenser=condenser)

        updated = self.service._apply_server_agent_overrides(
            agent, AgentType.DEFAULT, {}, uuid4(), 'user-1'
        )

        assert updated.llm.usage_id == 'agent'
        assert updated.condenser.llm.usage_id == 'condenser'

    def test_apply_server_overrides_condenser_non_openhands_model(self):
        """Condenser usage_id is set even for non-openhands models (no metadata)."""
        from openhands.sdk.context.condenser import LLMSummarizingCondenser

        llm = LLM(model='gpt-4', api_key='k', usage_id='agent')
        condenser = LLMSummarizingCondenser(llm=llm)
        agent = Agent(llm=llm, tools=[], condenser=condenser)

        updated = self.service._apply_server_agent_overrides(
            agent, AgentType.DEFAULT, {}, uuid4(), 'user-1'
        )

        # Non-openhands model: main LLM unchanged, but condenser still gets usage_id
        assert updated.condenser.llm.usage_id == 'condenser'

    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.get_default_tools',
        return_value=[],
    )
    @pytest.mark.asyncio
    async def test_build_start_conversation_request_for_user_integration(
        self, _mock_tools
    ):
        """Test the main _build_start_conversation_request_for_user method integration."""
        self.mock_user_context.get_user_info.return_value = self.mock_user

        mock_secrets = {'GITHUB_TOKEN': StaticSecret(value=SecretStr('tok'))}
        real_llm = LLM(model='gpt-4', api_key=SecretStr('test-key'))
        mock_mcp_config = {'default': {'url': 'test'}}
        test_conversation_id = uuid4()

        self.service._setup_secrets_for_git_providers = AsyncMock(
            return_value=mock_secrets
        )
        self.service._configure_llm_and_mcp = AsyncMock(
            return_value=(real_llm, mock_mcp_config)
        )

        result = await self.service._build_start_conversation_request_for_user(
            sandbox=self.mock_sandbox,
            conversation_id=test_conversation_id,
            initial_message=None,
            system_message_suffix='Test suffix',
            git_provider=ProviderType.GITHUB,
            working_dir='/test/dir',
            agent_type=AgentType.DEFAULT,
            llm_model='gpt-4',
            remote_workspace=None,
            selected_repository='test/repo',
        )

        assert isinstance(result, StartConversationRequest)
        assert result.conversation_id == test_conversation_id
        assert result.agent.llm.model == 'gpt-4'
        # Secrets are injected via agent_context
        assert result.agent.agent_context.secrets == mock_secrets
        # System message suffix includes the original suffix and web host context
        assert 'Test suffix' in result.agent.agent_context.system_message_suffix
        assert '<HOST>' in result.agent.agent_context.system_message_suffix
        assert (
            'https://test.example.com'
            in result.agent.agent_context.system_message_suffix
        )
        # Workspace points to the repo subdirectory
        assert result.workspace.working_dir == '/test/dir/repo'

        self.service._setup_secrets_for_git_providers.assert_called_once_with(
            self.mock_user
        )
        self.service._configure_llm_and_mcp.assert_called_once_with(
            self.mock_user, 'gpt-4', test_conversation_id
        )

    @pytest.mark.asyncio
    async def test_find_running_sandbox_for_user_found(self):
        """Test _find_running_sandbox_for_user when a running sandbox is found."""
        # Arrange
        user_id = 'test_user_123'
        self.mock_user_context.get_user_id.return_value = user_id

        # Create mock sandboxes
        running_sandbox = Mock(spec=SandboxInfo)
        running_sandbox.id = 'sandbox_1'
        running_sandbox.status = SandboxStatus.RUNNING
        running_sandbox.created_by_user_id = user_id

        other_user_sandbox = Mock(spec=SandboxInfo)
        other_user_sandbox.id = 'sandbox_2'
        other_user_sandbox.status = SandboxStatus.RUNNING
        other_user_sandbox.created_by_user_id = 'other_user'

        paused_sandbox = Mock(spec=SandboxInfo)
        paused_sandbox.id = 'sandbox_3'
        paused_sandbox.status = SandboxStatus.PAUSED
        paused_sandbox.created_by_user_id = user_id

        # Mock sandbox service search
        mock_page = Mock(spec=SandboxPage)
        mock_page.items = [other_user_sandbox, running_sandbox, paused_sandbox]
        mock_page.next_page_id = None
        self.mock_sandbox_service.search_sandboxes = AsyncMock(return_value=mock_page)

        # Act
        result = await self.service._find_running_sandbox_for_user()

        # Assert
        assert result == running_sandbox
        self.mock_user_context.get_user_id.assert_called_once()
        self.mock_sandbox_service.search_sandboxes.assert_called_once_with(
            page_id=None, limit=100
        )

    @pytest.mark.asyncio
    async def test_find_running_sandbox_for_user_not_found(self):
        """Test _find_running_sandbox_for_user when no running sandbox is found."""
        # Arrange
        user_id = 'test_user_123'
        self.mock_user_context.get_user_id.return_value = user_id

        # Create mock sandboxes (none running for this user)
        other_user_sandbox = Mock(spec=SandboxInfo)
        other_user_sandbox.id = 'sandbox_1'
        other_user_sandbox.status = SandboxStatus.RUNNING
        other_user_sandbox.created_by_user_id = 'other_user'

        paused_sandbox = Mock(spec=SandboxInfo)
        paused_sandbox.id = 'sandbox_2'
        paused_sandbox.status = SandboxStatus.PAUSED
        paused_sandbox.created_by_user_id = user_id

        # Mock sandbox service search
        mock_page = Mock(spec=SandboxPage)
        mock_page.items = [other_user_sandbox, paused_sandbox]
        mock_page.next_page_id = None
        self.mock_sandbox_service.search_sandboxes = AsyncMock(return_value=mock_page)

        # Act
        result = await self.service._find_running_sandbox_for_user()

        # Assert
        assert result is None
        self.mock_user_context.get_user_id.assert_called_once()
        self.mock_sandbox_service.search_sandboxes.assert_called_once_with(
            page_id=None, limit=100
        )

    @pytest.mark.asyncio
    async def test_find_running_sandbox_for_user_exception_handling(self):
        """Test _find_running_sandbox_for_user handles exceptions gracefully."""
        # Arrange
        self.mock_user_context.get_user_id.side_effect = Exception('User context error')

        # Act
        with patch(
            'openhands.app_server.app_conversation.live_status_app_conversation_service._logger'
        ) as mock_logger:
            result = await self.service._find_running_sandbox_for_user()

        # Assert
        assert result is None
        mock_logger.warning.assert_called_once()
        assert (
            'Error finding running sandbox for user'
            in mock_logger.warning.call_args[0][0]
        )

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
    async def test_configure_llm_and_mcp_with_custom_remote_servers(self):
        """Test _configure_llm_and_mcp merges custom remote servers."""
        from openhands.core.config.mcp_config import MCPConfig, RemoteMCPServer

        self.mock_user.mcp_config = MCPConfig(
            mcpServers={
                'linear': RemoteMCPServer(
                    url='https://linear.app/sse', transport='sse', auth='linear_key'
                ),
                'notion': RemoteMCPServer(
                    url='https://notion.com/sse', transport='sse'
                ),
            }
        )
        self.mock_user_context.get_mcp_api_key.return_value = None

        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None, self.conversation_id
        )

        assert isinstance(llm, LLM)
        assert 'mcpServers' in mcp_config

        mcp_servers = mcp_config['mcpServers']
        assert 'default' in mcp_servers
        assert 'linear' in mcp_servers
        assert 'notion' in mcp_servers

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_with_custom_http_servers(self):
        """Test _configure_llm_and_mcp merges custom HTTP servers with timeout."""
        from openhands.core.config.mcp_config import MCPConfig, RemoteMCPServer

        self.mock_user.mcp_config = MCPConfig(
            mcpServers={
                'custom-http': RemoteMCPServer(
                    url='https://example.com/mcp',
                    transport='http',
                    auth='test_key',
                    timeout=120,
                )
            }
        )
        self.mock_user_context.get_mcp_api_key.return_value = None

        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None, self.conversation_id
        )

        assert isinstance(llm, LLM)
        mcp_servers = mcp_config['mcpServers']
        assert 'custom-http' in mcp_servers

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_with_custom_stdio_servers(self):
        """Test _configure_llm_and_mcp merges custom STDIO servers with explicit names."""
        from openhands.core.config.mcp_config import MCPConfig, StdioMCPServer

        self.mock_user.mcp_config = MCPConfig(
            mcpServers={
                'my-custom-server': StdioMCPServer(
                    command='npx',
                    args=['-y', 'my-package'],
                    env={'API_KEY': 'secret'},
                )
            }
        )
        self.mock_user_context.get_mcp_api_key.return_value = None

        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None, self.conversation_id
        )

        assert isinstance(llm, LLM)
        mcp_servers = mcp_config['mcpServers']

        assert 'my-custom-server' in mcp_servers
        server_config = mcp_servers['my-custom-server']
        assert server_config['command'] == 'npx'
        assert server_config['args'] == ['-y', 'my-package']
        assert server_config['env'] == {'API_KEY': 'secret'}

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_merges_system_and_custom_servers(self):
        """Test _configure_llm_and_mcp merges both system and custom MCP servers."""
        from openhands.core.config.mcp_config import (
            MCPConfig,
            RemoteMCPServer,
            StdioMCPServer,
        )

        self.mock_user.search_api_key = SecretStr('tavily_key')
        self.mock_user.mcp_config = MCPConfig(
            mcpServers={
                'custom-sse': RemoteMCPServer(
                    url='https://custom.com/sse', transport='sse'
                ),
                'custom-stdio': StdioMCPServer(command='node', args=['app.js']),
            }
        )
        self.mock_user_context.get_mcp_api_key.return_value = 'mcp_api_key'

        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None, self.conversation_id
        )

        mcp_servers = mcp_config['mcpServers']

        assert 'default' in mcp_servers
        assert 'tavily' in mcp_servers
        assert 'custom-sse' in mcp_servers
        assert 'custom-stdio' in mcp_servers

        assert len(mcp_servers) == 4

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_custom_config_error_handling(self):
        """Test _configure_llm_and_mcp handles invalid custom MCP config gracefully."""
        # Arrange
        invalid_mcp_config = Mock()
        invalid_mcp_config.model_dump.return_value = 'not-a-dict'
        self.mock_user._agent_settings_override = SimpleNamespace(
            mcp_config=invalid_mcp_config
        )
        self.service._configure_llm = Mock(
            return_value=LLM.model_validate({'model': 'gpt-4', 'usage_id': 'agent'})
        )
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None, self.conversation_id
        )

        # Assert - should still return valid config with system servers only
        assert isinstance(llm, LLM)
        mcp_servers = mcp_config['mcpServers']
        assert 'default' in mcp_servers

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_sdk_format_with_mcpservers_wrapper(self):
        """Test _configure_llm_and_mcp returns SDK-required format with mcpServers key."""
        # Arrange
        self.mock_user_context.get_mcp_api_key.return_value = 'mcp_key'

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None, self.conversation_id
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
        from openhands.core.config.mcp_config import MCPConfig

        self.mock_user.mcp_config = MCPConfig(mcpServers={})
        self.mock_user_context.get_mcp_api_key.return_value = None

        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None, self.conversation_id
        )

        mcp_servers = mcp_config['mcpServers']
        assert 'default' in mcp_servers
        assert len(mcp_servers) == 1

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_remote_server_without_auth(self):
        """Test _configure_llm_and_mcp handles remote servers without auth."""
        from openhands.core.config.mcp_config import MCPConfig, RemoteMCPServer

        self.mock_user.mcp_config = MCPConfig(
            mcpServers={
                'public': RemoteMCPServer(url='https://public.com/sse', transport='sse')
            }
        )
        self.mock_user_context.get_mcp_api_key.return_value = None

        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None, self.conversation_id
        )

        mcp_servers = mcp_config['mcpServers']
        assert 'public' in mcp_servers

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_http_server_default_timeout(self):
        """Test _configure_llm_and_mcp handles HTTP servers with default timeout."""
        from openhands.core.config.mcp_config import MCPConfig, RemoteMCPServer

        self.mock_user.mcp_config = MCPConfig(
            mcpServers={
                'http-server': RemoteMCPServer(
                    url='https://example.com/mcp', transport='http'
                )
            }
        )
        self.mock_user_context.get_mcp_api_key.return_value = None

        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None, self.conversation_id
        )

        mcp_servers = mcp_config['mcpServers']
        assert 'http-server' in mcp_servers

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_stdio_server_without_env(self):
        """Test _configure_llm_and_mcp handles STDIO servers without environment variables."""
        from openhands.core.config.mcp_config import MCPConfig, StdioMCPServer

        self.mock_user.mcp_config = MCPConfig(
            mcpServers={
                'simple-server': StdioMCPServer(command='node', args=['app.js'])
            }
        )
        self.mock_user_context.get_mcp_api_key.return_value = None

        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None, self.conversation_id
        )

        mcp_servers = mcp_config['mcpServers']
        assert 'simple-server' in mcp_servers
        server_config = mcp_servers['simple-server']
        assert server_config['command'] == 'node'
        assert server_config['args'] == ['app.js']

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_multiple_servers_same_type(self):
        """Test _configure_llm_and_mcp handles multiple custom servers of the same type."""
        from openhands.core.config.mcp_config import MCPConfig, RemoteMCPServer

        self.mock_user.mcp_config = MCPConfig(
            mcpServers={
                'server1': RemoteMCPServer(
                    url='https://server1.com/sse', transport='sse'
                ),
                'server2': RemoteMCPServer(
                    url='https://server2.com/sse', transport='sse'
                ),
                'server3': RemoteMCPServer(
                    url='https://server3.com/sse', transport='sse'
                ),
            }
        )
        self.mock_user_context.get_mcp_api_key.return_value = None

        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None, self.conversation_id
        )

        mcp_servers = mcp_config['mcpServers']

        assert 'server1' in mcp_servers
        assert 'server2' in mcp_servers
        assert 'server3' in mcp_servers

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_mixed_server_types(self):
        """Test _configure_llm_and_mcp handles all server types together."""
        from openhands.core.config.mcp_config import (
            MCPConfig,
            RemoteMCPServer,
            StdioMCPServer,
        )

        self.mock_user.mcp_config = MCPConfig(
            mcpServers={
                'sse-server': RemoteMCPServer(
                    url='https://sse.example.com/sse',
                    transport='sse',
                    auth='sse_key',
                ),
                'http-server': RemoteMCPServer(
                    url='https://shttp.example.com/mcp',
                    transport='http',
                    timeout=90,
                ),
                'stdio-server': StdioMCPServer(
                    command='npx',
                    args=['mcp-server'],
                    env={'TOKEN': 'value'},
                ),
            }
        )
        self.mock_user_context.get_mcp_api_key.return_value = None

        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None, self.conversation_id
        )

        mcp_servers = mcp_config['mcpServers']

        assert 'sse-server' in mcp_servers
        assert 'http-server' in mcp_servers
        assert 'stdio-server' in mcp_servers

        stdio_server = mcp_servers['stdio-server']
        assert stdio_server['command'] == 'npx'
        assert stdio_server['env'] == {'TOKEN': 'value'}

    # ------------------------------------------------------------------ #
    #  Regression tests: workspace.working_dir == project_dir             #
    # ------------------------------------------------------------------ #

    def test_get_project_dir_with_repo(self):
        """get_project_dir appends repo name to working_dir."""
        from openhands.app_server.app_conversation.app_conversation_service_base import (
            get_project_dir,
        )

        assert (
            get_project_dir('/workspace/project', 'OpenHands/software-agent-sdk')
            == '/workspace/project/software-agent-sdk'
        )
        assert get_project_dir('/w', 'org/repo-name') == '/w/repo-name'

    def test_get_project_dir_without_repo(self):
        """get_project_dir returns working_dir unchanged when no repo selected."""
        from openhands.app_server.app_conversation.app_conversation_service_base import (
            get_project_dir,
        )

        assert get_project_dir('/workspace/project', None) == '/workspace/project'
        assert get_project_dir('/workspace/project', '') == '/workspace/project'

    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.get_default_tools',
        return_value=[],
    )
    @pytest.mark.asyncio
    async def test_build_request_workspace_uses_project_dir(self, _mock_tools):
        """workspace.working_dir in StartConversationRequest must equal project_dir.

        This is the root cause of the V1 hook-stop bug: if workspace.working_dir
        points to the sandbox mount root (/workspace/project) instead of the
        cloned repo (/workspace/project/<repo>), the agent's CWD is wrong and
        .openhands/hooks/on_stop.sh is not found.
        """
        self.mock_user_context.get_user_info.return_value = self.mock_user

        real_llm = LLM(model='gpt-4', api_key=SecretStr('test-key'))

        self.service._setup_secrets_for_git_providers = AsyncMock(return_value={})
        self.service._configure_llm_and_mcp = AsyncMock(return_value=(real_llm, {}))

        result = await self.service._build_start_conversation_request_for_user(
            sandbox=self.mock_sandbox,
            conversation_id=uuid4(),
            initial_message=None,
            system_message_suffix=None,
            git_provider=None,
            working_dir='/workspace/project',
            selected_repository='OpenHands/software-agent-sdk',
        )

        assert (
            result.workspace.working_dir == '/workspace/project/software-agent-sdk'
        ), 'workspace.working_dir must point to the repo root, not the sandbox mount'

    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.get_default_tools',
        return_value=[],
    )
    @pytest.mark.asyncio
    async def test_build_request_no_repo_workspace_unchanged(self, _mock_tools):
        """Without selected_repository, workspace.working_dir == sandbox working_dir."""
        self.mock_user_context.get_user_info.return_value = self.mock_user

        real_llm = LLM(model='gpt-4', api_key=SecretStr('test-key'))

        self.service._setup_secrets_for_git_providers = AsyncMock(return_value={})
        self.service._configure_llm_and_mcp = AsyncMock(return_value=(real_llm, {}))

        result = await self.service._build_start_conversation_request_for_user(
            sandbox=self.mock_sandbox,
            conversation_id=uuid4(),
            initial_message=None,
            system_message_suffix=None,
            git_provider=None,
            working_dir='/workspace/project',
            selected_repository=None,
        )

        assert result.workspace.working_dir == '/workspace/project'

    @pytest.mark.asyncio
    async def test_search_app_conversations_with_sandbox_id_filter(self):
        """Test that search_app_conversations passes sandbox_id__eq to the info service.

        This verifies that the sandbox_id filter is correctly propagated through
        the service layer to the underlying info service.
        """
        from openhands.app_server.app_conversation.app_conversation_models import (
            AppConversationInfoPage,
        )

        # Create test data with different sandbox IDs
        sandbox_id_alpha = 'sandbox-alpha-123'
        sandbox_id_beta = 'sandbox-beta-456'

        conv_alpha = AppConversationInfo(
            id=uuid4(),
            created_by_user_id=None,
            sandbox_id=sandbox_id_alpha,
            title='Alpha Conversation',
        )
        conv_beta = AppConversationInfo(
            id=uuid4(),
            created_by_user_id=None,
            sandbox_id=sandbox_id_beta,
            title='Beta Conversation',
        )

        # Mock the info service to return filtered results based on sandbox_id__eq
        async def mock_search(sandbox_id__eq=None, **kwargs):
            if sandbox_id__eq == sandbox_id_alpha:
                return AppConversationInfoPage(items=[conv_alpha])
            elif sandbox_id__eq == sandbox_id_beta:
                return AppConversationInfoPage(items=[conv_beta])
            else:
                return AppConversationInfoPage(items=[conv_alpha, conv_beta])

        self.mock_app_conversation_info_service.search_app_conversation_info = (
            AsyncMock(side_effect=mock_search)
        )

        # Mock sandbox service to return running status for sandbox lookups
        self.mock_sandbox_service.batch_get_sandboxes = AsyncMock(return_value=[])

        # Test filtering by sandbox_id_alpha
        result = await self.service.search_app_conversations(
            sandbox_id__eq=sandbox_id_alpha
        )

        # Verify the info service was called with the correct sandbox_id__eq
        self.mock_app_conversation_info_service.search_app_conversation_info.assert_called()
        call_kwargs = self.mock_app_conversation_info_service.search_app_conversation_info.call_args[
            1
        ]
        assert call_kwargs.get('sandbox_id__eq') == sandbox_id_alpha

        # Verify only alpha conversation is returned
        assert len(result.items) == 1
        assert result.items[0].sandbox_id == sandbox_id_alpha

    @pytest.mark.asyncio
    async def test_count_app_conversations_with_sandbox_id_filter(self):
        """Test that count_app_conversations passes sandbox_id__eq to the info service.

        This verifies that the sandbox_id filter is correctly propagated through
        the service layer to the underlying info service for count operations.
        """
        sandbox_id = 'sandbox-count-test-789'

        # Mock the info service to return count based on sandbox_id__eq
        async def mock_count(sandbox_id__eq=None, **kwargs):
            if sandbox_id__eq == sandbox_id:
                return 3  # 3 conversations match this sandbox
            else:
                return 10  # 10 total conversations

        self.mock_app_conversation_info_service.count_app_conversation_info = AsyncMock(
            side_effect=mock_count
        )

        # Test counting with sandbox_id filter
        result = await self.service.count_app_conversations(sandbox_id__eq=sandbox_id)

        # Verify the info service was called with the correct sandbox_id__eq
        self.mock_app_conversation_info_service.count_app_conversation_info.assert_called_once()
        call_kwargs = self.mock_app_conversation_info_service.count_app_conversation_info.call_args[
            1
        ]
        assert call_kwargs.get('sandbox_id__eq') == sandbox_id

        # Verify filtered count is returned
        assert result == 3

    @pytest.mark.asyncio
    async def test_search_app_conversations_sandbox_id_filter_returns_empty(self):
        """Test that search with non-matching sandbox_id returns empty results."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            AppConversationInfoPage,
        )

        # Mock the info service to return empty for non-matching sandbox
        self.mock_app_conversation_info_service.search_app_conversation_info = (
            AsyncMock(return_value=AppConversationInfoPage(items=[]))
        )
        self.mock_sandbox_service.batch_get_sandboxes = AsyncMock(return_value=[])

        # Test filtering by non-existent sandbox_id
        result = await self.service.search_app_conversations(
            sandbox_id__eq='non-existent-sandbox'
        )

        # Verify empty results
        assert len(result.items) == 0


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
        self.mock_pending_message_service = Mock()

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
            pending_message_service=self.mock_pending_message_service,
            sandbox_startup_timeout=30,
            sandbox_startup_poll_frequency=1,
            max_num_conversations_per_sandbox=20,
            httpx_client=self.mock_httpx_client,
            web_url='https://test.example.com',
            openhands_provider_base_url='https://provider.example.com',
            access_token_hard_timeout=None,
            app_mode='test',
        )

        # Mock user info
        self.mock_user = _TestUserInfo(
            id='test_user_123',
            llm_model='gpt-4',
            llm_base_url='https://api.openai.com/v1',
            llm_api_key='test_api_key',
            confirmation_mode=False,
            search_api_key=None,
            mcp_config=None,
            security_analyzer=None,
        )

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
        'openhands.app_server.app_conversation.live_status_app_conversation_service.get_default_tools',
        return_value=[],
    )
    async def test_build_request_with_plugins(self, _mock_tools):
        """Plugins are converted to PluginSource and included in the request."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        self.mock_user_context.get_user_info.return_value = self.mock_user

        real_llm = LLM(model='gpt-4', api_key=SecretStr('test-key'))

        self.service._setup_secrets_for_git_providers = AsyncMock(return_value={})
        self.service._configure_llm_and_mcp = AsyncMock(return_value=(real_llm, {}))

        plugins = [
            PluginSpec(
                source='github:owner/my-plugin',
                ref='v1.0.0',
                parameters={'api_key': 'test123'},
            )
        ]

        result = await self.service._build_start_conversation_request_for_user(
            sandbox=self.mock_sandbox,
            conversation_id=uuid4(),
            initial_message=None,
            system_message_suffix=None,
            git_provider=None,
            working_dir='/workspace',
            plugins=plugins,
        )

        assert isinstance(result, StartConversationRequest)
        assert result.plugins is not None
        assert len(result.plugins) == 1
        assert result.plugins[0].source == 'github:owner/my-plugin'
        assert result.plugins[0].ref == 'v1.0.0'
        # Plugin params are folded into the initial message
        assert result.initial_message is not None
        assert (
            'Plugin Configuration Parameters:' in result.initial_message.content[0].text
        )
        assert '- api_key: test123' in result.initial_message.content[0].text

    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.get_default_tools',
        return_value=[],
    )
    @pytest.mark.asyncio
    async def test_build_request_without_plugins(self, _mock_tools):
        """Without plugins, result.plugins is None."""
        self.mock_user_context.get_user_info.return_value = self.mock_user

        real_llm = LLM(model='gpt-4', api_key=SecretStr('test-key'))

        self.service._setup_secrets_for_git_providers = AsyncMock(return_value={})
        self.service._configure_llm_and_mcp = AsyncMock(return_value=(real_llm, {}))

        result = await self.service._build_start_conversation_request_for_user(
            sandbox=self.mock_sandbox,
            conversation_id=uuid4(),
            initial_message=None,
            system_message_suffix=None,
            git_provider=None,
            working_dir='/workspace',
        )

        assert isinstance(result, StartConversationRequest)
        assert result.plugins is None

    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.get_default_tools',
        return_value=[],
    )
    @pytest.mark.asyncio
    async def test_build_request_plugin_with_repo_path(self, _mock_tools):
        """repo_path is propagated through to PluginSource."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        self.mock_user_context.get_user_info.return_value = self.mock_user

        real_llm = LLM(model='gpt-4', api_key=SecretStr('test-key'))

        self.service._setup_secrets_for_git_providers = AsyncMock(return_value={})
        self.service._configure_llm_and_mcp = AsyncMock(return_value=(real_llm, {}))

        plugins = [
            PluginSpec(
                source='github:owner/marketplace-repo',
                ref='main',
                repo_path='plugins/city-weather',
            )
        ]

        result = await self.service._build_start_conversation_request_for_user(
            sandbox=self.mock_sandbox,
            conversation_id=uuid4(),
            initial_message=None,
            system_message_suffix=None,
            git_provider=None,
            working_dir='/workspace',
            plugins=plugins,
        )

        assert result.plugins is not None
        assert len(result.plugins) == 1
        assert result.plugins[0].source == 'github:owner/marketplace-repo'
        assert result.plugins[0].ref == 'main'
        assert result.plugins[0].repo_path == 'plugins/city-weather'

    @patch(
        'openhands.app_server.app_conversation.live_status_app_conversation_service.get_default_tools',
        return_value=[],
    )
    @pytest.mark.asyncio
    async def test_build_request_multiple_plugins(self, _mock_tools):
        """Multiple plugins are all converted correctly."""
        from openhands.app_server.app_conversation.app_conversation_models import (
            PluginSpec,
        )

        self.mock_user_context.get_user_info.return_value = self.mock_user

        real_llm = LLM(model='gpt-4', api_key=SecretStr('test-key'))

        self.service._setup_secrets_for_git_providers = AsyncMock(return_value={})
        self.service._configure_llm_and_mcp = AsyncMock(return_value=(real_llm, {}))

        plugins = [
            PluginSpec(source='github:owner/security-plugin', ref='v2.0.0'),
            PluginSpec(
                source='github:owner/monorepo',
                repo_path='plugins/logging',
            ),
            PluginSpec(source='/local/path/to/plugin'),
        ]

        result = await self.service._build_start_conversation_request_for_user(
            sandbox=self.mock_sandbox,
            conversation_id=uuid4(),
            initial_message=None,
            system_message_suffix=None,
            git_provider=None,
            working_dir='/workspace',
            plugins=plugins,
        )

        assert result.plugins is not None
        assert len(result.plugins) == 3
        assert result.plugins[0].source == 'github:owner/security-plugin'
        assert result.plugins[0].ref == 'v2.0.0'
        assert result.plugins[1].source == 'github:owner/monorepo'
        assert result.plugins[1].repo_path == 'plugins/logging'
        assert result.plugins[2].source == '/local/path/to/plugin'


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


class TestLoadHooksFromWorkspace:
    """Test cases for _load_hooks_from_workspace method."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_user_context = Mock(spec=UserContext)
        self.mock_jwt_service = Mock()
        self.mock_sandbox_service = Mock()
        self.mock_sandbox_spec_service = Mock()
        self.mock_app_conversation_info_service = Mock()
        self.mock_app_conversation_start_task_service = Mock()
        self.mock_event_callback_service = Mock()
        self.mock_event_service = Mock()
        self.mock_httpx_client = AsyncMock()
        self.mock_pending_message_service = Mock()

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
            pending_message_service=self.mock_pending_message_service,
            sandbox_startup_timeout=30,
            sandbox_startup_poll_frequency=1,
            max_num_conversations_per_sandbox=20,
            httpx_client=self.mock_httpx_client,
            web_url='https://test.example.com',
            openhands_provider_base_url='https://provider.example.com',
            access_token_hard_timeout=None,
            app_mode='test',
        )

    @pytest.mark.asyncio
    async def test_load_hooks_from_workspace_success(self):
        """Test loading hooks from workspace when hooks.json exists."""
        # Arrange
        mock_remote_workspace = Mock(spec=AsyncRemoteWorkspace)
        mock_remote_workspace.host = 'http://agent-server:8000'
        mock_remote_workspace._headers = {'X-Session-API-Key': 'test-key'}

        hooks_response = {
            'hook_config': {
                'stop': [
                    {
                        'matcher': '*',
                        'hooks': [{'type': 'command', 'command': 'echo "stop hook"'}],
                    }
                ]
            }
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = hooks_response
        mock_response.raise_for_status = Mock()

        self.mock_httpx_client.post = AsyncMock(return_value=mock_response)

        # Act
        result = await self.service._load_hooks_from_workspace(
            mock_remote_workspace, '/workspace'
        )

        # Assert
        assert result is not None
        assert not result.is_empty()
        self.mock_httpx_client.post.assert_called_once_with(
            'http://agent-server:8000/api/hooks',
            json={'project_dir': '/workspace'},
            headers={
                'Content-Type': 'application/json',
                'X-Session-API-Key': 'test-key',
            },
            timeout=30.0,
        )

    @pytest.mark.asyncio
    async def test_load_hooks_from_workspace_file_not_found(self):
        """Test loading hooks when hooks.json does not exist."""
        # Arrange
        mock_remote_workspace = Mock(spec=AsyncRemoteWorkspace)
        mock_remote_workspace.host = 'http://agent-server:8000'
        mock_remote_workspace._headers = {}

        # Agent server returns hook_config: None when file not found
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'hook_config': None}
        mock_response.raise_for_status = Mock()

        self.mock_httpx_client.post = AsyncMock(return_value=mock_response)

        # Act
        result = await self.service._load_hooks_from_workspace(
            mock_remote_workspace, '/workspace'
        )

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_load_hooks_from_workspace_empty_hooks(self):
        """Test loading hooks when hooks.json is empty or has no hooks."""
        # Arrange
        mock_remote_workspace = Mock(spec=AsyncRemoteWorkspace)
        mock_remote_workspace.host = 'http://agent-server:8000'
        mock_remote_workspace._headers = {}

        # Agent server returns empty hook_config
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'hook_config': {}}
        mock_response.raise_for_status = Mock()

        self.mock_httpx_client.post = AsyncMock(return_value=mock_response)

        # Act
        result = await self.service._load_hooks_from_workspace(
            mock_remote_workspace, '/workspace'
        )

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_load_hooks_from_workspace_http_error(self):
        """Test loading hooks when HTTP request fails."""
        # Arrange
        mock_remote_workspace = Mock(spec=AsyncRemoteWorkspace)
        mock_remote_workspace.host = 'http://agent-server:8000'
        mock_remote_workspace._headers = {}

        self.mock_httpx_client.post = AsyncMock(
            side_effect=Exception('Connection error')
        )

        # Act
        result = await self.service._load_hooks_from_workspace(
            mock_remote_workspace, '/workspace'
        )

        # Assert
        assert result is None

    def test_get_project_dir_for_hooks_with_selected_repository(self):
        """Test get_project_dir_for_hooks with a selected repository."""
        from openhands.app_server.app_conversation.hook_loader import (
            get_project_dir_for_hooks,
        )

        result = get_project_dir_for_hooks(
            '/workspace/project',
            'OpenHands/software-agent-sdk',
        )
        assert result == '/workspace/project/software-agent-sdk'

    def test_get_project_dir_for_hooks_without_selected_repository(self):
        """Test get_project_dir_for_hooks without a selected repository."""
        from openhands.app_server.app_conversation.hook_loader import (
            get_project_dir_for_hooks,
        )

        result = get_project_dir_for_hooks('/workspace/project', None)
        assert result == '/workspace/project'

    def test_get_project_dir_for_hooks_with_empty_string(self):
        """Test get_project_dir_for_hooks with empty string repository."""
        from openhands.app_server.app_conversation.hook_loader import (
            get_project_dir_for_hooks,
        )

        # Empty string should be treated as no repository
        result = get_project_dir_for_hooks('/workspace/project', '')
        assert result == '/workspace/project'

    @pytest.mark.asyncio
    async def test_load_hooks_from_workspace_with_project_dir(self):
        """Test loading hooks with a pre-resolved project_dir.

        The caller is responsible for computing the project_dir (which
        already includes the repo name when a repo is selected).
        _load_hooks_from_workspace should use the project_dir as-is.
        """
        # Arrange
        mock_remote_workspace = Mock(spec=AsyncRemoteWorkspace)
        mock_remote_workspace.host = 'http://agent-server:8000'
        mock_remote_workspace._headers = {'X-Session-API-Key': 'test-key'}

        hooks_response = {
            'hook_config': {
                'stop': [
                    {
                        'matcher': '*',
                        'hooks': [{'type': 'command', 'command': 'echo "stop hook"'}],
                    }
                ]
            }
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = hooks_response
        mock_response.raise_for_status = Mock()

        self.mock_httpx_client.post = AsyncMock(return_value=mock_response)

        # Act - project_dir already includes repo name
        result = await self.service._load_hooks_from_workspace(
            mock_remote_workspace,
            '/workspace/project/software-agent-sdk',
        )

        # Assert
        assert result is not None
        assert not result.is_empty()
        # The project_dir should be passed as-is without doubling
        self.mock_httpx_client.post.assert_called_once_with(
            'http://agent-server:8000/api/hooks',
            json={'project_dir': '/workspace/project/software-agent-sdk'},
            headers={
                'Content-Type': 'application/json',
                'X-Session-API-Key': 'test-key',
            },
            timeout=30.0,
        )

    @pytest.mark.asyncio
    async def test_load_hooks_from_workspace_base_dir(self):
        """Test loading hooks with a base workspace directory (no repo selected)."""
        # Arrange
        mock_remote_workspace = Mock(spec=AsyncRemoteWorkspace)
        mock_remote_workspace.host = 'http://agent-server:8000'
        mock_remote_workspace._headers = {'X-Session-API-Key': 'test-key'}

        hooks_response = {
            'hook_config': {
                'stop': [
                    {
                        'matcher': '*',
                        'hooks': [{'type': 'command', 'command': 'echo "stop hook"'}],
                    }
                ]
            }
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = hooks_response
        mock_response.raise_for_status = Mock()

        self.mock_httpx_client.post = AsyncMock(return_value=mock_response)

        # Act - no repo selected, project_dir is base working_dir
        result = await self.service._load_hooks_from_workspace(
            mock_remote_workspace,
            '/workspace/project',
        )

        # Assert
        assert result is not None
        self.mock_httpx_client.post.assert_called_once_with(
            'http://agent-server:8000/api/hooks',
            json={'project_dir': '/workspace/project'},
            headers={
                'Content-Type': 'application/json',
                'X-Session-API-Key': 'test-key',
            },
            timeout=30.0,
        )
