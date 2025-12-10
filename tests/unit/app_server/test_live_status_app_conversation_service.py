"""Unit tests for the methods in LiveStatusAppConversationService."""

from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

import pytest

from openhands.agent_server.models import SendMessageRequest, StartConversationRequest
from openhands.app_server.app_conversation.app_conversation_models import AgentType
from openhands.app_server.app_conversation.live_status_app_conversation_service import (
    LiveStatusAppConversationService,
)
from openhands.app_server.sandbox.sandbox_models import SandboxInfo, SandboxStatus
from openhands.app_server.user.user_context import UserContext
from openhands.integrations.provider import ProviderType
from openhands.sdk import Agent
from openhands.sdk.llm import LLM
from openhands.sdk.secret import LookupSecret, StaticSecret
from openhands.sdk.workspace import LocalWorkspace
from openhands.sdk.workspace.remote.async_remote_workspace import AsyncRemoteWorkspace
from openhands.server.types import AppMode


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
        self.mock_httpx_client = Mock()

        # Create service instance
        self.service = LiveStatusAppConversationService(
            init_git_in_empty_workspace=True,
            user_context=self.mock_user_context,
            app_conversation_info_service=self.mock_app_conversation_info_service,
            app_conversation_start_task_service=self.mock_app_conversation_start_task_service,
            event_callback_service=self.mock_event_callback_service,
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
            keycloak_auth_cookie=None,
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
        from pydantic import SecretStr

        from openhands.integrations.provider import ProviderToken

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

        # Should be called twice, once for each provider
        assert self.mock_jwt_service.create_jws_token.call_count == 2

    @pytest.mark.asyncio
    async def test_setup_secrets_for_git_providers_with_saas_mode(self):
        """Test _setup_secrets_for_git_providers with SaaS mode (includes keycloak cookie)."""
        # Arrange
        from pydantic import SecretStr

        from openhands.integrations.provider import ProviderToken

        self.service.app_mode = 'saas'
        self.service.keycloak_auth_cookie = 'test_cookie'
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
        assert 'Cookie' in lookup_secret.headers
        assert lookup_secret.headers['Cookie'] == 'keycloak_auth=test_cookie'

    @pytest.mark.asyncio
    async def test_setup_secrets_for_git_providers_without_web_url(self):
        """Test _setup_secrets_for_git_providers without web URL (uses static token)."""
        # Arrange
        from pydantic import SecretStr

        from openhands.integrations.provider import ProviderToken

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
        self.mock_user_context.get_latest_token.assert_called_once_with(
            ProviderType.GITHUB
        )

    @pytest.mark.asyncio
    async def test_setup_secrets_for_git_providers_no_static_token(self):
        """Test _setup_secrets_for_git_providers when no static token is available."""
        # Arrange
        from pydantic import SecretStr

        from openhands.integrations.provider import ProviderToken

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

        assert 'default' in mcp_config
        assert mcp_config['default']['url'] == 'https://test.example.com/mcp/mcp'
        assert mcp_config['default']['headers']['X-Session-API-Key'] == 'mcp_api_key'

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
        assert 'default' in mcp_config
        assert 'headers' not in mcp_config['default']

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
        from pydantic import SecretStr

        self.mock_user.search_api_key = SecretStr('user_search_key')
        self.mock_user_context.get_mcp_api_key.return_value = 'mcp_api_key'

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        assert isinstance(llm, LLM)
        assert 'default' in mcp_config
        assert 'tavily' in mcp_config
        assert (
            mcp_config['tavily']['url']
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
        assert 'default' in mcp_config
        assert 'tavily' in mcp_config
        assert (
            mcp_config['tavily']['url']
            == 'https://mcp.tavily.com/mcp/?tavilyApiKey=env_tavily_key'
        )

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_tavily_user_key_takes_precedence(self):
        """Test _configure_llm_and_mcp user search_api_key takes precedence over env key."""
        # Arrange
        from pydantic import SecretStr

        self.mock_user.search_api_key = SecretStr('user_search_key')
        self.service.tavily_api_key = 'env_tavily_key'
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        assert isinstance(llm, LLM)
        assert 'tavily' in mcp_config
        assert (
            mcp_config['tavily']['url']
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
        assert 'default' in mcp_config
        assert 'tavily' not in mcp_config

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
        assert 'default' in mcp_config
        assert 'tavily' not in mcp_config

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_saas_mode_with_user_search_key(self):
        """Test _configure_llm_and_mcp adds tavily in SAAS mode when user has search_api_key.

        Even in SAAS mode, if the user has their own search_api_key, tavily should be added.
        """
        # Arrange - simulate SAAS mode with user having their own search key
        from pydantic import SecretStr

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
        assert 'default' in mcp_config
        assert 'tavily' in mcp_config
        assert (
            mcp_config['tavily']['url']
            == 'https://mcp.tavily.com/mcp/?tavilyApiKey=user_search_key'
        )

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_tavily_with_empty_user_search_key(self):
        """Test _configure_llm_and_mcp handles empty user search_api_key correctly."""
        # Arrange
        from pydantic import SecretStr

        self.mock_user.search_api_key = SecretStr('')  # Empty string
        self.service.tavily_api_key = 'env_tavily_key'
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        assert isinstance(llm, LLM)
        assert 'tavily' in mcp_config
        # Should fall back to env key since user key is empty
        assert (
            mcp_config['tavily']['url']
            == 'https://mcp.tavily.com/mcp/?tavilyApiKey=env_tavily_key'
        )

    @pytest.mark.asyncio
    async def test_configure_llm_and_mcp_tavily_with_whitespace_user_search_key(self):
        """Test _configure_llm_and_mcp handles whitespace-only user search_api_key correctly."""
        # Arrange
        from pydantic import SecretStr

        self.mock_user.search_api_key = SecretStr('   ')  # Whitespace only
        self.service.tavily_api_key = 'env_tavily_key'
        self.mock_user_context.get_mcp_api_key.return_value = None

        # Act
        llm, mcp_config = await self.service._configure_llm_and_mcp(
            self.mock_user, None
        )

        # Assert
        assert isinstance(llm, LLM)
        assert 'tavily' in mcp_config
        # Should fall back to env key since user key is whitespace only
        assert (
            mcp_config['tavily']['url']
            == 'https://mcp.tavily.com/mcp/?tavilyApiKey=env_tavily_key'
        )

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
            )

            # Assert
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
        mock_updated_agent = Mock(spec=Agent)
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
        mock_updated_agent = Mock(spec=Agent)
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
        mock_updated_agent = Mock(spec=Agent)
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
        )
        self.service._finalize_conversation_request.assert_called_once()
