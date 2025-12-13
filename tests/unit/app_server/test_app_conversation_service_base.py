"""Unit tests for git and security functionality in AppConversationServiceBase.

This module tests the git-related functionality, specifically the clone_or_init_git_repo method
and the recent bug fixes for git checkout operations.
"""

import subprocess
from types import MethodType
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest

from openhands.app_server.app_conversation.app_conversation_models import AgentType
from openhands.app_server.app_conversation.app_conversation_service_base import (
    AppConversationServiceBase,
)
from openhands.app_server.sandbox.sandbox_models import SandboxInfo
from openhands.app_server.user.user_context import UserContext


class MockUserInfo:
    """Mock class for UserInfo to simulate user settings."""

    def __init__(
        self, git_user_name: str | None = None, git_user_email: str | None = None
    ):
        self.git_user_name = git_user_name
        self.git_user_email = git_user_email


class MockCommandResult:
    """Mock class for command execution result."""

    def __init__(self, exit_code: int = 0, stderr: str = ''):
        self.exit_code = exit_code
        self.stderr = stderr


class MockWorkspace:
    """Mock class for AsyncRemoteWorkspace."""

    def __init__(self, working_dir: str = '/workspace'):
        self.working_dir = working_dir
        self.execute_command = AsyncMock(return_value=MockCommandResult())


class MockAppConversationServiceBase:
    """Mock class to test git functionality without complex dependencies."""

    def __init__(self):
        self.logger = MagicMock()

    async def clone_or_init_git_repo(
        self,
        workspace_path: str,
        repo_url: str,
        branch: str = 'main',
        timeout: int = 300,
    ) -> bool:
        """Clone or initialize a git repository.

        This is a simplified version of the actual method for testing purposes.
        """
        try:
            # Try to clone the repository
            clone_result = subprocess.run(
                ['git', 'clone', '--branch', branch, repo_url, workspace_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if clone_result.returncode == 0:
                self.logger.info(
                    f'Successfully cloned repository {repo_url} to {workspace_path}'
                )
                return True

            # If clone fails, try to checkout the branch
            checkout_result = subprocess.run(
                ['git', 'checkout', branch],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if checkout_result.returncode == 0:
                self.logger.info(f'Successfully checked out branch {branch}')
                return True
            else:
                self.logger.error(
                    f'Failed to checkout branch {branch}: {checkout_result.stderr}'
                )
                return False

        except subprocess.TimeoutExpired:
            self.logger.error(f'Git operation timed out after {timeout} seconds')
            return False
        except Exception as e:
            self.logger.error(f'Git operation failed: {str(e)}')
            return False


@pytest.fixture
def service():
    """Create a mock service instance for testing."""
    return MockAppConversationServiceBase()


@pytest.mark.asyncio
async def test_clone_or_init_git_repo_successful_clone(service):
    """Test successful git clone operation."""
    with patch('subprocess.run') as mock_run:
        # Mock successful clone
        mock_run.return_value = MagicMock(returncode=0, stderr='', stdout='Cloning...')

        result = await service.clone_or_init_git_repo(
            workspace_path='/tmp/test_repo',
            repo_url='https://github.com/test/repo.git',
            branch='main',
            timeout=300,
        )

        assert result is True
        mock_run.assert_called_once_with(
            [
                'git',
                'clone',
                '--branch',
                'main',
                'https://github.com/test/repo.git',
                '/tmp/test_repo',
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        service.logger.info.assert_called_with(
            'Successfully cloned repository https://github.com/test/repo.git to /tmp/test_repo'
        )


@pytest.mark.asyncio
async def test_clone_or_init_git_repo_clone_fails_checkout_succeeds(service):
    """Test git clone fails but checkout succeeds."""
    with patch('subprocess.run') as mock_run:
        # Mock clone failure, then checkout success
        mock_run.side_effect = [
            MagicMock(returncode=1, stderr='Clone failed', stdout=''),  # Clone fails
            MagicMock(
                returncode=0, stderr='', stdout='Switched to branch'
            ),  # Checkout succeeds
        ]

        result = await service.clone_or_init_git_repo(
            workspace_path='/tmp/test_repo',
            repo_url='https://github.com/test/repo.git',
            branch='feature-branch',
            timeout=300,
        )

        assert result is True
        assert mock_run.call_count == 2

        # Check clone call
        mock_run.assert_any_call(
            [
                'git',
                'clone',
                '--branch',
                'feature-branch',
                'https://github.com/test/repo.git',
                '/tmp/test_repo',
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Check checkout call
        mock_run.assert_any_call(
            ['git', 'checkout', 'feature-branch'],
            cwd='/tmp/test_repo',
            capture_output=True,
            text=True,
            timeout=300,
        )

        service.logger.info.assert_called_with(
            'Successfully checked out branch feature-branch'
        )


@pytest.mark.asyncio
async def test_clone_or_init_git_repo_both_operations_fail(service):
    """Test both git clone and checkout operations fail."""
    with patch('subprocess.run') as mock_run:
        # Mock both operations failing
        mock_run.side_effect = [
            MagicMock(returncode=1, stderr='Clone failed', stdout=''),  # Clone fails
            MagicMock(
                returncode=1, stderr='Checkout failed', stdout=''
            ),  # Checkout fails
        ]

        result = await service.clone_or_init_git_repo(
            workspace_path='/tmp/test_repo',
            repo_url='https://github.com/test/repo.git',
            branch='nonexistent-branch',
            timeout=300,
        )

        assert result is False
        assert mock_run.call_count == 2
        service.logger.error.assert_called_with(
            'Failed to checkout branch nonexistent-branch: Checkout failed'
        )


@pytest.mark.asyncio
async def test_clone_or_init_git_repo_timeout(service):
    """Test git operation timeout."""
    with patch('subprocess.run') as mock_run:
        # Mock timeout exception
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=['git', 'clone'], timeout=300
        )

        result = await service.clone_or_init_git_repo(
            workspace_path='/tmp/test_repo',
            repo_url='https://github.com/test/repo.git',
            branch='main',
            timeout=300,
        )

        assert result is False
        service.logger.error.assert_called_with(
            'Git operation timed out after 300 seconds'
        )


@pytest.mark.asyncio
async def test_clone_or_init_git_repo_exception(service):
    """Test git operation with unexpected exception."""
    with patch('subprocess.run') as mock_run:
        # Mock unexpected exception
        mock_run.side_effect = Exception('Unexpected error')

        result = await service.clone_or_init_git_repo(
            workspace_path='/tmp/test_repo',
            repo_url='https://github.com/test/repo.git',
            branch='main',
            timeout=300,
        )

        assert result is False
        service.logger.error.assert_called_with(
            'Git operation failed: Unexpected error'
        )


@pytest.mark.asyncio
async def test_clone_or_init_git_repo_custom_timeout(service):
    """Test git operation with custom timeout."""
    with patch('subprocess.run') as mock_run:
        # Mock successful clone with custom timeout
        mock_run.return_value = MagicMock(returncode=0, stderr='', stdout='Cloning...')

        result = await service.clone_or_init_git_repo(
            workspace_path='/tmp/test_repo',
            repo_url='https://github.com/test/repo.git',
            branch='main',
            timeout=600,  # Custom timeout
        )

        assert result is True
        mock_run.assert_called_once_with(
            [
                'git',
                'clone',
                '--branch',
                'main',
                'https://github.com/test/repo.git',
                '/tmp/test_repo',
            ],
            capture_output=True,
            text=True,
            timeout=600,  # Verify custom timeout is used
        )


@patch(
    'openhands.app_server.app_conversation.app_conversation_service_base.LLMSummarizingCondenser'
)
def test_create_condenser_default_agent_with_none_max_size(mock_condenser_class):
    """Test _create_condenser for DEFAULT agent with condenser_max_size = None uses default."""
    # Arrange
    mock_user_context = Mock(spec=UserContext)
    with patch.object(
        AppConversationServiceBase,
        '__abstractmethods__',
        set(),
    ):
        service = AppConversationServiceBase(
            init_git_in_empty_workspace=True,
            user_context=mock_user_context,
        )
        mock_llm = MagicMock()
        mock_llm_copy = MagicMock()
        mock_llm_copy.usage_id = 'condenser'
        mock_llm.model_copy.return_value = mock_llm_copy
        mock_condenser_instance = MagicMock()
        mock_condenser_class.return_value = mock_condenser_instance

        # Act
        service._create_condenser(mock_llm, AgentType.DEFAULT, None)

        # Assert
        mock_condenser_class.assert_called_once()
        call_kwargs = mock_condenser_class.call_args[1]
        # When condenser_max_size is None, max_size should not be passed (uses SDK default of 120)
        assert 'max_size' not in call_kwargs
        # keep_first is never passed (uses SDK default of 4)
        assert 'keep_first' not in call_kwargs
        assert call_kwargs['llm'].usage_id == 'condenser'
        mock_llm.model_copy.assert_called_once()


@patch(
    'openhands.app_server.app_conversation.app_conversation_service_base.LLMSummarizingCondenser'
)
def test_create_condenser_default_agent_with_custom_max_size(mock_condenser_class):
    """Test _create_condenser for DEFAULT agent with custom condenser_max_size."""
    # Arrange
    mock_user_context = Mock(spec=UserContext)
    with patch.object(
        AppConversationServiceBase,
        '__abstractmethods__',
        set(),
    ):
        service = AppConversationServiceBase(
            init_git_in_empty_workspace=True,
            user_context=mock_user_context,
        )
        mock_llm = MagicMock()
        mock_llm_copy = MagicMock()
        mock_llm_copy.usage_id = 'condenser'
        mock_llm.model_copy.return_value = mock_llm_copy
        mock_condenser_instance = MagicMock()
        mock_condenser_class.return_value = mock_condenser_instance

        # Act
        service._create_condenser(mock_llm, AgentType.DEFAULT, 150)

        # Assert
        mock_condenser_class.assert_called_once()
        call_kwargs = mock_condenser_class.call_args[1]
        assert call_kwargs['max_size'] == 150  # Custom value should be used
        # keep_first is never passed (uses SDK default of 4)
        assert 'keep_first' not in call_kwargs
        assert call_kwargs['llm'].usage_id == 'condenser'
        mock_llm.model_copy.assert_called_once()


@patch(
    'openhands.app_server.app_conversation.app_conversation_service_base.LLMSummarizingCondenser'
)
def test_create_condenser_plan_agent_with_none_max_size(mock_condenser_class):
    """Test _create_condenser for PLAN agent with condenser_max_size = None uses default."""
    # Arrange
    mock_user_context = Mock(spec=UserContext)
    with patch.object(
        AppConversationServiceBase,
        '__abstractmethods__',
        set(),
    ):
        service = AppConversationServiceBase(
            init_git_in_empty_workspace=True,
            user_context=mock_user_context,
        )
        mock_llm = MagicMock()
        mock_llm_copy = MagicMock()
        mock_llm_copy.usage_id = 'planning_condenser'
        mock_llm.model_copy.return_value = mock_llm_copy
        mock_condenser_instance = MagicMock()
        mock_condenser_class.return_value = mock_condenser_instance

        # Act
        service._create_condenser(mock_llm, AgentType.PLAN, None)

        # Assert
        mock_condenser_class.assert_called_once()
        call_kwargs = mock_condenser_class.call_args[1]
        # When condenser_max_size is None, max_size should not be passed (uses SDK default of 120)
        assert 'max_size' not in call_kwargs
        # keep_first is never passed (uses SDK default of 4)
        assert 'keep_first' not in call_kwargs
        assert call_kwargs['llm'].usage_id == 'planning_condenser'
        mock_llm.model_copy.assert_called_once()


@patch(
    'openhands.app_server.app_conversation.app_conversation_service_base.LLMSummarizingCondenser'
)
def test_create_condenser_plan_agent_with_custom_max_size(mock_condenser_class):
    """Test _create_condenser for PLAN agent with custom condenser_max_size."""
    # Arrange
    mock_user_context = Mock(spec=UserContext)
    with patch.object(
        AppConversationServiceBase,
        '__abstractmethods__',
        set(),
    ):
        service = AppConversationServiceBase(
            init_git_in_empty_workspace=True,
            user_context=mock_user_context,
        )
        mock_llm = MagicMock()
        mock_llm_copy = MagicMock()
        mock_llm_copy.usage_id = 'planning_condenser'
        mock_llm.model_copy.return_value = mock_llm_copy
        mock_condenser_instance = MagicMock()
        mock_condenser_class.return_value = mock_condenser_instance

        # Act
        service._create_condenser(mock_llm, AgentType.PLAN, 200)

        # Assert
        mock_condenser_class.assert_called_once()
        call_kwargs = mock_condenser_class.call_args[1]
        assert call_kwargs['max_size'] == 200  # Custom value should be used
        # keep_first is never passed (uses SDK default of 4)
        assert 'keep_first' not in call_kwargs
        assert call_kwargs['llm'].usage_id == 'planning_condenser'
        mock_llm.model_copy.assert_called_once()


# =============================================================================
# Tests for security analyzer helpers
# =============================================================================


@pytest.mark.parametrize('value', [None, '', 'none', 'NoNe'])
def test_create_security_analyzer_returns_none_for_empty_values(value):
    """_create_security_analyzer_from_string returns None for empty/none values."""
    # Arrange
    service, _ = _create_service_with_mock_user_context(
        MockUserInfo(), bind_methods=('_create_security_analyzer_from_string',)
    )

    # Act
    result = service._create_security_analyzer_from_string(value)

    # Assert
    assert result is None


def test_create_security_analyzer_returns_llm_analyzer():
    """_create_security_analyzer_from_string returns LLMSecurityAnalyzer for llm string."""
    # Arrange
    security_analyzer_str = 'llm'
    service, _ = _create_service_with_mock_user_context(
        MockUserInfo(), bind_methods=('_create_security_analyzer_from_string',)
    )

    # Act
    result = service._create_security_analyzer_from_string(security_analyzer_str)

    # Assert
    from openhands.sdk.security.llm_analyzer import LLMSecurityAnalyzer

    assert isinstance(result, LLMSecurityAnalyzer)


def test_create_security_analyzer_logs_warning_for_unknown_value():
    """_create_security_analyzer_from_string logs warning and returns None for unknown."""
    # Arrange
    unknown_value = 'custom'
    service, _ = _create_service_with_mock_user_context(
        MockUserInfo(), bind_methods=('_create_security_analyzer_from_string',)
    )

    # Act
    with patch(
        'openhands.app_server.app_conversation.app_conversation_service_base._logger'
    ) as mock_logger:
        result = service._create_security_analyzer_from_string(unknown_value)

    # Assert
    assert result is None
    mock_logger.warning.assert_called_once()


def test_select_confirmation_policy_when_disabled_returns_never_confirm():
    """_select_confirmation_policy returns NeverConfirm when confirmation_mode is False."""
    # Arrange
    confirmation_mode = False
    security_analyzer = 'llm'
    service, _ = _create_service_with_mock_user_context(
        MockUserInfo(), bind_methods=('_select_confirmation_policy',)
    )

    # Act
    policy = service._select_confirmation_policy(confirmation_mode, security_analyzer)

    # Assert
    from openhands.sdk.security.confirmation_policy import NeverConfirm

    assert isinstance(policy, NeverConfirm)


def test_select_confirmation_policy_llm_returns_confirm_risky():
    """_select_confirmation_policy uses ConfirmRisky when analyzer is llm."""
    # Arrange
    confirmation_mode = True
    security_analyzer = 'llm'
    service, _ = _create_service_with_mock_user_context(
        MockUserInfo(), bind_methods=('_select_confirmation_policy',)
    )

    # Act
    policy = service._select_confirmation_policy(confirmation_mode, security_analyzer)

    # Assert
    from openhands.sdk.security.confirmation_policy import ConfirmRisky

    assert isinstance(policy, ConfirmRisky)


@pytest.mark.parametrize('security_analyzer', [None, '', 'none', 'custom'])
def test_select_confirmation_policy_non_llm_returns_always_confirm(
    security_analyzer,
):
    """_select_confirmation_policy falls back to AlwaysConfirm for non-llm values."""
    # Arrange
    confirmation_mode = True
    service, _ = _create_service_with_mock_user_context(
        MockUserInfo(), bind_methods=('_select_confirmation_policy',)
    )

    # Act
    policy = service._select_confirmation_policy(confirmation_mode, security_analyzer)

    # Assert
    from openhands.sdk.security.confirmation_policy import AlwaysConfirm

    assert isinstance(policy, AlwaysConfirm)


@pytest.mark.asyncio
async def test_set_security_analyzer_skips_when_no_session_key():
    """_set_security_analyzer_from_settings exits early without session_api_key."""
    # Arrange
    agent_server_url = 'https://agent.example.com'
    conversation_id = uuid4()
    httpx_client = AsyncMock()
    service, _ = _create_service_with_mock_user_context(
        MockUserInfo(),
        bind_methods=(
            '_create_security_analyzer_from_string',
            '_set_security_analyzer_from_settings',
        ),
    )

    with patch.object(service, '_create_security_analyzer_from_string') as mock_create:
        # Act
        await service._set_security_analyzer_from_settings(
            agent_server_url=agent_server_url,
            session_api_key=None,
            conversation_id=conversation_id,
            security_analyzer_str='llm',
            httpx_client=httpx_client,
        )

    # Assert
    mock_create.assert_not_called()
    httpx_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_set_security_analyzer_skips_when_analyzer_none():
    """_set_security_analyzer_from_settings skips API call when analyzer resolves to None."""
    # Arrange
    agent_server_url = 'https://agent.example.com'
    session_api_key = 'session-key'
    conversation_id = uuid4()
    httpx_client = AsyncMock()
    service, _ = _create_service_with_mock_user_context(
        MockUserInfo(),
        bind_methods=(
            '_create_security_analyzer_from_string',
            '_set_security_analyzer_from_settings',
        ),
    )

    with patch.object(
        service, '_create_security_analyzer_from_string', return_value=None
    ) as mock_create:
        # Act
        await service._set_security_analyzer_from_settings(
            agent_server_url=agent_server_url,
            session_api_key=session_api_key,
            conversation_id=conversation_id,
            security_analyzer_str='none',
            httpx_client=httpx_client,
        )

    # Assert
    mock_create.assert_called_once_with('none')
    httpx_client.post.assert_not_called()


class DummyAnalyzer:
    """Simple analyzer stub for testing model_dump contract."""

    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self) -> dict:
        return self._payload


@pytest.mark.asyncio
async def test_set_security_analyzer_successfully_calls_agent_server():
    """_set_security_analyzer_from_settings posts analyzer payload when available."""
    # Arrange
    agent_server_url = 'https://agent.example.com'
    session_api_key = 'session-key'
    conversation_id = uuid4()
    analyzer_payload = {'type': 'llm'}
    httpx_client = AsyncMock()
    http_response = MagicMock()
    http_response.raise_for_status = MagicMock()
    httpx_client.post.return_value = http_response
    service, _ = _create_service_with_mock_user_context(
        MockUserInfo(),
        bind_methods=(
            '_create_security_analyzer_from_string',
            '_set_security_analyzer_from_settings',
        ),
    )

    analyzer = DummyAnalyzer(analyzer_payload)

    with (
        patch.object(
            service,
            '_create_security_analyzer_from_string',
            return_value=analyzer,
        ) as mock_create,
        patch(
            'openhands.app_server.app_conversation.app_conversation_service_base._logger'
        ) as mock_logger,
    ):
        # Act
        await service._set_security_analyzer_from_settings(
            agent_server_url=agent_server_url,
            session_api_key=session_api_key,
            conversation_id=conversation_id,
            security_analyzer_str='llm',
            httpx_client=httpx_client,
        )

    # Assert
    mock_create.assert_called_once_with('llm')
    httpx_client.post.assert_awaited_once_with(
        f'{agent_server_url}/api/conversations/{conversation_id}/security_analyzer',
        json={'security_analyzer': analyzer_payload},
        headers={'X-Session-API-Key': session_api_key},
        timeout=30.0,
    )
    http_response.raise_for_status.assert_called_once()
    mock_logger.info.assert_called()


@pytest.mark.asyncio
async def test_set_security_analyzer_logs_warning_on_failure():
    """_set_security_analyzer_from_settings warns but does not raise on errors."""
    # Arrange
    agent_server_url = 'https://agent.example.com'
    session_api_key = 'session-key'
    conversation_id = uuid4()
    analyzer_payload = {'type': 'llm'}
    httpx_client = AsyncMock()
    httpx_client.post.side_effect = RuntimeError('network down')
    service, _ = _create_service_with_mock_user_context(
        MockUserInfo(),
        bind_methods=(
            '_create_security_analyzer_from_string',
            '_set_security_analyzer_from_settings',
        ),
    )

    analyzer = DummyAnalyzer(analyzer_payload)

    with (
        patch.object(
            service,
            '_create_security_analyzer_from_string',
            return_value=analyzer,
        ) as mock_create,
        patch(
            'openhands.app_server.app_conversation.app_conversation_service_base._logger'
        ) as mock_logger,
    ):
        # Act
        await service._set_security_analyzer_from_settings(
            agent_server_url=agent_server_url,
            session_api_key=session_api_key,
            conversation_id=conversation_id,
            security_analyzer_str='llm',
            httpx_client=httpx_client,
        )

    # Assert
    mock_create.assert_called_once_with('llm')
    httpx_client.post.assert_awaited_once()
    mock_logger.warning.assert_called()


# =============================================================================
# Tests for _configure_git_user_settings
# =============================================================================


def _create_service_with_mock_user_context(
    user_info: MockUserInfo, bind_methods: tuple[str, ...] | None = None
) -> tuple:
    """Create a mock service with selected real methods bound for testing.

    Uses MagicMock for the service but binds the real method for testing.

    Returns a tuple of (service, mock_user_context) for testing.
    """
    mock_user_context = MagicMock()
    mock_user_context.get_user_info = AsyncMock(return_value=user_info)

    # Create a simple mock service and set required attribute
    service = MagicMock()
    service.user_context = mock_user_context
    methods_to_bind = ['_configure_git_user_settings']
    if bind_methods:
        methods_to_bind.extend(bind_methods)
        # Remove potential duplicates while keeping order
        methods_to_bind = list(dict.fromkeys(methods_to_bind))

    # Bind actual methods from the real class to test implementations directly
    for method_name in methods_to_bind:
        real_method = getattr(AppConversationServiceBase, method_name)
        setattr(service, method_name, MethodType(real_method, service))

    return service, mock_user_context


@pytest.fixture
def mock_workspace():
    """Create a mock workspace instance for testing."""
    return MockWorkspace(working_dir='/workspace/project')


@pytest.mark.asyncio
async def test_configure_git_user_settings_both_name_and_email(mock_workspace):
    """Test configuring both git user name and email."""
    user_info = MockUserInfo(
        git_user_name='Test User', git_user_email='test@example.com'
    )
    service, mock_user_context = _create_service_with_mock_user_context(user_info)

    await service._configure_git_user_settings(mock_workspace)

    # Verify get_user_info was called
    mock_user_context.get_user_info.assert_called_once()

    # Verify both git config commands were executed
    assert mock_workspace.execute_command.call_count == 2

    # Check git config user.name call
    mock_workspace.execute_command.assert_any_call(
        'git config --global user.name "Test User"', '/workspace/project'
    )

    # Check git config user.email call
    mock_workspace.execute_command.assert_any_call(
        'git config --global user.email "test@example.com"', '/workspace/project'
    )


@pytest.mark.asyncio
async def test_configure_git_user_settings_only_name(mock_workspace):
    """Test configuring only git user name."""
    user_info = MockUserInfo(git_user_name='Test User', git_user_email=None)
    service, _ = _create_service_with_mock_user_context(user_info)

    await service._configure_git_user_settings(mock_workspace)

    # Verify only user.name was configured
    assert mock_workspace.execute_command.call_count == 1
    mock_workspace.execute_command.assert_called_once_with(
        'git config --global user.name "Test User"', '/workspace/project'
    )


@pytest.mark.asyncio
async def test_configure_git_user_settings_only_email(mock_workspace):
    """Test configuring only git user email."""
    user_info = MockUserInfo(git_user_name=None, git_user_email='test@example.com')
    service, _ = _create_service_with_mock_user_context(user_info)

    await service._configure_git_user_settings(mock_workspace)

    # Verify only user.email was configured
    assert mock_workspace.execute_command.call_count == 1
    mock_workspace.execute_command.assert_called_once_with(
        'git config --global user.email "test@example.com"', '/workspace/project'
    )


@pytest.mark.asyncio
async def test_configure_git_user_settings_neither_set(mock_workspace):
    """Test when neither git user name nor email is set."""
    user_info = MockUserInfo(git_user_name=None, git_user_email=None)
    service, _ = _create_service_with_mock_user_context(user_info)

    await service._configure_git_user_settings(mock_workspace)

    # Verify no git config commands were executed
    mock_workspace.execute_command.assert_not_called()


@pytest.mark.asyncio
async def test_configure_git_user_settings_empty_strings(mock_workspace):
    """Test when git user name and email are empty strings."""
    user_info = MockUserInfo(git_user_name='', git_user_email='')
    service, _ = _create_service_with_mock_user_context(user_info)

    await service._configure_git_user_settings(mock_workspace)

    # Empty strings are falsy, so no commands should be executed
    mock_workspace.execute_command.assert_not_called()


@pytest.mark.asyncio
async def test_configure_git_user_settings_get_user_info_fails(mock_workspace):
    """Test handling of exception when get_user_info fails."""
    user_info = MockUserInfo()
    service, mock_user_context = _create_service_with_mock_user_context(user_info)
    mock_user_context.get_user_info = AsyncMock(
        side_effect=Exception('User info error')
    )

    # Should not raise exception, just log warning
    await service._configure_git_user_settings(mock_workspace)

    # Verify no git config commands were executed
    mock_workspace.execute_command.assert_not_called()


@pytest.mark.asyncio
async def test_configure_git_user_settings_name_command_fails(mock_workspace):
    """Test handling when git config user.name command fails."""
    user_info = MockUserInfo(
        git_user_name='Test User', git_user_email='test@example.com'
    )
    service, _ = _create_service_with_mock_user_context(user_info)

    # Make the first command fail (user.name), second succeed (user.email)
    mock_workspace.execute_command = AsyncMock(
        side_effect=[
            MockCommandResult(exit_code=1, stderr='Permission denied'),
            MockCommandResult(exit_code=0),
        ]
    )

    # Should not raise exception
    await service._configure_git_user_settings(mock_workspace)

    # Verify both commands were still attempted
    assert mock_workspace.execute_command.call_count == 2


@pytest.mark.asyncio
async def test_configure_git_user_settings_email_command_fails(mock_workspace):
    """Test handling when git config user.email command fails."""
    user_info = MockUserInfo(
        git_user_name='Test User', git_user_email='test@example.com'
    )
    service, _ = _create_service_with_mock_user_context(user_info)

    # Make the first command succeed (user.name), second fail (user.email)
    mock_workspace.execute_command = AsyncMock(
        side_effect=[
            MockCommandResult(exit_code=0),
            MockCommandResult(exit_code=1, stderr='Permission denied'),
        ]
    )

    # Should not raise exception
    await service._configure_git_user_settings(mock_workspace)

    # Verify both commands were still attempted
    assert mock_workspace.execute_command.call_count == 2


@pytest.mark.asyncio
async def test_configure_git_user_settings_special_characters_in_name(mock_workspace):
    """Test git user name with special characters."""
    user_info = MockUserInfo(
        git_user_name="Test O'Brien", git_user_email='test@example.com'
    )
    service, _ = _create_service_with_mock_user_context(user_info)

    await service._configure_git_user_settings(mock_workspace)

    # Verify the name is passed with special characters
    mock_workspace.execute_command.assert_any_call(
        'git config --global user.name "Test O\'Brien"', '/workspace/project'
    )


# =============================================================================
# Tests for _load_and_merge_all_skills with org skills
# =============================================================================


class TestLoadAndMergeAllSkillsWithOrgSkills:
    """Test _load_and_merge_all_skills includes organization skills."""

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_sandbox_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_global_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_user_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_org_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_repo_skills'
    )
    async def test_load_and_merge_includes_org_skills(
        self,
        mock_load_repo,
        mock_load_org,
        mock_load_user,
        mock_load_global,
        mock_load_sandbox,
    ):
        """Test that _load_and_merge_all_skills loads and merges org skills."""
        # Arrange
        mock_user_context = Mock(spec=UserContext)
        with patch.object(
            AppConversationServiceBase,
            '__abstractmethods__',
            set(),
        ):
            service = AppConversationServiceBase(
                init_git_in_empty_workspace=True,
                user_context=mock_user_context,
            )

            sandbox = Mock(spec=SandboxInfo)
            sandbox.exposed_urls = []
            remote_workspace = AsyncMock()

            # Create distinct mock skills for each source
            sandbox_skill = Mock()
            sandbox_skill.name = 'sandbox_skill'
            global_skill = Mock()
            global_skill.name = 'global_skill'
            user_skill = Mock()
            user_skill.name = 'user_skill'
            org_skill = Mock()
            org_skill.name = 'org_skill'
            repo_skill = Mock()
            repo_skill.name = 'repo_skill'

            mock_load_sandbox.return_value = [sandbox_skill]
            mock_load_global.return_value = [global_skill]
            mock_load_user.return_value = [user_skill]
            mock_load_org.return_value = [org_skill]
            mock_load_repo.return_value = [repo_skill]

            # Act
            result = await service._load_and_merge_all_skills(
                sandbox, remote_workspace, 'owner/repo', '/workspace'
            )

            # Assert
            assert len(result) == 5
            names = {s.name for s in result}
            assert names == {
                'sandbox_skill',
                'global_skill',
                'user_skill',
                'org_skill',
                'repo_skill',
            }
            mock_load_org.assert_called_once_with(
                remote_workspace, 'owner/repo', '/workspace', mock_user_context
            )

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_sandbox_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_global_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_user_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_org_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_repo_skills'
    )
    async def test_load_and_merge_org_skills_precedence(
        self,
        mock_load_repo,
        mock_load_org,
        mock_load_user,
        mock_load_global,
        mock_load_sandbox,
    ):
        """Test that org skills have correct precedence (higher than user, lower than repo)."""
        # Arrange
        mock_user_context = Mock(spec=UserContext)
        with patch.object(
            AppConversationServiceBase,
            '__abstractmethods__',
            set(),
        ):
            service = AppConversationServiceBase(
                init_git_in_empty_workspace=True,
                user_context=mock_user_context,
            )

            sandbox = Mock(spec=SandboxInfo)
            sandbox.exposed_urls = []
            remote_workspace = AsyncMock()

            # Create skills with same name but different sources
            user_skill = Mock()
            user_skill.name = 'common_skill'
            user_skill.source = 'user'

            org_skill = Mock()
            org_skill.name = 'common_skill'
            org_skill.source = 'org'

            repo_skill = Mock()
            repo_skill.name = 'common_skill'
            repo_skill.source = 'repo'

            mock_load_sandbox.return_value = []
            mock_load_global.return_value = []
            mock_load_user.return_value = [user_skill]
            mock_load_org.return_value = [org_skill]
            mock_load_repo.return_value = [repo_skill]

            # Act
            result = await service._load_and_merge_all_skills(
                sandbox, remote_workspace, 'owner/repo', '/workspace'
            )

            # Assert
            # Should have only one skill with repo source (highest precedence)
            assert len(result) == 1
            assert result[0].source == 'repo'

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_sandbox_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_global_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_user_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_org_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_repo_skills'
    )
    async def test_load_and_merge_org_skills_override_user_skills(
        self,
        mock_load_repo,
        mock_load_org,
        mock_load_user,
        mock_load_global,
        mock_load_sandbox,
    ):
        """Test that org skills override user skills for same name."""
        # Arrange
        mock_user_context = Mock(spec=UserContext)
        with patch.object(
            AppConversationServiceBase,
            '__abstractmethods__',
            set(),
        ):
            service = AppConversationServiceBase(
                init_git_in_empty_workspace=True,
                user_context=mock_user_context,
            )

            sandbox = Mock(spec=SandboxInfo)
            sandbox.exposed_urls = []
            remote_workspace = AsyncMock()

            # Create skills with same name
            user_skill = Mock()
            user_skill.name = 'shared_skill'
            user_skill.priority = 'low'

            org_skill = Mock()
            org_skill.name = 'shared_skill'
            org_skill.priority = 'high'

            mock_load_sandbox.return_value = []
            mock_load_global.return_value = []
            mock_load_user.return_value = [user_skill]
            mock_load_org.return_value = [org_skill]
            mock_load_repo.return_value = []

            # Act
            result = await service._load_and_merge_all_skills(
                sandbox, remote_workspace, 'owner/repo', '/workspace'
            )

            # Assert
            assert len(result) == 1
            assert result[0].priority == 'high'  # Org skill should win

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_sandbox_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_global_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_user_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_org_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_repo_skills'
    )
    async def test_load_and_merge_handles_org_skills_failure(
        self,
        mock_load_repo,
        mock_load_org,
        mock_load_user,
        mock_load_global,
        mock_load_sandbox,
    ):
        """Test that failure to load org skills doesn't break the overall process."""
        # Arrange
        mock_user_context = Mock(spec=UserContext)
        with patch.object(
            AppConversationServiceBase,
            '__abstractmethods__',
            set(),
        ):
            service = AppConversationServiceBase(
                init_git_in_empty_workspace=True,
                user_context=mock_user_context,
            )

            sandbox = Mock(spec=SandboxInfo)
            sandbox.exposed_urls = []
            remote_workspace = AsyncMock()

            global_skill = Mock()
            global_skill.name = 'global_skill'
            repo_skill = Mock()
            repo_skill.name = 'repo_skill'

            mock_load_sandbox.return_value = []
            mock_load_global.return_value = [global_skill]
            mock_load_user.return_value = []
            mock_load_org.return_value = []  # Org skills failed/empty
            mock_load_repo.return_value = [repo_skill]

            # Act
            result = await service._load_and_merge_all_skills(
                sandbox, remote_workspace, 'owner/repo', '/workspace'
            )

            # Assert
            # Should still have skills from other sources
            assert len(result) == 2
            names = {s.name for s in result}
            assert names == {'global_skill', 'repo_skill'}

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_sandbox_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_global_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_user_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_org_skills'
    )
    @patch(
        'openhands.app_server.app_conversation.app_conversation_service_base.load_repo_skills'
    )
    async def test_load_and_merge_no_selected_repository(
        self,
        mock_load_repo,
        mock_load_org,
        mock_load_user,
        mock_load_global,
        mock_load_sandbox,
    ):
        """Test skill loading when no repository is selected."""
        # Arrange
        mock_user_context = Mock(spec=UserContext)
        with patch.object(
            AppConversationServiceBase,
            '__abstractmethods__',
            set(),
        ):
            service = AppConversationServiceBase(
                init_git_in_empty_workspace=True,
                user_context=mock_user_context,
            )

            sandbox = Mock(spec=SandboxInfo)
            sandbox.exposed_urls = []
            remote_workspace = AsyncMock()

            global_skill = Mock()
            global_skill.name = 'global_skill'

            mock_load_sandbox.return_value = []
            mock_load_global.return_value = [global_skill]
            mock_load_user.return_value = []
            mock_load_org.return_value = []
            mock_load_repo.return_value = []

            # Act
            result = await service._load_and_merge_all_skills(
                sandbox, remote_workspace, None, '/workspace'
            )

            # Assert
            assert len(result) == 1
            # Org skills should be called even with None repository
            mock_load_org.assert_called_once_with(
                remote_workspace, None, '/workspace', mock_user_context
            )
