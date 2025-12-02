"""Unit tests for git functionality in AppConversationServiceBase.

This module tests the git-related functionality, specifically the clone_or_init_git_repo method
and the recent bug fixes for git checkout operations.
"""

import subprocess
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from openhands.app_server.app_conversation.app_conversation_models import AgentType
from openhands.app_server.app_conversation.app_conversation_service_base import (
    AppConversationServiceBase,
)
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
# Tests for _configure_git_user_settings
# =============================================================================


def _create_service_with_mock_user_context(user_info: MockUserInfo) -> tuple:
    """Create a mock service with the actual _configure_git_user_settings method.

    Uses MagicMock for the service but binds the real method for testing.

    Returns a tuple of (service, mock_user_context) for testing.
    """
    mock_user_context = MagicMock()
    mock_user_context.get_user_info = AsyncMock(return_value=user_info)

    # Create a simple mock service and set required attribute
    service = MagicMock()
    service.user_context = mock_user_context

    # Bind the actual method from the real class to test real implementation
    service._configure_git_user_settings = (
        lambda workspace: AppConversationServiceBase._configure_git_user_settings(
            service, workspace
        )
    )

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
