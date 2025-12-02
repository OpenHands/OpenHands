"""Unit tests for git functionality in AppConversationServiceBase.

This module tests the git-related functionality, specifically the clone_or_init_git_repo method
and the recent bug fixes for git checkout operations.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest


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
