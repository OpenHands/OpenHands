"""Tests for skill_loader module.

This module tests the loading of skills from the agent-server,
which centralizes all skill loading logic. The app-server acts as a
thin proxy that builds configs and calls the agent-server's /api/skills endpoint.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from openhands.app_server.app_conversation.skill_loader import (
    OrgConfig,
    SandboxConfig,
    SkillInfo,
    _convert_skill_info_to_skill,
    _determine_org_repo_path,
    _get_org_repository_url,
    _get_provider_type,
    _is_azure_devops_repository,
    _is_gitlab_repository,
    build_org_config,
    build_sandbox_config,
    load_skills_from_agent_server,
)
from openhands.app_server.sandbox.sandbox_models import (
    ExposedUrl,
    SandboxInfo,
    SandboxStatus,
)
from openhands.app_server.user.user_context import UserContext
from openhands.integrations.provider import ProviderType
from openhands.integrations.service_types import AuthenticationError
from openhands.sdk.context.skills import KeywordTrigger, Skill, TaskTrigger

# ===== Test Fixtures =====


@pytest.fixture
def mock_skill():
    """Create a mock Skill object."""
    skill = Mock()
    skill.name = 'test_skill'
    skill.content = 'Test content'
    return skill


@pytest.fixture
def mock_skills_list():
    """Create a list of mock Skill objects."""
    skills = []
    for i in range(3):
        skill = Mock()
        skill.name = f'skill_{i}'
        skill.content = f'Content {i}'
        skills.append(skill)
    return skills


@pytest.fixture
def mock_user_context():
    """Create a mock UserContext."""
    return AsyncMock(spec=UserContext)


@pytest.fixture
def mock_sandbox_info():
    """Create a mock SandboxInfo with exposed URLs."""
    return SandboxInfo(
        id='test-sandbox-123',
        created_by_user_id='user-123',
        sandbox_spec_id='spec-123',
        status=SandboxStatus.RUNNING,
        session_api_key='test-api-key',
        exposed_urls=[
            ExposedUrl(name='AGENT_SERVER', url='http://localhost:8000', port=8000),
            ExposedUrl(name='VSCODE', url='http://localhost:8080', port=8080),
        ],
    )


@pytest.fixture
def mock_sandbox_info_no_urls():
    """Create a mock SandboxInfo without exposed URLs."""
    return SandboxInfo(
        id='test-sandbox-123',
        created_by_user_id='user-123',
        sandbox_spec_id='spec-123',
        status=SandboxStatus.RUNNING,
        session_api_key='test-api-key',
        exposed_urls=None,
    )


# ===== Tests for New Functions =====


class TestGetProviderType:
    """Test _get_provider_type function."""

    @pytest.mark.asyncio
    @patch('openhands.app_server.app_conversation.skill_loader._is_gitlab_repository')
    @patch(
        'openhands.app_server.app_conversation.skill_loader._is_azure_devops_repository'
    )
    async def test_returns_gitlab_for_gitlab_repo(
        self, mock_is_azure, mock_is_gitlab, mock_user_context
    ):
        """Test returns 'gitlab' for GitLab repository."""
        # Arrange
        mock_is_gitlab.return_value = True
        mock_is_azure.return_value = False

        # Act
        result = await _get_provider_type('owner/repo', mock_user_context)

        # Assert
        assert result == 'gitlab'
        mock_is_gitlab.assert_called_once_with('owner/repo', mock_user_context)

    @pytest.mark.asyncio
    @patch('openhands.app_server.app_conversation.skill_loader._is_gitlab_repository')
    @patch(
        'openhands.app_server.app_conversation.skill_loader._is_azure_devops_repository'
    )
    async def test_returns_azure_for_azure_repo(
        self, mock_is_azure, mock_is_gitlab, mock_user_context
    ):
        """Test returns 'azure' for Azure DevOps repository."""
        # Arrange
        mock_is_gitlab.return_value = False
        mock_is_azure.return_value = True

        # Act
        result = await _get_provider_type('org/project/repo', mock_user_context)

        # Assert
        assert result == 'azure'
        mock_is_azure.assert_called_once_with('org/project/repo', mock_user_context)

    @pytest.mark.asyncio
    @patch('openhands.app_server.app_conversation.skill_loader._is_gitlab_repository')
    @patch(
        'openhands.app_server.app_conversation.skill_loader._is_azure_devops_repository'
    )
    async def test_returns_github_for_github_repo(
        self, mock_is_azure, mock_is_gitlab, mock_user_context
    ):
        """Test returns 'github' for GitHub repository (default)."""
        # Arrange
        mock_is_gitlab.return_value = False
        mock_is_azure.return_value = False

        # Act
        result = await _get_provider_type('owner/repo', mock_user_context)

        # Assert
        assert result == 'github'


class TestBuildOrgConfig:
    """Test build_org_config function."""

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.skill_loader._determine_org_repo_path'
    )
    @patch('openhands.app_server.app_conversation.skill_loader._get_org_repository_url')
    @patch('openhands.app_server.app_conversation.skill_loader._get_provider_type')
    async def test_builds_config_successfully(
        self, mock_get_provider, mock_get_url, mock_determine_path, mock_user_context
    ):
        """Test successfully building org config."""
        # Arrange
        mock_determine_path.return_value = ('owner/.openhands', 'owner')
        mock_get_url.return_value = 'https://token@github.com/owner/.openhands.git'
        mock_get_provider.return_value = 'github'

        # Act
        result = await build_org_config('owner/repo', mock_user_context)

        # Assert
        assert result is not None
        assert isinstance(result, OrgConfig)
        assert result.repository == 'owner/repo'
        assert result.provider == 'github'
        assert result.org_repo_url == 'https://token@github.com/owner/.openhands.git'
        assert result.org_name == 'owner'

    @pytest.mark.asyncio
    async def test_returns_none_when_no_repository(self, mock_user_context):
        """Test returns None when selected_repository is None."""
        # Act
        result = await build_org_config(None, mock_user_context)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_repository_has_insufficient_parts(
        self, mock_user_context
    ):
        """Test returns None when repository path has less than 2 parts."""
        # Act
        result = await build_org_config('repo', mock_user_context)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.skill_loader._determine_org_repo_path'
    )
    @patch('openhands.app_server.app_conversation.skill_loader._get_org_repository_url')
    async def test_returns_none_when_url_not_available(
        self, mock_get_url, mock_determine_path, mock_user_context
    ):
        """Test returns None when org repository URL cannot be retrieved."""
        # Arrange
        mock_determine_path.return_value = ('owner/.openhands', 'owner')
        mock_get_url.return_value = None

        # Act
        result = await build_org_config('owner/repo', mock_user_context)

        # Assert
        assert result is None


class TestBuildSandboxConfig:
    """Test build_sandbox_config function."""

    def test_builds_config_with_exposed_urls(self, mock_sandbox_info):
        """Test building sandbox config with exposed URLs."""
        # Act
        result = build_sandbox_config(mock_sandbox_info)

        # Assert
        assert result is not None
        assert isinstance(result, SandboxConfig)
        assert len(result.exposed_urls) == 2
        assert result.exposed_urls[0].name == 'AGENT_SERVER'
        assert result.exposed_urls[0].url == 'http://localhost:8000'
        assert result.exposed_urls[0].port == 8000

    def test_returns_none_when_no_exposed_urls(self, mock_sandbox_info_no_urls):
        """Test returns None when sandbox has no exposed URLs."""
        # Act
        result = build_sandbox_config(mock_sandbox_info_no_urls)

        # Assert
        assert result is None

    def test_returns_none_when_exposed_urls_is_empty_list(self):
        """Test returns None when exposed_urls is empty list."""
        # Arrange
        sandbox = SandboxInfo(
            id='test-sandbox',
            created_by_user_id='user-123',
            sandbox_spec_id='spec-123',
            status=SandboxStatus.RUNNING,
            session_api_key='test-key',
            exposed_urls=[],
        )

        # Act
        result = build_sandbox_config(sandbox)

        # Assert
        assert result is None


class TestConvertSkillInfoToSkill:
    """Test _convert_skill_info_to_skill function."""

    def test_converts_skill_with_keyword_trigger(self):
        """Test converting skill data with keyword triggers."""
        # Arrange
        skill_info = SkillInfo(
            name='test_skill',
            content='Test content',
            triggers=['test', 'testing'],
            source='repo',
            description='A test skill',
        )

        # Act
        skill = _convert_skill_info_to_skill(skill_info)

        # Assert
        assert isinstance(skill, Skill)
        assert skill.name == 'test_skill'
        assert skill.content == 'Test content'
        assert isinstance(skill.trigger, KeywordTrigger)
        assert skill.trigger.keywords == ['test', 'testing']
        assert skill.source == 'repo'
        assert skill.description == 'A test skill'

    def test_converts_skill_with_task_trigger(self):
        """Test converting skill data with task triggers (starting with /)."""
        # Arrange
        skill_info = SkillInfo(
            name='task_skill',
            content='Task content',
            triggers=['/task1', '/task2'],
            source='org',
        )

        # Act
        skill = _convert_skill_info_to_skill(skill_info)

        # Assert
        assert isinstance(skill, Skill)
        assert skill.name == 'task_skill'
        assert isinstance(skill.trigger, TaskTrigger)
        assert skill.trigger.triggers == ['/task1', '/task2']

    def test_converts_skill_without_trigger(self):
        """Test converting skill data without triggers."""
        # Arrange
        skill_info = SkillInfo(
            name='repo_skill',
            content='Repo content',
            source='project',
        )

        # Act
        skill = _convert_skill_info_to_skill(skill_info)

        # Assert
        assert isinstance(skill, Skill)
        assert skill.name == 'repo_skill'
        assert skill.trigger is None

    def test_converts_skill_with_empty_triggers(self):
        """Test converting skill data with empty triggers list."""
        # Arrange
        skill_info = SkillInfo(
            name='skill',
            content='Content',
            triggers=[],
        )

        # Act
        skill = _convert_skill_info_to_skill(skill_info)

        # Assert
        assert skill.trigger is None


class TestLoadSkillsFromAgentServer:
    """Test load_skills_from_agent_server function."""

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_loads_skills_successfully(self, mock_client_class):
        """Test successfully loading skills from agent-server."""
        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'skills': [
                {
                    'name': 'skill1',
                    'content': 'Content 1',
                    'triggers': ['keyword1'],
                },
                {
                    'name': 'skill2',
                    'content': 'Content 2',
                    'triggers': [],
                },
            ],
            'sources': {'public': 1, 'user': 1},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        # Act
        result = await load_skills_from_agent_server(
            agent_server_url='http://localhost:8000',
            session_api_key='test-key',
            project_dir='/workspace/project',
            org_config=OrgConfig(
                repository='owner/repo',
                provider='github',
                org_repo_url='https://github.com/owner/.openhands.git',
                org_name='owner',
            ),
            sandbox_config=SandboxConfig(exposed_urls=[]),
        )

        # Assert
        assert len(result) == 2
        assert result[0].name == 'skill1'
        assert result[1].name == 'skill2'
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == 'http://localhost:8000/api/skills'
        assert 'X-Session-API-Key' in call_args[1]['headers']
        assert call_args[1]['headers']['X-Session-API-Key'] == 'test-key'

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_handles_http_status_error(self, mock_client_class):
        """Test handling HTTP status error from agent-server."""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = 'Internal Server Error'

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                'Server error', request=MagicMock(), response=mock_response
            )
        )
        mock_client_class.return_value = mock_client

        # Act
        result = await load_skills_from_agent_server(
            agent_server_url='http://localhost:8000',
            session_api_key='test-key',
            project_dir='/workspace',
        )

        # Assert
        assert result == []

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_handles_request_error(self, mock_client_class):
        """Test handling request error (connection failure)."""
        # Arrange
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(
            side_effect=httpx.RequestError('Connection failed')
        )
        mock_client_class.return_value = mock_client

        # Act
        result = await load_skills_from_agent_server(
            agent_server_url='http://localhost:8000',
            session_api_key='test-key',
            project_dir='/workspace',
        )

        # Assert
        assert result == []


# ===== Tests for Organization Skills Functions (Still Existing) =====


class TestIsGitlabRepository:
    """Test _is_gitlab_repository helper function."""

    @pytest.mark.asyncio
    async def test_is_gitlab_repository_true(self):
        """Test GitLab repository detection returns True."""
        # Arrange
        mock_user_context = AsyncMock()
        mock_provider_handler = AsyncMock()
        mock_repository = Mock()
        mock_repository.git_provider = ProviderType.GITLAB

        mock_user_context.get_provider_handler.return_value = mock_provider_handler
        mock_provider_handler.verify_repo_provider.return_value = mock_repository

        # Act
        result = await _is_gitlab_repository('owner/repo', mock_user_context)

        # Assert
        assert result is True
        mock_provider_handler.verify_repo_provider.assert_called_once_with(
            'owner/repo', is_optional=True
        )

    @pytest.mark.asyncio
    async def test_is_gitlab_repository_false(self):
        """Test non-GitLab repository detection returns False."""
        # Arrange
        mock_user_context = AsyncMock()
        mock_provider_handler = AsyncMock()
        mock_repository = Mock()
        mock_repository.git_provider = ProviderType.GITHUB

        mock_user_context.get_provider_handler.return_value = mock_provider_handler
        mock_provider_handler.verify_repo_provider.return_value = mock_repository

        # Act
        result = await _is_gitlab_repository('owner/repo', mock_user_context)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_is_gitlab_repository_exception_handling(self):
        """Test exception handling returns False."""
        # Arrange
        mock_user_context = AsyncMock()
        mock_user_context.get_provider_handler.side_effect = Exception('API error')

        # Act
        result = await _is_gitlab_repository('owner/repo', mock_user_context)

        # Assert
        assert result is False


class TestIsAzureDevOpsRepository:
    """Test _is_azure_devops_repository helper function."""

    @pytest.mark.asyncio
    async def test_is_azure_devops_repository_true(self):
        """Test Azure DevOps repository detection returns True."""
        # Arrange
        mock_user_context = AsyncMock()
        mock_provider_handler = AsyncMock()
        mock_repository = Mock()
        mock_repository.git_provider = ProviderType.AZURE_DEVOPS

        mock_user_context.get_provider_handler.return_value = mock_provider_handler
        mock_provider_handler.verify_repo_provider.return_value = mock_repository

        # Act
        result = await _is_azure_devops_repository(
            'org/project/repo', mock_user_context
        )

        # Assert
        assert result is True
        mock_provider_handler.verify_repo_provider.assert_called_once_with(
            'org/project/repo', is_optional=True
        )

    @pytest.mark.asyncio
    async def test_is_azure_devops_repository_false(self):
        """Test non-Azure DevOps repository detection returns False."""
        # Arrange
        mock_user_context = AsyncMock()
        mock_provider_handler = AsyncMock()
        mock_repository = Mock()
        mock_repository.git_provider = ProviderType.GITHUB

        mock_user_context.get_provider_handler.return_value = mock_provider_handler
        mock_provider_handler.verify_repo_provider.return_value = mock_repository

        # Act
        result = await _is_azure_devops_repository('owner/repo', mock_user_context)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_is_azure_devops_repository_exception_handling(self):
        """Test exception handling returns False."""
        # Arrange
        mock_user_context = AsyncMock()
        mock_user_context.get_provider_handler.side_effect = Exception('Network error')

        # Act
        result = await _is_azure_devops_repository('owner/repo', mock_user_context)

        # Assert
        assert result is False


class TestDetermineOrgRepoPath:
    """Test _determine_org_repo_path helper function."""

    @pytest.mark.asyncio
    @patch('openhands.app_server.app_conversation.skill_loader._is_gitlab_repository')
    @patch(
        'openhands.app_server.app_conversation.skill_loader._is_azure_devops_repository'
    )
    async def test_github_repository_path(self, mock_is_azure, mock_is_gitlab):
        """Test org path for GitHub repository."""
        # Arrange
        mock_user_context = AsyncMock()
        mock_is_gitlab.return_value = False
        mock_is_azure.return_value = False

        # Act
        org_repo, org_name = await _determine_org_repo_path(
            'owner/repo', mock_user_context
        )

        # Assert
        assert org_repo == 'owner/.openhands'
        assert org_name == 'owner'

    @pytest.mark.asyncio
    @patch('openhands.app_server.app_conversation.skill_loader._is_gitlab_repository')
    @patch(
        'openhands.app_server.app_conversation.skill_loader._is_azure_devops_repository'
    )
    async def test_gitlab_repository_path(self, mock_is_azure, mock_is_gitlab):
        """Test org path for GitLab repository."""
        # Arrange
        mock_user_context = AsyncMock()
        mock_is_gitlab.return_value = True
        mock_is_azure.return_value = False

        # Act
        org_repo, org_name = await _determine_org_repo_path(
            'owner/repo', mock_user_context
        )

        # Assert
        assert org_repo == 'owner/openhands-config'
        assert org_name == 'owner'

    @pytest.mark.asyncio
    @patch('openhands.app_server.app_conversation.skill_loader._is_gitlab_repository')
    @patch(
        'openhands.app_server.app_conversation.skill_loader._is_azure_devops_repository'
    )
    async def test_azure_devops_repository_path(self, mock_is_azure, mock_is_gitlab):
        """Test org path for Azure DevOps repository."""
        # Arrange
        mock_user_context = AsyncMock()
        mock_is_gitlab.return_value = False
        mock_is_azure.return_value = True

        # Act
        org_repo, org_name = await _determine_org_repo_path(
            'org/project/repo', mock_user_context
        )

        # Assert
        assert org_repo == 'org/openhands-config/openhands-config'
        assert org_name == 'org'


class TestGetOrgRepositoryUrl:
    """Test _get_org_repository_url helper function."""

    @pytest.mark.asyncio
    async def test_successful_url_retrieval(self):
        """Test successfully retrieving authenticated URL."""
        # Arrange
        mock_user_context = AsyncMock()
        expected_url = 'https://token@github.com/owner/.openhands.git'
        mock_user_context.get_authenticated_git_url.return_value = expected_url

        # Act
        result = await _get_org_repository_url('owner/.openhands', mock_user_context)

        # Assert
        assert result == expected_url
        mock_user_context.get_authenticated_git_url.assert_called_once_with(
            'owner/.openhands', is_optional=True
        )

    @pytest.mark.asyncio
    async def test_authentication_error(self):
        """Test handling of authentication error returns None."""
        # Arrange
        mock_user_context = AsyncMock()
        mock_user_context.get_authenticated_git_url.side_effect = AuthenticationError(
            'Not found'
        )

        # Act
        result = await _get_org_repository_url('owner/.openhands', mock_user_context)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_general_exception(self):
        """Test handling of general exception returns None."""
        # Arrange
        mock_user_context = AsyncMock()
        mock_user_context.get_authenticated_git_url.side_effect = Exception(
            'Network error'
        )

        # Act
        result = await _get_org_repository_url('owner/.openhands', mock_user_context)

        # Assert
        assert result is None
