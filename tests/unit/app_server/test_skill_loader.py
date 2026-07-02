"""Tests for skill_loader module.

This module tests the loading of skills from the agent-server,
which centralizes all skill loading logic. The app-server acts as a
thin proxy that builds configs and calls the agent-server's /api/skills endpoint.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from openhands.app_server.app_conversation.skill_loader import (
    _MAX_ORG_CANDIDATES,
    OrgConfig,
    SandboxConfig,
    SkillInfo,
    _convert_skill_info_to_skill,
    _determine_org_repo_path,
    _get_org_repository_url,
    _get_provider_type,
    _is_azure_devops_repository,
    _is_gitlab_repository,
    build_org_configs,
    build_sandbox_config,
    load_skills_from_agent_server,
)
from openhands.app_server.integrations.provider import ProviderType
from openhands.app_server.integrations.service_types import AuthenticationError
from openhands.app_server.sandbox.sandbox_models import (
    ExposedUrl,
    SandboxInfo,
    SandboxStatus,
)
from openhands.app_server.user.user_context import UserContext
from openhands.sdk.skills import KeywordTrigger, Skill, TaskTrigger

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


class TestBuildOrgConfigs:
    """Test build_org_configs function."""

    @staticmethod
    def _make_provider_handler(logins, orgs=None):
        """Build a mock ProviderHandler.

        Args:
            logins: Mapping of ProviderType -> account login.
            orgs: Optional mapping of ProviderType -> list of org/group names.
        """
        orgs = orgs or {}
        handler = MagicMock()
        # Iterating a ProviderHandler's provider_tokens yields ProviderType keys.
        handler.provider_tokens = dict.fromkeys(logins)

        def _get_service(provider):
            service = MagicMock()
            user = MagicMock()
            user.login = logins[provider]
            service.get_user = AsyncMock(return_value=user)
            return service

        handler.get_service = MagicMock(side_effect=_get_service)
        handler.get_github_organizations = AsyncMock(
            return_value=orgs.get(ProviderType.GITHUB, [])
        )
        handler.get_gitlab_groups = AsyncMock(
            return_value=orgs.get(ProviderType.GITLAB, [])
        )
        handler.get_bitbucket_workspaces = AsyncMock(return_value=[])
        handler.get_bitbucket_dc_projects = AsyncMock(return_value=[])
        handler.get_azure_devops_organizations = AsyncMock(
            return_value=orgs.get(ProviderType.AZURE_DEVOPS, [])
        )
        return handler

    @pytest.mark.asyncio
    @patch('openhands.app_server.app_conversation.skill_loader._get_org_repository_url')
    async def test_loads_account_and_orgs_when_no_repository(
        self, mock_get_url, mock_user_context
    ):
        """Account and org repos (both .openhands and .agents) load with no repo."""
        # Arrange
        handler = self._make_provider_handler(
            logins={ProviderType.GITHUB: 'hieptl'},
            orgs={ProviderType.GITHUB: ['acme']},
        )
        mock_user_context.get_provider_handler = AsyncMock(return_value=handler)
        mock_get_url.side_effect = lambda path, ctx: f'https://git/{path}.git'

        # Act
        result = await build_org_configs(None, mock_user_context)

        # Assert
        assert {c.repository for c in result} == {
            'hieptl/.openhands',
            'hieptl/.agents',
            'acme/.openhands',
            'acme/.agents',
        }

    @pytest.mark.asyncio
    @patch('openhands.app_server.app_conversation.skill_loader._get_provider_type')
    @patch(
        'openhands.app_server.app_conversation.skill_loader._determine_org_repo_path'
    )
    @patch('openhands.app_server.app_conversation.skill_loader._get_org_repository_url')
    async def test_includes_selected_repo_owner_both_repos(
        self,
        mock_get_url,
        mock_determine_path,
        mock_get_provider,
        mock_user_context,
    ):
        """A selected GitHub repo loads the owner's .openhands and .agents repos."""
        # Arrange
        handler = self._make_provider_handler(logins={})  # no global owners
        mock_user_context.get_provider_handler = AsyncMock(return_value=handler)
        mock_determine_path.return_value = ('other/.openhands', 'other')
        mock_get_provider.return_value = 'github'
        mock_get_url.side_effect = lambda path, ctx: f'https://git/{path}.git'

        # Act
        result = await build_org_configs('other/web', mock_user_context)

        # Assert
        assert {c.org_repo_url for c in result} == {
            'https://git/other/.openhands.git',
            'https://git/other/.agents.git',
        }

    @pytest.mark.asyncio
    @patch('openhands.app_server.app_conversation.skill_loader._get_provider_type')
    @patch(
        'openhands.app_server.app_conversation.skill_loader._determine_org_repo_path'
    )
    @patch('openhands.app_server.app_conversation.skill_loader._get_org_repository_url')
    async def test_dedupes_when_owner_equals_login(
        self,
        mock_get_url,
        mock_determine_path,
        mock_get_provider,
        mock_user_context,
    ):
        """A repo shared by the account and the selected repo is resolved once."""
        # Arrange
        handler = self._make_provider_handler(logins={ProviderType.GITHUB: 'hieptl'})
        mock_user_context.get_provider_handler = AsyncMock(return_value=handler)
        mock_determine_path.return_value = ('hieptl/.openhands', 'hieptl')
        mock_get_provider.return_value = 'github'
        mock_get_url.side_effect = lambda path, ctx: f'https://git/{path}.git'

        # Act
        await build_org_configs('hieptl/web', mock_user_context)

        # Assert
        resolved_paths = sorted(call.args[0] for call in mock_get_url.call_args_list)
        assert resolved_paths == ['hieptl/.agents', 'hieptl/.openhands']

    @pytest.mark.asyncio
    @patch('openhands.app_server.app_conversation.skill_loader._get_org_repository_url')
    async def test_skips_agents_for_gitlab(self, mock_get_url, mock_user_context):
        """GitLab owners resolve only openhands-config (no .openhands/.agents)."""
        # Arrange
        handler = self._make_provider_handler(logins={ProviderType.GITLAB: 'mygroup'})
        mock_user_context.get_provider_handler = AsyncMock(return_value=handler)
        mock_get_url.side_effect = lambda path, ctx: f'https://git/{path}.git'

        # Act
        result = await build_org_configs(None, mock_user_context)

        # Assert
        assert {c.repository for c in result} == {'mygroup/openhands-config'}

    @pytest.mark.asyncio
    @patch('openhands.app_server.app_conversation.skill_loader._get_org_repository_url')
    async def test_filters_unresolvable_repos(self, mock_get_url, mock_user_context):
        """Repos whose authenticated URL does not resolve are excluded."""
        # Arrange
        handler = self._make_provider_handler(logins={ProviderType.GITHUB: 'hieptl'})
        mock_user_context.get_provider_handler = AsyncMock(return_value=handler)

        def _resolve(path, ctx):
            return f'https://git/{path}.git' if path == 'hieptl/.openhands' else None

        mock_get_url.side_effect = _resolve

        # Act
        result = await build_org_configs(None, mock_user_context)

        # Assert
        assert {c.repository for c in result} == {'hieptl/.openhands'}

    @pytest.mark.asyncio
    @patch('openhands.app_server.app_conversation.skill_loader._get_provider_type')
    @patch(
        'openhands.app_server.app_conversation.skill_loader._determine_org_repo_path'
    )
    @patch('openhands.app_server.app_conversation.skill_loader._get_org_repository_url')
    async def test_selected_repo_is_first_for_legacy_org_config(
        self,
        mock_get_url,
        mock_determine_path,
        mock_get_provider,
        mock_user_context,
    ):
        """The selected repo's config is org_configs[0].

        Older agent-servers read only the first/legacy ``org_config``, so the
        selected repository's org skills must stay first even when global
        account/org repos also resolve.
        """
        # Arrange: the user also has global account + org repos that resolve.
        handler = self._make_provider_handler(
            logins={ProviderType.GITHUB: 'hieptl'},
            orgs={ProviderType.GITHUB: ['acme']},
        )
        mock_user_context.get_provider_handler = AsyncMock(return_value=handler)
        mock_determine_path.return_value = ('selectedorg/.openhands', 'selectedorg')
        mock_get_provider.return_value = 'github'
        mock_get_url.side_effect = lambda path, ctx: f'https://git/{path}.git'

        # Act
        result = await build_org_configs('selectedorg/web', mock_user_context)

        # Assert
        assert result[0].repository == 'selectedorg/.openhands'
        assert result[0].org_name == 'selectedorg'

    @pytest.mark.asyncio
    @patch('openhands.app_server.app_conversation.skill_loader._get_org_repository_url')
    async def test_does_not_enumerate_bitbucket_dc_projects(
        self, mock_get_url, mock_user_context
    ):
        """Bitbucket DC's server-wide project enumeration is never triggered."""
        # Arrange
        handler = self._make_provider_handler(
            logins={ProviderType.BITBUCKET_DATA_CENTER: 'bbuser'}
        )
        mock_user_context.get_provider_handler = AsyncMock(return_value=handler)
        mock_get_url.side_effect = lambda path, ctx: f'https://git/{path}.git'

        # Act
        await build_org_configs(None, mock_user_context)

        # Assert
        handler.get_bitbucket_dc_projects.assert_not_called()

    @pytest.mark.asyncio
    @patch('openhands.app_server.app_conversation.skill_loader._get_org_repository_url')
    async def test_caps_candidate_fan_out(self, mock_get_url, mock_user_context):
        """No more than _MAX_ORG_CANDIDATES repos are verified per conversation."""
        # Arrange: 1 login + 40 orgs => 82 GitHub-style candidate paths.
        orgs = [f'org{i}' for i in range(40)]
        handler = self._make_provider_handler(
            logins={ProviderType.GITHUB: 'hieptl'},
            orgs={ProviderType.GITHUB: orgs},
        )
        mock_user_context.get_provider_handler = AsyncMock(return_value=handler)
        mock_get_url.side_effect = lambda path, ctx: f'https://git/{path}.git'

        # Act
        await build_org_configs(None, mock_user_context)

        # Assert
        assert len(mock_get_url.call_args_list) == _MAX_ORG_CANDIDATES


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

        org_config = OrgConfig(
            repository='owner/repo',
            provider='github',
            org_repo_url='https://github.com/owner/.openhands.git',
            org_name='owner',
        )

        # Act
        result = await load_skills_from_agent_server(
            agent_server_url='http://localhost:8000',
            session_api_key='test-key',
            project_dir='/workspace/project',
            org_configs=[org_config],
            sandbox_config=SandboxConfig(exposed_urls=[]),
        )

        # Assert
        assert len(result) == 2
        assert result[0].name == 'skill1'
        assert result[1].name == 'skill2'
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == 'http://localhost:8000/api/skills'
        assert call_args[1]['headers']['X-Session-API-Key'] == 'test-key'
        # The list form is sent, and the legacy single field mirrors its first entry.
        payload = call_args[1]['json']
        assert payload['org_configs'] == [org_config.model_dump()]
        assert payload['org_config'] == org_config.model_dump()

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


# ===== Tests for registered_marketplaces support =====


class TestLoadSkillsWithMarketplaces:
    """Test load_skills_from_agent_server with registered_marketplaces parameter."""

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_passes_registered_marketplaces_in_payload(self, mock_client_class):
        """Test that registered_marketplaces is included in the API payload."""
        from openhands.app_server.settings.settings_models import (
            MarketplaceRegistration,
        )

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'skills': [],
            'sources': {},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        marketplaces = [
            MarketplaceRegistration(
                name='public',
                source='github:OpenHands/skills',
                auto_load=True,
            ),
            MarketplaceRegistration(
                name='team',
                source='github:acme/plugins',
                ref='v1.0.0',
                repo_path='marketplaces/internal',
            ),
        ]

        # Act
        await load_skills_from_agent_server(
            agent_server_url='http://localhost:8000',
            session_api_key='test-key',
            project_dir='/workspace/project',
            registered_marketplaces=marketplaces,
        )

        # Assert
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        payload = call_args[1]['json']

        assert 'registered_marketplaces' in payload
        assert len(payload['registered_marketplaces']) == 2

        # Verify first marketplace
        mp1 = payload['registered_marketplaces'][0]
        assert mp1['name'] == 'public'
        assert mp1['source'] == 'github:OpenHands/skills'
        assert mp1['auto_load']
        # None values are stripped by model_dump()
        assert 'ref' not in mp1
        assert 'repo_path' not in mp1

        # Verify second marketplace
        mp2 = payload['registered_marketplaces'][1]
        assert mp2['name'] == 'team'
        assert mp2['source'] == 'github:acme/plugins'
        assert mp2['ref'] == 'v1.0.0'
        assert mp2['repo_path'] == 'marketplaces/internal'
        # auto_load=False is included (not stripped like None values)
        assert mp2['auto_load'] is False

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_handles_none_registered_marketplaces(self, mock_client_class):
        """None registered_marketplaces omits the key entirely from the payload.

        The agent-server field is a non-optional list, so sending ``null`` would
        fail validation and drop all skills; omitting the key uses its default.
        """
        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'skills': [],
            'sources': {},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        # Act
        await load_skills_from_agent_server(
            agent_server_url='http://localhost:8000',
            session_api_key='test-key',
            project_dir='/workspace/project',
            registered_marketplaces=None,
        )

        # Assert
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        payload = call_args[1]['json']

        # The key is omitted entirely (not sent as null) when not provided.
        assert 'registered_marketplaces' not in payload

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_handles_empty_registered_marketplaces(self, mock_client_class):
        """Test that empty registered_marketplaces list is handled correctly."""
        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'skills': [],
            'sources': {},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        # Act
        await load_skills_from_agent_server(
            agent_server_url='http://localhost:8000',
            session_api_key='test-key',
            project_dir='/workspace/project',
            registered_marketplaces=[],
        )

        # Assert
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        payload = call_args[1]['json']

        # Empty list is now preserved (semantic distinction from None)
        # Empty list = "explicitly no marketplaces", None = "not specified"
        assert payload.get('registered_marketplaces') == []
