import os
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import SecretStr

from openhands.app_server.app import app
from openhands.app_server.file_store.memory import InMemoryFileStore
from openhands.app_server.integrations.provider import ProviderToken, ProviderType
from openhands.app_server.integrations.service_types import UserGitInfo
from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.secrets.secrets_store import SecretsStore
from openhands.app_server.settings.file_settings_store import FileSettingsStore
from openhands.app_server.settings.settings_models import MarketplaceRegistration
from openhands.app_server.settings.settings_store import SettingsStore
from openhands.app_server.user.skills_router import (
    GLOBAL_SKILLS_DIR,
    _clone_marketplace_repo,
)
from openhands.app_server.user_auth.user_auth import UserAuth


class MockUserAuth(UserAuth):
    """Mock implementation of UserAuth for testing."""

    def __init__(self):
        self._settings = None
        self._settings_store = MagicMock()
        self._settings_store.load = AsyncMock(return_value=None)
        self._settings_store.store = AsyncMock()

    async def get_user_id(self) -> str | None:
        return 'test-user'

    async def get_user_email(self) -> str | None:
        return 'test-email@whatever.com'

    async def get_access_token(self) -> SecretStr | None:
        return SecretStr('test-token')

    async def get_provider_tokens(
        self,
    ) -> dict[ProviderType, ProviderToken] | None:
        return None

    async def get_user_settings_store(self) -> SettingsStore | None:
        return self._settings_store

    async def get_secrets_store(self) -> SecretsStore | None:
        return None

    async def get_secrets(self) -> Secrets | None:
        return None

    async def get_mcp_api_key(self) -> str | None:
        return None

    async def get_user_git_info(self) -> UserGitInfo | None:
        return None

    @classmethod
    async def get_instance(cls, request: Request) -> UserAuth:
        return MockUserAuth()

    @classmethod
    async def get_for_user(cls, user_id: str) -> UserAuth:
        return MockUserAuth()


@pytest.fixture
def test_client():
    with (
        patch.dict(os.environ, {'SESSION_API_KEY': ''}, clear=False),
        patch('openhands.app_server.utils.dependencies._SESSION_API_KEY', None),
        patch(
            'openhands.app_server.user_auth.user_auth.UserAuth.get_instance',
            return_value=MockUserAuth(),
        ),
        patch(
            'openhands.app_server.settings.file_settings_store.FileSettingsStore.get_instance',
            AsyncMock(return_value=FileSettingsStore(InMemoryFileStore())),
        ),
    ):
        client = TestClient(app)
        yield client


def _write_skill_file(
    dir_path: Path,
    name: str,
    skill_type: str = 'knowledge',
    triggers: list[str] | None = None,
) -> None:
    """Write a mock skill markdown file with frontmatter."""
    dir_path.mkdir(parents=True, exist_ok=True)
    lines = [
        '---',
        f'name: {name}',
        f'type: {skill_type}',
    ]
    if triggers:
        lines.append('triggers:')
        for t in triggers:
            lines.append(f'- {t}')
    lines.append('---')
    lines.append(f'{name} content')
    (dir_path / f'{name}.md').write_text('\n'.join(lines))


@pytest.mark.asyncio
async def test_skills_search_returns_skills(test_client, tmp_path):
    """Test that GET /api/v1/skills/search returns a paginated list of skills."""
    global_dir = tmp_path / 'global'
    _write_skill_file(global_dir, 'test_repo', skill_type='repo')
    _write_skill_file(
        global_dir, 'test_knowledge', skill_type='knowledge', triggers=['testword']
    )

    with (
        patch('openhands.app_server.user.skills_router.GLOBAL_SKILLS_DIR', global_dir),
        patch(
            'openhands.app_server.user.skills_router.USER_SKILLS_DIR',
            tmp_path / 'nonexistent',
        ),
    ):
        response = test_client.get('/api/v1/skills/search')

    assert response.status_code == 200
    data = response.json()
    assert 'items' in data
    assert 'next_page_id' in data
    assert len(data['items']) == 2

    # Verify skill structure
    skill_names = [s['name'] for s in data['items']]
    assert 'test_repo' in skill_names
    assert 'test_knowledge' in skill_names

    # Check knowledge skill has triggers
    knowledge_skill = next(s for s in data['items'] if s['name'] == 'test_knowledge')
    assert knowledge_skill['triggers'] == ['testword']
    assert knowledge_skill['type'] == 'knowledge'

    # Check repo skill has no triggers
    repo_skill = next(s for s in data['items'] if s['name'] == 'test_repo')
    assert repo_skill['triggers'] is None
    assert repo_skill['type'] == 'repo'

    # No next page when all results fit
    assert data['next_page_id'] is None


@pytest.mark.asyncio
async def test_skills_search_handles_missing_dirs(test_client, tmp_path):
    """Test that the endpoint handles missing directories gracefully."""
    with (
        patch(
            'openhands.app_server.user.skills_router.GLOBAL_SKILLS_DIR',
            tmp_path / 'no_such_dir',
        ),
        patch(
            'openhands.app_server.user.skills_router.USER_SKILLS_DIR',
            tmp_path / 'also_missing',
        ),
    ):
        response = test_client.get('/api/v1/skills/search')

    assert response.status_code == 200
    data = response.json()
    assert data['items'] == []
    assert data['next_page_id'] is None


@pytest.mark.asyncio
async def test_skills_search_sorted_by_source_then_name(test_client, tmp_path):
    """Test that skills are sorted by source (global first) then by name."""
    global_dir = tmp_path / 'global'
    user_dir = tmp_path / 'user'

    _write_skill_file(global_dir, 'z_global', skill_type='repo')
    _write_skill_file(global_dir, 'a_global', skill_type='repo')
    _write_skill_file(user_dir, 'b_user', skill_type='repo')

    with (
        patch('openhands.app_server.user.skills_router.GLOBAL_SKILLS_DIR', global_dir),
        patch('openhands.app_server.user.skills_router.USER_SKILLS_DIR', user_dir),
    ):
        response = test_client.get('/api/v1/skills/search')

    assert response.status_code == 200
    data = response.json()
    skills = data['items']

    # Global skills should come first, sorted by name
    assert skills[0]['name'] == 'a_global'
    assert skills[0]['source'] == 'global'
    assert skills[1]['name'] == 'z_global'
    assert skills[1]['source'] == 'global'
    # User skills should come last
    assert skills[2]['name'] == 'b_user'
    assert skills[2]['source'] == 'user'


@pytest.mark.asyncio
async def test_skills_search_pagination(test_client, tmp_path):
    """Test cursor-based pagination."""
    global_dir = tmp_path / 'global'
    _write_skill_file(global_dir, 'skill_a', skill_type='repo')
    _write_skill_file(global_dir, 'skill_b', skill_type='repo')
    _write_skill_file(global_dir, 'skill_c', skill_type='repo')

    with (
        patch('openhands.app_server.user.skills_router.GLOBAL_SKILLS_DIR', global_dir),
        patch(
            'openhands.app_server.user.skills_router.USER_SKILLS_DIR',
            tmp_path / 'nonexistent',
        ),
    ):
        # First page with limit=2
        response = test_client.get('/api/v1/skills/search', params={'limit': 2})
        assert response.status_code == 200
        data = response.json()
        assert len(data['items']) == 2
        assert data['items'][0]['name'] == 'skill_a'
        assert data['items'][1]['name'] == 'skill_b'
        assert data['next_page_id'] == 'skill_b'

        # Second page using next_page_id
        response = test_client.get(
            '/api/v1/skills/search',
            params={'limit': 2, 'page_id': data['next_page_id']},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data['items']) == 1
        assert data['items'][0]['name'] == 'skill_c'
        assert data['next_page_id'] is None


def test_global_skills_dir_points_to_repo_root():
    """Test that GLOBAL_SKILLS_DIR points to the correct location.

    This test validates that GLOBAL_SKILLS_DIR is correctly configured to point
    to the skills/ directory at the repo root. This prevents regressions like the
    one introduced in fb98faf4a where an incorrect path caused no skills to load.
    """
    # The directory should exist
    assert GLOBAL_SKILLS_DIR.exists(), (
        f'GLOBAL_SKILLS_DIR does not exist: {GLOBAL_SKILLS_DIR}'
    )

    # It should be named 'skills'
    assert GLOBAL_SKILLS_DIR.name == 'skills', (
        f"Expected directory name 'skills', got '{GLOBAL_SKILLS_DIR.name}'"
    )

    # It should contain at least one .md file (skill definition)
    md_files = list(GLOBAL_SKILLS_DIR.glob('*.md'))
    assert len(md_files) > 0, (
        f'GLOBAL_SKILLS_DIR contains no .md files: {GLOBAL_SKILLS_DIR}'
    )

    # Verify it's at repo root by checking for known skill files
    # (github.md is a core skill that should always exist)
    expected_skill = GLOBAL_SKILLS_DIR / 'github.md'
    assert expected_skill.exists(), (
        f'Expected skill file not found: {expected_skill}. '
        f'GLOBAL_SKILLS_DIR may be pointing to wrong location.'
    )


# Tests for marketplace-skills endpoint git clone failures


class TestMarketplaceSkillsCloneFailures:
    """Tests for git clone failure scenarios in marketplace-skills endpoint."""

    @pytest.mark.asyncio
    async def test_clone_invalid_repo_path(self):
        """Test that invalid repository path returns appropriate error."""
        mock_user_context = MagicMock()
        mock_user_context.get_provider_tokens = AsyncMock(return_value=None)
        mock_user_context.get_user_id = AsyncMock(return_value='test-user')

        marketplace = MarketplaceRegistration(
            name='test-marketplace',
            source='github:owner/valid-repo',
        )

        # Mock _parse_marketplace_source to return invalid path (empty = parse failure)
        with patch(
            'openhands.app_server.user.skills_router._parse_marketplace_source',
            return_value=('github', ''),
        ):
            result = await _clone_marketplace_repo(marketplace, mock_user_context)

        assert result[0] is None
        assert 'Invalid repository path' in result[1]

    @pytest.mark.asyncio
    async def test_clone_network_timeout(self):
        """Test that network timeout returns appropriate error."""

        mock_user_context = MagicMock()
        mock_user_context.get_provider_tokens = AsyncMock(return_value=None)
        mock_user_context.get_user_id = AsyncMock(return_value='test-user')

        marketplace = MarketplaceRegistration(
            name='test-marketplace',
            source='github:owner/nonexistent-repo-12345',
        )

        # Mock subprocess.run to raise TimeoutExpired
        with patch(
            'openhands.app_server.user.skills_router.subprocess.run'
        ) as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd='git clone', timeout=120
            )
            result = await _clone_marketplace_repo(marketplace, mock_user_context)

        assert result[0] is None
        assert 'timed out' in result[1]

    @pytest.mark.asyncio
    async def test_clone_git_failure_returns_error(self):
        """Test that git clone failure returns error message."""
        mock_user_context = MagicMock()
        mock_user_context.get_provider_tokens = AsyncMock(return_value=None)
        mock_user_context.get_user_id = AsyncMock(return_value='test-user')

        marketplace = MarketplaceRegistration(
            name='test-marketplace',
            source='github:owner/nonexistent-repo-xyz',
        )

        # Mock subprocess.run to return error
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stderr = 'Repository not found'

        with patch(
            'openhands.app_server.user.skills_router.subprocess.run'
        ) as mock_run:
            mock_run.return_value = mock_result
            result = await _clone_marketplace_repo(marketplace, mock_user_context)

        assert result[0] is None
        assert 'Git clone failed' in result[1]
        assert 'not found' in result[1].lower()

    @pytest.mark.asyncio
    async def test_clone_with_ref_checkout_failure(self):
        """Test that checkout ref failure returns appropriate error."""
        mock_user_context = MagicMock()
        mock_user_context.get_provider_tokens = AsyncMock(return_value=None)
        mock_user_context.get_user_id = AsyncMock(return_value='test-user')

        marketplace = MarketplaceRegistration(
            name='test-marketplace',
            source='github:owner/valid-repo',
            ref='nonexistent-branch',
        )

        # Mock subprocess.run for clone success, checkout failure
        mock_clone_result = MagicMock()
        mock_clone_result.returncode = 0

        mock_checkout_result = MagicMock()
        mock_checkout_result.returncode = 128
        mock_checkout_result.stderr = " pathspec 'nonexistent-branch' did not match"

        def run_side_effect(cmd, *args, **kwargs):
            if 'checkout' in cmd:
                return mock_checkout_result
            return mock_clone_result

        with patch(
            'openhands.app_server.user.skills_router.subprocess.run'
        ) as mock_run:
            with patch('openhands.app_server.user.skills_router._cleanup_clone_dir'):
                with patch('tempfile.mkdtemp', return_value=Path('/tmp/test_clone')):
                    mock_run.side_effect = run_side_effect
                    result = await _clone_marketplace_repo(
                        marketplace, mock_user_context
                    )

        assert result[0] is None
        assert 'Git checkout failed' in result[1]

    @pytest.mark.asyncio
    async def test_clone_repo_path_not_found(self):
        """Test that non-existent repo_path returns appropriate error."""
        mock_user_context = MagicMock()
        mock_user_context.get_provider_tokens = AsyncMock(return_value=None)
        mock_user_context.get_user_id = AsyncMock(return_value='test-user')

        marketplace = MarketplaceRegistration(
            name='test-marketplace',
            source='github:owner/valid-repo',
            repo_path='nonexistent/skills/dir',
        )

        mock_clone_result = MagicMock()
        mock_clone_result.returncode = 0

        def run_side_effect(cmd, *args, **kwargs):
            return mock_clone_result

        with patch(
            'openhands.app_server.user.skills_router.subprocess.run'
        ) as mock_run:
            with patch(
                'openhands.app_server.user.skills_router.Path.exists'
            ) as mock_exists:
                with patch(
                    'openhands.app_server.user.skills_router._cleanup_clone_dir'
                ):
                    with patch(
                        'tempfile.mkdtemp', return_value=Path('/tmp/test_clone')
                    ):
                        mock_run.side_effect = run_side_effect
                        # First call is clone, exists() returns False for repo_path
                        mock_exists.return_value = False
                        result = await _clone_marketplace_repo(
                            marketplace, mock_user_context
                        )

        assert result[0] is None
        assert 'Repo path not found' in result[1]

    @pytest.mark.asyncio
    async def test_clone_bbdc_personal_url_uses_pinned_authenticated_url(self):
        """A BBDC personal-repo URL clones via the host-matched, pinned token URL."""
        handler = AsyncMock()
        dc_url = 'https://user:pat@bitbucket.dc.example.com/scm/~jane/marketplace.git'
        handler.get_authenticated_git_url.return_value = dc_url

        mock_user_context = MagicMock()
        mock_user_context.get_provider_tokens = AsyncMock(
            return_value={
                ProviderType.BITBUCKET_DATA_CENTER: ProviderToken(
                    host='bitbucket.dc.example.com'
                )
            }
        )
        mock_user_context.get_user_id = AsyncMock(return_value='test-user')
        mock_user_context.get_provider_handler = AsyncMock(return_value=handler)

        marketplace = MarketplaceRegistration(
            name='personal',
            source='https://bitbucket.dc.example.com/scm/~jane/marketplace.git',
        )

        mock_clone = MagicMock(returncode=0)
        with patch(
            'openhands.app_server.user.skills_router.subprocess.run',
            return_value=mock_clone,
        ) as mock_run:
            with patch('tempfile.mkdtemp', return_value=Path('/tmp/test_clone')):
                await _clone_marketplace_repo(marketplace, mock_user_context)

        handler.get_authenticated_git_url.assert_awaited_once_with(
            '~jane/marketplace',
            specified_provider=ProviderType.BITBUCKET_DATA_CENTER,
        )
        clone_argv = mock_run.call_args[0][0]
        assert clone_argv[:3] == ['git', 'clone', '--']
        assert clone_argv[3] == dc_url

    @pytest.mark.asyncio
    async def test_clone_refuses_unmatched_custom_host_url(self):
        """A full-URL source on a host matching no provider is refused (no SSRF)."""
        mock_user_context = MagicMock()
        mock_user_context.get_provider_tokens = AsyncMock(
            return_value={
                ProviderType.BITBUCKET_DATA_CENTER: ProviderToken(
                    host='bitbucket.dc.example.com'
                )
            }
        )
        mock_user_context.get_user_id = AsyncMock(return_value='test-user')
        mock_user_context.get_provider_handler = AsyncMock()

        marketplace = MarketplaceRegistration(
            name='evil',
            source='https://attacker.example.com/scm/proj/repo.git',
        )

        with patch(
            'openhands.app_server.user.skills_router.subprocess.run'
        ) as mock_run:
            result = await _clone_marketplace_repo(marketplace, mock_user_context)

        assert result[0] is None
        assert 'Unsupported marketplace host' in result[1]
        mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_clone_public_github_url_works_with_non_github_provider(self):
        """A public github.com URL is a public clone even for a Bitbucket-only user."""
        mock_user_context = MagicMock()
        mock_user_context.get_provider_tokens = AsyncMock(
            return_value={
                ProviderType.BITBUCKET_DATA_CENTER: ProviderToken(
                    host='bitbucket.dc.example.com'
                )
            }
        )
        mock_user_context.get_user_id = AsyncMock(return_value='test-user')
        mock_user_context.get_provider_handler = AsyncMock()

        marketplace = MarketplaceRegistration(
            name='public-gh',
            source='https://github.com/OpenHands/skills.git',
        )

        mock_clone = MagicMock(returncode=0)
        with patch(
            'openhands.app_server.user.skills_router.subprocess.run',
            return_value=mock_clone,
        ) as mock_run:
            with patch('tempfile.mkdtemp', return_value=Path('/tmp/test_clone')):
                await _clone_marketplace_repo(marketplace, mock_user_context)

        # Not refused; cloned from the public github.com domain (no credentials).
        clone_argv = mock_run.call_args[0][0]
        assert clone_argv[3] == 'https://github.com/OpenHands/skills.git'
        mock_user_context.get_provider_handler.assert_not_awaited()
