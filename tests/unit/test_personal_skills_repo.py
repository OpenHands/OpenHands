"""Tests for the personal skills repo service."""

import subprocess
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from openhands.app_server.settings.settings_models import Settings
from openhands.server.personal_skills_repo import (
    _inject_token,
    _validate_git_url,
    clone_repo_at_commit,
    get_skills_dir_from_repo,
    load_personal_repo_skills,
    resolve_repo_commit,
)


class TestValidateGitUrl:
    def test_https_valid(self):
        _validate_git_url('https://github.com/user/repo.git')

    def test_http_valid(self):
        _validate_git_url('http://github.com/user/repo.git')

    def test_file_protocol_rejected(self):
        with pytest.raises(ValueError, match='Unsupported URL scheme'):
            _validate_git_url('file:///etc/passwd')

    def test_ssh_protocol_rejected(self):
        with pytest.raises(ValueError, match='Unsupported URL scheme'):
            _validate_git_url('ssh://git@github.com/user/repo.git')

    def test_no_scheme_rejected(self):
        with pytest.raises(ValueError, match='Unsupported URL scheme'):
            _validate_git_url('github.com/user/repo.git')

    def test_empty_host_rejected(self):
        with pytest.raises(ValueError, match='missing hostname'):
            _validate_git_url('https://')

    def test_git_protocol_rejected(self):
        with pytest.raises(ValueError, match='Unsupported URL scheme'):
            _validate_git_url('git://github.com/user/repo.git')


class TestInjectToken:
    def test_injects_into_https(self):
        result = _inject_token('https://github.com/user/repo.git', 'mytoken')
        assert result == 'https://x-access-token:mytoken@github.com/user/repo.git'

    def test_no_token_returns_unchanged(self):
        url = 'https://github.com/user/repo.git'
        assert _inject_token(url, None) == url

    def test_non_https_returns_unchanged(self):
        url = 'http://github.com/user/repo.git'
        assert _inject_token(url, 'mytoken') == url

    def test_preserves_path(self):
        result = _inject_token('https://gitlab.com/org/sub/repo.git', 'tok')
        assert result == 'https://x-access-token:tok@gitlab.com/org/sub/repo.git'

    def test_does_not_leak_token_to_wrong_host(self):
        """Token should only be injected into the netloc, not the path."""
        result = _inject_token('https://github.com/user/repo.git', 'secret')
        assert result.count('secret') == 1
        assert 'x-access-token:secret@github.com' in result


class TestResolveRepoCommit:
    def test_resolve_success(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = 'abc123def456\tHEAD\n'

        with patch('subprocess.run', return_value=mock_result):
            commit = resolve_repo_commit('https://github.com/user/repo')
            assert commit == 'abc123def456'

    def test_resolve_failure(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = 'fatal: not found'

        with patch('subprocess.run', return_value=mock_result):
            with pytest.raises(ValueError, match='Failed to resolve repo'):
                resolve_repo_commit('https://github.com/user/nonexistent')

    def test_resolve_empty_output(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ''

        with patch('subprocess.run', return_value=mock_result):
            with pytest.raises(ValueError, match='No HEAD ref found'):
                resolve_repo_commit('https://github.com/user/empty')

    def test_resolve_rejects_file_url(self):
        with pytest.raises(ValueError, match='Unsupported URL scheme'):
            resolve_repo_commit('file:///etc/passwd')

    def test_resolve_timeout(self):
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('git', 30)):
            with pytest.raises(ValueError, match='Timeout'):
                resolve_repo_commit('https://github.com/user/repo')

    def test_resolve_with_token(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = 'abc123\tHEAD\n'

        with patch('subprocess.run', return_value=mock_result) as mock_run:
            resolve_repo_commit('https://github.com/user/repo', token='mytoken')
            cmd = mock_run.call_args[0][0]
            assert 'x-access-token:mytoken@github.com' in cmd[2]


class TestCloneRepoAtCommit:
    def test_clone_success(self, tmp_path):
        clone_ok = MagicMock(returncode=0, stdout='', stderr='')
        checkout_ok = MagicMock(returncode=0, stdout='', stderr='')

        with (
            patch('subprocess.run', side_effect=[clone_ok, checkout_ok]),
            patch(
                'openhands.server.personal_skills_repo.PERSONAL_SKILLS_CACHE_DIR',
                tmp_path / 'cache',
            ),
        ):
            result = clone_repo_at_commit('https://github.com/user/repo', 'abc123')
            assert result == tmp_path / 'cache'

    def test_clone_failure_raises(self, tmp_path):
        clone_fail = MagicMock(returncode=1, stderr='clone error')

        with (
            patch('subprocess.run', return_value=clone_fail),
            patch(
                'openhands.server.personal_skills_repo.PERSONAL_SKILLS_CACHE_DIR',
                tmp_path / 'cache',
            ),
        ):
            with pytest.raises(ValueError, match='Clone failed'):
                clone_repo_at_commit('https://github.com/user/repo', 'abc123')

    def test_checkout_failure_raises(self, tmp_path):
        clone_ok = MagicMock(returncode=0, stdout='', stderr='')
        checkout_fail = MagicMock(returncode=1, stderr='checkout error')

        with (
            patch('subprocess.run', side_effect=[clone_ok, checkout_fail]),
            patch(
                'openhands.server.personal_skills_repo.PERSONAL_SKILLS_CACHE_DIR',
                tmp_path / 'cache',
            ),
        ):
            with pytest.raises(ValueError, match='Checkout failed'):
                clone_repo_at_commit('https://github.com/user/repo', 'abc123')

    def test_clone_timeout_cleans_up(self, tmp_path):
        cache = tmp_path / 'cache'

        with (
            patch('subprocess.run', side_effect=subprocess.TimeoutExpired('git', 120)),
            patch(
                'openhands.server.personal_skills_repo.PERSONAL_SKILLS_CACHE_DIR',
                cache,
            ),
        ):
            with pytest.raises(ValueError, match='Timeout'):
                clone_repo_at_commit('https://github.com/user/repo', 'abc123')
            assert not cache.exists()

    def test_skips_clone_if_already_at_commit(self, tmp_path):
        cache = tmp_path / 'cache'
        cache.mkdir()

        rev_parse_ok = MagicMock(returncode=0, stdout='abc123\n')

        with (
            patch('subprocess.run', return_value=rev_parse_ok) as mock_run,
            patch(
                'openhands.server.personal_skills_repo.PERSONAL_SKILLS_CACHE_DIR',
                cache,
            ),
        ):
            result = clone_repo_at_commit('https://github.com/user/repo', 'abc123')
            assert result == cache
            # Only rev-parse should be called, not clone
            assert mock_run.call_count == 1

    def test_rejects_file_url(self):
        with pytest.raises(ValueError, match='Unsupported URL scheme'):
            clone_repo_at_commit('file:///etc/passwd', 'abc123')


class TestLoadPersonalRepoSkills:
    def test_returns_empty_when_no_cache(self, tmp_path):
        with patch(
            'openhands.server.personal_skills_repo.PERSONAL_SKILLS_CACHE_DIR',
            tmp_path / 'nonexistent',
        ):
            assert load_personal_repo_skills() == []

    def test_loads_skill_from_frontmatter(self, tmp_path):
        skills_dir = tmp_path / 'skills'
        skills_dir.mkdir()
        (skills_dir / 'test-skill.md').write_text(
            '---\nname: my-skill\ntriggers:\n- docker\n---\nSkill content here'
        )

        with (
            patch(
                'openhands.server.personal_skills_repo.PERSONAL_SKILLS_CACHE_DIR',
                tmp_path,
            ),
            patch(
                'openhands.server.personal_skills_repo.get_skills_dir_from_repo',
                return_value=skills_dir,
            ),
        ):
            results = load_personal_repo_skills()
            assert len(results) == 1
            name, content, triggers = results[0]
            assert name == 'my-skill'
            assert 'Skill content here' in content
            assert triggers == ['docker']

    def test_skips_readme(self, tmp_path):
        skills_dir = tmp_path / 'skills'
        skills_dir.mkdir()
        (skills_dir / 'README.md').write_text('# Readme')
        (skills_dir / 'real-skill.md').write_text('---\nname: real\n---\nContent')

        with (
            patch(
                'openhands.server.personal_skills_repo.PERSONAL_SKILLS_CACHE_DIR',
                tmp_path,
            ),
            patch(
                'openhands.server.personal_skills_repo.get_skills_dir_from_repo',
                return_value=skills_dir,
            ),
        ):
            results = load_personal_repo_skills()
            assert len(results) == 1
            assert results[0][0] == 'real'

    def test_falls_back_to_stem_name(self, tmp_path):
        skills_dir = tmp_path / 'skills'
        skills_dir.mkdir()
        (skills_dir / 'my-tool.md').write_text('---\n---\nNo name in frontmatter')

        with (
            patch(
                'openhands.server.personal_skills_repo.PERSONAL_SKILLS_CACHE_DIR',
                tmp_path,
            ),
            patch(
                'openhands.server.personal_skills_repo.get_skills_dir_from_repo',
                return_value=skills_dir,
            ),
        ):
            results = load_personal_repo_skills()
            assert len(results) == 1
            assert results[0][0] == 'my-tool'


class TestGetSkillsDirFromRepo:
    def test_openhands_microagents_dir(self, tmp_path):
        (tmp_path / '.openhands' / 'microagents').mkdir(parents=True)
        assert (
            get_skills_dir_from_repo(tmp_path)
            == tmp_path / '.openhands' / 'microagents'
        )

    def test_skills_dir(self, tmp_path):
        (tmp_path / 'skills').mkdir()
        assert get_skills_dir_from_repo(tmp_path) == tmp_path / 'skills'

    def test_agents_skills_dir(self, tmp_path):
        (tmp_path / '.agents' / 'skills').mkdir(parents=True)
        assert get_skills_dir_from_repo(tmp_path) == tmp_path / '.agents' / 'skills'

    def test_fallback_to_root_with_md_files(self, tmp_path):
        (tmp_path / 'my-skill.md').write_text('# Skill')
        assert get_skills_dir_from_repo(tmp_path) == tmp_path

    def test_no_skills_dir(self, tmp_path):
        assert get_skills_dir_from_repo(tmp_path) is None


class TestSettingsPersonalSkillsRepoFields:
    def test_default_values(self):
        s = Settings()
        assert s.personal_skills_repo_url is None
        assert s.personal_skills_repo_commit is None
        assert s.personal_skills_repo_updated_at is None

    def test_set_values(self):
        now = datetime.now(timezone.utc)
        s = Settings(
            personal_skills_repo_url='https://github.com/user/skills',
            personal_skills_repo_commit='abc123',
            personal_skills_repo_updated_at=now,
        )
        assert s.personal_skills_repo_url == 'https://github.com/user/skills'
        assert s.personal_skills_repo_commit == 'abc123'

    def test_serialization_roundtrip(self):
        now = datetime.now(timezone.utc)
        s = Settings(
            personal_skills_repo_url='https://github.com/user/skills',
            personal_skills_repo_commit='abc123',
            personal_skills_repo_updated_at=now,
        )
        restored = Settings.model_validate(s.model_dump(mode='json'))
        assert restored.personal_skills_repo_url == s.personal_skills_repo_url
        assert restored.personal_skills_repo_commit == s.personal_skills_repo_commit
