"""Tests for skill_loader module.

This module tests the loading of skills from various sources
(global, user, and repository-level) into SDK Skill objects for V1 conversations.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from openhands.app_server.app_conversation.skill_loader import (
    _determine_repo_root,
    _find_and_load_global_skill_files,
    _find_and_load_skill_md_files,
    _load_special_files,
    _read_file_from_workspace,
    load_global_skills,
    load_repo_skills,
    merge_skills,
)

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
def mock_async_remote_workspace():
    """Create a mock AsyncRemoteWorkspace."""
    workspace = AsyncMock()
    return workspace


@pytest.fixture
def temp_skills_dir():
    """Create a temporary directory with test skill files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        # Create test skill files
        test_skill = """---
name: test_skill
triggers:
  - test
  - testing
---

# Test Skill

This is a test skill for testing purposes.
"""
        (root / 'test_skill.md').write_text(test_skill)

        another_skill = """---
name: another_skill
---

# Another Skill

Another test skill.
"""
        (root / 'another_skill.md').write_text(another_skill)

        # Create README.md which should be ignored
        (root / 'README.md').write_text('# README\n\nThis should be ignored.')

        yield root


@pytest.fixture
def command_result_success():
    """Create a successful command result."""
    result = Mock()
    result.exit_code = 0
    result.stdout = 'test output'
    return result


@pytest.fixture
def command_result_failure():
    """Create a failed command result."""
    result = Mock()
    result.exit_code = 1
    result.stdout = ''
    return result


# ===== Tests for Helper Functions =====


class TestDetermineRepoRoot:
    """Test _determine_repo_root helper function."""

    def test_with_selected_repository(self):
        """Test determining repo root with selected repository."""
        result = _determine_repo_root('/workspace/project', 'owner/repo-name')
        assert result == '/workspace/project/repo-name'

    def test_without_selected_repository(self):
        """Test determining repo root without selected repository."""
        result = _determine_repo_root('/workspace/project', None)
        assert result == '/workspace/project'

    def test_with_complex_repository_name(self):
        """Test with complex repository name."""
        result = _determine_repo_root('/workspace', 'org-name/complex-repo-123')
        assert result == '/workspace/complex-repo-123'


class TestReadFileFromWorkspace:
    """Test _read_file_from_workspace helper function."""

    @pytest.mark.asyncio
    async def test_successful_read(
        self, mock_async_remote_workspace, command_result_success
    ):
        """Test successfully reading a file from workspace."""
        command_result_success.stdout = 'file content\n'
        mock_async_remote_workspace.execute_command.return_value = (
            command_result_success
        )

        result = await _read_file_from_workspace(
            mock_async_remote_workspace, '/path/to/file.md', '/workspace'
        )

        assert result == 'file content\n'
        mock_async_remote_workspace.execute_command.assert_called_once_with(
            'cat /path/to/file.md', cwd='/workspace', timeout=10.0
        )

    @pytest.mark.asyncio
    async def test_file_not_found(
        self, mock_async_remote_workspace, command_result_failure
    ):
        """Test reading a non-existent file."""
        mock_async_remote_workspace.execute_command.return_value = (
            command_result_failure
        )

        result = await _read_file_from_workspace(
            mock_async_remote_workspace, '/nonexistent/file.md', '/workspace'
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_file(self, mock_async_remote_workspace):
        """Test reading an empty file."""
        result_obj = Mock()
        result_obj.exit_code = 0
        result_obj.stdout = '   '  # Only whitespace
        mock_async_remote_workspace.execute_command.return_value = result_obj

        result = await _read_file_from_workspace(
            mock_async_remote_workspace, '/empty/file.md', '/workspace'
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_command_exception(self, mock_async_remote_workspace):
        """Test handling exception during file read."""
        mock_async_remote_workspace.execute_command.side_effect = Exception(
            'Connection error'
        )

        result = await _read_file_from_workspace(
            mock_async_remote_workspace, '/path/to/file.md', '/workspace'
        )

        assert result is None


class TestLoadSpecialFiles:
    """Test _load_special_files helper function."""

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.skill_loader._read_file_from_workspace'
    )
    @patch('openhands.app_server.app_conversation.skill_loader.Skill')
    async def test_load_all_special_files(
        self,
        mock_skill_class,
        mock_read_file,
        mock_async_remote_workspace,
        mock_skills_list,
    ):
        """Test loading all special files successfully."""
        # Mock reading files - return content for each special file
        mock_read_file.side_effect = [
            'cursorrules content',
            'agents.md content',
            'agent.md content',
        ]

        # Mock skill creation
        mock_skill_class.load.side_effect = mock_skills_list

        result = await _load_special_files(
            mock_async_remote_workspace, '/repo', '/workspace'
        )

        assert len(result) == 3
        assert result == mock_skills_list
        assert mock_read_file.call_count == 3
        assert mock_skill_class.load.call_count == 3

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.skill_loader._read_file_from_workspace'
    )
    @patch('openhands.app_server.app_conversation.skill_loader.Skill')
    async def test_load_partial_special_files(
        self, mock_skill_class, mock_read_file, mock_async_remote_workspace, mock_skill
    ):
        """Test loading when only some special files exist."""
        # Only .cursorrules exists
        mock_read_file.side_effect = ['cursorrules content', None, None]
        mock_skill_class.load.return_value = mock_skill

        result = await _load_special_files(
            mock_async_remote_workspace, '/repo', '/workspace'
        )

        assert len(result) == 1
        assert result[0] == mock_skill
        assert mock_read_file.call_count == 3
        assert mock_skill_class.load.call_count == 1

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.skill_loader._read_file_from_workspace'
    )
    async def test_load_no_special_files(
        self, mock_read_file, mock_async_remote_workspace
    ):
        """Test when no special files exist."""
        mock_read_file.return_value = None

        result = await _load_special_files(
            mock_async_remote_workspace, '/repo', '/workspace'
        )

        assert len(result) == 0


class TestFindAndLoadSkillMdFiles:
    """Test _find_and_load_skill_md_files helper function."""

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.skill_loader._read_file_from_workspace'
    )
    @patch('openhands.app_server.app_conversation.skill_loader.Skill')
    async def test_find_and_load_files_success(
        self,
        mock_skill_class,
        mock_read_file,
        mock_async_remote_workspace,
        mock_skills_list,
    ):
        """Test successfully finding and loading skill .md files."""
        result_obj = Mock()
        result_obj.exit_code = 0
        result_obj.stdout = (
            '/repo/.openhands/skills/test1.md\n/repo/.openhands/skills/test2.md\n'
        )
        mock_async_remote_workspace.execute_command.return_value = result_obj

        mock_read_file.side_effect = ['content1', 'content2']
        mock_skill_class.load.side_effect = mock_skills_list[:2]

        result = await _find_and_load_skill_md_files(
            mock_async_remote_workspace, '/repo/.openhands/skills', '/workspace'
        )

        assert len(result) == 2
        assert result == mock_skills_list[:2]

        # Verify relative paths are used
        assert mock_skill_class.load.call_args_list[0][1]['path'] == 'test1.md'
        assert mock_skill_class.load.call_args_list[1][1]['path'] == 'test2.md'

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.skill_loader._read_file_from_workspace'
    )
    @patch('openhands.app_server.app_conversation.skill_loader.Skill')
    async def test_find_and_load_excludes_readme(
        self, mock_skill_class, mock_read_file, mock_async_remote_workspace, mock_skill
    ):
        """Test that README.md files are excluded."""
        result_obj = Mock()
        result_obj.exit_code = 0
        result_obj.stdout = (
            '/repo/.openhands/skills/test.md\n/repo/.openhands/skills/README.md\n'
        )
        mock_async_remote_workspace.execute_command.return_value = result_obj

        mock_read_file.return_value = 'content'
        mock_skill_class.load.return_value = mock_skill

        result = await _find_and_load_skill_md_files(
            mock_async_remote_workspace, '/repo/.openhands/skills', '/workspace'
        )

        assert len(result) == 1
        assert result[0] == mock_skill
        # Verify README.md was not processed
        assert mock_read_file.call_count == 1

    @pytest.mark.asyncio
    async def test_find_and_load_no_results(
        self, mock_async_remote_workspace, command_result_failure
    ):
        """Test when no files are found."""
        mock_async_remote_workspace.execute_command.return_value = (
            command_result_failure
        )

        result = await _find_and_load_skill_md_files(
            mock_async_remote_workspace, '/nonexistent', '/workspace'
        )

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_find_and_load_exception(self, mock_async_remote_workspace):
        """Test handling exception during file search."""
        mock_async_remote_workspace.execute_command.side_effect = Exception(
            'Command error'
        )

        result = await _find_and_load_skill_md_files(
            mock_async_remote_workspace, '/repo/.openhands/skills', '/workspace'
        )

        assert len(result) == 0

    @pytest.mark.asyncio
    @patch(
        'openhands.app_server.app_conversation.skill_loader._read_file_from_workspace'
    )
    async def test_find_and_load_some_missing(
        self, mock_read_file, mock_async_remote_workspace
    ):
        """Test loading when some files fail to read."""
        result_obj = Mock()
        result_obj.exit_code = 0
        result_obj.stdout = (
            '/repo/.openhands/skills/test1.md\n/repo/.openhands/skills/missing.md\n'
        )
        mock_async_remote_workspace.execute_command.return_value = result_obj

        mock_read_file.side_effect = ['content1', None]

        with patch(
            'openhands.app_server.app_conversation.skill_loader.Skill'
        ) as mock_skill_class:
            mock_skill = Mock()
            mock_skill_class.load.return_value = mock_skill

            result = await _find_and_load_skill_md_files(
                mock_async_remote_workspace,
                '/repo/.openhands/skills',
                '/workspace',
            )

            assert len(result) == 1
            assert mock_skill_class.load.call_count == 1


class TestFindAndLoadGlobalSkillFiles:
    """Test _find_and_load_global_skill_files helper function."""

    @patch('openhands.app_server.app_conversation.skill_loader.Skill')
    def test_find_and_load_global_files_success(
        self, mock_skill_class, temp_skills_dir, mock_skills_list
    ):
        """Test successfully finding and loading global skill files."""
        file_paths = list(temp_skills_dir.glob('*.md'))
        file_paths = [f for f in file_paths if f.name.lower() != 'readme.md']

        mock_skill_class.load.side_effect = mock_skills_list[: len(file_paths)]

        result = _find_and_load_global_skill_files(temp_skills_dir)

        # Should find and load .md files but not README.md
        assert len(result) == len(file_paths)
        assert mock_skill_class.load.call_count == len(file_paths)
        skill_names = [s.name for s in result]
        assert len(skill_names) == len(file_paths)

    @patch('openhands.app_server.app_conversation.skill_loader.Skill')
    def test_find_and_load_global_files_with_errors(
        self, mock_skill_class, temp_skills_dir, mock_skill
    ):
        """Test loading when some files fail to parse."""
        file_paths = list(temp_skills_dir.glob('*.md'))
        file_paths = [f for f in file_paths if f.name.lower() != 'readme.md']

        # First file succeeds, second file fails
        mock_skill_class.load.side_effect = [mock_skill, Exception('Parse error')]

        result = _find_and_load_global_skill_files(temp_skills_dir)

        assert len(result) == 1
        assert result[0] == mock_skill

    def test_find_and_load_global_files_empty_dir(self, tmp_path):
        """Test finding and loading files in empty directory."""
        result = _find_and_load_global_skill_files(tmp_path)
        assert len(result) == 0

    def test_find_and_load_global_files_nonexistent_dir(self):
        """Test finding and loading files in non-existent directory."""
        nonexistent = Path('/nonexistent/path')
        result = _find_and_load_global_skill_files(nonexistent)
        assert len(result) == 0


# ===== Tests for Main Loader Functions =====


class TestLoadGlobalSkills:
    """Test load_global_skills main function."""

    @patch('openhands.app_server.app_conversation.skill_loader.Path')
    @patch(
        'openhands.app_server.app_conversation.skill_loader._find_and_load_global_skill_files'
    )
    def test_load_global_skills_success(
        self,
        mock_find_and_load,
        mock_path_class,
        temp_skills_dir,
        mock_skills_list,
    ):
        """Test successfully loading global skills."""
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_class.return_value = mock_path_obj

        mock_find_and_load.return_value = mock_skills_list

        result = load_global_skills()

        assert len(result) == len(mock_skills_list)
        assert result == mock_skills_list

    @patch('openhands.app_server.app_conversation.skill_loader.Path')
    def test_load_global_skills_dir_not_exists(self, mock_path_class):
        """Test when global skills directory doesn't exist."""
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = False
        mock_path_class.return_value = mock_path_obj

        result = load_global_skills()

        assert len(result) == 0

    @patch('openhands.app_server.app_conversation.skill_loader.Path')
    @patch(
        'openhands.app_server.app_conversation.skill_loader._find_and_load_global_skill_files'
    )
    def test_load_global_skills_exception(self, mock_find_and_load, mock_path_class):
        """Test handling exception during global skill loading."""
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_class.return_value = mock_path_obj

        mock_find_and_load.side_effect = Exception('File system error')

        result = load_global_skills()

        assert len(result) == 0


class TestLoadRepoSkills:
    """Test load_repo_skills main function."""

    @pytest.mark.asyncio
    @patch('openhands.app_server.app_conversation.skill_loader._load_special_files')
    @patch(
        'openhands.app_server.app_conversation.skill_loader._find_and_load_skill_md_files'
    )
    async def test_load_repo_skills_success(
        self,
        mock_find_and_load,
        mock_load_special,
        mock_async_remote_workspace,
        mock_skills_list,
    ):
        """Test successfully loading repo skills."""
        special_skills = [mock_skills_list[0]]
        skills_dir_skills = [mock_skills_list[1]]
        microagents_dir_skills = [mock_skills_list[2]]

        mock_load_special.return_value = special_skills
        # Mock loading from both directories
        mock_find_and_load.side_effect = [skills_dir_skills, microagents_dir_skills]

        result = await load_repo_skills(
            mock_async_remote_workspace, 'owner/repo', '/workspace/project'
        )

        assert len(result) == 3
        # Verify all skills are present (merged with precedence)
        assert special_skills[0] in result
        assert skills_dir_skills[0] in result
        assert microagents_dir_skills[0] in result

    @pytest.mark.asyncio
    @patch('openhands.app_server.app_conversation.skill_loader._load_special_files')
    @patch(
        'openhands.app_server.app_conversation.skill_loader._find_and_load_skill_md_files'
    )
    async def test_load_repo_skills_no_selected_repository(
        self,
        mock_find_and_load,
        mock_load_special,
        mock_async_remote_workspace,
        mock_skills_list,
    ):
        """Test loading repo skills without selected repository."""
        mock_load_special.return_value = [mock_skills_list[0]]
        mock_find_and_load.return_value = []

        result = await load_repo_skills(
            mock_async_remote_workspace, None, '/workspace/project'
        )

        assert len(result) == 1
        # Verify repo root is working_dir when no repository selected
        mock_load_special.assert_called_once_with(
            mock_async_remote_workspace, '/workspace/project', '/workspace/project'
        )
        # Verify both directories were checked
        assert mock_find_and_load.call_count == 2

    @pytest.mark.asyncio
    @patch('openhands.app_server.app_conversation.skill_loader._load_special_files')
    async def test_load_repo_skills_exception(
        self, mock_load_special, mock_async_remote_workspace
    ):
        """Test handling exception during repo skill loading."""
        mock_load_special.side_effect = Exception('Workspace error')

        result = await load_repo_skills(
            mock_async_remote_workspace, 'owner/repo', '/workspace/project'
        )

        assert len(result) == 0


class TestMergeSkills:
    """Test merge_skills function."""

    def test_merge_skills_no_duplicates(self):
        """Test merging skills with no duplicates."""
        skill1 = Mock()
        skill1.name = 'skill1'
        skill2 = Mock()
        skill2.name = 'skill2'
        skill3 = Mock()
        skill3.name = 'skill3'

        result = merge_skills([[skill1], [skill2], [skill3]])

        assert len(result) == 3
        names = {s.name for s in result}
        assert names == {'skill1', 'skill2', 'skill3'}

    def test_merge_skills_with_duplicates(self):
        """Test merging skills with duplicates - later takes precedence."""
        skill1_v1 = Mock()
        skill1_v1.name = 'skill1'
        skill1_v1.version = 'v1'

        skill1_v2 = Mock()
        skill1_v2.name = 'skill1'
        skill1_v2.version = 'v2'

        skill2 = Mock()
        skill2.name = 'skill2'

        result = merge_skills([[skill1_v1, skill2], [skill1_v2]])

        assert len(result) == 2
        names = {s.name for s in result}
        assert names == {'skill1', 'skill2'}

        # Verify later version takes precedence
        skill1_result = next(s for s in result if s.name == 'skill1')
        assert skill1_result.version == 'v2'

    def test_merge_skills_empty_lists(self):
        """Test merging empty skill lists."""
        result = merge_skills([[], [], []])
        assert len(result) == 0

    def test_merge_skills_single_list(self):
        """Test merging single skill list."""
        skill1 = Mock()
        skill1.name = 'skill1'
        skill2 = Mock()
        skill2.name = 'skill2'

        result = merge_skills([[skill1, skill2]])

        assert len(result) == 2

    def test_merge_skills_precedence_order(self):
        """Test that skill precedence follows list order."""
        # Create three versions of the same skill
        skill_v1 = Mock()
        skill_v1.name = 'test_skill'
        skill_v1.priority = 'low'

        skill_v2 = Mock()
        skill_v2.name = 'test_skill'
        skill_v2.priority = 'medium'

        skill_v3 = Mock()
        skill_v3.name = 'test_skill'
        skill_v3.priority = 'high'

        # List order: low -> medium -> high
        # Should result in high priority (last one)
        result = merge_skills([[skill_v1], [skill_v2], [skill_v3]])

        assert len(result) == 1
        assert result[0].priority == 'high'

    def test_merge_skills_mixed_empty_and_filled(self):
        """Test merging with mix of empty and filled lists."""
        skill1 = Mock()
        skill1.name = 'skill1'
        skill2 = Mock()
        skill2.name = 'skill2'

        result = merge_skills([[], [skill1], [], [skill2], []])

        assert len(result) == 2


# ===== Integration Tests =====


class TestSkillLoaderIntegration:
    """Integration tests for the skill loader."""

    @pytest.mark.asyncio
    @patch('openhands.app_server.app_conversation.skill_loader.load_global_skills')
    @patch('openhands.sdk.context.skills.load_user_skills')
    @patch('openhands.app_server.app_conversation.skill_loader.load_repo_skills')
    async def test_full_loading_workflow(
        self,
        mock_load_repo,
        mock_load_user,
        mock_load_global,
        mock_async_remote_workspace,
    ):
        """Test the full workflow of loading all skill types."""
        # Create distinct mock skills for each source
        global_skill = Mock()
        global_skill.name = 'global_skill'

        user_skill = Mock()
        user_skill.name = 'user_skill'

        repo_skill = Mock()
        repo_skill.name = 'repo_skill'

        mock_load_global.return_value = [global_skill]
        mock_load_user.return_value = [user_skill]
        mock_load_repo.return_value = [repo_skill]

        # Simulate loading all sources
        global_skills = mock_load_global()
        user_skills = mock_load_user()
        repo_skills = await mock_load_repo(
            mock_async_remote_workspace, 'owner/repo', '/workspace'
        )

        # Merge all skills
        all_skills = merge_skills([global_skills, user_skills, repo_skills])

        assert len(all_skills) == 3
        names = {s.name for s in all_skills}
        assert names == {'global_skill', 'user_skill', 'repo_skill'}

    @pytest.mark.asyncio
    @patch('openhands.app_server.app_conversation.skill_loader.load_global_skills')
    @patch('openhands.sdk.context.skills.load_user_skills')
    @patch('openhands.app_server.app_conversation.skill_loader.load_repo_skills')
    async def test_loading_with_override_precedence(
        self,
        mock_load_repo,
        mock_load_user,
        mock_load_global,
        mock_async_remote_workspace,
    ):
        """Test that repo skills override user skills, and user skills override global."""
        # Create skills with same name but different sources
        global_skill = Mock()
        global_skill.name = 'common_skill'
        global_skill.source = 'global'

        user_skill = Mock()
        user_skill.name = 'common_skill'
        user_skill.source = 'user'

        repo_skill = Mock()
        repo_skill.name = 'common_skill'
        repo_skill.source = 'repo'

        mock_load_global.return_value = [global_skill]
        mock_load_user.return_value = [user_skill]
        mock_load_repo.return_value = [repo_skill]

        # Load and merge in correct precedence order
        global_skills = mock_load_global()
        user_skills = mock_load_user()
        repo_skills = await mock_load_repo(
            mock_async_remote_workspace, 'owner/repo', '/workspace'
        )

        all_skills = merge_skills([global_skills, user_skills, repo_skills])

        # Should have only one skill with repo source (highest precedence)
        assert len(all_skills) == 1
        assert all_skills[0].source == 'repo'
