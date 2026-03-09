import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from openhands.runtime.utils import git_changes, git_diff, git_handler
from openhands.runtime.utils.git_handler import CommandResult, GitHandler


@pytest.mark.skipif(sys.platform == 'win32', reason='Windows is not supported')
class TestGitHandler(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for our test repositories
        self.test_dir = tempfile.mkdtemp()
        self.origin_dir = os.path.join(self.test_dir, 'origin')
        self.local_dir = os.path.join(self.test_dir, 'local')

        # Create the directories
        os.makedirs(self.origin_dir, exist_ok=True)
        os.makedirs(self.local_dir, exist_ok=True)

        # Track executed commands for verification
        self.executed_commands = []
        self.created_files = []

        # Initialize the GitHandler with our mock functions
        self.git_handler = GitHandler(
            execute_shell_fn=self._execute_command, create_file_fn=self._create_file
        )
        self.git_handler.set_cwd(self.local_dir)

        self.git_handler.git_changes_cmd = f'python3 {git_changes.__file__}'
        self.git_handler.git_diff_cmd = f'python3 {git_diff.__file__} {{file_path}}'

        # Set up the git repositories
        self._setup_git_repos()

    def tearDown(self):
        # Clean up the temporary directories
        shutil.rmtree(self.test_dir)

    def _execute_command(self, cmd, cwd=None):
        """Execute a shell command and return the result."""
        result = subprocess.run(
            args=cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
        )
        stderr = result.stderr or b''
        stdout = result.stdout or b''
        return CommandResult((stderr + stdout).decode(), result.returncode)

    def run_command(self, cmd, cwd=None):
        result = self._execute_command(cmd, cwd)
        if result.exit_code != 0:
            raise RuntimeError(
                f'command_error:{cmd};{result.exit_code};{result.content}'
            )

    def _create_file(self, path, content):
        """Mock function for creating files."""
        self.created_files.append((path, content))
        try:
            with open(path, 'w') as f:
                f.write(content)
            return 0
        except Exception:
            return -1

    def write_file(
        self,
        dir: str,
        name: str,
        additional_content: tuple[str, ...] = ('Line 1', 'Line 2', 'Line 3'),
    ):
        with open(os.path.join(dir, name), 'w') as f:
            f.write(name)
            for line in additional_content:
                f.write('\n')
                f.write(line)
        assert os.path.exists(os.path.join(dir, name))

    def _setup_git_repos(self):
        """Set up real git repositories for testing."""
        # Set up origin repository
        self.run_command('git init --initial-branch=main', self.origin_dir)
        self._execute_command(
            "git config user.email 'test@example.com'", self.origin_dir
        )
        self._execute_command("git config user.name 'Test User'", self.origin_dir)

        # Set up the initial state...
        self.write_file(self.origin_dir, 'unchanged.txt')
        self.write_file(self.origin_dir, 'committed_modified.txt')
        self.write_file(self.origin_dir, 'staged_modified.txt')
        self.write_file(self.origin_dir, 'unstaged_modified.txt')
        self.write_file(self.origin_dir, 'committed_delete.txt')
        self.write_file(self.origin_dir, 'staged_delete.txt')
        self.write_file(self.origin_dir, 'unstaged_delete.txt')
        self.run_command("git add . && git commit -m 'Initial Commit'", self.origin_dir)

        # Clone the origin repository to local
        self.run_command(f'git clone "{self.origin_dir}" "{self.local_dir}"')

        self._execute_command(
            "git config user.email 'test@example.com'", self.local_dir
        )
        self._execute_command("git config user.name 'Test User'", self.local_dir)

        self.run_command('git checkout -b feature-branch', self.local_dir)

        # Setup committed changes...
        self.write_file(self.local_dir, 'committed_modified.txt', ('Line 4',))
        self.write_file(self.local_dir, 'committed_add.txt')
        os.remove(os.path.join(self.local_dir, 'committed_delete.txt'))
        self.run_command(
            "git add . && git commit -m 'First batch of changes'", self.local_dir
        )

        # Setup staged changes...
        self.write_file(self.local_dir, 'staged_modified.txt', ('Line 4',))
        self.write_file(self.local_dir, 'staged_add.txt')
        os.remove(os.path.join(self.local_dir, 'staged_delete.txt'))
        self.run_command('git add .', self.local_dir)

        # Setup unstaged changes...
        self.write_file(self.local_dir, 'unstaged_modified.txt', ('Line 4',))
        self.write_file(self.local_dir, 'unstaged_add.txt')
        os.remove(os.path.join(self.local_dir, 'unstaged_delete.txt'))

    def setup_nested(self):
        nested_1 = Path(self.local_dir, 'nested 1')
        nested_1.mkdir()
        nested_1 = str(nested_1)
        self.run_command('git init --initial-branch=main', nested_1)
        self._execute_command("git config user.email 'test@example.com'", nested_1)
        self._execute_command("git config user.name 'Test User'", nested_1)
        self.write_file(nested_1, 'committed_add.txt')
        self.run_command('git add .', nested_1)
        self.run_command('git commit -m "Initial Commit"', nested_1)
        self.write_file(nested_1, 'staged_add.txt')

        nested_2 = Path(self.local_dir, 'nested_2')
        nested_2.mkdir()
        nested_2 = str(nested_2)
        self.run_command('git init --initial-branch=main', nested_2)
        self._execute_command("git config user.email 'test@example.com'", nested_2)
        self._execute_command("git config user.name 'Test User'", nested_2)
        self.write_file(nested_2, 'committed_add.txt')
        self.run_command('git add .', nested_2)
        self.run_command('git commit -m "Initial Commit"', nested_2)
        self.write_file(nested_2, 'unstaged_add.txt')

    def test_get_git_changes(self):
        """Test with unpushed commits, staged commits, and unstaged commits"""
        changes = self.git_handler.get_git_changes()

        expected_changes = [
            {'status': 'A', 'path': 'committed_add.txt'},
            {'status': 'D', 'path': 'committed_delete.txt'},
            {'status': 'M', 'path': 'committed_modified.txt'},
            {'status': 'A', 'path': 'staged_add.txt'},
            {'status': 'D', 'path': 'staged_delete.txt'},
            {'status': 'M', 'path': 'staged_modified.txt'},
            {'status': 'A', 'path': 'unstaged_add.txt'},
            {'status': 'D', 'path': 'unstaged_delete.txt'},
            {'status': 'M', 'path': 'unstaged_modified.txt'},
        ]

        assert changes == expected_changes

    def test_get_git_changes_after_push(self):
        """Test with staged commits, and unstaged commits"""
        self.run_command('git push -u origin feature-branch', self.local_dir)
        changes = self.git_handler.get_git_changes()

        expected_changes = [
            {'status': 'A', 'path': 'staged_add.txt'},
            {'status': 'D', 'path': 'staged_delete.txt'},
            {'status': 'M', 'path': 'staged_modified.txt'},
            {'status': 'A', 'path': 'unstaged_add.txt'},
            {'status': 'D', 'path': 'unstaged_delete.txt'},
            {'status': 'M', 'path': 'unstaged_modified.txt'},
        ]

        assert changes == expected_changes

    def test_get_git_changes_nested_repos(self):
        """Test with staged commits, and unstaged commits"""
        self.setup_nested()

        changes = self.git_handler.get_git_changes()

        expected_changes = [
            {'status': 'A', 'path': 'committed_add.txt'},
            {'status': 'D', 'path': 'committed_delete.txt'},
            {'status': 'M', 'path': 'committed_modified.txt'},
            {'status': 'A', 'path': 'nested 1/committed_add.txt'},
            {'status': 'A', 'path': 'nested 1/staged_add.txt'},
            {'status': 'A', 'path': 'nested_2/committed_add.txt'},
            {'status': 'A', 'path': 'nested_2/unstaged_add.txt'},
            {'status': 'A', 'path': 'staged_add.txt'},
            {'status': 'D', 'path': 'staged_delete.txt'},
            {'status': 'M', 'path': 'staged_modified.txt'},
            {'status': 'A', 'path': 'unstaged_add.txt'},
            {'status': 'D', 'path': 'unstaged_delete.txt'},
            {'status': 'M', 'path': 'unstaged_modified.txt'},
        ]

        assert changes == expected_changes

    def test_get_git_diff_staged_modified(self):
        """Test on a staged modified"""
        diff = self.git_handler.get_git_diff('staged_modified.txt')
        expected_diff = {
            'original': 'staged_modified.txt\nLine 1\nLine 2\nLine 3',
            'modified': 'staged_modified.txt\nLine 4',
        }
        assert diff == expected_diff

    def test_get_git_diff_unchanged(self):
        """Test that get_git_diff delegates to the git_diff module."""
        diff = self.git_handler.get_git_diff('unchanged.txt')
        expected_diff = {
            'original': 'unchanged.txt\nLine 1\nLine 2\nLine 3',
            'modified': 'unchanged.txt\nLine 1\nLine 2\nLine 3',
        }
        assert diff == expected_diff

    def test_get_git_diff_unpushed(self):
        """Test that get_git_diff delegates to the git_diff module."""
        diff = self.git_handler.get_git_diff('committed_modified.txt')
        expected_diff = {
            'original': 'committed_modified.txt\nLine 1\nLine 2\nLine 3',
            'modified': 'committed_modified.txt\nLine 4',
        }
        assert diff == expected_diff

    def test_get_git_diff_unstaged_add(self):
        """Test that get_git_diff delegates to the git_diff module."""
        diff = self.git_handler.get_git_diff('unstaged_add.txt')
        expected_diff = {
            'original': '',
            'modified': 'unstaged_add.txt\nLine 1\nLine 2\nLine 3',
        }
        assert diff == expected_diff

    def test_get_git_changes_fallback(self):
        """Test that get_git_changes falls back to creating a script file when needed."""
        # Break the git changes command
        with patch(
            'openhands.runtime.utils.git_handler.GIT_CHANGES_CMD',
            'non-existant-command',
        ):
            self.git_handler.git_changes_cmd = git_handler.GIT_CHANGES_CMD

            changes = self.git_handler.get_git_changes()

            expected_changes = [
                {'status': 'A', 'path': 'committed_add.txt'},
                {'status': 'D', 'path': 'committed_delete.txt'},
                {'status': 'M', 'path': 'committed_modified.txt'},
                {'status': 'A', 'path': 'staged_add.txt'},
                {'status': 'D', 'path': 'staged_delete.txt'},
                {'status': 'M', 'path': 'staged_modified.txt'},
                {'status': 'A', 'path': 'unstaged_add.txt'},
                {'status': 'D', 'path': 'unstaged_delete.txt'},
                {'status': 'M', 'path': 'unstaged_modified.txt'},
            ]

            assert changes == expected_changes

    def test_get_git_diff_fallback(self):
        """Test that get_git_diff delegates to the git_diff module."""
        # Break the git diff command
        with patch(
            'openhands.runtime.utils.git_handler.GIT_DIFF_CMD', 'non-existant-command'
        ):
            self.git_handler.git_diff_cmd = git_handler.GIT_DIFF_CMD

            diff = self.git_handler.get_git_diff('unchanged.txt')
            expected_diff = {
                'original': 'unchanged.txt\nLine 1\nLine 2\nLine 3',
                'modified': 'unchanged.txt\nLine 1\nLine 2\nLine 3',
            }
            assert diff == expected_diff

    def test_get_git_diff_command_injection_is_sanitized(self):
        """Verify that a malicious path does not execute injected shell commands."""
        sentinel = os.path.join(self.test_dir, 'injected')
        # A payload that would create the sentinel file if injection were possible
        malicious_path = f'"; touch {sentinel}; echo "'

        # get_git_diff should raise (no such file) rather than executing the payload
        with self.assertRaises(ValueError):
            self.git_handler.get_git_diff(malicious_path)

        assert not os.path.exists(sentinel), (
            'Shell injection succeeded: sentinel file was created'
        )

    def test_get_git_diff_no_cwd(self):
        """Raises ValueError('no_dir_in_git_diff') when cwd has not been set."""
        handler = GitHandler(
            execute_shell_fn=self._execute_command,
            create_file_fn=self._create_file,
        )
        with self.assertRaises(ValueError) as ctx:
            handler.get_git_diff('some_file.txt')
        assert str(ctx.exception) == 'no_dir_in_git_diff'

    def test_get_git_diff_double_fallback_raises(self):
        """Raises ValueError('error_in_git_diff') when cmd differs from GIT_DIFF_CMD and still fails."""
        # Simulate being in fallback mode: cmd != GIT_DIFF_CMD but it still fails
        self.git_handler.git_diff_cmd = 'non-existent-command {file_path}'
        with self.assertRaises(ValueError) as ctx:
            self.git_handler.get_git_diff('unchanged.txt')
        assert str(ctx.exception) == 'error_in_git_diff'


class TestGitShowCmdBuilder:
    """Pure unit tests for _make_git_show_cmd — no filesystem or subprocess access."""

    def test_plain_path(self):
        cmd = git_diff._make_git_show_cmd('abc123', 'file.txt')
        assert shlex.split(cmd) == ['git', 'show', 'abc123:file.txt']

    def test_path_with_spaces(self):
        cmd = git_diff._make_git_show_cmd('abc123', 'file with spaces.txt')
        assert shlex.split(cmd) == ['git', 'show', 'abc123:file with spaces.txt']

    def test_path_with_single_quote(self):
        cmd = git_diff._make_git_show_cmd('abc123', "it's a test.txt")
        assert shlex.split(cmd) == ['git', 'show', "abc123:it's a test.txt"]

    def test_injection_attempt(self):
        cmd = git_diff._make_git_show_cmd('abc123', '"; rm -rf /; echo "')
        assert shlex.split(cmd) == ['git', 'show', 'abc123:"; rm -rf /; echo "']


def test_get_git_diff_file_too_large():
    """Raises ValueError('file_to_large') when the file exceeds the size limit."""
    with patch('os.path.getsize', return_value=git_diff.MAX_FILE_SIZE_FOR_GIT_DIFF + 1):
        with pytest.raises(ValueError, match='file_to_large'):
            git_diff.get_git_diff('/nonexistent/path.txt')


def test_get_git_diff_no_repository():
    """Raises ValueError('no_repository') when the file is outside any git repository."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, 'file.txt')
        Path(file_path).write_text('content')
        with pytest.raises(ValueError, match='no_repository'):
            git_diff.get_git_diff(file_path)
