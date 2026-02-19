"""Tests for safe_resolve_path defense-in-depth path containment checks."""

import os
from pathlib import Path

import pytest

from openhands.runtime.utils.files import safe_resolve_path


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace directory with test files."""
    (tmp_path / 'file.txt').write_text('hello')
    subdir = tmp_path / 'subdir'
    subdir.mkdir()
    (subdir / 'nested.txt').write_text('nested')
    return tmp_path


class TestSafeResolvePath:
    def test_absolute_path_within_workspace(self, workspace):
        result = safe_resolve_path(str(workspace / 'file.txt'), str(workspace))
        assert result == workspace / 'file.txt'

    def test_relative_path_within_workspace(self, workspace):
        result = safe_resolve_path('file.txt', str(workspace))
        assert result == workspace / 'file.txt'

    def test_nested_relative_path(self, workspace):
        result = safe_resolve_path('subdir/nested.txt', str(workspace))
        assert result == workspace / 'subdir' / 'nested.txt'

    def test_relative_path_with_dotdot_staying_inside(self, workspace):
        result = safe_resolve_path('subdir/../file.txt', str(workspace))
        assert result == workspace / 'file.txt'

    def test_relative_path_with_dotdot_escaping(self, workspace):
        with pytest.raises(PermissionError, match='outside the allowed directory'):
            safe_resolve_path('../etc/passwd', str(workspace))

    def test_absolute_path_outside_workspace(self, workspace):
        with pytest.raises(PermissionError, match='outside the allowed directory'):
            safe_resolve_path('/etc/passwd', str(workspace))

    def test_symlink_inside_workspace(self, workspace):
        target = workspace / 'file.txt'
        link = workspace / 'link.txt'
        link.symlink_to(target)
        result = safe_resolve_path('link.txt', str(workspace))
        assert result == workspace / 'file.txt'

    def test_symlink_escaping_workspace(self, workspace):
        link = workspace / 'escape_link'
        link.symlink_to('/etc')
        with pytest.raises(PermissionError, match='outside the allowed directory'):
            safe_resolve_path('escape_link/passwd', str(workspace))

    def test_defaults_to_cwd_when_no_base_dir(self):
        cwd = os.getcwd()
        result = safe_resolve_path('.')
        assert result == Path(os.path.realpath(cwd))

    def test_multiple_dotdot_traversal(self, workspace):
        with pytest.raises(PermissionError, match='outside the allowed directory'):
            safe_resolve_path('subdir/../../..', str(workspace))

    def test_dot_path(self, workspace):
        result = safe_resolve_path('.', str(workspace))
        assert result == workspace

    def test_workspace_root_path(self, workspace):
        result = safe_resolve_path(str(workspace), str(workspace))
        assert result == workspace

    def test_nonexistent_file_within_workspace(self, workspace):
        # Should not raise even if file doesn't exist - we only check containment
        result = safe_resolve_path('nonexistent.txt', str(workspace))
        assert result == workspace / 'nonexistent.txt'
