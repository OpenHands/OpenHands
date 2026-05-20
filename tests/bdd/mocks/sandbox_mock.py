"""Mock sandbox with in-memory filesystem simulation.

Provides a simulated runtime environment for testing agent tool execution
without requiring a real Docker sandbox.

Usage:
    sandbox = MockSandbox()
    sandbox.write_file("src/main.py", "print('hello')")
    result = await sandbox.execute("cat src/main.py")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ExecutionResult:
    """Result of a sandbox command execution."""

    def __init__(
        self,
        command: str,
        exit_code: int = 0,
        stdout: str = '',
        stderr: str = '',
    ) -> None:
        """Initialize execution result.

        Args:
            command: The command that was executed
            exit_code: Command exit code
            stdout: Standard output
            stderr: Standard error
        """
        self.command = command
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.success = exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'command': self.command,
            'exit_code': self.exit_code,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'success': self.success,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f'ExecutionResult(command={self.command!r}, '
            f'exit_code={self.exit_code}, success={self.success})'
        )


class MockFilesystem:
    """In-memory filesystem for sandbox testing."""

    def __init__(self, root_path: str = '/home/user') -> None:
        """Initialize mock filesystem.

        Args:
            root_path: Root path for the filesystem (default: /home/user)
        """
        self.root_path = Path(root_path)
        self.files: dict[str, str] = {}
        self.directories: set[Path] = {self.root_path, self.root_path / 'project'}
        self._seed_default_files()

    def _seed_default_files(self) -> None:
        """Seed filesystem with default files."""
        self.files[str(self.root_path / 'project' / 'README.md')] = (
            '# Project\n\nThis is a test project.\n'
        )
        self.files[str(self.root_path / 'project' / 'main.py')] = (
            "#!/usr/bin/env python3\n\ndef main():\n    print('Hello, World!')\n"
        )

    def write_file(self, path: str, content: str) -> None:
        """Write file to filesystem.

        Args:
            path: File path
            content: File content
        """
        file_path = self._resolve_path(path)
        self.files[str(file_path)] = content
        # Ensure parent directory exists
        parent = file_path.parent
        if parent not in self.directories:
            self.directories.add(parent)

    def read_file(self, path: str) -> str | None:
        """Read file from filesystem.

        Args:
            path: File path

        Returns:
            File content or None if not found
        """
        file_path = self._resolve_path(path)
        return self.files.get(str(file_path))

    def delete_file(self, path: str) -> bool:
        """Delete file from filesystem.

        Args:
            path: File path

        Returns:
            True if deleted, False if not found
        """
        file_path = self._resolve_path(path)
        if str(file_path) in self.files:
            del self.files[str(file_path)]
            return True
        return False

    def list_files(self, directory: str = '.') -> list[str]:
        """List files in directory.

        Args:
            directory: Directory path

        Returns:
            List of file paths
        """
        dir_path = self._resolve_path(directory)
        matching_files = []
        for file_path in self.files.keys():
            file_obj = Path(file_path)
            if file_obj.parent == dir_path:
                matching_files.append(file_path)
        return sorted(matching_files)

    def file_exists(self, path: str) -> bool:
        """Check if file exists.

        Args:
            path: File path

        Returns:
            True if file exists
        """
        file_path = self._resolve_path(path)
        return str(file_path) in self.files

    def _resolve_path(self, path: str) -> Path:
        """Resolve relative path to absolute path.

        Args:
            path: File path (absolute or relative)

        Returns:
            Resolved Path object
        """
        if path.startswith('/'):
            return Path(path)
        return self.root_path / 'project' / path

    def reset(self) -> None:
        """Clear filesystem and restore defaults."""
        self.files.clear()
        self.directories = {self.root_path, self.root_path / 'project'}
        self._seed_default_files()

    def get_tree(self) -> str:
        """Get filesystem tree representation."""
        lines = ['project/']
        for file_path in sorted(self.files.keys()):
            rel_path = Path(file_path).relative_to(self.root_path)
            lines.append(f'  {rel_path}')
        return '\n'.join(lines)


class MockSandbox:
    """Mock sandbox runtime environment for testing.

    Simulates sandbox behavior without Docker, supporting filesystem
    operations and command execution.
    """

    def __init__(self, root_path: str = '/home/user') -> None:
        """Initialize mock sandbox.

        Args:
            root_path: Root filesystem path
        """
        self.filesystem = MockFilesystem(root_path)
        self.execution_history: list[ExecutionResult] = []
        self.working_directory = str(self.filesystem.root_path / 'project')

    def write_file(self, path: str, content: str) -> None:
        """Write file to sandbox.

        Args:
            path: File path
            content: File content
        """
        self.filesystem.write_file(path, content)

    def read_file(self, path: str) -> str | None:
        """Read file from sandbox.

        Args:
            path: File path

        Returns:
            File content or None if not found
        """
        return self.filesystem.read_file(path)

    async def execute(self, command: str) -> ExecutionResult:
        """Execute command in sandbox.

        Simulates common shell commands without actually executing them.

        Args:
            command: Shell command to execute

        Returns:
            ExecutionResult with output
        """
        result = self._simulate_command(command)
        self.execution_history.append(result)
        return result

    def _simulate_command(self, command: str) -> ExecutionResult:
        """Simulate command execution.

        Args:
            command: Shell command

        Returns:
            ExecutionResult
        """
        # cat command
        if command.startswith('cat '):
            file_path = command[4:].strip()
            content = self.filesystem.read_file(file_path)
            if content is not None:
                return ExecutionResult(command, stdout=content)
            return ExecutionResult(
                command,
                exit_code=1,
                stderr=f'cat: {file_path}: No such file or directory',
            )

        # find command
        if command.startswith('find '):
            files = self.filesystem.list_files()
            stdout = '\n'.join(files)
            return ExecutionResult(command, stdout=stdout)

        # ls command
        if command.startswith('ls '):
            dir_path = command[3:].strip() or '.'
            files = self.filesystem.list_files(dir_path)
            stdout = '\n'.join(Path(f).name for f in files)
            return ExecutionResult(command, stdout=stdout)

        # echo command
        if command.startswith('echo '):
            text = command[5:].strip().strip('\'"')
            return ExecutionResult(command, stdout=text)

        # pwd command
        if command == 'pwd':
            return ExecutionResult(command, stdout=self.working_directory)

        # mkdir command
        if command.startswith('mkdir '):
            path = command[6:].strip()
            resolved = self.filesystem._resolve_path(path)
            self.filesystem.directories.add(resolved)
            return ExecutionResult(command)

        # rm command
        if command.startswith('rm '):
            file_path = command[3:].strip()
            if self.filesystem.delete_file(file_path):
                return ExecutionResult(command)
            return ExecutionResult(
                command,
                exit_code=1,
                stderr=f'rm: {file_path}: No such file or directory',
            )

        # Default: command not found simulation
        return ExecutionResult(
            command, exit_code=127, stderr=f'command not found: {command}'
        )

    def reset(self) -> None:
        """Reset sandbox to initial state."""
        self.filesystem.reset()
        self.execution_history.clear()
        self.working_directory = str(self.filesystem.root_path / 'project')

    def get_execution_history(self) -> list[ExecutionResult]:
        """Get command execution history.

        Returns:
            List of ExecutionResult objects
        """
        return self.execution_history.copy()

    def get_filesystem_tree(self) -> str:
        """Get filesystem tree representation.

        Returns:
            Formatted filesystem tree
        """
        return self.filesystem.get_tree()

    def get_stats(self) -> dict[str, Any]:
        """Get sandbox statistics.

        Returns:
            Dict with sandbox stats
        """
        return {
            'file_count': len(self.filesystem.files),
            'execution_count': len(self.execution_history),
            'working_directory': self.working_directory,
        }
