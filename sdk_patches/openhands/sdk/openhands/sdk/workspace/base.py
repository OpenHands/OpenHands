from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Any

from pydantic import BeforeValidator, Field

from openhands.sdk.git.models import GitChange, GitDiff
from openhands.sdk.logger import get_logger
from openhands.sdk.utils.models import DiscriminatedUnionMixin
from openhands.sdk.workspace.models import CommandResult, FileOperationResult


logger = get_logger(__name__)


def _convert_path_to_str(v: str | Path) -> str:
    """Convert Path objects to string for working_dir."""
    if isinstance(v, Path):
        return str(v)
    return v


class BaseWorkspace(DiscriminatedUnionMixin, ABC):
    """Abstract base class for workspace implementations.

    Workspaces provide a sandboxed environment where agents can execute commands,
    read/write files, and perform other operations. All workspace implementations
    support the context manager protocol for safe resource management.

    Example:
        >>> with workspace:
        ...     result = workspace.execute_command("echo 'hello'")
        ...     content = workspace.read_file("example.txt")
    """

    working_dir: Annotated[
        str,
        BeforeValidator(_convert_path_to_str),
        Field(
            description=(
                "The working directory for agent operations and tool execution. "
                "Accepts both string paths and Path objects. "
                "Path objects are automatically converted to strings."
            )
        ),
    ]

    def __enter__(self) -> "BaseWorkspace":
        """Enter the workspace context.

        Returns:
            Self for use in with statements
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the workspace context and cleanup resources.

        Default implementation performs no cleanup. Subclasses should override
        to add cleanup logic (e.g., stopping containers, closing connections).

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        pass

    @abstractmethod
    def execute_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: float = 30.0,
    ) -> CommandResult:
        """Execute a bash command on the system.

        Args:
            command: The bash command to execute
            cwd: Working directory for the command (optional)
            timeout: Timeout in seconds (defaults to 30.0)

        Returns:
            CommandResult: Result containing stdout, stderr, exit_code, and other
                metadata

        Raises:
            Exception: If command execution fails
        """
        ...

    @abstractmethod
    def file_upload(
        self,
        source_path: str | Path,
        destination_path: str | Path,
    ) -> FileOperationResult:
        """Upload a file to the system.

        Args:
            source_path: Path to the source file
            destination_path: Path where the file should be uploaded

        Returns:
            FileOperationResult: Result containing success status and metadata

        Raises:
            Exception: If file upload fails
        """
        ...

    @abstractmethod
    def file_download(
        self,
        source_path: str | Path,
        destination_path: str | Path,
    ) -> FileOperationResult:
        """Download a file from the system.

        Args:
            source_path: Path to the source file on the system
            destination_path: Path where the file should be downloaded

        Returns:
            FileOperationResult: Result containing success status and metadata

        Raises:
            Exception: If file download fails
        """
        ...

    @abstractmethod
    def git_changes(self, path: str | Path) -> list[GitChange]:
        """Get the git changes for the repository at the path given.

        Args:
            path: Path to the git repository

        Returns:
            list[GitChange]: List of changes

        Raises:
            Exception: If path is not a git repository or getting changes failed
        """

    @abstractmethod
    def git_diff(self, path: str | Path) -> GitDiff:
        """Get the git diff for the file at the path given.

        Args:
            path: Path to the file

        Returns:
            GitDiff: Git diff

        Raises:
            Exception: If path is not a git repository or getting diff failed
        """
