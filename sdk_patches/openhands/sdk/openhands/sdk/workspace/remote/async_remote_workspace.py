from collections.abc import Generator
from pathlib import Path
from typing import Any

import httpx
from pydantic import PrivateAttr

from openhands.sdk.git.models import GitChange, GitDiff
from openhands.sdk.workspace.models import CommandResult, FileOperationResult
from openhands.sdk.workspace.remote.remote_workspace_mixin import RemoteWorkspaceMixin


class AsyncRemoteWorkspace(RemoteWorkspaceMixin):
    """Async Remote Workspace Implementation."""

    _client: httpx.AsyncClient | None = PrivateAttr(default=None)

    async def reset_client(self) -> None:
        """Reset the HTTP client to force re-initialization.

        This is useful when connection parameters (host, api_key) have changed
        and the client needs to be recreated with new values.
        """
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
        self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        client = self._client
        if client is None:
            # Configure reasonable timeouts for HTTP requests
            # - connect: 10 seconds to establish connection
            # - read: 60 seconds to read response (for LLM operations)
            # - write: 10 seconds to send request
            # - pool: 10 seconds to get connection from pool
            timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
            client = httpx.AsyncClient(
                base_url=self.host, timeout=timeout, headers=self._headers
            )
            self._client = client
        return client

    async def _execute(self, generator: Generator[dict[str, Any], httpx.Response, Any]):
        try:
            kwargs = next(generator)
            while True:
                response = await self.client.request(**kwargs)
                kwargs = generator.send(response)
        except StopIteration as e:
            return e.value

    async def execute_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: float = 30.0,
    ) -> CommandResult:
        """Execute a bash command on the remote system.

        This method starts a bash command via the remote agent server API,
        then polls for the output until the command completes.

        Args:
            command: The bash command to execute
            cwd: Working directory (optional)
            timeout: Timeout in seconds

        Returns:
            CommandResult: Result with stdout, stderr, exit_code, and other metadata
        """
        generator = self._execute_command_generator(command, cwd, timeout)
        result = await self._execute(generator)
        return result

    async def file_upload(
        self,
        source_path: str | Path,
        destination_path: str | Path,
    ) -> FileOperationResult:
        """Upload a file to the remote system.

        Reads the local file and sends it to the remote system via HTTP API.

        Args:
            source_path: Path to the local source file
            destination_path: Path where the file should be uploaded on remote system

        Returns:
            FileOperationResult: Result with success status and metadata
        """
        generator = self._file_upload_generator(source_path, destination_path)
        result = await self._execute(generator)
        return result

    async def file_download(
        self,
        source_path: str | Path,
        destination_path: str | Path,
    ) -> FileOperationResult:
        """Download a file from the remote system.

        Requests the file from the remote system via HTTP API and saves it locally.

        Args:
            source_path: Path to the source file on remote system
            destination_path: Path where the file should be saved locally

        Returns:
            FileOperationResult: Result with success status and metadata
        """
        generator = self._file_download_generator(source_path, destination_path)
        result = await self._execute(generator)
        return result

    async def git_changes(self, path: str | Path) -> list[GitChange]:
        """Get the git changes for the repository at the path given.

        Args:
            path: Path to the git repository

        Returns:
            list[GitChange]: List of changes

        Raises:
            Exception: If path is not a git repository or getting changes failed
        """
        generator = self._git_changes_generator(path)
        result = await self._execute(generator)
        return result

    async def git_diff(self, path: str | Path) -> GitDiff:
        """Get the git diff for the file at the path given.

        Args:
            path: Path to the file

        Returns:
            GitDiff: Git diff

        Raises:
            Exception: If path is not a git repository or getting diff failed
        """
        generator = self._git_diff_generator(path)
        result = await self._execute(generator)
        return result
