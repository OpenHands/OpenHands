import asyncio
from abc import ABC, abstractmethod

from openhands.app_server.sandbox.sandbox_models import (
    SandboxInfo,
    SandboxPage,
    SandboxStatus,
)
from openhands.app_server.services.injector import Injector
from openhands.sdk.utils.models import DiscriminatedUnionMixin
from openhands.sdk.utils.paging import page_iterator


class SandboxService(ABC):
    """Service for accessing sandboxes in which conversations may be run."""

    @abstractmethod
    async def search_sandboxes(
        self,
        page_id: str | None = None,
        limit: int = 100,
    ) -> SandboxPage:
        """Search for sandboxes."""

    @abstractmethod
    async def get_sandbox(self, sandbox_id: str) -> SandboxInfo | None:
        """Get a single sandbox. Return None if the sandbox was not found."""

    @abstractmethod
    async def get_sandbox_by_session_api_key(
        self, session_api_key: str
    ) -> SandboxInfo | None:
        """Get a single sandbox by session API key. Return None if the sandbox was not found."""

    async def batch_get_sandboxes(
        self, sandbox_ids: list[str]
    ) -> list[SandboxInfo | None]:
        """Get a batch of sandboxes, returning None for any which were not found."""
        results = await asyncio.gather(
            *[self.get_sandbox(sandbox_id) for sandbox_id in sandbox_ids]
        )
        return results

    @abstractmethod
    async def start_sandbox(self, sandbox_spec_id: str | None = None) -> SandboxInfo:
        """Begin the process of starting a sandbox.

        Return the info on the new sandbox. If no spec is selected, use the default.
        """

    @abstractmethod
    async def resume_sandbox(self, sandbox_id: str) -> bool:
        """Begin the process of resuming a sandbox.

        Return True if the sandbox exists and is being resumed or is already running.
        Return False if the sandbox did not exist.
        """

    @abstractmethod
    async def pause_sandbox(self, sandbox_id: str) -> bool:
        """Begin the process of pausing a sandbox.

        Return True if the sandbox exists and is being paused or is already paused.
        Return False if the sandbox did not exist.
        """

    @abstractmethod
    async def delete_sandbox(self, sandbox_id: str) -> bool:
        """Begin the process of deleting a sandbox (which may involve stopping it).

        Return False if the sandbox did not exist.
        """

    async def pause_old_sandboxes(self, max_num_sandboxes: int) -> list[str]:
        """Pause the oldest sandboxes if there are more than max_num_sandboxes running.
        In a multi user environment, this will pause sandboxes only for the current user.

        Args:
            max_num_sandboxes: Maximum number of sandboxes to keep running

        Returns:
            List of sandbox IDs that were paused
        """
        if max_num_sandboxes <= 0:
            raise ValueError('max_num_sandboxes must be greater than 0')

        # Get all running sandboxes (iterate through all pages)
        running_sandboxes = []
        async for sandbox in page_iterator(self.search_sandboxes, limit=100):
            if sandbox.status == SandboxStatus.RUNNING:
                running_sandboxes.append(sandbox)

        # If we're within the limit, no cleanup needed
        if len(running_sandboxes) <= max_num_sandboxes:
            return []

        # Sort by creation time (oldest first)
        running_sandboxes.sort(key=lambda x: x.created_at)

        # Determine how many to pause
        num_to_pause = len(running_sandboxes) - max_num_sandboxes
        sandboxes_to_pause = running_sandboxes[:num_to_pause]

        # Stop the oldest sandboxes
        paused_sandbox_ids = []
        for sandbox in sandboxes_to_pause:
            try:
                success = await self.pause_sandbox(sandbox.id)
                if success:
                    paused_sandbox_ids.append(sandbox.id)
            except Exception:
                # Continue trying to pause other sandboxes even if one fails
                pass

        return paused_sandbox_ids


class SandboxServiceInjector(DiscriminatedUnionMixin, Injector[SandboxService], ABC):
    pass
