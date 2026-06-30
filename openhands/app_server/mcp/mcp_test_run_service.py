"""Abstract service for MCP server test run persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from openhands.app_server.mcp.mcp_test_models import (
    McpServerTestRun,
    McpServerTestRunPage,
    McpServerTestRunSortOrder,
)
from openhands.app_server.services.injector import Injector
from openhands.sdk.utils.models import DiscriminatedUnionMixin


class McpServerTestRunService(ABC):
    @abstractmethod
    async def create_test_run(self, run: McpServerTestRun) -> McpServerTestRun:
        """Persist a new test run."""

    @abstractmethod
    async def save_test_run(self, run: McpServerTestRun) -> McpServerTestRun:
        """Update an existing test run."""

    @abstractmethod
    async def get_test_run(self, test_id: UUID) -> McpServerTestRun | None:
        """Get a single test run by id."""

    @abstractmethod
    async def get_running_test_run(self, server_id: str) -> McpServerTestRun | None:
        """Return an in-flight test run for the given server, if any."""

    @abstractmethod
    async def get_latest_test_run(self, server_id: str) -> McpServerTestRun | None:
        """Return the most recent test run for the given server."""

    @abstractmethod
    async def search_test_runs(
        self,
        server_id: str | None = None,
        sort_order: McpServerTestRunSortOrder = McpServerTestRunSortOrder.CREATED_AT_DESC,
        page_id: str | None = None,
        limit: int = 20,
    ) -> McpServerTestRunPage:
        """Search test runs, optionally filtered by server id."""


class McpServerTestRunServiceInjector(
    DiscriminatedUnionMixin, Injector[McpServerTestRunService], ABC
):
    pass
