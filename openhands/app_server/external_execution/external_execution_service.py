from abc import ABC, abstractmethod
from typing import Any

from openhands.app_server.services.injector import Injector
from openhands.sdk.utils.models import DiscriminatedUnionMixin


class ExternalExecutionService(ABC):
    """Async external execution backend for durable, long-running work.

    This is intentionally separate from SandboxService. Sandboxes provide an
    interactive OpenHands agent-server environment; external execution is for
    submitting durable jobs and polling their logs/artifacts over time.
    """

    @abstractmethod
    async def estimate_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Estimate cost/resources for a job payload."""

    @abstractmethod
    async def submit_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit a job and return provider metadata including a job id."""

    @abstractmethod
    async def get_job(self, job_id: str) -> dict[str, Any]:
        """Return provider metadata for a submitted job."""

    @abstractmethod
    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Return provider status metadata for a submitted job."""

    @abstractmethod
    async def get_job_logs(
        self, job_id: str, limit: int | None = None, cursor: str | int | None = None
    ) -> dict[str, Any]:
        """Return or page through provider logs for a submitted job."""

    @abstractmethod
    async def cancel_job(
        self, job_id: str, reason: str | None = None
    ) -> dict[str, Any]:
        """Cancel a submitted job."""

    @abstractmethod
    async def list_artifacts(self, job_id: str) -> dict[str, Any]:
        """List artifacts produced by a submitted job."""

    @abstractmethod
    async def get_artifact_download_url(
        self, job_id: str, artifact_id: str
    ) -> dict[str, Any]:
        """Return a provider download URL for an artifact."""


class ExternalExecutionServiceInjector(
    DiscriminatedUnionMixin, Injector[ExternalExecutionService], ABC
):
    pass
