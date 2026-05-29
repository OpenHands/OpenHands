from dataclasses import dataclass
from typing import Any, AsyncGenerator

from fastapi import Request
from pydantic import Field, SecretStr

from openhands.app_server.external_execution.external_execution_service import (
    ExternalExecutionService,
    ExternalExecutionServiceInjector,
)
from openhands.app_server.external_execution.junglegrid_client import JungleGridClient
from openhands.app_server.services.injector import InjectorState


@dataclass
class JungleGridExecutionService(ExternalExecutionService):
    """Jungle Grid external execution adapter.

    This service is async-first: callers submit work, receive provider metadata
    such as a job id, and poll status/logs/artifacts without blocking OpenHands'
    default local, Docker, process, or remote sandbox behavior.
    """

    client: JungleGridClient

    async def estimate_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.client.estimate_job(payload)

    async def submit_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.client.submit_job(payload)

    async def get_job(self, job_id: str) -> dict[str, Any]:
        return await self.client.get_job(job_id)

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        return await self.client.get_job_status(job_id)

    async def get_job_logs(
        self, job_id: str, limit: int | None = None, cursor: str | int | None = None
    ) -> dict[str, Any]:
        return await self.client.get_job_logs(job_id, limit=limit, cursor=cursor)

    async def cancel_job(
        self, job_id: str, reason: str | None = None
    ) -> dict[str, Any]:
        return await self.client.cancel_job(job_id, reason=reason)

    async def list_artifacts(self, job_id: str) -> dict[str, Any]:
        return await self.client.list_artifacts(job_id)

    async def get_artifact_download_url(
        self, job_id: str, artifact_id: str
    ) -> dict[str, Any]:
        return await self.client.get_artifact_download_url(job_id, artifact_id)


class JungleGridExecutionServiceInjector(ExternalExecutionServiceInjector):
    """Dependency injector for Jungle Grid external execution."""

    api_key: SecretStr = Field(description='Jungle Grid API key')
    base_url: str = Field(description='Jungle Grid API base URL')
    workspace_id: str | None = Field(
        default=None,
        description='Optional Jungle Grid workspace or project identifier',
    )

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[ExternalExecutionService, None]:
        from openhands.app_server.config import get_httpx_client

        async with get_httpx_client(state, request) as httpx_client:
            yield JungleGridExecutionService(
                client=JungleGridClient(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    workspace_id=self.workspace_id,
                    httpx_client=httpx_client,
                )
            )
