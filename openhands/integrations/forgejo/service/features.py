from __future__ import annotations

import base64
from typing import Any

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.forgejo.service.base import ForgejoMixinBase
from openhands.integrations.service_types import (
    MicroagentContentResponse,
    MicroagentResponse,
    ProviderType,
    ResourceNotFoundError,
    SuggestedTask,
)


class ForgejoFeaturesMixin(ForgejoMixinBase):
    """Microagent and feature helpers for Forgejo."""

    async def _get_cursorrules_url(self, repository: str) -> str:
        owner, repo = self._split_repo(repository)
        return self._build_repo_api_url(owner, repo, 'contents', '.cursorrules')

    async def _get_microagents_directory_url(
        self, repository: str, microagents_path: str
    ) -> str:
        owner, repo = self._split_repo(repository)
        normalized_path = microagents_path.strip('/')
        return self._build_repo_api_url(owner, repo, 'contents', normalized_path)

    def _get_microagents_directory_params(self, microagents_path: str) -> dict | None:
        return None

    def _is_valid_microagent_file(self, item: dict[str, Any] | None) -> bool:
        if not isinstance(item, dict):
            return False
        if item.get('type') != 'file':
            return False
        name = item.get('name', '')
        return isinstance(name, str) and (
            name.endswith('.md') or name.endswith('.cursorrules')
        )

    def _get_file_name_from_item(self, item: dict[str, Any] | None) -> str:
        if not isinstance(item, dict):
            return ''
        name = item.get('name')
        return name if isinstance(name, str) else ''

    def _get_file_path_from_item(
        self, item: dict[str, Any] | None, microagents_path: str
    ) -> str:
        file_name = self._get_file_name_from_item(item)
        if not microagents_path:
            return file_name
        return f'{microagents_path.strip("/")}/{file_name}'

    async def get_microagents(self, repository: str) -> list[MicroagentResponse]:  # type: ignore[override]
        microagents_path = self._determine_microagents_path(repository)
        microagents: list[MicroagentResponse] = []

        try:
            directory_url = await self._get_microagents_directory_url(
                repository, microagents_path
            )
            items, _ = await self._make_request(directory_url)
        except ResourceNotFoundError:
            items = []
        except Exception as exc:
            # Fail gracefully if the directory cannot be inspected
            self._log_microagent_warning(repository, str(exc))
            items = []

        if isinstance(items, list):
            for item in items:
                if self._is_valid_microagent_file(item):
                    file_name = self._get_file_name_from_item(item)
                    file_path = self._get_file_path_from_item(item, microagents_path)
                    microagents.append(
                        self._create_microagent_response(file_name, file_path)
                    )

        cursorrules = await self._check_cursorrules_file(repository)
        if cursorrules:
            microagents.append(cursorrules)

        return microagents

    async def get_microagent_content(
        self, repository: str, file_path: str
    ) -> MicroagentContentResponse:  # type: ignore[override]
        owner, repo = self._split_repo(repository)
        normalized_path = file_path.lstrip('/')
        url = self._build_repo_api_url(owner, repo, 'contents', normalized_path)

        response, _ = await self._make_request(url)
        content = response.get('content') or ''
        encoding = (response.get('encoding') or 'base64').lower()

        if encoding == 'base64':
            try:
                decoded = base64.b64decode(content).decode('utf-8')
            except Exception:
                decoded = ''
        else:
            decoded = content

        try:
            return self._parse_microagent_content(decoded, file_path)
        except Exception:
            return MicroagentContentResponse(
                content=decoded,
                path=file_path,
                triggers=[],
                git_provider=ProviderType.FORGEJO.value,
            )

    async def get_suggested_tasks(self) -> list[SuggestedTask]:  # type: ignore[override]
        # Suggested tasks are not yet implemented for Forgejo.
        return []

    def _log_microagent_warning(self, repository: str, message: str) -> None:
        logger.debug(f'Forgejo microagent scan warning for {repository}: {message}')
