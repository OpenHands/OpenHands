from __future__ import annotations

from openhands.app_server.integrations.forgejo.service.base import ForgejoMixinBase
from openhands.app_server.integrations.service_types import SuggestedTask


class ForgejoFeaturesMixin(ForgejoMixinBase):
    """Microagent and feature helpers for Forgejo."""

    async def get_suggested_tasks(self) -> list[SuggestedTask]:  # type: ignore[override]
        # Suggested tasks are not yet implemented for Forgejo.
        return []
