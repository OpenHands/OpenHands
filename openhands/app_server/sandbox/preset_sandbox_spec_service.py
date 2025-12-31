from dataclasses import dataclass

from openhands.app_server.sandbox.sandbox_spec_models import (
    SandboxSpecInfo,
    SandboxSpecInfoPage,
)
from openhands.app_server.sandbox.sandbox_spec_service import (
    SandboxSpecService,
)


@dataclass
class PresetSandboxSpecService(SandboxSpecService):
    """Service which uses a preset set of sandbox specs."""

    specs: list[SandboxSpecInfo]

    async def search_sandbox_specs(
        self, page_id: str | None = None, limit: int = 100
    ) -> SandboxSpecInfoPage:
        """Search for sandbox specs with pagination support."""
        # Apply pagination
        start_idx = 0
        if page_id:
            try:
                start_idx = int(page_id)
            except ValueError:
                start_idx = 0

        end_idx = start_idx + limit
        paginated_specs = self.specs[start_idx:end_idx]

        # Determine next page ID
        next_page_id = None
        if end_idx < len(self.specs):
            next_page_id = str(end_idx)

        return SandboxSpecInfoPage(items=paginated_specs, next_page_id=next_page_id)

    async def get_sandbox_spec(self, sandbox_spec_id: str) -> SandboxSpecInfo | None:
        """Get a single sandbox spec by ID, returning None if not found."""
        for spec in self.specs:
            if spec.id == sandbox_spec_id:
                return spec
        return None

    async def get_default_sandbox_spec(self) -> SandboxSpecInfo:
        return self.specs[0]
