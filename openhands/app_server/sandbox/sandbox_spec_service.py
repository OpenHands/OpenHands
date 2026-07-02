import asyncio
import logging
import os
from abc import ABC, abstractmethod

from openhands.agent_server import env_parser
from openhands.app_server.errors import SandboxError
from openhands.app_server.sandbox.sandbox_spec_models import (
    SandboxSpecInfo,
    SandboxSpecInfoPage,
)
from openhands.app_server.services.injector import Injector
from openhands.sdk.utils.models import DiscriminatedUnionMixin

# The version of the agent server to use for deployments.
# Typically this will be the same as the values from the pyproject.toml
AGENT_SERVER_IMAGE = 'ghcr.io/openhands/agent-server:1.30.0-python'


class SandboxSpecService(ABC):
    """Service for managing Sandbox specs.

    At present this is read only. The plan is that later this class will allow building
    and deleting sandbox specs and limiting access by user and group. It would also be
    nice to be able to set the desired number of warm sandboxes for a spec and scale
    this up and down.
    """

    @abstractmethod
    async def search_sandbox_specs(
        self, page_id: str | None = None, limit: int = 100
    ) -> SandboxSpecInfoPage:
        """Search for sandbox specs."""

    @abstractmethod
    async def get_sandbox_spec(self, sandbox_spec_id: str) -> SandboxSpecInfo | None:
        """Get a single sandbox spec, returning None if not found."""

    async def get_default_sandbox_spec(self) -> SandboxSpecInfo:
        """Get the default sandbox spec."""
        page = await self.search_sandbox_specs()
        if not page.items:
            raise SandboxError('No sandbox specs available!')
        return page.items[0]

    async def batch_get_sandbox_specs(
        self, sandbox_spec_ids: list[str]
    ) -> list[SandboxSpecInfo | None]:
        """Get a batch of sandbox specs, returning None for any not found."""
        results = await asyncio.gather(
            *[
                self.get_sandbox_spec(sandbox_spec_id)
                for sandbox_spec_id in sandbox_spec_ids
            ]
        )
        return results


class SandboxSpecServiceInjector(
    DiscriminatedUnionMixin, Injector[SandboxSpecService], ABC
):
    pass


async def resolve_sandbox_spec(
    sandbox_spec_id: str | None,
    user_default_spec_id: str | None,
    sandbox_spec_service: SandboxSpecService,
    logger: logging.Logger,
) -> SandboxSpecInfo:
    """Return the SandboxSpecInfo to use for a new sandbox.

    Resolution order:
    1. ``sandbox_spec_id`` (caller-explicit) — not found is a hard error.
    2. ``user_default_spec_id`` (user preference) — if missing, log a warning
       and fall back to the system default.
    3. System default (first spec returned by the service).
    """
    from_user_default = sandbox_spec_id is None and user_default_spec_id is not None
    effective_id = (
        sandbox_spec_id if sandbox_spec_id is not None else user_default_spec_id
    )

    if effective_id is None:
        return await sandbox_spec_service.get_default_sandbox_spec()

    spec = await sandbox_spec_service.get_sandbox_spec(effective_id)
    if spec is not None:
        return spec

    if from_user_default:
        logger.warning(
            'User default sandbox spec %r not found; falling back to system default.',
            effective_id,
        )
        return await sandbox_spec_service.get_default_sandbox_spec()

    raise ValueError(f'Sandbox Spec {effective_id!r} not found')


def get_agent_server_image() -> str:
    agent_server_image_repository = os.getenv('AGENT_SERVER_IMAGE_REPOSITORY')
    agent_server_image_tag = os.getenv('AGENT_SERVER_IMAGE_TAG')
    if agent_server_image_repository and agent_server_image_tag:
        return f'{agent_server_image_repository}:{agent_server_image_tag}'
    return AGENT_SERVER_IMAGE


def is_custom_agent_server_image() -> bool:
    """True only when an admin pinned a custom sandbox image (tag differs from the
    release-default tag). Default/upgrade installs keep the release tag, never gated."""
    tag = os.getenv('AGENT_SERVER_IMAGE_TAG')
    return bool(tag) and tag != AGENT_SERVER_IMAGE.rsplit(':', 1)[-1]


# Prefixes for environment variables that should be auto-forwarded to agent-server
# These are typically configuration variables that affect the agent's behavior
AUTO_FORWARD_PREFIXES = ('LLM_', 'LMNR_')


def get_agent_server_env() -> dict[str, str]:
    """Get environment variables to be injected into agent server sandbox environments.

    This function combines two sources of environment variables:

    1. **Auto-forwarded variables**: Environment variables with certain prefixes
       (e.g., LLM_*, LMNR_*) are automatically forwarded to the agent-server container.
       This ensures that LLM configuration like timeouts and retry settings
       work correctly in the two-container V1 architecture, as well as
       Laminar monitoring/analytics configuration.

    2. **Explicit overrides via OH_AGENT_SERVER_ENV**: A JSON string that allows
       setting arbitrary environment variables in the agent-server container.
       Values set here take precedence over auto-forwarded variables.

    Auto-forwarded prefixes:
        - LLM_* : LLM configuration (timeout, retries, model settings, etc.)
        - LMNR_* : Laminar monitoring/analytics configuration

    Usage:
        # Auto-forwarding (no action needed):
        export LLM_TIMEOUT=3600
        export LLM_NUM_RETRIES=10
        # These will automatically be available in the agent-server

        # Auto-forwarding for Laminar:
        export LMNR_PROJECT_API_KEY=your-api-key
        export LMNR_BASE_URL=https://app.lmnr.ai
        # These will automatically be available in the agent-server

        # Explicit override via JSON:
        OH_AGENT_SERVER_ENV='{"DEBUG": "true", "CUSTOM_VAR": "value"}'

        # Override an auto-forwarded variable:
        export LLM_TIMEOUT=3600  # Would be auto-forwarded as 3600
        OH_AGENT_SERVER_ENV='{"LLM_TIMEOUT": "7200"}'  # Overrides to 7200

    Returns:
        dict[str, str]: Dictionary of environment variable names to values.
                       Returns empty dict if no variables are found.

    Raises:
        JSONDecodeError: If OH_AGENT_SERVER_ENV contains invalid JSON.
    """
    result: dict[str, str] = {}

    # Step 1: Auto-forward environment variables with recognized prefixes
    for key, value in os.environ.items():
        if any(key.startswith(prefix) for prefix in AUTO_FORWARD_PREFIXES):
            result[key] = value

    # Step 2: Apply explicit overrides from OH_AGENT_SERVER_ENV
    # These take precedence over auto-forwarded variables
    explicit_env = env_parser.from_env(dict[str, str], 'OH_AGENT_SERVER_ENV')
    result.update(explicit_env)

    return result
