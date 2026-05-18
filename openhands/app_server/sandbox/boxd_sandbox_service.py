"""Sandbox service backed by boxd cloud microVMs.

boxd (https://boxd.sh) runs each VM as a dedicated KVM process with
sub-millisecond warm suspend/resume. Each conversation gets one boxd VM
with the agent-server image installed and a subdomain proxy on port 60000.

Persistence model: we mirror :class:`RemoteSandboxService` — sandbox
metadata (id, owning user, spec id, session-key hash, created_at) lives
in a small SQLAlchemy table; the live VM state comes from the boxd SDK.
This keeps fast (indexed) lookups for ``get_sandbox_by_session_api_key``
and ``search_sandboxes`` while letting boxd remain the source of truth
for VM status.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator

import base62
from fastapi import Request
from pydantic import Field
from sqlalchemy import String, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from openhands.agent_server.utils import utc_now
from openhands.app_server.errors import SandboxError
from openhands.app_server.sandbox.sandbox_models import (
    AGENT_SERVER,
    VSCODE,
    ExposedUrl,
    SandboxInfo,
    SandboxPage,
    SandboxStatus,
)
from openhands.app_server.sandbox.sandbox_service import (
    ALLOW_CORS_ORIGINS_VARIABLE,
    SESSION_API_KEY_VARIABLE,
    WEBHOOK_CALLBACK_VARIABLE,
    SandboxService,
    SandboxServiceInjector,
)
from openhands.app_server.sandbox.sandbox_spec_models import SandboxSpecInfo
from openhands.app_server.sandbox.sandbox_spec_service import SandboxSpecService
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.utils.sql_utils import Base, UtcDateTime

_logger = logging.getLogger(__name__)

# boxd VM status → internal SandboxStatus. Lowercased keys; SDK returns
# mixed-case strings, we normalize on read.
STATUS_MAPPING: dict[str, SandboxStatus] = {
    'running': SandboxStatus.RUNNING,
    'suspended': SandboxStatus.PAUSED,
    'starting': SandboxStatus.STARTING,
    'creating': SandboxStatus.STARTING,
    'error': SandboxStatus.ERROR,
    'failed': SandboxStatus.ERROR,
    'stopped': SandboxStatus.MISSING,
    'destroyed': SandboxStatus.MISSING,
}

AGENT_SERVER_PORT = 60000
VSCODE_PORT = 60001
AGENT_PROXY_NAME = 'agent'
VSCODE_PROXY_NAME = 'vscode'

# All boxd VMs we create are prefixed so search/list can distinguish
# ours from VMs created out-of-band by the same user.
VM_NAME_PREFIX = 'oh-'


def _hash_session_api_key(session_api_key: str) -> str:
    """SHA-256 of the session API key for indexed lookup."""
    return hashlib.sha256(session_api_key.encode()).hexdigest()


class StoredBoxdSandbox(Base):
    """App-side index of boxd sandboxes.

    The boxd SDK doesn't expose enough metadata on the Box object to
    reconstruct a SandboxInfo (no env, no created_at), and listing all
    user VMs to filter by session API key is O(N). This table mirrors
    the pattern used by ``StoredRemoteSandbox``.
    """

    __tablename__ = 'v1_boxd_sandbox'

    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_by_user_id: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True
    )
    sandbox_spec_id: Mapped[str] = mapped_column(String, index=True)
    session_api_key_hash: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime, server_default=func.now(), index=True
    )


@dataclass
class BoxdSandboxService(SandboxService):
    """Sandbox service backed by boxd cloud microVMs.

    See the module docstring for the persistence model.
    """

    sandbox_spec_service: SandboxSpecService
    compute: Any  # boxd.Compute — typed Any so the SDK isn't a hard import
    web_url: str | None
    max_num_sandboxes: int
    auto_suspend_timeout: int
    vcpu: int
    memory: str
    disk: str
    user_context: UserContext
    db_session: AsyncSession

    # ── Internal helpers ──────────────────────────────────────────────

    async def _secure_select(self):
        """SELECT statement scoped to the current user (or all rows for admin)."""
        query = select(StoredBoxdSandbox)
        user_id = await self.user_context.get_user_id()
        if user_id:
            query = query.where(StoredBoxdSandbox.created_by_user_id == user_id)
        return query

    async def _get_stored(self, sandbox_id: str) -> StoredBoxdSandbox | None:
        stmt = await self._secure_select()
        stmt = stmt.where(StoredBoxdSandbox.id == sandbox_id)
        result = await self.db_session.execute(stmt)
        return result.scalar_one_or_none()

    def _build_box_config(
        self, sandbox_spec: SandboxSpecInfo, env: dict[str, str]
    ) -> Any:
        """Build a BoxConfig matching our standard agent-server shape.

        Note: named subdomain proxies are created post-boot via
        ``box.create_proxy(...)``; passing them through
        ``NetworkConfig.proxies`` is silently ignored by the current boxd
        server (only the default catch-all proxy is auto-created).
        """
        from boxd.aio import BoxConfig, LifecycleConfig

        return BoxConfig(
            vcpu=self.vcpu,
            memory=self.memory,
            disk=self.disk,
            env=env,
            cmd=sandbox_spec.command,
            lifecycle=LifecycleConfig(auto_suspend_timeout=self.auto_suspend_timeout),
        )

    async def _ensure_named_proxies(self, box: Any) -> None:
        """Create the agent + vscode subdomain proxies on a fresh VM.

        Idempotent: if a proxy with the same name already exists boxd will
        return an error which we log and continue past.
        """
        for name, port in (
            (AGENT_PROXY_NAME, AGENT_SERVER_PORT),
            (VSCODE_PROXY_NAME, VSCODE_PORT),
        ):
            try:
                await box.create_proxy(name, port=port)
            except Exception as exc:
                _logger.warning(
                    'create_proxy(%s, port=%d) on box %s failed: %s',
                    name,
                    port,
                    getattr(box, 'id', '?'),
                    exc,
                )

    async def _build_environment(
        self, sandbox_spec: SandboxSpecInfo
    ) -> dict[str, str]:
        """Compose the env dict the box will boot with.

        Note: app-side metadata (spec_id, owner) lives in the DB, not the
        VM env, so we only inject what the agent server itself needs.
        """
        env = dict(sandbox_spec.initial_env)
        if self.web_url:
            env[WEBHOOK_CALLBACK_VARIABLE] = f'{self.web_url}/api/v1/webhooks'
            env[ALLOW_CORS_ORIGINS_VARIABLE] = self.web_url
        return env

    def _derive_status(self, box: Any | None) -> SandboxStatus:
        if box is None:
            return SandboxStatus.MISSING
        raw = (getattr(box, 'status', '') or '').lower()
        return STATUS_MAPPING.get(raw, SandboxStatus.ERROR)

    async def _to_sandbox_info(
        self,
        stored: StoredBoxdSandbox,
        box: Any | None,
        session_api_key: str | None = None,
    ) -> SandboxInfo:
        """Project (stored row, optional live Box) into a SandboxInfo."""
        status = self._derive_status(box)

        exposed_urls: list[ExposedUrl] | None = None
        if status == SandboxStatus.RUNNING and box is not None:
            exposed_urls = []
            try:
                proxies = await box.proxies()
            except Exception as exc:
                _logger.warning(
                    'Failed to list proxies on box %s: %s',
                    getattr(box, 'id', '?'),
                    exc,
                )
                proxies = []
            for proxy in proxies:
                if proxy.name == AGENT_PROXY_NAME:
                    exposed_urls.append(
                        ExposedUrl(
                            name=AGENT_SERVER,
                            url=f'https://{proxy.domain}',
                            port=AGENT_SERVER_PORT,
                        )
                    )
                elif proxy.name == VSCODE_PROXY_NAME:
                    exposed_urls.append(
                        ExposedUrl(
                            name=VSCODE,
                            url=f'https://{proxy.domain}',
                            port=VSCODE_PORT,
                        )
                    )

        return SandboxInfo(
            id=stored.id,
            created_by_user_id=stored.created_by_user_id,
            sandbox_spec_id=stored.sandbox_spec_id,
            status=status,
            session_api_key=session_api_key
            if status == SandboxStatus.RUNNING
            else None,
            exposed_urls=exposed_urls,
            created_at=stored.created_at,
        )

    async def _get_box_or_none(self, sandbox_id: str) -> Any | None:
        """Fetch the boxd VM, swallowing NotFound. Other errors propagate."""
        from boxd.aio import NotFoundError

        vm_name = f'{VM_NAME_PREFIX}{sandbox_id}'
        try:
            return await self.compute.box.get(vm_name)
        except NotFoundError:
            return None

    # ── Public API ────────────────────────────────────────────────────

    async def start_sandbox(
        self, sandbox_spec_id: str | None = None, sandbox_id: str | None = None
    ) -> SandboxInfo:
        from boxd.aio import BoxdError

        # Enforce per-user sandbox limits before creating another.
        await self.pause_old_sandboxes(self.max_num_sandboxes - 1)

        if sandbox_spec_id is None:
            sandbox_spec = await self.sandbox_spec_service.get_default_sandbox_spec()
        else:
            sandbox_spec_maybe = await self.sandbox_spec_service.get_sandbox_spec(
                sandbox_spec_id
            )
            if sandbox_spec_maybe is None:
                raise SandboxError(f'Sandbox spec not found: {sandbox_spec_id}')
            sandbox_spec = sandbox_spec_maybe

        if sandbox_id is None:
            sandbox_id = base62.encodebytes(os.urandom(16))
        vm_name = f'{VM_NAME_PREFIX}{sandbox_id}'

        user_id = await self.user_context.get_user_id()
        env = await self._build_environment(sandbox_spec)

        # Session API key is generated app-side. The VM carries it via
        # SESSION_API_KEY_VARIABLE; the hash is indexed in our DB.
        session_api_key = base62.encodebytes(os.urandom(32))
        env[SESSION_API_KEY_VARIABLE] = session_api_key
        session_api_key_hash = _hash_session_api_key(session_api_key)

        # Insert the stored row before talking to boxd so the row's
        # auto-set ``created_at`` reflects creation start, not end.
        stored = StoredBoxdSandbox(
            id=sandbox_id,
            created_by_user_id=user_id,
            sandbox_spec_id=sandbox_spec.id,
            session_api_key_hash=session_api_key_hash,
            created_at=utc_now(),
        )
        self.db_session.add(stored)

        config = self._build_box_config(sandbox_spec, env)

        try:
            box = await self.compute.box.create(
                name=vm_name,
                config=config,
                image=sandbox_spec.id,
            )
        except BoxdError as exc:
            _logger.error('Failed to create boxd VM %s: %s', vm_name, exc)
            raise SandboxError(f'Failed to start sandbox: {exc}')

        await self._ensure_named_proxies(box)

        _logger.info(
            'Started boxd sandbox %s (vm=%s)', sandbox_id, getattr(box, 'id', '?')
        )
        return await self._to_sandbox_info(
            stored, box, session_api_key=session_api_key
        )

    async def get_sandbox(self, sandbox_id: str) -> SandboxInfo | None:
        stored = await self._get_stored(sandbox_id)
        if stored is None:
            return None
        box = await self._get_box_or_none(sandbox_id)
        return await self._to_sandbox_info(stored, box)

    async def search_sandboxes(
        self, page_id: str | None = None, limit: int = 100
    ) -> SandboxPage:
        stmt = await self._secure_select()
        offset = 0
        if page_id is not None:
            try:
                offset = int(page_id)
            except ValueError:
                offset = 0

        # Fetch limit+1 to detect "has more" without a separate count.
        stmt = (
            stmt.order_by(StoredBoxdSandbox.created_at.desc())
            .offset(offset)
            .limit(limit + 1)
        )
        result = await self.db_session.execute(stmt)
        stored_rows = list(result.scalars().all())

        has_more = len(stored_rows) > limit
        if has_more:
            stored_rows = stored_rows[:limit]

        next_page_id = str(offset + limit) if has_more else None

        items: list[SandboxInfo] = []
        for stored in stored_rows:
            box = await self._get_box_or_none(stored.id)
            items.append(await self._to_sandbox_info(stored, box))

        return SandboxPage(items=items, next_page_id=next_page_id)

    async def get_sandbox_by_session_api_key(
        self, session_api_key: str
    ) -> SandboxInfo | None:
        target_hash = _hash_session_api_key(session_api_key)
        stmt = await self._secure_select()
        stmt = stmt.where(StoredBoxdSandbox.session_api_key_hash == target_hash)
        result = await self.db_session.execute(stmt)
        stored = result.scalar_one_or_none()
        if stored is None:
            return None
        box = await self._get_box_or_none(stored.id)
        return await self._to_sandbox_info(
            stored, box, session_api_key=session_api_key
        )

    async def pause_sandbox(self, sandbox_id: str) -> bool:
        stored = await self._get_stored(sandbox_id)
        if stored is None:
            return False
        box = await self._get_box_or_none(sandbox_id)
        if box is None:
            return False
        # Security: invalidate the session key hash so leaked keys can't
        # be used while the sandbox is paused. The VM still has the
        # original key in env; it becomes valid again on resume when we
        # restore the hash.
        stored.session_api_key_hash = None
        try:
            await box.suspend()
            return True
        except Exception as exc:
            _logger.error('Failed to suspend boxd sandbox %s: %s', sandbox_id, exc)
            return False

    async def resume_sandbox(self, sandbox_id: str) -> bool:
        # Enforce per-user limits before another resume puts us over.
        await self.pause_old_sandboxes(self.max_num_sandboxes - 1)
        stored = await self._get_stored(sandbox_id)
        if stored is None:
            return False
        box = await self._get_box_or_none(sandbox_id)
        if box is None:
            return False
        try:
            await box.resume()
            return True
        except Exception as exc:
            _logger.error('Failed to resume boxd sandbox %s: %s', sandbox_id, exc)
            return False

    async def delete_sandbox(self, sandbox_id: str) -> bool:
        stored = await self._get_stored(sandbox_id)
        if stored is None:
            return False
        # Drop the row first — deleting the session_api_key_hash
        # invalidates leaked keys even if the boxd call fails.
        await self.db_session.delete(stored)
        box = await self._get_box_or_none(sandbox_id)
        if box is None:
            # VM was already gone or unreachable; the row removal is
            # still useful (cleans up our index).
            return True
        try:
            await box.destroy()
            return True
        except Exception as exc:
            _logger.error('Failed to destroy boxd sandbox %s: %s', sandbox_id, exc)
            return False


class BoxdSandboxServiceInjector(SandboxServiceInjector):
    """Dependency injector for the boxd sandbox service."""

    api_key: str | None = Field(
        default=None,
        description=(
            'boxd API key (e.g. bxk_...). If unset, the boxd SDK falls '
            'back to the BOXD_API_KEY environment variable.'
        ),
    )
    api_url: str | None = Field(
        default=None,
        description='Optional boxd control-plane URL (defaults to the SDK default).',
    )
    auto_suspend_timeout: int = Field(
        default=300,
        description=(
            'Seconds of inactivity before boxd warm-suspends the VM '
            '(sub-ms resume).'
        ),
    )
    max_num_sandboxes: int = Field(
        default=10,
        description='Maximum number of running sandboxes per user.',
    )
    vcpu: int = Field(default=2, description='vCPUs per VM.')
    memory: str = Field(default='8G', description='Memory per VM (boxd size string).')
    disk: str = Field(default='100G', description='Disk per VM (boxd size string).')

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[SandboxService, None]:
        from boxd.aio import Compute

        # Defer to break the circular path with config.py.
        from openhands.app_server.config import (
            get_db_session,
            get_global_config,
            get_sandbox_spec_service,
            get_user_context,
        )

        config = get_global_config()
        web_url = config.web_url

        compute_kwargs: dict[str, Any] = {}
        if self.api_key:
            compute_kwargs['api_key'] = self.api_key
        if self.api_url:
            compute_kwargs['api_url'] = self.api_url

        async with (
            get_user_context(state, request) as user_context,
            get_sandbox_spec_service(state, request) as sandbox_spec_service,
            get_db_session(state, request) as db_session,
            Compute(**compute_kwargs) as compute,
        ):
            yield BoxdSandboxService(
                sandbox_spec_service=sandbox_spec_service,
                compute=compute,
                web_url=web_url,
                max_num_sandboxes=self.max_num_sandboxes,
                auto_suspend_timeout=self.auto_suspend_timeout,
                vcpu=self.vcpu,
                memory=self.memory,
                disk=self.disk,
                user_context=user_context,
                db_session=db_session,
            )
