import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator
from urllib.parse import urlparse
from uuid import UUID

import base62
import httpx
from fastapi import Request
from pydantic import Field
from sqlalchemy import Boolean, String, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from openhands.agent_server.models import (
    ConversationInfo,
    EventPage,
)
from openhands.agent_server.utils import utc_now
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
)
from openhands.app_server.errors import ConcurrencyLimitError, SandboxError
from openhands.app_server.sandbox.sandbox_models import (
    AGENT_SERVER,
    VSCODE,
    WORKER_1,
    WORKER_2,
    ExposedUrl,
    SandboxInfo,
    SandboxPage,
    SandboxRecord,
    SandboxStatus,
)
from openhands.app_server.sandbox.sandbox_service import (
    ALLOW_CORS_ORIGINS_VARIABLE,
    WEBHOOK_CALLBACK_VARIABLE,
    SandboxService,
    SandboxServiceInjector,
)
from openhands.app_server.sandbox.sandbox_spec_models import SandboxSpecInfo
from openhands.app_server.sandbox.sandbox_spec_service import SandboxSpecService
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import ADMIN, USER_CONTEXT_ATTR
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.utils.sql_utils import Base, UtcDateTime
from openhands.sdk.utils.paging import page_iterator

_logger = logging.getLogger(__name__)
polling_task: asyncio.Task | None = None
STATUS_MAPPING = {
    'running': SandboxStatus.RUNNING,
    'paused': SandboxStatus.PAUSED,
    'stopped': SandboxStatus.MISSING,
    'starting': SandboxStatus.STARTING,
    'error': SandboxStatus.ERROR,
}
AGENT_SERVER_PORT = 60000
VSCODE_PORT = 60001
WORKER_1_PORT = 12000
WORKER_2_PORT = 12001


def _hash_session_api_key(session_api_key: str) -> str:
    """Hash a session API key using SHA-256."""
    return hashlib.sha256(session_api_key.encode()).hexdigest()


class StoredRemoteSandbox(Base):
    """Local storage for remote sandbox info.

    The remote runtime API does not return some variables we need, and does not
    return stopped runtimes in list operations, so we need a local copy. We use
    the remote api as a source of truth on what is currently running, not what was
    run historicallly."""

    __tablename__ = 'v1_remote_sandbox'

    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_by_user_id: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True
    )
    sandbox_spec_id: Mapped[str] = mapped_column(
        String, index=True
    )  # shadows runtime['image']
    session_api_key_hash: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime, server_default=func.now(), index=True
    )
    is_paused: Mapped[bool] = mapped_column(Boolean, default=False, server_default='0')


@dataclass
class RemoteSandboxService(SandboxService):
    """Sandbox service that uses HTTP to communicate with a remote runtime API.

    This service adapts the legacy RemoteRuntime HTTP protocol to work with
    the new Sandbox interface.
    """

    sandbox_spec_service: SandboxSpecService
    api_url: str
    api_key: str
    web_url: str | None
    resource_factor: int
    runtime_class: str | None
    start_sandbox_timeout: int
    max_num_sandboxes: int
    user_context: UserContext
    httpx_client: httpx.AsyncClient
    db_session: AsyncSession

    async def _send_runtime_api_request(
        self, method: str, path: str, **kwargs: Any
    ) -> httpx.Response:
        """Send a request to the remote runtime API."""
        try:
            url = self.api_url + path
            return await self.httpx_client.request(
                method, url, headers={'X-API-Key': self.api_key}, **kwargs
            )
        except httpx.TimeoutException:
            _logger.error(f'No response received within timeout for URL: {url}')
            raise
        except httpx.HTTPError as e:
            _logger.error(f'HTTP error for URL {url}: {e}')
            raise

    def _to_sandbox_info(
        self, stored: StoredRemoteSandbox, runtime: dict[str, Any] | None = None
    ):
        status = self._get_sandbox_status_from_runtime(runtime)

        # Get session_api_key and exposed urls
        if runtime:
            session_api_key = runtime['session_api_key']
            if status == SandboxStatus.RUNNING:
                exposed_urls = []
                url = runtime.get('url', None)
                if url:
                    runtime_id = runtime['runtime_id']
                    exposed_urls.append(
                        ExposedUrl(name=AGENT_SERVER, url=url, port=AGENT_SERVER_PORT)
                    )
                    vscode_url = (
                        _build_service_url(url, 'vscode', runtime_id)
                        + f'?tkn={session_api_key}&folder=%2Fworkspace%2Fproject'
                    )
                    exposed_urls.append(
                        ExposedUrl(name=VSCODE, url=vscode_url, port=VSCODE_PORT)
                    )
                    exposed_urls.append(
                        ExposedUrl(
                            name=WORKER_1,
                            url=_build_service_url(url, 'work-1', runtime_id),
                            port=WORKER_1_PORT,
                        )
                    )
                    exposed_urls.append(
                        ExposedUrl(
                            name=WORKER_2,
                            url=_build_service_url(url, 'work-2', runtime_id),
                            port=WORKER_2_PORT,
                        )
                    )
            else:
                exposed_urls = None
        else:
            session_api_key = None
            exposed_urls = None

        sandbox_spec_id = stored.sandbox_spec_id
        return SandboxInfo(
            id=stored.id,
            created_by_user_id=stored.created_by_user_id,
            sandbox_spec_id=sandbox_spec_id,
            status=status,
            session_api_key=session_api_key,
            exposed_urls=exposed_urls,
            created_at=stored.created_at,
        )

    def _get_sandbox_status_from_runtime(
        self, runtime: dict[str, Any] | None
    ) -> SandboxStatus:
        """Derive a SandboxStatus from the runtime info.

        The status field is now the source of truth for sandbox status. It accounts
        for both pod readiness and ingress availability, making it more reliable than
        pod_status which only reflected pod state.
        """
        if not runtime:
            return SandboxStatus.MISSING

        runtime_status = runtime.get('status')
        if runtime_status:
            status = STATUS_MAPPING.get(runtime_status.lower(), None)
            if status is not None:
                return status

        return SandboxStatus.MISSING

    async def _secure_select(self):
        query = select(StoredRemoteSandbox)
        user_id = await self.user_context.get_user_id()
        if user_id:
            query = query.where(StoredRemoteSandbox.created_by_user_id == user_id)
        return query

    async def _get_user_effective_sandbox_limit(self) -> int:
        """Get the effective sandbox limit for the current user.

        Delegates to UserContext.get_max_concurrent_sandboxes() which handles:
        1. OrgMember.max_concurrent_sandboxes_override (if not NULL) - enterprise only
        2. Org.max_concurrent_sandboxes (org default) - enterprise only
        3. Global fallback (self.max_num_sandboxes) - OSS mode

        Returns:
            int: The effective maximum number of concurrent sandboxes
        """
        return await self.user_context.get_max_concurrent_sandboxes(
            self.max_num_sandboxes
        )

    async def _count_user_running_sandboxes(self) -> int:
        """Count the number of running (non-paused) sandboxes for the current user.

        Queries the DB directly by created_by_user_id and is_paused, avoiding
        a global runtime API call for every concurrency check.

        Returns:
            int: Number of running sandboxes
        """
        query = await self._secure_select()
        query = query.filter(StoredRemoteSandbox.is_paused == False)  # noqa: E712
        result = await self.db_session.execute(query)
        return len(result.scalars().all())

    async def _get_stored_sandbox(self, sandbox_id: str) -> StoredRemoteSandbox | None:
        stmt = await self._secure_select()
        stmt = stmt.where(StoredRemoteSandbox.id == sandbox_id)
        result = await self.db_session.execute(stmt)
        stored_sandbox = result.scalar_one_or_none()
        return stored_sandbox

    async def _get_runtime(self, sandbox_id: str) -> dict[str, Any]:
        response = await self._send_runtime_api_request(
            'GET',
            f'/sessions/{sandbox_id}',
        )
        response.raise_for_status()
        runtime_data = response.json()
        return runtime_data

    async def _get_runtimes_batch(
        self, sandbox_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Get multiple runtimes in a single batch request.

        Args:
            sandbox_ids: List of sandbox IDs to fetch

        Returns:
            Dictionary mapping sandbox_id to runtime data
        """
        if not sandbox_ids:
            return {}

        # Build query parameters for the batch endpoint
        params = [('ids', sandbox_id) for sandbox_id in sandbox_ids]

        response = await self._send_runtime_api_request(
            'GET',
            '/sessions/batch',
            params=params,
        )
        response.raise_for_status()
        batch_data = response.json()

        # The batch endpoint should return a list of runtimes
        # Convert to a dictionary keyed by session_id for easy lookup
        runtimes_by_id = {}
        for runtime in batch_data:
            if runtime and 'session_id' in runtime:
                runtimes_by_id[runtime['session_id']] = runtime

        return runtimes_by_id

    async def _init_environment(
        self, sandbox_spec: SandboxSpecInfo, sandbox_id: str
    ) -> dict[str, str]:
        """Initialize the environment variables for the sandbox."""
        environment = sandbox_spec.initial_env.copy()

        # If a public facing url is defined, add a callback to the agent server environment.
        if self.web_url:
            environment[WEBHOOK_CALLBACK_VARIABLE] = f'{self.web_url}/api/v1/webhooks'
            # We specify CORS settings only if there is a public facing url - otherwise
            # we are probably in local development and the only url in use is localhost
            environment[ALLOW_CORS_ORIGINS_VARIABLE] = self.web_url

        # Add worker port environment variables so the agent knows which ports to use
        # for web applications. These match the ports exposed via the WORKER_1 and
        # WORKER_2 URLs.
        environment[WORKER_1] = str(WORKER_1_PORT)
        environment[WORKER_2] = str(WORKER_2_PORT)

        return environment

    async def search_sandboxes(
        self,
        page_id: str | None = None,
        limit: int = 100,
    ) -> SandboxPage:
        stmt = await self._secure_select()

        # Handle pagination
        if page_id is not None:
            # Parse page_id to get offset or cursor
            try:
                offset = int(page_id)
                stmt = stmt.offset(offset)
            except ValueError:
                # If page_id is not a valid integer, start from beginning
                offset = 0
        else:
            offset = 0

        # Apply limit and get one extra to check if there are more results
        stmt = stmt.limit(limit + 1).order_by(StoredRemoteSandbox.created_at.desc())

        result = await self.db_session.execute(stmt)
        stored_sandboxes = result.scalars().all()

        # Check if there are more results
        has_more = len(stored_sandboxes) > limit
        if has_more:
            stored_sandboxes = stored_sandboxes[:limit]

        # Calculate next page ID
        next_page_id = None
        if has_more:
            next_page_id = str(offset + limit)

        # Batch fetch runtime data for all sandboxes
        sandbox_ids = [stored_sandbox.id for stored_sandbox in stored_sandboxes]
        runtimes_by_id = await self._get_runtimes_batch(sandbox_ids)

        # Convert stored sandboxes to domain models with runtime data
        items = [
            self._to_sandbox_info(stored_sandbox, runtimes_by_id.get(stored_sandbox.id))
            for stored_sandbox in stored_sandboxes
        ]

        return SandboxPage(items=items, next_page_id=next_page_id)

    async def get_sandbox(self, sandbox_id: str) -> SandboxInfo | None:
        """Get a single sandbox by checking its corresponding runtime."""
        stored_sandbox = await self._get_stored_sandbox(sandbox_id)
        if stored_sandbox is None:
            return None

        runtime = None
        try:
            runtime = await self._get_runtime(stored_sandbox.id)
        except Exception:
            _logger.exception(
                f'Error getting runtime: {stored_sandbox.id}', stack_info=True
            )

        return self._to_sandbox_info(stored_sandbox, runtime)

    async def get_sandbox_by_session_api_key(
        self, session_api_key: str
    ) -> SandboxInfo | None:
        """Get a single sandbox by session API key using the stored hash."""
        session_api_key_hash = _hash_session_api_key(session_api_key)

        stmt = await self._secure_select()
        stmt = stmt.where(
            StoredRemoteSandbox.session_api_key_hash == session_api_key_hash
        )
        result = await self.db_session.execute(stmt)
        stored_sandbox = result.scalar_one_or_none()

        if stored_sandbox is None:
            return None

        try:
            runtime = await self._get_runtime(stored_sandbox.id)
            return self._to_sandbox_info(stored_sandbox, runtime)
        except Exception:
            _logger.exception(
                f'Error getting runtime for sandbox {stored_sandbox.id}',
                stack_info=True,
            )
            return self._to_sandbox_info(stored_sandbox, None)

    async def check_concurrency_limit(self) -> None:
        """Check if the user has reached their concurrent sandbox limit.

        This check is performed synchronously before creating a task to allow
        the API to return a 429 error immediately instead of failing asynchronously.

        Raises:
            ConcurrencyLimitError: If the user has reached their limit
        """
        effective_limit = await self._get_user_effective_sandbox_limit()
        current_count = await self._count_user_running_sandboxes()

        if current_count >= effective_limit:
            _logger.info(
                f'User has reached sandbox limit: {current_count}/{effective_limit}'
            )
            raise ConcurrencyLimitError(
                detail={
                    'error': 'CONCURRENCY_LIMIT_REACHED',
                    'message': (
                        f'You have reached your limit of {effective_limit} '
                        'concurrent conversations. Please close an existing '
                        'conversation to start a new one.'
                    ),
                    'limit': effective_limit,
                    'current': current_count,
                }
            )

    async def get_sandbox_record_by_session_api_key(
        self, session_api_key: str
    ) -> SandboxRecord | None:
        """Get persisted sandbox identity by session API key — DB lookup only, no runtime call."""
        session_api_key_hash = _hash_session_api_key(session_api_key)

        stmt = await self._secure_select()
        stmt = stmt.where(
            StoredRemoteSandbox.session_api_key_hash == session_api_key_hash
        )
        result = await self.db_session.execute(stmt)
        stored_sandbox = result.scalar_one_or_none()

        if stored_sandbox is None:
            return None

        return SandboxRecord(
            id=stored_sandbox.id,
            created_by_user_id=stored_sandbox.created_by_user_id,
        )

    async def start_sandbox(
        self, sandbox_spec_id: str | None = None, sandbox_id: str | None = None
    ) -> SandboxInfo:
        """Start a new sandbox by creating a remote runtime.

        Uses SELECT FOR UPDATE within a nested transaction to acquire a row-level
        lock, preventing race conditions (TOCTOU) when checking concurrency limits.
        The lock is released after the sandbox record is inserted (before the
        long-running runtime startup) to avoid blocking concurrent requests.
        """
        try:
            # Get sandbox spec early (before locking) to minimize lock hold time
            if sandbox_spec_id is None:
                sandbox_spec = (
                    await self.sandbox_spec_service.get_default_sandbox_spec()
                )
            else:
                sandbox_spec_maybe = await self.sandbox_spec_service.get_sandbox_spec(
                    sandbox_spec_id
                )
                if sandbox_spec_maybe is None:
                    raise ValueError('Sandbox Spec not found')
                sandbox_spec = sandbox_spec_maybe

            # Create a unique id, use provided sandbox_id if available
            if sandbox_id is None:
                sandbox_id = base62.encodebytes(os.urandom(16))

            # Get user id for locking and record creation
            user_id = await self.user_context.get_user_id()

            # Use a nested transaction (savepoint) for the lock + check + insert.
            # This allows us to release the row-level lock after inserting the
            # sandbox record, before the long-running runtime startup begins.
            # Without this, the lock would be held for the entire request duration
            # (up to 120s for sandbox startup), blocking concurrent requests.
            async with self.db_session.begin_nested():
                # Acquire row-level lock to prevent TOCTOU race condition.
                # This serializes concurrent sandbox creation requests for the same user.
                # We lock the user's most recent sandbox row (if any) to coordinate access.
                # If no sandbox exists, the INSERT below will still serialize due to
                # unique constraint checks, and the next request will have a row to lock.
                await self.db_session.execute(
                    select(StoredRemoteSandbox.id)
                    .filter(StoredRemoteSandbox.created_by_user_id == user_id)
                    .order_by(StoredRemoteSandbox.created_at.desc())
                    .limit(1)
                    .with_for_update()
                )

                # Get user's effective limit and current count (now serialized)
                effective_limit = await self._get_user_effective_sandbox_limit()
                current_count = await self._count_user_running_sandboxes()

                # Check if user has reached their limit
                if current_count >= effective_limit:
                    _logger.info(
                        f'User has reached sandbox limit: {current_count}/{effective_limit}'
                    )
                    raise ConcurrencyLimitError(
                        detail={
                            'error': 'CONCURRENCY_LIMIT_REACHED',
                            'message': (
                                f'You have reached your limit of {effective_limit} '
                                'concurrent conversations. Please close an existing '
                                'conversation to start a new one.'
                            ),
                            'limit': effective_limit,
                            'current': current_count,
                        }
                    )

                # Enforce sandbox limits by cleaning up old sandboxes if approaching limit
                # Use effective_limit - 1 to leave room for the new sandbox
                if current_count >= effective_limit - 1:
                    await self.pause_old_sandboxes(effective_limit - 1)

                # Store the sandbox record (this reserves the slot)
                stored_sandbox = StoredRemoteSandbox(
                    id=sandbox_id,
                    created_by_user_id=user_id,
                    sandbox_spec_id=sandbox_spec.id,
                    created_at=utc_now(),
                )
                self.db_session.add(stored_sandbox)

            # Nested transaction committed - lock is now released.
            # The sandbox record exists, so the concurrency slot is reserved.
            # Other requests can now proceed with their own concurrency checks.

            # Prepare environment variables
            environment = await self._init_environment(sandbox_spec, sandbox_id)

            # Prepare start request
            start_request: dict[str, Any] = {
                'image': sandbox_spec.id,  # Use sandbox_spec.id as the container image
                'command': sandbox_spec.command,
                'working_dir': '/workspace',
                'environment': environment,
                'session_id': sandbox_id,  # Use sandbox_id as session_id
                'resource_factor': self.resource_factor,
                'run_as_user': 10001,
                'run_as_group': 10001,
                'fs_group': 10001,
            }

            # Add runtime class if specified
            if self.runtime_class == 'sysbox':
                start_request['runtime_class'] = 'sysbox-runc'

            # Start the runtime
            response = await self._send_runtime_api_request(
                'POST',
                '/start',
                json=start_request,
            )
            response.raise_for_status()
            runtime_data = response.json()

            # Store the session_api_key hash for efficient lookups
            session_api_key = runtime_data.get('session_api_key')
            if session_api_key:
                stored_sandbox.session_api_key_hash = _hash_session_api_key(
                    session_api_key
                )

            # Log runtime assignment for observability
            runtime_id = runtime_data.get('runtime_id', 'unknown')
            _logger.info(f'Started sandbox {sandbox_id} with runtime_id={runtime_id}')

            return self._to_sandbox_info(stored_sandbox, runtime_data)

        except ConcurrencyLimitError:
            # Re-raise concurrency limit errors without wrapping
            raise
        except httpx.HTTPError as e:
            _logger.error(f'Failed to start sandbox: {e}')
            raise SandboxError(f'Failed to start sandbox: {e}')

    async def resume_sandbox(self, sandbox_id: str) -> bool:
        """Resume a paused sandbox.

        Security: When a sandbox is resumed, the runtime-api generates a new
        session_api_key and returns it. This invalidates any previously leaked
        keys and ensures that only the new key can be used to access secrets.
        """
        # Enforce sandbox limits by cleaning up old sandboxes
        await self.pause_old_sandboxes(self.max_num_sandboxes - 1)

        try:
            stored_sandbox = await self._get_stored_sandbox(sandbox_id)
            if not stored_sandbox:
                return False
            runtime_data = await self._get_runtime(sandbox_id)
            response = await self._send_runtime_api_request(
                'POST',
                '/resume',
                json={'runtime_id': runtime_data['runtime_id']},
            )
            if response.status_code == 404:
                return False
            response.raise_for_status()

            # Security: Update stored session_api_key with the new key returned
            # by the runtime-api. The old key was invalidated on resume.
            response_data = response.json()
            new_session_api_key = response_data.get('session_api_key')
            if new_session_api_key:
                stored_sandbox.session_api_key_hash = _hash_session_api_key(
                    new_session_api_key
                )
                _logger.info(
                    f'Updated session_api_key_hash for sandbox {sandbox_id} after resume'
                )

            stored_sandbox.is_paused = False
            return True
        except httpx.HTTPError as e:
            _logger.error(f'Error resuming sandbox {sandbox_id}: {e}')
            return False

    async def pause_sandbox(self, sandbox_id: str) -> bool:
        """Pause a running sandbox.

        Security: Clears the session_api_key_hash to invalidate any existing
        session keys, preventing leaked keys from being used while paused.
        """
        try:
            stored_sandbox = await self._get_stored_sandbox(sandbox_id)
            if not stored_sandbox:
                return False

            # Security: Invalidate the session API key hash to prevent
            # leaked keys from being used while the sandbox is paused.
            stored_sandbox.session_api_key_hash = None
            stored_sandbox.is_paused = True

            runtime_data = await self._get_runtime(sandbox_id)
            response = await self._send_runtime_api_request(
                'POST',
                '/pause',
                json={'runtime_id': runtime_data['runtime_id']},
            )
            if response.status_code == 404:
                return False
            response.raise_for_status()
            return True

        except httpx.HTTPError as e:
            _logger.error(f'Error pausing sandbox {sandbox_id}: {e}')
            return False

    async def delete_sandbox(self, sandbox_id: str) -> bool:
        """Delete a sandbox by stopping its runtime.

        Security: Deleting the stored_sandbox record also removes the
        session_api_key_hash, invalidating any leaked session keys.
        """
        try:
            stored_sandbox = await self._get_stored_sandbox(sandbox_id)
            if not stored_sandbox:
                return False
            # Deleting the record also removes the session_api_key_hash,
            # which invalidates any leaked session keys.
            await self.db_session.delete(stored_sandbox)
            runtime_data = await self._get_runtime(sandbox_id)
            response = await self._send_runtime_api_request(
                'POST',
                '/stop',
                json={'runtime_id': runtime_data['runtime_id']},
            )
            if response.status_code != 404:
                response.raise_for_status()
            return True
        except httpx.HTTPError as e:
            _logger.error(f'Error deleting sandbox {sandbox_id}: {e}')
            return False

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

        # Query running sandboxes from DB directly using is_paused flag, avoiding
        # a global runtime API call that returns all users' runtimes.
        query = await self._secure_select()
        query = query.filter(StoredRemoteSandbox.is_paused == False).order_by(  # noqa: E712
            StoredRemoteSandbox.created_at.desc()
        )
        running_sandboxes = list(await self.db_session.execute(query))

        # If we're within the limit, no cleanup needed
        if len(running_sandboxes) <= max_num_sandboxes:
            return []

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

    async def batch_get_sandboxes(
        self, sandbox_ids: list[str]
    ) -> list[SandboxInfo | None]:
        """Get a batch of sandboxes, returning None for any which were not found."""
        if not sandbox_ids:
            return []
        query = await self._secure_select()
        query = query.filter(StoredRemoteSandbox.id.in_(sandbox_ids))
        stored_remote_sandboxes = await self.db_session.execute(query)
        stored_remote_sandboxes_by_id = {
            stored_remote_sandbox[0].id: stored_remote_sandbox[0]
            for stored_remote_sandbox in stored_remote_sandboxes
        }
        runtimes_by_id = await self._get_runtimes_batch(
            list(stored_remote_sandboxes_by_id)
        )
        results = []
        for sandbox_id in sandbox_ids:
            stored_remote_sandbox = stored_remote_sandboxes_by_id.get(sandbox_id)
            result = None
            if stored_remote_sandbox:
                runtime = runtimes_by_id.get(sandbox_id)
                result = self._to_sandbox_info(stored_remote_sandbox, runtime)
            results.append(result)
        return results


def _build_service_url(url: str, service_name: str, runtime_id: str) -> str:
    """Build a service URL for the given service name.

    Handles both path-based and subdomain-based routing:
    - Path mode (url path starts with /{runtime_id}): returns {scheme}://{netloc}/{runtime_id}/{service_name}
    - Subdomain mode: returns {scheme}://{service_name}-{netloc}{path}
    """
    parsed = urlparse(url)
    scheme, netloc, path = parsed.scheme, parsed.netloc, parsed.path or '/'
    # Path mode if runtime_url path starts with /{id}
    path_mode = path.startswith(f'/{runtime_id}')
    if path_mode:
        return f'{scheme}://{netloc}/{runtime_id}/{service_name}'
    else:
        return f'{scheme}://{service_name}-{netloc}{path}'


async def poll_agent_servers(api_url: str, api_key: str, sleep_interval: int):
    """When the app server does not have a public facing url, we poll the agent
    servers for the most recent data.

    This is because webhook callbacks cannot be invoked.

    IMPORTANT: DB sessions are scoped tightly to avoid holding connections across
    network I/O. Services are imported locally inside the function bodies to
    ensure they are resolved in the correct context. We use a
    "fetch -> release -> network -> re-acquire -> write" pattern.
    """
    from openhands.app_server.config import (
        get_app_conversation_info_service,
        get_db_session,
        get_httpx_client,
    )

    while True:
        try:
            state = InjectorState()
            # We allow access to all items here
            setattr(state, USER_CONTEXT_ATTR, ADMIN)

            try:
                # Get the list of running sandboxes using the runtime api /list endpoint.
                # (This will not return runtimes that have been stopped for a while)
                async with get_httpx_client(state) as httpx_client:
                    response = await httpx_client.get(
                        f'{api_url}/list', headers={'X-API-Key': api_key}
                    )
                    response.raise_for_status()
                    runtimes = response.json()['runtimes']
                    runtimes_by_sandbox_id = {
                        runtime['session_id']: runtime
                        for runtime in runtimes
                        # The runtime API currently reports a running status when
                        # pods are still starting. Resync can tolerate this.
                        if runtime['status'] == 'running'
                    }

                # Phase 1: Read - fetch all conversations into a list with a short DB session
                # This releases the DB session before any network I/O
                conversations_to_refresh: list[AppConversationInfo] = []
                async with (
                    get_app_conversation_info_service(
                        state
                    ) as app_conversation_info_service,
                    get_db_session(state) as _db_session,
                ):
                    async for app_conversation_info in page_iterator(
                        app_conversation_info_service.search_app_conversation_info
                    ):
                        conversations_to_refresh.append(app_conversation_info)

                _logger.debug(
                    f'Found {len(conversations_to_refresh)} conversations to check'
                )

                # Phase 2: Network I/O - fetch httpx client and do all network operations
                # WITHOUT any DB session held
                async with get_httpx_client(state) as httpx_client:
                    matches = 0
                    for app_conversation_info in conversations_to_refresh:
                        runtime = runtimes_by_sandbox_id.get(
                            app_conversation_info.sandbox_id
                        )
                        if runtime:
                            matches += 1
                            await refresh_conversation(
                                app_conversation_info=app_conversation_info,
                                runtime=runtime,
                                httpx_client=httpx_client,
                            )
                    _logger.debug(
                        f'Matched {len(runtimes_by_sandbox_id)} Runtimes with {matches} Conversations.'
                    )

            except Exception as exc:
                _logger.exception(
                    f'Error when polling agent servers: {exc}', stack_info=True
                )

            # Sleep between retries
            await asyncio.sleep(sleep_interval)

        except asyncio.CancelledError:
            return


async def refresh_conversation(
    app_conversation_info: AppConversationInfo,
    runtime: dict[str, Any],
    httpx_client: httpx.AsyncClient,
):
    """Refresh a conversation.

    Grab ConversationInfo and all events from the agent server and make sure they
    exist in the app server.

    IMPORTANT: This function acquires its own short-lived DB sessions for writes,
    never holding a session across network I/O. Uses a "fetch -> release -> write"
    pattern per conversation.
    """
    from openhands.app_server.config import (
        get_app_conversation_info_service,
        get_db_session,
        get_event_callback_service,
        get_event_service,
    )

    state = InjectorState()
    setattr(state, USER_CONTEXT_ATTR, ADMIN)

    _logger.debug(f'Started Refreshing Conversation {app_conversation_info.id}')
    try:
        url = runtime['url']

        # TODO: Maybe we can use RemoteConversation here?

        # Phase 1: Network I/O - First get conversation...
        conversation_url = f'{url}/api/conversations/{app_conversation_info.id.hex}'
        response = await httpx_client.get(
            conversation_url, headers={'X-Session-API-Key': runtime['session_api_key']}
        )
        response.raise_for_status()

        updated_conversation_info = ConversationInfo.model_validate(response.json())

        app_conversation_info.updated_at = updated_conversation_info.updated_at

        # TODO: This is a temp fix - the agent server is storing metrics in a new format
        # We should probably update the data structures and to store / display the more
        # explicit metrics
        try:
            app_conversation_info.metrics = (
                updated_conversation_info.stats.get_combined_metrics()
            )
        except Exception:
            _logger.exception('error_updating_conversation_metrics', stack_info=True)

        # Phase 2: Write - acquire DB session and save conversation info
        # (short-lived session, no network I/O held)
        async with (
            get_db_session(state) as _db_session,
            get_app_conversation_info_service(state) as app_conversation_info_service,
        ):
            await app_conversation_info_service.save_app_conversation_info(
                app_conversation_info
            )

        # Phase 3: Network I/O - fetch events (no DB session held)
        # TODO: It would be nice to have an updated_at__gte filter parameter in the
        # agent server so that we don't pull the full event list each time
        event_url = (
            f'{url}/api/conversations/{app_conversation_info.id.hex}/events/search'
        )

        async def fetch_events_page(page_id: str | None = None) -> EventPage:
            """Helper function to fetch a page of events from the agent server."""
            params: dict[str, str] = {}
            if page_id:
                params['page_id'] = page_id
            response = await httpx_client.get(
                event_url,
                params=params,
                headers={'X-Session-API-Key': runtime['session_api_key']},
            )
            response.raise_for_status()
            return EventPage.model_validate(response.json())

        async for event in page_iterator(fetch_events_page):
            # Phase 4: Write - acquire DB session for each event save
            # (short-lived session per event, no network I/O held)
            async with (
                get_db_session(state) as _db_session,
                get_event_service(state) as event_service,
                get_event_callback_service(state) as event_callback_service,
            ):
                existing = await event_service.get_event(
                    app_conversation_info.id, UUID(event.id)
                )
                if existing is None:
                    await event_service.save_event(app_conversation_info.id, event)
                    await event_callback_service.execute_callbacks(
                        app_conversation_info.id, event
                    )

        _logger.debug(f'Finished Refreshing Conversation {app_conversation_info.id}')

    except Exception as exc:
        _logger.exception(f'Error Refreshing Conversation: {exc}', stack_info=True)


class RemoteSandboxServiceInjector(SandboxServiceInjector):
    """Dependency injector for remote sandbox services."""

    api_url: str = Field(description='The API URL for remote runtimes')
    api_key: str = Field(description='The API Key for remote runtimes')
    polling_interval: int = Field(
        default=15,
        description=(
            'The sleep time between poll operations against agent servers when there is '
            'no public facing web_url'
        ),
    )
    resource_factor: int = Field(
        default=1,
        description='Factor by which to scale resources in sandbox: 1, 2, 4, or 8',
    )
    runtime_class: str = Field(
        default='gvisor',
        description='can be "gvisor" or "sysbox" (support docker inside runtime + more stable)',
    )
    start_sandbox_timeout: int = Field(
        default=120,
        description=(
            'The max time to wait for a sandbox to start before considering it to '
            'be in an error state.'
        ),
    )
    max_num_sandboxes: int = Field(
        default=10,
        description='Maximum number of sandboxes allowed to run simultaneously',
    )

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[SandboxService, None]:
        # Define inline to prevent circular lookup
        from openhands.app_server.config import (
            get_db_session,
            get_global_config,
            get_httpx_client,
            get_sandbox_spec_service,
            get_user_context,
        )

        # If no public facing web url is defined, poll for changes as callbacks will be unavailable.
        # This is primarily used for local development rather than production
        config = get_global_config()
        web_url = config.web_url
        if web_url is None or 'localhost' in web_url:
            global polling_task
            if polling_task is None:
                polling_task = asyncio.create_task(
                    poll_agent_servers(
                        api_url=self.api_url,
                        api_key=self.api_key,
                        sleep_interval=self.polling_interval,
                    )
                )
        async with (
            get_user_context(state, request) as user_context,
            get_sandbox_spec_service(state, request) as sandbox_spec_service,
            get_httpx_client(state, request) as httpx_client,
            get_db_session(state, request) as db_session,
        ):
            yield RemoteSandboxService(
                sandbox_spec_service=sandbox_spec_service,
                api_url=self.api_url,
                api_key=self.api_key,
                web_url=web_url,
                resource_factor=self.resource_factor,
                runtime_class=self.runtime_class,
                start_sandbox_timeout=self.start_sandbox_timeout,
                max_num_sandboxes=self.max_num_sandboxes,
                user_context=user_context,
                httpx_client=httpx_client,
                db_session=db_session,
            )
