"""Archive a remote sandbox's workspace to object storage before deletion.

This is the app-server side of the archive-before-delete flow (APP-2403). It
pulls a workspace archive from the in-pod agent-server endpoint
(``GET /api/file/archive``, added in software-agent-sdk AGE-1871) and stores it,
plus a small manifest, in object storage so the agent's work survives sandbox
deletion and can feed eval/dataset creation.

It covers the *explicit-delete-while-running* path. The dominant idle/expiry
reap is handled separately in runtime-api at pause time (PLTF-3112), because
that deletion never reaches the app-server.

Configuration is environment-driven (wired by APP-2405) and the feature is a
no-op unless ``RUNTIME_FILE_ARCHIVE_ENABLED`` is set.
"""

import asyncio
import json
import logging
import os
from typing import Any

from openhands.agent_server.utils import utc_now
from openhands.app_server.file_store.files import FileStore
from openhands.app_server.file_store.google_cloud import GoogleCloudFileStore

_logger = logging.getLogger(__name__)

_ARCHIVE_SUFFIX = {'git-delta': 'patch', 'tar.gz': 'tar.gz', 'zip': 'zip'}


def archive_enabled() -> bool:
    return os.getenv('RUNTIME_FILE_ARCHIVE_ENABLED', 'false').lower() in ('true', '1')


def archive_required() -> bool:
    """When true, an archive failure blocks deletion so it can be retried."""
    return os.getenv('RUNTIME_FILE_ARCHIVE_REQUIRED', 'false').lower() in ('true', '1')


def _archive_bucket() -> str:
    return os.getenv('RUNTIME_FILE_ARCHIVE_BUCKET', '')


def _archive_prefix() -> str:
    return os.getenv('RUNTIME_FILE_ARCHIVE_PREFIX', 'workspace-archives')


def _archive_format() -> str:
    return os.getenv('RUNTIME_FILE_ARCHIVE_FORMAT', 'git-delta')


def _archive_path() -> str:
    # The git repo lives at /workspace/project; /workspace itself is not a repo.
    return os.getenv('RUNTIME_FILE_ARCHIVE_PATH', '/workspace/project')


def _archive_timeout() -> float:
    return float(os.getenv('RUNTIME_FILE_ARCHIVE_TIMEOUT', '120'))


def _get_archive_file_store() -> FileStore:
    """Object store for archives. Currently Google Cloud Storage."""
    return GoogleCloudFileStore(bucket_name=_archive_bucket())


async def archive_workspace(
    httpx_client: Any,
    runtime: dict[str, Any],
    sandbox_id: str,
    conversation_id: str | None = None,
) -> bool:
    """Archive the sandbox's workspace; return whether deletion may proceed.

    Returns True when the workspace was archived, or when archiving failed but
    is not required (best-effort). Returns False only when archiving is
    required and could not be completed, so the caller can leave the sandbox
    intact and retry the delete later. Never raises.
    """
    agent_server_url = runtime.get('url')
    session_api_key = runtime.get('session_api_key')
    if not agent_server_url:
        _logger.warning(
            'Workspace archive skipped for %s: runtime has no agent-server URL',
            sandbox_id,
        )
        return not archive_required()
    if not _archive_bucket():
        _logger.warning(
            'Workspace archive enabled for %s but RUNTIME_FILE_ARCHIVE_BUCKET '
            'is not set; skipping',
            sandbox_id,
        )
        return not archive_required()

    fmt = _archive_format()
    suffix = _ARCHIVE_SUFFIX.get(fmt, 'patch')
    headers = {'X-Session-API-Key': session_api_key} if session_api_key else {}

    try:
        response = await httpx_client.get(
            f'{agent_server_url}/api/file/archive',
            params={'path': _archive_path(), 'format': fmt},
            headers=headers,
            timeout=_archive_timeout(),
        )
        if response.status_code != 200:
            _logger.warning(
                'Workspace archive for %s: agent-server returned %s; skipping',
                sandbox_id,
                response.status_code,
            )
            return not archive_required()
        data = response.content
        base_commit = response.headers.get('X-Archive-Base-Commit', '')
    except Exception as e:
        _logger.warning('Workspace archive fetch failed for %s: %s', sandbox_id, e)
        return not archive_required()

    try:
        store = _get_archive_file_store()
        ts = utc_now().strftime('%Y%m%dT%H%M%SZ')
        base_path = f'{_archive_prefix()}/{sandbox_id}/{ts}'
        await asyncio.to_thread(store.write, f'{base_path}.{suffix}', data)
        manifest = json.dumps(
            {
                'sandbox_id': sandbox_id,
                # For cloud conversations the sandbox id is the
                # conversation_id.hex; use it as the archive's conversation key.
                'conversation_id': conversation_id or sandbox_id,
                'base_commit': base_commit,
                'format': fmt,
                'source_path': _archive_path(),
                'byte_count': len(data),
                'created_at': ts,
            },
            sort_keys=True,
        ).encode('utf-8')
        await asyncio.to_thread(store.write, f'{base_path}.manifest.json', manifest)
        _logger.info(
            'Archived workspace for %s (%d bytes) to %s.%s',
            sandbox_id,
            len(data),
            base_path,
            suffix,
        )
        return True
    except Exception as e:
        _logger.exception('Workspace archive upload failed for %s: %s', sandbox_id, e)
        return not archive_required()
