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
from openhands.app_server.settings.settings_models import (
    SandboxGroupingStrategy,
    grouped_workspace_dir,
)

_logger = logging.getLogger(__name__)

# Only the formats the SDK GET /api/file/archive producer accepts
# (ArchiveFormat = git-delta | tar.gz). Anything else (e.g. the long-removed
# 'zip') is rejected by the producer with a 422, so it must not be advertised
# here — see _supported_format() which validates before issuing the request.
_ARCHIVE_SUFFIX = {'git-delta': 'patch', 'tar.gz': 'tar.gz'}


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


def _archive_base_path() -> str:
    # The workspace base. A repo-backed conversation clones the repo into a
    # {base}/{repo_name} subdirectory, so this base is usually not itself a git
    # repo — the agent-server archive endpoint auto-descends to the actual repo
    # beneath the given path (software-agent-sdk _resolve_git_repo_root), so we
    # only need to point it at the (possibly grouped) workspace base here.
    return os.getenv('RUNTIME_FILE_ARCHIVE_PATH', '/workspace/project')


def _archive_path(
    grouping_strategy: SandboxGroupingStrategy,
    conversation_id_hex: str,
) -> str:
    """Path of the workspace to archive (the agent-server resolves the repo).

    Mirrors the relocation applied at conversation start in
    live_status_app_conversation_service (working_dir → grouped_workspace_dir):
    NO_GROUPING keeps the bare base dir, any grouping nests it under the
    conversation id. The agent-server then descends from this base to the
    cloned repo ({base}/[{hex}/]{repo_name}); hardcoding the base without the
    grouping nesting would point at the wrong conversation's directory.
    """
    return grouped_workspace_dir(
        _archive_base_path(), grouping_strategy, conversation_id_hex
    )


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
    grouping_strategy: SandboxGroupingStrategy = SandboxGroupingStrategy.NO_GROUPING,
) -> bool:
    """Archive the sandbox's workspace; return whether deletion may proceed.

    Returns True when the workspace was archived, when there is nothing to
    archive (the agent-server reports the path is missing or not a git repo —
    a 400/404), or when archiving failed but is not required (best-effort).
    Returns False only when archiving is required and hit a genuinely transient
    failure (5xx / network / a non-"nothing-to-capture" 4xx such as 401/422),
    so the caller can leave the sandbox intact for a later retry (an explicit
    re-delete, or the runtime-api primary path at the eventual idle reap).
    Never raises.

    A pure configuration error (unsupported RUNTIME_FILE_ARCHIVE_FORMAT, or
    RUNTIME_FILE_ARCHIVE_BUCKET unset) cannot be fixed by retrying, so it is
    logged loudly and the delete is allowed to proceed rather than wedging every
    delete forever when archiving is required.
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
        # Misconfiguration, not a transient failure: no amount of retrying makes
        # a missing bucket appear. Proceed (with a loud error) so a
        # REQUIRED-without-bucket setup does not block every sandbox delete.
        _logger.error(
            'Workspace archive enabled for %s but RUNTIME_FILE_ARCHIVE_BUCKET '
            'is not set; proceeding with delete (fix the config to capture)',
            sandbox_id,
        )
        return True

    fmt = _archive_format()
    if fmt not in _ARCHIVE_SUFFIX:
        # The SDK producer would 422 this; treat it as a hard config error
        # (mirrors runtime-api._validate_archive_format) rather than letting the
        # 4xx be misread as "nothing to archive" and silently deleting.
        _logger.error(
            'Workspace archive for %s: unsupported RUNTIME_FILE_ARCHIVE_FORMAT '
            '%r (valid: %s); skipping archive',
            sandbox_id,
            fmt,
            sorted(_ARCHIVE_SUFFIX),
        )
        return not archive_required()
    suffix = _ARCHIVE_SUFFIX[fmt]
    headers = {'X-Session-API-Key': session_api_key} if session_api_key else {}
    # Conversation id keys the archive and locates the (possibly grouped) repo.
    # For cloud conversations the sandbox id is the conversation_id.hex.
    conversation_key = conversation_id or sandbox_id
    archive_path = _archive_path(grouping_strategy, conversation_key)

    try:
        response = await httpx_client.get(
            f'{agent_server_url}/api/file/archive',
            params={'path': archive_path, 'format': fmt},
            headers=headers,
            timeout=_archive_timeout(),
        )
        if response.status_code != 200:
            # The agent-server (software-agent-sdk GET /api/file/archive) returns
            # 400 (not a directory / not a git repo / bad base_ref) or 404 (path
            # missing) when there is genuinely nothing to capture — those are
            # permanent for this workspace, so let the delete proceed even under
            # REQUIRED. Every OTHER status (401 auth, 422 bad format, 429, any
            # 5xx, etc.) is a real failure that retrying could fix, so it must
            # block a REQUIRED delete instead of being misread as "no data".
            permanent = response.status_code in (400, 404)
            _logger.warning(
                'Workspace archive for %s: agent-server returned %s; %s',
                sandbox_id,
                response.status_code,
                'nothing to archive' if permanent else 'retryable failure',
            )
            return permanent or not archive_required()
        data = response.content
        base_commit = response.headers.get('X-Archive-Base-Commit', '')
    except Exception as e:
        # Network/timeout error: genuinely transient.
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
                'conversation_id': conversation_key,
                'base_commit': base_commit,
                'format': fmt,
                'source_path': archive_path,
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
