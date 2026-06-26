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
    # Default to 'both' for now — keep the compact git-delta AND a self-contained
    # full tar.gz in the bucket until the storage policy is settled (infra#1444).
    return os.getenv('RUNTIME_FILE_ARCHIVE_FORMAT', 'both')


def _formats_to_capture() -> list[str] | None:
    """Resolve RUNTIME_FILE_ARCHIVE_FORMAT to the list of formats to upload.

    'both' captures the git-delta AND the full tar.gz; a single format captures
    just that one. Returns None for an unsupported value (a hard config error the
    SDK producer would 422), so the caller can log + skip instead of mis-reading
    it as "nothing to archive".
    """
    fmt = _archive_format()
    if fmt == 'both':
        return ['git-delta', 'tar.gz']
    if fmt in _ARCHIVE_SUFFIX:
        return [fmt]
    return None


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

    formats = _formats_to_capture()
    if formats is None:
        # The SDK producer would 422 this; treat it as a hard config error
        # (mirrors runtime-api._validate_archive_format) rather than letting the
        # 4xx be misread as "nothing to archive" and silently deleting.
        _logger.error(
            'Workspace archive for %s: unsupported RUNTIME_FILE_ARCHIVE_FORMAT '
            '%r (valid: %s); skipping archive',
            sandbox_id,
            _archive_format(),
            ['git-delta', 'tar.gz', 'both'],
        )
        return not archive_required()

    headers = {'X-Session-API-Key': session_api_key} if session_api_key else {}
    # Conversation id keys the archive and locates the (possibly grouped) repo.
    # For cloud conversations the sandbox id is the conversation_id.hex.
    conversation_key = conversation_id or sandbox_id
    archive_path = _archive_path(grouping_strategy, conversation_key)
    ts = utc_now().strftime('%Y%m%dT%H%M%SZ')
    base_path = f'{_archive_prefix()}/{sandbox_id}/{ts}'

    # 'both' uploads each format under its own suffix ({ts}.patch + {ts}.tar.gz),
    # each with its own manifest. base_commit only rides the git-delta response
    # header, so capture it there and reuse it for the tar.gz manifest.
    retryable_failure = False
    base_commit = ''
    for fmt in formats:
        suffix = _ARCHIVE_SUFFIX[fmt]
        try:
            response = await httpx_client.get(
                f'{agent_server_url}/api/file/archive',
                params={'path': archive_path, 'format': fmt},
                headers=headers,
                timeout=_archive_timeout(),
            )
        except Exception as e:
            # Network/timeout error: genuinely transient.
            _logger.warning(
                'Workspace archive fetch (%s) failed for %s: %s', fmt, sandbox_id, e
            )
            retryable_failure = True
            continue

        if response.status_code != 200:
            # 400 (not a directory / not a git repo / bad base_ref) or 404 (path
            # missing) means there is genuinely nothing to capture for this
            # format — permanent, so it must not block a REQUIRED delete. Every
            # OTHER status (401 auth, 422 bad format, 429, any 5xx) is retryable.
            permanent = response.status_code in (400, 404)
            _logger.warning(
                'Workspace archive (%s) for %s: agent-server returned %s; %s',
                fmt,
                sandbox_id,
                response.status_code,
                'nothing to archive' if permanent else 'retryable failure',
            )
            if not permanent:
                retryable_failure = True
            continue

        data = response.content
        header_base = response.headers.get('X-Archive-Base-Commit', '')
        if header_base:
            base_commit = header_base

        try:
            store = _get_archive_file_store()
            await asyncio.to_thread(store.write, f'{base_path}.{suffix}', data)
            manifest = json.dumps(
                {
                    'sandbox_id': sandbox_id,
                    'conversation_id': conversation_key,
                    'phase': 'final',
                    'base_commit': base_commit,
                    'format': fmt,
                    'source_path': archive_path,
                    'byte_count': len(data),
                    'created_at': ts,
                },
                sort_keys=True,
            ).encode('utf-8')
            await asyncio.to_thread(
                store.write, f'{base_path}.{suffix}.manifest.json', manifest
            )
            _logger.info(
                'Archived workspace (%s) for %s (%d bytes) to %s.%s',
                fmt,
                sandbox_id,
                len(data),
                base_path,
                suffix,
            )
        except Exception as e:
            _logger.exception(
                'Workspace archive upload (%s) failed for %s: %s', fmt, sandbox_id, e
            )
            retryable_failure = True

    # Deletion may proceed unless archiving is REQUIRED and a retryable failure
    # left us short of the data we were meant to capture.
    if archive_required() and retryable_failure:
        return False
    return True


def initial_archive_enabled() -> bool:
    """Whether to capture the workspace's INITIAL state (before the first step).

    Independent of ``RUNTIME_FILE_ARCHIVE_ENABLED`` (the delete/pause capture of
    the *final* state) so the pre-agent snapshot can be toggled on its own. Off
    by default — like every other capture knob, nothing happens until enabled.
    """
    return os.getenv(
        'RUNTIME_FILE_ARCHIVE_INITIAL_ENABLED', 'false'
    ).lower() in ('true', '1')


def _initial_archive_format() -> str:
    """Format for the initial snapshot. Defaults to a self-contained tar.gz.

    At conversation start the working tree has no changes yet, so a ``git-delta``
    would be empty; a full ``tar.gz`` is the only format that captures anything
    and, unlike a delta keyed to ``base_commit``, it survives the upstream repo
    or branch later disappearing (the fragile re-clone path we want to avoid).
    """
    return os.getenv('RUNTIME_FILE_ARCHIVE_INITIAL_FORMAT', 'tar.gz')


async def archive_initial_workspace(
    httpx_client: Any,
    *,
    agent_server_url: str | None,
    session_api_key: str | None,
    project_dir: str,
    sandbox_id: str,
    conversation_id: str | None = None,
    base_commit: str = '',
) -> bool:
    """Snapshot the workspace BEFORE the agent's first step; return success.

    Captures the repo exactly as cloned (option A — the pre- vs post-setup choice
    is the open design question tracked in All-Hands-AI/infra#1444) as a
    self-contained ``tar.gz`` plus a ``phase=initial`` manifest, so evals have the
    true starting state even if the source repo later disappears.

    This is strictly best-effort: it never raises and never blocks conversation
    startup. A failure (feature off, misconfig, agent-server hiccup) just means no
    initial snapshot for this run, logged and swallowed. Returns True only when an
    archive was actually written.
    """
    if not initial_archive_enabled():
        return False
    if not agent_server_url:
        _logger.warning(
            'Initial workspace archive skipped for %s: no agent-server URL',
            sandbox_id,
        )
        return False
    if not _archive_bucket():
        _logger.error(
            'Initial workspace archive enabled for %s but '
            'RUNTIME_FILE_ARCHIVE_BUCKET is not set; skipping initial snapshot',
            sandbox_id,
        )
        return False

    fmt = _initial_archive_format()
    if fmt not in _ARCHIVE_SUFFIX:
        _logger.error(
            'Initial workspace archive for %s: unsupported '
            'RUNTIME_FILE_ARCHIVE_INITIAL_FORMAT %r (valid: %s); skipping',
            sandbox_id,
            fmt,
            sorted(_ARCHIVE_SUFFIX),
        )
        return False
    suffix = _ARCHIVE_SUFFIX[fmt]
    headers = {'X-Session-API-Key': session_api_key} if session_api_key else {}

    try:
        response = await httpx_client.get(
            f'{agent_server_url}/api/file/archive',
            params={'path': project_dir, 'format': fmt},
            headers=headers,
            timeout=_archive_timeout(),
        )
        if response.status_code != 200:
            _logger.warning(
                'Initial workspace archive for %s: agent-server returned %s; '
                'no initial snapshot',
                sandbox_id,
                response.status_code,
            )
            return False
        data = response.content
        # tar.gz carries no base-commit header (git-delta sets it); fall back to
        # the caller-provided HEAD sha so the initial snapshot still records the
        # commit it came from.
        captured_base = (
            response.headers.get('X-Archive-Base-Commit', '') or base_commit
        )
    except Exception as e:
        _logger.warning(
            'Initial workspace archive fetch failed for %s: %s', sandbox_id, e
        )
        return False

    try:
        store = _get_archive_file_store()
        ts = utc_now().strftime('%Y%m%dT%H%M%SZ')
        conversation_key = conversation_id or sandbox_id
        # Nest under /initial/ so it never collides with the final capture, which
        # writes to {prefix}/{sandbox_id}/{ts}.
        base_path = f'{_archive_prefix()}/{sandbox_id}/initial/{ts}'
        await asyncio.to_thread(store.write, f'{base_path}.{suffix}', data)
        manifest = json.dumps(
            {
                'sandbox_id': sandbox_id,
                'conversation_id': conversation_key,
                'phase': 'initial',
                'base_commit': captured_base,
                'format': fmt,
                'source_path': project_dir,
                'byte_count': len(data),
                'created_at': ts,
            },
            sort_keys=True,
        ).encode('utf-8')
        await asyncio.to_thread(store.write, f'{base_path}.manifest.json', manifest)
        _logger.info(
            'Archived INITIAL workspace for %s (%d bytes) to %s.%s',
            sandbox_id,
            len(data),
            base_path,
            suffix,
        )
        return True
    except Exception as e:
        _logger.exception(
            'Initial workspace archive upload failed for %s: %s', sandbox_id, e
        )
        return False
