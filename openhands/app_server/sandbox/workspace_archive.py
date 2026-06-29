"""Archive a remote sandbox's workspace to object storage before deletion.

Pulls a workspace archive from the in-pod agent-server endpoint
(``GET /api/file/archive``) and stores it, plus a small manifest, in object
storage so the agent's work — production OpenHands Cloud workspace state —
survives sandbox deletion and is preserved for downstream use (e.g. dataset/eval
creation).

It covers the *explicit-delete-while-running* path. The dominant idle/expiry
reap is handled separately in runtime-api at pause time, because that deletion
never reaches the app-server.

Configuration is environment-driven and the feature is a no-op unless
``RUNTIME_FILE_ARCHIVE_ENABLED`` is set.
"""

import asyncio
import json
import logging
import os
import tempfile
from typing import Any

import httpx

from openhands.agent_server.utils import utc_now
from openhands.app_server.file_store import get_file_store
from openhands.app_server.file_store.files import FileStore

_logger = logging.getLogger(__name__)

# Formats the SDK GET /api/file/archive producer accepts (git-delta | tar.gz);
# anything else 422s, so validate before issuing the request.
_ARCHIVE_SUFFIX = {'git-delta': 'patch', 'tar.gz': 'tar.gz'}


def _archive_request_params(path: str, fmt: str) -> dict[str, str]:
    """Query params for GET /api/file/archive.

    The tar.gz is the SELF-CONTAINED full capture, so disable the endpoint's
    default excludes for it: otherwise agent output under dist/build/node_modules
    and the repo's .git history are dropped and it is no more complete than the
    git-delta (defeating the whole point of capturing 'both'). Credential-bearing
    git internals are still scrubbed server-side even with excludes off. git-delta
    keeps the defaults — it is the compact companion, not the full capture.
    """
    params = {'path': path, 'format': fmt}
    if fmt == 'tar.gz':
        params['use_default_excludes'] = 'false'
    return params


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
    # Default to 'both' — the compact git-delta AND a self-contained full tar.gz.
    # git-delta alone is lossy as a sole capture: it respects the repo's
    # .gitignore (so agent-authored gitignored files are dropped) and needs the
    # base tree to reconstruct, whereas the tar.gz is self-contained and captures
    # those files. Keep both until the storage cost is measured, then narrow to
    # 'git-delta' (+ bucket lifecycle) if warranted (infra#1444).
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


def _float_env(name: str, default: float) -> float:
    """Parse a float env var, falling back to default on a non-numeric value.

    A bad override (``'120s'``, a stray newline) must not raise on every archive
    call — that would wedge every REQUIRED delete forever.
    """
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        _logger.warning('Invalid %s=%r; using %s', name, raw, default)
        return default
    if value <= 0:
        # A non-positive timeout/deadline would make httpx raise on every archive
        # (the wedge this guard exists to prevent); fall back to the safe default.
        _logger.warning('Non-positive %s=%r; using %s', name, raw, default)
        return default
    return value


def _archive_timeout() -> float:
    # Must cover the agent-server git build budget (read-tree 60 + add 300 + diff
    # 300 = up to ~660s) before the first response byte flows, or large repos
    # ReadTimeout and never capture. The final archive runs in the detached
    # delete finalizer, so a long wait here doesn't block any user request; the
    # initial snapshot is separately bounded by initial_archive_deadline().
    return _float_env('RUNTIME_FILE_ARCHIVE_TIMEOUT', 660.0)


def _archive_store_type() -> str:
    # Default to GCS to preserve current behavior; local/s3 also work. NOT
    # 'memory' — it is text-only (read() returns str) and would corrupt the
    # binary archive (see InMemoryFileStore).
    return os.getenv('RUNTIME_FILE_ARCHIVE_STORE_TYPE', 'google_cloud')


def _get_archive_file_store() -> FileStore:
    """Object store for archives, built via the backend-portable factory."""
    return get_file_store(_archive_store_type(), _archive_bucket())


def _cleanup_tempfile(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


async def _stream_to_tempfile(response: Any) -> tuple[str, int]:
    """Stream a 200 response body to a temp file; return (path, byte_count).

    Avoids buffering the whole archive in app-server RAM (OOM risk under
    concurrent large deletes). Cleans up its own file if streaming fails.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False)
    byte_count = 0
    try:
        async for chunk in response.aiter_bytes():
            tmp.write(chunk)
            byte_count += len(chunk)
        tmp.close()
        return tmp.name, byte_count
    except BaseException:
        tmp.close()
        _cleanup_tempfile(tmp.name)
        raise


def _write_file_to_store(store: FileStore, name: str, path: str) -> None:
    """Stream a temp file to the store, never buffering the whole archive in RAM.

    The download was streamed to a tempfile precisely to avoid holding the
    archive in memory (see ``_stream_to_tempfile``); ``write_from_path`` keeps
    that guarantee on upload (GCS/local/S3 stream from disk) instead of reading
    the whole file back with ``store.write(name, f.read())``.
    """
    store.write_from_path(name, path)


async def archive_workspace(
    httpx_client: httpx.AsyncClient,
    runtime: dict[str, Any],
    sandbox_id: str,
    *,
    archive_path: str,
    conversation_id: str | None = None,
) -> bool:
    """Archive the workspace at ``archive_path``; return whether delete may proceed.

    ``archive_path`` is resolved by the caller — the path pinned at conversation
    creation, NOT re-derived here from live settings — so a capture can never be
    misrouted to the wrong directory. The agent-server descends from it to the
    cloned repo.

    Returns True when the workspace was archived, when the path holds nothing to
    archive (agent-server 400: not a directory / not a git repo), or when
    archiving failed but is not required (best-effort). Returns False when
    archiving is required and either hit a transient failure (5xx / network /
    422 / 429) or could not confirm a capture (401 auth / 404 missing path), so
    the caller leaves the sandbox intact for the idle-reap retry. Never raises.

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
        # Unsupported RUNTIME_FILE_ARCHIVE_FORMAT is a pure config error, exactly
        # like the unset bucket above: no retry makes a valid format appear, so
        # proceed loudly rather than wedging every REQUIRED delete forever (the
        # app-server has no idle-reap backstop). Validated here so a bad format
        # never reaches the producer (which would 422 it).
        _logger.error(
            'Workspace archive for %s: unsupported RUNTIME_FILE_ARCHIVE_FORMAT '
            '%r (valid: %s); proceeding with delete (fix the config to capture)',
            sandbox_id,
            _archive_format(),
            ['git-delta', 'tar.gz', 'both'],
        )
        return True

    headers = {'X-Session-API-Key': session_api_key} if session_api_key else {}
    # For cloud conversations the sandbox id is the conversation_id.hex.
    conversation_key = conversation_id or sandbox_id
    ts = utc_now().strftime('%Y%m%dT%H%M%SZ')
    # Key by conversation, not just sandbox: under grouping a sandbox is shared by
    # siblings, and the 1s ts is not unique — without the conversation segment two
    # sibling captures in the same second overwrite each other at the object level.
    base_path = f'{_archive_prefix()}/{sandbox_id}/{conversation_key}/{ts}'

    # 'both' uploads each format under its own suffix ({ts}.patch + {ts}.tar.gz),
    # each with its own manifest. base_commit only rides the git-delta response
    # header, so capture it there and reuse it for the tar.gz manifest.
    # A retry under REQUIRED re-uploads under a fresh {ts}; any blob/manifest left
    # by a partially-failed prior attempt becomes an orphan reaped by the bucket
    # lifecycle policy (we favor capture completeness over upload dedup).
    retryable_failure = False
    # A capture we could not confirm happened (401 auth / 404 missing path):
    # under REQUIRED this must NOT permit teardown — it is the misrouted-path
    # symptom this feature most needs to guard against, not "nothing to archive".
    unconfirmed_capture = False
    base_commit = ''
    # One store per call (not per format) — building it lazily spins up a client.
    store = _get_archive_file_store()
    for fmt in formats:
        suffix = _ARCHIVE_SUFFIX[fmt]
        tmp_path: str | None = None
        byte_count = 0
        try:
            async with httpx_client.stream(
                'GET',
                f'{agent_server_url}/api/file/archive',
                params=_archive_request_params(archive_path, fmt),
                headers=headers,
                timeout=_archive_timeout(),
            ) as response:
                if response.status_code != 200:
                    code = response.status_code
                    if code == 400:
                        # Path exists but holds no archivable repo (not a git
                        # repo / not a directory). A positive "nothing here" —
                        # safe to skip this format and proceed.
                        detail = 'nothing to archive'
                    elif code in (401, 404):
                        # 401 auth rejected / 404 path missing: the capture did
                        # NOT happen and this is not a confirmed-empty workspace,
                        # so it must block a REQUIRED delete (idle reap retries).
                        unconfirmed_capture = True
                        detail = 'capture unconfirmed (auth/path)'
                    else:
                        # 422 / 429 / 5xx — transient.
                        retryable_failure = True
                        detail = 'retryable failure'
                    _logger.warning(
                        'Workspace archive (%s) for %s: agent-server returned %s; %s',
                        fmt,
                        sandbox_id,
                        code,
                        detail,
                    )
                    continue
                header_base = response.headers.get('X-Archive-Base-Commit', '')
                if header_base:
                    base_commit = header_base
                # Stream to disk so the archive never sits whole in RAM.
                tmp_path, byte_count = await _stream_to_tempfile(response)
        except Exception as e:
            # Network/timeout error: genuinely transient.
            _logger.warning(
                'Workspace archive fetch (%s) failed for %s: %s', fmt, sandbox_id, e
            )
            retryable_failure = True
            _cleanup_tempfile(tmp_path)
            continue

        assert tmp_path is not None  # set on the 200 path above
        try:
            await asyncio.to_thread(
                _write_file_to_store, store, f'{base_path}.{suffix}', tmp_path
            )
            manifest = json.dumps(
                {
                    'sandbox_id': sandbox_id,
                    'conversation_id': conversation_key,
                    'phase': 'final',
                    'base_commit': base_commit,
                    'format': fmt,
                    'source_path': archive_path,
                    'byte_count': byte_count,
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
                byte_count,
                base_path,
                suffix,
            )
        except Exception as e:
            _logger.exception(
                'Workspace archive upload (%s) failed for %s: %s', fmt, sandbox_id, e
            )
            retryable_failure = True
        finally:
            _cleanup_tempfile(tmp_path)

    # Deletion may proceed unless archiving is REQUIRED and we either hit a
    # retryable failure or could not confirm a capture (401/404) — both leave us
    # short of the data we were meant to preserve.
    if archive_required() and (retryable_failure or unconfirmed_capture):
        return False
    return True


def initial_archive_enabled() -> bool:
    """Whether to capture the workspace's INITIAL state (before the first step).

    Independent of ``RUNTIME_FILE_ARCHIVE_ENABLED`` (the delete/pause capture of
    the *final* state) so the pre-agent snapshot can be toggled on its own. Off
    by default — like every other capture knob, nothing happens until enabled.
    """
    return os.getenv('RUNTIME_FILE_ARCHIVE_INITIAL_ENABLED', 'false').lower() in (
        'true',
        '1',
    )


def initial_archive_deadline() -> float:
    """Hard ceiling (seconds) on how long the inline, pre-setup initial snapshot
    may delay conversation startup.

    The snapshot is awaited before setup.sh so it captures the repo exactly as
    cloned, with no concurrent mutation. ``wait_for`` only charges the time
    actually spent, so a fast snapshot adds no latency; this caps the worst case
    (a large repo or a hung endpoint) so it can never dominate startup. An overrun
    is logged and startup proceeds without the snapshot. A fresh clone's tar.gz is
    just the repo tree, so the default comfortably covers typical repos; raise it
    for a very large monorepo.
    """
    return _float_env('RUNTIME_FILE_ARCHIVE_INITIAL_DEADLINE', 120.0)


def _initial_archive_format() -> str:
    """Format for the initial snapshot. Defaults to a self-contained tar.gz.

    At conversation start the working tree has no changes yet, so a ``git-delta``
    would be empty; a full ``tar.gz`` is the only format that captures anything
    and, unlike a delta keyed to ``base_commit``, it survives the upstream repo
    or branch later disappearing (the fragile re-clone path we want to avoid).
    """
    return os.getenv('RUNTIME_FILE_ARCHIVE_INITIAL_FORMAT', 'tar.gz')


async def archive_initial_workspace(
    httpx_client: httpx.AsyncClient,
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
    self-contained ``tar.gz`` plus a ``phase=initial`` manifest, so the snapshot
    records the true starting state even if the source repo later disappears.

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

    tmp_path: str | None = None
    byte_count = 0
    try:
        async with httpx_client.stream(
            'GET',
            f'{agent_server_url}/api/file/archive',
            params=_archive_request_params(project_dir, fmt),
            headers=headers,
            timeout=_archive_timeout(),
        ) as response:
            if response.status_code != 200:
                _logger.warning(
                    'Initial workspace archive for %s: agent-server returned %s; '
                    'no initial snapshot',
                    sandbox_id,
                    response.status_code,
                )
                return False
            # tar.gz carries no base-commit header (git-delta sets it); fall back
            # to the caller-provided HEAD sha so the snapshot still records the
            # commit it came from.
            captured_base = (
                response.headers.get('X-Archive-Base-Commit', '') or base_commit
            )
            # Stream to disk so the archive never sits whole in RAM.
            tmp_path, byte_count = await _stream_to_tempfile(response)
    except Exception as e:
        _logger.warning(
            'Initial workspace archive fetch failed for %s: %s', sandbox_id, e
        )
        _cleanup_tempfile(tmp_path)
        return False

    assert tmp_path is not None  # set on the 200 path above
    try:
        store = _get_archive_file_store()
        ts = utc_now().strftime('%Y%m%dT%H%M%SZ')
        conversation_key = conversation_id or sandbox_id
        # Key by conversation (siblings share a grouped sandbox) and nest under
        # /initial/ so it never collides with that conversation's final capture
        # ({prefix}/{sandbox_id}/{conversation_key}/{ts}).
        blob_name = (
            f'{_archive_prefix()}/{sandbox_id}/{conversation_key}/initial/{ts}.{suffix}'
        )
        await asyncio.to_thread(_write_file_to_store, store, blob_name, tmp_path)
        manifest = json.dumps(
            {
                'sandbox_id': sandbox_id,
                'conversation_id': conversation_key,
                'phase': 'initial',
                'base_commit': captured_base,
                'format': fmt,
                'source_path': project_dir,
                'byte_count': byte_count,
                'created_at': ts,
            },
            sort_keys=True,
        ).encode('utf-8')
        # Shared contract: manifest = blob + '.manifest.json' (was dropping the
        # format suffix, so a downstream enricher could never locate it).
        await asyncio.to_thread(store.write, f'{blob_name}.manifest.json', manifest)
        _logger.info(
            'Archived INITIAL workspace for %s (%d bytes) to %s',
            sandbox_id,
            byte_count,
            blob_name,
        )
        return True
    except Exception as e:
        _logger.exception(
            'Initial workspace archive upload failed for %s: %s', sandbox_id, e
        )
        return False
    finally:
        _cleanup_tempfile(tmp_path)
