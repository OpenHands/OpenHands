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
from urllib.parse import unquote

import httpx

from openhands.agent_server.utils import utc_now
from openhands.app_server.file_store import get_file_store
from openhands.app_server.file_store.files import FileStore

_logger = logging.getLogger(__name__)

# Formats the SDK GET /api/file/archive producer accepts (git-delta | tar.gz);
# anything else 422s, so validate before issuing the request.
_ARCHIVE_SUFFIX = {'git-delta': 'patch', 'tar.gz': 'tar.gz'}

_REPO_METADATA_HEADERS = {
    'repo_remote': 'X-Archive-Repo-Remote',
    'branch': 'X-Archive-Branch',
    'head_commit': 'X-Archive-Head-Commit',
}
_REPO_ROOT_HEADER = 'X-Archive-Repo-Root'


def _extract_repo_metadata(
    headers: Any, existing: dict[str, str] | None = None
) -> dict[str, str]:
    """Pull repo-identity fields from an archive response's headers.

    Falls back to ``existing`` per-key so a later response missing a header
    (e.g. a format the agent-server can't probe) doesn't clobber a value an
    earlier response in the same capture already found.
    """
    fallback = existing or {}
    return {
        key: unquote(headers.get(header_name, '')) or fallback.get(key, '')
        for key, header_name in _REPO_METADATA_HEADERS.items()
    }


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


def _manifest_enrichment_enabled() -> bool:
    """Whether to probe packages and runtime versions into the manifest."""
    return os.getenv('RUNTIME_FILE_ARCHIVE_ENRICH', 'true').lower() in ('true', '1')


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
    # delete finalizer, so a long wait here doesn't block any user request.
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


_PROBE_TIMEOUT = 15.0
_MAX_PACKAGES_PER_MANAGER = 2000
_ENVIRONMENT_CMD = (
    'echo "python=$(.venv/bin/python --version 2>/dev/null '
    '|| python3 --version 2>/dev/null || true)"; '
    'echo "node=$(node --version 2>/dev/null || true)"; '
    'echo "os=$(. /etc/os-release 2>/dev/null; echo "${ID:-} ${VERSION_ID:-}")"'
)


def _parse_pip_list(out: str) -> dict[str, str]:
    """Parse ``pip list --format=json`` output into ``{name: version}``."""
    try:
        data = json.loads(out or '[]')
    except json.JSONDecodeError:
        return {}
    result: dict[str, str] = {}
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        name, version = item.get('name'), item.get('version')
        if name and version:
            result[name] = version
        if len(result) >= _MAX_PACKAGES_PER_MANAGER:
            break
    return result


def _parse_npm_ls(out: str) -> dict[str, str]:
    """Parse ``npm ls --json`` output (top-level deps) into ``{name: version}``."""
    try:
        data = json.loads(out or '{}')
    except json.JSONDecodeError:
        return {}
    deps = data.get('dependencies') if isinstance(data, dict) else None
    if not isinstance(deps, dict):
        return {}
    result: dict[str, str] = {}
    for name, meta in deps.items():
        version = meta.get('version') if isinstance(meta, dict) else None
        if name and version:
            result[name] = version
        if len(result) >= _MAX_PACKAGES_PER_MANAGER:
            break
    return result


def _parse_runtime(out: str) -> dict[str, str]:
    """Parse the `key=value` runtime lines into ``{python, node, os}``."""
    result: dict[str, str] = {}
    for line in out.splitlines():
        key, _, value = line.partition('=')
        key, value = key.strip(), value.strip()
        if key not in ('python', 'node', 'os') or not value:
            continue
        if key == 'python':
            value = value.removeprefix('Python ').strip()
        elif key == 'node':
            value = value.lstrip('v')
        if value:
            result[key] = value
    return result


async def _run_probe(
    httpx_client: httpx.AsyncClient,
    agent_server_url: str,
    headers: dict[str, str],
    cwd: str,
    command: str,
) -> str:
    """Run one command in the workspace; '' on any failure (never raises)."""
    try:
        response = await httpx_client.post(
            f'{agent_server_url}/api/bash/execute_bash_command',
            json={'command': command, 'cwd': cwd, 'timeout': _PROBE_TIMEOUT},
            headers=headers,
            timeout=_PROBE_TIMEOUT + 1,
        )
        if response.status_code != 200:
            return ''
        data = response.json()
    except Exception as e:
        _logger.debug('Workspace probe %r failed: %s', command, e)
        return ''
    output = data.get('stdout') if isinstance(data, dict) else None
    return output if isinstance(output, str) else ''


async def _probe_workspace(
    httpx_client: httpx.AsyncClient,
    agent_server_url: str,
    headers: dict[str, str],
    cwd: str,
) -> dict[str, Any]:
    """Collect installed package and runtime versions."""
    if not _manifest_enrichment_enabled():
        return {}

    async def _run(command: str) -> str:
        return await _run_probe(httpx_client, agent_server_url, headers, cwd, command)

    result: dict[str, Any] = {}
    packages: dict[str, dict[str, str]] = {}
    # Prefer a project venv / uv (where the agent's installs actually land)
    # before the system interpreter; each emits the same --format=json shape.
    pip = _parse_pip_list(
        await _run(
            '.venv/bin/python -m pip list --format=json 2>/dev/null '
            '|| uv pip list --format=json 2>/dev/null '
            '|| python3 -m pip list --format=json 2>/dev/null'
        )
    )
    if pip:
        packages['pip'] = pip
    npm = _parse_npm_ls(await _run('npm ls --json --depth=0 2>/dev/null'))
    if npm:
        packages['npm'] = npm
    if packages:
        result['packages'] = packages

    environment = _parse_runtime(await _run(_ENVIRONMENT_CMD))
    if environment:
        result['environment'] = environment
    return result


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
    # Repo identity rides the response headers (empty against an agent-server
    # image predating them — graceful) and is reused across formats so each
    # manifest is self-describing (repo / branch / captured HEAD).
    repo_metadata = dict.fromkeys(_REPO_METADATA_HEADERS, '')
    enrichment: dict[str, Any] = {}
    enrichment_probed = False
    probe_path = archive_path
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
                repo_metadata = _extract_repo_metadata(response.headers, repo_metadata)
                response_repo_root = response.headers.get(_REPO_ROOT_HEADER, '')
                if response_repo_root:
                    probe_path = unquote(response_repo_root)
                # Stream to disk so the archive never sits whole in RAM.
                tmp_path, byte_count = await _stream_to_tempfile(response)
            if not enrichment_probed:
                enrichment_probed = True
                try:
                    enrichment = await _probe_workspace(
                        httpx_client,
                        agent_server_url,
                        headers,
                        probe_path,
                    )
                except Exception as e:
                    _logger.debug(
                        'Workspace enrichment skipped for %s: %s', sandbox_id, e
                    )
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
                    **repo_metadata,
                    'packages': enrichment.get('packages', {}),
                    'environment': enrichment.get('environment', {}),
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
