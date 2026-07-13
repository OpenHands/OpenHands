#!/usr/bin/env python3
"""Sanitized live smoke for PR #15110 LLM profile secret recovery.

This script is a PR artifact, not product code. It exercises the enterprise
HTTP profile API, restarts the SaaS app, reloads through SaasSettingsStore, and
invokes the recovered profile through the SDK/LiteLLM completion path.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import UUID

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTERPRISE_DIR = REPO_ROOT / 'enterprise'
FRONTEND_DIR = Path('/tmp/oh-pr15110-empty-frontend')
DOCKER_PG = 'oh-pr15110-pg'
PROFILE_NAME = 'local-openai-compatible-smoke'
MODEL = 'openai/gpt-4o-mini'
EXPECTED_COMPLETION = 'local-smoke-completion-ok'
MASK = '**********'


def sha16(value: str | bytes | None) -> str | None:
    if value is None:
        return None
    data = value if isinstance(value, bytes) else value.encode()
    return hashlib.sha256(data).hexdigest()[:16]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def run(
    cmd: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    timeout: int = 120,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    merged_env.pop('SESSION_API_KEY', None)
    merged_env.pop('VIRTUAL_ENV', None)
    merged_env.pop('POETRY_ACTIVE', None)
    merged_env.setdefault('OPENHANDS_SUPPRESS_BANNER', '1')
    merged_env.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        timeout=timeout,
        check=True,
    )


def http_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = 20,
) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    req_headers = {'Content-Type': 'application/json'}
    if headers:
        req_headers.update(headers)
    req = Request(url, data=data, method=method, headers=req_headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            return resp.status, json.loads(payload.decode() or '{}')
    except HTTPError as exc:
        payload = exc.read()
        with contextlib.suppress(Exception):
            return exc.code, json.loads(payload.decode() or '{}')
        return exc.code, payload.decode(errors='replace')


def wait_http(url: str, *, timeout: int = 90) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=3) as resp:
                if 200 <= resp.status < 500:
                    return
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f'Timed out waiting for {url}: {last_error}')


def wait_port_closed(port: int, *, timeout: int = 20) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(('127.0.0.1', port)) != 0:
                return
        time.sleep(0.2)
    raise RuntimeError(f'Port {port} did not close')


class MockOpenAIHandler(BaseHTTPRequestHandler):
    expected_api_key: str = ''
    requests_seen: list[dict[str, Any]] = []

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:
        length = int(self.headers.get('Content-Length', '0'))
        body = self.rfile.read(length)
        auth_header = self.headers.get('Authorization')
        scheme = None
        token = None
        if auth_header:
            parts = auth_header.split(' ', 1)
            scheme = parts[0]
            token = parts[1] if len(parts) == 2 else ''

        parsed_body: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            parsed_body = json.loads(body.decode() or '{}')

        matches = bool(token) and token == self.expected_api_key
        self.requests_seen.append(
            {
                'path': self.path,
                'method': 'POST',
                'auth_present': auth_header is not None,
                'auth_scheme': scheme,
                'auth_sha16': sha16(token),
                'auth_matches_expected': matches,
                'body_sha16': sha16(body),
                'body_model': parsed_body.get('model'),
                'body_message_count': len(parsed_body.get('messages') or []),
            }
        )

        if self.path not in {'/v1/chat/completions', '/chat/completions'}:
            self._send_json(404, {'error': {'message': 'not found'}})
            return

        if not matches:
            self._send_json(401, {'error': {'message': 'invalid api key'}})
            return

        self._send_json(
            200,
            {
                'id': 'chatcmpl-local-smoke',
                'object': 'chat.completion',
                'created': 1,
                'model': parsed_body.get('model') or MODEL,
                'choices': [
                    {
                        'index': 0,
                        'message': {
                            'role': 'assistant',
                            'content': EXPECTED_COMPLETION,
                        },
                        'finish_reason': 'stop',
                    }
                ],
                'usage': {
                    'prompt_tokens': 3,
                    'completion_tokens': 4,
                    'total_tokens': 7,
                },
            },
        )


def start_mock_server(port: int, expected_api_key: str) -> ThreadingHTTPServer:
    MockOpenAIHandler.expected_api_key = expected_api_key
    MockOpenAIHandler.requests_seen = []
    server = ThreadingHTTPServer(('127.0.0.1', port), MockOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def exercise_mock_rejection(port: int, wrong_key: str) -> dict[str, Any]:
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': 'ping'}]}
    missing = http_json(
        'POST', f'http://127.0.0.1:{port}/v1/chat/completions', body=body
    )[0]
    wrong = http_json(
        'POST',
        f'http://127.0.0.1:{port}/v1/chat/completions',
        headers={'Authorization': f'Bearer {wrong_key}'},
        body=body,
    )[0]
    return {'missing_auth_status': missing, 'wrong_auth_status': wrong}


def branch_env(db_name: str, jwt_secret: str, redis_port: int) -> dict[str, str]:
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    return {
        'OPENHANDS_CONFIG_CLS': 'server.config.SaaSServerConfig',
        'DB_HOST': '127.0.0.1',
        'DB_PORT': '15432',
        'DB_USER': 'postgres',
        'DB_PASSWORD': 'postgres',
        'DB_NAME': db_name,
        'JWT_SECRET': jwt_secret,
        'REDIS_HOST': '127.0.0.1',
        'REDIS_PORT': str(redis_port),
        'FRONTEND_DIRECTORY': str(FRONTEND_DIR),
        'POSTHOG_CLIENT_KEY': 'phc_pr15110_local_smoke_dummy',
        'OPENHANDS_LLM_PROVIDER_ROUTE': 'direct',
    }


def reset_database(db_name: str) -> None:
    run(
        [
            'sudo',
            'docker',
            'exec',
            DOCKER_PG,
            'dropdb',
            '-U',
            'postgres',
            '--if-exists',
            db_name,
        ],
        timeout=60,
    )
    run(
        [
            'sudo',
            'docker',
            'exec',
            DOCKER_PG,
            'createdb',
            '-U',
            'postgres',
            db_name,
        ],
        timeout=60,
    )


def migrate(env: dict[str, str]) -> str:
    result = run(
        ['poetry', 'run', 'alembic', 'upgrade', 'head'],
        cwd=ENTERPRISE_DIR,
        env=env,
        timeout=180,
    )
    return result.stdout or ''


def start_app(env: dict[str, str], port: int, log_path: Path) -> subprocess.Popen[str]:
    log_file = log_path.open('w')
    merged_env = os.environ.copy()
    merged_env.update(env)
    merged_env.pop('SESSION_API_KEY', None)
    merged_env.pop('VIRTUAL_ENV', None)
    merged_env.pop('POETRY_ACTIVE', None)
    merged_env.setdefault('OPENHANDS_SUPPRESS_BANNER', '1')
    merged_env.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    proc = subprocess.Popen(
        [
            'poetry',
            'run',
            'uvicorn',
            'saas_server:app',
            '--host',
            '127.0.0.1',
            '--port',
            str(port),
        ],
        cwd=ENTERPRISE_DIR,
        env=merged_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._smoke_log_file = log_file  # type: ignore[attr-defined]
    wait_http(f'http://127.0.0.1:{port}/health')
    return proc


def stop_app(proc: subprocess.Popen[str] | None, port: int) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=20)
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=10)
    log_file = getattr(proc, '_smoke_log_file', None)
    if log_file:
        log_file.close()
    wait_port_closed(port)


async def bootstrap_identity(api_key: str, org_name: str) -> dict[str, str]:
    sys.path.insert(0, str(ENTERPRISE_DIR))
    from uuid import uuid4

    from sqlalchemy import select
    from storage.api_key import ApiKey
    from storage.database import a_session_maker
    from storage.org import Org
    from storage.org_member import OrgMember
    from storage.role import Role
    from storage.user import User

    user_id = uuid4()
    async with a_session_maker() as session:
        owner_role = (
            (await session.execute(select(Role).where(Role.name == 'owner')))
            .scalars()
            .first()
        )
        if owner_role is None:
            raise RuntimeError('owner role was not created by migrations')
        org = Org(name=org_name)
        session.add(org)
        await session.flush()
        user = User(
            id=user_id,
            current_org_id=org.id,
            email=f'{org_name}@example.invalid',
            email_verified=True,
            enable_sound_notifications=False,
            language='en',
            user_consents_to_analytics=False,
            onboarding_completed=True,
        )
        member = OrgMember(
            org_id=org.id,
            user_id=user_id,
            role_id=owner_role.id,
            llm_api_key='bootstrap-unused',
            has_custom_llm_api_key=False,
            agent_settings_diff={},
            conversation_settings_diff={},
        )
        key = ApiKey(
            key=api_key,
            user_id=str(user_id),
            org_id=org.id,
            name='pr15110-local-smoke',
        )
        session.add_all([user, member, key])
        await session.commit()
    return {'user_id': str(user_id), 'org_id': str(org.id)}


def save_profile_through_api(
    app_port: int, org_id: str, api_key: str, llm_key: str, mock_port: int
) -> dict[str, Any]:
    profile_url = (
        f'http://127.0.0.1:{app_port}/api/organizations/{org_id}/profiles/'
        f'{PROFILE_NAME}'
    )
    payload = {
        'llm': {
            'model': MODEL,
            'base_url': f'http://127.0.0.1:{mock_port}/v1',
            'api_key': llm_key,
            'num_retries': 0,
            'retry_min_wait': 0,
            'retry_max_wait': 0,
            'timeout': 20,
            'stream': False,
        },
        'include_secrets': True,
    }
    headers = {'Authorization': f'Bearer {api_key}', 'X-Org-Id': org_id}
    save_status, save_body = http_json(
        'POST', profile_url, headers=headers, body=payload
    )
    get_status, get_body = http_json('GET', profile_url, headers=headers)
    return {
        'save_status': save_status,
        'save_body_sha16': sha16(json.dumps(save_body, sort_keys=True)),
        'get_status_before_restart': get_status,
        'get_body_before_restart_sha16': sha16(json.dumps(get_body, sort_keys=True)),
        'get_body_before_restart_contains_mask': MASK in json.dumps(get_body),
        'get_body_before_restart_contains_raw_secret': llm_key in json.dumps(get_body),
    }


def get_profile_after_restart(
    app_port: int, org_id: str, api_key: str, llm_key: str
) -> dict[str, Any]:
    profile_url = (
        f'http://127.0.0.1:{app_port}/api/organizations/{org_id}/profiles/'
        f'{PROFILE_NAME}'
    )
    headers = {'Authorization': f'Bearer {api_key}', 'X-Org-Id': org_id}
    status, body = http_json('GET', profile_url, headers=headers)
    encoded = json.dumps(body, sort_keys=True)
    return {
        'get_status_after_restart': status,
        'get_body_after_restart_sha16': sha16(encoded),
        'get_body_after_restart_contains_mask': MASK in encoded,
        'get_body_after_restart_contains_raw_secret': llm_key in encoded,
        'get_body_after_restart_model': (body.get('llm') or {}).get('model')
        if isinstance(body, dict)
        else None,
        'get_body_after_restart_base_url_sha16': sha16(
            (body.get('llm') or {}).get('base_url')
        )
        if isinstance(body, dict)
        else None,
    }


async def load_profile_and_invoke(
    user_id: str, org_id: str, llm_key: str, managed_proxy_url: str
) -> dict[str, Any]:
    sys.path.insert(0, str(ENTERPRISE_DIR))
    from storage.saas_settings_store import SaasSettingsStore

    from openhands.app_server.settings.llm_profiles import resolve_profile_llm
    from openhands.sdk.llm import Message, TextContent

    store = SaasSettingsStore(user_id, effective_org_id=UUID(org_id))
    loaded = await store.load()
    profile = loaded.llm_profiles.get(PROFILE_NAME) if loaded else None
    if profile is None:
        return {'loaded_profile_present': False, 'completion_success': False}

    recovered_key = (
        profile.api_key.get_secret_value()
        if hasattr(profile.api_key, 'get_secret_value')
        else profile.api_key
    )
    resolved = resolve_profile_llm(
        profile,
        managed_proxy_url=managed_proxy_url,
        fallback_api_key=None,
    )
    response = resolved.completion(
        [
            Message(
                role='user',
                content=[TextContent(text='Return the deterministic smoke string.')],
            )
        ],
        stream=False,
    )
    content = getattr(response, 'content', None)
    if content is None:
        message = getattr(response, 'message', None)
        message_content = getattr(message, 'content', None)
        if message_content:
            first_content = message_content[0]
            content = getattr(first_content, 'text', None)
    return {
        'loaded_profile_present': True,
        'loaded_profile_model': profile.model,
        'loaded_profile_base_url_sha16': sha16(profile.base_url),
        'loaded_profile_api_key_sha16': sha16(recovered_key),
        'loaded_profile_api_key_matches_sentinel': recovered_key == llm_key,
        'resolved_profile_base_url_sha16': sha16(resolved.base_url),
        'completion_success': content == EXPECTED_COMPLETION,
        'completion_content_sha16': sha16(content),
        'completion_content': content,
    }


async def inspect_storage(org_id: str, llm_key: str) -> dict[str, Any]:
    sys.path.insert(0, str(ENTERPRISE_DIR))
    from sqlalchemy import text
    from storage.database import a_session_maker

    async with a_session_maker() as session:
        row = (
            (
                await session.execute(
                    text(
                        'SELECT llm_profiles::text AS llm_profiles '
                        'FROM "org" WHERE id = :org_id'
                    ),
                    {'org_id': org_id},
                )
            )
            .mappings()
            .first()
        )
    raw = '' if row is None or row['llm_profiles'] is None else row['llm_profiles']
    parsed = None
    parseable = False
    with contextlib.suppress(Exception):
        parsed = json.loads(raw)
        parseable = True

    visible_markers: list[str] = []
    for marker in [PROFILE_NAME, MODEL, '127.0.0.1', '/v1']:
        if marker in raw:
            visible_markers.append(marker)

    return {
        'llm_profiles_sql_text_sha16': sha16(raw),
        'llm_profiles_sql_text_length': len(raw),
        'llm_profiles_json_parseable': parseable,
        'llm_profiles_json_type': type(parsed).__name__ if parseable else None,
        'llm_profiles_contains_raw_secret': llm_key in raw,
        'llm_profiles_contains_mask': MASK in raw,
        'llm_profiles_contains_encrypted_leaf_marker': 'gAAAA' in raw,
        'llm_profiles_visible_markers': visible_markers,
    }


def assert_no_raw_secret_in_artifact(
    artifact: dict[str, Any], secrets_to_check: list[str]
) -> None:
    encoded = json.dumps(artifact, sort_keys=True)
    leaked = [name for name in secrets_to_check if name and name in encoded]
    if leaked:
        raise RuntimeError(
            f'artifact contains raw secret material: {len(leaked)} item(s)'
        )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch-label', required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--db-name', required=True)
    parser.add_argument('--app-port', type=int, required=True)
    parser.add_argument('--mock-port', type=int, required=True)
    parser.add_argument('--redis-port', type=int, default=16379)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--logs-dir', type=Path, required=True)
    args = parser.parse_args()

    head = run(['git', 'rev-parse', 'HEAD']).stdout.strip()
    if head != args.expected_sha:
        raise RuntimeError(f'expected checkout {args.expected_sha}, got {head}')

    args.logs_dir.mkdir(parents=True, exist_ok=True)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)

    llm_key = 'sk-local-smoke-' + secrets.token_urlsafe(24)
    wrong_key = 'sk-local-wrong-' + secrets.token_urlsafe(12)
    app_api_key = 'sk-oh-local-smoke-' + secrets.token_urlsafe(24)
    jwt_secret = f'pr15110-{args.branch_label}-jwt-' + secrets.token_urlsafe(16)
    env = branch_env(args.db_name, jwt_secret, args.redis_port)

    artifact: dict[str, Any] = {
        'branch_label': args.branch_label,
        'head_sha': head,
        'started_at': utc_now(),
        'db_name': args.db_name,
        'app_port': args.app_port,
        'mock_port': args.mock_port,
        'profile_name': PROFILE_NAME,
        'model': MODEL,
        'sentinel_key_sha16': sha16(llm_key),
        'app_api_key_sha16': sha16(app_api_key),
        'wrong_key_sha16': sha16(wrong_key),
    }

    app_proc: subprocess.Popen[str] | None = None
    mock_server: ThreadingHTTPServer | None = None
    try:
        reset_database(args.db_name)
        migration_log = migrate(env)
        artifact['migration'] = {
            'ran_to_head': True,
            'log_sha16': sha16(migration_log),
            'mentions_secret_storage_rewrite': (
                'Rewrite settings secrets to field-level encrypted JSON'
                in migration_log
            ),
        }

        os.environ.update(env)
        identity = await bootstrap_identity(
            app_api_key, f'pr15110_{args.branch_label}_{int(time.time())}'
        )
        artifact['identity'] = {
            'user_id': identity['user_id'],
            'org_id': identity['org_id'],
        }

        mock_server = start_mock_server(args.mock_port, llm_key)
        artifact['mock_endpoint_self_test'] = exercise_mock_rejection(
            args.mock_port, wrong_key
        )

        first_log = args.logs_dir / f'{args.branch_label}-app-before-restart.log'
        app_proc = start_app(env, args.app_port, first_log)
        artifact['app_before_restart'] = {
            'health': 200,
            'log_sha16': sha16(first_log.read_text(errors='replace')),
        }
        artifact['api_save'] = save_profile_through_api(
            args.app_port,
            identity['org_id'],
            app_api_key,
            llm_key,
            args.mock_port,
        )
        stop_app(app_proc, args.app_port)
        app_proc = None

        restart_log = args.logs_dir / f'{args.branch_label}-app-after-restart.log'
        app_proc = start_app(env, args.app_port, restart_log)
        artifact['app_after_restart'] = {
            'health': 200,
            'log_sha16': sha16(restart_log.read_text(errors='replace')),
        }
        artifact['api_get_after_restart'] = get_profile_after_restart(
            args.app_port, identity['org_id'], app_api_key, llm_key
        )
        artifact['storage_after_restart'] = await inspect_storage(
            identity['org_id'], llm_key
        )
        artifact[
            'load_and_llm_invocation_after_restart'
        ] = await load_profile_and_invoke(
            identity['user_id'],
            identity['org_id'],
            llm_key,
            managed_proxy_url=f'http://127.0.0.1:{args.mock_port}/v1',
        )
        artifact['mock_endpoint_requests'] = MockOpenAIHandler.requests_seen
        artifact['completed_at'] = utc_now()
        artifact['overall_success'] = bool(
            artifact['api_save']['save_status'] == 200
            and artifact['api_get_after_restart']['get_status_after_restart'] == 200
            and artifact['load_and_llm_invocation_after_restart'].get(
                'loaded_profile_api_key_matches_sentinel'
            )
            and artifact['load_and_llm_invocation_after_restart'].get(
                'completion_success'
            )
            and not artifact['storage_after_restart'][
                'llm_profiles_contains_raw_secret'
            ]
            and not artifact['api_get_after_restart'][
                'get_body_after_restart_contains_raw_secret'
            ]
            and artifact['mock_endpoint_self_test']['missing_auth_status'] == 401
            and artifact['mock_endpoint_self_test']['wrong_auth_status'] == 401
        )
    except Exception as exc:
        artifact['completed_at'] = utc_now()
        artifact['overall_success'] = False
        artifact['error_type'] = exc.__class__.__name__
        artifact['error'] = (
            str(exc).replace(llm_key, '<redacted>').replace(app_api_key, '<redacted>')
        )
    finally:
        stop_app(app_proc, args.app_port)
        if mock_server is not None:
            mock_server.shutdown()
            mock_server.server_close()

    assert_no_raw_secret_in_artifact(
        artifact, [llm_key, app_api_key, wrong_key, jwt_secret]
    )
    args.artifact.write_text(json.dumps(artifact, indent=2, sort_keys=True) + '\n')
    print(
        json.dumps(
            {
                'branch_label': args.branch_label,
                'head_sha': head,
                'artifact': str(args.artifact),
                'overall_success': artifact['overall_success'],
                'sentinel_key_sha16': artifact['sentinel_key_sha16'],
            },
            sort_keys=True,
        )
    )
    return 0 if artifact['overall_success'] else 1


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
