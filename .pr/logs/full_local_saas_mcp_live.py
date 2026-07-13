"""Full local SaaS evidence for PR #15103 / issue #15226.

Runs against a disposable local PostgreSQL/Redis-backed enterprise app.
The script intentionally prints only fingerprints and structural checks, never
raw API keys, bearer tokens, cookies, session keys, or MCP credentials.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastmcp import FastMCP
from sqlalchemy import delete, select
from starlette.middleware import Middleware
from starlette.types import Receive, Scope, Send

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTERPRISE_ROOT = REPO_ROOT / 'enterprise'
APP_HOST = '127.0.0.1'
APP_PORT = int(os.getenv('OH_PR15103_APP_PORT', '13000'))
APP_URL = f'http://{APP_HOST}:{APP_PORT}'
LLM_PORT = int(os.getenv('OH_PR15103_LLM_PORT', '13080'))
MCP_PORT = int(os.getenv('OH_PR15103_MCP_PORT', '13081'))
STATE_DIR = Path(os.getenv('OH_PR15103_STATE_DIR', '/tmp/oh-pr15103-full-local'))
JWT_SECRET = os.getenv('JWT_SECRET', 'pr15103-local-jwt-secret')

ORG_ID = uuid.UUID('15103000-0000-4000-8000-000000000001')
ADMIN_ID = uuid.UUID('15103000-0000-4000-8000-000000000002')
PEER_ID = uuid.UUID('15103000-0000-4000-8000-000000000003')

SENSITIVE_MARKERS: set[str] = set()
MCP_TRAFFIC: list[dict[str, Any]] = []
LLM_TRAFFIC: list[dict[str, Any]] = []


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop('SESSION_API_KEY', None)
    env.update(
        {
            'PYTHONPATH': f'{ENTERPRISE_ROOT}:{REPO_ROOT}:{env.get("PYTHONPATH", "")}',
            'OPENHANDS_CONFIG_CLS': 'server.config.SaaSServerConfig',
            'OPENHANDS_SUPPRESS_BANNER': '1',
            'JWT_SECRET': JWT_SECRET,
            'OH_PERSISTENCE_DIR': str(STATE_DIR / 'persistence'),
            'RUNTIME': 'process',
            'OH_SANDBOX_KIND': 'ProcessSandboxServiceInjector',
            'OH_SANDBOX_BASE_WORKING_DIR': str(STATE_DIR / 'sandboxes'),
            'OH_SANDBOX_BASE_PORT': '13100',
            'FRONTEND_DIRECTORY': str(REPO_ROOT / 'frontend' / 'build'),
            'POSTHOG_CLIENT_KEY': 'local-posthog-disabled',
            'LAMINAR_API_KEY': '',
            'REDIS_HOST': os.getenv('REDIS_HOST', '127.0.0.1'),
            'REDIS_PORT': os.getenv('REDIS_PORT', '16379'),
            'DB_HOST': os.getenv('DB_HOST', '127.0.0.1'),
            'DB_PORT': os.getenv('DB_PORT', '15432'),
            'DB_NAME': os.getenv('DB_NAME', 'openhands_pr15103'),
            'DB_USER': os.getenv('DB_USER', 'postgres'),
            'DB_PASS': os.getenv('DB_PASS', 'postgres'),
            'LLM_BASE_URL': f'http://127.0.0.1:{LLM_PORT}/v1',
            'LITE_LLM_API_URL': f'http://127.0.0.1:{LLM_PORT}/litellm',
            'LITE_LLM_API_KEY': 'local-litellm-admin',
            'LITE_LLM_TEAM_ID': 'local-pr15103-team',
            'OPENHANDS_LLM_PROVIDER_ROUTE': 'direct',
            'OPENHANDS_DEFAULT_LLM_MODEL': 'openai/local-deterministic',
            'OPENHANDS_DEFAULT_LLM_BASE_URL': f'http://127.0.0.1:{LLM_PORT}/v1',
            'OPENHANDS_DEFAULT_LLM_API_KEY': '',
        }
    )
    return env


def fp(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def redact_obj(value: Any) -> Any:
    text = json.dumps(value, default=str)
    for marker in sorted(SENSITIVE_MARKERS, key=len, reverse=True):
        if marker:
            text = text.replace(marker, f'<redacted:{fp(marker)}>')
    return json.loads(text)


def _find_free_port(start: int) -> int:
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((APP_HOST, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f'no free port starting at {start}')


def _secret(label: str) -> str:
    value = f'{label}-{secrets.token_urlsafe(18)}'
    SENSITIVE_MARKERS.add(value)
    return value


def _llm_app(expected_api_key: str) -> FastAPI:
    app = FastAPI()
    state = {'calls': 0}

    def completion_payload(
        *,
        model: str,
        message: dict[str, Any],
        finish_reason: str,
    ) -> dict[str, Any]:
        return {
            'id': f'chatcmpl-{secrets.token_hex(6)}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': model,
            'choices': [
                {
                    'index': 0,
                    'message': message,
                    'finish_reason': finish_reason,
                }
            ],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 10,
                'total_tokens': 20,
            },
        }

    async def stream_payload(
        *,
        model: str,
        message: dict[str, Any],
        finish_reason: str,
    ) -> AsyncIterator[bytes]:
        chunk_base = {
            'id': f'chatcmpl-{secrets.token_hex(6)}',
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': model,
        }
        if message.get('tool_calls'):
            tool_call = message['tool_calls'][0]
            chunks = [
                {'role': 'assistant'},
                {
                    'tool_calls': [
                        {
                            'index': 0,
                            'id': tool_call['id'],
                            'type': 'function',
                            'function': {
                                'name': tool_call['function']['name'],
                                'arguments': tool_call['function']['arguments'],
                            },
                        }
                    ]
                },
            ]
        else:
            chunks = [{'role': 'assistant'}, {'content': message.get('content', '')}]
        for delta in chunks:
            payload = dict(chunk_base)
            payload['choices'] = [{'index': 0, 'delta': delta, 'finish_reason': None}]
            yield f'data: {json.dumps(payload)}\n\n'.encode()
        final = dict(chunk_base)
        final['choices'] = [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}]
        yield f'data: {json.dumps(final)}\n\n'.encode()
        yield b'data: [DONE]\n\n'

    @app.post('/v1/chat/completions')
    async def chat_completions(request: Request):
        authorization = request.headers.get('authorization', '')
        authorized = authorization == f'Bearer {expected_api_key}'
        if not authorized:
            LLM_TRAFFIC.append(
                {
                    'call': state['calls'] + 1,
                    'authorized': False,
                    'authorization_fp': fp(authorization),
                }
            )
            return JSONResponse({'error': 'unauthorized'}, status_code=401)
        body = await request.json()
        state['calls'] += 1
        model = body.get('model', 'openai/local-deterministic')
        tools = body.get('tools') or []
        tool_names = [
            ((tool.get('function') or {}).get('name'))
            for tool in tools
            if isinstance(tool, dict)
        ]
        messages = body.get('messages') or []
        saw_tool_result = any(msg.get('role') == 'tool' for msg in messages)
        target_tool = next(
            (name for name in tool_names if name and 'preserved_auth_probe' in name),
            None,
        )
        LLM_TRAFFIC.append(
            {
                'call': state['calls'],
                'has_tools': bool(tools),
                'tool_count': len(tool_names),
                'target_tool_seen': target_tool is not None,
                'saw_tool_result': saw_tool_result,
                'stream': bool(body.get('stream')),
                'authorized': True,
                'authorization_fp': fp(authorization),
            }
        )
        if target_tool and not saw_tool_result:
            message = {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': 'call_preserved_auth_probe',
                        'type': 'function',
                        'function': {'name': target_tool, 'arguments': '{}'},
                    }
                ],
            }
            finish_reason = 'tool_calls'
        else:
            message = {
                'role': 'assistant',
                'content': 'preserved MCP tool call complete',
            }
            finish_reason = 'stop'
        if body.get('stream'):
            return StreamingResponse(
                stream_payload(
                    model=model,
                    message=message,
                    finish_reason=finish_reason,
                ),
                media_type='text/event-stream',
            )
        return completion_payload(
            model=model,
            message=message,
            finish_reason=finish_reason,
        )

    @app.get('/v1/models')
    async def models():
        return {'object': 'list', 'data': [{'id': 'openai/local-deterministic'}]}

    return app


class McpAuthCaptureMiddleware:
    def __init__(self, app: Any, *, expected_bearer: str, expected_header: str):
        self.app = app
        self.expected_bearer = expected_bearer
        self.expected_header = expected_header

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope['type'] != 'http' or not scope.get('path', '').startswith('/mcp'):
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode('latin1').lower(): value.decode('latin1')
            for key, value in scope.get('headers', [])
        }
        auth = headers.get('authorization', '')
        extra = headers.get('x-pr15103-header', '')
        authorized = (
            auth == f'Bearer {self.expected_bearer}' and extra == self.expected_header
        )
        MCP_TRAFFIC.append(
            {
                'path': scope.get('path'),
                'method': scope.get('method'),
                'authorization_fp': fp(auth),
                'extra_header_fp': fp(extra),
                'authorized': authorized,
            }
        )
        if not authorized:
            response = JSONResponse({'error': 'unauthorized'}, status_code=401)
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


def _mcp_app(expected_bearer: str, expected_header: str) -> Any:
    mcp = FastMCP('preserved-auth-evidence')

    @mcp.tool()
    def preserved_auth_probe() -> str:
        return 'mcp-live-tool-marker'

    return mcp.http_app(
        path='/mcp',
        middleware=[
            Middleware(
                McpAuthCaptureMiddleware,
                expected_bearer=expected_bearer,
                expected_header=expected_header,
            )
        ],
        transport='streamable-http',
    )


def _run_uvicorn(app: FastAPI, port: int) -> uvicorn.Server:
    config = uvicorn.Config(app, host=APP_HOST, port=port, log_level='warning')
    server = uvicorn.Server(config)
    return server


async def _start_server(app: FastAPI, port: int) -> tuple[uvicorn.Server, asyncio.Task]:
    server = _run_uvicorn(app, port)
    task = asyncio.create_task(server.serve())
    deadline = time.monotonic() + 20
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                await client.get(f'http://{APP_HOST}:{port}/')
                return server, task
            except Exception:
                await asyncio.sleep(0.1)
    raise RuntimeError(f'server on port {port} did not start')


async def _stop_server(server: uvicorn.Server, task: asyncio.Task) -> None:
    server.should_exit = True
    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(task, timeout=10)


def _start_app_process() -> subprocess.Popen[str]:
    env = _env()
    log_path = STATE_DIR / 'saas-app.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open('a', buffering=1)
    proc = subprocess.Popen(
        [
            sys.executable,
            '-m',
            'uvicorn',
            'saas_server:app',
            '--host',
            APP_HOST,
            '--port',
            str(APP_PORT),
        ],
        cwd=ENTERPRISE_ROOT,
        env=env,
        stdout=handle,
        stderr=handle,
        text=True,
    )
    return proc


async def _wait_app(client: httpx.AsyncClient) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            resp = await client.get(f'{APP_URL}/health')
            if resp.status_code < 500:
                return
        except Exception:
            pass
        await asyncio.sleep(0.25)
    raise RuntimeError('enterprise app did not start')


async def _stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        await asyncio.to_thread(proc.wait, 10)
    except subprocess.TimeoutExpired:
        proc.kill()
        await asyncio.to_thread(proc.wait)


async def seed_identity() -> dict[str, str]:
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from storage.api_key import ApiKey
    from storage.api_key_store import ApiKeyStore
    from storage.database import _get_db_session_injector
    from storage.org import Org
    from storage.org_member import OrgMember
    from storage.role import Role
    from storage.user import User

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    db_injector = _get_db_session_injector()
    async_engine = await db_injector.get_async_db_engine()
    session_maker = async_sessionmaker(async_engine, expire_on_commit=False)

    admin_key = ApiKeyStore().generate_api_key()
    peer_key = ApiKeyStore().generate_api_key()
    wrong_key = ApiKeyStore().generate_api_key()
    SENSITIVE_MARKERS.update({admin_key, peer_key, wrong_key})

    async with session_maker() as session:
        await session.execute(
            delete(ApiKey).where(ApiKey.user_id.in_([str(ADMIN_ID), str(PEER_ID)]))
        )
        await session.execute(delete(OrgMember).where(OrgMember.org_id == ORG_ID))
        await session.execute(delete(User).where(User.id.in_([ADMIN_ID, PEER_ID])))
        await session.execute(delete(Org).where(Org.id == ORG_ID))
        await session.commit()

        role = await session.scalar(select(Role).where(Role.id == 3))
        if role is None:
            session.add(Role(id=3, name='member', rank=1000))
        org = Org(
            id=ORG_ID,
            name=f'pr15103-local-{secrets.token_hex(4)}',
            org_version=1,
            agent_settings={},
            conversation_settings={},
            enable_proactive_conversation_starters=True,
        )
        session.add(org)
        session.add_all(
            [
                User(
                    id=ADMIN_ID,
                    current_org_id=ORG_ID,
                    accepted_tos=datetime.now(UTC).replace(tzinfo=None),
                    email='pr15103-admin@example.invalid',
                    email_verified=True,
                    enable_sound_notifications=False,
                    user_consents_to_analytics=False,
                    onboarding_completed=True,
                    sandbox_grouping_strategy='NO_GROUPING',
                ),
                User(
                    id=PEER_ID,
                    current_org_id=ORG_ID,
                    accepted_tos=datetime.now(UTC).replace(tzinfo=None),
                    email='pr15103-peer@example.invalid',
                    email_verified=True,
                    enable_sound_notifications=False,
                    user_consents_to_analytics=False,
                    onboarding_completed=True,
                    sandbox_grouping_strategy='NO_GROUPING',
                ),
            ]
        )
        session.add_all(
            [
                OrgMember(
                    org_id=ORG_ID,
                    user_id=ADMIN_ID,
                    role_id=3,
                    llm_api_key='seed-admin-llm-placeholder',
                    has_custom_llm_api_key=True,
                    agent_settings_diff={},
                    conversation_settings_diff={},
                    status='active',
                ),
                OrgMember(
                    org_id=ORG_ID,
                    user_id=PEER_ID,
                    role_id=3,
                    llm_api_key='seed-peer-llm-placeholder',
                    has_custom_llm_api_key=True,
                    agent_settings_diff={},
                    conversation_settings_diff={},
                    status='active',
                ),
                ApiKey(
                    key=admin_key,
                    user_id=str(ADMIN_ID),
                    org_id=ORG_ID,
                    name='pr15103-admin',
                ),
                ApiKey(
                    key=peer_key,
                    user_id=str(PEER_ID),
                    org_id=ORG_ID,
                    name='pr15103-peer',
                ),
            ]
        )
        await session.commit()

    return {'admin': admin_key, 'peer': peer_key, 'wrong': wrong_key}


def auth_headers(key: str) -> dict[str, str]:
    return {'Authorization': f'Bearer {key}', 'X-Org-Id': str(ORG_ID)}


async def get_json(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    key: str,
    *,
    json_body: Any | None = None,
    extra_headers: dict[str, str] | None = None,
    expected: int = 200,
) -> Any:
    headers = auth_headers(key)
    if extra_headers:
        headers.update(extra_headers)
    resp = await client.request(
        method, f'{APP_URL}{path}', headers=headers, json=json_body
    )
    if resp.status_code != expected:
        raise AssertionError(
            f'{method} {path} expected {expected}, got {resp.status_code}: {resp.text[:500]}'
        )
    if not resp.content:
        return None
    return resp.json()


def has_raw_marker(value: Any) -> bool:
    text = json.dumps(value, default=str)
    return any(marker and marker in text for marker in SENSITIVE_MARKERS)


def get_mcp_servers(settings: dict[str, Any]) -> dict[str, Any]:
    agent = settings.get('agent_settings') or {}
    mcp_config = agent.get('mcp_config') or {}
    if 'mcpServers' in mcp_config:
        return mcp_config.get('mcpServers') or {}
    return mcp_config


async def wait_for_ready_task(
    client: httpx.AsyncClient, key: str, task_id: str
) -> dict[str, Any]:
    deadline = time.monotonic() + 120
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        page = await get_json(
            client,
            'GET',
            '/api/v1/app-conversations/start-tasks/search',
            key,
        )
        for item in page.get('items', []):
            if item.get('id') == task_id:
                last = item
                if item.get('status') in {'READY', 'ERROR'}:
                    return item
        await asyncio.sleep(1)
    raise AssertionError(f'start task did not finish; last={redact_obj(last)}')


async def wait_for_conversation_stop(
    client: httpx.AsyncClient, key: str, conversation_id: str
) -> dict[str, Any]:
    deadline = time.monotonic() + 120
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        data = await get_json(
            client,
            'GET',
            f'/api/v1/app-conversations?ids={conversation_id}',
            key,
        )
        conv = (data or [None])[0]
        last = conv
        status = str((conv or {}).get('execution_status') or '').lower()
        if conv and status in {'stopped', 'finished', 'error'}:
            return conv
        await asyncio.sleep(1)
    return last or {}


def secret_leaf_omitted(value: Any, key: str) -> bool:
    if value is None:
        return True
    if not isinstance(value, dict):
        return False
    return key not in value or value.get(key) in {None, '**********'}


async def run_proof() -> dict[str, Any]:
    global LLM_PORT, MCP_PORT

    os.environ.pop('SESSION_API_KEY', None)
    os.environ.update(_env())
    for name in (
        'DB_HOST',
        'DB_PORT',
        'DB_NAME',
        'DB_USER',
        'DB_PASS',
        'REDIS_HOST',
        'REDIS_PORT',
    ):
        os.environ.setdefault(name, _env()[name])
    os.environ.setdefault('JWT_SECRET', JWT_SECRET)
    os.environ.setdefault('OH_PERSISTENCE_DIR', str(STATE_DIR / 'persistence'))
    os.environ.setdefault('OPENHANDS_CONFIG_CLS', 'server.config.SaaSServerConfig')

    llm_key = _secret('llm-key')
    llm_server, llm_task = await _start_server(
        _llm_app(llm_key),
        _find_free_port(LLM_PORT),
    )
    LLM_PORT = llm_server.config.port
    os.environ.update(_env())

    http_bearer = _secret('http-bearer')
    header_secret = _secret('header-secret')
    stdio_secret = _secret('stdio-env')
    stdio_other = _secret('stdio-other')
    mcp_server, mcp_task = await _start_server(
        _mcp_app(http_bearer, header_secret),
        _find_free_port(MCP_PORT),
    )
    MCP_PORT = mcp_server.config.port

    keys = await seed_identity()
    app_proc = _start_app_process()
    async with httpx.AsyncClient(timeout=30.0) as client:
        await _wait_app(client)

        missing_llm_payload = {
            'agent_settings_diff': {
                'llm': {
                    'model': 'openai/local-deterministic',
                    'base_url': f'http://127.0.0.1:{LLM_PORT}/v1',
                    'api_key': None,
                },
                'mcp_config': {},
            }
        }
        await get_json(
            client,
            'POST',
            '/api/v1/settings',
            keys['admin'],
            json_body=missing_llm_payload,
        )
        bad_task = await get_json(
            client,
            'POST',
            '/api/v1/app-conversations',
            keys['admin'],
            json_body={
                'initial_message': {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'This should fail without an LLM key.'}
                    ],
                    'run': True,
                }
            },
        )
        bad_done = await wait_for_ready_task(client, keys['admin'], bad_task['id'])
        bad_conv_id = bad_done.get('app_conversation_id')
        if bad_conv_id:
            await wait_for_conversation_stop(client, keys['admin'], bad_conv_id)
        missing_llm_rejected = bad_done.get('status') == 'ERROR' or any(
            event.get('authorized') is False for event in LLM_TRAFFIC
        )
        LLM_TRAFFIC.clear()

        settings_payload = {
            'agent_settings_diff': {
                'llm': {
                    'model': 'openai/local-deterministic',
                    'base_url': f'http://127.0.0.1:{LLM_PORT}/v1',
                    'api_key': llm_key,
                },
                'mcp_config': {
                    'native-http': {
                        'url': f'http://127.0.0.1:{MCP_PORT}/mcp',
                        'transport': 'streamable-http',
                        'auth': {'strategy': 'bearer', 'value': http_bearer},
                        'headers': {'X-PR15103-Header': header_secret},
                    },
                    'stdio': {
                        'command': 'python',
                        'args': ['-c', "print('stdio placeholder')"],
                        'env': {'API_KEY': stdio_secret, 'OTHER_VAR': stdio_other},
                    },
                },
            },
            'conversation_settings_diff': {'max_iterations': 4},
        }
        await get_json(
            client,
            'POST',
            '/api/v1/settings',
            keys['admin'],
            json_body=settings_payload,
        )

        redacted_get = await get_json(client, 'GET', '/api/v1/settings', keys['admin'])
        users_me = await get_json(client, 'GET', '/api/v1/users/me', keys['admin'])
        peer_get = await get_json(client, 'GET', '/api/v1/settings', keys['peer'])

        redacted_servers = get_mcp_servers(redacted_get)
        redacted_http = redacted_servers.get('native-http', {})
        redacted_stdio = redacted_servers.get('stdio', {})
        redacted_get_no_raw = not has_raw_marker(redacted_get)
        users_me_no_raw = not has_raw_marker(users_me)
        peer_has_no_mcp = get_mcp_servers(peer_get) == {}

        edited = redacted_servers
        edited['unrelated'] = {
            'url': f'http://127.0.0.1:{MCP_PORT}/mcp',
            'transport': 'streamable-http',
            'auth': {'strategy': 'none'},
        }
        await get_json(
            client,
            'POST',
            '/api/v1/settings',
            keys['admin'],
            json_body={'agent_settings_diff': {'mcp_config': edited}},
        )

        await _stop_process(app_proc)
        app_proc = _start_app_process()
        await _wait_app(client)

        wrong_mcp_payload = {
            'agent_settings_diff': {
                'mcp_config': {
                    'native-http': {
                        'url': f'http://127.0.0.1:{MCP_PORT}/mcp',
                        'transport': 'streamable-http',
                        'auth': {'strategy': 'bearer', 'value': 'wrong'},
                        'headers': {'X-PR15103-Header': header_secret},
                    }
                }
            },
            'conversation_settings_diff': {'max_iterations': 2},
        }
        await get_json(
            client,
            'POST',
            '/api/v1/settings',
            keys['peer'],
            json_body={
                'agent_settings_diff': {
                    'llm': {
                        'model': 'openai/local-deterministic',
                        'base_url': f'http://127.0.0.1:{LLM_PORT}/v1',
                        'api_key': llm_key,
                    },
                    **wrong_mcp_payload['agent_settings_diff'],
                },
                'conversation_settings_diff': wrong_mcp_payload[
                    'conversation_settings_diff'
                ],
            },
        )
        wrong_task = await get_json(
            client,
            'POST',
            '/api/v1/app-conversations',
            keys['peer'],
            json_body={
                'initial_message': {
                    'role': 'user',
                    'content': [{'type': 'text', 'text': 'Call preserved_auth_probe.'}],
                    'run': True,
                }
            },
        )
        wrong_done = await wait_for_ready_task(client, keys['peer'], wrong_task['id'])
        wrong_conv_id = wrong_done.get('app_conversation_id')
        if wrong_conv_id:
            await wait_for_conversation_stop(client, keys['peer'], wrong_conv_id)
        wrong_mcp_rejected = any(event['authorized'] is False for event in MCP_TRAFFIC)
        MCP_TRAFFIC.clear()
        LLM_TRAFFIC.clear()

        start = await get_json(
            client,
            'POST',
            '/api/v1/app-conversations',
            keys['admin'],
            json_body={
                'initial_message': {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': 'Call preserved_auth_probe exactly once, then stop.',
                        }
                    ],
                    'run': True,
                }
            },
        )
        ready = await wait_for_ready_task(client, keys['admin'], start['id'])
        if ready.get('status') != 'READY':
            raise AssertionError(
                f'conversation did not become READY: {redact_obj(ready)}'
            )
        conv = await wait_for_conversation_stop(
            client, keys['admin'], ready['app_conversation_id']
        )

        post_restart_get = await get_json(
            client, 'GET', '/api/v1/settings', keys['admin']
        )
        post_servers = get_mcp_servers(post_restart_get)

    await _stop_process(app_proc)
    await _stop_server(mcp_server, mcp_task)
    await _stop_server(llm_server, llm_task)

    checks = {
        'missing_llm_rejected': missing_llm_rejected,
        'wrong_mcp_credentials_rejected': wrong_mcp_rejected,
        'api_redaction_no_raw_settings': redacted_get_no_raw,
        'api_redaction_no_raw_users_me': users_me_no_raw,
        'peer_member_cannot_see_mcp': peer_has_no_mcp,
        'redacted_http_auth_not_exposed': (
            (redacted_http.get('auth') or {}).get('strategy') == 'bearer'
            and secret_leaf_omitted(redacted_http.get('auth'), 'value')
        ),
        'redacted_http_header_not_exposed': secret_leaf_omitted(
            redacted_http.get('headers'), 'X-PR15103-Header'
        ),
        'redacted_stdio_env_not_exposed': secret_leaf_omitted(
            redacted_stdio.get('env'), 'API_KEY'
        ),
        'unrelated_mcp_edit_survived_restart': set(post_servers)
        >= {'native-http', 'stdio', 'unrelated'},
        'fresh_conversation_ready': ready.get('status') == 'READY',
        'fresh_conversation_terminal_or_running': str(
            conv.get('execution_status') or ''
        ).lower()
        in {'stopped', 'finished', 'running', ''},
        'llm_saw_target_tool': any(
            item.get('target_tool_seen') for item in LLM_TRAFFIC
        ),
        'llm_saw_tool_result': any(item.get('saw_tool_result') for item in LLM_TRAFFIC),
        'mcp_authorized_tool_traffic': any(
            item.get('authorized') for item in MCP_TRAFFIC
        ),
    }
    return {
        'result': 'PASS' if all(checks.values()) else 'FAIL',
        'refs': {
            'head': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True
            ).strip(),
            'main': subprocess.check_output(
                ['git', 'rev-parse', 'origin/main'], cwd=REPO_ROOT, text=True
            ).strip(),
        },
        'setup': {
            'app_url': APP_URL,
            'db': {
                'host': os.getenv('DB_HOST', '127.0.0.1'),
                'port': os.getenv('DB_PORT', '15432'),
                'name': os.getenv('DB_NAME', 'openhands_pr15103'),
            },
            'redis': {
                'host': os.getenv('REDIS_HOST', '127.0.0.1'),
                'port': os.getenv('REDIS_PORT', '16379'),
            },
            'runtime': 'process',
            'auth_path': 'SaasUserAuth bearer API key via production api_keys table',
        },
        'secret_fingerprints': {
            'llm_api_key': fp(llm_key),
            'mcp_bearer': fp(http_bearer),
            'mcp_header': fp(header_secret),
            'stdio_api_key': fp(stdio_secret),
            'stdio_other': fp(stdio_other),
        },
        'redacted_shapes': {
            'http_auth': redact_obj(redacted_http.get('auth')),
            'http_headers': redact_obj(redacted_http.get('headers')),
            'stdio_env': redact_obj(redacted_stdio.get('env')),
            'post_restart_servers': sorted(post_servers),
        },
        'llm_traffic': LLM_TRAFFIC,
        'mcp_traffic': MCP_TRAFFIC,
        'conversation': {
            'start_task_status': ready.get('status'),
            'app_conversation_id_fp': fp(ready.get('app_conversation_id')),
            'execution_status': conv.get('execution_status'),
            'sandbox_status': conv.get('sandbox_status'),
        },
        'checks': checks,
    }


async def main() -> None:
    result = await run_proof()
    print(json.dumps(redact_obj(result), indent=2, sort_keys=True))
    if result['result'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    asyncio.run(main())
