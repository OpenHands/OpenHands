from __future__ import annotations

import hashlib
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from openhands.app_server.sandbox.sandbox_models import (
    AGENT_SERVER,
    ExposedUrl,
    SandboxInfo,
    SandboxPage,
    SandboxRecord,
    SandboxStatus,
)
from openhands.app_server.sandbox.sandbox_service import (
    SandboxService,
    SandboxServiceInjector,
)
from openhands.app_server.sandbox.sandbox_spec_models import (
    SandboxSpecInfo,
    SandboxSpecInfoPage,
)
from openhands.app_server.sandbox.sandbox_spec_service import (
    SandboxSpecService,
    SandboxSpecServiceInjector,
)
from openhands.app_server.services.injector import InjectorState


def fingerprint(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]


def _bearer_token(header_value: str | None) -> str | None:
    if not header_value or not header_value.startswith('Bearer '):
        return None
    return header_value.removeprefix('Bearer ')


def _redact_key_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        'fingerprint': fingerprint(record.get('key')),
        'user_id': record.get('user_id'),
        'team_id': record.get('team_id'),
        'key_alias': record.get('key_alias'),
        'metadata': record.get('metadata') or {},
    }


def _extract_llm_key(payload: Any) -> str | None:
    if isinstance(payload, dict):
        if 'llm' in payload and isinstance(payload['llm'], dict):
            value = payload['llm'].get('api_key')
            if isinstance(value, str):
                return value
        for value in payload.values():
            nested = _extract_llm_key(value)
            if nested:
                return nested
    elif isinstance(payload, list):
        for value in payload:
            nested = _extract_llm_key(value)
            if nested:
                return nested
    return None


def _redact_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        redacted: dict[str, Any] = {}
        for key, value in payload.items():
            lowered = key.lower()
            if lowered in {'api_key', 'key'} and isinstance(value, str):
                redacted[key] = {
                    'redacted': True,
                    'fingerprint': fingerprint(value),
                }
            elif 'token' in lowered and isinstance(value, str):
                redacted[key] = {
                    'redacted': True,
                    'fingerprint': fingerprint(value),
                }
            else:
                redacted[key] = _redact_payload(value)
        return redacted
    if isinstance(payload, list):
        return [_redact_payload(value) for value in payload]
    if isinstance(payload, str):
        value = re.sub(
            r'(?:sk-live|live-llm)-[A-Za-z0-9-]+',
            '<redacted-local-llm-key>',
            payload,
        )
        value = re.sub(
            r'(?:sk-oh|live-api)-[A-Za-z0-9-]+',
            '<redacted-local-api-key>',
            value,
        )
        value = re.sub(r'http://127\.0\.0\.1:\d+', 'http://127.0.0.1:<port>', value)
        value = re.sub(r'127\.0\.0\.1:\d+', '127.0.0.1:<port>', value)
        return value
    return payload


@dataclass
class StubState:
    keys: dict[str, dict[str, Any]] = field(default_factory=dict)
    deleted_keys: list[dict[str, Any]] = field(default_factory=list)
    generated_keys: list[dict[str, Any]] = field(default_factory=list)
    verify_calls: list[dict[str, Any]] = field(default_factory=list)
    delete_calls: list[dict[str, Any]] = field(default_factory=list)
    agent_start_calls: list[dict[str, Any]] = field(default_factory=list)
    profile_calls: list[dict[str, Any]] = field(default_factory=list)
    hook_calls: list[dict[str, Any]] = field(default_factory=list)
    skill_calls: list[dict[str, Any]] = field(default_factory=list)
    bash_calls: list[dict[str, Any]] = field(default_factory=list)
    generation_counter: int = 0

    def reset(self, initial_keys: list[dict[str, Any]]) -> None:
        self.keys = {item['key']: dict(item) for item in initial_keys}
        self.deleted_keys = []
        self.generated_keys = []
        self.verify_calls = []
        self.delete_calls = []
        self.agent_start_calls = []
        self.profile_calls = []
        self.hook_calls = []
        self.skill_calls = []
        self.bash_calls = []
        self.generation_counter = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            'keys': [_redact_key_record(record) for record in self.keys.values()],
            'deleted_keys': self.deleted_keys,
            'generated_keys': self.generated_keys,
            'verify_calls': self.verify_calls,
            'delete_calls': self.delete_calls,
            'agent_start_calls': self.agent_start_calls,
            'profile_calls': self.profile_calls,
            'hook_calls': self.hook_calls,
            'skill_calls': self.skill_calls,
            'bash_calls': self.bash_calls,
        }


stub_state = StubState()
stub_app = FastAPI(title='OpenHands issue 15022 live evidence stub')


class ResetPayload(BaseModel):
    initial_keys: list[dict[str, Any]]


class DeleteKeyPayload(BaseModel):
    key: str
    reason: str | None = None


@stub_app.get('/alive')
@stub_app.get('/health')
async def alive() -> dict[str, bool]:
    return {'alive': True}


@stub_app.post('/__test/reset')
async def reset(payload: ResetPayload) -> dict[str, Any]:
    stub_state.reset(payload.initial_keys)
    return stub_state.snapshot()


@stub_app.post('/__test/delete_key')
async def delete_key(payload: DeleteKeyPayload) -> dict[str, Any]:
    record = stub_state.keys.pop(payload.key, None)
    deleted = {
        'fingerprint': fingerprint(payload.key),
        'reason': payload.reason,
        'found': record is not None,
    }
    stub_state.deleted_keys.append(deleted)
    return deleted


@stub_app.get('/__test/state')
async def state() -> dict[str, Any]:
    return stub_state.snapshot()


@stub_app.get('/v1/models')
async def models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token = _bearer_token(authorization)
    valid = bool(token and token in stub_state.keys)
    stub_state.verify_calls.append(
        {
            'fingerprint': fingerprint(token),
            'valid': valid,
        }
    )
    if not valid:
        raise HTTPException(status_code=401, detail='token_not_found_in_db')
    return {'data': [{'id': 'openhands/gpt-5.5', 'object': 'model'}]}


@stub_app.post('/key/generate')
async def key_generate(request: Request) -> dict[str, str]:
    body = await request.json()
    stub_state.generation_counter += 1
    key = f'live-llm-generated-{stub_state.generation_counter}-{uuid4().hex[:16]}'
    record = {
        'key': key,
        'user_id': body.get('user_id'),
        'team_id': body.get('team_id'),
        'key_alias': body.get('key_alias'),
        'metadata': body.get('metadata') or {},
    }
    stub_state.keys[key] = record
    stub_state.generated_keys.append(_redact_key_record(record))
    return {'key': key}


@stub_app.post('/key/delete')
async def key_delete(request: Request) -> dict[str, Any]:
    body = await request.json()
    keys = [key for key in body.get('keys', []) if isinstance(key, str)]
    aliases = [alias for alias in body.get('key_aliases', []) if isinstance(alias, str)]
    removed: list[dict[str, Any]] = []
    for key in keys:
        record = stub_state.keys.pop(key, None)
        removed.append(
            {
                'mode': 'key',
                'fingerprint': fingerprint(key),
                'found': record is not None,
            }
        )
    for alias in aliases:
        matches = [
            key
            for key, record in stub_state.keys.items()
            if record.get('key_alias') == alias
        ]
        for key in matches:
            stub_state.keys.pop(key, None)
        removed.append(
            {
                'mode': 'alias',
                'key_alias': alias,
                'removed_count': len(matches),
            }
        )
    stub_state.delete_calls.append(
        {'request': _redact_payload(body), 'removed': removed}
    )
    if keys and not any(item['found'] for item in removed if item['mode'] == 'key'):
        raise HTTPException(status_code=404, detail='key_not_found')
    return {'deleted': True, 'removed': removed}


@stub_app.get('/user/info')
async def user_info(user_id: str) -> dict[str, Any]:
    return {
        'keys': [
            {
                'key_name': record['key'],
                'key_alias': record.get('key_alias'),
                'team_id': record.get('team_id'),
                'metadata': record.get('metadata') or {},
            }
            for record in stub_state.keys.values()
            if record.get('user_id') == user_id
        ]
    }


@stub_app.post('/api/profiles/{name}')
async def upsert_profile(name: str, request: Request) -> dict[str, Any]:
    body = await request.json()
    stub_state.profile_calls.append(
        {'method': 'POST', 'name': name, 'body': _redact_payload(body)}
    )
    return {'name': name}


@stub_app.get('/api/profiles')
async def list_profiles() -> dict[str, list[Any]]:
    stub_state.profile_calls.append({'method': 'GET'})
    return {'profiles': []}


@stub_app.delete('/api/profiles/{name}')
async def delete_profile(name: str) -> dict[str, bool]:
    stub_state.profile_calls.append({'method': 'DELETE', 'name': name})
    return {'deleted': True}


@stub_app.post('/api/hooks')
async def hooks(request: Request) -> dict[str, Any]:
    body = await request.json()
    stub_state.hook_calls.append(_redact_payload(body))
    return {'hook_config': None}


@stub_app.post('/api/skills')
async def skills(request: Request) -> dict[str, Any]:
    body = await request.json()
    stub_state.skill_calls.append(_redact_payload(body))
    return {'skills': [], 'sources': {}}


@stub_app.post('/api/bash/start_bash_command')
async def start_bash_command(request: Request) -> dict[str, str]:
    body = await request.json()
    command_id = uuid4().hex
    stub_state.bash_calls.append(
        {'id': command_id, 'request': _redact_payload(body), 'completed': True}
    )
    return {'id': command_id}


@stub_app.get('/api/bash/bash_events/search')
async def search_bash_events(command_id__eq: str | None = None) -> dict[str, Any]:
    return {
        'items': [
            {
                'id': uuid4().hex,
                'kind': 'BashOutput',
                'order': 0,
                'stdout': '',
                'stderr': '',
                'exit_code': 0,
                'command_id': command_id__eq,
            }
        ]
    }


@stub_app.post('/api/conversations')
async def create_conversation(request: Request) -> dict[str, Any]:
    body = await request.json()
    llm_key = _extract_llm_key(body)
    accepted = bool(llm_key and llm_key in stub_state.keys)
    call = {
        'llm_key_fingerprint': fingerprint(llm_key),
        'accepted': accepted,
        'conversation_id': body.get('conversation_id'),
        'request': _redact_payload(body),
    }
    if not accepted:
        call['rejection_detail'] = 'token_not_found_in_db'
    stub_state.agent_start_calls.append(call)
    if not accepted:
        raise HTTPException(status_code=500, detail='token_not_found_in_db')
    response = dict(body)
    response['id'] = body.get('conversation_id') or str(uuid4())
    return response


@dataclass
class LocalSandboxService(SandboxService):
    agent_server_url: str
    session_api_key: str = 'session-live-evidence'
    sandbox_id: str = 'sandbox-live-evidence'
    sandbox_spec_id: str = 'sandbox-spec-live-evidence'

    def _info(self) -> SandboxInfo:
        return SandboxInfo(
            id=self.sandbox_id,
            created_by_user_id=None,
            sandbox_spec_id=self.sandbox_spec_id,
            status=SandboxStatus.RUNNING,
            session_api_key=self.session_api_key,
            exposed_urls=[
                ExposedUrl(name=AGENT_SERVER, url=self.agent_server_url, port=0)
            ],
        )

    async def search_sandboxes(
        self, page_id: str | None = None, limit: int = 100
    ) -> SandboxPage:
        return SandboxPage(items=[self._info()])

    async def get_sandbox(self, sandbox_id: str) -> SandboxInfo | None:
        if sandbox_id != self.sandbox_id:
            return None
        return self._info()

    async def get_sandbox_by_session_api_key(
        self, session_api_key: str
    ) -> SandboxInfo | None:
        if session_api_key != self.session_api_key:
            return None
        return self._info()

    async def get_sandbox_record_by_session_api_key(
        self, session_api_key: str
    ) -> SandboxRecord | None:
        if session_api_key != self.session_api_key:
            return None
        return SandboxRecord(id=self.sandbox_id, created_by_user_id=None)

    async def start_sandbox(
        self, sandbox_spec_id: str | None = None, sandbox_id: str | None = None
    ) -> SandboxInfo:
        return self._info()

    async def resume_sandbox(self, sandbox_id: str) -> bool:
        return sandbox_id == self.sandbox_id

    async def pause_sandbox(self, sandbox_id: str) -> bool:
        return sandbox_id == self.sandbox_id

    async def stop_sandbox(self, sandbox_id: str) -> bool:
        return sandbox_id == self.sandbox_id

    async def delete_sandbox(self, sandbox_id: str) -> bool:
        return sandbox_id == self.sandbox_id


class LocalSandboxServiceInjector(SandboxServiceInjector):
    agent_server_url: str

    async def inject(self, state: InjectorState, request: Request | None = None):
        yield LocalSandboxService(agent_server_url=self.agent_server_url)


@dataclass
class LocalSandboxSpecService(SandboxSpecService):
    sandbox_spec_id: str = 'sandbox-spec-live-evidence'

    def _info(self) -> SandboxSpecInfo:
        return SandboxSpecInfo(
            id=self.sandbox_spec_id,
            command=None,
            initial_env={},
            working_dir='/workspace',
        )

    async def search_sandbox_specs(
        self, page_id: str | None = None, limit: int = 100
    ) -> SandboxSpecInfoPage:
        return SandboxSpecInfoPage(items=[self._info()])

    async def get_sandbox_spec(self, sandbox_spec_id: str) -> SandboxSpecInfo | None:
        if sandbox_spec_id != self.sandbox_spec_id:
            return None
        return self._info()


class LocalSandboxSpecServiceInjector(SandboxSpecServiceInjector):
    async def inject(self, state: InjectorState, request: Request | None = None):
        yield LocalSandboxSpecService()


@asynccontextmanager
async def close_config_resources():
    from openhands.app_server.config import get_global_config

    try:
        yield
    finally:
        config = get_global_config()
        await config.db_session.close()


def patch_local_services() -> None:
    from openhands.app_server.config import get_global_config

    agent_url = os.environ['LIVE_EVIDENCE_AGENT_URL'].rstrip('/')
    config = get_global_config()
    config.sandbox = LocalSandboxServiceInjector(agent_server_url=agent_url)
    config.sandbox_spec = LocalSandboxSpecServiceInjector()
    if config.app_conversation is not None:
        config.app_conversation.sandbox_startup_timeout = 10
        config.app_conversation.sandbox_startup_poll_frequency = 1
