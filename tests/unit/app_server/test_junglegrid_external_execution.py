import os
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from pydantic import SecretStr

from openhands.app_server.external_execution.junglegrid_client import (
    JungleGridApiError,
    JungleGridClient,
)


def _clean_env() -> dict[str, str]:
    env = {}
    for key in ['PATH', 'HOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'TMPDIR', 'TMP', 'TEMP']:
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def test_junglegrid_external_execution_disabled_without_env():
    from openhands.app_server.config import config_from_env

    with patch.dict(os.environ, _clean_env(), clear=True):
        config = config_from_env()

    assert config.external_execution is None


def test_junglegrid_external_execution_enabled_from_env():
    from openhands.app_server.config import config_from_env
    from openhands.app_server.external_execution.junglegrid_service import (
        JungleGridExecutionServiceInjector,
    )

    env = _clean_env()
    env['JUNGLEGRID_API_KEY'] = 'jg_test_key'
    env['JUNGLEGRID_API_BASE'] = 'https://api.junglegrid.example/'
    env['JUNGLEGRID_WORKSPACE_ID'] = 'workspace-123'

    with patch.dict(os.environ, env, clear=True):
        config = config_from_env()

    assert isinstance(config.external_execution, JungleGridExecutionServiceInjector)
    assert config.external_execution.api_key.get_secret_value() == 'jg_test_key'
    assert config.external_execution.base_url == 'https://api.junglegrid.example'
    assert config.external_execution.workspace_id == 'workspace-123'


@pytest.mark.parametrize(
    ('method_name', 'args', 'expected_method', 'expected_path', 'expected_body'),
    [
        (
            'estimate_job',
            ({'workload_type': 'batch'},),
            'POST',
            '/v1/mcp/jobs/estimate',
            {'workload_type': 'batch'},
        ),
        (
            'submit_job',
            ({'name': 'tests', 'workload_type': 'batch', 'image': 'python:3.12'},),
            'POST',
            '/v1/mcp/jobs',
            {'name': 'tests', 'workload_type': 'batch', 'image': 'python:3.12'},
        ),
        ('get_job', ('job 123',), 'GET', '/v1/mcp/jobs/job%20123', None),
        ('get_job_status', ('job 123',), 'GET', '/v1/mcp/jobs/job%20123', None),
        (
            'get_job_logs',
            ('job 123',),
            'GET',
            '/v1/mcp/jobs/job%20123/logs',
            None,
        ),
        (
            'cancel_job',
            ('job 123',),
            'POST',
            '/v1/mcp/jobs/job%20123/cancel',
            {'reason': 'Cancelled via OpenHands'},
        ),
        (
            'list_artifacts',
            ('job 123',),
            'GET',
            '/v1/mcp/jobs/job%20123/artifacts',
            None,
        ),
        (
            'get_artifact_download_url',
            ('job 123', 'artifact 456'),
            'POST',
            '/v1/mcp/jobs/job%20123/artifacts/artifact%20456/download',
            None,
        ),
    ],
)
@pytest.mark.asyncio
async def test_junglegrid_client_methods_call_documented_endpoints(
    method_name, args, expected_method, expected_path, expected_body
):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={'ok': True, 'data': {'status': 'ok'}})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as httpx_client:
        client = JungleGridClient(
            base_url='https://api.junglegrid.example/',
            api_key=SecretStr('jg_test_key'),
            httpx_client=httpx_client,
        )

        result = await getattr(client, method_name)(*args)

    assert result == {'status': 'ok'}
    assert len(requests) == 1
    request = requests[0]
    assert request.method == expected_method
    assert request.url == httpx.URL(f'https://api.junglegrid.example{expected_path}')
    assert request.headers['Accept'] == 'application/json'
    assert request.headers['Authorization'] == 'Bearer jg_test_key'
    assert request.headers['Content-Type'] == 'application/json'
    if expected_body is None:
        assert request.content == b''
    else:
        assert _request_json(request) == expected_body


@pytest.mark.asyncio
async def test_junglegrid_client_get_job_logs_supports_pagination():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={'items': []})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as httpx_client:
        client = JungleGridClient(
            base_url='https://api.junglegrid.example',
            api_key=SecretStr('jg_test_key'),
            httpx_client=httpx_client,
        )

        assert await client.get_job_logs('job 123', limit=50, cursor='cursor-1') == {
            'items': []
        }

    assert requests[0].url == httpx.URL(
        'https://api.junglegrid.example/v1/mcp/jobs/job%20123/logs'
        '?limit=50&cursor=cursor-1'
    )


@pytest.mark.asyncio
async def test_junglegrid_client_cancel_job_accepts_reason():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={'status': 'cancelled'})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as httpx_client:
        client = JungleGridClient(
            base_url='https://api.junglegrid.example',
            api_key=SecretStr('jg_test_key'),
            httpx_client=httpx_client,
        )

        await client.cancel_job('job-123', reason='user requested')

    assert _request_json(requests[0]) == {'reason': 'user requested'}


@pytest.mark.asyncio
async def test_junglegrid_client_raises_structured_api_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                'error': {
                    'code': 'FORBIDDEN',
                    'message': 'api key missing jobs:read scope',
                }
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as httpx_client:
        client = JungleGridClient(
            base_url='https://api.junglegrid.example',
            api_key=SecretStr('jg_test_key'),
            httpx_client=httpx_client,
        )

        with pytest.raises(JungleGridApiError) as exc_info:
            await client.get_job('job-123')

    assert exc_info.value.status_code == 403
    assert exc_info.value.code == 'FORBIDDEN'
    assert str(exc_info.value) == 'api key missing jobs:read scope'


def test_junglegrid_external_execution_supports_legacy_env_aliases():
    from openhands.app_server.config import config_from_env
    from openhands.app_server.external_execution.junglegrid_service import (
        JungleGridExecutionServiceInjector,
    )

    env = _clean_env()
    env['JUNGLE_GRID_API_KEY'] = 'jg_test_key'
    env['JUNGLE_GRID_API_URL'] = 'https://api.junglegrid.example/'

    with patch.dict(os.environ, env, clear=True):
        config = config_from_env()

    assert isinstance(config.external_execution, JungleGridExecutionServiceInjector)
    assert config.external_execution.api_key.get_secret_value() == 'jg_test_key'
    assert config.external_execution.base_url == 'https://api.junglegrid.example'


def test_junglegrid_external_execution_uses_documented_default_base_url():
    from openhands.app_server.config import config_from_env

    env = _clean_env()
    env['JUNGLEGRID_API_KEY'] = 'jg_test_key'

    with patch.dict(os.environ, env, clear=True):
        config = config_from_env()

    assert config.external_execution is not None
    assert config.external_execution.base_url == 'https://api.junglegrid.dev'


def _request_json(request: httpx.Request) -> Any:
    return httpx.Response(200, request=request, content=request.content).json()
