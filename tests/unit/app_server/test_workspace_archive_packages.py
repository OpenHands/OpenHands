"""Test package and runtime manifest enrichment."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from openhands.app_server.sandbox import workspace_archive as wa

# --- parsers ---------------------------------------------------------------


def test_parse_pip_list():
    out = json.dumps(
        [
            {'name': 'requests', 'version': '2.31.0'},
            {'name': 'flask', 'version': '3.0.0'},
        ]
    )
    assert wa._parse_pip_list(out) == {'requests': '2.31.0', 'flask': '3.0.0'}


def test_parse_pip_list_bad_input():
    assert wa._parse_pip_list('not json') == {}
    assert wa._parse_pip_list('') == {}
    assert wa._parse_pip_list('{"unexpected": "shape"}') == {}
    assert wa._parse_pip_list('[1]') == {}


def test_parse_npm_ls_top_level():
    out = json.dumps(
        {
            'dependencies': {
                'express': {'version': '4.18.2'},
                'lodash': {'version': '4.17.21'},
            }
        }
    )
    assert wa._parse_npm_ls(out) == {'express': '4.18.2', 'lodash': '4.17.21'}


def test_parse_npm_ls_bad_input():
    assert wa._parse_npm_ls('') == {}
    assert wa._parse_npm_ls('not json') == {}
    assert wa._parse_npm_ls(json.dumps({'no': 'deps'})) == {}
    assert wa._parse_npm_ls(json.dumps({'dependencies': []})) == {}


def test_parse_caps_entries():
    big = json.dumps(
        [
            {'name': f'p{i}', 'version': '1'}
            for i in range(wa._MAX_PACKAGES_PER_MANAGER + 50)
        ]
    )
    assert len(wa._parse_pip_list(big)) == wa._MAX_PACKAGES_PER_MANAGER


def test_parse_runtime_strips_prefixes():
    out = 'python=Python 3.12.4\nnode=v20.11.0\nos=ubuntu 24.04'
    assert wa._parse_runtime(out) == {
        'python': '3.12.4',
        'node': '20.11.0',
        'os': 'ubuntu 24.04',
    }


def test_parse_runtime_omits_empty():
    assert wa._parse_runtime('python=Python 3.12\nnode=\nos= ') == {'python': '3.12'}
    assert '2>&1' not in wa._ENVIRONMENT_CMD


def test_extract_repo_metadata_decodes_percent_encoding():
    headers = {
        'X-Archive-Repo-Remote': (
            'https%3A%2F%2Fgithub.com%2Fexample%2Ffeature%252Frepo.git'
        ),
        'X-Archive-Branch': 'caf%C3%A9%25branch',
    }
    assert wa._extract_repo_metadata(headers) == {
        'repo_remote': 'https://github.com/example/feature%2Frepo.git',
        'branch': 'café%branch',
        'head_commit': '',
    }


def _response(stdout: str, status_code: int = 200):
    response = MagicMock(status_code=status_code)
    response.json.return_value = {'stdout': stdout}
    return response


_ENV_OUT = 'python=Python 3.12.4\nnode=v20.11.0\nos=ubuntu 24.04\n'


@pytest.mark.asyncio
async def test_probe_workspace_full():
    pip_json = json.dumps([{'name': 'requests', 'version': '2.31.0'}])
    npm_json = json.dumps({'dependencies': {'express': {'version': '4.18.2'}}})
    client = MagicMock()
    client.post = AsyncMock(
        side_effect=[_response(pip_json), _response(npm_json), _response(_ENV_OUT)]
    )
    result = await wa._probe_workspace(
        client, 'http://host', {'X-Session-API-Key': 'key'}, '/repo'
    )

    assert result == {
        'packages': {'pip': {'requests': '2.31.0'}, 'npm': {'express': '4.18.2'}},
        'environment': {'python': '3.12.4', 'node': '20.11.0', 'os': 'ubuntu 24.04'},
    }


@pytest.mark.asyncio
async def test_probe_workspace_omits_absent_tools():
    client = MagicMock()
    client.post = AsyncMock(side_effect=[_response(''), _response(''), _response('')])
    assert await wa._probe_workspace(client, 'http://host', {}, '/repo') == {}


@pytest.mark.asyncio
async def test_probe_workspace_never_raises():
    client = MagicMock()
    client.post = AsyncMock(side_effect=RuntimeError('agent-server unreachable'))
    assert await wa._probe_workspace(client, 'http://host', {}, '/repo') == {}


@pytest.mark.asyncio
async def test_run_probe_posts_command():
    client = MagicMock()
    client.post = AsyncMock(return_value=_response('done'))
    headers = {'X-Session-API-Key': 'key'}

    assert await wa._run_probe(client, 'http://host', headers, '/repo', 'cmd') == 'done'
    client.post.assert_awaited_once_with(
        'http://host/api/bash/execute_bash_command',
        json={'command': 'cmd', 'cwd': '/repo', 'timeout': wa._PROBE_TIMEOUT},
        headers=headers,
        timeout=wa._PROBE_TIMEOUT + 1,
    )


@pytest.mark.asyncio
async def test_probe_disabled_by_env(monkeypatch):
    monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_ENRICH', 'false')
    client = MagicMock()
    client.post = AsyncMock()

    assert await wa._probe_workspace(client, 'h', {}, '/r') == {}
    client.post.assert_not_awaited()
