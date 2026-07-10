"""Test package and runtime manifest enrichment."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

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


class _Result:
    def __init__(self, stdout: str, exit_code: int = 0):
        self.stdout = stdout
        self.exit_code = exit_code


_ENV_OUT = 'python=Python 3.12.4\nnode=v20.11.0\nos=ubuntu 24.04\n'


@pytest.mark.asyncio
async def test_probe_workspace_full():
    pip_json = json.dumps([{'name': 'requests', 'version': '2.31.0'}])
    npm_json = json.dumps({'dependencies': {'express': {'version': '4.18.2'}}})

    async def fake_exec(command, cwd, timeout):
        if 'pip list' in command:
            return _Result(pip_json)
        if 'npm ls' in command:
            return _Result(npm_json, exit_code=1)  # nonzero but valid JSON
        if command == wa._ENVIRONMENT_CMD:
            return _Result(_ENV_OUT)
        return _Result('')

    with patch.object(
        wa.AsyncRemoteWorkspace, 'execute_command', side_effect=fake_exec
    ):
        result = await wa._probe_workspace('http://host', 'key', '/repo')

    assert result == {
        'packages': {'pip': {'requests': '2.31.0'}, 'npm': {'express': '4.18.2'}},
        'environment': {'python': '3.12.4', 'node': '20.11.0', 'os': 'ubuntu 24.04'},
    }


@pytest.mark.asyncio
async def test_probe_workspace_omits_absent_tools():
    async def fake_exec(command, cwd, timeout):
        return _Result('')  # no tool present / no output

    with patch.object(
        wa.AsyncRemoteWorkspace, 'execute_command', side_effect=fake_exec
    ):
        assert await wa._probe_workspace('http://host', 'key', '/repo') == {}


@pytest.mark.asyncio
async def test_probe_workspace_never_raises():
    async def boom(command, cwd, timeout):
        raise RuntimeError('agent-server unreachable')

    with patch.object(wa.AsyncRemoteWorkspace, 'execute_command', side_effect=boom):
        assert await wa._probe_workspace('http://host', 'key', '/repo') == {}


@pytest.mark.asyncio
async def test_run_probe_closes_client_without_blocking_event_loop(monkeypatch):
    workspace = MagicMock()

    async def execute_command(command, cwd, timeout):
        time.sleep(0.05)
        return _Result('done')

    workspace.execute_command = execute_command
    workspace.reset_client = AsyncMock()
    monkeypatch.setattr(wa, 'AsyncRemoteWorkspace', lambda **kwargs: workspace)

    probe = asyncio.create_task(wa._run_probe('http://host', 'key', '/repo', 'cmd'))
    await asyncio.sleep(0.01)

    assert not probe.done()
    assert await probe == 'done'
    workspace.reset_client.assert_awaited_once()


@pytest.mark.asyncio
async def test_probe_disabled_by_env(monkeypatch):
    monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_ENRICH', 'false')

    async def fail(command, cwd, timeout):
        raise AssertionError('must not probe when disabled')

    with patch.object(wa.AsyncRemoteWorkspace, 'execute_command', side_effect=fail):
        assert await wa._probe_workspace('h', 'k', '/r') == {}
