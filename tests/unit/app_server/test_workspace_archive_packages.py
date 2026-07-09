"""Package snapshot in the workspace-archive manifest."""

import json
from unittest.mock import patch

import pytest

from openhands.app_server.sandbox import workspace_archive as wa


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
            'name': 'proj',
            'dependencies': {
                'express': {'version': '4.18.2'},
                'lodash': {'version': '4.17.21'},
            },
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


class _Result:
    def __init__(self, stdout: str, exit_code: int = 0):
        self.stdout = stdout
        self.exit_code = exit_code


@pytest.mark.asyncio
async def test_probe_packages_pip_and_npm():
    pip_json = json.dumps([{'name': 'requests', 'version': '2.31.0'}])
    # npm ls exits non-zero on peer-dep gripes but still emits valid JSON.
    npm_json = json.dumps({'dependencies': {'express': {'version': '4.18.2'}}})

    async def fake_exec(command, cwd, timeout):
        if 'pip list' in command:
            return _Result(pip_json)
        if 'npm ls' in command:
            return _Result(npm_json, exit_code=1)
        return _Result('')

    with patch.object(
        wa.AsyncRemoteWorkspace, 'execute_command', side_effect=fake_exec
    ):
        pkgs = await wa._probe_packages('http://host', 'key', '/workspace/project/repo')

    assert pkgs == {
        'pip': {'requests': '2.31.0'},
        'npm': {'express': '4.18.2'},
    }


@pytest.mark.asyncio
async def test_probe_packages_omits_absent_managers():
    async def fake_exec(command, cwd, timeout):
        return _Result('')  # neither manager present

    with patch.object(
        wa.AsyncRemoteWorkspace, 'execute_command', side_effect=fake_exec
    ):
        assert await wa._probe_packages('http://host', 'key', '/c') == {}


@pytest.mark.asyncio
async def test_probe_packages_never_raises_on_command_failure():
    async def boom(command, cwd, timeout):
        raise RuntimeError('agent-server unreachable')

    with patch.object(wa.AsyncRemoteWorkspace, 'execute_command', side_effect=boom):
        assert await wa._probe_packages('http://host', 'key', '/c') == {}


@pytest.mark.asyncio
async def test_probe_packages_disabled_by_env(monkeypatch):
    monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_PACKAGES', 'false')

    async def fail(command, cwd, timeout):
        raise AssertionError('must not probe when disabled')

    with patch.object(wa.AsyncRemoteWorkspace, 'execute_command', side_effect=fail):
        assert await wa._probe_packages('http://host', 'key', '/c') == {}
