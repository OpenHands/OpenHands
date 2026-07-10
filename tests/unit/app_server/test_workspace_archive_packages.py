"""Manifest enrichment probes (packages / runtime / lockfiles / git / run)."""

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
    # os with no /etc/os-release resolves to blank -> omitted
    assert wa._parse_runtime('python=Python 3.12\nnode=\nos= ') == {'python': '3.12'}


def test_parse_lockfiles():
    out = (
        f'{"a" * 64}  requirements.txt\n'
        f'{"b" * 64}  uv.lock\n'
        'garbage line\n'
        f'{"c" * 64} *package.json'  # binary-mode marker stripped
    )
    assert wa._parse_lockfiles(out) == {
        'requirements.txt': 'a' * 64,
        'uv.lock': 'b' * 64,
        'package.json': 'c' * 64,
    }


def test_split_env_lockfiles():
    out = f'python=Python 3.12.4\n{wa._ENV_MARKER}\n{"a" * 64}  uv.lock'
    env, locks = wa._split_env_lockfiles(out)
    assert env == {'python': '3.12.4'}
    assert locks == {'uv.lock': 'a' * 64}


def test_parse_git_changes_numstat():
    out = f'commits=3\n{wa._NUMSTAT_MARKER}\n12\t4\tsrc/a.py\n5\t0\tsrc/b.py\n-\t-\timg.png'
    # binary file (-\t-) counts toward files_changed but not ins/dels
    assert wa._parse_git_changes(out) == {
        'commits': 3,
        'files_changed': 3,
        'insertions': 17,
        'deletions': 4,
    }


def test_parse_git_changes_no_diff():
    assert wa._parse_git_changes(f'commits=0\n{wa._NUMSTAT_MARKER}\n') == {'commits': 0}


def test_is_sha():
    assert wa._is_sha('a' * 40) and wa._is_sha('deadbeef')
    assert not wa._is_sha('') and not wa._is_sha('nothex-XYZ')


# --- probes ----------------------------------------------------------------


class _Result:
    def __init__(self, stdout: str, exit_code: int = 0):
        self.stdout = stdout
        self.exit_code = exit_code


_ENV_OUT = (
    f'python=Python 3.12.4\nnode=v20.11.0\nos=ubuntu 24.04\n{wa._ENV_MARKER}\n'
    f'{"a" * 64}  requirements.txt'
)


@pytest.mark.asyncio
async def test_probe_workspace_full():
    pip_json = json.dumps([{'name': 'requests', 'version': '2.31.0'}])
    npm_json = json.dumps({'dependencies': {'express': {'version': '4.18.2'}}})

    async def fake_exec(command, cwd, timeout):
        if 'pip list' in command:
            return _Result(pip_json)
        if 'npm ls' in command:
            return _Result(npm_json, exit_code=1)  # nonzero but valid JSON
        if wa._ENV_MARKER in command:
            return _Result(_ENV_OUT)
        return _Result('')

    with patch.object(
        wa.AsyncRemoteWorkspace, 'execute_command', side_effect=fake_exec
    ):
        result = await wa._probe_workspace('http://host', 'key', '/repo')

    assert result == {
        'packages': {'pip': {'requests': '2.31.0'}, 'npm': {'express': '4.18.2'}},
        'environment': {'python': '3.12.4', 'node': '20.11.0', 'os': 'ubuntu 24.04'},
        'lockfiles': {'requirements.txt': 'a' * 64},
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
async def test_probe_git_changes_requires_shas():
    async def fail(command, cwd, timeout):
        raise AssertionError('must not run git for non-sha refs')

    with patch.object(wa.AsyncRemoteWorkspace, 'execute_command', side_effect=fail):
        assert await wa._probe_git_changes('h', 'k', '/r', 'DETACHED', 'x') == {}


@pytest.mark.asyncio
async def test_probe_git_changes_parses():
    git_out = f'commits=2\n{wa._NUMSTAT_MARKER}\n10\t2\tf.py'
    commands = []

    async def fake_exec(command, cwd, timeout):
        commands.append(command)
        return _Result(git_out)

    with patch.object(
        wa.AsyncRemoteWorkspace, 'execute_command', side_effect=fake_exec
    ):
        gc = await wa._probe_git_changes('h', 'k', '/r', 'a' * 40, 'b' * 40)
    assert gc == {'commits': 2, 'files_changed': 1, 'insertions': 10, 'deletions': 2}
    assert f'git diff --numstat {"a" * 40} 2>/dev/null' in commands[0]


@pytest.mark.asyncio
async def test_probe_disabled_by_env(monkeypatch):
    monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_ENRICH', 'false')

    async def fail(command, cwd, timeout):
        raise AssertionError('must not probe when disabled')

    with patch.object(wa.AsyncRemoteWorkspace, 'execute_command', side_effect=fail):
        assert await wa._probe_workspace('h', 'k', '/r') == {}
        assert await wa._probe_git_changes('h', 'k', '/r', 'a' * 40, 'b' * 40) == {}


# --- run metrics -----------------------------------------------------------


def _client_returning(payload, status=200):
    class _Resp:
        status_code = status

        def json(self):
            return payload

    client = MagicMock()
    client.get = AsyncMock(return_value=_Resp())
    return client


@pytest.mark.asyncio
async def test_fetch_run_metrics_full():
    client = _client_returning(
        {
            'metrics': {
                'model_name': 'claude-opus-4-8',
                'accumulated_cost': 0.42,
                'accumulated_token_usage': {
                    'prompt_tokens': 1200,
                    'completion_tokens': 300,
                },
            },
            'execution_status': 'finished',
            'created_at': '2026-07-10T00:00:00',
            'updated_at': '2026-07-10T00:01:30',
        }
    )
    run = await wa._fetch_run_metrics(
        client, 'http://h', {}, 'c1425d4f35cc47d9804c125fe0af02aa'
    )
    assert run == {
        'model': 'claude-opus-4-8',
        'cost': 0.42,
        'prompt_tokens': 1200,
        'completion_tokens': 300,
        'status': 'finished',
        'duration_seconds': 90.0,
    }
    # hex conversation id is normalized to a hyphenated UUID for the GET
    assert client.get.call_args.args[0].endswith(
        '/api/conversations/c1425d4f-35cc-47d9-804c-125fe0af02aa'
    )


@pytest.mark.asyncio
async def test_fetch_run_metrics_skips_default_model():
    client = _client_returning(
        {'metrics': {'model_name': 'default', 'accumulated_cost': 0}}
    )
    run = await wa._fetch_run_metrics(client, 'http://h', {}, 'c1')
    assert 'model' not in run and run.get('cost') == 0


@pytest.mark.asyncio
async def test_fetch_run_metrics_non_200_and_no_id():
    assert (
        await wa._fetch_run_metrics(_client_returning({}, status=404), 'h', {}, 'c')
        == {}
    )
    assert await wa._fetch_run_metrics(_client_returning({}), 'h', {}, None) == {}
