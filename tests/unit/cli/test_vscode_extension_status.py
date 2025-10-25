import json
import pathlib
from unittest import mock

import pytest

from openhands.cli.vscode_extension import attempt_vscode_extension_install


def _make_fake_vsix(tmp_path):
    oh_dir = tmp_path / '.openhands'
    oh_dir.mkdir(exist_ok=True)
    vsix_path = oh_dir / 'openhands-vscode-0.0.1.vsix'
    vsix_path.write_text('dummy')

    class DummyAsFile:
        def __enter__(self):
            return vsix_path

        def __exit__(self, *args):
            pass

    return vsix_path, DummyAsFile


@pytest.fixture
def set_home(tmp_path, monkeypatch):
    monkeypatch.setattr(pathlib.Path, 'home', lambda: tmp_path)
    return tmp_path


def test_legacy_flag_stale_removed_and_install_attempted(set_home, monkeypatch):
    tmp_path = set_home
    monkeypatch.setenv('TERM_PROGRAM', 'vscode')
    oh_dir = tmp_path / '.openhands'
    oh_dir.mkdir(exist_ok=True)
    legacy_flag = oh_dir / '.vscode_extension_installed'
    legacy_flag.write_text('1')

    # Simulate editor CLI present, extension not installed, then bundled install success
    def fake_run(args, capture_output=True, text=True, check=False, **kwargs):
        if args[:2] == ['code', '--version']:
            return mock.Mock(returncode=0, stdout='1.0.0', stderr='')
        if args[:2] == ['code', '--list-extensions']:
            return mock.Mock(returncode=0, stdout='', stderr='')
        if args[:2] == ['code', '--install-extension']:
            return mock.Mock(returncode=0, stdout='', stderr='')
        raise AssertionError(f'unexpected args: {args}')

    monkeypatch.setattr('subprocess.run', fake_run)
    vsix_path, DummyAsFile = _make_fake_vsix(tmp_path)
    monkeypatch.setattr('importlib.resources.as_file', lambda p: DummyAsFile())
    monkeypatch.setattr('importlib.resources.files', lambda _: pathlib.Path(oh_dir))

    attempt_vscode_extension_install()

    assert legacy_flag.exists()
    status_path = oh_dir / '.editor_extension_status.json'
    assert status_path.exists()
    status = json.loads(status_path.read_text())
    assert 'vscode' in status
    assert status['vscode'].get('last_success')


def test_permanent_failure_when_cli_missing(set_home, monkeypatch):
    tmp_path = set_home
    monkeypatch.setenv('TERM_PROGRAM', 'vscode')

    # No editor CLI available: --version raises FileNotFoundError
    def fake_run(args, capture_output=True, text=True, check=False, **kwargs):
        raise FileNotFoundError('not found')

    monkeypatch.setattr('subprocess.run', fake_run)

    prints = []
    monkeypatch.setattr(
        'builtins.print', lambda *a, **k: prints.append(' '.join(map(str, a)))
    )

    attempt_vscode_extension_install()

    status = json.loads(
        (tmp_path / '.openhands' / '.editor_extension_status.json').read_text()
    )
    assert status['vscode']['permanent_failure'] == 'command_not_found'
    # Ensure informative message is printed
    assert any('no editor CLI found' in p for p in prints)


def test_reset_clears_permanent_failure_and_allows_retry(set_home, monkeypatch):
    tmp_path = set_home
    monkeypatch.setenv('TERM_PROGRAM', 'vscode')
    oh_dir = tmp_path / '.openhands'
    oh_dir.mkdir(exist_ok=True)
    # Seed status with permanent failure
    status_path = oh_dir / '.editor_extension_status.json'
    status_path.write_text(
        json.dumps(
            {
                'vscode': {
                    'permanent_failure': 'command_not_found',
                    'attempts': 3,
                    'last_attempt': '2025-01-01T00:00:00Z',
                }
            }
        )
    )

    # Set reset knob
    monkeypatch.setenv('OPENHANDS_RESET_VSCODE', '1')

    # Now simulate available CLI and successful bundled install
    def fake_run(args, capture_output=True, text=True, check=False, **kwargs):
        if args[:2] == ['code', '--version']:
            return mock.Mock(returncode=0, stdout='1.0.0', stderr='')
        if args[:2] == ['code', '--list-extensions']:
            return mock.Mock(returncode=0, stdout='', stderr='')
        if args[:2] == ['code', '--install-extension']:
            return mock.Mock(returncode=0, stdout='', stderr='')
        raise AssertionError(f'unexpected args: {args}')

    monkeypatch.setattr('subprocess.run', fake_run)
    vsix_path, DummyAsFile = _make_fake_vsix(tmp_path)
    monkeypatch.setattr('importlib.resources.as_file', lambda p: DummyAsFile())
    monkeypatch.setattr('importlib.resources.files', lambda _: pathlib.Path(oh_dir))

    attempt_vscode_extension_install()

    status = json.loads(status_path.read_text())
    assert status['vscode'].get('permanent_failure') in (None, '')
    assert status['vscode'].get('last_success')


def test_backoff_skip(set_home, monkeypatch):
    tmp_path = set_home
    monkeypatch.setenv('TERM_PROGRAM', 'vscode')
    oh_dir = tmp_path / '.openhands'
    oh_dir.mkdir(exist_ok=True)
    status_path = oh_dir / '.editor_extension_status.json'
    status_path.write_text(
        json.dumps({'vscode': {'attempts': 2, 'last_attempt': '2099-01-01T00:00:00Z'}})
    )

    # Editor CLI present so skip is due to backoff, not missing CLI
    def fake_run(args, capture_output=True, text=True, check=False, **kwargs):
        if args[:2] == ['code', '--version']:
            return mock.Mock(returncode=0, stdout='1.0.0', stderr='')
        # Should not call --list-extensions because backoff blocks it
        raise AssertionError(f'unexpected args: {args}')

    monkeypatch.setattr('subprocess.run', fake_run)

    prints = []
    monkeypatch.setattr(
        'builtins.print', lambda *a, **k: prints.append(' '.join(map(str, a)))
    )

    attempt_vscode_extension_install()

    # Should show backoff message with remaining time hint
    assert any('Will retry later' in p for p in prints)


def test_success_sets_last_success_and_clears_failure(set_home, monkeypatch):
    tmp_path = set_home
    monkeypatch.setenv('TERM_PROGRAM', 'vscode')

    def fake_run(args, capture_output=True, text=True, check=False, **kwargs):
        if args[:2] == ['code', '--version']:
            return mock.Mock(returncode=0, stdout='1.0.0', stderr='')
        if args[:2] == ['code', '--list-extensions']:
            return mock.Mock(returncode=0, stdout='', stderr='')
        if args[:2] == ['code', '--install-extension']:
            return mock.Mock(returncode=0, stdout='', stderr='')
        raise AssertionError(f'unexpected args: {args}')

    monkeypatch.setattr('subprocess.run', fake_run)
    vsix_path, DummyAsFile = _make_fake_vsix(tmp_path)
    oh_dir = tmp_path / '.openhands'
    monkeypatch.setattr('importlib.resources.as_file', lambda p: DummyAsFile())
    monkeypatch.setattr('importlib.resources.files', lambda _: pathlib.Path(oh_dir))

    attempt_vscode_extension_install()

    status = json.loads((oh_dir / '.editor_extension_status.json').read_text())
    assert status['vscode'].get('last_success')
    assert status['vscode'].get('permanent_failure') in (None, '')


def test_editor_variants_preference(set_home, monkeypatch):
    tmp_path = set_home
    monkeypatch.setenv('TERM_PROGRAM', 'vscode')

    # Both code and code-insiders available -> we should accept the first one queried
    # We simulate that both return 0 for --version, and we only proceed with bundled install once.
    calls = []

    def fake_run(args, capture_output=True, text=True, check=False, **kwargs):
        calls.append(args)
        if args[:2] in (['code', '--version'], ['code-insiders', '--version']):
            return mock.Mock(returncode=0, stdout='1.0', stderr='')
        if args[:2] == ['code', '--list-extensions']:
            return mock.Mock(returncode=0, stdout='', stderr='')
        if args[:2] == ['code', '--install-extension']:
            return mock.Mock(returncode=0, stdout='', stderr='')
        raise AssertionError(f'unexpected args: {args}')

    monkeypatch.setattr('subprocess.run', fake_run)
    vsix_path, DummyAsFile = _make_fake_vsix(tmp_path)
    oh_dir = tmp_path / '.openhands'
    monkeypatch.setattr('importlib.resources.as_file', lambda p: DummyAsFile())
    monkeypatch.setattr('importlib.resources.files', lambda _: pathlib.Path(oh_dir))

    attempt_vscode_extension_install()

    # Ensure --version was queried for both candidates
    assert ['code', '--version'] in calls
    assert ['code-insiders', '--version'] in calls
    # Ensure the install proceeded using 'code' (first candidate)
    assert any(c[:2] == ['code', '--install-extension'] for c in calls)
