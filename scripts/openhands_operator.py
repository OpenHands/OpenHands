#!/usr/bin/env python3
"""Prepare, validate, and start this OpenHands fork safely.

This module uses only the Python standard library so it can diagnose a fresh
checkout before OpenHands dependencies are installed.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlparse

MIN_PYTHON = (3, 12, 0)
MAX_PYTHON = (3, 14, 0)
MIN_NODE = (22, 12, 0)
MIN_POETRY = (1, 8, 0)
VALID_STATUSES = frozenset({'pass', 'warning', 'error'})
GENERIC_PROVIDER_KEYS = ('LLM_MODEL', 'LLM_API_KEY')
OPENCODE_GO_KEYS = (
    'OPENCODE_GO_MODEL',
    'OPENCODE_GO_BASE_URL',
    'OPENCODE_GO_API_KEY',
)
_VERSION_PATTERN = re.compile(r'(?<!\d)(\d+)\.(\d+)(?:\.(\d+))?')


@dataclass(frozen=True)
class CheckResult:
    """One readiness check result."""

    name: str
    status: str
    message: str

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f'Unsupported check status: {self.status}')

    def to_dict(self) -> dict[str, str]:
        return {
            'name': self.name,
            'status': self.status,
            'message': self.message,
        }


@dataclass(frozen=True)
class ReadinessReport:
    """A collection of readiness check results."""

    results: list[CheckResult]

    def _count(self, status: str) -> int:
        return sum(result.status == status for result in self.results)

    @property
    def passes(self) -> int:
        return self._count('pass')

    @property
    def warnings(self) -> int:
        return self._count('warning')

    @property
    def errors(self) -> int:
        return self._count('error')

    @property
    def ready(self) -> bool:
        return self.errors == 0


def _result(name: str, status: str, message: str) -> CheckResult:
    return CheckResult(name=name, status=status, message=message)


def parse_version(value: str) -> tuple[int, int, int] | None:
    """Extract the first semantic version-like value from command output."""

    match = _VERSION_PATTERN.search(value)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def parse_port(value: str) -> int:
    """Parse and validate a TCP port for argparse."""

    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('port must be an integer') from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError('port must be between 1 and 65535')
    return port


def _format_version(version: tuple[int, int, int]) -> str:
    return '.'.join(str(part) for part in version)


def _version_in_range(
    version: tuple[int, int, int],
    minimum: tuple[int, int, int],
    maximum: tuple[int, int, int] | None = None,
) -> bool:
    return version >= minimum and (maximum is None or version < maximum)


def _run_probe(command: Sequence[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return completed.returncode, (completed.stdout or completed.stderr).strip()


def _check_available_command(name: str, executable: str) -> CheckResult:
    if shutil.which(executable) is None:
        return _result(
            name,
            'error',
            f'{name} is not installed or is not available on PATH.',
        )
    return _result(name, 'pass', f'{name} is available.')


def _check_versioned_command(
    *,
    name: str,
    executable: str,
    arguments: Sequence[str],
    minimum: tuple[int, int, int],
    maximum: tuple[int, int, int] | None = None,
) -> CheckResult:
    available = _check_available_command(name, executable)
    if available.status == 'error':
        return available

    return_code, output = _run_probe((executable, *arguments))
    if return_code != 0:
        return _result(name, 'error', f'{name} could not be executed successfully.')

    version = parse_version(output)
    if version is None:
        return _result(
            name,
            'error',
            f'Could not determine the installed {name} version.',
        )

    if not _version_in_range(version, minimum, maximum):
        requirement = f'>={_format_version(minimum)}'
        if maximum is not None:
            requirement += f', <{_format_version(maximum)}'
        return _result(
            name,
            'error',
            f'{name} {_format_version(version)} is incompatible; required {requirement}.',
        )

    return _result(
        name,
        'pass',
        f'{name} {_format_version(version)} is compatible.',
    )


def _check_python() -> CheckResult:
    version = sys.version_info[:3]
    if not _version_in_range(version, MIN_PYTHON, MAX_PYTHON):
        return _result(
            'Python',
            'error',
            (
                f'Python {_format_version(version)} is incompatible; required '
                f'>={_format_version(MIN_PYTHON)}, <{_format_version(MAX_PYTHON)}.'
            ),
        )
    return _result(
        'Python',
        'pass',
        f'Python {_format_version(version)} is compatible.',
    )


def _check_docker() -> list[CheckResult]:
    cli = _check_available_command('Docker CLI', 'docker')
    if cli.status == 'error':
        return [
            _result(
                'Docker CLI',
                'error',
                'Docker is required for the docker runtime but is unavailable.',
            )
        ]

    return_code, _ = _run_probe(('docker', 'info'))
    daemon = (
        _result('Docker daemon', 'pass', 'Docker daemon is reachable.')
        if return_code == 0
        else _result(
            'Docker daemon',
            'error',
            (
                'Docker is installed but the daemon is not reachable. Start '
                'Docker Desktop/Engine or deliberately use --runtime local.'
            ),
        )
    )
    return [cli, daemon]


def _check_workspace(workspace: Path) -> CheckResult:
    if not workspace.exists():
        return _result(
            'Workspace',
            'warning',
            (
                f'Workspace does not exist: {workspace}. Run bootstrap or use '
                'start --bootstrap to create it.'
            ),
        )
    if not workspace.is_dir():
        return _result(
            'Workspace',
            'error',
            f'Workspace path is not a directory: {workspace}.',
        )
    if not os.access(workspace, os.W_OK | os.X_OK):
        return _result(
            'Workspace',
            'error',
            f'Workspace is not writable: {workspace}.',
        )
    return _result('Workspace', 'pass', f'Workspace is ready: {workspace}.')


def _is_loopback_host(host: str) -> bool:
    if host.strip().lower() == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _check_remote_access(
    backend_host: str,
    frontend_host: str,
    allow_remote_access: bool,
) -> CheckResult:
    remote_hosts = list(
        dict.fromkeys(
            host
            for host in (backend_host, frontend_host)
            if not _is_loopback_host(host)
        )
    )
    if not remote_hosts:
        return _result(
            'Remote access',
            'pass',
            'Backend and frontend are bound to loopback addresses.',
        )

    hosts = ', '.join(remote_hosts)
    if not allow_remote_access:
        return _result(
            'Remote access',
            'error',
            (
                f'Refusing non-loopback binding ({hosts}) without '
                '--allow-remote-access. Use TLS, access control, and a hardened '
                'reverse proxy before exposing OpenHands.'
            ),
        )
    return _result(
        'Remote access',
        'pass',
        (
            f'Non-loopback binding was explicitly acknowledged for {hosts}. '
            'Protect it with TLS, access control, and a hardened reverse proxy.'
        ),
    )


def _check_port(name: str, host: str, port: int) -> CheckResult:
    address_family = socket.AF_INET6 if ':' in host else socket.AF_INET
    sock = socket.socket(address_family, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except (OSError, OverflowError):
        return _result(
            name,
            'error',
            f'{name} port {host}:{port} is already in use or unavailable.',
        )
    finally:
        sock.close()
    return _result(name, 'pass', f'{name} port {host}:{port} is available.')


def resolve_provider_mode(env: Mapping[str, str], mode: str) -> str:
    """Resolve provider auto-detection without exposing values."""

    if mode != 'auto':
        return mode
    if any(env.get(key, '').strip() for key in OPENCODE_GO_KEYS):
        return 'opencode-go'
    generic_keys = (*GENERIC_PROVIDER_KEYS, 'LLM_BASE_URL')
    if any(env.get(key, '').strip() for key in generic_keys):
        return 'generic'
    return 'none'


def _base_url_error(value: str) -> str | None:
    parsed = urlparse(value)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        return 'must be an absolute http:// or https:// URL'
    if parsed.username is not None or parsed.password is not None:
        return 'must not contain embedded credentials'
    return None


def _missing_keys(env: Mapping[str, str], keys: Sequence[str]) -> list[str]:
    return [key for key in keys if not env.get(key, '').strip()]


def validate_provider(
    env: Mapping[str, str],
    mode: str,
    require_provider: bool,
) -> list[CheckResult]:
    """Validate provider configuration using presence checks only."""

    resolved = resolve_provider_mode(env, mode)
    if resolved == 'none':
        if mode == 'none' and not require_provider:
            return [
                _result(
                    'LLM provider',
                    'pass',
                    'Provider validation was disabled explicitly.',
                )
            ]
        return [
            _result(
                'LLM provider',
                'error' if require_provider else 'warning',
                (
                    'No environment-based LLM provider is configured. You may '
                    'configure a model in the OpenHands Settings UI after startup.'
                ),
            )
        ]

    if resolved == 'generic':
        missing = _missing_keys(env, GENERIC_PROVIDER_KEYS)
        if missing:
            return [
                _result(
                    'Generic LLM provider',
                    'error',
                    'Generic provider configuration is incomplete; missing: '
                    + ', '.join(missing)
                    + '.',
                )
            ]
        base_url = env.get('LLM_BASE_URL', '').strip()
        error = _base_url_error(base_url) if base_url else None
        if error is not None:
            return [
                _result(
                    'Generic LLM provider',
                    'error',
                    f'LLM_BASE_URL {error}.',
                )
            ]
        return [
            _result(
                'Generic LLM provider',
                'pass',
                'Generic provider model and API key are present.',
            )
        ]

    if resolved == 'opencode-go':
        missing = _missing_keys(env, OPENCODE_GO_KEYS)
        if missing:
            return [
                _result(
                    'OpenCode Go provider',
                    'error',
                    'OpenCode Go configuration is incomplete; missing: '
                    + ', '.join(missing)
                    + '.',
                )
            ]
        error = _base_url_error(env['OPENCODE_GO_BASE_URL'].strip())
        if error is not None:
            return [
                _result(
                    'OpenCode Go provider',
                    'error',
                    f'OPENCODE_GO_BASE_URL {error}.',
                )
            ]
        return [
            _result(
                'OpenCode Go provider',
                'pass',
                'OpenCode Go model, base URL, and API key are present.',
            )
        ]

    return [
        _result(
            'LLM provider',
            'error',
            f'Unsupported provider mode: {resolved}.',
        )
    ]


def build_child_environment(
    env: Mapping[str, str],
    provider_mode: str,
) -> dict[str, str]:
    """Map a complete OpenCode Go profile into OpenHands child variables."""

    child = dict(env)
    if resolve_provider_mode(env, provider_mode) != 'opencode-go':
        return child

    model = env.get('OPENCODE_GO_MODEL', '').strip()
    base_url = env.get('OPENCODE_GO_BASE_URL', '').strip()
    api_key = env.get('OPENCODE_GO_API_KEY', '').strip()
    if not (model and base_url and api_key):
        return child

    child['LLM_MODEL'] = (
        model if model.startswith('openai/') else f'openai/{model}'
    )
    child['LLM_BASE_URL'] = base_url
    child['LLM_API_KEY'] = api_key
    return child


def bootstrap_workspace(
    *,
    repo_root: Path,
    workspace: Path,
    create_config: bool,
) -> list[str]:
    """Create local operator state without overwriting existing config."""

    actions: list[str] = []
    if workspace.exists():
        if not workspace.is_dir():
            raise NotADirectoryError(f'Workspace path is not a directory: {workspace}')
        actions.append(f'Workspace already exists: {workspace}')
    else:
        workspace.mkdir(parents=True, exist_ok=False)
        actions.append(f'Created workspace: {workspace}')

    if create_config:
        template = repo_root / 'config.template.toml'
        config = repo_root / 'config.toml'
        if config.exists():
            actions.append('config.toml already exists; left unchanged.')
        elif not template.is_file():
            actions.append(
                'config.template.toml is missing; config.toml was not created.'
            )
        else:
            shutil.copyfile(template, config)
            actions.append('Created config.toml from config.template.toml.')
    return actions


def _apply_strict_mode(
    results: list[CheckResult], strict: bool
) -> list[CheckResult]:
    if not strict:
        return results
    return [
        _result(item.name, 'error', item.message)
        if item.status == 'warning'
        else item
        for item in results
    ]


def collect_readiness(
    *,
    env: Mapping[str, str],
    runtime: str,
    workspace: Path,
    backend_host: str,
    backend_port: int,
    frontend_host: str,
    frontend_port: int,
    provider_mode: str,
    require_provider: bool,
    strict: bool,
    skip_system_checks: bool,
    skip_port_checks: bool,
    allow_remote_access: bool,
) -> ReadinessReport:
    results: list[CheckResult] = []
    if not skip_system_checks:
        results.extend(
            [
                _check_python(),
                _check_versioned_command(
                    name='Node.js',
                    executable='node',
                    arguments=('--version',),
                    minimum=MIN_NODE,
                ),
                _check_available_command('npm', 'npm'),
                _check_versioned_command(
                    name='Poetry',
                    executable='poetry',
                    arguments=('--version',),
                    minimum=MIN_POETRY,
                ),
                _check_available_command('Git', 'git'),
                _check_available_command('make', 'make'),
                _check_available_command('netcat (nc)', 'nc'),
            ]
        )
        if runtime == 'docker':
            results.extend(_check_docker())
        else:
            results.append(
                _result(
                    'Runtime isolation',
                    'warning',
                    (
                        'Local runtime is selected. Agents may access the host '
                        'filesystem; use only in a trusted environment.'
                    ),
                )
            )

    results.extend(
        [
            _check_workspace(workspace),
            _check_remote_access(
                backend_host,
                frontend_host,
                allow_remote_access,
            ),
        ]
    )
    if not skip_port_checks:
        results.extend(
            [
                _check_port('Backend', backend_host, backend_port),
                _check_port('Frontend', frontend_host, frontend_port),
            ]
        )
    results.extend(validate_provider(env, provider_mode, require_provider))
    return ReadinessReport(_apply_strict_mode(results, strict))


def render_report(report: ReadinessReport, as_json: bool) -> str:
    """Render a deterministic report that never includes provider values."""

    if as_json:
        return json.dumps(
            {
                'ready': report.ready,
                'summary': {
                    'passes': report.passes,
                    'warnings': report.warnings,
                    'errors': report.errors,
                },
                'results': [result.to_dict() for result in report.results],
            },
            indent=2,
            sort_keys=True,
        )

    lines = [
        'OpenHands operator readiness: '
        + ('READY' if report.ready else 'BLOCKED')
    ]
    labels = {'pass': 'PASS', 'warning': 'WARN', 'error': 'ERROR'}
    lines.extend(
        f'[{labels[item.status]}] {item.name}: {item.message}'
        for item in report.results
    )
    lines.append(
        'Summary: '
        f'{report.passes} passed, {report.warnings} warnings, '
        f'{report.errors} errors.'
    )
    return '\n'.join(lines)


def execute_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
) -> int:
    """Execute a build or launch command and return its exit status."""

    try:
        return subprocess.call(list(command), cwd=cwd, env=dict(env))
    except OSError as exc:
        print(f'Unable to execute {command[0]}: {exc}', file=sys.stderr)
        return 127


def _resolve_workspace(repo_root: Path, value: str) -> Path:
    workspace = Path(value).expanduser()
    return (
        workspace.resolve()
        if workspace.is_absolute()
        else (repo_root / workspace).resolve()
    )


def _safe_launch_description(
    command: Sequence[str],
    child_env: Mapping[str, str],
    provider_mode: str,
) -> str:
    visible = [
        f"RUNTIME={child_env.get('RUNTIME', '')}",
        f"WORKSPACE_BASE={child_env.get('WORKSPACE_BASE', '')}",
    ]
    if 'INSTALL_DOCKER' in child_env:
        visible.append(f"INSTALL_DOCKER={child_env['INSTALL_DOCKER']}")
    if resolve_provider_mode(child_env, provider_mode) in {'generic', 'opencode-go'}:
        visible.extend(
            [
                f"LLM_MODEL={child_env.get('LLM_MODEL', '<unset>')}",
                'LLM_BASE_URL=<set>'
                if child_env.get('LLM_BASE_URL')
                else 'LLM_BASE_URL=<unset>',
                'LLM_API_KEY=<set>'
                if child_env.get('LLM_API_KEY')
                else 'LLM_API_KEY=<unset>',
            ]
        )
    return '\n'.join(
        [
            'Launch command: ' + ' '.join(command),
            'Environment: ' + ' '.join(visible),
        ]
    )


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--runtime',
        choices=('docker', 'local'),
        default=os.environ.get('OPENHANDS_RUNTIME', 'docker'),
        help='Agent runtime to validate and launch (default: docker).',
    )
    parser.add_argument(
        '--provider',
        choices=('auto', 'generic', 'opencode-go', 'none'),
        default=os.environ.get('OPENHANDS_PROVIDER', 'auto'),
        help='Environment provider profile to validate (default: auto).',
    )
    parser.add_argument(
        '--workspace',
        default=os.environ.get('OPENHANDS_WORKSPACE', 'workspace'),
        help='Workspace directory, relative to the repository unless absolute.',
    )
    parser.add_argument(
        '--backend-host',
        default=os.environ.get('BACKEND_HOST', '127.0.0.1'),
    )
    parser.add_argument(
        '--backend-port',
        type=parse_port,
        default=os.environ.get('BACKEND_PORT', '3000'),
    )
    parser.add_argument(
        '--frontend-host',
        default=os.environ.get('FRONTEND_HOST', '127.0.0.1'),
    )
    parser.add_argument(
        '--frontend-port',
        type=parse_port,
        default=os.environ.get('FRONTEND_PORT', '3001'),
    )
    parser.add_argument(
        '--allow-remote-access',
        action='store_true',
        help='Acknowledge non-loopback binding behind independent security.',
    )
    parser.add_argument(
        '--require-provider',
        action='store_true',
        help='Fail when no environment-based provider is configured.',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat every warning as an error.',
    )
    parser.add_argument(
        '--skip-system-checks',
        action='store_true',
        help='Skip executable/runtime probes (for focused CI or tests).',
    )
    parser.add_argument(
        '--skip-port-checks',
        action='store_true',
        help='Skip backend and frontend port availability checks.',
    )
    parser.add_argument('--json', action='store_true', help='Print JSON output.')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Prepare, validate, and start the OpenHands fork safely.'
    )
    commands = parser.add_subparsers(dest='command', required=True)

    doctor = commands.add_parser('doctor', help='Check operational readiness.')
    _add_common_options(doctor)

    bootstrap = commands.add_parser(
        'bootstrap',
        help='Create the workspace and optional local config.',
    )
    bootstrap.add_argument(
        '--workspace',
        default=os.environ.get('OPENHANDS_WORKSPACE', 'workspace'),
    )
    bootstrap.add_argument(
        '--create-config',
        action='store_true',
        help='Copy config.template.toml when config.toml is absent.',
    )
    bootstrap.add_argument('--json', action='store_true')

    start = commands.add_parser(
        'start',
        help='Validate and launch using existing make targets.',
    )
    _add_common_options(start)
    start.add_argument(
        '--bootstrap',
        action='store_true',
        help='Create the workspace before validation.',
    )
    start.add_argument(
        '--create-config',
        action='store_true',
        help='With --bootstrap, copy the config template when absent.',
    )
    start.add_argument('--build', action='store_true', help='Run make build first.')
    start.add_argument(
        '--dry-run',
        action='store_true',
        help='Print sanitized actions without changing files or launching.',
    )
    return parser


def _run_bootstrap(
    *,
    repo_root: Path,
    workspace: Path,
    create_config: bool,
    as_json: bool,
) -> int:
    try:
        actions = bootstrap_workspace(
            repo_root=repo_root,
            workspace=workspace,
            create_config=create_config,
        )
    except OSError as exc:
        if as_json:
            print(json.dumps({'ok': False, 'error': str(exc)}, indent=2))
        else:
            print(f'Bootstrap failed: {exc}', file=sys.stderr)
        return 1

    print(
        json.dumps({'ok': True, 'actions': actions}, indent=2)
        if as_json
        else '\n'.join(actions)
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    workspace = _resolve_workspace(repo_root, args.workspace)

    if args.command == 'bootstrap':
        return _run_bootstrap(
            repo_root=repo_root,
            workspace=workspace,
            create_config=args.create_config,
            as_json=args.json,
        )

    if args.command == 'start' and args.create_config and not args.bootstrap:
        parser.error('--create-config requires --bootstrap')

    if args.command == 'start' and args.bootstrap:
        if args.dry_run:
            if not args.json:
                print(f'Bootstrap preview: would ensure workspace {workspace}.')
                if args.create_config:
                    print(
                        'Bootstrap preview: would copy config.template.toml only '
                        'when config.toml is absent.'
                    )
        else:
            bootstrap_exit = _run_bootstrap(
                repo_root=repo_root,
                workspace=workspace,
                create_config=args.create_config,
                as_json=False,
            )
            if bootstrap_exit != 0:
                return bootstrap_exit

    report = collect_readiness(
        env=os.environ,
        runtime=args.runtime,
        workspace=workspace,
        backend_host=args.backend_host,
        backend_port=args.backend_port,
        frontend_host=args.frontend_host,
        frontend_port=args.frontend_port,
        provider_mode=args.provider,
        require_provider=args.require_provider,
        strict=args.strict,
        skip_system_checks=args.skip_system_checks,
        skip_port_checks=args.skip_port_checks,
        allow_remote_access=args.allow_remote_access,
    )
    print(render_report(report, as_json=args.json))
    if not report.ready or args.command == 'doctor':
        return 0 if report.ready else 1

    child_env = build_child_environment(os.environ, args.provider)
    child_env['RUNTIME'] = args.runtime
    child_env['WORKSPACE_BASE'] = str(workspace)
    if args.runtime == 'local':
        child_env['INSTALL_DOCKER'] = '0'

    build_command = ('make', 'build')
    launch_command = (
        'make',
        'run',
        f'BACKEND_HOST={args.backend_host}',
        f'BACKEND_PORT={args.backend_port}',
        f'FRONTEND_HOST={args.frontend_host}',
        f'FRONTEND_PORT={args.frontend_port}',
    )
    if args.dry_run:
        if args.build:
            print('Build command: ' + ' '.join(build_command))
        print(_safe_launch_description(launch_command, child_env, args.provider))
        return 0

    if args.build:
        build_exit = execute_command(build_command, cwd=repo_root, env=child_env)
        if build_exit != 0:
            return build_exit
    return execute_command(launch_command, cwd=repo_root, env=child_env)


if __name__ == '__main__':
    raise SystemExit(main())
