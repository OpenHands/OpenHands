from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from seed_live_db import (
    API_KEYS,
    BYOK_BYOR_KEY,
    BYOK_LLM_KEY,
    MANAGED_REFRESH_OLD_KEY,
    MANAGED_START_OLD_KEY,
    fingerprint,
)

ROOT = Path(__file__).resolve().parents[2]
HARNESS_DIR = Path(__file__).resolve().parent
SUPPORT_DIR = HARNESS_DIR
RUNS_DIR = HARNESS_DIR / 'runs'
ENTERPRISE_PYTHON = ['poetry', '--project', 'enterprise', 'run', 'python']


def run(command: list[str], *, env: dict[str, str] | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        output = sanitize_text(exc.stdout or '')
        raise RuntimeError(
            f'command failed ({exc.returncode}): {" ".join(command)}\n{output}'
        ) from exc
    return result.stdout


def free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def wait_for(url: str, timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code < 500:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.4)
    raise RuntimeError(f'timed out waiting for {url}: {last_error}')


def safe_env(
    base: dict[str, str],
    persistence_dir: Path,
    stub_url: str,
    frontend_dir: Path,
) -> dict[str, str]:
    env = {
        'PATH': base.get('PATH', ''),
        'HOME': base.get('HOME', ''),
        'PYTHONPATH': f'{ROOT / "enterprise"}:{SUPPORT_DIR}:{ROOT}',
        'OPENHANDS_SUPPRESS_BANNER': '1',
        'OPENHANDS_CONFIG_CLS': 'server.config.SaaSServerConfig',
        'SERVE_FRONTEND': 'false',
        'FRONTEND_DIRECTORY': str(frontend_dir),
        'POSTHOG_CLIENT_KEY': 'phc-live-evidence-local',
        'OH_PERSISTENCE_DIR': str(persistence_dir),
        'DB_HOST': '',
        'GCP_DB_INSTANCE': '',
        'LITE_LLM_API_URL': stub_url,
        'LITE_LLM_API_KEY': 'litellm-admin-live-evidence',
        'OPENHANDS_PROVIDER_BASE_URL': stub_url,
        'LIVE_EVIDENCE_AGENT_URL': stub_url,
        'ENABLE_JIRA': 'false',
        'ENABLE_JIRA_DC': 'false',
        'ENABLE_LINEAR': 'false',
        'ENABLE_BILLING': 'false',
        'OH_APP_CONVERSATION_INFO_KIND': (
            'server.utils.saas_app_conversation_info_injector.'
            'SaasAppConversationInfoServiceInjector'
        ),
        'SESSION_API_KEY': '',
        'LLM_API_KEY': '',
        'OPENHANDS_API_KEY': '',
        'GITHUB_TOKEN': '',
        'DD_API_KEY': '',
        'DD_APP_KEY': '',
        'DD_SITE': '',
    }
    return env


def start_uvicorn(
    module: str,
    port: int,
    log_path: Path,
    *,
    env: dict[str, str],
) -> subprocess.Popen[str]:
    log_file = log_path.open('w')
    return subprocess.Popen(
        [
            *ENTERPRISE_PYTHON,
            '-m',
            'uvicorn',
            module,
            '--host',
            '127.0.0.1',
            '--port',
            str(port),
            '--log-level',
            'info',
        ],
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=8)


def sanitize_text(value: str) -> str:
    value = re.sub(
        r'(?:sk-live|live-llm)-[A-Za-z0-9-]+',
        '<redacted-local-llm-key>',
        value,
    )
    value = re.sub(
        r'(?:sk-oh|live-api)-[A-Za-z0-9-]+',
        '<redacted-local-api-key>',
        value,
    )
    value = value.replace(
        'litellm-admin-live-evidence', '<redacted-local-litellm-admin>'
    )
    value = value.replace('session-live-evidence', '<redacted-local-session-key>')
    value = re.sub(r'http://127\.0\.0\.1:\d+', 'http://127.0.0.1:<port>', value)
    value = re.sub(r'127\.0\.0\.1:\d+', '127.0.0.1:<port>', value)
    return value


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value)
    return value


def sanitize_file(path: Path) -> None:
    if not path.exists():
        return
    path.write_text(sanitize_text(path.read_text(errors='replace')))


def sanitize_logs(scenario_dir: Path) -> None:
    for log_path in scenario_dir.glob('*.log'):
        sanitize_file(log_path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_json(payload), indent=2, sort_keys=True) + '\n')


def parse_json_output(output: str) -> Any:
    decoder = json.JSONDecoder()
    parsed: Any = None
    for index, char in enumerate(output):
        if char not in '[{':
            continue
        try:
            value, end = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if output[index + end :].strip():
            parsed = value
            continue
        return value
    if parsed is not None:
        return parsed
    raise ValueError(
        f'no JSON payload found in command output: {sanitize_text(output)}'
    )


def poll_task(app_url: str, api_key: str, task_id: str) -> dict[str, Any]:
    headers = {'Authorization': f'Bearer {api_key}'}
    last_payload: Any = None
    for _ in range(60):
        response = httpx.get(
            f'{app_url}/api/v1/app-conversations/start-tasks',
            params=[('ids', task_id)],
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        last_payload = response.json()[0]
        if last_payload and last_payload.get('status') in {'READY', 'ERROR'}:
            return last_payload
        time.sleep(0.5)
    raise RuntimeError(f'task {task_id} did not finish: {last_payload}')


def start_conversation(
    app_url: str, api_key: str
) -> tuple[httpx.Response, dict[str, Any]]:
    response = httpx.post(
        f'{app_url}/api/v1/app-conversations',
        headers={'Authorization': f'Bearer {api_key}'},
        json={},
        timeout=20,
    )
    response.raise_for_status()
    task = response.json()
    return response, poll_task(app_url, api_key, task['id'])


def call_refresh(app_url: str, api_key: str) -> dict[str, Any]:
    response = httpx.post(
        f'{app_url}/api/keys/llm/managed/refresh',
        headers={'Authorization': f'Bearer {api_key}'},
        timeout=20,
    )
    body: Any
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return {'status_code': response.status_code, 'body': body}


def git_checkout(ref: str) -> str:
    return run(['git', 'checkout', '--detach', ref]).strip()


def current_sha() -> str:
    return run(['git', 'rev-parse', 'HEAD']).strip()


def run_ref(label: str, ref: str, run_dir: Path) -> dict[str, Any]:
    scenario_dir = run_dir / label
    scenario_dir.mkdir(parents=True, exist_ok=True)
    git_checkout(ref)
    sha = current_sha()

    stub_port = free_port()
    app_port = free_port()
    stub_url = f'http://127.0.0.1:{stub_port}'
    app_url = f'http://127.0.0.1:{app_port}'
    persistence_dir = scenario_dir / 'persistence'
    frontend_dir = scenario_dir / 'frontend'
    frontend_dir.mkdir(parents=True, exist_ok=True)
    (frontend_dir / 'index.html').write_text(
        '<!doctype html><title>live evidence</title>\n'
    )
    db_path = persistence_dir / 'openhands.db'
    seed_output_path = scenario_dir / 'seed.json'
    env = safe_env(os.environ, persistence_dir, stub_url, frontend_dir)

    run(
        [
            *ENTERPRISE_PYTHON,
            str(SUPPORT_DIR / 'seed_live_db.py'),
            str(db_path),
            stub_url,
            str(seed_output_path),
        ],
        env=env,
    )
    seed_payload = json.loads(seed_output_path.read_text())
    initial_raw_keys = seed_payload.pop('initial_litellm_keys_raw')
    write_json(seed_output_path, seed_payload)

    stub = start_uvicorn(
        'live_services:stub_app', stub_port, scenario_dir / 'stub.log', env=env
    )
    app: subprocess.Popen[str] | None = None
    try:
        wait_for(f'{stub_url}/alive')
        reset_response = httpx.post(
            f'{stub_url}/__test/reset',
            json={'initial_keys': initial_raw_keys},
            timeout=10,
        )
        reset_response.raise_for_status()
        write_json(scenario_dir / 'stub-reset.json', reset_response.json())

        app = start_uvicorn(
            'live_harness_app:app', app_port, scenario_dir / 'app.log', env=env
        )
        wait_for(f'{app_url}/saas')

        before_probe = parse_json_output(
            run(
                [
                    *ENTERPRISE_PYTHON,
                    str(SUPPORT_DIR / 'probe_live_db.py'),
                    str(db_path),
                ],
                env=env,
            )
        )
        write_json(scenario_dir / 'db-before.json', before_probe)

        observations: dict[str, Any] = {
            'label': label,
            'sha': sha,
            'stub_url_redacted': 'http://127.0.0.1:<stub-port>',
            'app_url_redacted': 'http://127.0.0.1:<app-port>',
            'key_fingerprints': {
                'managed_refresh_old': fingerprint(MANAGED_REFRESH_OLD_KEY),
                'managed_start_old': fingerprint(MANAGED_START_OLD_KEY),
                'byok_llm': fingerprint(BYOK_LLM_KEY),
            },
        }

        if label == 'main':
            httpx.post(
                f'{stub_url}/__test/delete_key',
                json={
                    'key': MANAGED_START_OLD_KEY,
                    'reason': 'simulate upstream LiteLLM deletion while DB still references key',
                },
                timeout=10,
            ).raise_for_status()
            observations['refresh_endpoint_probe'] = call_refresh(
                app_url, API_KEYS['managed_refresh']
            )
            _, task = start_conversation(app_url, API_KEYS['managed_start'])
            observations['conversation_task'] = task
        else:
            httpx.post(
                f'{stub_url}/__test/delete_key',
                json={
                    'key': MANAGED_REFRESH_OLD_KEY,
                    'reason': 'simulate upstream LiteLLM deletion before explicit refresh',
                },
                timeout=10,
            ).raise_for_status()
            observations['managed_refresh'] = call_refresh(
                app_url, API_KEYS['managed_refresh']
            )

            httpx.post(
                f'{stub_url}/__test/delete_key',
                json={
                    'key': MANAGED_START_OLD_KEY,
                    'reason': 'simulate upstream LiteLLM deletion before startup self-heal',
                },
                timeout=10,
            ).raise_for_status()
            _, task = start_conversation(app_url, API_KEYS['managed_start'])
            observations['conversation_task'] = task
            observations['byok_refresh'] = call_refresh(app_url, API_KEYS['byok'])

        after_probe = parse_json_output(
            run(
                [
                    *ENTERPRISE_PYTHON,
                    str(SUPPORT_DIR / 'probe_live_db.py'),
                    str(db_path),
                ],
                env=env,
            )
        )
        stub_state = httpx.get(f'{stub_url}/__test/state', timeout=10).json()
        observations['db_after'] = after_probe
        observations['stub_state'] = stub_state
        write_json(scenario_dir / 'observations.json', observations)
        write_json(scenario_dir / 'db-after.json', after_probe)
        write_json(scenario_dir / 'stub-state.json', stub_state)
        return observations
    finally:
        if app is not None:
            stop_process(app)
        stop_process(stub)
        sanitize_logs(scenario_dir)


def summarize(
    main: dict[str, Any], pr: dict[str, Any], run_dir: Path
) -> dict[str, Any]:
    byok_member = next(
        member
        for member in pr['db_after']['members']
        if member['user_id'] == '33333333-3333-3333-3333-333333333333'
    )
    summary = {
        'created_at': datetime.now(UTC).isoformat(),
        'main': {
            'sha': main['sha'],
            'refresh_endpoint_status': main['refresh_endpoint_probe']['status_code'],
            'conversation_status': main['conversation_task'].get('status'),
            'conversation_detail': main['conversation_task'].get('detail'),
            'agent_start_accepted': [
                call['accepted'] for call in main['stub_state']['agent_start_calls']
            ],
        },
        'pr': {
            'sha': pr['sha'],
            'managed_refresh_status': pr['managed_refresh']['status_code'],
            'conversation_status': pr['conversation_task'].get('status'),
            'byok_refresh_status': pr['byok_refresh']['status_code'],
            'byok_llm_key_unchanged': byok_member['llm_key_fingerprint']
            == fingerprint(BYOK_LLM_KEY),
            'byok_byor_key_unchanged': byok_member['byor_key_fingerprint']
            == fingerprint(BYOK_BYOR_KEY),
            'agent_start_accepted': [
                call['accepted'] for call in pr['stub_state']['agent_start_calls']
            ],
            'generated_key_metadata': [
                call['metadata'] for call in pr['stub_state']['generated_keys']
            ],
        },
        'artifact_dir': str(run_dir.relative_to(ROOT)),
    }
    write_json(run_dir / 'summary.json', summary)
    return summary


def main() -> None:
    global SUPPORT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument('--main-ref', default='origin/main')
    parser.add_argument('--pr-ref', default='origin/pr-15023')
    parser.add_argument('--restore-ref', default='managed-llm-key-refresh-15022')
    args = parser.parse_args()

    run_id = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    original_ref = run(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip()
    original_sha = current_sha()
    support_dir = Path(tempfile.mkdtemp(prefix='issue-15022-harness-'))
    for filename in (
        'live_harness_app.py',
        'live_services.py',
        'probe_live_db.py',
        'seed_live_db.py',
    ):
        shutil.copy2(HARNESS_DIR / filename, support_dir / filename)
    SUPPORT_DIR = support_dir

    write_json(
        run_dir / 'refs.json',
        {
            'main_ref': args.main_ref,
            'pr_ref': args.pr_ref,
            'restore_ref': args.restore_ref,
            'original_ref': original_ref,
            'original_sha': original_sha,
        },
    )

    try:
        main_observations = run_ref('main', args.main_ref, run_dir)
        pr_observations = run_ref('pr', args.pr_ref, run_dir)
        summary = summarize(main_observations, pr_observations, run_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        run(['git', 'checkout', args.restore_ref])
        shutil.rmtree(support_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
