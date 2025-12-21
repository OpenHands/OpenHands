"""Run many Fleet tasks under a single Fleet trace job using OpenHands locally.

This is an *outer harness*:
- Creates one Fleet trace job (so all tasks show up together in the dashboard)
- Loads tasks from Fleet
- For each task:
  - Creates a Fleet environment instance (remote)
  - Runs an OpenHands session locally against that env via MCP tools
  - Runs the Fleet verifier (if present)
  - Completes/fails the Fleet session with verifier_execution_id

Notes:
- Requires optional deps: `fleet-python` (install via `poetry install --with fleet`)
- Intended for headless/eval use (no GUI / no signal handlers).
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import os
from typing import Any

from openhands.core.config import OpenHandsConfig
from openhands.core.config.utils import load_from_toml
from openhands.core.config.sandbox_config import SandboxConfig
from openhands.core.loop import run_agent_until_done
from openhands.core.setup import create_agent, create_controller, create_memory, create_runtime
from openhands.events import EventSource, EventStreamSubscriber
from openhands.events.action import MessageAction
from openhands.events.observation import AgentStateChangedObservation
from openhands.core.schema import AgentState
from openhands.llm.llm_registry import LLMRegistry
from openhands.mcp.utils import add_mcp_tools_to_agent
from openhands.server.services.conversation_stats import ConversationStats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run Fleet tasks under one Fleet trace job.')
    p.add_argument('--config', required=True, help='Path to OpenHands config.toml')
    p.add_argument('--project-key', default=None, help='Fleet project key to load tasks from')
    p.add_argument('--env-key', default=None, help='Fleet environment key to load tasks from')
    p.add_argument(
        '--task-keys',
        default=None,
        help='Comma-separated Fleet task keys to run (overrides --project-key)',
    )
    p.add_argument('--job-name', default='openhands-fleet-run', help='Fleet trace job name')
    p.add_argument('--max-concurrent', type=int, default=4, help='Max concurrent tasks')
    p.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help='Optional OpenHands max_iterations override for each task',
    )
    return p.parse_args()

def _require_valid_sandbox_section(config_path: str) -> None:
    """Fail-fast on invalid [sandbox] config.

    `load_from_toml()` intentionally logs-and-continues on validation errors.
    For the eval harness, we want incorrect config to fail at load time with the real reason.
    """
    import tomllib
    from pydantic import ValidationError

    with open(config_path, 'rb') as f:
        data = tomllib.load(f)
    sandbox = data.get('sandbox', None)
    if sandbox is None:
        return
    if not isinstance(sandbox, dict):
        raise ValueError(f'Invalid [sandbox] section in {config_path}: expected a table/dict.')
    try:
        # Validate strictly; this will raise if keys/types are wrong.
        SandboxConfig.from_toml_section(sandbox)
    except ValidationError as e:
        raise ValueError(f'Invalid [sandbox] section in {config_path}: {e}') from e
    except Exception as e:  # noqa: BLE001
        raise ValueError(f'Invalid [sandbox] section in {config_path}: {e}') from e


async def _run_one_task(
    *,
    base_config: OpenHandsConfig,
    job_id: str,
    task: Any,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    try:
        from openenv.fleet import FleetMCPTools  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "This runner requires OpenEnv Fleet support. Install: pip install 'openenv[fleet]' "
            "(or `poetry install --with fleet`)."
        ) from e

    async with semaphore:
        cfg = copy.deepcopy(base_config)
        cfg.fleet_session_export_enabled = True
        cfg.fleet_session_export_job_id = job_id
        cfg.fleet_session_export_task_key = getattr(task, 'key', None) or 'task'

        fleet_api_key = cfg.sandbox.fleet_api_key or os.getenv('FLEET_API_KEY')
        if not fleet_api_key:
            raise ValueError('Fleet API key is required (set [sandbox].fleet_api_key or FLEET_API_KEY)')
        # Some fleet-python code paths rely on env var lookup rather than global configure().
        os.environ.setdefault('FLEET_API_KEY', fleet_api_key)

        # Create env instance for this task (includes env_key + data_key + env_variables).
        env = await task.make(image_type='mcp')
        cfg.fleet_session_export_instance_id = getattr(env, 'instance_id', None)

        # OpenHands runtime/session id: use task key for determinism (trim to keep it short).
        sid = str(getattr(task, 'key', 'task'))[:32]

        llm_registry = LLMRegistry(cfg)
        agent = create_agent(cfg, llm_registry)
        runtime = create_runtime(cfg, llm_registry, sid=sid, headless_mode=True, agent=agent)

        # Attach MCP tools for this already-provisioned env (no provisioning in runtime.connect()).
        root = getattr(getattr(env, 'urls', None), 'root', None)
        if not isinstance(root, str) or not root:
            raise RuntimeError('Fleet env is missing urls.root')

        runtime.orch = env
        runtime.tools = FleetMCPTools(
            api_key=fleet_api_key,
            mcp_urls=(f'{root}api/v1/mcp', f'{root}mcp'),
        )
        await env.reset()

        # Fleet MCP endpoints can be temporarily unavailable right after provisioning.
        # OpenEnv's FleetMCPTools will union endpoints; if it returns 0 tools, retry briefly.
        tools_action = None
        for attempt in range(6):
            tools_action = await runtime.tools.list_tools()
            if getattr(tools_action, 'tools', None):
                break
            await asyncio.sleep(0.75 * (2**attempt) if attempt < 3 else 3.0)
        if not tools_action or not getattr(tools_action, 'tools', None):
            raise RuntimeError('No MCP tools discovered for Fleet env (endpoints may be down / still starting).')

        runtime.available_tools = tools_action.tools
        runtime._runtime_initialized = True  # type: ignore[attr-defined]

        # Memory + tools
        event_stream = runtime.event_stream
        memory = create_memory(
            runtime=runtime,
            event_stream=event_stream,
            sid=sid,
            selected_repository=cfg.sandbox.selected_repo,
            repo_directory=None,
            conversation_instructions=None,
            working_dir=str(runtime.workspace_root),
        )
        if agent.config.enable_mcp:
            await add_mcp_tools_to_agent(agent, runtime, memory)

        # Controller
        conversation_stats = ConversationStats(event_stream.file_store, event_stream.sid, None)
        controller, _ = create_controller(agent, runtime, cfg, conversation_stats, headless_mode=True)

        # Auto-continue if agent requests user input.
        def _on_event(ev):
            if isinstance(ev, AgentStateChangedObservation) and ev.agent_state == AgentState.AWAITING_USER_INPUT:
                event_stream.add_event(MessageAction(content='Please continue.'), EventSource.USER)

        event_stream.subscribe(EventStreamSubscriber.MAIN, _on_event, f'fleet_job_runner:{sid}')

        # Kick off task
        event_stream.add_event(MessageAction(content=getattr(task, 'prompt', '')), EventSource.USER)

        end_states = [
            AgentState.FINISHED,
            AgentState.REJECTED,
            AgentState.ERROR,
            AgentState.PAUSED,
            AgentState.STOPPED,
        ]
        await run_agent_until_done(controller, runtime, memory, end_states)

        state = controller.get_state()
        final_answer = ''
        try:
            # Prefer CodeAct FinishTool message (final_thought) if present.
            for ev in reversed(state.history):
                if hasattr(ev, 'action') and getattr(ev, 'action', None) == 'finish':
                    final_answer = getattr(ev, 'final_thought', '') or ''
                    break
        except Exception:
            final_answer = ''

        # Verify + complete/fail Fleet session
        verification_success = None
        verifier_execution_id = None
        try:
            if getattr(task, 'verifier', None) and state.agent_state == AgentState.FINISHED:
                v = await task.verify_detailed_async(env=env, final_answer=final_answer)
                verification_success = bool(getattr(v, 'success', False))
                verifier_execution_id = getattr(v, 'execution_id', None)
        except Exception:
            verification_success = False

        exporter = getattr(agent, 'fleet_session_exporter', None)
        if exporter is not None and getattr(exporter, 'enabled', False):
            if verification_success is False:
                exporter.fail(verifier_execution_id=verifier_execution_id)
            else:
                # Default: complete even if no verifier exists (or verifier success is None).
                exporter.complete(verifier_execution_id=verifier_execution_id)

        # Cleanup env
        try:
            await env.close()
        except Exception:
            pass

        return {
            'task_key': getattr(task, 'key', None),
            'session_id': getattr(exporter, 'session_id', None) if exporter else None,
            'agent_state': str(state.agent_state),
            'verification_success': verification_success,
            'verifier_execution_id': verifier_execution_id,
        }


async def main() -> int:
    args = _parse_args()

    # Optional dep: fleet-python (task loading + verifier execution).
    try:
        import fleet  # type: ignore[import-not-found]
        from fleet._async import load_tasks  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        raise ImportError('This runner requires `fleet-python`. Install: `pip install fleet-python`') from e

    base_cfg = OpenHandsConfig()
    load_from_toml(base_cfg, args.config)
    _require_valid_sandbox_section(args.config)

    if args.max_steps is not None:
        base_cfg.max_iterations = int(args.max_steps)

    # Configure fleet-python client (required for job + task APIs).
    fleet_api_key = base_cfg.sandbox.fleet_api_key or os.getenv('FLEET_API_KEY')
    if not fleet_api_key:
        raise ValueError(
            f'Fleet API key is required. Checked: [sandbox].fleet_api_key in {args.config} and env FLEET_API_KEY.'
        )
    # Some fleet-python code paths rely on env var lookup rather than global configure().
    os.environ.setdefault('FLEET_API_KEY', fleet_api_key)
    try:
        fleet.configure(
            api_key=fleet_api_key,
            base_url=base_cfg.fleet_session_export_base_url or os.getenv('FLEET_BASE_URL'),
        )
    except Exception:
        # Older fleet-python versions may not accept base_url. Configure at least api_key.
        fleet.configure(api_key=fleet_api_key)

    # One job for all tasks.
    job_id = await fleet.job_async(name=args.job_name)

    if args.task_keys:
        keys = [k.strip() for k in args.task_keys.split(',') if k.strip()]
        tasks = await load_tasks(keys=keys)
    elif args.env_key:
        tasks = await load_tasks(env_key=args.env_key)
    else:
        if not args.project_key:
            raise ValueError('Provide --project-key or --env-key or --task-keys')
        tasks = await load_tasks(project_key=args.project_key)

    sem = asyncio.Semaphore(max(1, int(args.max_concurrent)))
    results = await asyncio.gather(
        *[
            _run_one_task(base_config=base_cfg, job_id=job_id, task=t, semaphore=sem)
            for t in tasks
        ],
        return_exceptions=True,
    )

    # Print a small summary (machine-friendly).
    ok = 0
    for r in results:
        if isinstance(r, Exception):
            print({'error': str(r)})
        else:
            print(r)
            ok += 1

    print({'job_id': job_id, 'tasks_total': len(tasks), 'tasks_ran': ok})
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))


