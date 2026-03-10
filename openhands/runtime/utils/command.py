# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import os

from openhands.core.config import OpenHandsConfig
from openhands.core.logger import openhands_logger as logger
from openhands.runtime.plugins import PluginRequirement

DEFAULT_PYTHON_PREFIX = [
    '/openhands/micromamba/bin/micromamba',
    'run',
    '-n',
    'openhands',
    'poetry',
    'run',
]
DEFAULT_MAIN_MODULE = 'openhands.runtime.action_execution_server'

RUNTIME_USERNAME = os.getenv('RUNTIME_USERNAME')
RUNTIME_UID = os.getenv('RUNTIME_UID')


def get_action_execution_server_startup_command(
    server_port: int,
    plugins: list[PluginRequirement],
    app_config: OpenHandsConfig,
    python_prefix: list[str] = DEFAULT_PYTHON_PREFIX,
    override_user_id: int | None = None,
    override_username: str | None = None,
    main_module: str = DEFAULT_MAIN_MODULE,
    python_executable: str = 'python',
) -> list[str]:
    sandbox_config = app_config.sandbox
    logger.debug(f'app_config {vars(app_config)}')
    logger.debug(f'sandbox_config {vars(sandbox_config)}')
    logger.debug(f'RUNTIME_USERNAME {RUNTIME_USERNAME}, RUNTIME_UID {RUNTIME_UID}')
    logger.debug(
        f'override_username {override_username}, override_user_id {override_user_id}'
    )

    # Plugin args
    plugin_args = []
    if plugins is not None and len(plugins) > 0:
        plugin_args = ['--plugins'] + [plugin.name for plugin in plugins]

    # Browsergym stuffs
    browsergym_args = []
    if sandbox_config.browsergym_eval_env is not None:
        browsergym_args = [
            '--browsergym-eval-env'
        ] + sandbox_config.browsergym_eval_env.split(' ')

    username = (
        override_username
        or RUNTIME_USERNAME
        or ('openhands' if app_config.run_as_openhands else 'root')
    )
    user_id = (
        override_user_id or RUNTIME_UID or (1000 if app_config.run_as_openhands else 0)
    )
    logger.debug(f'username {username}, user_id {user_id}')

    base_cmd = [
        *python_prefix,
        python_executable,
        '-u',
        '-m',
        main_module,
        str(server_port),
        '--working-dir',
        app_config.workspace_mount_path_in_sandbox,
        *plugin_args,
        '--username',
        username,
        '--user-id',
        str(user_id),
        *browsergym_args,
    ]

    if not app_config.enable_browser:
        base_cmd.append('--no-enable-browser')
    logger.debug(f'get_action_execution_server_startup_command: {base_cmd}')

    return base_cmd
