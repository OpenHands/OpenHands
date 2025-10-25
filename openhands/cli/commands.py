import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import tomlkit
from prompt_toolkit import HTML, print_formatted_text
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import clear, print_container
from prompt_toolkit.widgets import Frame, TextArea
from pydantic import ValidationError

from openhands.cli.settings import (
    display_settings,
    modify_llm_settings_advanced,
    modify_llm_settings_basic,
    modify_search_api_settings,
)
from openhands.cli.tui import (
    COLOR_GREY,
    UsageMetrics,
    cli_confirm,
    create_prompt_session,
    display_help,
    display_mcp_errors,
    display_shutdown_message,
    display_status,
    read_prompt_input,
)
from openhands.cli.utils import (
    add_local_config_trusted_dir,
    get_local_config_trusted_dirs,
    read_file,
    write_to_file,
)
from openhands.core.config import (
    OpenHandsConfig,
)
from openhands.core.config.mcp_config import (
    MCPSHTTPServerConfig,
    MCPSSEServerConfig,
    MCPStdioServerConfig,
)
from openhands.core.schema import AgentState
from openhands.core.schema.exit_reason import ExitReason
from openhands.events import EventSource
from openhands.events.action import (
    ChangeAgentStateAction,
    LoopRecoveryAction,
    MessageAction,
)
from openhands.events.stream import EventStream
from openhands.storage.settings.file_settings_store import FileSettingsStore


async def collect_input(config: OpenHandsConfig, prompt_text: str) -> str | None:
    """Collect user input with cancellation support.

    Args:
        config: OpenHands configuration
        prompt_text: Text to display to user

    Returns:
        str | None: User input string, or None if user cancelled
    """
    print_formatted_text(prompt_text, end=' ')
    user_input = await read_prompt_input(config, '', multiline=False)

    # Check for cancellation
    if user_input.strip().lower() in ['/exit', '/cancel', 'cancel']:
        return None

    return user_input.strip()


def restart_cli() -> None:
    """Restart the CLI by replacing the current process."""
    print_formatted_text('🔄 Restarting OpenHands CLI...')

    # Get the current Python executable and script arguments
    python_executable = sys.executable
    script_args = sys.argv

    # Use os.execv to replace the current process
    # This preserves the original command line arguments
    try:
        os.execv(python_executable, [python_executable] + script_args)
    except Exception as e:
        print_formatted_text(f'❌ Failed to restart CLI: {e}')
        print_formatted_text(
            'Please restart OpenHands manually for changes to take effect.'
        )


async def prompt_for_restart(config: OpenHandsConfig) -> bool:
    """Prompt user if they want to restart the CLI and return their choice."""
    print_formatted_text('📝 MCP server configuration updated successfully!')
    print_formatted_text('The changes will take effect after restarting OpenHands.')

    prompt_session = create_prompt_session(config)

    while True:
        try:
            with patch_stdout():
                response = await prompt_session.prompt_async(
                    HTML(
                        '<gold>Would you like to restart OpenHands now? (y/n): </gold>'
                    )
                )
                response = response.strip().lower() if response else ''

                if response in ['y', 'yes']:
                    return True
                elif response in ['n', 'no']:
                    return False
                else:
                    print_formatted_text('Please enter "y" for yes or "n" for no.')
        except (KeyboardInterrupt, EOFError):
            return False


async def handle_commands(
    command: str,
    event_stream: EventStream,
    usage_metrics: UsageMetrics,
    sid: str,
    config: OpenHandsConfig,
    current_dir: str,
    settings_store: FileSettingsStore,
    agent_state: str,
) -> tuple[bool, bool, bool, ExitReason]:
    close_repl = False
    reload_microagents = False
    new_session_requested = False
    exit_reason = ExitReason.ERROR

    if command == '/exit':
        close_repl = handle_exit_command(
            config,
            event_stream,
            usage_metrics,
            sid,
        )
        if close_repl:
            exit_reason = ExitReason.INTENTIONAL
    elif command == '/help':
        handle_help_command()
    elif command == '/init':
        close_repl, reload_microagents = await handle_init_command(
            config, event_stream, current_dir
        )
    elif command == '/status':
        handle_status_command(usage_metrics, sid)
    elif command == '/new':
        close_repl, new_session_requested = handle_new_command(
            config, event_stream, usage_metrics, sid
        )
        if close_repl:
            exit_reason = ExitReason.INTENTIONAL
    elif command == '/settings':
        await handle_settings_command(config, settings_store)
    elif command.startswith('/resume'):
        close_repl, new_session_requested = await handle_resume_command(
            command, event_stream, agent_state
        )
    elif command == '/mcp':
        await handle_mcp_command(config)
    else:
        close_repl = True
        action = MessageAction(content=command)
        event_stream.add_event(action, EventSource.USER)

    return close_repl, reload_microagents, new_session_requested, exit_reason


def handle_exit_command(
    config: OpenHandsConfig,
    event_stream: EventStream,
    usage_metrics: UsageMetrics,
    sid: str,
) -> bool:
    close_repl = False

    confirm_exit = (
        cli_confirm(config, '\nTerminate session?', ['Yes, proceed', 'No, dismiss'])
        == 0
    )

    if confirm_exit:
        event_stream.add_event(
            ChangeAgentStateAction(AgentState.STOPPED),
            EventSource.ENVIRONMENT,
        )
        display_shutdown_message(usage_metrics, sid)
        close_repl = True

    return close_repl


def handle_help_command() -> None:
    display_help()


async def handle_init_command(
    config: OpenHandsConfig, event_stream: EventStream, current_dir: str
) -> tuple[bool, bool]:
    REPO_MD_CREATE_PROMPT = """
        Please explore this repository. Create the file .openhands/microagents/repo.md with:
            - A description of the project
            - An overview of the file structure
            - Any information on how to run tests or other relevant commands
            - Any other information that would be helpful to a brand new developer
        Keep it short--just a few paragraphs will do.
    """
    close_repl = False
    reload_microagents = False

    if config.runtime in ('local', 'cli'):
        init_repo = await init_repository(config, current_dir)
        if init_repo:
            event_stream.add_event(
                MessageAction(content=REPO_MD_CREATE_PROMPT),
                EventSource.USER,
            )
            reload_microagents = True
            close_repl = True
    else:
        print_formatted_text(
            '\nRepository initialization through the CLI is only supported for CLI and local runtimes.\n'
        )

    return close_repl, reload_microagents


def handle_status_command(usage_metrics: UsageMetrics, sid: str) -> None:
    display_status(usage_metrics, sid)


def handle_new_command(
    config: OpenHandsConfig,
    event_stream: EventStream,
    usage_metrics: UsageMetrics,
    sid: str,
) -> tuple[bool, bool]:
    close_repl = False
    new_session_requested = False

    new_session_requested = (
        cli_confirm(
            config,
            '\nCurrent session will be terminated and you will lose the conversation history.\n\nContinue?',
            ['Yes, proceed', 'No, dismiss'],
        )
        == 0
    )

    if new_session_requested:
        close_repl = True
        new_session_requested = True
        event_stream.add_event(
            ChangeAgentStateAction(AgentState.STOPPED),
            EventSource.ENVIRONMENT,
        )
        # For /new command, we skip the shutdown message to provide a cleaner transition
        # The user explicitly requested a new session, so showing old stats is not helpful
        from prompt_toolkit.shortcuts import clear

        clear()

    return close_repl, new_session_requested


async def handle_settings_command(
    config: OpenHandsConfig,
    settings_store: FileSettingsStore,
) -> None:
    display_settings(config)
    modify_settings = cli_confirm(
        config,
        '\nWhich settings would you like to modify?',
        [
            'LLM (Basic)',
            'LLM (Advanced)',
            'Search API (Optional)',
            'Go back',
        ],
    )

    if modify_settings == 0:
        await modify_llm_settings_basic(config, settings_store)
    elif modify_settings == 1:
        await modify_llm_settings_advanced(config, settings_store)
    elif modify_settings == 2:
        await modify_search_api_settings(config, settings_store)


# FIXME: Currently there's an issue with the actual 'resume' behavior.
# Setting the agent state to RUNNING will currently freeze the agent without continuing with the rest of the task.
# This is a workaround to handle the resume command for the time being. Replace user message with the state change event once the issue is fixed.
async def handle_resume_command(
    command: str,
    event_stream: EventStream,
    agent_state: str,
) -> tuple[bool, bool]:
    close_repl = True
    new_session_requested = False

    if agent_state != AgentState.PAUSED:
        close_repl = False
        print_formatted_text(
            HTML(
                '<ansired>Error: Agent is not paused. /resume command is only available when agent is paused.</ansired>'
            )
        )
        return close_repl, new_session_requested

    # Check if this is a loop recovery resume with an option
    if command.strip() != '/resume':
        # Parse the option from the command (e.g., '/resume 1', '/resume 2')
        parts = command.strip().split()
        if len(parts) == 2 and parts[1] in ['1', '2']:
            option = parts[1]
            # Send the option as a message to be handled by the controller
            event_stream.add_event(
                LoopRecoveryAction(option=int(option)),
                EventSource.USER,
            )
        else:
            # Invalid format, send as regular resume
            event_stream.add_event(
                MessageAction(content='continue'),
                EventSource.USER,
            )
    else:
        # Regular resume without loop recovery option
        event_stream.add_event(
            MessageAction(content='continue'),
            EventSource.USER,
        )

    # event_stream.add_event(
    #     ChangeAgentStateAction(AgentState.RUNNING),
    #     EventSource.ENVIRONMENT,
    # )

    return close_repl, new_session_requested


async def init_repository(config: OpenHandsConfig, current_dir: str) -> bool:
    repo_file_path = Path(current_dir) / '.openhands' / 'microagents' / 'repo.md'
    init_repo = False

    if repo_file_path.exists():
        try:
            # Path.exists() ensures repo_file_path is not None, so we can safely pass it to read_file
            content = await asyncio.get_event_loop().run_in_executor(
                None, read_file, repo_file_path
            )

            print_formatted_text(
                'Repository instructions file (repo.md) already exists.\n'
            )

            container = Frame(
                TextArea(
                    text=content,
                    read_only=True,
                    style=COLOR_GREY,
                    wrap_lines=True,
                ),
                title='Repository Instructions (repo.md)',
                style=f'fg:{COLOR_GREY}',
            )
            print_container(container)
            print_formatted_text('')  # Add a newline after the frame

            init_repo = (
                cli_confirm(
                    config,
                    'Do you want to re-initialize?',
                    ['Yes, re-initialize', 'No, dismiss'],
                )
                == 0
            )

            if init_repo:
                write_to_file(repo_file_path, '')
        except Exception:
            print_formatted_text('Error reading repository instructions file (repo.md)')
            init_repo = False
    else:
        print_formatted_text(
            '\nRepository instructions file will be created by exploring the repository.\n'
        )

        init_repo = (
            cli_confirm(
                config,
                'Do you want to proceed?',
                ['Yes, create', 'No, dismiss'],
            )
            == 0
        )

    return init_repo


def check_folder_security_agreement(config: OpenHandsConfig, current_dir: str) -> bool:
    # Directories trusted by user for the CLI to use as workspace
    # Config from ~/.openhands/config.toml overrides the app config

    app_config_trusted_dirs = config.sandbox.trusted_dirs
    local_config_trusted_dirs = get_local_config_trusted_dirs()

    trusted_dirs = local_config_trusted_dirs
    if not local_config_trusted_dirs:
        trusted_dirs = app_config_trusted_dirs

    is_trusted = current_dir in trusted_dirs

    if not is_trusted:
        security_frame = Frame(
            TextArea(
                text=(
                    f' Do you trust the files in this folder?\n\n'
                    f'   {current_dir}\n\n'
                    ' OpenHands may read and execute files in this folder with your permission.'
                ),
                style=COLOR_GREY,
                read_only=True,
                wrap_lines=True,
            ),
            style=f'fg:{COLOR_GREY}',
        )

        clear()
        print_container(security_frame)
        print_formatted_text('')

        confirm = (
            cli_confirm(
                config, 'Do you wish to continue?', ['Yes, proceed', 'No, exit']
            )
            == 0
        )

        if confirm:
            add_local_config_trusted_dir(current_dir)

        return confirm

    return True


async def handle_mcp_command(config: OpenHandsConfig) -> None:
    """Handle MCP command with interactive menu."""
    action = cli_confirm(
        config,
        'MCP Server Configuration',
        [
            'List configured servers',
            'Add new server',
            'Remove server',
            'View errors',
            'Go back',
        ],
    )

    if action == 0:  # List
        display_mcp_servers(config)
    elif action == 1:  # Add
        await add_mcp_server(config)
    elif action == 2:  # Remove
        await remove_mcp_server(config)
    elif action == 3:  # View errors
        handle_mcp_errors_command()
    # action == 4 is "Go back", do nothing


def display_mcp_servers(config: OpenHandsConfig) -> None:
    """Display MCP server configuration information."""
    mcp_config = config.mcp

    # Count the different types of servers
    sse_count = len(mcp_config.sse_servers)
    stdio_count = len(mcp_config.stdio_servers)
    shttp_count = len(mcp_config.shttp_servers)
    total_count = sse_count + stdio_count + shttp_count

    if total_count == 0:
        print_formatted_text(
            'No custom MCP servers configured. See the documentation to learn more:\n'
            '  https://docs.all-hands.dev/usage/how-to/cli-mode#using-mcp-servers'
        )
    else:
        print_formatted_text(
            f'Configured MCP servers:\n'
            f'  • SSE servers: {sse_count}\n'
            f'  • Stdio servers: {stdio_count}\n'
            f'  • SHTTP servers: {shttp_count}\n'
            f'  • Total: {total_count}'
        )

        # Show details for each type if they exist
        if sse_count > 0:
            print_formatted_text('SSE Servers:')
            for idx, sse_server in enumerate(mcp_config.sse_servers, 1):
                print_formatted_text(f'  {idx}. {sse_server.url}')
            print_formatted_text('')

        if stdio_count > 0:
            print_formatted_text('Stdio Servers:')
            for idx, stdio_server in enumerate(mcp_config.stdio_servers, 1):
                print_formatted_text(
                    f'  {idx}. {stdio_server.name} ({stdio_server.command})'
                )
            print_formatted_text('')

        if shttp_count > 0:
            print_formatted_text('SHTTP Servers:')
            for idx, shttp_server in enumerate(mcp_config.shttp_servers, 1):
                print_formatted_text(f'  {idx}. {shttp_server.url}')
            print_formatted_text('')


def handle_mcp_errors_command() -> None:
    """Display MCP connection errors."""
    display_mcp_errors()


def get_config_file_path() -> Path:
    """Get the path to the config file. By default, we use config.toml in the current working directory. If not found, we use ~/.openhands/config.toml."""
    # Check if config.toml exists in the current directory
    current_dir = Path.cwd() / 'config.toml'
    if current_dir.exists():
        return current_dir

    # Fallback to the user's home directory
    return Path.home() / '.openhands' / 'config.toml'


def load_config_file(file_path: Path) -> dict:
    """Load the config file, creating it if it doesn't exist."""
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                return dict(tomlkit.load(f))
        except Exception:
            pass

    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return {}


def save_config_file(config_data: dict, file_path: Path) -> None:
    """Save the config file with proper MCP formatting."""
    doc = tomlkit.document()

    for key, value in config_data.items():
        if key == 'mcp':
            # Handle MCP section specially
            mcp_section = tomlkit.table()

            for mcp_key, mcp_value in value.items():
                # Create array with inline tables for server configurations
                server_array = tomlkit.array()
                for server_config in mcp_value:
                    if isinstance(server_config, dict):
                        # Create inline table for each server
                        inline_table = tomlkit.inline_table()
                        for server_key, server_val in server_config.items():
                            inline_table[server_key] = server_val
                        server_array.append(inline_table)
                    else:
                        # Handle non-dict values (like string URLs)
                        server_array.append(server_config)
                mcp_section[mcp_key] = server_array

            doc[key] = mcp_section
        else:
            # Handle non-MCP sections normally
            doc[key] = value

    with open(file_path, 'w') as f:
        f.write(tomlkit.dumps(doc))


def _ensure_mcp_config_structure(config_data: dict) -> None:
    """Ensure MCP configuration structure exists in config data."""
    if 'mcp' not in config_data:
        config_data['mcp'] = {}


def _add_server_to_config(server_type: str, server_config: dict) -> Path:
    """Add a server configuration to the config file."""
    config_file_path = get_config_file_path()
    config_data = load_config_file(config_file_path)
    _ensure_mcp_config_structure(config_data)

    if server_type not in config_data['mcp']:
        config_data['mcp'][server_type] = []

    config_data['mcp'][server_type].append(server_config)
    save_config_file(config_data, config_file_path)

    return config_file_path


async def add_mcp_server(config: OpenHandsConfig) -> None:
    """Add a new MCP server configuration."""
    # Choose transport type
    transport_type = cli_confirm(
        config,
        'Select MCP server transport type:',
        [
            'SSE (Server-Sent Events)',
            'Stdio (Standard Input/Output)',
            'SHTTP (Streamable HTTP)',
            'Cancel',
        ],
    )

    if transport_type == 3:  # Cancel
        return

    try:
        if transport_type == 0:  # SSE
            await add_sse_server(config)
        elif transport_type == 1:  # Stdio
            await add_stdio_server(config)
        elif transport_type == 2:  # SHTTP
            await add_shttp_server(config)
    except Exception as e:
        print_formatted_text(f'Error adding MCP server: {e}')


async def add_sse_server(config: OpenHandsConfig) -> None:
    """Add an SSE MCP server."""
    print_formatted_text('Adding SSE MCP Server')

    while True:  # Retry loop for the entire form
        # Collect all inputs
        url = await collect_input(config, '\nEnter server URL:')
        if url is None:
            print_formatted_text('Operation cancelled.')
            return

        api_key = await collect_input(
            config, '\nEnter API key (optional, press Enter to skip):'
        )
        if api_key is None:
            print_formatted_text('Operation cancelled.')
            return

        # Convert empty string to None for optional field
        api_key = api_key if api_key else None

        # Validate all inputs at once
        try:
            server = MCPSSEServerConfig(url=url, api_key=api_key)
            break  # Success - exit retry loop

        except ValidationError as e:
            # Show all errors at once
            print_formatted_text('❌ Please fix the following errors:')
            for error in e.errors():
                field = error['loc'][0] if error['loc'] else 'unknown'
                print_formatted_text(f'  • {field}: {error["msg"]}')

            if cli_confirm(config, '\nTry again?') != 0:
                print_formatted_text('Operation cancelled.')
                return

    # Save to config file
    server_config = {'url': server.url}
    if server.api_key:
        server_config['api_key'] = server.api_key

    config_file_path = _add_server_to_config('sse_servers', server_config)
    print_formatted_text(f'✓ SSE MCP server added to {config_file_path}: {server.url}')

    # Prompt for restart
    if await prompt_for_restart(config):
        restart_cli()


async def add_stdio_server(config: OpenHandsConfig) -> None:
    """Add a Stdio MCP server."""
    print_formatted_text('Adding Stdio MCP Server')

    # Get existing server names to check for duplicates
    existing_names = [server.name for server in config.mcp.stdio_servers]

    while True:  # Retry loop for the entire form
        # Collect all inputs
        name = await collect_input(config, '\nEnter server name:')
        if name is None:
            print_formatted_text('Operation cancelled.')
            return

        command = await collect_input(config, "\nEnter command (e.g., 'uvx', 'npx'):")
        if command is None:
            print_formatted_text('Operation cancelled.')
            return

        args_input = await collect_input(
            config,
            '\nEnter arguments (optional, e.g., "-y server-package arg1"):',
        )
        if args_input is None:
            print_formatted_text('Operation cancelled.')
            return

        env_input = await collect_input(
            config,
            '\nEnter environment variables (KEY=VALUE format, comma-separated, optional):',
        )
        if env_input is None:
            print_formatted_text('Operation cancelled.')
            return

        # Check for duplicate server names
        if name in existing_names:
            print_formatted_text(f"❌ Server name '{name}' already exists.")
            if cli_confirm(config, '\nTry again?') != 0:
                print_formatted_text('Operation cancelled.')
                return
            continue

        # Validate all inputs at once
        try:
            server = MCPStdioServerConfig(
                name=name,
                command=command,
                args=args_input,  # type: ignore  # Will be parsed by Pydantic validator
                env=env_input,  # type: ignore  # Will be parsed by Pydantic validator
            )
            break  # Success - exit retry loop

        except ValidationError as e:
            # Show all errors at once
            print_formatted_text('❌ Please fix the following errors:')
            for error in e.errors():
                field = error['loc'][0] if error['loc'] else 'unknown'
                print_formatted_text(f'  • {field}: {error["msg"]}')

            if cli_confirm(config, '\nTry again?') != 0:
                print_formatted_text('Operation cancelled.')
                return

    # Save to config file
    server_config: dict[str, Any] = {
        'name': server.name,
        'command': server.command,
    }
    if server.args:
        server_config['args'] = server.args
    if server.env:
        server_config['env'] = server.env

    config_file_path = _add_server_to_config('stdio_servers', server_config)
    print_formatted_text(
        f'✓ Stdio MCP server added to {config_file_path}: {server.name}'
    )

    # Prompt for restart
    if await prompt_for_restart(config):
        restart_cli()


async def add_shttp_server(config: OpenHandsConfig) -> None:
    """Add an SHTTP MCP server."""
    print_formatted_text('Adding SHTTP MCP Server')

    while True:  # Retry loop for the entire form
        # Collect all inputs
        url = await collect_input(config, '\nEnter server URL:')
        if url is None:
            print_formatted_text('Operation cancelled.')
            return

        api_key = await collect_input(
            config, '\nEnter API key (optional, press Enter to skip):'
        )
        if api_key is None:
            print_formatted_text('Operation cancelled.')
            return

        # Convert empty string to None for optional field
        api_key = api_key if api_key else None

        # Validate all inputs at once
        try:
            server = MCPSHTTPServerConfig(url=url, api_key=api_key)
            break  # Success - exit retry loop

        except ValidationError as e:
            # Show all errors at once
            print_formatted_text('❌ Please fix the following errors:')
            for error in e.errors():
                field = error['loc'][0] if error['loc'] else 'unknown'
                print_formatted_text(f'  • {field}: {error["msg"]}')

            if cli_confirm(config, '\nTry again?') != 0:
                print_formatted_text('Operation cancelled.')
                return

    # Save to config file
    server_config = {'url': server.url}
    if server.api_key:
        server_config['api_key'] = server.api_key

    config_file_path = _add_server_to_config('shttp_servers', server_config)
    print_formatted_text(
        f'✓ SHTTP MCP server added to {config_file_path}: {server.url}'
    )

    # Prompt for restart
    if await prompt_for_restart(config):
        restart_cli()


async def remove_mcp_server(config: OpenHandsConfig) -> None:
    """Remove an MCP server configuration."""
    mcp_config = config.mcp

    # Collect all servers with their types
    servers: list[tuple[str, str, object]] = []

    # Add SSE servers
    for sse_server in mcp_config.sse_servers:
        servers.append(('SSE', sse_server.url, sse_server))

    # Add Stdio servers
    for stdio_server in mcp_config.stdio_servers:
        servers.append(('Stdio', stdio_server.name, stdio_server))

    # Add SHTTP servers
    for shttp_server in mcp_config.shttp_servers:
        servers.append(('SHTTP', shttp_server.url, shttp_server))

    if not servers:
        print_formatted_text('No MCP servers configured to remove.')
        return

    # Create choices for the user
    choices = []
    for server_type, identifier, _ in servers:
        choices.append(f'{server_type}: {identifier}')
    choices.append('Cancel')

    # Let user choose which server to remove
    choice = cli_confirm(config, 'Select MCP server to remove:', choices)

    if choice == len(choices) - 1:  # Cancel
        return

    # Remove the selected server
    server_type, identifier, _ = servers[choice]

    # Confirm removal
    confirm = cli_confirm(
        config,
        f'Are you sure you want to remove {server_type} server "{identifier}"?',
        ['Yes, remove', 'Cancel'],
    )

    if confirm == 1:  # Cancel
        return

    # Load config file and remove the server
    config_file_path = get_config_file_path()
    config_data = load_config_file(config_file_path)

    _ensure_mcp_config_structure(config_data)

    removed = False

    if server_type == 'SSE' and 'sse_servers' in config_data['mcp']:
        config_data['mcp']['sse_servers'] = [
            s for s in config_data['mcp']['sse_servers'] if s.get('url') != identifier
        ]
        removed = True
    elif server_type == 'Stdio' and 'stdio_servers' in config_data['mcp']:
        config_data['mcp']['stdio_servers'] = [
            s
            for s in config_data['mcp']['stdio_servers']
            if s.get('name') != identifier
        ]
        removed = True
    elif server_type == 'SHTTP' and 'shttp_servers' in config_data['mcp']:
        config_data['mcp']['shttp_servers'] = [
            s for s in config_data['mcp']['shttp_servers'] if s.get('url') != identifier
        ]
        removed = True

    if removed:
        save_config_file(config_data, config_file_path)
        print_formatted_text(
            f'✓ {server_type} MCP server "{identifier}" removed from {config_file_path}.'
        )

        # Prompt for restart
        if await prompt_for_restart(config):
            restart_cli()
    else:
        print_formatted_text(f'Failed to remove {server_type} server "{identifier}".')
