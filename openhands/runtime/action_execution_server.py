"""This is the main file for the runtime client.
It is responsible for executing actions received from OpenHands backend and producing observations.

NOTE: this will be executed inside the docker sandbox.
"""

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import shutil
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from zipfile import ZipFile

import puremagic
from binaryornot.check import is_binary
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from openhands_aci.editor.editor import OHEditor
from openhands_aci.editor.exceptions import ToolError
from openhands_aci.editor.results import ToolResult
from openhands_aci.utils.diff import get_diff
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException
from uvicorn import run

from openhands.core.config.mcp_config import MCPStdioServerConfig
from openhands.core.exceptions import BrowserUnavailableException
from openhands.core.logger import get_uvicorn_json_log_config
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    BrowseInteractiveAction,
    BrowseURLAction,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    FileWriteAction,
    IPythonRunCellAction,
)
from openhands.events.event import FileEditSource, FileReadSource
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    FileDownloadObservation,
    FileEditObservation,
    FileReadObservation,
    FileWriteObservation,
    IPythonRunCellObservation,
    Observation,
)
from openhands.events.serialization import event_from_dict, event_to_dict
from openhands.runtime.browser import browse
from openhands.runtime.browser.browser_env import BrowserEnv
from openhands.runtime.file_viewer_server import start_file_viewer_server

# Import our custom MCP Proxy Manager
from openhands.runtime.mcp.proxy import MCPProxyManager
from openhands.runtime.plugins import ALL_PLUGINS, JupyterPlugin, Plugin, VSCodePlugin
from openhands.runtime.utils import find_available_tcp_port
from openhands.runtime.utils.bash import BashSession
from openhands.runtime.utils.files import insert_lines, read_lines
from openhands.runtime.utils.memory_monitor import MemoryMonitor
from openhands.runtime.utils.runtime_init import init_user_and_working_directory
from openhands.runtime.utils.system_stats import (
    get_system_stats,
    update_last_execution_time,
)
from openhands.utils.async_utils import call_sync_from_async, wait_all

if sys.platform == 'win32':
    from openhands.runtime.utils.windows_bash import WindowsPowershellSession


class ActionRequest(BaseModel):
    action: dict


ROOT_GID = 0

SESSION_API_KEY = os.environ.get('SESSION_API_KEY')
api_key_header = APIKeyHeader(name='X-Session-API-Key', auto_error=False)


def verify_api_key(api_key: str = Depends(api_key_header)):
    if SESSION_API_KEY and api_key != SESSION_API_KEY:
        raise HTTPException(status_code=403, detail='Invalid API Key')
    return api_key


def _execute_file_editor(
    editor: OHEditor,
    command: str,
    path: str,
    file_text: str | None = None,
    view_range: list[int] | None = None,
    old_str: str | None = None,
    new_str: str | None = None,
    insert_line: int | str | None = None,
    enable_linting: bool = False,
) -> tuple[str, tuple[str | None, str | None]]:
    """Execute file editor command and handle exceptions.

    Args:
        editor: The OHEditor instance
        command: Editor command to execute
        path: File path
        file_text: Optional file text content
        view_range: Optional view range tuple (start, end)
        old_str: Optional string to replace
        new_str: Optional replacement string
        insert_line: Optional line number for insertion (can be int or str)
        enable_linting: Whether to enable linting

    Returns:
        tuple: A tuple containing the output string and a tuple of old and new file content
    """
    result: ToolResult | None = None

    # Convert insert_line from string to int if needed
    if insert_line is not None and isinstance(insert_line, str):
        try:
            insert_line = int(insert_line)
        except ValueError:
            return (
                f"ERROR:\nInvalid insert_line value: '{insert_line}'. Expected an integer.",
                (None, None),
            )

    try:
        result = editor(
            command=command,
            path=path,
            file_text=file_text,
            view_range=view_range,
            old_str=old_str,
            new_str=new_str,
            insert_line=insert_line,
            enable_linting=enable_linting,
        )
    except ToolError as e:
        result = ToolResult(error=e.message)
    except TypeError as e:
        # Handle unexpected arguments or type errors
        return f'ERROR:\n{str(e)}', (None, None)

    if result.error:
        return f'ERROR:\n{result.error}', (None, None)

    if not result.output:
        logger.warning(f'No output from file_editor for {path}')
        return '', (None, None)

    return result.output, (result.old_content, result.new_content)


class ActionExecutor:
    """ActionExecutor is running inside docker sandbox.
    It is responsible for executing actions received from OpenHands backend and producing observations.
    """

    def __init__(
        self,
        plugins_to_load: list[Plugin],
        work_dir: str,
        username: str,
        user_id: int,
        enable_browser: bool,
        browsergym_eval_env: str | None,
    ) -> None:
        self.plugins_to_load = plugins_to_load
        self._initial_cwd = work_dir
        self.username = username
        self.user_id = user_id
        _updated_user_id = init_user_and_working_directory(
            username=username, user_id=self.user_id, initial_cwd=work_dir
        )
        if _updated_user_id is not None:
            self.user_id = _updated_user_id

        self.bash_session: BashSession | 'WindowsPowershellSession' | None = None  # type: ignore[name-defined]
        self.lock = asyncio.Lock()
        self.plugins: dict[str, Plugin] = {}
        self.file_editor = OHEditor(workspace_root=self._initial_cwd)
        self.enable_browser = enable_browser
        self.browser: BrowserEnv | None = None
        self.browser_init_task: asyncio.Task | None = None
        self.browsergym_eval_env = browsergym_eval_env

        if (not self.enable_browser) and self.browsergym_eval_env:
            raise BrowserUnavailableException(
                'Browser environment is not enabled in config, but browsergym_eval_env is set'
            )

        self.start_time = time.time()
        self.last_execution_time = self.start_time
        self._initialized = False
        self.downloaded_files: list[str] = []
        self.downloads_directory = '/workspace/.downloads'

        self.max_memory_gb: int | None = None
        if _override_max_memory_gb := os.environ.get('RUNTIME_MAX_MEMORY_GB', None):
            self.max_memory_gb = int(_override_max_memory_gb)
            logger.info(
                f'Setting max memory to {self.max_memory_gb}GB (according to the RUNTIME_MAX_MEMORY_GB environment variable)'
            )
        else:
            logger.info('No max memory limit set, using all available system memory')

        self.memory_monitor = MemoryMonitor(
            enable=os.environ.get('RUNTIME_MEMORY_MONITOR', 'False').lower()
            in ['true', '1', 'yes']
        )
        self.memory_monitor.start_monitoring()

    @property
    def initial_cwd(self):
        return self._initial_cwd

    async def _init_browser_async(self):
        """Initialize the browser asynchronously."""
        if not self.enable_browser:
            logger.info('Browser environment is not enabled in config')
            return

        if sys.platform == 'win32':
            logger.warning('Browser environment not supported on windows')
            return

        logger.debug('Initializing browser asynchronously')
        try:
            self.browser = BrowserEnv(self.browsergym_eval_env)
            logger.debug('Browser initialized asynchronously')
        except Exception as e:
            logger.exception(f'Failed to initialize browser: {e}')
            self.browser = None

    async def _ensure_browser_ready(self):
        """Ensure the browser is ready for use."""
        if self.browser is None:
            if self.browser_init_task is None:
                # Start browser initialization if it hasn't been started
                self.browser_init_task = asyncio.create_task(self._init_browser_async())
            elif self.browser_init_task.done():
                # If the task is done but browser is still None, restart initialization
                self.browser_init_task = asyncio.create_task(self._init_browser_async())

            # Wait for browser to be initialized
            if self.browser_init_task:
                logger.debug('Waiting for browser to be ready...')
                await self.browser_init_task

            # Check if browser was successfully initialized
            if self.browser is None:
                raise BrowserUnavailableException('Browser initialization failed')

        # If we get here, the browser is ready
        logger.debug('Browser is ready')

    def _create_bash_session(self, cwd: str | None = None):
        if sys.platform == 'win32':
            return WindowsPowershellSession(  # type: ignore[name-defined]
                work_dir=cwd or self._initial_cwd,
                username=self.username,
                no_change_timeout_seconds=int(
                    os.environ.get('NO_CHANGE_TIMEOUT_SECONDS', 10)
                ),
                max_memory_mb=self.max_memory_gb * 1024 if self.max_memory_gb else None,
            )
        else:
            bash_session = BashSession(
                work_dir=cwd or self._initial_cwd,
                username=self.username,
                no_change_timeout_seconds=int(
                    os.environ.get('NO_CHANGE_TIMEOUT_SECONDS', 10)
                ),
                max_memory_mb=self.max_memory_gb * 1024 if self.max_memory_gb else None,
            )
            bash_session.initialize()
            return bash_session

    async def ainit(self):
        # bash needs to be initialized first
        logger.debug('Initializing bash session')
        self.bash_session = self._create_bash_session()
        logger.debug('Bash session initialized')

        # Start browser initialization in the background
        self.browser_init_task = asyncio.create_task(self._init_browser_async())
        logger.debug('Browser initialization started in background')

        await wait_all(
            (self._init_plugin(plugin) for plugin in self.plugins_to_load),
            timeout=int(os.environ.get('INIT_PLUGIN_TIMEOUT', '120')),
        )
        logger.debug('All plugins initialized')

        # This is a temporary workaround
        # TODO: refactor AgentSkills to be part of JupyterPlugin
        # AFTER ServerRuntime is deprecated
        logger.debug('Initializing AgentSkills')
        if 'agent_skills' in self.plugins and 'jupyter' in self.plugins:
            obs = await self.run_ipython(
                IPythonRunCellAction(
                    code='from openhands.runtime.plugins.agent_skills.agentskills import *\n'
                )
            )
            logger.debug(f'AgentSkills initialized: {obs}')

        logger.debug('Initializing bash commands')
        await self._init_bash_commands()

        logger.debug('Runtime client initialized.')
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    async def _init_plugin(self, plugin: Plugin):
        assert self.bash_session is not None
        # VSCode plugin needs runtime_id for path-based routing when using Gateway API
        if isinstance(plugin, VSCodePlugin):
            runtime_id = os.environ.get('RUNTIME_ID')
            await plugin.initialize(self.username, runtime_id=runtime_id)
        else:
            await plugin.initialize(self.username)
        self.plugins[plugin.name] = plugin
        logger.debug(f'Initializing plugin: {plugin.name}')

        if isinstance(plugin, JupyterPlugin):
            # Escape backslashes in Windows path
            cwd = self.bash_session.cwd.replace('\\', '/')
            await self.run_ipython(
                IPythonRunCellAction(code=f'import os; os.chdir(r"{cwd}")')
            )

    async def _init_bash_commands(self):
        # You can add any bash commands you want to run on startup here
        # It is empty because: Git configuration is now handled by the runtime client after connection
        INIT_COMMANDS = []
        is_windows = sys.platform == 'win32'

        # Determine no-pager command
        if is_windows:
            no_pager_cmd = 'function git { git.exe --no-pager $args }'
        else:
            no_pager_cmd = 'alias git="git --no-pager"'

        INIT_COMMANDS.append(no_pager_cmd)

        # Hack: for some reason when you set the openhands user to anything but root, tmux changes out
        # of the mount directory on the first invocation.
        if self.user_id != 0:
            INIT_COMMANDS.append(f'cd {self._initial_cwd}')

        logger.info(f'Initializing by running {len(INIT_COMMANDS)} bash commands...')
        for command in INIT_COMMANDS:
            action = CmdRunAction(command=command)
            action.set_hard_timeout(300)
            logger.debug(f'Executing init command: {command}')
            obs = await self.run(action)
            assert isinstance(obs, CmdOutputObservation)
            logger.debug(
                f'Init command outputs (exit code: {obs.exit_code}): {obs.content}'
            )
            assert obs.exit_code == 0
        logger.debug('Bash init commands completed')

    async def run_action(self, action) -> Observation:
        async with self.lock:
            action_type = action.action
            observation = await getattr(self, action_type)(action)
            return observation

    async def run(
        self, action: CmdRunAction
    ) -> CmdOutputObservation | ErrorObservation:
        try:
            bash_session = self.bash_session
            if action.is_static:
                bash_session = self._create_bash_session(action.cwd)
            assert bash_session is not None
            obs = await call_sync_from_async(bash_session.execute, action)
            return obs
        except Exception as e:
            logger.exception(f'Error running command: {e}')
            return ErrorObservation(str(e))

    async def run_ipython(self, action: IPythonRunCellAction) -> Observation:
        assert self.bash_session is not None
        if 'jupyter' in self.plugins:
            _jupyter_plugin: JupyterPlugin = self.plugins['jupyter']  # type: ignore
            # This is used to make AgentSkills in Jupyter aware of the
            # current working directory in Bash
            jupyter_cwd = getattr(self, '_jupyter_cwd', None)
            if self.bash_session.cwd != jupyter_cwd:
                logger.debug(
                    f'{self.bash_session.cwd} != {jupyter_cwd} -> reset Jupyter PWD'
                )
                # escape windows paths
                cwd = self.bash_session.cwd.replace('\\', '/')
                reset_jupyter_cwd_code = f'import os; os.chdir("{cwd}")'
                _aux_action = IPythonRunCellAction(code=reset_jupyter_cwd_code)
                _reset_obs: IPythonRunCellObservation = await _jupyter_plugin.run(
                    _aux_action
                )
                logger.debug(
                    f'Changed working directory in IPython to: {self.bash_session.cwd}. Output: {_reset_obs}'
                )
                self._jupyter_cwd = self.bash_session.cwd

            obs: IPythonRunCellObservation = await _jupyter_plugin.run(action)
            obs.content = obs.content.rstrip()

            if action.include_extra:
                obs.content += (
                    f'\n[Jupyter current working directory: {self.bash_session.cwd}]'
                )
                obs.content += f'\n[Jupyter Python interpreter: {_jupyter_plugin.python_interpreter_path}]'
            return obs
        else:
            raise RuntimeError(
                'JupyterRequirement not found. Unable to run IPython action.'
            )

    def _resolve_path(self, path: str, working_dir: str) -> str:
        filepath = Path(path)
        if not filepath.is_absolute():
            return str(Path(working_dir) / filepath)
        return str(filepath)

    async def read(self, action: FileReadAction) -> Observation:
        assert self.bash_session is not None

        # Cannot read binary files
        if is_binary(action.path):
            return ErrorObservation('ERROR_BINARY_FILE')

        if action.impl_source == FileReadSource.OH_ACI:
            result_str, _ = _execute_file_editor(
                self.file_editor,
                command='view',
                path=action.path,
                view_range=action.view_range,
            )

            return FileReadObservation(
                content=result_str,
                path=action.path,
                impl_source=FileReadSource.OH_ACI,
            )

        # NOTE: the client code is running inside the sandbox,
        # so there's no need to check permission
        working_dir = self.bash_session.cwd
        filepath = self._resolve_path(action.path, working_dir)
        try:
            if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                with open(filepath, 'rb') as file:
                    image_data = file.read()
                    encoded_image = base64.b64encode(image_data).decode('utf-8')
                    mime_type, _ = mimetypes.guess_type(filepath)
                    if mime_type is None:
                        mime_type = 'image/png'  # default to PNG if mime type cannot be determined
                    encoded_image = f'data:{mime_type};base64,{encoded_image}'

                return FileReadObservation(path=filepath, content=encoded_image)
            elif filepath.lower().endswith('.pdf'):
                with open(filepath, 'rb') as file:
                    pdf_data = file.read()
                    encoded_pdf = base64.b64encode(pdf_data).decode('utf-8')
                    encoded_pdf = f'data:application/pdf;base64,{encoded_pdf}'
                return FileReadObservation(path=filepath, content=encoded_pdf)
            elif filepath.lower().endswith(('.mp4', '.webm', '.ogg')):
                with open(filepath, 'rb') as file:
                    video_data = file.read()
                    encoded_video = base64.b64encode(video_data).decode('utf-8')
                    mime_type, _ = mimetypes.guess_type(filepath)
                    if mime_type is None:
                        mime_type = 'video/mp4'  # default to MP4 if MIME type cannot be determined
                    encoded_video = f'data:{mime_type};base64,{encoded_video}'

                return FileReadObservation(path=filepath, content=encoded_video)

            with open(filepath, 'r', encoding='utf-8') as file:
                lines = read_lines(file.readlines(), action.start, action.end)
        except FileNotFoundError:
            return ErrorObservation(
                f'File not found: {filepath}. Your current working directory is {working_dir}.'
            )
        except UnicodeDecodeError:
            return ErrorObservation(f'File could not be decoded as utf-8: {filepath}.')
        except IsADirectoryError:
            return ErrorObservation(
                f'Path is a directory: {filepath}. You can only read files'
            )

        code_view = ''.join(lines)
        return FileReadObservation(path=filepath, content=code_view)

    async def write(self, action: FileWriteAction) -> Observation:
        assert self.bash_session is not None
        working_dir = self.bash_session.cwd
        filepath = self._resolve_path(action.path, working_dir)

        insert = action.content.split('\n')
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        file_exists = os.path.exists(filepath)
        if file_exists:
            file_stat = os.stat(filepath)
        else:
            file_stat = None

        mode = 'w' if not file_exists else 'r+'
        try:
            with open(filepath, mode, encoding='utf-8') as file:
                if mode != 'w':
                    all_lines = file.readlines()
                    new_file = insert_lines(insert, all_lines, action.start, action.end)
                else:
                    new_file = [i + '\n' for i in insert]

                file.seek(0)
                file.writelines(new_file)
                file.truncate()

        except FileNotFoundError:
            return ErrorObservation(f'File not found: {filepath}')
        except IsADirectoryError:
            return ErrorObservation(
                f'Path is a directory: {filepath}. You can only write to files'
            )
        except UnicodeDecodeError:
            return ErrorObservation(f'File could not be decoded as utf-8: {filepath}')

        # Attempt to handle file permissions
        try:
            if file_exists:
                assert file_stat is not None
                # restore the original file permissions if the file already exists
                os.chmod(filepath, file_stat.st_mode)
                os.chown(filepath, file_stat.st_uid, file_stat.st_gid)
            else:
                # set the new file permissions if the file is new
                os.chmod(filepath, 0o664)
                os.chown(filepath, self.user_id, self.user_id)
        except PermissionError as e:
            return ErrorObservation(
                f'File {filepath} written, but failed to change ownership and permissions: {e}'
            )
        return FileWriteObservation(content='', path=filepath)

    async def edit(self, action: FileEditAction) -> Observation:
        assert action.impl_source == FileEditSource.OH_ACI
        result_str, (old_content, new_content) = _execute_file_editor(
            self.file_editor,
            command=action.command,
            path=action.path,
            file_text=action.file_text,
            old_str=action.old_str,
            new_str=action.new_str,
            insert_line=action.insert_line,
            enable_linting=False,
        )

        return FileEditObservation(
            content=result_str,
            path=action.path,
            old_content=action.old_str,
            new_content=action.new_str,
            impl_source=FileEditSource.OH_ACI,
            diff=get_diff(
                old_contents=old_content or '',
                new_contents=new_content or '',
                filepath=action.path,
            ),
        )

    async def browse(self, action: BrowseURLAction) -> Observation:
        if self.browser is None:
            return ErrorObservation(
                'Browser functionality is not supported or disabled.'
            )
        await self._ensure_browser_ready()
        return await browse(action, self.browser, self.initial_cwd)

    async def browse_interactive(self, action: BrowseInteractiveAction) -> Observation:
        if self.browser is None:
            return ErrorObservation(
                'Browser functionality is not supported or disabled.'
            )
        await self._ensure_browser_ready()
        browser_observation = await browse(action, self.browser, self.initial_cwd)
        if not browser_observation.error:
            return browser_observation
        else:
            curr_files = os.listdir(self.downloads_directory)
            new_download = False
            for file in curr_files:
                if file not in self.downloaded_files:
                    new_download = True
                    self.downloaded_files.append(file)
                    break  # FIXME: assuming only one file will be downloaded for simplicity

            if not new_download:
                return browser_observation
            else:
                # A new file is downloaded in self.downloads_directory, shift file to /workspace
                src_path = os.path.join(
                    self.downloads_directory, self.downloaded_files[-1]
                )
                # Guess extension of file using puremagic and add it to tgt_path file name
                file_ext = ''
                try:
                    guesses = puremagic.magic_file(src_path)
                    if len(guesses) > 0:
                        ext = guesses[0].extension.strip()
                        if len(ext) > 0:
                            file_ext = ext
                except Exception as _:
                    pass

                tgt_path = os.path.join(
                    '/workspace', f'file_{len(self.downloaded_files)}{file_ext}'
                )
                shutil.copy(src_path, tgt_path)
                file_download_obs = FileDownloadObservation(
                    content=f'Execution of the previous action {action.browser_actions} resulted in a file download. The downloaded file is saved at location: {tgt_path}',
                    file_path=tgt_path,
                )
                return file_download_obs

    def close(self):
        self.memory_monitor.stop_monitoring()
        if self.bash_session is not None:
            self.bash_session.close()
        if self.browser is not None:
            self.browser.close()


if __name__ == '__main__':
    logger.warning('Starting Action Execution Server')
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int, help='Port to listen on')
    parser.add_argument('--working-dir', type=str, help='Working directory')
    parser.add_argument('--plugins', type=str, help='Plugins to initialize', nargs='+')
    parser.add_argument(
        '--username', type=str, help='User to run as', default='openhands'
    )
    parser.add_argument('--user-id', type=int, help='User ID to run as', default=1000)
    parser.add_argument(
        '--enable-browser',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable the browser environment',
    )
    parser.add_argument(
        '--browsergym-eval-env',
        type=str,
        help='BrowserGym environment used for browser evaluation',
        default=None,
    )

    # example: python client.py 8000 --working-dir /workspace --plugins JupyterRequirement
    args = parser.parse_args()

    # Start the file viewer server in a separate thread
    logger.info('Starting file viewer server')
    _file_viewer_port = find_available_tcp_port(
        min_port=args.port + 1, max_port=min(args.port + 1024, 65535)
    )
    server_url, _ = start_file_viewer_server(port=_file_viewer_port)
    logger.info(f'File viewer server started at {server_url}')

    plugins_to_load: list[Plugin] = []
    if args.plugins:
        for plugin in args.plugins:
            if plugin not in ALL_PLUGINS:
                raise ValueError(f'Plugin {plugin} not found')
            plugins_to_load.append(ALL_PLUGINS[plugin]())  # type: ignore

    client: ActionExecutor | None = None
    mcp_proxy_manager: MCPProxyManager | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global client, mcp_proxy_manager
        logger.info('Initializing ActionExecutor...')
        client = ActionExecutor(
            plugins_to_load,
            work_dir=args.working_dir,
            username=args.username,
            user_id=args.user_id,
            enable_browser=args.enable_browser,
            browsergym_eval_env=args.browsergym_eval_env,
        )
        await client.ainit()
        logger.info('ActionExecutor initialized.')

        # Check if we're on Windows
        is_windows = sys.platform == 'win32'

        # Initialize and mount MCP Proxy Manager (skip on Windows)
        if is_windows:
            logger.info('Skipping MCP Proxy initialization on Windows')
            mcp_proxy_manager = None
        else:
            logger.info('Initializing MCP Proxy Manager...')
            # Create a MCP Proxy Manager
            mcp_proxy_manager = MCPProxyManager(
                auth_enabled=bool(SESSION_API_KEY),
                api_key=SESSION_API_KEY,
                logger_level=logger.getEffectiveLevel(),
            )
            mcp_proxy_manager.initialize()
            # Mount the proxy to the app
            allowed_origins = ['*']
            try:
                await mcp_proxy_manager.mount_to_app(app, allowed_origins)
            except Exception as e:
                logger.error(f'Error mounting MCP Proxy: {e}', exc_info=True)
                raise RuntimeError(f'Cannot mount MCP Proxy: {e}')

        yield

        # Clean up & release the resources
        logger.info('Shutting down MCP Proxy Manager...')
        if mcp_proxy_manager:
            del mcp_proxy_manager
            mcp_proxy_manager = None
        else:
            logger.info('MCP Proxy Manager instance not found for shutdown.')

        logger.info('Closing ActionExecutor...')
        if client:
            try:
                client.close()
                logger.info('ActionExecutor closed successfully.')
            except Exception as e:
                logger.error(f'Error closing ActionExecutor: {e}', exc_info=True)
        else:
            logger.info('ActionExecutor instance not found for closing.')
        logger.info('Shutdown complete.')

    app = FastAPI(lifespan=lifespan)

    # TODO below 3 exception handlers were recommended by Sonnet.
    # Are these something we should keep?
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception('Unhandled exception occurred:')
        return JSONResponse(
            status_code=500,
            content={'detail': 'An unexpected error occurred. Please try again later.'},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.exception(f'HTTP exception occurred: {exc.detail}')
        return JSONResponse(status_code=exc.status_code, content={'detail': exc.detail})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        logger.exception(f'Validation error occurred: {exc}')
        return JSONResponse(
            status_code=422,
            content={
                'detail': 'Invalid request parameters',
                'errors': str(exc.errors()),
            },
        )

    @app.middleware('http')
    async def authenticate_requests(request: Request, call_next):
        if request.url.path != '/alive' and request.url.path != '/server_info':
            try:
                verify_api_key(request.headers.get('X-Session-API-Key'))
            except HTTPException as e:
                return JSONResponse(
                    status_code=e.status_code, content={'detail': e.detail}
                )
        response = await call_next(request)
        return response

    @app.get('/server_info')
    async def get_server_info():
        assert client is not None
        current_time = time.time()
        uptime = current_time - client.start_time
        idle_time = current_time - client.last_execution_time

        response = {
            'uptime': uptime,
            'idle_time': idle_time,
            'resources': get_system_stats(),
        }
        logger.info('Server info endpoint response: %s', response)
        return response

    @app.post('/execute_action')
    async def execute_action(action_request: ActionRequest):
        assert client is not None
        try:
            action = event_from_dict(action_request.action)
            if not isinstance(action, Action):
                raise HTTPException(status_code=400, detail='Invalid action type')
            client.last_execution_time = time.time()
            observation = await client.run_action(action)
            return event_to_dict(observation)
        except Exception as e:
            logger.exception(f'Error while running /execute_action: {str(e)}')
            raise HTTPException(
                status_code=500,
                detail=f'Internal server error: {str(e)}',
            )
        finally:
            update_last_execution_time()

    @app.post('/update_mcp_server')
    async def update_mcp_server(request: Request):
        # Check if we're on Windows
        is_windows = sys.platform == 'win32'

        # Access the global mcp_proxy_manager variable
        global mcp_proxy_manager

        if is_windows:
            # On Windows, just return a success response without doing anything
            logger.info(
                'MCP server update request received on Windows - skipping as MCP is disabled'
            )
            return JSONResponse(
                status_code=200,
                content={
                    'detail': 'MCP server update skipped (MCP is disabled on Windows)',
                    'router_error_log': '',
                },
            )

        # Non-Windows implementation
        if mcp_proxy_manager is None:
            raise HTTPException(
                status_code=500, detail='MCP Proxy Manager is not initialized'
            )

        # Get the request body
        mcp_tools_to_sync = await request.json()
        if not isinstance(mcp_tools_to_sync, list):
            raise HTTPException(
                status_code=400, detail='Request must be a list of MCP tools to sync'
            )
        logger.info(
            f'Updating MCP server with tools: {json.dumps(mcp_tools_to_sync, indent=2)}'
        )
        mcp_tools_to_sync = [MCPStdioServerConfig(**tool) for tool in mcp_tools_to_sync]
        try:
            await mcp_proxy_manager.update_and_remount(app, mcp_tools_to_sync, ['*'])
            logger.info('MCP Proxy Manager updated and remounted successfully')
            router_error_log = ''
        except Exception as e:
            logger.error(f'Error updating MCP Proxy Manager: {e}', exc_info=True)
            router_error_log = str(e)

        return JSONResponse(
            status_code=200,
            content={
                'detail': 'MCP server updated successfully',
                'router_error_log': router_error_log,
            },
        )

    @app.post('/upload_file')
    async def upload_file(
        file: UploadFile,
        destination: str = '/',
        recursive: bool = False,
    ):
        assert client is not None

        try:
            # Ensure the destination directory exists
            if not os.path.isabs(destination):
                raise HTTPException(
                    status_code=400, detail='Destination must be an absolute path'
                )

            full_dest_path = destination
            if not os.path.exists(full_dest_path):
                os.makedirs(full_dest_path, exist_ok=True)

            if recursive or file.filename.endswith('.zip'):
                # For recursive uploads, we expect a zip file
                if not file.filename.endswith('.zip'):
                    raise HTTPException(
                        status_code=400, detail='Recursive uploads must be zip files'
                    )

                zip_path = os.path.join(full_dest_path, file.filename)
                with open(zip_path, 'wb') as buffer:
                    shutil.copyfileobj(file.file, buffer)

                # Extract the zip file
                shutil.unpack_archive(zip_path, full_dest_path)
                os.remove(zip_path)  # Remove the zip file after extraction

                logger.debug(
                    f'Uploaded file {file.filename} and extracted to {destination}'
                )
            else:
                # For single file uploads
                file_path = os.path.join(full_dest_path, file.filename)
                with open(file_path, 'wb') as buffer:
                    shutil.copyfileobj(file.file, buffer)
                logger.debug(f'Uploaded file {file.filename} to {destination}')

            return JSONResponse(
                content={
                    'filename': file.filename,
                    'destination': destination,
                    'recursive': recursive,
                },
                status_code=200,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get('/download_files')
    def download_file(path: str):
        logger.debug('Downloading files')
        try:
            if not os.path.isabs(path):
                raise HTTPException(
                    status_code=400, detail='Path must be an absolute path'
                )

            if not os.path.exists(path):
                raise HTTPException(status_code=404, detail='File not found')

            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
                with ZipFile(temp_zip, 'w') as zipf:
                    for root, _, files in os.walk(path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zipf.write(
                                file_path, arcname=os.path.relpath(file_path, path)
                            )
                return FileResponse(
                    path=temp_zip.name,
                    media_type='application/zip',
                    filename=f'{os.path.basename(path)}.zip',
                    background=BackgroundTask(lambda: os.unlink(temp_zip.name)),
                )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get('/alive')
    async def alive():
        if client is None or not client.initialized:
            return {'status': 'not initialized'}
        return {'status': 'ok'}

    # ================================
    # VSCode-specific operations
    # ================================

    @app.get('/vscode/connection_token')
    async def get_vscode_connection_token():
        assert client is not None
        if 'vscode' in client.plugins:
            plugin: VSCodePlugin = client.plugins['vscode']  # type: ignore
            return {'token': plugin.vscode_connection_token}
        else:
            return {'token': None}

    # ================================
    # File-specific operations for UI
    # ================================

    @app.post('/list_files')
    async def list_files(request: Request):
        """List files in the specified path.

        This function retrieves a list of files from the agent's runtime file store,
        excluding certain system and hidden files/directories.

        To list files:
        ```sh
        curl -X POST -d '{"path": "/"}' http://localhost:3000/list_files
        ```

        Args:
            request (Request): The incoming request object.
            path (str, optional): The path to list files from. Defaults to '/'.

        Returns:
            list: A list of file names in the specified path.

        Raises:
            HTTPException: If there's an error listing the files.
        """
        assert client is not None

        # get request as dict
        request_dict = await request.json()
        path = request_dict.get('path', None)

        # Get the full path of the requested directory
        if path is None:
            full_path = client.initial_cwd
        elif os.path.isabs(path):
            full_path = path
        else:
            full_path = os.path.join(client.initial_cwd, path)

        if not os.path.exists(full_path):
            # if user just removed a folder, prevent server error 500 in UI
            return JSONResponse(content=[])

        try:
            # Check if the directory exists
            if not os.path.exists(full_path) or not os.path.isdir(full_path):
                return JSONResponse(content=[])

            entries = os.listdir(full_path)

            # Separate directories and files
            directories = []
            files = []
            for entry in entries:
                # Remove leading slash and any parent directory components
                entry_relative = entry.lstrip('/').split('/')[-1]

                # Construct the full path by joining the base path with the relative entry path
                full_entry_path = os.path.join(full_path, entry_relative)
                if os.path.exists(full_entry_path):
                    is_dir = os.path.isdir(full_entry_path)
                    if is_dir:
                        # add trailing slash to directories
                        # required by FE to differentiate directories and files
                        entry = entry.rstrip('/') + '/'
                        directories.append(entry)
                    else:
                        files.append(entry)

            # Sort directories and files separately
            directories.sort(key=lambda s: s.lower())
            files.sort(key=lambda s: s.lower())

            # Combine sorted directories and files
            sorted_entries = directories + files
            return JSONResponse(content=sorted_entries)

        except Exception as e:
            logger.exception(f'Error listing files: {e}')
            return JSONResponse(content=[])

    logger.debug(f'Starting action execution API on port {args.port}')
    # When LOG_JSON=1, provide a JSON log config to Uvicorn so error/access logs are structured
    log_config = None
    if os.getenv('LOG_JSON', '0') in ('1', 'true', 'True'):
        log_config = get_uvicorn_json_log_config()
    run(app, host='0.0.0.0', port=args.port, log_config=log_config, use_colors=False)
