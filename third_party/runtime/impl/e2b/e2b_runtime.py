import os
from typing import Callable

from openhands.core.config import OpenHandsConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    BrowseURLAction,
    BrowseInteractiveAction,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    FileWriteAction,
    IPythonRunCellAction,
)
from openhands.events.observation import (
    BrowserOutputObservation,
    CmdOutputObservation,
    ErrorObservation,
    FileEditObservation,
    FileReadObservation,
    FileWriteObservation,
    IPythonRunCellObservation,
    Observation,
)
from openhands.events.stream import EventStream
from openhands.integrations.provider import PROVIDER_TOKEN_TYPE
from openhands.sdk.llm import LLMRegistry
from openhands.runtime.impl.action_execution.action_execution_client import (
    ActionExecutionClient,
)
from openhands.runtime.plugins import PluginRequirement
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.runtime.utils.files import insert_lines, read_lines
from openhands.utils.async_utils import call_sync_from_async
from third_party.runtime.impl.e2b.filestore import E2BFileStore
from third_party.runtime.impl.e2b.sandbox import E2BBox, E2BSandbox


class E2BRuntime(ActionExecutionClient):
    # Class-level cache for sandbox IDs
    _sandbox_id_cache: dict[str, str] = {}
    
    def __init__(
        self,
        config: OpenHandsConfig,
        event_stream: EventStream,
        llm_registry: LLMRegistry,
        sid: str = "default",
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_callback: Callable | None = None,
        attach_to_existing: bool = False,
        headless_mode: bool = True,
        user_id: str | None = None,
        git_provider_tokens: PROVIDER_TOKEN_TYPE | None = None,
        sandbox: E2BSandbox | None = None,
    ):
        if config.workspace_base is not None:
            logger.warning(
                "Setting workspace_base is not supported in the E2B runtime. "
                "E2B provides its own isolated filesystem."
            )
        
        super().__init__(
            config=config,
            event_stream=event_stream,
            llm_registry=llm_registry,
            sid=sid,
            plugins=plugins,
            env_vars=env_vars,
            status_callback=status_callback,
            attach_to_existing=attach_to_existing,
            headless_mode=headless_mode,
            user_id=user_id,
            git_provider_tokens=git_provider_tokens,
        )
        self.sandbox = sandbox
        self.file_store = None
        self.api_url = None
        self._action_server_port = 8000
        self._runtime_initialized = False

    @property
    def action_execution_server_url(self) -> str:
        """Return the URL of the action execution server."""
        if not self.api_url:
            raise RuntimeError("E2B runtime not connected. Call connect() first.")
        return self.api_url

    async def connect(self) -> None:
        """Initialize E2B sandbox and start action execution server."""
        self.set_runtime_status(RuntimeStatus.STARTING_RUNTIME)
        
        try:
            if self.attach_to_existing and self.sandbox is None:
                try:
                    cached_sandbox_id = self.__class__._sandbox_id_cache.get(self.sid)
                    
                    if cached_sandbox_id:
                        try:
                            self.sandbox = E2BBox(self.config.sandbox, sandbox_id=cached_sandbox_id)
                            logger.info(f"Successfully attached to existing E2B sandbox: {cached_sandbox_id}")
                        except Exception as e:
                            logger.warning(f"Failed to connect to cached sandbox {cached_sandbox_id}: {e}")
                            del self.__class__._sandbox_id_cache[self.sid]
                            self.sandbox = None
                        
                except Exception as e:
                    logger.warning(f"Failed to attach to existing sandbox: {e}. Will create a new one.")
            
            # Create E2B sandbox if not provided
            if self.sandbox is None:
                try:
                    self.sandbox = E2BSandbox(self.config.sandbox)
                    sandbox_id = self.sandbox.sandbox.sandbox_id
                    logger.info(f"E2B sandbox created with ID: {sandbox_id}")
                    
                    self.__class__._sandbox_id_cache[self.sid] = sandbox_id
                except Exception as e:
                    logger.error(f"Failed to create E2B sandbox: {e}")
                    raise
            
            if not isinstance(self.sandbox, (E2BSandbox, E2BBox)):
                raise ValueError("E2BRuntime requires an E2BSandbox or E2BBox")
            
            self.file_store = E2BFileStore(self.sandbox.filesystem)
            
            # E2B doesn't use action execution server - set dummy URL
            self.api_url = "direct://e2b-sandbox"
            
            workspace_dir = self.config.workspace_mount_path_in_sandbox
            if workspace_dir:
                try:
                    exit_code, output = self.sandbox.execute(f"sudo mkdir -p {workspace_dir}")
                    if exit_code == 0:
                        self.sandbox.execute(f"sudo chmod 777 {workspace_dir}")
                        logger.info(f"Created workspace directory: {workspace_dir}")
                    else:
                        logger.warning(f"Failed to create workspace directory: {output}")
                except Exception as e:
                    logger.warning(f"Failed to create workspace directory: {e}")
            
            await call_sync_from_async(self.setup_initial_env)
            
            self._runtime_initialized = True
            self.set_runtime_status(RuntimeStatus.READY)
            logger.info("E2B runtime connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect E2B runtime: {e}")
            self.set_runtime_status(RuntimeStatus.FAILED)
            raise

    async def close(self) -> None:
        """Close the E2B runtime."""
        if self._runtime_closed:
            return
            
        self._runtime_closed = True
        
        if self.sandbox:
            try:
                
                if not self.attach_to_existing:
                    self.sandbox.close()
                    if self.sid in self.__class__._sandbox_id_cache:
                        del self.__class__._sandbox_id_cache[self.sid]
                    logger.info("E2B sandbox closed and removed from cache")
                else:
                    logger.info("E2B runtime connection closed, sandbox kept running for reuse")
                    
            except Exception as e:
                logger.error(f"Error closing E2B sandbox: {e}")
        
        parent_close = super().close()
        if parent_close is not None:
            await parent_close

    def run(self, action: CmdRunAction) -> Observation:
        """Execute command using E2B's native execute method."""
        if self.sandbox is None:
            return ErrorObservation("E2B sandbox not initialized")
            
        try:
            timeout = action.timeout if action.timeout else self.config.sandbox.timeout
            exit_code, output = self.sandbox.execute(action.command, timeout=timeout)
            return CmdOutputObservation(
                content=output,
                command=action.command,
                exit_code=exit_code
            )
        except Exception as e:
            return ErrorObservation(f"Failed to execute command: {e}")
    
    def run_ipython(self, action: IPythonRunCellAction) -> Observation:
        """Execute IPython code using E2B's code interpreter."""
        if self.sandbox is None:
            return ErrorObservation("E2B sandbox not initialized")
        
        try:
            result = self.sandbox.sandbox.run_code(action.code)
            
            outputs = []
            if hasattr(result, 'results') and result.results:
                for r in result.results:
                    if hasattr(r, 'text') and r.text:
                        outputs.append(r.text)
                    elif hasattr(r, 'html') and r.html:
                        outputs.append(r.html)
                    elif hasattr(r, 'png') and r.png:
                        outputs.append(f"[Image data: {len(r.png)} bytes]")
                        
            if hasattr(result, 'error') and result.error:
                return ErrorObservation(f"IPython error: {result.error}")
                
            return IPythonRunCellObservation(
                content='\n'.join(outputs) if outputs else '',
                code=action.code
            )
        except Exception as e:
            return ErrorObservation(f"Failed to execute IPython code: {e}")

    def read(self, action: FileReadAction) -> Observation:
        if self.file_store is None:
            return ErrorObservation("E2B file store not initialized. Call connect() first.")
            
        try:
            content = self.file_store.read(action.path)
            lines = read_lines(content.split("\n"), action.start, action.end)
            code_view = "".join(lines)
            return FileReadObservation(code_view, path=action.path)
        except Exception as e:
            return ErrorObservation(f"Failed to read file: {e}")

    def write(self, action: FileWriteAction) -> Observation:
        if self.file_store is None:
            return ErrorObservation("E2B file store not initialized. Call connect() first.")
            
        try:
            if action.start == 0 and action.end == -1:
                self.file_store.write(action.path, action.content)
                return FileWriteObservation(content="", path=action.path)
            
            files = self.file_store.list(action.path)
            if action.path in files:
                all_lines = self.file_store.read(action.path).split("\n")
                new_file = insert_lines(
                    action.content.split("\n"), all_lines, action.start, action.end
                )
                self.file_store.write(action.path, "".join(new_file))
                return FileWriteObservation("", path=action.path)
            else:
                # Create a new file
                self.file_store.write(action.path, action.content)
                return FileWriteObservation(content="", path=action.path)
        except Exception as e:
            return ErrorObservation(f"Failed to write file: {e}")
    
    def edit(self, action: FileEditAction) -> Observation:
        """Edit a file using E2B's file system."""
        if self.file_store is None:
            return ErrorObservation("E2B file store not initialized. Call connect() first.")
            
        try:
            if action.path in self.file_store.list(action.path):
                content = self.file_store.read(action.path)
            else:
                return ErrorObservation(f"File {action.path} not found")
            
            lines = content.split('\n')
            if action.start < 0 or action.end > len(lines):
                return ErrorObservation(f"Invalid line range: {action.start}-{action.end}")
                
            new_lines = lines[:action.start] + action.content.split('\n') + lines[action.end:]
            new_content = '\n'.join(new_lines)
            
            self.file_store.write(action.path, new_content)
            
            return FileEditObservation(
                content='',
                path=action.path,
                old_content='\n'.join(lines[action.start:action.end]),
                start=action.start,
                end=action.end
            )
        except Exception as e:
            return ErrorObservation(f"Failed to edit file: {e}")
    
    def browse(self, action: BrowseURLAction) -> Observation:
        """Browse a URL using E2B's browser capabilities."""
        if self.sandbox is None:
            return ErrorObservation("E2B sandbox not initialized")
            
        try:
            exit_code, output = self.sandbox.execute(f"curl -s -L '{action.url}'")
            if exit_code != 0:
                exit_code, output = self.sandbox.execute(f"wget -qO- '{action.url}'")
                
            if exit_code != 0:
                return ErrorObservation(f"Failed to fetch URL: {output}")
                
            return BrowserOutputObservation(
                content=output,
                url=action.url,
                screenshot=None,
                error=None if exit_code == 0 else output
            )
        except Exception as e:
            return ErrorObservation(f"Failed to browse URL: {e}")
    
    def browse_interactive(self, action: BrowseInteractiveAction) -> Observation:
        """Interactive browsing is not supported in E2B."""
        return ErrorObservation(
            "Interactive browsing is not supported in E2B runtime. "
            "Use browse() for simple URL fetching or consider using a different runtime."
        )
    
    def list_files(self, path: str | None = None) -> list[str]:
        """List files in the sandbox."""
        if self.sandbox is None:
            logger.warning("Cannot list files: E2B sandbox not initialized")
            return []
            
        if path is None:
            path = self.config.workspace_mount_path_in_sandbox or '/workspace'
            
        try:
            exit_code, output = self.sandbox.execute(f"find {path} -maxdepth 1 -type f -o -type d")
            if exit_code == 0:
                files = [line.strip() for line in output.strip().split('\n') if line.strip()]
                return [f.replace(path + '/', '') if f.startswith(path + '/') else f for f in files]
            else:
                logger.warning(f"Failed to list files in {path}: {output}")
                return []
        except Exception as e:
            logger.warning(f"Error listing files: {e}")
            return []
    
    def add_env_vars(self, env_vars: dict[str, str]) -> None:
        """Add environment variables to the E2B sandbox."""
        if self.sandbox is None:
            logger.warning("Cannot add env vars: E2B sandbox not initialized")
            return
            
        if not hasattr(self, '_env_vars'):
            self._env_vars = {}
        self._env_vars.update(env_vars)
        
        for key, value in env_vars.items():
            try:
                escaped_value = value.replace("'", "'\"'\"'")
                cmd = f"export {key}='{escaped_value}'"
                self.sandbox.execute(cmd)
                logger.debug(f"Set env var: {key}")
            except Exception as e:
                logger.warning(f"Failed to set env var {key}: {e}")
    
    def get_working_directory(self) -> str:
        """Get the current working directory."""
        if self.sandbox is None:
            return self.config.workspace_mount_path_in_sandbox or '/workspace'
        try:
            exit_code, output = self.sandbox.execute("pwd")
            if exit_code == 0:
                return output.strip()
        except Exception:
            pass
        return self.config.workspace_mount_path_in_sandbox or '/workspace'
    
    def get_mcp_config(self, extra_stdio_servers: list | None = None) -> dict:
        """Get MCP configuration for E2B runtime."""
        return {
            'stdio_servers': extra_stdio_servers or []
        }
    
    def check_if_alive(self) -> None:
        """Check if the E2B sandbox is alive."""
        if self.sandbox is None:
            raise RuntimeError("E2B sandbox not initialized")
        return
    
    def copy_to(self, host_src: str, sandbox_dest: str, recursive: bool = False) -> None:
        """Copy files to the E2B sandbox."""
        if self.sandbox is None:
            raise RuntimeError("E2B sandbox not initialized")
        self.sandbox.copy_to(host_src, sandbox_dest, recursive)
    
    def get_vscode_token(self) -> str:
        """E2B doesn't support VSCode integration."""
        return ""
    
    @classmethod
    def setup(cls, config: OpenHandsConfig, headless_mode: bool = False) -> None:
        """Set up the E2B runtime environment."""
        logger.info("E2B runtime setup called")
        pass
    
    @classmethod
    def teardown(cls, config: OpenHandsConfig) -> None:
        """Tear down the E2B runtime environment."""
        logger.info("E2B runtime teardown called")
        pass
