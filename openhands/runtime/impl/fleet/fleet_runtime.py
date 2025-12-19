import inspect
import time
from typing import Any, Dict, List
import copy
from pathlib import Path

from openhands.runtime.base import Runtime
from openhands.core.config import OpenHandsConfig
from openhands.events.action import (
    Action,
    CmdRunAction,
    FileReadAction,
    FileWriteAction,
    FileEditAction,
    BrowseInteractiveAction,
    BrowseURLAction,
    MCPAction
)
from openhands.events.observation import (
    CmdOutputObservation,
    FileReadObservation,
    FileWriteObservation,
    FileEditObservation,
    BrowserOutputObservation,
    Observation,
    ErrorObservation,
    MCPObservation,
)
from openhands.events.observation.mcp import MCPImage
from openhands.core.logger import openhands_logger as logger
from openhands.runtime.impl.fleet.trace_manager import FleetTraceManager

# Placeholder imports for Fleet SDK - handling import errors gracefully
try:
    from openenv.fleet import FleetEnvClient, FleetMCPTools  # type: ignore[import-not-found]
except ImportError:
    FleetEnvClient = None
    FleetMCPTools = None

class FleetRuntime(Runtime):
    """
    Runtime implementation that connects to a remote Fleet environment.
    Uses FleetEnvClient for orchestration and FleetMCPTools for action execution.
    """

    def __init__(self, config: OpenHandsConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.orch = None
        self.tools = None
        self.available_tools = []
        self.trace_manager = FleetTraceManager(
            api_url=self.config.sandbox.fleet_trace_api_url,
            api_key=self.config.sandbox.fleet_trace_api_key
        )

    async def connect(self):
        """Provision and connect to the Fleet environment."""
        if not FleetEnvClient:
            raise ImportError(
                "openenv-core[fleet] is not installed. "
                "Please install it with: pip install 'openenv-core[fleet]'"
            )

        api_key = self.config.sandbox.fleet_api_key
        env_key = self.config.sandbox.fleet_env_key

        if not api_key:
            raise ValueError("fleet_api_key is required for FleetRuntime")
        if not env_key:
            raise ValueError("fleet_env_key is required for FleetRuntime")

        self.log('info', f'Connecting to Fleet environment: {env_key}')

        try:
            # 1. Initialize Fleet Clients
            # Note: This is based on the pseudocode in the README.
            # Adjust if the actual SDK signature differs.
            maybe = FleetEnvClient.from_fleet(api_key=api_key, env_key=env_key)
            self.orch, self.tools = await maybe if inspect.isawaitable(maybe) else maybe

            # 2. Reset the remote environment
            self.log('info', 'Resetting environment...')
            await self.orch.reset()

            # 3. Discover available tools
            self.log('info', 'Discovering tools...')
            maybe_tools = self.tools.list_tools()
            tool_list_action = (
                await maybe_tools if inspect.isawaitable(maybe_tools) else maybe_tools
            )
            self.available_tools = tool_list_action.tools
            tool_names = []
            try:
                for tool in self.available_tools:
                    fn = tool.get('function', {}) if isinstance(tool, dict) else {}
                    name = fn.get('name')
                    if name:
                        tool_names.append(name)
            except Exception:
                tool_names = []
            self.log('info', f'Discovered {len(self.available_tools)} tools: {sorted(tool_names)}')

            self._runtime_initialized = True

        except Exception as e:
            self.log('error', f"Failed to connect to Fleet: {e}")
            raise

    # --- Helper for Logging and Tracing ---

    def log(self, level: str, message: str) -> None:
        """Override log to ensure correct formatting"""
        msg = f'[FleetRuntime {self.sid}] {message}'
        getattr(logger, level)(msg, stacklevel=2)

    # --- Action Mapping ---

    async def run(self, action: CmdRunAction) -> Observation:
        """Map CmdRunAction to Fleet's 'bash' or 'computer' tool."""
        self.trace_manager.trace_action("run", {"command": action.command})
        try:
            # OpenHands CmdRunAction expects a shell.
            # We look for a 'bash' or 'computer' tool in Fleet.
            # This is a heuristic; actual tool name depends on the Fleet image.

            tool_name = "bash"
            args = {"command": action.command}

            # Check if 'computer' is the preferred tool (e.g. Anthropic definition)
            # which might handle shell commands differently or we might default to 'bash'
            # For now, let's assume there is a bash-capable tool.

            result = await self.tools.call_tool(tool_name, args)

            # Assuming result structure has content/exit_code
            # Adjust based on actual CallToolResult
            obs = CmdOutputObservation(
                content=str(result.content),
                exit_code=getattr(result, 'exit_code', 0)
            )
            self.trace_manager.trace_observation("run", obs)
            return obs
        except Exception as e:
            self.log('error', f"Error executing run: {e}")
            self.trace_manager.trace_error("run", str(e))
            return ErrorObservation(f"Failed to execute command: {e}")

    async def read(self, action: FileReadAction) -> Observation:
        """Map FileReadAction to Fleet's file reading tool."""
        self.trace_manager.trace_action("read", {"path": action.path})
        try:
            # Assuming a 'read_file' tool exists
            result = await self.tools.call_tool("read_file", {"path": action.path})
            obs = FileReadObservation(content=str(result.content), path=action.path)
            self.trace_manager.trace_observation("read", obs)
            return obs
        except Exception as e:
            self.trace_manager.trace_error("read", str(e))
            return ErrorObservation(f"Failed to read file: {e}")

    async def write(self, action: FileWriteAction) -> Observation:
        self.trace_manager.trace_action("write", {"path": action.path})
        try:
            # Assuming 'write_file' tool
            await self.tools.call_tool("write_file", {"path": action.path, "content": action.content})
            obs = FileWriteObservation(content="", path=action.path)
            self.trace_manager.trace_observation("write", obs)
            return obs
        except Exception as e:
            self.trace_manager.trace_error("write", str(e))
            return ErrorObservation(f"Failed to write file: {e}")

    async def edit(self, action: FileEditAction) -> Observation:
        # For complex editing, we might use a specific tool or fallback to generic replacement
        # if the Fleet env supports str_replace_editor
        return ErrorObservation("FileEditAction not yet fully implemented for FleetRuntime")

    async def browse(self, action: BrowseURLAction) -> Observation:
        return ErrorObservation("BrowseURLAction not implemented. Use 'computer' tool via MCPAction instead.")

    async def browse_interactive(self, action: BrowseInteractiveAction) -> Observation:
        return ErrorObservation("BrowseInteractiveAction not implemented. Use 'computer' tool via MCPAction instead.")

    async def call_tool_mcp(self, action: MCPAction) -> Observation:
        """Directly pass MCP actions to Fleet and parse results including images."""
        start = time.monotonic()
        args_preview: Dict[str, Any] = {}
        try:
            if isinstance(action.tool_args, dict):
                # Avoid logging huge payloads (e.g., screenshots); only log keys + small primitives.
                for k, v in action.tool_args.items():
                    if isinstance(v, (str, int, float, bool)) and len(str(v)) <= 200:
                        args_preview[k] = v
                    else:
                        args_preview[k] = f'<{type(v).__name__}>'
        except Exception:
            args_preview = {}

        self.log('debug', f'MCP call -> {action.tool_name} args={args_preview}')
        self.trace_manager.trace_action(
            'mcp', {'tool': action.tool_name, 'args_preview': args_preview}
        )
        try:
            result = await self.tools.call_tool(action.tool_name, action.tool_args)
            dur_ms = int((time.monotonic() - start) * 1000)

            # Parse MCP result to extract text and images
            text_content, images, is_error = self._parse_mcp_result(result)

            self.log(
                'debug',
                f'MCP result <- {action.tool_name} ({dur_ms}ms): '
                f'{len(text_content)} chars, {len(images)} images',
            )

            obs = MCPObservation(
                content=text_content,
                name=action.tool_name,
                arguments=action.tool_args or {},
                images=images,
                is_error=is_error,
            )
            self.trace_manager.trace_observation('mcp', obs)
            return obs
        except Exception as e:
            dur_ms = int((time.monotonic() - start) * 1000)
            self.log('warning', f'MCP error <- {action.tool_name} ({dur_ms}ms): {e}')
            self.trace_manager.trace_error('mcp', str(e))
            return ErrorObservation(f'MCP Tool call failed: {e}')

    def _parse_mcp_result(self, result: Any) -> tuple[str, list[MCPImage], bool]:
        """Parse an MCP CallToolResult to extract text content and images.

        MCP results have a .content list with items like:
        - TextContent(type="text", text="...")
        - ImageContent(type="image", data="base64...", mimeType="image/png")

        Returns:
            (text_content, images, is_error)
        """
        text_parts: list[str] = []
        images: list[MCPImage] = []
        is_error = False

        # Check for error flag
        if hasattr(result, 'isError'):
            is_error = bool(result.isError)

        # Get content list
        content_list = getattr(result, 'content', None)
        if content_list is None:
            # Fallback: result might be a dict
            if isinstance(result, dict):
                content_list = result.get('content', [])
                is_error = result.get('isError', False)
            else:
                # Unknown format, stringify
                return str(result), [], False

        # Parse each content item
        for item in content_list:
            item_type = self._get_attr_or_key(item, 'type')

            if item_type == 'text':
                text = self._get_attr_or_key(item, 'text', '')
                if text:
                    text_parts.append(text)

            elif item_type == 'image':
                data = self._get_attr_or_key(item, 'data', '')
                mime_type = self._get_attr_or_key(item, 'mimeType', 'image/png')
                if data:
                    images.append(MCPImage(data=data, mime_type=mime_type))

        text_content = '\n'.join(text_parts) if text_parts else ''
        return text_content, images, is_error

    def _get_attr_or_key(self, obj: Any, key: str, default: Any = None) -> Any:
        """Get attribute or dict key from an object."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # --- Required Abstract Methods (Stubs) ---
    # These might need actual implementation depending on how deep the integration goes

    def copy_to(self, host_src: str, sandbox_dest: str, recursive: bool = False):
        raise NotImplementedError("copy_to not implemented for FleetRuntime")

    def list_files(self, path: str | None = None) -> List[str]:
        # Could map to 'ls' or 'list_files' tool
        return []

    def copy_from(self, path: str) -> Path:
        raise NotImplementedError("copy_from not implemented for FleetRuntime")

    def run_ipython(self, action):
        return ErrorObservation("IPython not supported")

    def get_mcp_config(self, extra_stdio_servers=None):
        # We don't use local MCP servers with Fleet, Fleet IS the MCP server
        # But we need to return something if called
        return self.config.mcp

    @property
    def vscode_url(self) -> str | None:
        return None
