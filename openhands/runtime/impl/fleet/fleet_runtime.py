import inspect
import time
from typing import Any, Dict, List
from pathlib import Path

from openhands.runtime.base import Runtime
from openhands.core.config import OpenHandsConfig
from openhands.events.action import (
    CmdRunAction,
    FileReadAction,
    FileWriteAction,
    FileEditAction,
    BrowseInteractiveAction,
    BrowseURLAction,
    MCPAction
)
from openhands.events.observation import (
    Observation,
    ErrorObservation,
    MCPObservation,
)
from openhands.events.observation.mcp import MCPImage
from openhands.core.logger import openhands_logger as logger

# Canonical OpenEnv import surface (no guessing / fallback imports).
try:
    from openenv.fleet import FleetEnvClient, FleetMCPTools  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    FleetEnvClient = None  # type: ignore[assignment]
    FleetMCPTools = None  # type: ignore[assignment]

class OpenEnvRuntime(Runtime):
    """
    Runtime implementation that connects to a remote Fleet environment via OpenEnv.
    Uses OpenEnv's FleetEnvClient (orchestration) and FleetMCPTools (MCP actions).
    """

    def __init__(self, config: OpenHandsConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.orch = None
        self.tools = None
        self.available_tools = []

    async def connect(self):
        """Provision and connect to the Fleet environment."""
        if FleetEnvClient is None or FleetMCPTools is None:
            raise ImportError(
                "OpenEnv runtime requires OpenEnv Fleet support. "
                "Install it (or use `poetry install --with fleet`): pip install 'openenv[fleet]'"
            )

        api_key = self.config.sandbox.fleet_api_key
        env_key = self.config.sandbox.fleet_env_key

        if not api_key:
            raise ValueError("fleet_api_key is required for OpenEnvRuntime")
        if not env_key:
            raise ValueError("fleet_env_key is required for OpenEnvRuntime")

        self.log('info', f'Connecting to Fleet environment: {env_key}')

        try:
            # 1) Provision remote instance + tool handle via OpenEnv.
            # OpenEnv returns: (orchestrator HTTP client, MCP tools client).
            self.orch, self.tools = FleetEnvClient.from_fleet(
                api_key=api_key,
                env_key=env_key,
                image_type="mcp",
            )

            self.log('info', 'Discovering tools...')
            # 2) Reset the episode (orch is usually sync; tolerate async impls)
            self.log('info', 'Resetting environment...')
            maybe_reset = self.orch.reset()
            if inspect.isawaitable(maybe_reset):
                await maybe_reset

            tool_list_action = await self.tools.list_tools()
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
        msg = f'[OpenEnvRuntime {self.sid}] {message}'
        getattr(logger, level)(msg, stacklevel=2)

    # --- Action Mapping ---

    async def run(self, action: CmdRunAction) -> Observation:
        """Map CmdRunAction to Fleet's 'bash' or 'computer' tool."""
        return ErrorObservation(
            "CmdRunAction is not supported in OpenEnvRuntime. "
            "Fleet environments expose a unified action space via MCP tools; use MCPAction."
        )

    async def read(self, action: FileReadAction) -> Observation:
        """Map FileReadAction to Fleet's file reading tool."""
        return ErrorObservation(
            "FileReadAction is not supported in OpenEnvRuntime. "
            "Use MCPAction with the environment's file tool (if exposed) or computer/browser tools."
        )

    async def write(self, action: FileWriteAction) -> Observation:
        return ErrorObservation(
            "FileWriteAction is not supported in OpenEnvRuntime. "
            "Use MCPAction with the environment's file tool (if exposed)."
        )

    async def edit(self, action: FileEditAction) -> Observation:
        # For complex editing, we might use a specific tool or fallback to generic replacement
        # if the Fleet env supports str_replace_editor
        return ErrorObservation("FileEditAction not yet fully implemented for OpenEnvRuntime")

    async def browse(self, action: BrowseURLAction) -> Observation:
        return ErrorObservation("BrowseURLAction not implemented. Use 'computer' tool via MCPAction instead.")

    async def browse_interactive(self, action: BrowseInteractiveAction) -> Observation:
        return ErrorObservation("BrowseInteractiveAction not implemented. Use 'computer' tool via MCPAction instead.")

    async def call_tool_mcp(self, action: MCPAction) -> Observation:
        """Directly pass MCP actions to Fleet and parse results including images."""
        start = time.monotonic()
        args_preview: Dict[str, Any] = {}
        try:
            if isinstance(action.arguments, dict):
                # Avoid logging huge payloads (e.g., screenshots); only log keys + small primitives.
                for k, v in action.arguments.items():
                    if isinstance(v, (str, int, float, bool)) and len(str(v)) <= 200:
                        args_preview[k] = v
                    else:
                        args_preview[k] = f'<{type(v).__name__}>'
        except Exception:
            args_preview = {}

        self.log('debug', f'MCP call -> {action.name} args={args_preview}')
        try:
            result = await self.tools.call_tool(action.name, action.arguments)
            dur_ms = int((time.monotonic() - start) * 1000)

            # Parse MCP result to extract text and images
            text_content, images, is_error = self._parse_mcp_result(result)

            self.log(
                'debug',
                f'MCP result <- {action.name} ({dur_ms}ms): '
                f'{len(text_content)} chars, {len(images)} images',
            )

            obs = MCPObservation(
                content=text_content,
                name=action.name,
                arguments=action.arguments or {},
                images=images,
                is_error=is_error,
            )
            return obs
        except Exception as e:
            dur_ms = int((time.monotonic() - start) * 1000)
            self.log('warning', f'MCP error <- {action.name} ({dur_ms}ms): {e}')
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
        raise NotImplementedError("copy_to not implemented for OpenEnvRuntime")

    def list_files(self, path: str | None = None) -> List[str]:
        # Could map to 'ls' or 'list_files' tool
        return []

    def copy_from(self, path: str) -> Path:
        raise NotImplementedError("copy_from not implemented for OpenEnvRuntime")

    def run_ipython(self, action):
        return ErrorObservation("IPython not supported")

    def get_mcp_config(self, extra_stdio_servers=None):
        # We don't use local MCP servers with Fleet, Fleet IS the MCP server
        # But we need to return something if called
        return self.config.mcp

    @property
    def vscode_url(self) -> str | None:
        return None
