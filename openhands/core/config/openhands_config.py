# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import os
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from openhands.core import logger
from openhands.core.config.agent_config import AgentConfig
from openhands.core.config.config_utils import (
    DEFAULT_WORKSPACE_MOUNT_PATH_IN_SANDBOX,
    OH_DEFAULT_AGENT,
    OH_MAX_ITERATIONS,
    model_defaults_to_dict,
)
from openhands.core.config.extended_config import ExtendedConfig
from openhands.core.config.kubernetes_config import KubernetesConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.core.config.mcp_config import MCPConfig
from openhands.core.config.sandbox_config import SandboxConfig
from openhands.core.config.security_config import SecurityConfig


class OpenHandsConfig(BaseModel):
    """Configuration for the app.

    Attributes:
        llms: Dictionary mapping LLM names to their configurations.
            The default configuration is stored under the 'llm' key.
        agents: Dictionary mapping agent names to their configurations.
            The default configuration is stored under the 'agent' key.
        default_agent: Name of the default agent to use.
        sandbox: Sandbox configuration settings.
        security: Security configuration settings.
        runtime: Runtime environment identifier.
        file_store: Type of file store to use.
        file_store_path: Path to the file store.
        file_store_web_hook_url: Optional url for file store web hook
        file_store_web_hook_headers: Optional headers for file_store web hook
        enable_browser: Whether to enable the browser environment
        save_trajectory_path: Either a folder path to store trajectories with auto-generated filenames, or a designated trajectory file path.
        save_screenshots_in_trajectory: Whether to save screenshots in trajectory (in encoded image format).
        replay_trajectory_path: Path to load trajectory and replay. If provided, trajectory would be replayed first before user's instruction.
        search_api_key: API key for Tavily search engine (https://tavily.com/).
        workspace_base (deprecated): Base path for the workspace. Defaults to `./workspace` as absolute path.
        workspace_mount_path (deprecated): Path to mount the workspace. Defaults to `workspace_base`.
        workspace_mount_path_in_sandbox (deprecated): Path to mount the workspace in sandbox. Defaults to `/workspace`.
        workspace_mount_rewrite (deprecated): Path to rewrite the workspace mount path.
        cache_dir: Path to cache directory. Defaults to `/tmp/cache`.
        run_as_openhands: Whether to run as openhands.
        max_iterations: Maximum number of iterations allowed.
        max_budget_per_task: Maximum budget per task, agent stops if exceeded.
        disable_color: Whether to disable terminal colors. For terminals that don't support color.
        debug: Whether to enable debugging mode.
        file_uploads_max_file_size_mb: Maximum file upload size in MB. `0` means unlimited.
        file_uploads_restrict_file_types: Whether to restrict upload file types.
        file_uploads_allowed_extensions: Allowed file extensions. `['.*']` allows all.
        cli_multiline_input: Whether to enable multiline input in CLI. When disabled,
            input is read line by line. When enabled, input continues until /exit command.
        mcp_host: Host for OpenHands' default MCP server
        mcp: MCP configuration settings.
        git_user_name: Git user name for commits made by the agent.
        git_user_email: Git user email for commits made by the agent.
    """

    llms: dict[str, LLMConfig] = Field(default_factory=dict)
    agents: dict[str, AgentConfig] = Field(default_factory=dict)
    default_agent: str = Field(default=OH_DEFAULT_AGENT)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    extended: ExtendedConfig = Field(default_factory=lambda: ExtendedConfig({}))
    runtime: str = Field(default='docker')
    file_store: str = Field(default='local')
    file_store_path: str = Field(default='~/.openhands')
    file_store_web_hook_url: str | None = Field(default=None)
    file_store_web_hook_headers: dict | None = Field(default=None)
    file_store_web_hook_batch: bool = Field(default=False)
    enable_browser: bool = Field(default=True)
    save_trajectory_path: str | None = Field(default=None)
    save_screenshots_in_trajectory: bool = Field(default=False)
    replay_trajectory_path: str | None = Field(default=None)
    search_api_key: SecretStr | None = Field(
        default=None,
        description='API key for Tavily search engine (https://tavily.com/). Required for search functionality.',
    )

    workspace_base: str | None = Field(default=None)
    workspace_mount_path_in_sandbox: str = Field(
        default=DEFAULT_WORKSPACE_MOUNT_PATH_IN_SANDBOX
    )

    # Deprecated parameters - will be removed in a future version
    workspace_mount_path: str | None = Field(default=None)
    workspace_mount_rewrite: str | None = Field(default=None)
    # End of deprecated parameters

    cache_dir: str = Field(default='/tmp/cache')
    run_as_openhands: bool = Field(default=True)
    max_iterations: int = Field(default=OH_MAX_ITERATIONS)
    max_budget_per_task: float | None = Field(default=None)
    init_git_in_empty_workspace: bool = Field(default=False)

    disable_color: bool = Field(default=False)
    jwt_secret: SecretStr | None = Field(default=None)
    debug: bool = Field(default=False)
    file_uploads_max_file_size_mb: int = Field(default=0)
    file_uploads_restrict_file_types: bool = Field(default=False)
    file_uploads_allowed_extensions: list[str] = Field(default_factory=lambda: ['.*'])

    cli_multiline_input: bool = Field(default=False)
    conversation_max_age_seconds: int = Field(default=864000)  # 10 days in seconds
    enable_default_condenser: bool = Field(default=True)
    max_concurrent_conversations: int = Field(
        default=3
    )  # Maximum number of concurrent agent loops allowed per user
    client_wait_timeout: int = Field(
        default=30,
        description='Timeout in seconds for waiting for websocket client connection during initialization',
    )
    mcp_host: str = Field(default=f'localhost:{os.getenv("port", 3000)}')
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    kubernetes: KubernetesConfig = Field(default_factory=KubernetesConfig)
    git_user_name: str = Field(
        default='openhands', description='Git user name for commits made by the agent'
    )
    git_user_email: str = Field(
        default='openhands@all-hands.dev',
        description='Git user email for commits made by the agent',
    )

    defaults_dict: ClassVar[dict] = {}

    model_config = ConfigDict(extra='forbid')

    def get_llm_config(self, name: str = 'llm') -> LLMConfig:
        """'llm' is the name for default config (for backward compatibility prior to 0.8)."""
        if name in self.llms:
            return self.llms[name]
        if name is not None and name != 'llm':
            logger.openhands_logger.warning(
                f'llm config group {name} not found, using default config'
            )
        if 'llm' not in self.llms:
            self.llms['llm'] = LLMConfig()
        return self.llms['llm']

    def set_llm_config(self, value: LLMConfig, name: str = 'llm') -> None:
        self.llms[name] = value

    def get_agent_config(self, name: str = 'agent') -> AgentConfig:
        """'agent' is the name for default config (for backward compatibility prior to 0.8)."""
        if name in self.agents:
            return self.agents[name]
        if 'agent' not in self.agents:
            self.agents['agent'] = AgentConfig()
        return self.agents['agent']

    def set_agent_config(self, value: AgentConfig, name: str = 'agent') -> None:
        self.agents[name] = value

    def get_agent_to_llm_config_map(self) -> dict[str, LLMConfig]:
        """Get a map of agent names to llm configs."""
        return {name: self.get_llm_config_from_agent(name) for name in self.agents}

    def get_llm_config_from_agent_config(self, agent_config: AgentConfig):
        llm_config_name = (
            agent_config.llm_config if agent_config.llm_config is not None else 'llm'
        )
        return self.get_llm_config(llm_config_name)

    def get_llm_config_from_agent(self, name: str = 'agent') -> LLMConfig:
        agent_config: AgentConfig = self.get_agent_config(name)
        return self.get_llm_config_from_agent_config(agent_config)

    def get_agent_configs(self) -> dict[str, AgentConfig]:
        return self.agents

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook, called when the instance is created with only default values."""
        super().model_post_init(__context)

        if not OpenHandsConfig.defaults_dict:  # Only set defaults_dict if it's empty
            OpenHandsConfig.defaults_dict = model_defaults_to_dict(self)
