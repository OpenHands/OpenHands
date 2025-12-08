from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from openhands.core.config.condenser_config import (
    CondenserConfig,
    ConversationWindowCondenserConfig,
)
from openhands.core.config.extended_config import ExtendedConfig
from openhands.core.config.model_routing_config import ModelRoutingConfig
from openhands.core.logger import openhands_logger as logger
from openhands.utils.import_utils import get_impl


class AgentConfig(BaseModel):
    cli_mode: bool = Field(default=False)
    """Whether the agent is running in CLI mode. This can be used to disable certain tools that are not supported in CLI mode."""
    llm_config: str | None = Field(default=None)
    """The name of the llm config to use. If specified, this will override global llm config."""
    classpath: str | None = Field(default=None)
    """The classpath of the agent to use. To be used for custom agents that are not defined in the openhands.agenthub package."""
    system_prompt_filename: str = Field(default='system_prompt.j2')
    """Filename of the system prompt template file within the agent's prompt directory. Defaults to 'system_prompt.j2'."""
    enable_browsing: bool = Field(default=True)
    """Whether to enable browsing tool.
    Note: If using CLIRuntime, browsing is not implemented and should be disabled."""
    enable_llm_editor: bool = Field(default=False)
    """Whether to enable LLM editor tool"""
    enable_editor: bool = Field(default=True)
    """Whether to enable the standard editor tool (str_replace_editor), only has an effect if enable_llm_editor is False."""
    enable_jupyter: bool = Field(default=True)
    """Whether to enable Jupyter tool.
    Note: If using CLIRuntime, Jupyter use is not implemented and should be disabled."""
    enable_cmd: bool = Field(default=True)
    """Whether to enable bash tool"""
    enable_think: bool = Field(default=True)
    """Whether to enable think tool"""
    enable_finish: bool = Field(default=True)
    """Whether to enable finish tool"""
    enable_condensation_request: bool = Field(default=False)
    """Whether to enable condensation request tool"""
    enable_prompt_extensions: bool = Field(default=True)
    """Whether to enable prompt extensions"""
    enable_mcp: bool = Field(default=True)
    """Whether to enable MCP tools"""
    disabled_microagents: list[str] = Field(default_factory=list)
    """A list of microagents to disable (by name, without .py extension, e.g. ["github", "lint"]). Default is None."""
    enable_history_truncation: bool = Field(default=True)
    """Whether history should be truncated to continue the session when hitting LLM context length limit."""
    enable_som_visual_browsing: bool = Field(default=True)
    """Whether to enable SoM (Set of Marks) visual browsing."""
    enable_plan_mode: bool = Field(default=True)
    """Whether to enable plan mode, which uses the long horizon system message and add the new tool - task_tracker - for planning, tracking and executing complex tasks."""
    enable_stuck_detection: bool = Field(default=True)
    """Whether to enable stuck/loop detection. When disabled, the agent will not automatically detect and recover from loops."""
    condenser: CondenserConfig = Field(
        # The default condenser is set to the conversation window condenser -- if
        # we use NoOp and the conversation hits the LLM context length limit,
        # the agent will generate a condensation request which will never be
        # handled.
        default_factory=lambda: ConversationWindowCondenserConfig()
    )
    model_routing: ModelRoutingConfig = Field(default_factory=ModelRoutingConfig)
    """Model routing configuration settings."""
    extended: ExtendedConfig = Field(default_factory=lambda: ExtendedConfig({}))
    """Extended configuration for the agent."""
    runtime: str | None = Field(default=None)
    """Runtime type (e.g., 'docker', 'local', 'cli') used for runtime-specific tool behavior."""

    model_config = ConfigDict(extra='forbid')

    @property
    def resolved_system_prompt_filename(self) -> str:
        """
        Returns the appropriate system prompt filename based on the agent configuration.
        When enable_plan_mode is True, automatically uses the long horizon system prompt
        unless a custom system_prompt_filename was explicitly set (not the default).
        """
        if self.enable_plan_mode and self.system_prompt_filename == 'system_prompt.j2':
            return 'system_prompt_long_horizon.j2'
        return self.system_prompt_filename

    @classmethod
    def from_toml_section(cls, data: dict) -> dict[str, AgentConfig]:
        """Create a mapping of AgentConfig instances from a toml dictionary representing the [agent] section.

        The default configuration is built from all non-dict keys in data.
        Then, each key with a dict value is treated as a custom agent configuration, and its values override
        the default configuration.

        Example:
        Apply generic agent config with custom agent overrides, e.g.
            [agent]
            enable_prompt_extensions = false
            [agent.BrowsingAgent]
            enable_prompt_extensions = true
        results in prompt_extensions being true for BrowsingAgent but false for others.

        Returns:
            dict[str, AgentConfig]: A mapping where the key "agent" corresponds to the default configuration
            and additional keys represent custom configurations.
        """
        # Initialize the result mapping
        agent_mapping: dict[str, AgentConfig] = {}

        # Extract base config data (non-dict values)
        base_data = {}
        custom_sections: dict[str, dict] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                custom_sections[key] = value
            else:
                base_data[key] = value

        # Try to create the base config
        try:
            base_config = cls.model_validate(base_data)
            agent_mapping['agent'] = base_config
        except ValidationError as e:
            logger.warning(f'Invalid base agent configuration: {e}. Using defaults.')
            # If base config fails, create a default one
            base_config = cls()
            # Still add it to the mapping
            agent_mapping['agent'] = base_config

        # Process each custom section independently
        for name, overrides in custom_sections.items():
            try:
                # Merge base config with overrides
                merged = {**base_config.model_dump(), **overrides}
                if merged.get('classpath'):
                    # if an explicit classpath is given, try to load it and look up its config model class
                    from openhands.controller.agent import Agent

                    try:
                        agent_cls = get_impl(Agent, merged.get('classpath'))
                        custom_config = agent_cls.config_model.model_validate(merged)
                    except Exception as e:
                        logger.warning(
                            f'Failed to load custom agent class [{merged.get("classpath")}]: {e}. Using default config model.'
                        )
                        custom_config = cls.model_validate(merged)
                else:
                    # otherwise, try to look up the agent class by name (i.e. if it's a built-in)
                    # if that fails, just use the default AgentConfig class.
                    try:
                        agent_cls = Agent.get_cls(name)
                        custom_config = agent_cls.config_model.model_validate(merged)
                    except Exception:
                        # otherwise, just fall back to the default config model
                        custom_config = cls.model_validate(merged)
                agent_mapping[name] = custom_config
            except ValidationError as e:
                logger.warning(
                    f'Invalid agent configuration for [{name}]: {e}. This section will be skipped.'
                )
                # Skip this custom section but continue with others
                continue

        return agent_mapping
