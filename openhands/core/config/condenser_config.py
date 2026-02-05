# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from openhands.core import logger
from openhands.core.config.llm_config import LLMConfig


class NoOpCondenserConfig(BaseModel):
    """Configuration for NoOpCondenser."""

    type: Literal['noop'] = Field(default='noop')

    model_config = ConfigDict(extra='forbid')


class ObservationMaskingCondenserConfig(BaseModel):
    """Configuration for ObservationMaskingCondenser."""

    type: Literal['observation_masking'] = Field(default='observation_masking')
    attention_window: int = Field(
        default=100,
        description='The number of most-recent events where observations will not be masked.',
        ge=1,
    )

    model_config = ConfigDict(extra='forbid')


class BrowserOutputCondenserConfig(BaseModel):
    """Configuration for the BrowserOutputCondenser."""

    type: Literal['browser_output_masking'] = Field(default='browser_output_masking')
    attention_window: int = Field(
        default=1,
        description='The number of most recent browser output observations that will not be masked.',
        ge=1,
    )


class RecentEventsCondenserConfig(BaseModel):
    """Configuration for RecentEventsCondenser."""

    type: Literal['recent'] = Field(default='recent')

    # at least one event by default, because the best guess is that it is the user task
    keep_first: int = Field(
        default=1,
        description='The number of initial events to condense.',
        ge=0,
    )
    max_events: int = Field(
        default=100, description='Maximum number of events to keep.', ge=1
    )

    model_config = ConfigDict(extra='forbid')


class LLMSummarizingCondenserConfig(BaseModel):
    """Configuration for LLMCondenser."""

    type: Literal['llm'] = Field(default='llm')
    llm_config: LLMConfig = Field(
        ..., description='Configuration for the LLM to use for condensing.'
    )

    # at least one event by default, because the best guess is that it's the user task
    keep_first: int = Field(
        default=1,
        description='Number of initial events to always keep in history.',
        ge=0,
    )
    max_size: int = Field(
        default=100,
        description='Maximum size of the condensed history before triggering forgetting.',
        ge=2,
    )
    max_event_length: int = Field(
        default=10_000,
        description='Maximum length of the event representations to be passed to the LLM.',
    )

    model_config = ConfigDict(extra='forbid')


class AmortizedForgettingCondenserConfig(BaseModel):
    """Configuration for AmortizedForgettingCondenser."""

    type: Literal['amortized'] = Field(default='amortized')
    max_size: int = Field(
        default=100,
        description='Maximum size of the condensed history before triggering forgetting.',
        ge=2,
    )

    # at least one event by default, because the best guess is that it's the user task
    keep_first: int = Field(
        default=1,
        description='Number of initial events to always keep in history.',
        ge=0,
    )

    model_config = ConfigDict(extra='forbid')


class LLMAttentionCondenserConfig(BaseModel):
    """Configuration for LLMAttentionCondenser."""

    type: Literal['llm_attention'] = Field(default='llm_attention')
    llm_config: LLMConfig = Field(
        ..., description='Configuration for the LLM to use for attention.'
    )
    max_size: int = Field(
        default=100,
        description='Maximum size of the condensed history before triggering forgetting.',
        ge=2,
    )

    # at least one event by default, because the best guess is that it's the user task
    keep_first: int = Field(
        default=1,
        description='Number of initial events to always keep in history.',
        ge=0,
    )

    model_config = ConfigDict(extra='forbid')


class StructuredSummaryCondenserConfig(BaseModel):
    """Configuration for StructuredSummaryCondenser instances."""

    type: Literal['structured'] = Field(default='structured')
    llm_config: LLMConfig = Field(
        ..., description='Configuration for the LLM to use for condensing.'
    )

    # at least one event by default, because the best guess is that it's the user task
    keep_first: int = Field(
        default=1,
        description='Number of initial events to always keep in history.',
        ge=0,
    )
    max_size: int = Field(
        default=100,
        description='Maximum size of the condensed history before triggering forgetting.',
        ge=2,
    )
    max_event_length: int = Field(
        default=10_000,
        description='Maximum length of the event representations to be passed to the LLM.',
    )

    model_config = ConfigDict(extra='forbid')


class CondenserPipelineConfig(BaseModel):
    """Configuration for the CondenserPipeline."""

    type: Literal['pipeline'] = Field(default='pipeline')
    condensers: list[CondenserConfig] = Field(
        default_factory=list,
        description='List of condenser configurations to be used in the pipeline.',
    )

    model_config = ConfigDict(extra='forbid')


class ConversationWindowCondenserConfig(BaseModel):
    """Configuration for ConversationWindowCondenser.

    Not currently supported by the TOML or ENV_VAR configuration strategies.
    """

    type: Literal['conversation_window'] = Field(default='conversation_window')

    model_config = ConfigDict(extra='forbid')


# Type alias for convenience
CondenserConfig = (
    NoOpCondenserConfig
    | ObservationMaskingCondenserConfig
    | BrowserOutputCondenserConfig
    | RecentEventsCondenserConfig
    | LLMSummarizingCondenserConfig
    | AmortizedForgettingCondenserConfig
    | LLMAttentionCondenserConfig
    | StructuredSummaryCondenserConfig
    | CondenserPipelineConfig
    | ConversationWindowCondenserConfig
)


def condenser_config_from_toml_section(
    data: dict, llm_configs: dict | None = None
) -> dict[str, CondenserConfig]:
    """Create a CondenserConfig instance from a toml dictionary representing the [condenser] section.

    For CondenserConfig, the handling is different since it's a union type. The type of condenser
    is determined by the 'type' field in the section.

    Example:
    Parse condenser config like:
        [condenser]
        type = "noop"

    For condensers that require an LLM config, you can specify the name of an LLM config:
        [condenser]
        type = "llm"
        llm_config = "my_llm"  # References [llm.my_llm] section

    Args:
        data: The TOML dictionary representing the [condenser] section.
        llm_configs: Optional dictionary of LLMConfig objects keyed by name.

    Returns:
        dict[str, CondenserConfig]: A mapping where the key "condenser" corresponds to the configuration.
    """
    # Initialize the result mapping
    condenser_mapping: dict[str, CondenserConfig] = {}

    # Process config
    try:
        # Determine which condenser type to use based on 'type' field
        condenser_type = data.get('type', 'noop')

        # Handle LLM config reference if needed
        if (
            condenser_type in ('llm', 'llm_attention')
            and 'llm_config' in data
            and isinstance(data['llm_config'], str)
        ):
            llm_config_name = data['llm_config']
            if llm_configs and llm_config_name in llm_configs:
                # Replace the string reference with the actual LLMConfig object
                data_copy = data.copy()
                data_copy['llm_config'] = llm_configs[llm_config_name]
                config = create_condenser_config(condenser_type, data_copy)
            else:
                logger.openhands_logger.warning(
                    f"LLM config '{llm_config_name}' not found for condenser. Using default LLMConfig."
                )
                # Create a default LLMConfig if the referenced one doesn't exist
                data_copy = data.copy()
                # Try to use the fallback 'llm' config
                if llm_configs is not None:
                    data_copy['llm_config'] = llm_configs.get('llm')
                config = create_condenser_config(condenser_type, data_copy)
        else:
            config = create_condenser_config(condenser_type, data)

        condenser_mapping['condenser'] = config
    except (ValidationError, ValueError) as e:
        logger.openhands_logger.warning(
            f'Invalid condenser configuration: {e}. Using NoOpCondenserConfig.'
        )
        # Default to NoOpCondenserConfig if config fails
        config = NoOpCondenserConfig(type='noop')
        condenser_mapping['condenser'] = config

    return condenser_mapping


# For backward compatibility
from_toml_section = condenser_config_from_toml_section


def create_condenser_config(condenser_type: str, data: dict) -> CondenserConfig:
    """Create a CondenserConfig instance based on the specified type.

    Args:
        condenser_type: The type of condenser to create.
        data: The configuration data.

    Returns:
        A CondenserConfig instance.

    Raises:
        ValueError: If the condenser type is unknown.
        ValidationError: If the provided data fails validation for the condenser type.
    """
    # Mapping of condenser types to their config classes
    condenser_classes = {
        'noop': NoOpCondenserConfig,
        'observation_masking': ObservationMaskingCondenserConfig,
        'recent': RecentEventsCondenserConfig,
        'llm': LLMSummarizingCondenserConfig,
        'amortized': AmortizedForgettingCondenserConfig,
        'llm_attention': LLMAttentionCondenserConfig,
        'structured': StructuredSummaryCondenserConfig,
        'pipeline': CondenserPipelineConfig,
        'conversation_window': ConversationWindowCondenserConfig,
        'browser_output_masking': BrowserOutputCondenserConfig,
    }

    if condenser_type not in condenser_classes:
        raise ValueError(f'Unknown condenser type: {condenser_type}')

    # Create and validate the config using direct instantiation
    # Explicitly handle ValidationError to provide more context
    try:
        config_class = condenser_classes[condenser_type]
        # Use type casting to help mypy understand the return type
        return cast(CondenserConfig, config_class(**data))
    except ValidationError as e:
        # Just re-raise with a more descriptive message, but don't try to pass the errors
        # which can cause compatibility issues with different pydantic versions
        raise ValueError(
            f"Validation failed for condenser type '{condenser_type}': {e}"
        )
