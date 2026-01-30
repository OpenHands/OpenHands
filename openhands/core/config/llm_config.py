# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from openhands.core.logger import LOG_DIR
from openhands.core.logger import openhands_logger as logger


class LLMConfig(BaseModel):
    """Configuration for the LLM model.

    Attributes:
        model: The model to use.
        api_key: The API key to use.
        base_url: The base URL for the API. This is necessary for local LLMs.
        api_version: The version of the API.
        aws_access_key_id: The AWS access key ID.
        aws_secret_access_key: The AWS secret access key.
        aws_region_name: The AWS region name.
        num_retries: The number of retries to attempt.
        retry_multiplier: The multiplier for the exponential backoff.
        retry_min_wait: The minimum time to wait between retries, in seconds. This is exponential backoff minimum. For models with very low limits, this can be set to 15-20.
        retry_max_wait: The maximum time to wait between retries, in seconds. This is exponential backoff maximum.
        timeout: The timeout for the API.
        max_message_chars: The approximate max number of characters in the content of an event included in the prompt to the LLM. Larger observations are truncated.
        temperature: The temperature for the API.
        top_p: The top p for the API.
        top_k: The top k for the API.
        custom_llm_provider: The custom LLM provider to use. This is undocumented in openhands, and normally not used. It is documented on the litellm side.
        max_input_tokens: The maximum number of input tokens. Note that this is currently unused, and the value at runtime is actually the total tokens in OpenAI (e.g. 128,000 tokens for GPT-4).
        max_output_tokens: The maximum number of output tokens. This is sent to the LLM.
        input_cost_per_token: The cost per input token. This will available in logs for the user to check.
        output_cost_per_token: The cost per output token. This will available in logs for the user to check.
        ollama_base_url: The base URL for the OLLAMA API.
        drop_params: Drop any unmapped (unsupported) params without causing an exception.
        modify_params: Modify params allows litellm to do transformations like adding a default message, when a message is empty.
        disable_vision: If model is vision capable, this option allows to disable image processing (useful for cost reduction).
        caching_prompt: Use the prompt caching feature if provided by the LLM and supported by the provider.
        log_completions: Whether to log LLM completions to the state.
        log_completions_folder: The folder to log LLM completions to. Required if log_completions is True.
        custom_tokenizer: A custom tokenizer to use for token counting.
        native_tool_calling: Whether to use native tool calling if supported by the model. Can be True, False, or not set.
        reasoning_effort: The effort to put into reasoning. This is a string that can be one of 'low', 'medium', 'high', or 'none'. Can apply to all reasoning models.
        seed: The seed to use for the LLM.
        safety_settings: Safety settings for models that support them (like Mistral AI and Gemini).
        for_routing: Whether this LLM is used for routing. This is set to True for models used in conjunction with the main LLM in the model routing feature.
        completion_kwargs: Custom kwargs to pass to litellm.completion.
    """

    model: str = Field(default='claude-opus-4-5-20251101')
    api_key: SecretStr | None = Field(default=None)
    base_url: str | None = Field(default=None)
    api_version: str | None = Field(default=None)
    aws_access_key_id: SecretStr | None = Field(default=None)
    aws_secret_access_key: SecretStr | None = Field(default=None)
    aws_region_name: str | None = Field(default=None)
    openrouter_site_url: str = Field(default='https://docs.all-hands.dev/')
    openrouter_app_name: str = Field(default='OpenHands')
    # total wait time: 8 + 16 + 32 + 64 = 120 seconds
    num_retries: int = Field(default=5)
    retry_multiplier: float = Field(default=8)
    retry_min_wait: int = Field(default=8)
    retry_max_wait: int = Field(default=64)
    timeout: int | None = Field(default=None)
    max_message_chars: int = Field(
        default=30_000
    )  # maximum number of characters in an observation's content when sent to the llm
    temperature: float = Field(default=0.0)
    top_p: float = Field(default=1.0)
    top_k: float | None = Field(default=None)
    custom_llm_provider: str | None = Field(default=None)
    max_input_tokens: int | None = Field(default=None)
    max_output_tokens: int | None = Field(default=None)
    input_cost_per_token: float | None = Field(default=None)
    output_cost_per_token: float | None = Field(default=None)
    ollama_base_url: str | None = Field(default=None)
    # This setting can be sent in each call to litellm
    drop_params: bool = Field(default=True)
    # Note: this setting is actually global, unlike drop_params
    modify_params: bool = Field(default=True)
    disable_vision: bool | None = Field(default=None)
    disable_stop_word: bool | None = Field(default=False)
    caching_prompt: bool = Field(default=True)
    log_completions: bool = Field(default=False)
    log_completions_folder: str = Field(default=os.path.join(LOG_DIR, 'completions'))
    custom_tokenizer: str | None = Field(default=None)
    native_tool_calling: bool | None = Field(default=None)
    reasoning_effort: str | None = Field(default=None)
    seed: int | None = Field(default=None)
    safety_settings: list[dict[str, str]] | None = Field(
        default=None,
        description='Safety settings for models that support them (like Mistral AI and Gemini)',
    )
    for_routing: bool = Field(default=False)
    # The number of correction attempts for the LLM draft editor
    correct_num: int = Field(default=5)
    completion_kwargs: dict[str, Any] | None = Field(
        default=None,
        description='Custom kwargs to pass to litellm.completion',
    )

    model_config = ConfigDict(extra='forbid')

    @classmethod
    def from_toml_section(cls, data: dict) -> dict[str, LLMConfig]:
        """Create a mapping of LLMConfig instances from a toml dictionary representing the [llm] section.

        The default configuration is built from all non-dict keys in data.
        Then, each key with a dict value (e.g. [llm.random_name]) is treated as a custom LLM configuration,
        and its values override the default configuration.

        Example:
        Apply generic LLM config with custom LLM overrides, e.g.
            [llm]
            model=...
            num_retries = 5
            [llm.claude]
            model="claude-3-5-sonnet"
        results in num_retries APPLIED to claude-3-5-sonnet.

        Returns:
            dict[str, LLMConfig]: A mapping where the key "llm" corresponds to the default configuration
            and additional keys represent custom configurations.
        """
        # Initialize the result mapping
        llm_mapping: dict[str, LLMConfig] = {}

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
            llm_mapping['llm'] = base_config
        except ValidationError:
            logger.warning(
                'Cannot parse [llm] config from toml. Continuing with defaults.'
            )
            # If base config fails, create a default one
            base_config = cls()
            # Still add it to the mapping
            llm_mapping['llm'] = base_config

        # Process each custom section independently
        for name, overrides in custom_sections.items():
            try:
                # Merge base config with overrides
                merged = {**base_config.model_dump(), **overrides}
                custom_config = cls.model_validate(merged)
                llm_mapping[name] = custom_config
            except ValidationError:
                logger.warning(
                    f'Cannot parse [{name}] config from toml. This section will be skipped.'
                )
                # Skip this custom section but continue with others
                continue

        return llm_mapping

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook to assign OpenRouter-related variables to environment variables.

        This ensures that these values are accessible to litellm at runtime.
        """
        super().model_post_init(__context)

        # Assign OpenRouter-specific variables to environment variables
        if self.openrouter_site_url:
            os.environ['OR_SITE_URL'] = self.openrouter_site_url
        if self.openrouter_app_name:
            os.environ['OR_APP_NAME'] = self.openrouter_app_name

        # Do not set a default reasoning_effort. Leave as None unless user-configured.

        # Set an API version by default for Azure models
        # Required for newer models.
        # Azure issue: https://github.com/OpenHands/OpenHands/issues/7755
        if self.model.startswith('azure') and self.api_version is None:
            self.api_version = '2024-12-01-preview'

        # Set AWS credentials as environment variables for LiteLLM Bedrock
        if self.aws_access_key_id:
            os.environ['AWS_ACCESS_KEY_ID'] = self.aws_access_key_id.get_secret_value()
        if self.aws_secret_access_key:
            os.environ['AWS_SECRET_ACCESS_KEY'] = (
                self.aws_secret_access_key.get_secret_value()
            )
        if self.aws_region_name:
            os.environ['AWS_REGION_NAME'] = self.aws_region_name
