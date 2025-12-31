from __future__ import annotations

import copy
import json
import os
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Literal, get_args, get_origin

import httpx  # noqa: F401
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from openhands.sdk.llm.utils.model_info import get_litellm_model_info
from openhands.sdk.utils.pydantic_secrets import serialize_secret, validate_secret


if TYPE_CHECKING:  # type hints only, avoid runtime import cycle
    from openhands.sdk.tool.tool import ToolDefinition

from openhands.sdk.utils.pydantic_diff import pretty_pydantic_diff


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import litellm

from typing import cast

from litellm import (
    ChatCompletionToolParam,
    CustomStreamWrapper,
    ResponseInputParam,
    completion as litellm_completion,
)
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout as LiteLLMTimeout,
)
from litellm.responses.main import responses as litellm_responses
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import ModelResponse
from litellm.utils import (
    create_pretrained_tokenizer,
    supports_vision,
    token_counter,
)

from openhands.sdk.llm.exceptions import (
    LLMNoResponseError,
    map_provider_exception,
)

# OpenHands utilities
from openhands.sdk.llm.llm_response import LLMResponse
from openhands.sdk.llm.message import (
    Message,
)
from openhands.sdk.llm.mixins.non_native_fc import NonNativeToolCallingMixin
from openhands.sdk.llm.options.chat_options import select_chat_options
from openhands.sdk.llm.options.responses_options import select_responses_options
from openhands.sdk.llm.streaming import (
    TokenCallbackType,
)
from openhands.sdk.llm.utils.metrics import Metrics, MetricsSnapshot
from openhands.sdk.llm.utils.model_features import get_default_temperature, get_features
from openhands.sdk.llm.utils.retry_mixin import RetryMixin
from openhands.sdk.llm.utils.telemetry import Telemetry
from openhands.sdk.logger import ENV_LOG_DIR, get_logger


logger = get_logger(__name__)

__all__ = ["LLM"]


# Exceptions we retry on
LLM_RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
    LiteLLMTimeout,
    InternalServerError,
    LLMNoResponseError,
)


class LLM(BaseModel, RetryMixin, NonNativeToolCallingMixin):
    """Language model interface for OpenHands agents.

    The LLM class provides a unified interface for interacting with various
    language models through the litellm library. It handles model configuration,
    API authentication,
    retry logic, and tool calling capabilities.

    Example:
        >>> from openhands.sdk import LLM
        >>> from pydantic import SecretStr
        >>> llm = LLM(
        ...     model="claude-sonnet-4-20250514",
        ...     api_key=SecretStr("your-api-key"),
        ...     usage_id="my-agent"
        ... )
        >>> # Use with agent or conversation
    """

    # =========================================================================
    # Config fields
    # =========================================================================
    model: str = Field(default="claude-sonnet-4-20250514", description="Model name.")
    api_key: str | SecretStr | None = Field(default=None, description="API key.")
    base_url: str | None = Field(default=None, description="Custom base URL.")
    api_version: str | None = Field(
        default=None, description="API version (e.g., Azure)."
    )

    aws_access_key_id: str | SecretStr | None = Field(default=None)
    aws_secret_access_key: str | SecretStr | None = Field(default=None)
    aws_region_name: str | None = Field(default=None)

    openrouter_site_url: str = Field(default="https://docs.all-hands.dev/")
    openrouter_app_name: str = Field(default="OpenHands")

    num_retries: int = Field(default=5, ge=0)
    retry_multiplier: float = Field(default=8.0, ge=0)
    retry_min_wait: int = Field(default=8, ge=0)
    retry_max_wait: int = Field(default=64, ge=0)

    timeout: int | None = Field(default=None, ge=0, description="HTTP timeout (s).")

    max_message_chars: int = Field(
        default=30_000,
        ge=1,
        description="Approx max chars in each event/content sent to the LLM.",
    )

    temperature: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Sampling temperature for response generation. "
            "Defaults to 0 for most models and provider default for reasoning models."
        ),
    )
    top_p: float | None = Field(default=1.0, ge=0, le=1)
    top_k: float | None = Field(default=None, ge=0)

    custom_llm_provider: str | None = Field(default=None)
    max_input_tokens: int | None = Field(
        default=None,
        ge=1,
        description="The maximum number of input tokens. "
        "Note that this is currently unused, and the value at runtime is actually"
        " the total tokens in OpenAI (e.g. 128,000 tokens for GPT-4).",
    )
    max_output_tokens: int | None = Field(
        default=None,
        ge=1,
        description="The maximum number of output tokens. This is sent to the LLM.",
    )
    model_canonical_name: str | None = Field(
        default=None,
        description=(
            "Optional canonical model name for feature registry lookups. "
            "The OpenHands SDK maintains a model feature registry that "
            "maps model names to capabilities (e.g., vision support, "
            "prompt caching, responses API support). When using proxied or "
            "aliased model identifiers, set this field to the canonical "
            "model name (e.g., 'openai/gpt-4o') to ensure correct "
            "capability detection. If not provided, the 'model' field "
            "will be used for capability lookups."
        ),
    )
    extra_headers: dict[str, str] | None = Field(
        default=None,
        description="Optional HTTP headers to forward to LiteLLM requests.",
    )
    input_cost_per_token: float | None = Field(
        default=None,
        ge=0,
        description="The cost per input token. This will available in logs for user.",
    )
    output_cost_per_token: float | None = Field(
        default=None,
        ge=0,
        description="The cost per output token. This will available in logs for user.",
    )
    ollama_base_url: str | None = Field(default=None)

    stream: bool = Field(
        default=False,
        description=(
            "Enable streaming responses from the LLM. "
            "When enabled, the provided `on_token` callback in .completions "
            "and .responses will be invoked for each chunk of tokens."
        ),
    )
    drop_params: bool = Field(default=True)
    modify_params: bool = Field(
        default=True,
        description="Modify params allows litellm to do transformations like adding"
        " a default message, when a message is empty.",
    )
    disable_vision: bool | None = Field(
        default=None,
        description="If model is vision capable, this option allows to disable image "
        "processing (useful for cost reduction).",
    )
    disable_stop_word: bool | None = Field(
        default=False, description="Disable using of stop word."
    )
    caching_prompt: bool = Field(default=True, description="Enable caching of prompts.")
    log_completions: bool = Field(
        default=False, description="Enable logging of completions."
    )
    log_completions_folder: str = Field(
        default=os.path.join(ENV_LOG_DIR, "completions"),
        description="The folder to log LLM completions to. "
        "Required if log_completions is True.",
    )
    custom_tokenizer: str | None = Field(
        default=None, description="A custom tokenizer to use for token counting."
    )
    native_tool_calling: bool = Field(
        default=True,
        description="Whether to use native tool calling.",
    )
    force_string_serializer: bool | None = Field(
        default=None,
        description=(
            "Force using string content serializer when sending to LLM API. "
            "If None (default), auto-detect based on model. "
            "Useful for providers that do not support list content, "
            "like HuggingFace and Groq."
        ),
    )
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "none"] | None = Field(
        default="high",
        description="The effort to put into reasoning. "
        "This is a string that can be one of 'low', 'medium', 'high', 'xhigh', "
        "or 'none'. "
        "Can apply to all reasoning models.",
    )
    reasoning_summary: Literal["auto", "concise", "detailed"] | None = Field(
        default=None,
        description="The level of detail for reasoning summaries. "
        "This is a string that can be one of 'auto', 'concise', or 'detailed'. "
        "Requires verified OpenAI organization. Only sent when explicitly set.",
    )
    enable_encrypted_reasoning: bool = Field(
        default=True,
        description="If True, ask for ['reasoning.encrypted_content'] "
        "in Responses API include.",
    )
    # Prompt cache retention only applies to GPT-5+ models; filtered in chat options
    prompt_cache_retention: str | None = Field(
        default="24h",
        description=(
            "Retention policy for prompt cache. Only sent for GPT-5+ models; "
            "explicitly stripped for all other models."
        ),
    )
    extended_thinking_budget: int | None = Field(
        default=200_000,
        description="The budget tokens for extended thinking, "
        "supported by Anthropic models.",
    )
    seed: int | None = Field(
        default=None, description="The seed to use for random number generation."
    )
    safety_settings: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "Safety settings for models that support them (like Mistral AI and Gemini)"
        ),
    )
    usage_id: str = Field(
        default="default",
        serialization_alias="usage_id",
        description=(
            "Unique usage identifier for the LLM. Used for registry lookups, "
            "telemetry, and spend tracking."
        ),
    )
    litellm_extra_body: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional key-value pairs to pass to litellm's extra_body parameter. "
            "This is useful for custom inference endpoints that need additional "
            "parameters for configuration, routing, or advanced features. "
            "NOTE: Not all LLM providers support extra_body parameters. Some providers "
            "(e.g., OpenAI) may reject requests with unrecognized options. "
            "This is commonly supported by: "
            "- LiteLLM proxy servers (routing metadata, tracing) "
            "- vLLM endpoints (return_token_ids, etc.) "
            "- Custom inference clusters "
            "Examples: "
            "- Proxy routing: {'trace_version': '1.0.0', 'tags': ['agent:my-agent']} "
            "- vLLM features: {'return_token_ids': True}"
        ),
    )

    # =========================================================================
    # Internal fields (excluded from dumps)
    # =========================================================================
    retry_listener: SkipJsonSchema[
        Callable[[int, int, BaseException | None], None] | None
    ] = Field(
        default=None,
        exclude=True,
    )
    _metrics: Metrics | None = PrivateAttr(default=None)
    # ===== Plain class vars (NOT Fields) =====
    # When serializing, these fields (SecretStr) will be dump to "****"
    # When deserializing, these fields will be ignored and we will override
    # them from the LLM instance provided at runtime.
    OVERRIDE_ON_SERIALIZE: tuple[str, ...] = (
        "api_key",
        "aws_access_key_id",
        "aws_secret_access_key",
        # Dynamic runtime metadata for telemetry/routing that can differ across sessions
        # and should not cause resume-time diffs. Always prefer the runtime value.
        "litellm_extra_body",
    )

    # Runtime-only private attrs
    _model_info: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)
    _telemetry: Telemetry | None = PrivateAttr(default=None)

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True
    )

    # =========================================================================
    # Validators
    # =========================================================================
    @field_validator("api_key", "aws_access_key_id", "aws_secret_access_key")
    @classmethod
    def _validate_secrets(cls, v: str | SecretStr | None, info) -> SecretStr | None:
        return validate_secret(v, info)

    @model_validator(mode="before")
    @classmethod
    def _coerce_inputs(cls, data):
        if not isinstance(data, dict):
            return data
        d = dict(data)

        model_val = d.get("model")
        if not model_val:
            raise ValueError("model must be specified in LLM")

        # Azure default version
        if model_val.startswith("azure") and not d.get("api_version"):
            d["api_version"] = "2024-12-01-preview"

        # Provider rewrite: openhands/* -> litellm_proxy/*
        if model_val.startswith("openhands/"):
            model_name = model_val.removeprefix("openhands/")
            d["model"] = f"litellm_proxy/{model_name}"
            # Set base_url (default to the app proxy when base_url is unset or None)
            # Use `or` instead of dict.get() to handle explicit None values
            d["base_url"] = d.get("base_url") or "https://llm-proxy.app.all-hands.dev/"

        # HF doesn't support the OpenAI default value for top_p (1)
        if model_val.startswith("huggingface"):
            if d.get("top_p", 1.0) == 1.0:
                d["top_p"] = 0.9

        return d

    @model_validator(mode="after")
    def _set_env_side_effects(self):
        if self.openrouter_site_url:
            os.environ["OR_SITE_URL"] = self.openrouter_site_url
        if self.openrouter_app_name:
            os.environ["OR_APP_NAME"] = self.openrouter_app_name
        if self.aws_access_key_id:
            assert isinstance(self.aws_access_key_id, SecretStr)
            os.environ["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id.get_secret_value()
        if self.aws_secret_access_key:
            assert isinstance(self.aws_secret_access_key, SecretStr)
            os.environ["AWS_SECRET_ACCESS_KEY"] = (
                self.aws_secret_access_key.get_secret_value()
            )
        if self.aws_region_name:
            os.environ["AWS_REGION_NAME"] = self.aws_region_name

        # Metrics + Telemetry wiring
        if self._metrics is None:
            self._metrics = Metrics(model_name=self.model)

        self._telemetry = Telemetry(
            model_name=self.model,
            log_enabled=self.log_completions,
            log_dir=self.log_completions_folder if self.log_completions else None,
            input_cost_per_token=self.input_cost_per_token,
            output_cost_per_token=self.output_cost_per_token,
            metrics=self._metrics,
        )

        # Tokenizer
        if self.custom_tokenizer:
            self._tokenizer = create_pretrained_tokenizer(self.custom_tokenizer)

        # Capabilities + model info
        self._init_model_info_and_caps()

        if self.temperature is None:
            self.temperature = get_default_temperature(self.model)

        logger.debug(
            f"LLM ready: model={self.model} base_url={self.base_url} "
            f"reasoning_effort={self.reasoning_effort} "
            f"temperature={self.temperature}"
        )
        return self

    def _retry_listener_fn(
        self, attempt_number: int, num_retries: int, _err: BaseException | None
    ) -> None:
        if self.retry_listener is not None:
            self.retry_listener(attempt_number, num_retries, _err)
        if self._telemetry is not None and _err is not None:
            self._telemetry.on_error(_err)

    # =========================================================================
    # Serializers
    # =========================================================================
    @field_serializer(
        "api_key", "aws_access_key_id", "aws_secret_access_key", when_used="always"
    )
    def _serialize_secrets(self, v: SecretStr | None, info):
        return serialize_secret(v, info)

    # =========================================================================
    # Public API
    # =========================================================================
    @property
    def metrics(self) -> Metrics:
        """Get usage metrics for this LLM instance.

        Returns:
            Metrics object containing token usage, costs, and other statistics.

        Example:
            >>> cost = llm.metrics.accumulated_cost
            >>> print(f"Total cost: ${cost}")
        """
        assert self._metrics is not None, (
            "Metrics should be initialized after model validation"
        )
        return self._metrics

    @property
    def telemetry(self) -> Telemetry:
        """Get telemetry handler for this LLM instance.

        Returns:
            Telemetry object for managing logging and metrics callbacks.

        Example:
            >>> llm.telemetry.set_log_completions_callback(my_callback)
        """
        assert self._telemetry is not None, (
            "Telemetry should be initialized after model validation"
        )
        return self._telemetry

    def restore_metrics(self, metrics: Metrics) -> None:
        # Only used by ConversationStats to seed metrics
        self._metrics = metrics

    def completion(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition] | None = None,
        _return_metrics: bool = False,
        add_security_risk_prediction: bool = False,
        on_token: TokenCallbackType | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion from the language model.

        This is the method for getting responses from the model via Completion API.
        It handles message formatting, tool calling, and response processing.

        Returns:
            LLMResponse containing the model's response and metadata.

        Raises:
            ValueError: If streaming is requested (not supported).

        Example:
            >>> from openhands.sdk.llm import Message, TextContent
            >>> messages = [Message(role="user", content=[TextContent(text="Hello")])]
            >>> response = llm.completion(messages)
            >>> print(response.content)
        """
        enable_streaming = bool(kwargs.get("stream", False)) or self.stream
        if enable_streaming:
            if on_token is None:
                raise ValueError("Streaming requires an on_token callback")
            kwargs["stream"] = True

        # 1) serialize messages
        formatted_messages = self.format_messages_for_llm(messages)

        # 2) choose function-calling strategy
        use_native_fc = self.native_tool_calling
        original_fncall_msgs = copy.deepcopy(formatted_messages)

        # Convert Tool objects to ChatCompletionToolParam once here
        cc_tools: list[ChatCompletionToolParam] = []
        if tools:
            cc_tools = [
                t.to_openai_tool(
                    add_security_risk_prediction=add_security_risk_prediction
                )
                for t in tools
            ]

        use_mock_tools = self.should_mock_tool_calls(cc_tools)
        if use_mock_tools:
            logger.debug(
                "LLM.completion: mocking function-calling via prompt "
                f"for model {self.model}"
            )
            formatted_messages, kwargs = self.pre_request_prompt_mock(
                formatted_messages, cc_tools or [], kwargs
            )

        # 3) normalize provider params
        # Only pass tools when native FC is active
        kwargs["tools"] = cc_tools if (bool(cc_tools) and use_native_fc) else None
        has_tools_flag = bool(cc_tools) and use_native_fc
        # Behavior-preserving: delegate to select_chat_options
        call_kwargs = select_chat_options(self, kwargs, has_tools=has_tools_flag)

        # 4) optional request logging context (kept small)
        assert self._telemetry is not None
        log_ctx = None
        if self._telemetry.log_enabled:
            log_ctx = {
                "messages": formatted_messages[:],  # already simple dicts
                "tools": tools,
                "kwargs": {k: v for k, v in call_kwargs.items()},
                "context_window": self.max_input_tokens or 0,
            }
            if tools and not use_native_fc:
                log_ctx["raw_messages"] = original_fncall_msgs

        # 5) do the call with retries
        @self.retry_decorator(
            num_retries=self.num_retries,
            retry_exceptions=LLM_RETRY_EXCEPTIONS,
            retry_min_wait=self.retry_min_wait,
            retry_max_wait=self.retry_max_wait,
            retry_multiplier=self.retry_multiplier,
            retry_listener=self._retry_listener_fn,
        )
        def _one_attempt(**retry_kwargs) -> ModelResponse:
            assert self._telemetry is not None
            self._telemetry.on_request(log_ctx=log_ctx)
            # Merge retry-modified kwargs (like temperature) with call_kwargs
            final_kwargs = {**call_kwargs, **retry_kwargs}
            resp = self._transport_call(
                messages=formatted_messages,
                **final_kwargs,
                enable_streaming=enable_streaming,
                on_token=on_token,
            )
            raw_resp: ModelResponse | None = None
            if use_mock_tools:
                raw_resp = copy.deepcopy(resp)
                resp = self.post_response_prompt_mock(
                    resp, nonfncall_msgs=formatted_messages, tools=cc_tools
                )
            # 6) telemetry
            self._telemetry.on_response(resp, raw_resp=raw_resp)

            # Ensure at least one choice.
            # Gemini sometimes returns empty choices; we raise LLMNoResponseError here
            # inside the retry boundary so it is retried.
            if not resp.get("choices") or len(resp["choices"]) < 1:
                raise LLMNoResponseError(
                    "Response choices is less than 1. Response: " + str(resp)
                )

            return resp

        try:
            resp = _one_attempt()

            # Convert the first choice to an OpenHands Message
            first_choice = resp["choices"][0]
            message = Message.from_llm_chat_message(first_choice["message"])

            # Get current metrics snapshot
            metrics_snapshot = MetricsSnapshot(
                model_name=self.metrics.model_name,
                accumulated_cost=self.metrics.accumulated_cost,
                max_budget_per_task=self.metrics.max_budget_per_task,
                accumulated_token_usage=self.metrics.accumulated_token_usage,
            )

            # Create and return LLMResponse
            return LLMResponse(
                message=message, metrics=metrics_snapshot, raw_response=resp
            )
        except Exception as e:
            self._telemetry.on_error(e)
            mapped = map_provider_exception(e)
            if mapped is not e:
                raise mapped from e
            raise

    # =========================================================================
    # Responses API (non-stream, v1)
    # =========================================================================
    def responses(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition] | None = None,
        include: list[str] | None = None,
        store: bool | None = None,
        _return_metrics: bool = False,
        add_security_risk_prediction: bool = False,
        on_token: TokenCallbackType | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Alternative invocation path using OpenAI Responses API via LiteLLM.

        Maps Message[] -> (instructions, input[]) and returns LLMResponse.
        """
        # Streaming not yet supported
        if kwargs.get("stream", False) or self.stream or on_token is not None:
            raise ValueError("Streaming is not supported for Responses API yet")

        # Build instructions + input list using dedicated Responses formatter
        instructions, input_items = self.format_messages_for_responses(messages)

        # Convert Tool objects to Responses ToolParam
        # (Responses path always supports function tools)
        resp_tools = (
            [
                t.to_responses_tool(
                    add_security_risk_prediction=add_security_risk_prediction
                )
                for t in tools
            ]
            if tools
            else None
        )

        # Normalize/override Responses kwargs consistently
        call_kwargs = select_responses_options(
            self, kwargs, include=include, store=store
        )

        # Optional request logging
        assert self._telemetry is not None
        log_ctx = None
        if self._telemetry.log_enabled:
            log_ctx = {
                "llm_path": "responses",
                "input": input_items[:],
                "tools": tools,
                "kwargs": {k: v for k, v in call_kwargs.items()},
                "context_window": self.max_input_tokens or 0,
            }

        # Perform call with retries
        @self.retry_decorator(
            num_retries=self.num_retries,
            retry_exceptions=LLM_RETRY_EXCEPTIONS,
            retry_min_wait=self.retry_min_wait,
            retry_max_wait=self.retry_max_wait,
            retry_multiplier=self.retry_multiplier,
            retry_listener=self._retry_listener_fn,
        )
        def _one_attempt(**retry_kwargs) -> ResponsesAPIResponse:
            assert self._telemetry is not None
            self._telemetry.on_request(log_ctx=log_ctx)
            final_kwargs = {**call_kwargs, **retry_kwargs}
            with self._litellm_modify_params_ctx(self.modify_params):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    typed_input: ResponseInputParam | str = (
                        cast(ResponseInputParam, input_items) if input_items else ""
                    )
                    # Extract api_key value with type assertion for type checker
                    api_key_value: str | None = None
                    if self.api_key:
                        assert isinstance(self.api_key, SecretStr)
                        api_key_value = self.api_key.get_secret_value()

                    ret = litellm_responses(
                        model=self.model,
                        input=typed_input,
                        instructions=instructions,
                        tools=resp_tools,
                        api_key=api_key_value,
                        api_base=self.base_url,
                        api_version=self.api_version,
                        timeout=self.timeout,
                        drop_params=self.drop_params,
                        seed=self.seed,
                        **final_kwargs,
                    )
                    assert isinstance(ret, ResponsesAPIResponse), (
                        f"Expected ResponsesAPIResponse, got {type(ret)}"
                    )
                    # telemetry (latency, cost). Token usage mapping we handle after.
                    self._telemetry.on_response(ret)
                    return ret

        try:
            resp: ResponsesAPIResponse = _one_attempt()

            # Parse output -> Message (typed)
            # Cast to a typed sequence
            # accepted by from_llm_responses_output
            output_seq = cast(Sequence[Any], resp.output or [])
            message = Message.from_llm_responses_output(output_seq)

            metrics_snapshot = MetricsSnapshot(
                model_name=self.metrics.model_name,
                accumulated_cost=self.metrics.accumulated_cost,
                max_budget_per_task=self.metrics.max_budget_per_task,
                accumulated_token_usage=self.metrics.accumulated_token_usage,
            )

            return LLMResponse(
                message=message, metrics=metrics_snapshot, raw_response=resp
            )
        except Exception as e:
            self._telemetry.on_error(e)
            mapped = map_provider_exception(e)
            if mapped is not e:
                raise mapped from e
            raise

    # =========================================================================
    # Transport + helpers
    # =========================================================================
    def _transport_call(
        self,
        *,
        messages: list[dict[str, Any]],
        enable_streaming: bool = False,
        on_token: TokenCallbackType | None = None,
        **kwargs,
    ) -> ModelResponse:
        # litellm.modify_params is GLOBAL; guard it for thread-safety
        with self._litellm_modify_params_ctx(self.modify_params):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=DeprecationWarning, module="httpx.*"
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r".*content=.*upload.*",
                    category=DeprecationWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r"There is no current event loop",
                    category=DeprecationWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    category=DeprecationWarning,
                    message="Accessing the 'model_fields' attribute.*",
                )
                # Extract api_key value with type assertion for type checker
                api_key_value: str | None = None
                if self.api_key:
                    assert isinstance(self.api_key, SecretStr)
                    api_key_value = self.api_key.get_secret_value()

                # Some providers need renames handled in _normalize_call_kwargs.
                ret = litellm_completion(
                    model=self.model,
                    api_key=api_key_value,
                    api_base=self.base_url,
                    api_version=self.api_version,
                    timeout=self.timeout,
                    drop_params=self.drop_params,
                    seed=self.seed,
                    messages=messages,
                    **kwargs,
                )
                if enable_streaming and on_token is not None:
                    assert isinstance(ret, CustomStreamWrapper)
                    chunks = []
                    for chunk in ret:
                        on_token(chunk)
                        chunks.append(chunk)
                    ret = litellm.stream_chunk_builder(chunks, messages=messages)

                assert isinstance(ret, ModelResponse), (
                    f"Expected ModelResponse, got {type(ret)}"
                )
                return ret

    @contextmanager
    def _litellm_modify_params_ctx(self, flag: bool):
        old = getattr(litellm, "modify_params", None)
        try:
            litellm.modify_params = flag
            yield
        finally:
            litellm.modify_params = old

    # =========================================================================
    # Capabilities, formatting, and info
    # =========================================================================
    def _model_name_for_capabilities(self) -> str:
        """Return canonical name for capability lookups (e.g., vision support)."""
        return self.model_canonical_name or self.model

    def _init_model_info_and_caps(self) -> None:
        self._model_info = get_litellm_model_info(
            secret_api_key=self.api_key,
            base_url=self.base_url,
            model=self._model_name_for_capabilities(),
        )

        # Context window and max_output_tokens
        if (
            self.max_input_tokens is None
            and self._model_info is not None
            and isinstance(self._model_info.get("max_input_tokens"), int)
        ):
            self.max_input_tokens = self._model_info.get("max_input_tokens")

        if self.max_output_tokens is None:
            if any(
                m in self.model
                for m in [
                    "claude-3-7-sonnet",
                    "claude-sonnet-4",
                    "kimi-k2-thinking",
                ]
            ):
                self.max_output_tokens = (
                    64000  # practical cap (litellm may allow 128k with header)
                )
                logger.debug(
                    f"Setting max_output_tokens to {self.max_output_tokens} "
                    f"for {self.model}"
                )
            elif self._model_info is not None:
                if isinstance(self._model_info.get("max_output_tokens"), int):
                    self.max_output_tokens = self._model_info.get("max_output_tokens")
                elif isinstance(self._model_info.get("max_tokens"), int):
                    self.max_output_tokens = self._model_info.get("max_tokens")

        if "o3" in self.model:
            o3_limit = 100000
            if self.max_output_tokens is None or self.max_output_tokens > o3_limit:
                self.max_output_tokens = o3_limit
                logger.debug(
                    "Clamping max_output_tokens to %s for %s",
                    self.max_output_tokens,
                    self.model,
                )

    def vision_is_active(self) -> bool:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return not self.disable_vision and self._supports_vision()

    def _supports_vision(self) -> bool:
        """Acquire from litellm if model is vision capable.

        Returns:
            bool: True if model is vision capable. Return False if model not
                supported by litellm.
        """
        # litellm.supports_vision currently returns False for 'openai/gpt-...' or 'anthropic/claude-...' (with prefixes)  # noqa: E501
        # but model_info will have the correct value for some reason.
        # we can go with it, but we will need to keep an eye if model_info is correct for Vertex or other providers  # noqa: E501
        # remove when litellm is updated to fix https://github.com/BerriAI/litellm/issues/5608  # noqa: E501
        # Check both the full model name and the name after proxy prefix for vision support  # noqa: E501
        model_for_caps = self._model_name_for_capabilities()
        return (
            supports_vision(model_for_caps)
            or supports_vision(model_for_caps.split("/")[-1])
            or (
                self._model_info is not None
                and self._model_info.get("supports_vision", False)
            )
            or False  # fallback to False if model_info is None
        )

    def is_caching_prompt_active(self) -> bool:
        """Check if prompt caching is supported and enabled for current model.

        Returns:
            boolean: True if prompt caching is supported and enabled for the given
                model.
        """
        if not self.caching_prompt:
            return False
        # We don't need to look-up model_info, because
        # only Anthropic models need explicit caching breakpoints
        return (
            self.caching_prompt
            and get_features(self._model_name_for_capabilities()).supports_prompt_cache
        )

    def uses_responses_api(self) -> bool:
        """Whether this model uses the OpenAI Responses API path."""

        # by default, uses = supports
        return get_features(self._model_name_for_capabilities()).supports_responses_api

    @property
    def model_info(self) -> dict | None:
        """Returns the model info dictionary."""
        return self._model_info

    # =========================================================================
    # Utilities preserved from previous class
    # =========================================================================
    def _apply_prompt_caching(self, messages: list[Message]) -> None:
        """Applies caching breakpoints to the messages.

        For new Anthropic API, we only need to mark the last user or
          tool message as cacheable.
        """
        if len(messages) > 0 and messages[0].role == "system":
            messages[0].content[-1].cache_prompt = True
        # NOTE: this is only needed for anthropic
        for message in reversed(messages):
            if message.role in ("user", "tool"):
                message.content[
                    -1
                ].cache_prompt = True  # Last item inside the message content
                break

    def format_messages_for_llm(self, messages: list[Message]) -> list[dict]:
        """Formats Message objects for LLM consumption."""

        messages = copy.deepcopy(messages)
        if self.is_caching_prompt_active():
            self._apply_prompt_caching(messages)

        for message in messages:
            message.cache_enabled = self.is_caching_prompt_active()
            message.vision_enabled = self.vision_is_active()
            message.function_calling_enabled = self.native_tool_calling
            model_features = get_features(self._model_name_for_capabilities())
            message.force_string_serializer = (
                self.force_string_serializer
                if self.force_string_serializer is not None
                else model_features.force_string_serializer
            )
            message.send_reasoning_content = model_features.send_reasoning_content

        formatted_messages = [message.to_chat_dict() for message in messages]

        return formatted_messages

    def format_messages_for_responses(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Prepare (instructions, input[]) for the OpenAI Responses API.

        - Skips prompt caching flags and string serializer concerns
        - Uses Message.to_responses_value to get either instructions (system)
         or input items (others)
        - Concatenates system instructions into a single instructions string
        """
        msgs = copy.deepcopy(messages)

        # Determine vision based on model detection
        vision_active = self.vision_is_active()

        # Assign system instructions as a string, collect input items
        instructions: str | None = None
        input_items: list[dict[str, Any]] = []
        for m in msgs:
            val = m.to_responses_value(vision_enabled=vision_active)
            if isinstance(val, str):
                s = val.strip()
                if not s:
                    continue
                instructions = (
                    s if instructions is None else f"{instructions}\n\n---\n\n{s}"
                )
            else:
                if val:
                    input_items.extend(val)
        return instructions, input_items

    def get_token_count(self, messages: list[Message]) -> int:
        logger.debug(
            "Message objects now include serialized tool calls in token counting"
        )
        formatted_messages = self.format_messages_for_llm(messages)
        try:
            return int(
                token_counter(
                    model=self.model,
                    messages=formatted_messages,
                    custom_tokenizer=self._tokenizer,
                )
            )
        except Exception as e:
            logger.error(
                f"Error getting token count for model {self.model}\n{e}"
                + (
                    f"\ncustom_tokenizer: {self.custom_tokenizer}"
                    if self.custom_tokenizer
                    else ""
                ),
                exc_info=True,
            )
            return 0

    # =========================================================================
    # Serialization helpers
    # =========================================================================
    @classmethod
    def load_from_json(cls, json_path: str) -> LLM:
        with open(json_path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def load_from_env(cls, prefix: str = "LLM_") -> LLM:
        TRUTHY = {"true", "1", "yes", "on"}

        def _unwrap_type(t: Any) -> Any:
            origin = get_origin(t)
            if origin is None:
                return t
            args = [a for a in get_args(t) if a is not type(None)]
            return args[0] if args else t

        def _cast_value(raw: str, t: Any) -> Any:
            t = _unwrap_type(t)
            if t is SecretStr:
                return SecretStr(raw)
            if t is bool:
                return raw.lower() in TRUTHY
            if t is int:
                try:
                    return int(raw)
                except ValueError:
                    return None
            if t is float:
                try:
                    return float(raw)
                except ValueError:
                    return None
            origin = get_origin(t)
            if (origin in (list, dict, tuple)) or (
                isinstance(t, type) and issubclass(t, BaseModel)
            ):
                try:
                    return json.loads(raw)
                except Exception:
                    pass
            return raw

        data: dict[str, Any] = {}
        fields: dict[str, Any] = {
            name: f.annotation
            for name, f in cls.model_fields.items()
            if not getattr(f, "exclude", False)
        }

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            field_name = key[len(prefix) :].lower()
            if field_name not in fields:
                continue
            v = _cast_value(value, fields[field_name])
            if v is not None:
                data[field_name] = v
        return cls(**data)

    def resolve_diff_from_deserialized(self, persisted: LLM) -> LLM:
        """Resolve differences between a deserialized LLM and the current instance.

        This is due to fields like api_key being serialized to "****" in dumps,
        and we want to ensure that when loading from a file, we still use the
        runtime-provided api_key in the self instance.

        Return a new LLM instance equivalent to `persisted` but with
        explicitly whitelisted fields (e.g. api_key) taken from `self`.
        """
        if persisted.__class__ is not self.__class__:
            raise ValueError(
                f"Cannot resolve_diff_from_deserialized between {self.__class__} "
                f"and {persisted.__class__}"
            )

        # Copy allowed fields from runtime llm into the persisted llm
        llm_updates = {}
        persisted_dump = persisted.model_dump(context={"expose_secrets": True})
        for field in self.OVERRIDE_ON_SERIALIZE:
            if field in persisted_dump.keys():
                llm_updates[field] = getattr(self, field)
        if llm_updates:
            reconciled = persisted.model_copy(update=llm_updates)
        else:
            reconciled = persisted

        dump = self.model_dump(context={"expose_secrets": True})
        reconciled_dump = reconciled.model_dump(context={"expose_secrets": True})
        if dump != reconciled_dump:
            raise ValueError(
                "The LLM provided is different from the one in persisted state.\n"
                f"Diff: {pretty_pydantic_diff(self, reconciled)}"
            )
        return reconciled
