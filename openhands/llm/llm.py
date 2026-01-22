# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# V1 replacement for this module lives in the Software Agent SDK.
import copy
import os
import time
import warnings
from functools import partial
from typing import Any, Callable, cast

import httpx

from openhands.core.config import LLMConfig
from openhands.llm.metrics import Metrics
from openhands.llm.model_features import get_features

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import litellm

from litellm import Message as LiteLLMMessage
from litellm import ModelInfo, PromptTokensDetails
from litellm import completion as litellm_completion
from litellm import completion_cost as litellm_completion_cost
from litellm.exceptions import (
    APIConnectionError,
    BadGatewayError,
    RateLimitError,
    ServiceUnavailableError,
)
from litellm.types.utils import CostPerToken, ModelResponse, Usage
from litellm.utils import create_pretrained_tokenizer

from openhands.core.exceptions import LLMNoResponseError
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.llm.debug_mixin import DebugMixin
from openhands.llm.fn_call_converter import (
    STOP_WORDS,
    convert_fncall_messages_to_non_fncall_messages,
    convert_non_fncall_messages_to_fncall_messages,
)
from openhands.llm.retry_mixin import RetryMixin

__all__ = ['LLM']

# tuple of exceptions to retry on
LLM_RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
    BadGatewayError,
    litellm.Timeout,
    litellm.InternalServerError,
    LLMNoResponseError,
)


class LLM(RetryMixin, DebugMixin):
    """The LLM class represents a Language Model instance.

    Attributes:
        config: an LLMConfig object specifying the configuration of the LLM.
    """

    def __init__(
        self,
        config: LLMConfig,
        service_id: str,
        metrics: Metrics | None = None,
        retry_listener: Callable[[int, int], None] | None = None,
    ) -> None:
        """Initializes the LLM. If LLMConfig is passed, its values will be the fallback.

        Passing simple parameters always overrides config.

        Args:
            config: The LLM configuration.
            metrics: The metrics to use.
        """
        self._tried_model_info = False
        self.cost_metric_supported: bool = True
        self.config: LLMConfig = copy.deepcopy(config)
        self.service_id = service_id
        self.metrics: Metrics = (
            metrics if metrics is not None else Metrics(model_name=config.model)
        )

        self.model_info: ModelInfo | None = None
        self._function_calling_active: bool = False
        self.retry_listener = retry_listener
        if self.config.log_completions:
            if self.config.log_completions_folder is None:
                raise RuntimeError(
                    'log_completions_folder is required when log_completions is enabled'
                )
            os.makedirs(self.config.log_completions_folder, exist_ok=True)

        # call init_model_info to initialize config.max_output_tokens
        # which is used in partial function
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.init_model_info()
        if self.vision_is_active():
            logger.debug('LLM: model has vision enabled')
        if self.is_caching_prompt_active():
            logger.debug('LLM: caching prompt enabled')
        if self.is_function_calling_active():
            logger.debug('LLM: model supports function calling')

        # if using a custom tokenizer, make sure it's loaded and accessible in the format expected by litellm
        if self.config.custom_tokenizer is not None:
            self.tokenizer = create_pretrained_tokenizer(self.config.custom_tokenizer)
        else:
            self.tokenizer = None

        # set up the completion function
        kwargs: dict[str, Any] = {
            'temperature': self.config.temperature,
            'max_completion_tokens': self.config.max_output_tokens,
        }
        if self.config.top_k is not None:
            # openai doesn't expose top_k
            # litellm will handle it a bit differently than the openai-compatible params
            kwargs['top_k'] = self.config.top_k
        if self.config.top_p is not None:
            # openai doesn't expose top_p, but litellm does
            kwargs['top_p'] = self.config.top_p

        # Handle OpenHands provider - rewrite to litellm_proxy
        if self.config.model.startswith('openhands/'):
            model_name = self.config.model.removeprefix('openhands/')
            self.config.model = f'litellm_proxy/{model_name}'
            self.config.base_url = 'https://llm-proxy.app.all-hands.dev/'
            logger.debug(
                f'Rewrote openhands/{model_name} to {self.config.model} with base URL {self.config.base_url}'
            )

        features = get_features(self.config.model)
        if features.supports_reasoning_effort:
            # For Gemini models, only map 'low' to optimized thinking budget
            # Let other reasoning_effort values pass through to API as-is
            if 'gemini-2.5-pro' in self.config.model:
                logger.debug(
                    f'Gemini model {self.config.model} with reasoning_effort {self.config.reasoning_effort}'
                )
                if self.config.reasoning_effort in {None, 'low', 'none'}:
                    kwargs['thinking'] = {'budget_tokens': 128}
                    kwargs['allowed_openai_params'] = ['thinking']
                    kwargs.pop('reasoning_effort', None)
                else:
                    kwargs['reasoning_effort'] = self.config.reasoning_effort
                logger.debug(
                    f'Gemini model {self.config.model} with reasoning_effort {self.config.reasoning_effort} mapped to thinking {kwargs.get("thinking")}'
                )
            elif any(
                k in self.config.model
                for k in ('claude-sonnet-4-5', 'claude-haiku-4-5-20251001')
            ):
                # don't send reasoning_effort to specific Claude Sonnet/Haiku 4.5 variants
                kwargs.pop('reasoning_effort', None)
            else:
                if self.config.reasoning_effort is not None:
                    kwargs['reasoning_effort'] = self.config.reasoning_effort
            kwargs.pop(
                'temperature'
            )  # temperature is not supported for reasoning models
            kwargs.pop('top_p')  # reasoning model like o3 doesn't support top_p
        # Azure issue: https://github.com/OpenHands/OpenHands/issues/6777
        if self.config.model.startswith('azure'):
            kwargs['max_tokens'] = self.config.max_output_tokens
            kwargs.pop('max_completion_tokens')

        # Add safety settings for models that support them
        if 'mistral' in self.config.model.lower() and self.config.safety_settings:
            kwargs['safety_settings'] = self.config.safety_settings
        elif 'gemini' in self.config.model.lower() and self.config.safety_settings:
            kwargs['safety_settings'] = self.config.safety_settings

        # support AWS Bedrock provider
        kwargs['aws_region_name'] = self.config.aws_region_name
        if self.config.aws_access_key_id:
            kwargs['aws_access_key_id'] = (
                self.config.aws_access_key_id.get_secret_value()
            )
        if self.config.aws_secret_access_key:
            kwargs['aws_secret_access_key'] = (
                self.config.aws_secret_access_key.get_secret_value()
            )

        # Explicitly disable Anthropic extended thinking for Opus 4.1 to avoid
        # requiring 'thinking' content blocks. See issue #10510.
        if 'claude-opus-4-1' in self.config.model.lower():
            kwargs['thinking'] = {'type': 'disabled'}

        # Anthropic constraint: Opus 4.1, Opus 4.5, and Sonnet 4 models cannot accept both temperature and top_p
        # Prefer temperature (drop top_p) if both are specified.
        _model_lower = self.config.model.lower()
        # Apply to Opus 4.1, Opus 4.5, and Sonnet 4 models to avoid API errors
        if (
            ('claude-opus-4-1' in _model_lower)
            or ('claude-opus-4-5' in _model_lower)
            or ('claude-sonnet-4' in _model_lower)
        ) and ('temperature' in kwargs and 'top_p' in kwargs):
            kwargs.pop('top_p', None)

        # Add completion_kwargs if present
        if self.config.completion_kwargs is not None:
            kwargs.update(self.config.completion_kwargs)

        self._completion = partial(
            litellm_completion,
            model=self.config.model,
            api_key=self.config.api_key.get_secret_value()
            if self.config.api_key
            else None,
            base_url=self.config.base_url,
            api_version=self.config.api_version,
            custom_llm_provider=self.config.custom_llm_provider,
            timeout=self.config.timeout,
            drop_params=self.config.drop_params,
            seed=self.config.seed,
            **kwargs,
        )

        self._completion_unwrapped = self._completion

        @self.retry_decorator(
            num_retries=self.config.num_retries,
            retry_exceptions=LLM_RETRY_EXCEPTIONS,
            retry_min_wait=self.config.retry_min_wait,
            retry_max_wait=self.config.retry_max_wait,
            retry_multiplier=self.config.retry_multiplier,
            retry_listener=self.retry_listener,
        )
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper for the litellm completion function. Logs the input and output of the completion function."""
            from openhands.io import json

            messages_kwarg: (
                dict[str, Any] | Message | list[dict[str, Any]] | list[Message]
            ) = []
            mock_function_calling = not self.is_function_calling_active()

            # some callers might send the model and messages directly
            # litellm allows positional args, like completion(model, messages, **kwargs)
            if len(args) > 1:
                # ignore the first argument if it's provided (it would be the model)
                # design wise: we don't allow overriding the configured values
                # implementation wise: the partial function set the model as a kwarg already
                # as well as other kwargs
                messages_kwarg = args[1] if len(args) > 1 else args[0]
                kwargs['messages'] = messages_kwarg

                # remove the first args, they're sent in kwargs
                args = args[2:]
            elif 'messages' in kwargs:
                messages_kwarg = kwargs['messages']

            # ensure we work with a list of messages
            messages_list = (
                messages_kwarg if isinstance(messages_kwarg, list) else [messages_kwarg]
            )
            # Format Message objects to dict format if needed
            messages: list[dict] = []
            if messages_list and isinstance(messages_list[0], Message):
                messages = self.format_messages_for_llm(
                    cast(list[Message], messages_list)
                )
            else:
                messages = cast(list[dict[str, Any]], messages_list)

            kwargs['messages'] = messages

            # handle conversion of to non-function calling messages if needed
            original_fncall_messages = copy.deepcopy(messages)
            mock_fncall_tools = None
            # if the agent or caller has defined tools, and we mock via prompting, convert the messages
            if mock_function_calling and 'tools' in kwargs:
                add_in_context_learning_example = True
                if (
                    'openhands-lm' in self.config.model
                    or 'devstral' in self.config.model
                ):
                    add_in_context_learning_example = False

                messages = convert_fncall_messages_to_non_fncall_messages(
                    messages,
                    kwargs['tools'],
                    add_in_context_learning_example=add_in_context_learning_example,
                )
                kwargs['messages'] = messages

                # add stop words if the model supports it and stop words are not disabled
                if (
                    get_features(self.config.model).supports_stop_words
                    and not self.config.disable_stop_word
                ):
                    kwargs['stop'] = STOP_WORDS

                mock_fncall_tools = kwargs.pop('tools')
                if 'openhands-lm' in self.config.model:
                    # If we don't have this, we might run into issue when serving openhands-lm
                    # using SGLang
                    # BadRequestError: litellm.BadRequestError: OpenAIException - Error code: 400 - {'object': 'error', 'message': '400', 'type': 'Failed to parse fc related info to json format!', 'param': None, 'code': 400}
                    kwargs['tool_choice'] = 'none'
                else:
                    # tool_choice should not be specified when mocking function calling
                    kwargs.pop('tool_choice', None)

            # if we have no messages, something went very wrong
            if not messages:
                raise ValueError(
                    'The messages list is empty. At least one message is required.'
                )

            # log the entire LLM prompt
            self.log_prompt(messages)

            # set litellm modify_params to the configured value
            # True by default to allow litellm to do transformations like adding a default message, when a message is empty
            # NOTE: this setting is global; unlike drop_params, it cannot be overridden in the litellm completion partial
            litellm.modify_params = self.config.modify_params

            # if we're not using litellm proxy, remove the extra_body
            if 'litellm_proxy' not in self.config.model:
                kwargs.pop('extra_body', None)

            # Record start time for latency measurement
            start_time = time.time()
            # we don't support streaming here, thus we get a ModelResponse

            # Suppress httpx deprecation warnings during LiteLLM calls
            # This prevents the "Use 'content=<...>' to upload raw bytes/text content" warning
            # that appears when LiteLLM makes HTTP requests to LLM providers
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', category=DeprecationWarning, module='httpx.*'
                )
                warnings.filterwarnings(
                    'ignore',
                    message=r'.*content=.*upload.*',
                    category=DeprecationWarning,
                )
                resp: ModelResponse = self._completion_unwrapped(*args, **kwargs)

            # Calculate and record latency
            latency = time.time() - start_time
            response_id = resp.get('id', 'unknown')
            self.metrics.add_response_latency(latency, response_id)

            non_fncall_response = copy.deepcopy(resp)

            # if we mocked function calling, and we have tools, convert the response back to function calling format
            if mock_function_calling and mock_fncall_tools is not None:
                if len(resp.choices) < 1:
                    raise LLMNoResponseError(
                        'Response choices is less than 1 - This is only seen in Gemini models so far. Response: '
                        + str(resp)
                    )

                non_fncall_response_message = resp.choices[0].message
                # messages is already a list with proper typing from line 223
                fn_call_messages_with_response = (
                    convert_non_fncall_messages_to_fncall_messages(
                        messages + [non_fncall_response_message], mock_fncall_tools
                    )
                )
                fn_call_response_message = fn_call_messages_with_response[-1]
                if not isinstance(fn_call_response_message, LiteLLMMessage):
                    fn_call_response_message = LiteLLMMessage(
                        **fn_call_response_message
                    )
                resp.choices[0].message = fn_call_response_message

            # Check if resp has 'choices' key with at least one item
            if not resp.get('choices') or len(resp['choices']) < 1:
                raise LLMNoResponseError(
                    'Response choices is less than 1 - This is only seen in Gemini models so far. Response: '
                    + str(resp)
                )

            # log the LLM response
            self.log_response(resp)

            # post-process the response first to calculate cost
            cost = self._post_completion(resp)

            # log for evals or other scripts that need the raw completion
            if self.config.log_completions:
                assert self.config.log_completions_folder is not None
                log_file = os.path.join(
                    self.config.log_completions_folder,
                    f'{self.config.model.replace("/", "__")}-{time.time()}.json',
                )

                # set up the dict to be logged
                _d = {
                    'messages': messages,
                    'response': resp,
                    'args': args,
                    'kwargs': {
                        k: v
                        for k, v in kwargs.items()
                        if k not in ('messages', 'client')
                    },
                    'timestamp': time.time(),
                    'cost': cost,
                }

                # if non-native function calling, save messages/response separately
                if mock_function_calling:
                    # Overwrite response as non-fncall to be consistent with messages
                    _d['response'] = non_fncall_response

                    # Save fncall_messages/response separately
                    _d['fncall_messages'] = original_fncall_messages
                    _d['fncall_response'] = resp
                with open(log_file, 'w') as f:
                    f.write(json.dumps(_d))

            return resp

        self._completion = wrapper

    @property
    def completion(self) -> Callable:
        """Decorator for the litellm completion function.

        Check the complete documentation at https://litellm.vercel.app/docs/completion
        """
        return self._completion

    def init_model_info(self) -> None:
        if self._tried_model_info:
            return
        self._tried_model_info = True
        try:
            if self.config.model.startswith('openrouter'):
                self.model_info = litellm.get_model_info(self.config.model)
        except Exception as e:
            logger.debug(f'Error getting model info: {e}')

        if self.config.model.startswith('litellm_proxy/'):
            # IF we are using LiteLLM proxy, get model info from LiteLLM proxy
            # GET {base_url}/v1/model/info with litellm_model_id as path param
            base_url = self.config.base_url.strip() if self.config.base_url else ''
            if not base_url.startswith(('http://', 'https://')):
                base_url = 'http://' + base_url

            response = httpx.get(
                f'{base_url}/v1/model/info',
                headers={
                    'Authorization': f'Bearer {self.config.api_key.get_secret_value() if self.config.api_key else None}'
                },
            )

            try:
                resp_json = response.json()
                if 'data' not in resp_json:
                    logger.info(
                        f'No data field in model info response from LiteLLM proxy: {resp_json}'
                    )
                all_model_info = resp_json.get('data', [])
            except Exception as e:
                logger.info(f'Error parsing JSON response from LiteLLM proxy: {e}')
                all_model_info = []
            current_model_info = next(
                (
                    info
                    for info in all_model_info
                    if info['model_name']
                    == self.config.model.removeprefix('litellm_proxy/')
                ),
                None,
            )
            if current_model_info:
                self.model_info = current_model_info['model_info']
                logger.debug(f'Got model info from litellm proxy: {self.model_info}')

        # Last two attempts to get model info from NAME
        if not self.model_info:
            try:
                self.model_info = litellm.get_model_info(
                    self.config.model.split(':')[0]
                )
            # noinspection PyBroadException
            except Exception:
                pass
        if not self.model_info:
            try:
                self.model_info = litellm.get_model_info(
                    self.config.model.split('/')[-1]
                )
            # noinspection PyBroadException
            except Exception:
                pass
        from openhands.io import json

        logger.debug(
            f'Model info: {json.dumps({"model": self.config.model, "base_url": self.config.base_url}, indent=2)}'
        )

        if self.config.model.startswith('huggingface'):
            # HF doesn't support the OpenAI default value for top_p (1)
            logger.debug(
                f'Setting top_p to 0.9 for Hugging Face model: {self.config.model}'
            )
            self.config.top_p = 0.9 if self.config.top_p == 1 else self.config.top_p

        # Set max_input_tokens from model info if not explicitly set
        if (
            self.config.max_input_tokens is None
            and self.model_info is not None
            and 'max_input_tokens' in self.model_info
            and isinstance(self.model_info['max_input_tokens'], int)
        ):
            self.config.max_input_tokens = self.model_info['max_input_tokens']

        # Set max_output_tokens from model info if not explicitly set
        if self.config.max_output_tokens is None:
            # Special case for Claude Sonnet models
            sonnet_models = [
                'claude-3-7-sonnet',
                'claude-3.7-sonnet',
                'claude-sonnet-4',
                'claude-sonnet-4-5-20250929',
                'claude-haiku-4-5-20251001',
            ]
            if any(model in self.config.model for model in sonnet_models):
                self.config.max_output_tokens = 64000  # litellm set max to 128k, but that requires a header to be set
            # Try to get from model info
            elif self.model_info is not None:
                # max_output_tokens has precedence over max_tokens
                if 'max_output_tokens' in self.model_info and isinstance(
                    self.model_info['max_output_tokens'], int
                ):
                    self.config.max_output_tokens = self.model_info['max_output_tokens']
                elif 'max_tokens' in self.model_info and isinstance(
                    self.model_info['max_tokens'], int
                ):
                    self.config.max_output_tokens = self.model_info['max_tokens']

        # Initialize function calling using centralized model features
        features = get_features(self.config.model)
        if self.config.native_tool_calling is None:
            self._function_calling_active = features.supports_function_calling
        else:
            self._function_calling_active = self.config.native_tool_calling

    def vision_is_active(self) -> bool:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return not self.config.disable_vision and self._supports_vision()

    def _supports_vision(self) -> bool:
        """Acquire from litellm if model is vision capable.

        Returns:
            bool: True if model is vision capable. Return False if model not supported by litellm.
        """

        # Allow manual override via environment variable
        if os.getenv('OPENHANDS_FORCE_VISION', '').lower() in (
            '1',
            'true',
            'yes',
            'on',
        ):
            return True

        # litellm.supports_vision currently returns False for 'openai/gpt-...' or 'anthropic/claude-...' (with prefixes)
        # but model_info will have the correct value for some reason.
        # we can go with it, but we will need to keep an eye if model_info is correct for Vertex or other providers
        # remove when litellm is updated to fix https://github.com/BerriAI/litellm/issues/5608
        # Check both the full model name and the name after proxy prefix for vision support
        return (
            litellm.supports_vision(self.config.model)
            or litellm.supports_vision(self.config.model.split('/')[-1])
            or (
                self.model_info is not None
                and self.model_info.get('supports_vision', False)
            )
        )

    def is_caching_prompt_active(self) -> bool:
        """Check if prompt caching is supported and enabled for current model.

        Returns:
            boolean: True if prompt caching is supported and enabled for the given model.
        """
        if not self.config.caching_prompt:
            return False
        # We don't need to look-up model_info, because only Anthropic models need explicit caching breakpoints
        return get_features(self.config.model).supports_prompt_cache

    def is_function_calling_active(self) -> bool:
        """Returns whether function calling is supported and enabled for this LLM instance.

        The result is cached during initialization for performance.
        """
        return self._function_calling_active

    def _post_completion(self, response: ModelResponse) -> float:
        """Post-process the completion response.

        Logs the cost and usage stats of the completion call.
        """
        try:
            cur_cost = self._completion_cost(response)
        except Exception:
            cur_cost = 0

        stats = ''
        if self.cost_metric_supported:
            # keep track of the cost
            stats = 'Cost: %.2f USD | Accumulated Cost: %.2f USD\n' % (
                cur_cost,
                self.metrics.accumulated_cost,
            )

        # Add latency to stats if available
        if self.metrics.response_latencies:
            latest_latency = self.metrics.response_latencies[-1]
            stats += 'Response Latency: %.3f seconds\n' % latest_latency.latency

        usage: Usage | None = response.get('usage')
        response_id = response.get('id', 'unknown')

        if usage:
            # keep track of the input and output tokens
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)

            if prompt_tokens:
                stats += 'Input tokens: ' + str(prompt_tokens)

            if completion_tokens:
                stats += (
                    (' | ' if prompt_tokens else '')
                    + 'Output tokens: '
                    + str(completion_tokens)
                    + '\n'
                )

            # read the prompt cache hit, if any
            prompt_tokens_details: PromptTokensDetails = usage.get(
                'prompt_tokens_details'
            )
            cache_hit_tokens = (
                prompt_tokens_details.cached_tokens
                if prompt_tokens_details and prompt_tokens_details.cached_tokens
                else 0
            )
            if cache_hit_tokens:
                stats += 'Input tokens (cache hit): ' + str(cache_hit_tokens) + '\n'

            # For Anthropic, the cache writes have a different cost than regular input tokens
            # but litellm doesn't separate them in the usage stats
            # we can read it from the provider-specific extra field
            model_extra = usage.get('model_extra', {})
            cache_write_tokens = model_extra.get('cache_creation_input_tokens', 0)
            if cache_write_tokens:
                stats += 'Input tokens (cache write): ' + str(cache_write_tokens) + '\n'

            # Get context window from model info
            context_window = 0
            if self.model_info and 'max_input_tokens' in self.model_info:
                context_window = self.model_info['max_input_tokens']
                logger.debug(f'Using context window: {context_window}')

            # Record in metrics
            # We'll treat cache_hit_tokens as "cache read" and cache_write_tokens as "cache write"
            self.metrics.add_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_read_tokens=cache_hit_tokens,
                cache_write_tokens=cache_write_tokens,
                context_window=context_window,
                response_id=response_id,
            )

        # log the stats
        if stats:
            logger.debug(stats)

        return cur_cost

    def get_token_count(self, messages: list[dict] | list[Message]) -> int:
        """Get the number of tokens in a list of messages. Use dicts for better token counting.

        Args:
            messages (list): A list of messages, either as a list of dicts or as a list of Message objects.

        Returns:
            int: The number of tokens.
        """
        # attempt to convert Message objects to dicts, litellm expects dicts
        if (
            isinstance(messages, list)
            and len(messages) > 0
            and isinstance(messages[0], Message)
        ):
            logger.info(
                'Message objects now include serialized tool calls in token counting'
            )
            # Assert the expected type for format_messages_for_llm
            assert isinstance(messages, list) and all(
                isinstance(m, Message) for m in messages
            ), 'Expected list of Message objects'

            # We've already asserted that messages is a list of Message objects
            # Use explicit typing to satisfy mypy
            messages_typed: list[Message] = messages  # type: ignore
            messages = self.format_messages_for_llm(messages_typed)

        # try to get the token count with the default litellm tokenizers
        # or the custom tokenizer if set for this LLM configuration
        try:
            return int(
                litellm.token_counter(
                    model=self.config.model,
                    messages=messages,
                    custom_tokenizer=self.tokenizer,
                )
            )
        except Exception as e:
            # limit logspam in case token count is not supported
            logger.error(
                f'Error getting token count for\n model {self.config.model}\n{e}'
                + (
                    f'\ncustom_tokenizer: {self.config.custom_tokenizer}'
                    if self.config.custom_tokenizer is not None
                    else ''
                )
            )
            return 0

    def _is_local(self) -> bool:
        """Determines if the system is using a locally running LLM.

        Returns:
            boolean: True if executing a local model.
        """
        if self.config.base_url is not None:
            for substring in ['localhost', '127.0.0.1', '0.0.0.0']:
                if substring in self.config.base_url:
                    return True
        elif self.config.model is not None:
            if self.config.model.startswith('ollama'):
                return True
        return False

    def _completion_cost(self, response: Any) -> float:
        """Calculate completion cost and update metrics with running total.

        Calculate the cost of a completion response based on the model. Local models are treated as free.
        Add the current cost into total cost in metrics.

        Args:
            response: A response from a model invocation.

        Returns:
            number: The cost of the response.
        """
        if not self.cost_metric_supported:
            return 0.0

        extra_kwargs = {}
        if (
            self.config.input_cost_per_token is not None
            and self.config.output_cost_per_token is not None
        ):
            cost_per_token = CostPerToken(
                input_cost_per_token=self.config.input_cost_per_token,
                output_cost_per_token=self.config.output_cost_per_token,
            )
            logger.debug(f'Using custom cost per token: {cost_per_token}')
            extra_kwargs['custom_cost_per_token'] = cost_per_token

        # try directly get response_cost from response
        _hidden_params = getattr(response, '_hidden_params', {})
        cost = _hidden_params.get('additional_headers', {}).get(
            'llm_provider-x-litellm-response-cost', None
        )
        if cost is not None:
            cost = float(cost)
            logger.debug(f'Got response_cost from response: {cost}')

        try:
            if cost is None:
                try:
                    cost = litellm_completion_cost(
                        completion_response=response, **extra_kwargs
                    )
                except Exception as e:
                    logger.debug(f'Error getting cost from litellm: {e}')

            if cost is None:
                _model_name = '/'.join(self.config.model.split('/')[1:])
                cost = litellm_completion_cost(
                    completion_response=response, model=_model_name, **extra_kwargs
                )
                logger.debug(
                    f'Using fallback model name {_model_name} to get cost: {cost}'
                )
            self.metrics.add_cost(float(cost))
            return float(cost)
        except Exception:
            self.cost_metric_supported = False
            logger.debug('Cost calculation not supported for this model.')
        return 0.0

    def __str__(self) -> str:
        if self.config.api_version:
            return f'LLM(model={self.config.model}, api_version={self.config.api_version}, base_url={self.config.base_url})'
        elif self.config.base_url:
            return f'LLM(model={self.config.model}, base_url={self.config.base_url})'
        return f'LLM(model={self.config.model})'

    def __repr__(self) -> str:
        return str(self)

    def format_messages_for_llm(self, messages: Message | list[Message]) -> list[dict]:
        if isinstance(messages, Message):
            messages = [messages]

        # set flags to know how to serialize the messages
        for message in messages:
            message.cache_enabled = self.is_caching_prompt_active()
            message.vision_enabled = self.vision_is_active()
            message.function_calling_enabled = self.is_function_calling_active()
            if 'deepseek' in self.config.model:
                message.force_string_serializer = True
            if 'kimi-k2-instruct' in self.config.model and 'groq' in self.config.model:
                message.force_string_serializer = True
            if any(
                k in self.config.model
                for k in (
                    'openrouter/anthropic/claude-sonnet-4',
                    'openrouter/anthropic/claude-sonnet-4-5-20250929',
                    'openrouter/anthropic/claude-haiku-4-5-20251001',
                )
            ):
                message.force_string_serializer = True

        # let pydantic handle the serialization
        return [message.model_dump() for message in messages]
