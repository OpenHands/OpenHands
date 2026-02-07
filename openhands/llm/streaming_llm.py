# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import asyncio
from functools import partial
from typing import Any, Callable

from openhands.core.exceptions import UserCancelledError
from openhands.core.logger import openhands_logger as logger
from openhands.llm.async_llm import LLM_RETRY_EXCEPTIONS, AsyncLLM


class StreamingLLM(AsyncLLM):
    """Streaming LLM class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._async_streaming_completion = partial(
            self._call_acompletion,
            model=self.config.model,
            api_key=self.config.api_key.get_secret_value()
            if self.config.api_key
            else None,
            base_url=self.config.base_url,
            api_version=self.config.api_version,
            custom_llm_provider=self.config.custom_llm_provider,
            max_tokens=self.config.max_output_tokens,
            timeout=self.config.timeout,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            drop_params=self.config.drop_params,
            stream=True,  # Ensure streaming is enabled
        )

        async_streaming_completion_unwrapped = self._async_streaming_completion

        @self.retry_decorator(
            num_retries=self.config.num_retries,
            retry_exceptions=LLM_RETRY_EXCEPTIONS,
            retry_min_wait=self.config.retry_min_wait,
            retry_max_wait=self.config.retry_max_wait,
            retry_multiplier=self.config.retry_multiplier,
        )
        async def async_streaming_completion_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Use shared message extraction from base class
            args, messages = self._extract_messages(args, kwargs)

            # Apply reasoning effort with Gemini/Claude handling from base class
            self._apply_reasoning_effort(kwargs)

            self.log_prompt(messages)

            try:
                # Directly call and await litellm_acompletion
                resp = await async_streaming_completion_unwrapped(*args, **kwargs)

                # Track the last chunk for post-completion processing
                last_chunk = None

                # For streaming we iterate over the chunks
                async for chunk in resp:
                    # Check for cancellation before yielding the chunk
                    if (
                        hasattr(self.config, 'on_cancel_requested_fn')
                        and self.config.on_cancel_requested_fn is not None
                        and await self.config.on_cancel_requested_fn()
                    ):
                        raise UserCancelledError(
                            'LLM request cancelled due to CANCELLED state'
                        )
                    # with streaming, it is "delta", not "message"!
                    message_back = chunk['choices'][0]['delta'].get('content', '')
                    if message_back:
                        self.log_response(message_back)
                    last_chunk = chunk

                    yield chunk

                # Call _post_completion once after stream completes with the
                # final chunk (which typically contains usage data)
                if last_chunk is not None:
                    self._post_completion(last_chunk)

            except UserCancelledError:
                logger.debug('LLM request cancelled by user.')
                raise
            except Exception as e:
                logger.error(f'Completion Error occurred:\n{e}')
                raise

            finally:
                # sleep for 0.1 seconds to allow the stream to be flushed
                if kwargs.get('stream', False):
                    await asyncio.sleep(0.1)

        self._async_streaming_completion = async_streaming_completion_wrapper

    @property
    def async_streaming_completion(self) -> Callable:
        """Decorator for the async litellm acompletion function with streaming."""
        return self._async_streaming_completion
