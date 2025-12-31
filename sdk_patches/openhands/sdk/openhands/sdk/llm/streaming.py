from collections.abc import Callable

from litellm.types.utils import ModelResponseStream


# Type alias for stream chunks
LLMStreamChunk = ModelResponseStream

TokenCallbackType = Callable[[LLMStreamChunk], None]
