from dataclasses import dataclass, field

from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class LLMResponseObservation(Observation):
    """Observation capturing LLM response metadata for tracing/debugging.

    This is emitted after each LLM call to capture reasoning traces,
    token usage, and latency for observability.
    """

    observation: str = ObservationType.LLM_RESPONSE

    # Model info
    model: str = ''

    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Timing
    latency_ms: int = 0

    # Raw response content (reasoning/chain-of-thought)
    # This is the full text content from the LLM before parsing into actions
    response_content: str = ''

    # Tool calls summary (list of tool names called)
    tool_calls: list[str] = field(default_factory=list)

    # Cost (if available)
    cost: float | None = None

    @property
    def message(self) -> str:
        return f'LLM response: {self.total_tokens} tokens, {self.latency_ms}ms'

