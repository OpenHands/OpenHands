# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import copy
import time

from pydantic import BaseModel, Field


class Cost(BaseModel):
    model: str
    cost: float
    timestamp: float = Field(default_factory=time.time)


class ResponseLatency(BaseModel):
    """Metric tracking the round-trip time per completion call."""

    model: str
    latency: float
    response_id: str


class TokenUsage(BaseModel):
    """Metric tracking detailed token usage per completion call."""

    model: str = Field(default='')
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    cache_read_tokens: int = Field(default=0)
    cache_write_tokens: int = Field(default=0)
    context_window: int = Field(default=0)
    per_turn_token: int = Field(default=0)
    response_id: str = Field(default='')

    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        """Add two TokenUsage instances together."""
        return TokenUsage(
            model=self.model,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            context_window=max(self.context_window, other.context_window),
            per_turn_token=other.per_turn_token,
            response_id=self.response_id,
        )


class Metrics:
    """Metrics class can record various metrics during running and evaluation.
    We track:
      - accumulated_cost and costs
      - max_budget_per_task (budget limit)
      - A list of ResponseLatency
      - A list of TokenUsage (one per call).
    """

    def __init__(self, model_name: str = 'default') -> None:
        self._accumulated_cost: float = 0.0
        self._max_budget_per_task: float | None = None
        self._costs: list[Cost] = []
        self._response_latencies: list[ResponseLatency] = []
        self.model_name = model_name
        self._token_usages: list[TokenUsage] = []
        self._accumulated_token_usage: TokenUsage = TokenUsage(
            model=model_name,
            prompt_tokens=0,
            completion_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            context_window=0,
            response_id='',
        )

    @property
    def accumulated_cost(self) -> float:
        return self._accumulated_cost

    @accumulated_cost.setter
    def accumulated_cost(self, value: float) -> None:
        if value < 0:
            raise ValueError('Total cost cannot be negative.')
        self._accumulated_cost = value

    @property
    def max_budget_per_task(self) -> float | None:
        return self._max_budget_per_task

    @max_budget_per_task.setter
    def max_budget_per_task(self, value: float | None) -> None:
        self._max_budget_per_task = value

    @property
    def costs(self) -> list[Cost]:
        return self._costs

    @property
    def response_latencies(self) -> list[ResponseLatency]:
        if not hasattr(self, '_response_latencies'):
            self._response_latencies = []
        return self._response_latencies

    @response_latencies.setter
    def response_latencies(self, value: list[ResponseLatency]) -> None:
        self._response_latencies = value

    @property
    def token_usages(self) -> list[TokenUsage]:
        if not hasattr(self, '_token_usages'):
            self._token_usages = []
        return self._token_usages

    @token_usages.setter
    def token_usages(self, value: list[TokenUsage]) -> None:
        self._token_usages = value

    @property
    def accumulated_token_usage(self) -> TokenUsage:
        """Get the accumulated token usage, initializing it if it doesn't exist."""
        if not hasattr(self, '_accumulated_token_usage'):
            self._accumulated_token_usage = TokenUsage(
                model=self.model_name,
                prompt_tokens=0,
                completion_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                context_window=0,
                response_id='',
            )
        return self._accumulated_token_usage

    def add_cost(self, value: float) -> None:
        if value < 0:
            raise ValueError('Added cost cannot be negative.')
        self._accumulated_cost += value
        self._costs.append(Cost(cost=value, model=self.model_name))

    def add_response_latency(self, value: float, response_id: str) -> None:
        self._response_latencies.append(
            ResponseLatency(
                latency=max(0.0, value), model=self.model_name, response_id=response_id
            )
        )

    def add_token_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        context_window: int,
        response_id: str,
    ) -> None:
        """Add a single usage record."""
        # Token each turn for calculating context usage.
        per_turn_token = prompt_tokens + completion_tokens

        usage = TokenUsage(
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            context_window=context_window,
            per_turn_token=per_turn_token,
            response_id=response_id,
        )
        self._token_usages.append(usage)

        # Update accumulated token usage using the __add__ operator
        self._accumulated_token_usage = self.accumulated_token_usage + TokenUsage(
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            context_window=context_window,
            per_turn_token=per_turn_token,
            response_id='',
        )

    def merge(self, other: 'Metrics') -> None:
        """Merge 'other' metrics into this one."""
        self._accumulated_cost += other.accumulated_cost

        # Keep the max_budget_per_task from other if it's set and this one isn't
        if self._max_budget_per_task is None and other.max_budget_per_task is not None:
            self._max_budget_per_task = other.max_budget_per_task

        self._costs += other._costs
        # use the property so older picked objects that lack the field won't crash
        self.token_usages += other.token_usages
        self.response_latencies += other.response_latencies

        # Merge accumulated token usage using the __add__ operator
        self._accumulated_token_usage = (
            self.accumulated_token_usage + other.accumulated_token_usage
        )

    def get(self) -> dict:
        """Return the metrics in a dictionary."""
        return {
            'accumulated_cost': self._accumulated_cost,
            'max_budget_per_task': self._max_budget_per_task,
            'accumulated_token_usage': self.accumulated_token_usage.model_dump(),
            'costs': [cost.model_dump() for cost in self._costs],
            'response_latencies': [
                latency.model_dump() for latency in self._response_latencies
            ],
            'token_usages': [usage.model_dump() for usage in self._token_usages],
        }

    def log(self) -> str:
        """Log the metrics."""
        metrics = self.get()
        logs = ''
        for key, value in metrics.items():
            logs += f'{key}: {value}\n'
        return logs

    def copy(self) -> 'Metrics':
        """Create a deep copy of the Metrics object."""
        return copy.deepcopy(self)

    def diff(self, baseline: 'Metrics') -> 'Metrics':
        """Calculate the difference between current metrics and a baseline.

        This is useful for tracking metrics for specific operations like delegates.

        Args:
            baseline: A metrics object representing the baseline state

        Returns:
            A new Metrics object containing only the differences since the baseline
        """
        result = Metrics(self.model_name)

        # Calculate cost difference
        result._accumulated_cost = self._accumulated_cost - baseline._accumulated_cost

        # Include only costs that were added after the baseline
        if baseline._costs:
            last_baseline_timestamp = baseline._costs[-1].timestamp
            result._costs = [
                cost for cost in self._costs if cost.timestamp > last_baseline_timestamp
            ]
        else:
            result._costs = self._costs.copy()

        # Include only response latencies that were added after the baseline
        result._response_latencies = self._response_latencies[
            len(baseline._response_latencies) :
        ]

        # Include only token usages that were added after the baseline
        result._token_usages = self._token_usages[len(baseline._token_usages) :]

        # Calculate accumulated token usage difference
        base_usage = baseline.accumulated_token_usage
        current_usage = self.accumulated_token_usage

        result._accumulated_token_usage = TokenUsage(
            model=self.model_name,
            prompt_tokens=current_usage.prompt_tokens - base_usage.prompt_tokens,
            completion_tokens=current_usage.completion_tokens
            - base_usage.completion_tokens,
            cache_read_tokens=current_usage.cache_read_tokens
            - base_usage.cache_read_tokens,
            cache_write_tokens=current_usage.cache_write_tokens
            - base_usage.cache_write_tokens,
            context_window=current_usage.context_window,
            per_turn_token=0,
            response_id='',
        )

        return result

    def __repr__(self) -> str:
        return f'Metrics({self.get()}'
