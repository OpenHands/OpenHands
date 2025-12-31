import copy
import time
from typing import final

from pydantic import BaseModel, Field, field_validator, model_validator


class Cost(BaseModel):
    model: str
    cost: float = Field(ge=0.0, description="Cost must be non-negative")
    timestamp: float = Field(default_factory=time.time)

    @field_validator("cost")
    @classmethod
    def validate_cost(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Cost cannot be negative")
        return v


class ResponseLatency(BaseModel):
    """Metric tracking the round-trip time per completion call."""

    model: str
    latency: float = Field(ge=0.0, description="Latency must be non-negative")
    response_id: str

    @field_validator("latency")
    @classmethod
    def validate_latency(cls, v: float) -> float:
        return max(0.0, v)


class TokenUsage(BaseModel):
    """Metric tracking detailed token usage per completion call."""

    model: str = Field(default="")
    prompt_tokens: int = Field(
        default=0, ge=0, description="Prompt tokens must be non-negative"
    )
    completion_tokens: int = Field(
        default=0, ge=0, description="Completion tokens must be non-negative"
    )
    cache_read_tokens: int = Field(
        default=0, ge=0, description="Cache read tokens must be non-negative"
    )
    cache_write_tokens: int = Field(
        default=0, ge=0, description="Cache write tokens must be non-negative"
    )
    reasoning_tokens: int = Field(
        default=0, ge=0, description="Reasoning tokens must be non-negative"
    )
    context_window: int = Field(
        default=0, ge=0, description="Context window must be non-negative"
    )
    per_turn_token: int = Field(
        default=0, ge=0, description="Per turn tokens must be non-negative"
    )
    response_id: str = Field(default="")

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage instances together."""
        return TokenUsage(
            model=self.model,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            context_window=max(self.context_window, other.context_window),
            per_turn_token=other.per_turn_token,
            response_id=self.response_id,
        )


class MetricsSnapshot(BaseModel):
    """A snapshot of metrics at a point in time.

    Does not include lists of individual costs, latencies, or token usages.
    """

    model_name: str = Field(default="default", description="Name of the model")
    accumulated_cost: float = Field(
        default=0.0, ge=0.0, description="Total accumulated cost, must be non-negative"
    )
    max_budget_per_task: float | None = Field(
        default=None, description="Maximum budget per task"
    )
    accumulated_token_usage: TokenUsage | None = Field(
        default=None, description="Accumulated token usage across all calls"
    )


@final
class Metrics(MetricsSnapshot):
    """Metrics class can record various metrics during running and evaluation.
    We track:
      - accumulated_cost and costs
      - max_budget_per_task (budget limit)
      - A list of ResponseLatency
      - A list of TokenUsage (one per call).
    """

    costs: list[Cost] = Field(
        default_factory=list, description="List of individual costs"
    )
    response_latencies: list[ResponseLatency] = Field(
        default_factory=list, description="List of response latencies"
    )
    token_usages: list[TokenUsage] = Field(
        default_factory=list, description="List of token usage records"
    )

    @field_validator("accumulated_cost")
    @classmethod
    def validate_accumulated_cost(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Total cost cannot be negative.")
        return v

    @model_validator(mode="after")
    def initialize_accumulated_token_usage(self) -> "Metrics":
        if self.accumulated_token_usage is None:
            self.accumulated_token_usage = TokenUsage(
                model=self.model_name,
                prompt_tokens=0,
                completion_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
                context_window=0,
                response_id="",
            )
        return self

    def get_snapshot(self) -> MetricsSnapshot:
        """Get a snapshot of the current metrics without the detailed lists."""
        return MetricsSnapshot(
            model_name=self.model_name,
            accumulated_cost=self.accumulated_cost,
            max_budget_per_task=self.max_budget_per_task,
            accumulated_token_usage=copy.deepcopy(self.accumulated_token_usage)
            if self.accumulated_token_usage
            else None,
        )

    def add_cost(self, value: float) -> None:
        if value < 0:
            raise ValueError("Added cost cannot be negative.")
        self.accumulated_cost += value
        self.costs.append(Cost(cost=value, model=self.model_name))

    def add_response_latency(self, value: float, response_id: str) -> None:
        self.response_latencies.append(
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
        reasoning_tokens: int = 0,
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
            reasoning_tokens=reasoning_tokens,
            context_window=context_window,
            per_turn_token=per_turn_token,
            response_id=response_id,
        )
        self.token_usages.append(usage)

        # Update accumulated token usage using the __add__ operator
        new_usage = TokenUsage(
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            reasoning_tokens=reasoning_tokens,
            context_window=context_window,
            per_turn_token=per_turn_token,
            response_id="",
        )
        if self.accumulated_token_usage is None:
            self.accumulated_token_usage = new_usage
        else:
            self.accumulated_token_usage = self.accumulated_token_usage + new_usage

    def merge(self, other: "Metrics") -> None:
        """Merge 'other' metrics into this one."""
        self.accumulated_cost += other.accumulated_cost

        # Keep the max_budget_per_task from other if it's set and this one isn't
        if self.max_budget_per_task is None and other.max_budget_per_task is not None:
            self.max_budget_per_task = other.max_budget_per_task

        self.costs += other.costs
        self.token_usages += other.token_usages
        self.response_latencies += other.response_latencies

        # Merge accumulated token usage using the __add__ operator
        if self.accumulated_token_usage is None:
            self.accumulated_token_usage = other.accumulated_token_usage
        elif other.accumulated_token_usage is not None:
            self.accumulated_token_usage = (
                self.accumulated_token_usage + other.accumulated_token_usage
            )

    def get(self) -> dict:
        """Return the metrics in a dictionary."""
        return {
            "accumulated_cost": self.accumulated_cost,
            "max_budget_per_task": self.max_budget_per_task,
            "accumulated_token_usage": self.accumulated_token_usage.model_dump()
            if self.accumulated_token_usage
            else None,
            "costs": [cost.model_dump() for cost in self.costs],
            "response_latencies": [
                latency.model_dump() for latency in self.response_latencies
            ],
            "token_usages": [usage.model_dump() for usage in self.token_usages],
        }

    def log(self) -> str:
        """Log the metrics."""
        metrics = self.get()
        logs = ""
        for key, value in metrics.items():
            logs += f"{key}: {value}\n"
        return logs

    def deep_copy(self) -> "Metrics":
        """Create a deep copy of the Metrics object."""
        return copy.deepcopy(self)

    def diff(self, baseline: "Metrics") -> "Metrics":
        """Calculate the difference between current metrics and a baseline.

        This is useful for tracking metrics for specific operations like delegates.

        Args:
            baseline: A metrics object representing the baseline state

        Returns:
            A new Metrics object containing only the differences since the baseline
        """
        result = Metrics(model_name=self.model_name)

        # Calculate cost difference
        result.accumulated_cost = self.accumulated_cost - baseline.accumulated_cost

        # Include only costs that were added after the baseline
        if baseline.costs:
            last_baseline_timestamp = baseline.costs[-1].timestamp
            result.costs = [
                cost for cost in self.costs if cost.timestamp > last_baseline_timestamp
            ]
        else:
            result.costs = self.costs.copy()

        # Include only response latencies that were added after the baseline
        result.response_latencies = self.response_latencies[
            len(baseline.response_latencies) :
        ]

        # Include only token usages that were added after the baseline
        result.token_usages = self.token_usages[len(baseline.token_usages) :]

        # Calculate accumulated token usage difference
        base_usage = baseline.accumulated_token_usage
        current_usage = self.accumulated_token_usage

        if current_usage is not None and base_usage is not None:
            result.accumulated_token_usage = TokenUsage(
                model=self.model_name,
                prompt_tokens=current_usage.prompt_tokens - base_usage.prompt_tokens,
                completion_tokens=current_usage.completion_tokens
                - base_usage.completion_tokens,
                cache_read_tokens=current_usage.cache_read_tokens
                - base_usage.cache_read_tokens,
                cache_write_tokens=current_usage.cache_write_tokens
                - base_usage.cache_write_tokens,
                reasoning_tokens=current_usage.reasoning_tokens
                - base_usage.reasoning_tokens,
                context_window=current_usage.context_window,
                per_turn_token=0,
                response_id="",
            )
        elif current_usage is not None:
            result.accumulated_token_usage = current_usage
        else:
            result.accumulated_token_usage = None

        return result

    def __repr__(self) -> str:
        return f"Metrics({self.get()}"
