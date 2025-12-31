import json
import os
import time
import uuid
import warnings
from collections.abc import Callable
from typing import Any, ClassVar

from litellm.cost_calculator import completion_cost as litellm_completion_cost
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.utils import CostPerToken, ModelResponse, Usage
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from openhands.sdk.llm.utils.metrics import Metrics
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class Telemetry(BaseModel):
    """
    Handles latency, token/cost accounting, and optional logging.
    All runtime state (like start times) lives in private attrs.
    """

    # --- Config fields ---
    model_name: str = Field(default="unknown", description="Name of the LLM model")
    log_enabled: bool = Field(default=False, description="Whether to log completions")
    log_dir: str | None = Field(
        default=None, description="Directory to write logs if enabled"
    )
    input_cost_per_token: float | None = Field(
        default=None, ge=0, description="Custom Input cost per token (USD)"
    )
    output_cost_per_token: float | None = Field(
        default=None, ge=0, description="Custom Output cost per token (USD)"
    )

    metrics: Metrics = Field(..., description="Metrics collector instance")

    # --- Runtime fields (not serialized) ---
    _req_start: float = PrivateAttr(default=0.0)
    _req_ctx: dict[str, Any] = PrivateAttr(default_factory=dict)
    _last_latency: float = PrivateAttr(default=0.0)
    _log_completions_callback: Callable[[str, str], None] | None = PrivateAttr(
        default=None
    )
    _stats_update_callback: Callable[[], None] | None = PrivateAttr(default=None)

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True
    )

    # ---------- Lifecycle ----------
    def set_log_completions_callback(
        self, callback: Callable[[str, str], None] | None
    ) -> None:
        """Set a callback function for logging instead of writing to file.

        Args:
            callback: A function that takes (filename, log_data) and handles the log.
                     Used for streaming logs in remote execution contexts.
        """
        self._log_completions_callback = callback

    def set_stats_update_callback(self, callback: Callable[[], None] | None) -> None:
        """Set a callback function to be notified when stats are updated.

        Args:
            callback: A function called whenever metrics are updated.
                     Used for streaming stats updates in remote execution contexts.
        """
        self._stats_update_callback = callback

    def on_request(self, log_ctx: dict | None) -> None:
        self._req_start = time.time()
        self._req_ctx = log_ctx or {}

    def on_response(
        self,
        resp: ModelResponse | ResponsesAPIResponse,
        raw_resp: ModelResponse | None = None,
    ) -> Metrics:
        """
        Side-effects:
          - records latency, tokens, cost into Metrics
          - optionally writes a JSON log file
        """
        # 1) latency
        self._last_latency = time.time() - (self._req_start or time.time())
        response_id = resp.id
        self.metrics.add_response_latency(self._last_latency, response_id)

        # 2) cost
        cost = self._compute_cost(resp)
        # Intentionally skip logging zero-cost (0.0) responses; only record
        # positive cost
        if cost:
            self.metrics.add_cost(cost)

        # 3) tokens - use typed usage field when available
        usage = getattr(resp, "usage", None)

        if usage and self._has_meaningful_usage(usage):
            self._record_usage(
                usage, response_id, self._req_ctx.get("context_window", 0)
            )

        # 4) optional logging
        if self.log_enabled:
            self.log_llm_call(resp, cost, raw_resp=raw_resp)

        # 5) notify about stats update
        if self._stats_update_callback is not None:
            try:
                self._stats_update_callback()
            except Exception:
                logger.exception("Stats update callback failed", exc_info=True)

        return self.metrics.deep_copy()

    def on_error(self, _err: BaseException) -> None:
        # Stub for error tracking / counters
        return

    # ---------- Helpers ----------
    def _has_meaningful_usage(self, usage: Usage | ResponseAPIUsage | None) -> bool:
        """Check if usage has meaningful (non-zero) token counts.

        Supports both Chat Completions Usage and Responses API Usage shapes.
        """
        if usage is None:
            return False
        try:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            if prompt_tokens is None:
                prompt_tokens = getattr(usage, "input_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", None)
            if completion_tokens is None:
                completion_tokens = getattr(usage, "output_tokens", 0)

            pt = int(prompt_tokens or 0)
            ct = int(completion_tokens or 0)
            return pt > 0 or ct > 0
        except Exception:
            return False

    def _record_usage(
        self, usage: Usage | ResponseAPIUsage, response_id: str, context_window: int
    ) -> None:
        """
        Record token usage, supporting both Chat Completions Usage and
        Responses API Usage.

        Chat shape:
          - prompt_tokens, completion_tokens
          - prompt_tokens_details.cached_tokens
          - completion_tokens_details.reasoning_tokens
          - _cache_creation_input_tokens for cache_write
        Responses shape:
          - input_tokens, output_tokens
          - input_tokens_details.cached_tokens
          - output_tokens_details.reasoning_tokens
        """
        prompt_tokens = int(
            getattr(usage, "prompt_tokens", None)
            or getattr(usage, "input_tokens", 0)
            or 0
        )
        completion_tokens = int(
            getattr(usage, "completion_tokens", None)
            or getattr(usage, "output_tokens", 0)
            or 0
        )

        cache_read = 0
        p_details = getattr(usage, "prompt_tokens_details", None) or getattr(
            usage, "input_tokens_details", None
        )
        if p_details is not None:
            cache_read = int(getattr(p_details, "cached_tokens", 0) or 0)

        # Kimi-K2-thinking populate usage.cached_tokens field
        if not cache_read and hasattr(usage, "cached_tokens"):
            cache_read = int(getattr(usage, "cached_tokens", 0) or 0)

        reasoning_tokens = 0
        c_details = getattr(usage, "completion_tokens_details", None) or getattr(
            usage, "output_tokens_details", None
        )
        if c_details is not None:
            reasoning_tokens = int(getattr(c_details, "reasoning_tokens", 0) or 0)

        # Chat-specific: litellm may set a hidden cache write field
        cache_write = int(getattr(usage, "_cache_creation_input_tokens", 0) or 0)

        self.metrics.add_token_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            reasoning_tokens=reasoning_tokens,
            context_window=context_window,
            response_id=response_id,
        )

    def _compute_cost(self, resp: ModelResponse | ResponsesAPIResponse) -> float | None:
        """Try provider header → litellm direct. Return None on failure."""
        extra_kwargs = {}
        if (
            self.input_cost_per_token is not None
            and self.output_cost_per_token is not None
        ):
            cost_per_token = CostPerToken(
                input_cost_per_token=self.input_cost_per_token,
                output_cost_per_token=self.output_cost_per_token,
            )
            logger.debug(f"Using custom cost per token: {cost_per_token}")
            extra_kwargs["custom_cost_per_token"] = cost_per_token

        try:
            hidden = getattr(resp, "_hidden_params", {}) or {}
            cost = hidden.get("additional_headers", {}).get(
                "llm_provider-x-litellm-response-cost"
            )
            if cost is not None:
                return float(cost)
        except Exception as e:
            logger.debug(f"Failed to get cost from LiteLLM headers: {e}")

        # move on to litellm cost calculator
        # Handle model name properly - if it doesn't contain "/", use as-is
        model_parts = self.model_name.split("/")
        if len(model_parts) > 1:
            extra_kwargs["model"] = "/".join(model_parts[1:])
        else:
            extra_kwargs["model"] = self.model_name
        try:
            return float(
                litellm_completion_cost(completion_response=resp, **extra_kwargs)
            )
        except Exception as e:
            warnings.warn(f"Cost calculation failed: {e}")
            return None

    def log_llm_call(
        self,
        resp: ModelResponse | ResponsesAPIResponse,
        cost: float | None,
        raw_resp: ModelResponse | ResponsesAPIResponse | None = None,
    ) -> None:
        # Skip if neither file logging nor callback is configured
        if not self.log_dir and not self._log_completions_callback:
            return
        try:
            # Prepare filename and log data
            filename = (
                f"{self.model_name.replace('/', '__')}-"
                f"{time.time():.3f}-"
                f"{uuid.uuid4().hex[:4]}.json"
            )

            data = self._req_ctx.copy()
            data["response"] = (
                resp  # ModelResponse | ResponsesAPIResponse;
                # serialized via _safe_json
            )
            data["cost"] = float(cost or 0.0)
            data["timestamp"] = time.time()
            data["latency_sec"] = self._last_latency

            # Usage summary (prompt, completion, reasoning tokens) for quick inspection
            try:
                usage = getattr(resp, "usage", None)
                if usage:
                    prompt_tokens = int(
                        getattr(usage, "prompt_tokens", None)
                        or getattr(usage, "input_tokens", 0)
                        or 0
                    )
                    completion_tokens = int(
                        getattr(usage, "completion_tokens", None)
                        or getattr(usage, "output_tokens", 0)
                        or 0
                    )
                    details = getattr(
                        usage, "completion_tokens_details", None
                    ) or getattr(usage, "output_tokens_details", None)
                    reasoning_tokens = (
                        int(getattr(details, "reasoning_tokens", 0) or 0)
                        if details
                        else 0
                    )
                    p_details = getattr(
                        usage, "prompt_tokens_details", None
                    ) or getattr(usage, "input_tokens_details", None)
                    cache_read_tokens = (
                        int(getattr(p_details, "cached_tokens", 0) or 0)
                        if p_details
                        else 0
                    )

                    data["usage_summary"] = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "reasoning_tokens": reasoning_tokens,
                        "cache_read_tokens": cache_read_tokens,
                    }
            except Exception:
                # Best-effort only; don't fail logging
                pass

            # Raw response *before* nonfncall -> call conversion
            if raw_resp:
                data["raw_response"] = (
                    raw_resp  # ModelResponse | ResponsesAPIResponse;
                    # serialized via _safe_json
                )
            # Pop duplicated tools to avoid logging twice
            if (
                "tools" in data
                and isinstance(data.get("kwargs"), dict)
                and "tools" in data["kwargs"]
            ):
                data["kwargs"].pop("tools")

            log_data = json.dumps(data, default=_safe_json, ensure_ascii=False)

            # Use callback if set (for remote execution), otherwise write to file
            if self._log_completions_callback:
                self._log_completions_callback(filename, log_data)
            elif self.log_dir:
                # Create log directory if it doesn't exist
                os.makedirs(self.log_dir, exist_ok=True)
                if not os.access(self.log_dir, os.W_OK):
                    raise PermissionError(f"log_dir is not writable: {self.log_dir}")

                fname = os.path.join(self.log_dir, filename)
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(log_data)
        except Exception as e:
            warnings.warn(f"Telemetry logging failed: {e}")


def _safe_json(obj: Any) -> Any:
    # Centralized serializer for telemetry logs.
    # Prefer robust serialization for Pydantic models first to avoid cycles.
    # Typed LiteLLM responses
    if isinstance(obj, ModelResponse) or isinstance(obj, ResponsesAPIResponse):
        return obj.model_dump(mode="json", exclude_none=True)

    # Any Pydantic BaseModel (e.g., ToolDefinition, ChatCompletionToolParam, etc.)
    if isinstance(obj, BaseModel):
        # Use Pydantic's serializer which respects field exclusions (e.g., executors)
        return obj.model_dump(mode="json", exclude_none=True)

    # Fallbacks for other non-serializable objects used elsewhere in the log payload
    try:
        return obj.__dict__
    except Exception:
        return str(obj)
