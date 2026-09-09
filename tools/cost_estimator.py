"""Pre-assignment cost estimates, budget alerts, and actual-vs-estimate compare."""

from __future__ import annotations

from typing import Any

from kanban import KanbanError, KanbanStore

DEFAULT_MODEL = "claude-sonnet"
# USD per 1M tokens.
MODEL_PRICES: dict[str, dict[str, float]] = {
    "claude-sonnet": {"input": 3.0, "output": 15.0},
    "claude-opus": {"input": 15.0, "output": 75.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "glm-5.2": {"input": 0.5, "output": 1.5},
}
BASE_INPUT_TOKENS = 4000
DESCRIPTION_CHARS_PER_TOKEN = 4
OUTPUT_RATIO = 0.35
PRIORITY_MULTIPLIER = {"P0": 1.75, "P1": 1.35, "P2": 1.0, "P3": 0.7}
TOOL_CALLS_BY_COMPLEXITY = {"low": 4, "medium": 10, "high": 22}
AGENT_TIME_BY_COMPLEXITY = {"low": 120.0, "medium": 420.0, "high": 1200.0}
HIGH_DESCRIPTION_CHARS = 800
MEDIUM_DESCRIPTION_CHARS = 200
BUDGET_OK = "ok"
BUDGET_WARNING = "warning"
BUDGET_OVER = "over"
BUDGET_WARNING_RATIO = 0.8


def resolve_model(model: str | None) -> str:
    name = (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    if name not in MODEL_PRICES:
        raise KanbanError(f"Unknown model pricing for {name}")
    return name


def complexity_for(priority: str, description: str) -> str:
    length = len(description or "")
    if priority == "P0" or length >= HIGH_DESCRIPTION_CHARS:
        return "high"
    if priority == "P1" or length >= MEDIUM_DESCRIPTION_CHARS:
        return "medium"
    return "low"


def estimate_tokens(priority: str, title: str, description: str) -> int:
    multiplier = PRIORITY_MULTIPLIER.get(priority, 1.0)
    extra = (len(title or "") + len(description or "")) // DESCRIPTION_CHARS_PER_TOKEN
    return max(1, int((BASE_INPUT_TOKENS + extra) * multiplier))


def estimate_cost_usd(tokens: int, model: str | None = None) -> float:
    prices = MODEL_PRICES[resolve_model(model)]
    output_tokens = int(tokens * OUTPUT_RATIO)
    input_tokens = tokens - output_tokens
    usd = (
        input_tokens * prices["input"] + output_tokens * prices["output"]
    ) / 1_000_000
    return round(usd, 6)


def estimate_task(
    *,
    title: str,
    description: str = "",
    priority: str = "P2",
    model: str | None = None,
) -> dict[str, Any]:
    model_name = resolve_model(model)
    complexity = complexity_for(priority, description)
    tokens = estimate_tokens(priority, title, description)
    return {
        "model": model_name,
        "complexity": complexity,
        "estimate_tokens": tokens,
        "estimate_cost": estimate_cost_usd(tokens, model_name),
        "tool_calls": TOOL_CALLS_BY_COMPLEXITY[complexity],
        "agent_time": AGENT_TIME_BY_COMPLEXITY[complexity],
    }


def budget_status(
    spent: float,
    pending_estimate: float,
    cost_cap: float | None,
) -> dict[str, Any]:
    if cost_cap is None:
        return {
            "status": BUDGET_OK,
            "spent": spent,
            "pending_estimate": pending_estimate,
            "projected": spent + pending_estimate,
            "cost_cap": None,
        }
    projected = spent + pending_estimate
    if projected > cost_cap:
        status = BUDGET_OVER
    elif projected >= cost_cap * BUDGET_WARNING_RATIO:
        status = BUDGET_WARNING
    else:
        status = BUDGET_OK
    return {
        "status": status,
        "spent": spent,
        "pending_estimate": pending_estimate,
        "projected": projected,
        "cost_cap": cost_cap,
    }


def apply_estimate(
    store: KanbanStore,
    card_id: str,
    model: str | None = None,
    cost_cap: float | None = None,
) -> dict[str, Any]:
    card = store.get_card(card_id)
    estimate = estimate_task(
        title=str(card.get("title") or ""),
        description=str(card.get("description") or ""),
        priority=str(card.get("priority") or "P2"),
        model=model or card.get("model_used"),
    )
    costs = store.board_costs(card["board_id"])
    spent = float(costs["total_actual_cost"] or 0)
    alert = budget_status(spent, estimate["estimate_cost"], cost_cap)
    updated = store.update_card(
        card_id,
        estimate_tokens=estimate["estimate_tokens"],
        estimate_cost=estimate["estimate_cost"],
        tool_calls=estimate["tool_calls"],
        agent_time=estimate["agent_time"],
        model_used=estimate["model"],
    )
    if hasattr(store, "append_activity"):
        updated = store.append_activity(
            card_id,
            (
                f"Estimated ${estimate['estimate_cost']:.4f} "
                f"({estimate['estimate_tokens']} tokens, {estimate['model']})"
            ),
        )
    return {"card": updated, "estimate": estimate, "budget": alert}


def compare_estimate(card: dict[str, Any]) -> dict[str, Any]:
    estimate_cost = float(card.get("estimate_cost") or 0)
    actual_cost = float(card.get("actual_cost") or 0)
    estimate_tokens = int(card.get("estimate_tokens") or 0)
    actual_tokens = int(card.get("actual_tokens") or 0)
    return {
        "estimate_cost": estimate_cost,
        "actual_cost": actual_cost,
        "cost_delta": round(actual_cost - estimate_cost, 6),
        "estimate_tokens": estimate_tokens,
        "actual_tokens": actual_tokens,
        "token_delta": actual_tokens - estimate_tokens,
        "over_estimate": actual_cost > estimate_cost if estimate_cost else False,
    }


def record_actuals(
    store: KanbanStore,
    card_id: str,
    *,
    actual_tokens: int | None,
    actual_cost: float | None,
    tool_calls: int | None = None,
    agent_time: float | None = None,
    model_used: str | None = None,
) -> dict[str, Any]:
    card = store.update_card(
        card_id,
        actual_tokens=actual_tokens,
        actual_cost=actual_cost,
        tool_calls=tool_calls,
        agent_time=agent_time,
        model_used=model_used,
    )
    comparison = compare_estimate(card)
    if hasattr(store, "append_activity"):
        card = store.append_activity(
            card_id,
            (
                f"Actual ${comparison['actual_cost']:.4f} vs estimate "
                f"${comparison['estimate_cost']:.4f} "
                f"(delta {comparison['cost_delta']:+.4f})"
            ),
        )
    return {"card": card, "comparison": comparison}
