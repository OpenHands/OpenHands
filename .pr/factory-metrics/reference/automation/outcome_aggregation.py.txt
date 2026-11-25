"""In-memory aggregation for factory-scoped outcome reads."""

from collections.abc import Sequence
from statistics import mean

from openhands.automation.models import AutomationOutcomeValue
from openhands.automation.outcome_values import OutcomeAggregation


def aggregate_outcome_values(
    values: Sequence[AutomationOutcomeValue], operation: OutcomeAggregation
) -> float | int | None:
    if operation == "count":
        return len(values)
    if operation == "count_true":
        return sum(1 for value in values if value.boolean_value is True)
    if operation == "count_false":
        return sum(1 for value in values if value.boolean_value is False)
    numbers = [
        value.numeric_value for value in values if value.numeric_value is not None
    ]
    if not numbers:
        return None
    if operation == "sum":
        return sum(numbers)
    if operation == "avg":
        return mean(numbers)
    if operation == "min":
        return min(numbers)
    if operation == "max":
        return max(numbers)
    raise ValueError(f"Unsupported aggregate operation: {operation}")
