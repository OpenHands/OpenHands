"""Typed outcome value validation."""

import json
import math
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal


# Persisted factory registries and parser configs depend on this cap; lowering it
# would make existing saved outcome definitions invalid.
MAX_OUTCOME_KEYS = 50
MAX_OUTCOME_PAYLOAD_BYTES = 64 * 1024
MAX_OUTCOME_STRING_LENGTH = 2048
OUTCOME_KEY_PATTERN = r"^[a-zA-Z0-9_.-]+$"

OutcomeValueType = Literal["number", "string", "boolean", "datetime", "json"]
OutcomeAggregation = Literal[
    "sum", "count", "avg", "min", "max", "count_true", "count_false"
]


def validate_outcome_payload_size(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(payload, allow_nan=False, ensure_ascii=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"payload must be strict JSON: {exc}") from exc
    if len(encoded) > MAX_OUTCOME_PAYLOAD_BYTES:
        raise ValueError(
            f"payload exceeds maximum size of {MAX_OUTCOME_PAYLOAD_BYTES} bytes"
        )
    return payload


def validate_outcome_key(key: str) -> str:
    if len(key) > 100 or not re.fullmatch(OUTCOME_KEY_PATTERN, key):
        raise ValueError(f"Invalid outcome key: {key!r}")
    return key


def _validate_string(value: str, key: str) -> str:
    if len(value) > MAX_OUTCOME_STRING_LENGTH:
        raise ValueError(
            f"Outcome value '{key}' exceeds {MAX_OUTCOME_STRING_LENGTH} characters"
        )
    return value


def _validate_number(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Outcome value '{key}' is not a number")
    if abs(value) > 2**53:
        raise ValueError(f"Outcome value {key!r} exceeds the 2^53 numeric limit")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Outcome value '{key}' must be finite")
    return numeric


def _validate_datetime(value: Any, key: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"Outcome value '{key}' is not an ISO datetime string")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"Outcome value '{key}' is not an ISO datetime string"
        ) from exc


def _validate_json_value(value: Any, key: str) -> Any:
    if value is None:
        raise ValueError(f"Outcome value '{key}' cannot be null")
    try:
        json.dumps(value, allow_nan=False, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Outcome value '{key}' is not strict JSON: {exc}") from exc
    return value


def _typed_columns(
    key: str, value: Any, value_type: OutcomeValueType
) -> dict[str, Any]:
    columns: dict[str, Any] = {
        "key": key,
        "value_type": value_type,
        "numeric_value": None,
        "text_value": None,
        "boolean_value": None,
        "datetime_value": None,
        "json_value": None,
    }
    if value_type == "number":
        columns["numeric_value"] = _validate_number(value, key)
    elif value_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"Outcome value '{key}' is not a string")
        columns["text_value"] = _validate_string(value, key)
    elif value_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"Outcome value '{key}' is not a boolean")
        columns["boolean_value"] = value
    elif value_type == "datetime":
        columns["datetime_value"] = _validate_datetime(value, key)
    elif value_type == "json":
        columns["json_value"] = _validate_json_value(value, key)
    else:
        raise ValueError(f"Unsupported outcome value type: {value_type}")
    return columns


def _flat_value_type(value: Any, key: str) -> OutcomeValueType:
    if value is None:
        raise ValueError(f"Outcome value '{key}' cannot be null")
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        _validate_number(value, key)
        return "number"
    if isinstance(value, str):
        _validate_string(value, key)
        return "string"
    raise ValueError(
        f"Outcome value '{key}' must be a string, finite number, or boolean"
    )


def extract_outcome_value_columns(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a flat payload without parser configuration."""
    validate_outcome_payload_size(payload)
    if len(payload) > MAX_OUTCOME_KEYS:
        raise ValueError(f"At most {MAX_OUTCOME_KEYS} outcome keys are allowed")
    return [
        _typed_columns(validate_outcome_key(key), value, _flat_value_type(value, key))
        for key, value in payload.items()
    ]


def normalize_outcome_values(values: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate already-typed facts before persistence."""
    if not 0 < len(values) <= MAX_OUTCOME_KEYS:
        raise ValueError(f"Submit between 1 and {MAX_OUTCOME_KEYS} outcome values")
    validate_outcome_payload_size({"values": values})
    keys = [item["key"] for item in values]
    if len(keys) != len(set(keys)):
        raise ValueError("Outcome keys must be unique within a submission")
    return [
        _typed_columns(
            validate_outcome_key(item["key"]), item["value"], item["value_type"]
        )
        for item in values
    ]
