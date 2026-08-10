"""Structured MCP tool result helpers."""

from __future__ import annotations

import json
from typing import Any


def ok(data: dict[str, Any]) -> str:
    return json.dumps({"ok": True, **data}, default=str)


def err(error: str, **extra: Any) -> str:
    return json.dumps({"ok": False, "error": error, **extra}, default=str)
