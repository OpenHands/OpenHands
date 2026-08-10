"""Confirmation gate for intrusive MCP tools (blueprint §5.4)."""

from __future__ import annotations

import os
import secrets
import uuid
from typing import Any, Literal

AutonomyMode = Literal["manual", "semi_autonomous", "autonomous"]

AUTONOMY_MODE_ENV = "PENTEST_AUTONOMY_MODE"
DEFAULT_AUTONOMY_MODE: AutonomyMode = "semi_autonomous"

ACTIVE_TOOLS = frozenset({"zap_active_scan", "sqlmap_run", "nuclei_intrusive"})
# Outside MVP Fase 1 — autonomous mode does not block extra tools yet.
MAX_RISK_TOOLS: frozenset[str] = frozenset()

# request_id → approval token (stub until UI confirmation channel lands)
_pending: dict[str, str] = {}
_approved_tokens: set[str] = set()


class ConfirmationRequiredError(Exception):
    code = "confirmation_required"

    def __init__(self, tool_name: str, request_id: str, payload: dict[str, Any]):
        self.tool_name = tool_name
        self.request_id = request_id
        self.payload = payload
        super().__init__(
            f"Confirmation required for {tool_name} (request_id={request_id})"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "tool_name": self.tool_name,
            "request_id": self.request_id,
            "message": str(self),
            "payload": self.payload,
        }


def clear_confirmation_state() -> None:
    """Test helper — reset in-memory approval stubs."""
    _pending.clear()
    _approved_tokens.clear()


def approve_confirmation(request_id: str) -> str:
    """
    Stub approval channel (AC-187-6).

    Returns the token the caller must pass as confirmation_token /
    OPENHANDS_CONFIRMATION_TOKEN on re-run.
    """
    token = _pending.get(request_id) or f"approved:{request_id}:{secrets.token_hex(8)}"
    _pending[request_id] = token
    _approved_tokens.add(token)
    return token


def get_autonomy_mode() -> AutonomyMode:
    """
    Server-side autonomy only — never trust LLM/tool args.

    Source of truth: ``PENTEST_AUTONOMY_MODE`` (default ``semi_autonomous``).
    Unknown values fail closed to semi_autonomous.
    """
    raw = os.environ.get(AUTONOMY_MODE_ENV, DEFAULT_AUTONOMY_MODE).strip().lower()
    if raw in ("manual", "semi_autonomous", "autonomous"):
        return raw  # type: ignore[return-value]
    return DEFAULT_AUTONOMY_MODE


def _needs_gate(tool_name: str, autonomy_mode: AutonomyMode) -> bool:
    if autonomy_mode == "manual":
        return True
    if autonomy_mode == "semi_autonomous":
        return tool_name in ACTIVE_TOOLS
    # autonomous — only MAX_RISK_TOOLS (empty in Fase 1 MVP)
    return tool_name in MAX_RISK_TOOLS


async def require_confirmation(
    tool_name: str,
    payload: dict[str, Any],
    *,
    confirmation_token: str | None = None,
) -> None:
    """
    Gate intrusive tools using server-side autonomy (``PENTEST_AUTONOMY_MODE``).

    - manual: always requires approval
    - semi_autonomous: requires if tool_name in ACTIVE_TOOLS
    - autonomous: only blocks MAX_RISK_TOOLS (empty in Fase 1 MVP)

    Agent/tool arguments must never supply autonomy — that was HIGH-2 bypass.

    On first call without a valid token, raises ConfirmationRequiredError with
    request_id. Re-run with the approved token to proceed.
    """
    mode = get_autonomy_mode()

    if not _needs_gate(tool_name, mode):
        return

    if confirmation_token and confirmation_token in _approved_tokens:
        return

    env_token = os.environ.get("OPENHANDS_CONFIRMATION_TOKEN", "").strip()
    if confirmation_token and env_token and confirmation_token == env_token:
        return
    if env_token and env_token in _approved_tokens:
        return

    request_id = str(uuid.uuid4())
    token = f"approved:{request_id}:{secrets.token_hex(8)}"
    _pending[request_id] = token
    raise ConfirmationRequiredError(tool_name, request_id, payload)
