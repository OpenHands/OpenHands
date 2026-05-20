# pyright: reportIncompatibleMethodOverride=false
"""Memory-integrity event callback processor.

Runs lightweight scans over conversation events to surface signals of agent
memory poisoning (OWASP ASI06): prompt-injection markers, sensitive-data
exfiltration markers, size anomalies, and an append-only SHA-256 fingerprint
chain that can be used for offline forensic comparison.

Design notes
------------
- The agent's true memory/context lives in ``openhands-sdk``; this processor
  observes events *after* they reach the app server. It is therefore an
  audit/detection layer, not an in-line "block-before-write" guard. The
  ``BLOCK`` policy translates to recording an ``ERROR`` callback result that
  surfaces the violation via the existing event-callback result store.
- Two backends are supported behind a single processor surface:
    * ``BUILTIN`` (default): zero dependencies, regex-based detectors plus
      SHA-256 chaining. Always available.
    * ``OWASP_AGENT_MEMORY_GUARD``: opt-in adapter that *lazily* imports the
      ``agent_memory_guard`` PyPI package (`OWASP Incubator project
      <https://owasp.org/www-project-agent-memory-guard/>`_). The package is
      **not** declared as a project dependency; ``ImportError`` is raised with
      install guidance if the backend is requested without the package being
      installed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from enum import Enum
from threading import Lock
from typing import Any, ClassVar
from uuid import UUID

from pydantic import Field

from openhands.app_server.event_callback.event_callback_models import (
    EventCallback,
    EventCallbackProcessor,
)
from openhands.app_server.event_callback.event_callback_result_models import (
    EventCallbackResult,
    EventCallbackResultStatus,
)
from openhands.sdk import Event
from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.llm import ImageContent, TextContent
from openhands.sdk.utils.redact import redact_text_secrets

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public enums
# ---------------------------------------------------------------------------


class MemoryIntegrityPolicy(str, Enum):
    """How to react when a finding is produced."""

    AUDIT = 'AUDIT'  # record SUCCESS with findings in detail; no signal change
    WARN = 'WARN'  # log a warning and record SUCCESS with findings
    BLOCK = 'BLOCK'  # record an ERROR result so operators can react


class MemoryIntegrityBackend(str, Enum):
    """Selectable detector backend."""

    BUILTIN = 'BUILTIN'
    OWASP_AGENT_MEMORY_GUARD = 'OWASP_AGENT_MEMORY_GUARD'


class FindingSeverity(str, Enum):
    INFO = 'INFO'
    WARNING = 'WARNING'
    CRITICAL = 'CRITICAL'


# ---------------------------------------------------------------------------
# Finding model
# ---------------------------------------------------------------------------


def _finding(code: str, severity: FindingSeverity, detail: str) -> dict[str, str]:
    return {'code': code, 'severity': severity.value, 'detail': detail}


# ---------------------------------------------------------------------------
# Built-in detectors
# ---------------------------------------------------------------------------


# Prompt-injection markers. Kept intentionally narrow to limit false positives;
# the goal here is high-signal, not exhaustive — operators who need broader
# coverage can opt into the OWASP backend.
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        'injection.ignore_previous',
        re.compile(
            r'ignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+instructions?',
            re.IGNORECASE,
        ),
    ),
    (
        'injection.disregard_system',
        re.compile(
            r'disregard\s+(the\s+)?(system|above|previous)\s+(prompt|instructions?)?',
            re.IGNORECASE,
        ),
    ),
    (
        'injection.role_override',
        re.compile(
            r'you\s+are\s+now\s+(?:a|an|the)\s+\w+',
            re.IGNORECASE,
        ),
    ),
    (
        'injection.fake_system_tag',
        re.compile(r'<\|(?:im_start|system|assistant)\|>', re.IGNORECASE),
    ),
    (
        'injection.developer_mode',
        re.compile(r'(developer|jailbreak|DAN)\s+mode\s+enabled', re.IGNORECASE),
    ),
)

# Sensitive-material markers. These suggest the agent's memory now carries
# credentials or private keys an attacker might exfiltrate on the next turn.
_LEAK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        'leak.private_key',
        re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----'),
    ),
    ('leak.aws_access_key', re.compile(r'\bAKIA[0-9A-Z]{16}\b')),
    ('leak.github_token', re.compile(r'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b')),
    ('leak.stripe_live_key', re.compile(r'\bsk_live_[A-Za-z0-9]{16,}\b')),
    ('leak.slack_token', re.compile(r'\bxox[abprs]-[A-Za-z0-9-]{10,}\b')),
)


# ---------------------------------------------------------------------------
# Per-process fingerprint chain
# ---------------------------------------------------------------------------


# EventCallbackProcessor instances are serialized to/from JSON per invocation,
# so we cannot keep state on the instance itself. Use a module-level registry
# keyed by conversation id. The chain is best-effort within a process lifetime
# — it gives operators something concrete to diff against an external log,
# which is the same property baselines have in the upstream OWASP package.
_CHAIN_LOCK = Lock()
_FINGERPRINTS: dict[UUID, str] = {}


def _extend_chain(conversation_id: UUID, event: Event, text: str) -> str:
    h = hashlib.sha256()
    with _CHAIN_LOCK:
        prev = _FINGERPRINTS.get(conversation_id, '')
        h.update(prev.encode('utf-8'))
        h.update(str(event.id).encode('utf-8'))
        h.update(str(event.timestamp).encode('utf-8'))
        h.update(text.encode('utf-8', errors='replace'))
        new_fp = h.hexdigest()
        _FINGERPRINTS[conversation_id] = new_fp
    return new_fp


def _reset_chain(conversation_id: UUID) -> None:
    """Test/operator hook: drop the in-memory chain for a conversation."""
    with _CHAIN_LOCK:
        _FINGERPRINTS.pop(conversation_id, None)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def _extract_text(event: Event) -> str:
    """Best-effort textual view of an event for scanning.

    For events that can be converted to an LLM message (i.e. things that
    actually enter the model's context window) we scan the message text. For
    other events we fall back to the redacted ``str(event)``.
    """
    if isinstance(event, LLMConvertibleEvent):
        try:
            message = event.to_llm_message()
        except Exception:  # pragma: no cover - defensive
            return redact_text_secrets(str(event))
        parts: list[str] = []
        for piece in message.content:
            if isinstance(piece, TextContent):
                parts.append(piece.text)
            elif isinstance(piece, ImageContent):
                parts.append(f'[image:{len(piece.image_urls)}]')
        return '\n'.join(parts)
    return redact_text_secrets(str(event))


# ---------------------------------------------------------------------------
# Built-in backend
# ---------------------------------------------------------------------------


def _scan_builtin(text: str, max_text_bytes: int) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []

    encoded_len = len(text.encode('utf-8', errors='replace'))
    if encoded_len > max_text_bytes:
        findings.append(
            _finding(
                'anomaly.size',
                FindingSeverity.WARNING,
                f'event text {encoded_len} bytes exceeds threshold {max_text_bytes}',
            )
        )

    for code, pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            findings.append(
                _finding(
                    code,
                    FindingSeverity.CRITICAL,
                    f'matched injection pattern {code}',
                )
            )

    for code, pattern in _LEAK_PATTERNS:
        if pattern.search(text):
            findings.append(
                _finding(
                    code,
                    FindingSeverity.CRITICAL,
                    f'matched sensitive-material pattern {code}',
                )
            )

    return findings


# ---------------------------------------------------------------------------
# OWASP agent-memory-guard adapter (lazy)
# ---------------------------------------------------------------------------


_OWASP_INSTALL_HINT = (
    'OWASP_AGENT_MEMORY_GUARD backend requires the optional '
    "'agent-memory-guard' package. Install it with "
    "'pip install agent-memory-guard' (or add it to your environment)."
)


def _scan_owasp(text: str, event_id: str) -> list[dict[str, str]]:
    """Run the optional OWASP agent-memory-guard scan.

    The package is imported lazily so that selecting BUILTIN never pulls it
    in. A clean ImportError with install guidance is raised if the package
    is missing.
    """
    try:
        from agent_memory_guard import (  # type: ignore[import-not-found]
            MemoryGuard,
            Policy,
            PolicyViolation,
        )
    except ImportError as exc:  # pragma: no cover - exercised via test
        raise ImportError(_OWASP_INSTALL_HINT) from exc

    guard = MemoryGuard(policy=Policy.strict())
    try:
        # The upstream API raises PolicyViolation on a blocked write; we use
        # the event id as the key so each event is screened independently.
        guard.write(f'openhands.event.{event_id}', text)
    except PolicyViolation as exc:
        return [
            _finding(
                'owasp.policy_violation',
                FindingSeverity.CRITICAL,
                str(exc),
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class MemoryIntegrityCallbackProcessor(EventCallbackProcessor):
    """Scan conversation events for memory-poisoning signals.

    Attach via the standard ``POST /api/v1/event-callbacks`` endpoint. Each
    event matching the callback's filter is screened by the selected backend;
    findings are recorded on the callback result (status ``SUCCESS`` for
    ``AUDIT``/``WARN`` and ``ERROR`` for ``BLOCK``) and made available
    through ``GET /api/v1/event-callback-results``.

    Attributes:
    ----------
    policy:
        How to react to findings. ``AUDIT`` records them silently, ``WARN``
        also logs a warning, ``BLOCK`` flips the result status to ``ERROR``.
    backend:
        Detector backend. ``BUILTIN`` (default) uses the in-tree detectors;
        ``OWASP_AGENT_MEMORY_GUARD`` delegates to the optional
        ``agent-memory-guard`` package which must be installed separately.
    max_text_bytes:
        Size threshold for the size-anomaly finding (BUILTIN only).
    record_chain_fingerprint:
        When true, every screened event extends a per-conversation SHA-256
        chain; the latest fingerprint is appended to the finding payload as
        an INFO entry so operators can diff against an external audit log.
    """

    # Marker so other code can find this processor without an isinstance
    # import (avoids touching modules that don't already depend on this one).
    KIND: ClassVar[str] = 'MemoryIntegrityCallbackProcessor'

    policy: MemoryIntegrityPolicy = Field(default=MemoryIntegrityPolicy.AUDIT)
    backend: MemoryIntegrityBackend = Field(default=MemoryIntegrityBackend.BUILTIN)
    max_text_bytes: int = Field(default=1_048_576, ge=1_024)
    record_chain_fingerprint: bool = Field(default=True)

    async def __call__(
        self,
        conversation_id: UUID,
        callback: EventCallback,
        event: Event,
    ) -> EventCallbackResult | None:
        text = _extract_text(event)

        try:
            if self.backend is MemoryIntegrityBackend.OWASP_AGENT_MEMORY_GUARD:
                findings = _scan_owasp(text, str(event.id))
            else:
                findings = _scan_builtin(text, self.max_text_bytes)
        except ImportError as exc:
            # Surface the missing optional dep as a callback ERROR result so
            # the operator sees it; do not crash the callback loop.
            _logger.error('memory-integrity backend unavailable: %s', exc)
            return EventCallbackResult(
                status=EventCallbackResultStatus.ERROR,
                event_callback_id=callback.id,
                event_id=event.id,
                conversation_id=conversation_id,
                detail=str(exc),
            )

        if self.record_chain_fingerprint:
            fp = _extend_chain(conversation_id, event, text)
            findings.append(
                _finding(
                    'integrity.chain_fingerprint',
                    FindingSeverity.INFO,
                    fp,
                )
            )

        critical_or_warning = any(
            f['severity']
            in (FindingSeverity.CRITICAL.value, FindingSeverity.WARNING.value)
            for f in findings
        )

        if self.policy is MemoryIntegrityPolicy.WARN and critical_or_warning:
            _logger.warning(
                'memory-integrity findings on conversation %s event %s: %s',
                conversation_id,
                event.id,
                redact_text_secrets(json.dumps(findings)),
            )

        status = EventCallbackResultStatus.SUCCESS
        if self.policy is MemoryIntegrityPolicy.BLOCK and critical_or_warning:
            status = EventCallbackResultStatus.ERROR
            _logger.error(
                'memory-integrity BLOCK on conversation %s event %s: %s',
                conversation_id,
                event.id,
                redact_text_secrets(json.dumps(findings)),
            )

        return EventCallbackResult(
            status=status,
            event_callback_id=callback.id,
            event_id=event.id,
            conversation_id=conversation_id,
            detail=_serialize_detail(findings),
        )


def _serialize_detail(findings: list[dict[str, Any]]) -> str:
    """Compact, redacted JSON for the result ``detail`` column."""
    return redact_text_secrets(
        json.dumps({'findings': findings}, separators=(',', ':'))
    )
