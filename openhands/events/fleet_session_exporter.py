"""Fleet session logging exporter.

This integrates OpenHands with Fleet's session logging API (via fleet-sdk).

Design:
- Fleet session logging is a *client-side* concern: it needs access to the LLM
  prompt history + raw model response. Therefore the main hook is in the agent
  (where `history` and `response` exist).
- Session completion (complete/fail) is intentionally *explicit* so outer harnesses
  can run verifiers and complete with a verifier_execution_id.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openhands.core.logger import openhands_logger as logger
from openhands.events.stream import EventStream


@dataclass(frozen=True)
class FleetSessionExportConfig:
    enabled: bool = False
    api_key: str | None = None
    base_url: str | None = None
    job_id: str | None = None
    task_key: str | None = None
    instance_id: str | None = None
    model: str | None = None


class FleetSessionExporter:
    """Export LLM-call traces to Fleet Sessions.

    This uses fleet-sdk's Session.log(history, response) which sends to Fleet's
    `/v1/traces/logs` endpoint and allows the backend to normalize formats.
    """

    def __init__(self, event_stream: EventStream, cfg: FleetSessionExportConfig):
        self._event_stream = event_stream
        self._cfg = cfg
        self._session: Any | None = None
        self._enabled = bool(cfg.enabled)
        self._failed = False

    @property
    def enabled(self) -> bool:
        return self._enabled and not self._failed

    @property
    def session_id(self) -> str | None:
        return getattr(self._session, 'session_id', None) if self._session else None

    def start(self) -> None:
        if not self.enabled:
            return
        logger.info(
            'Fleet session exporter enabled',
            extra={
                'session_id': self._event_stream.sid,
                'fleet_job_id': self._cfg.job_id,
                'fleet_task_key': self._cfg.task_key,
            },
        )

    def stop(self) -> None:
        # No-op (explicit completion only; no event-stream subscription).
        return

    def _ensure_session(self) -> Any:
        if self._session is not None:
            return self._session

        # Optional dependency
        try:
            import fleet  # type: ignore[import-not-found]
        except Exception as e:  # noqa: BLE001
            self._failed = True
            raise ImportError(
                "Fleet session export requires fleet-sdk. Install it and ensure it's importable as `fleet`."
            ) from e

        # Configure fleet client explicitly (avoid relying on env-only behavior).
        fleet.configure(api_key=self._cfg.api_key, base_url=self._cfg.base_url)

        self._session = fleet.session(
            job_id=self._cfg.job_id,
            config={
                'openhands_session_id': self._event_stream.sid,
            },
            model=self._cfg.model,
            task_key=self._cfg.task_key,
            instance_id=self._cfg.instance_id,
        )
        return self._session

    def log_llm_call(self, history: list[Any], response: Any) -> None:
        """Log a single LLM call to Fleet.

        Args:
            history: The message list sent to the LLM.
            response: The provider response object returned by OpenHands' LLM router.
        """
        if not self.enabled:
            return
        try:
            session = self._ensure_session()
            ingest = session.log(history, response)
            # First successful log creates a session id.
            if ingest and getattr(ingest, 'created_new_session', False):
                logger.info(
                    'Fleet session created',
                    extra={
                        'openhands_session_id': self._event_stream.sid,
                        'fleet_session_id': getattr(ingest, 'session_id', None),
                    },
                )
        except Exception as e:  # noqa: BLE001
            # Best-effort: don't break agent loop
            self._failed = True
            logger.warning(
                f'Fleet session export failed; disabling exporter: {type(e).__name__}: {e}',
                extra={'openhands_session_id': self._event_stream.sid},
            )

    def complete(self, verifier_execution_id: str | None = None) -> None:
        """Mark the Fleet session as completed successfully (best-effort)."""
        if not self.enabled:
            return
        try:
            session = self._ensure_session()
            session.complete(verifier_execution_id=verifier_execution_id)
        except Exception as e:  # noqa: BLE001
            self._failed = True
            logger.warning(
                f'Fleet session complete failed; disabling exporter: {type(e).__name__}: {e}',
                extra={'openhands_session_id': self._event_stream.sid},
            )

    def fail(self, verifier_execution_id: str | None = None) -> None:
        """Mark the Fleet session as failed (best-effort)."""
        if not self.enabled:
            return
        try:
            session = self._ensure_session()
            session.fail(verifier_execution_id=verifier_execution_id)
        except Exception as e:  # noqa: BLE001
            self._failed = True
            logger.warning(
                f'Fleet session fail failed; disabling exporter: {type(e).__name__}: {e}',
                extra={'openhands_session_id': self._event_stream.sid},
            )


def build_fleet_session_export_config(
    *,
    enabled: bool,
    api_key: str | None,
    base_url: str | None,
    job_id: str | None,
    task_key: str | None,
    instance_id: str | None,
    model: str | None,
) -> FleetSessionExportConfig:
    """Build config with env var fallbacks (matches fleet-sdk conventions)."""
    if not enabled:
        return FleetSessionExportConfig(enabled=False)

    return FleetSessionExportConfig(
        enabled=True,
        api_key=api_key or os.getenv('FLEET_API_KEY'),
        base_url=base_url or os.getenv('FLEET_BASE_URL'),
        job_id=job_id or os.getenv('FLEET_JOB_ID'),
        task_key=task_key or os.getenv('FLEET_TASK_KEY'),
        instance_id=instance_id or os.getenv('FLEET_INSTANCE_ID'),
        model=model or os.getenv('FLEET_MODEL'),
    )


