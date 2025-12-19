"""Fleet session logging exporter.

This integrates OpenHands with Fleet's session logging API (via fleet-sdk).

Key design points:
- Fleet session logging is a *client-side* concern: it needs access to the LLM
  prompt history + raw model response.
- Therefore, the primary hook is inside the agent step() where `messages` and
  `response` exist.
- We also optionally subscribe to the OpenHands EventStream to mark sessions as
  completed when the agent finishes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from openhands.core.logger import openhands_logger as logger
from openhands.events.event import Event, EventSource
from openhands.events.stream import EventStream, EventStreamSubscriber


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
        self._subscriber_id = EventStreamSubscriber.MAIN
        self._callback_id = f'fleet_session_exporter:{event_stream.sid}'
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
        # Subscribe to session lifecycle events (best-effort)
        self._event_stream.subscribe(self._subscriber_id, self._on_event, self._callback_id)
        logger.info(
            'Fleet session exporter enabled',
            extra={
                'session_id': self._event_stream.sid,
                'fleet_job_id': self._cfg.job_id,
                'fleet_task_key': self._cfg.task_key,
            },
        )

    def stop(self) -> None:
        try:
            self._event_stream.unsubscribe(self._subscriber_id, self._callback_id)
        except Exception:
            pass

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

    def _on_event(self, event: Event) -> None:
        """Watch for completion signals and mark Fleet session complete."""
        if not self.enabled or self._session is None:
            return
        try:
            # Avoid circular imports
            from openhands.events.action.agent import AgentFinishAction

            if isinstance(event, AgentFinishAction) and event.source == EventSource.AGENT:
                self._session.complete()
        except Exception:
            # best-effort
            return


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


