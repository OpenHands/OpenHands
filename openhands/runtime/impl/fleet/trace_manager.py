import os
import requests
from typing import Any, Dict
from openhands.core.logger import openhands_logger as logger

class FleetTraceManager:
    """
    Manages tracing for FleetRuntime operations.
    Can send traces to an external API if configured.
    """
    def __init__(self, api_url: str | None = None, api_key: str | None = None):
        self.api_url = api_url or os.getenv("FLEET_TRACE_API_URL")
        self.api_key = api_key or os.getenv("FLEET_TRACE_API_KEY")
        self.enabled = bool(self.api_url)
        if self.enabled:
            logger.info(f"Fleet tracing enabled. Sending traces to {self.api_url}")
        else:
            logger.debug("Fleet tracing disabled (no API URL provided)")

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log an event to the Fleet Trace API.

        Args:
            event_type: The type of event (e.g., 'action_start', 'action_complete', 'error')
            data: Dictionary containing event data
        """
        if not self.enabled:
            return

        payload = {
            "event_type": event_type,
            "data": data,
            "timestamp": "TODO: isoformat_timestamp" # Should be added
        }

        try:
            # TODO: Make this async to avoid blocking runtime
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.post(self.api_url, json=payload, headers=headers, timeout=5)
            if response.status_code >= 400:
                logger.warning(f"Failed to send trace to Fleet API: {response.status_code} {response.text}")
        except Exception as e:
            logger.warning(f"Error sending trace to Fleet API: {e}")

    def trace_action(self, action_name: str, args: Dict[str, Any]):
        self.log_event("action_start", {"action": action_name, "args": args})

    def trace_observation(self, action_name: str, result: Any):
        # Convert result to serializable format if needed
        self.log_event("action_complete", {"action": action_name, "result": str(result)})

    def trace_error(self, action_name: str, error: str):
        self.log_event("action_error", {"action": action_name, "error": error})

