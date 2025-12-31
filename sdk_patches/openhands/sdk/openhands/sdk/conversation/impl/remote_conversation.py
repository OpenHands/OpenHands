import asyncio
import json
import os
import threading
import time
import uuid
from collections.abc import Mapping
from typing import SupportsIndex, overload
from urllib.parse import urlparse

import httpx
import websockets

from openhands.sdk.agent.base import AgentBase
from openhands.sdk.conversation.base import BaseConversation, ConversationStateProtocol
from openhands.sdk.conversation.conversation_stats import ConversationStats
from openhands.sdk.conversation.events_list_base import EventsListBase
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.conversation.secret_registry import SecretValue
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.conversation.types import (
    ConversationCallbackType,
    ConversationID,
    StuckDetectionThresholds,
)
from openhands.sdk.conversation.visualizer import (
    ConversationVisualizerBase,
    DefaultConversationVisualizer,
)
from openhands.sdk.event.base import Event
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.event.conversation_state import (
    FULL_STATE_KEY,
    ConversationStateUpdateEvent,
)
from openhands.sdk.event.llm_completion_log import LLMCompletionLogEvent
from openhands.sdk.hooks import (
    HookConfig,
    HookEventProcessor,
    HookEventType,
    HookManager,
)
from openhands.sdk.llm import LLM, Message, TextContent
from openhands.sdk.logger import DEBUG, get_logger
from openhands.sdk.observability.laminar import observe
from openhands.sdk.security.analyzer import SecurityAnalyzerBase
from openhands.sdk.security.confirmation_policy import (
    ConfirmationPolicyBase,
)
from openhands.sdk.workspace import LocalWorkspace, RemoteWorkspace


logger = get_logger(__name__)


def _send_request(
    client: httpx.Client,
    method: str,
    url: str,
    acceptable_status_codes: set[int] | None = None,
    **kwargs,
) -> httpx.Response:
    try:
        response = client.request(method, url, **kwargs)
        if acceptable_status_codes and response.status_code in acceptable_status_codes:
            return response
        response.raise_for_status()
        return response
    except httpx.HTTPStatusError as e:
        content = None
        try:
            content = e.response.json()
        except Exception:
            content = e.response.text
        logger.error(
            "HTTP request failed (%d %s): %s",
            e.response.status_code,
            e.response.reason_phrase,
            content,
            exc_info=True,
        )
        raise e
    except httpx.RequestError as e:
        logger.error(f"Request failed: {e}", exc_info=DEBUG)
        raise e


class WebSocketCallbackClient:
    """Minimal WS client: connects, forwards events, retries on error."""

    host: str
    conversation_id: str
    callback: ConversationCallbackType
    api_key: str | None
    _thread: threading.Thread | None
    _stop: threading.Event

    def __init__(
        self,
        host: str,
        conversation_id: str,
        callback: ConversationCallbackType,
        api_key: str | None = None,
    ):
        self.host = host
        self.conversation_id = conversation_id
        self.callback = callback
        self.api_key = api_key
        self._thread = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._thread:
            return
        self._stop.set()
        self._thread.join(timeout=5)
        self._thread = None

    def _run(self) -> None:
        try:
            asyncio.run(self._client_loop())
        except RuntimeError:
            # Fallback in case of an already running loop in rare environments
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._client_loop())
            loop.close()

    async def _client_loop(self) -> None:
        parsed = urlparse(self.host)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        base = f"{ws_scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
        ws_url = f"{base}/sockets/events/{self.conversation_id}"

        # Add API key as query parameter if provided
        if self.api_key:
            ws_url += f"?session_api_key={self.api_key}"

        delay = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(ws_url) as ws:
                    delay = 1.0
                    async for message in ws:
                        if self._stop.is_set():
                            break
                        try:
                            event = Event.model_validate(json.loads(message))
                            self.callback(event)
                        except Exception:
                            logger.exception(
                                "ws_event_processing_error", stack_info=True
                            )
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception:
                logger.debug("ws_connect_retry", exc_info=True)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)


class RemoteEventsList(EventsListBase):
    """A list-like, read-only view of remote conversation events.

    On first access it fetches existing events from the server. Afterwards,
    it relies on the WebSocket stream to incrementally append new events.
    """

    _client: httpx.Client
    _conversation_id: str
    _cached_events: list[Event]
    _cached_event_ids: set[str]
    _lock: threading.RLock

    def __init__(self, client: httpx.Client, conversation_id: str):
        self._client = client
        self._conversation_id = conversation_id
        self._cached_events: list[Event] = []
        self._cached_event_ids: set[str] = set()
        self._lock = threading.RLock()
        # Initial fetch to sync existing events
        self._do_full_sync()

    def _do_full_sync(self) -> None:
        """Perform a full sync with the remote API."""
        logger.debug(f"Performing full sync for conversation {self._conversation_id}")

        events = []
        page_id = None

        while True:
            params = {"limit": 100}
            if page_id:
                params["page_id"] = page_id

            resp = _send_request(
                self._client,
                "GET",
                f"/api/conversations/{self._conversation_id}/events/search",
                params=params,
            )
            data = resp.json()

            events.extend([Event.model_validate(item) for item in data["items"]])

            if not data.get("next_page_id"):
                break
            page_id = data["next_page_id"]

        self._cached_events = events
        self._cached_event_ids.update(e.id for e in events)
        logger.debug(f"Full sync completed, {len(events)} events cached")

    def add_event(self, event: Event) -> None:
        """Add a new event to the local cache (called by WebSocket callback)."""
        with self._lock:
            # Check if event already exists to avoid duplicates
            if event.id not in self._cached_event_ids:
                self._cached_events.append(event)
                self._cached_event_ids.add(event.id)
                logger.debug(f"Added event {event.id} to local cache")

    def append(self, event: Event) -> None:
        """Add a new event to the list (for compatibility with EventLog interface)."""
        self.add_event(event)

    def create_default_callback(self) -> ConversationCallbackType:
        """Create a default callback that adds events to this list."""

        def callback(event: Event) -> None:
            self.add_event(event)

        return callback

    def __len__(self) -> int:
        return len(self._cached_events)

    @overload
    def __getitem__(self, index: int) -> Event: ...

    @overload
    def __getitem__(self, index: slice) -> list[Event]: ...

    def __getitem__(self, index: SupportsIndex | slice) -> Event | list[Event]:
        with self._lock:
            return self._cached_events[index]

    def __iter__(self):
        with self._lock:
            return iter(self._cached_events)


class RemoteState(ConversationStateProtocol):
    """A state-like interface for accessing remote conversation state."""

    _client: httpx.Client
    _conversation_id: str
    _events: RemoteEventsList
    _cached_state: dict | None
    _lock: threading.RLock

    def __init__(self, client: httpx.Client, conversation_id: str):
        self._client = client
        self._conversation_id = conversation_id
        self._events = RemoteEventsList(client, conversation_id)

        # Cache for state information to avoid REST calls
        self._cached_state = None
        self._lock = threading.RLock()

    def _get_conversation_info(self) -> dict:
        """Fetch the latest conversation info from the remote API."""
        with self._lock:
            # Return cached state if available
            if self._cached_state is not None:
                return self._cached_state

            # Fallback to REST API if no cached state
            resp = _send_request(
                self._client, "GET", f"/api/conversations/{self._conversation_id}"
            )
            state = resp.json()
            self._cached_state = state
            return state

    def update_state_from_event(self, event: ConversationStateUpdateEvent) -> None:
        """Update cached state from a ConversationStateUpdateEvent."""
        with self._lock:
            # Handle full state snapshot
            if event.key == FULL_STATE_KEY:
                # Update cached state with the full snapshot
                if self._cached_state is None:
                    self._cached_state = {}
                self._cached_state.update(event.value)
            else:
                # Handle individual field updates
                if self._cached_state is None:
                    self._cached_state = {}
                self._cached_state[event.key] = event.value

    def create_state_update_callback(self) -> ConversationCallbackType:
        """Create a callback that updates state from ConversationStateUpdateEvent."""

        def callback(event: Event) -> None:
            if isinstance(event, ConversationStateUpdateEvent):
                self.update_state_from_event(event)

        return callback

    @property
    def events(self) -> RemoteEventsList:
        """Access to the events list."""
        return self._events

    @property
    def id(self) -> ConversationID:
        """The conversation ID."""
        return uuid.UUID(self._conversation_id)

    @property
    def execution_status(self) -> ConversationExecutionStatus:
        """The current conversation execution status."""
        info = self._get_conversation_info()
        status_str = info.get("execution_status")
        if status_str is None:
            raise RuntimeError(
                "execution_status missing in conversation info: " + str(info)
            )
        return ConversationExecutionStatus(status_str)

    @execution_status.setter
    def execution_status(self, value: ConversationExecutionStatus) -> None:
        """Set execution status is No-OP for RemoteConversation.

        # For remote conversations, execution status is managed server-side
        # This setter is provided for test compatibility but doesn't actually change remote state  # noqa: E501
        """  # noqa: E501
        raise NotImplementedError(
            f"Setting execution_status on RemoteState has no effect. "
            f"Remote execution status is managed server-side. Attempted to set: {value}"
        )

    @property
    def confirmation_policy(self) -> ConfirmationPolicyBase:
        """The confirmation policy."""
        info = self._get_conversation_info()
        policy_data = info.get("confirmation_policy")
        if policy_data is None:
            raise RuntimeError(
                "confirmation_policy missing in conversation info: " + str(info)
            )
        return ConfirmationPolicyBase.model_validate(policy_data)

    @property
    def security_analyzer(self) -> SecurityAnalyzerBase | None:
        """The security analyzer."""
        info = self._get_conversation_info()
        analyzer_data = info.get("security_analyzer")
        if analyzer_data:
            return SecurityAnalyzerBase.model_validate(analyzer_data)

        return None

    @property
    def activated_knowledge_skills(self) -> list[str]:
        """List of activated knowledge skills."""
        info = self._get_conversation_info()
        return info.get("activated_knowledge_skills", [])

    @property
    def agent(self):
        """The agent configuration (fetched from remote)."""
        info = self._get_conversation_info()
        agent_data = info.get("agent")
        if agent_data is None:
            raise RuntimeError("agent missing in conversation info: " + str(info))
        return AgentBase.model_validate(agent_data)

    @property
    def workspace(self):
        """The working directory (fetched from remote)."""
        info = self._get_conversation_info()
        workspace = info.get("workspace")
        if workspace is None:
            raise RuntimeError("workspace missing in conversation info: " + str(info))
        return workspace

    @property
    def persistence_dir(self):
        """The persistence directory (fetched from remote)."""
        info = self._get_conversation_info()
        persistence_dir = info.get("persistence_dir")
        if persistence_dir is None:
            raise RuntimeError(
                "persistence_dir missing in conversation info: " + str(info)
            )
        return persistence_dir

    @property
    def stats(self) -> ConversationStats:
        """Get conversation stats (fetched from remote)."""
        info = self._get_conversation_info()
        stats_data = info.get("stats", {})
        return ConversationStats.model_validate(stats_data)

    def model_dump(self, **_kwargs):
        """Get a dictionary representation of the remote state."""
        info = self._get_conversation_info()
        return info

    def model_dump_json(self, **kwargs):
        """Get a JSON representation of the remote state."""
        return json.dumps(self.model_dump(**kwargs))

    # Context manager methods for compatibility with ConversationState
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class RemoteConversation(BaseConversation):
    _id: uuid.UUID
    _state: "RemoteState"
    _visualizer: ConversationVisualizerBase | None
    _ws_client: "WebSocketCallbackClient | None"
    agent: AgentBase
    _callbacks: list[ConversationCallbackType]
    max_iteration_per_run: int
    workspace: RemoteWorkspace
    _client: httpx.Client
    _hook_processor: HookEventProcessor | None
    _cleanup_initiated: bool

    def __init__(
        self,
        agent: AgentBase,
        workspace: RemoteWorkspace,
        conversation_id: ConversationID | None = None,
        callbacks: list[ConversationCallbackType] | None = None,
        max_iteration_per_run: int = 500,
        stuck_detection: bool = True,
        stuck_detection_thresholds: (
            StuckDetectionThresholds | Mapping[str, int] | None
        ) = None,
        hook_config: HookConfig | None = None,
        visualizer: (
            type[ConversationVisualizerBase] | ConversationVisualizerBase | None
        ) = DefaultConversationVisualizer,
        secrets: Mapping[str, SecretValue] | None = None,
        **_: object,
    ) -> None:
        """Remote conversation proxy that talks to an agent server.

        Args:
            agent: Agent configuration (will be sent to the server)
            workspace: The working directory for agent operations and tool execution.
            conversation_id: Optional existing conversation id to attach to
            callbacks: Optional callbacks to receive events (not yet streamed)
            max_iteration_per_run: Max iterations configured on server
            stuck_detection: Whether to enable stuck detection on server
            stuck_detection_thresholds: Optional configuration for stuck detection
                      thresholds. Can be a StuckDetectionThresholds instance or
                      a dict with keys: 'action_observation', 'action_error',
                      'monologue', 'alternating_pattern'. Values are integers
                      representing the number of repetitions before triggering.
            hook_config: Optional hook configuration for session hooks
            visualizer: Visualization configuration. Can be:
                       - ConversationVisualizerBase subclass: Class to instantiate
                         (default: ConversationVisualizer)
                       - ConversationVisualizerBase instance: Use custom visualizer
                       - None: No visualization
            secrets: Optional secrets to initialize the conversation with
        """
        super().__init__()  # Initialize base class with span tracking
        self.agent = agent
        self._callbacks = callbacks or []
        self.max_iteration_per_run = max_iteration_per_run
        self.workspace = workspace
        self._client = workspace.client
        self._hook_processor = None
        self._cleanup_initiated = False

        if conversation_id is None:
            # Import here to avoid circular imports
            from openhands.sdk.tool.registry import get_tool_module_qualnames

            tool_qualnames = get_tool_module_qualnames()
            logger.debug(f"Sending tool_module_qualnames to server: {tool_qualnames}")
            payload = {
                "agent": agent.model_dump(
                    mode="json", context={"expose_secrets": True}
                ),
                "initial_message": None,
                "max_iterations": max_iteration_per_run,
                "stuck_detection": stuck_detection,
                # We need to convert RemoteWorkspace to LocalWorkspace for the server
                "workspace": LocalWorkspace(
                    working_dir=self.workspace.working_dir
                ).model_dump(),
                # Include tool module qualnames for dynamic registration on server
                "tool_module_qualnames": tool_qualnames,
            }
            if stuck_detection_thresholds is not None:
                # Convert to StuckDetectionThresholds if dict, then serialize
                if isinstance(stuck_detection_thresholds, Mapping):
                    threshold_config = StuckDetectionThresholds(
                        **stuck_detection_thresholds
                    )
                else:
                    threshold_config = stuck_detection_thresholds
                payload["stuck_detection_thresholds"] = threshold_config.model_dump()
            resp = _send_request(
                self._client, "POST", "/api/conversations", json=payload
            )
            data = resp.json()
            # Expect a ConversationInfo
            cid = data.get("id") or data.get("conversation_id")
            if not cid:
                raise RuntimeError(
                    "Invalid response from server: missing conversation id"
                )
            self._id = uuid.UUID(cid)
        else:
            # Attach to existing
            self._id = conversation_id
            # Validate it exists
            _send_request(self._client, "GET", f"/api/conversations/{self._id}")

        # Initialize the remote state
        self._state = RemoteState(self._client, str(self._id))

        # Add default callback to maintain local event state
        default_callback = self._state.events.create_default_callback()
        self._callbacks.append(default_callback)

        # Add callback to update state from websocket events
        state_update_callback = self._state.create_state_update_callback()
        self._callbacks.append(state_update_callback)

        # Add callback to handle LLM completion logs
        # Register callback if any LLM has log_completions enabled
        if any(llm.log_completions for llm in agent.get_all_llms()):
            llm_log_callback = self._create_llm_completion_log_callback()
            self._callbacks.append(llm_log_callback)

        # Handle visualization configuration
        if isinstance(visualizer, ConversationVisualizerBase):
            # Use custom visualizer instance
            self._visualizer = visualizer
            # Initialize the visualizer with conversation state
            self._visualizer.initialize(self._state)
            self._callbacks.append(self._visualizer.on_event)
        elif isinstance(visualizer, type) and issubclass(
            visualizer, ConversationVisualizerBase
        ):
            # Instantiate the visualizer class with appropriate parameters
            self._visualizer = visualizer()
            # Initialize with state
            self._visualizer.initialize(self._state)
            self._callbacks.append(self._visualizer.on_event)
        else:
            # No visualization (visualizer is None)
            self._visualizer = None

        # Compose all callbacks into a single callback
        composed_callback = BaseConversation.compose_callbacks(self._callbacks)

        # Initialize WebSocket client for callbacks
        self._ws_client = WebSocketCallbackClient(
            host=self.workspace.host,
            conversation_id=str(self._id),
            callback=composed_callback,
            api_key=self.workspace.api_key,
        )
        self._ws_client.start()

        # Initialize secrets if provided
        if secrets:
            # Convert dict[str, str] to dict[str, SecretValue]
            secret_values: dict[str, SecretValue] = {k: v for k, v in secrets.items()}
            self.update_secrets(secret_values)

        self._start_observability_span(str(self._id))
        if hook_config is not None:
            unsupported = (
                HookEventType.PRE_TOOL_USE,
                HookEventType.POST_TOOL_USE,
                HookEventType.USER_PROMPT_SUBMIT,
                HookEventType.STOP,
            )
            if any(hook_config.has_hooks_for_event(t) for t in unsupported):
                logger.warning(
                    "RemoteConversation only supports SessionStart/SessionEnd hooks; "
                    "other hook types will not be enforced."
                )
            hook_manager = HookManager(
                config=hook_config,
                working_dir=os.getcwd(),
                session_id=str(self._id),
            )
            self._hook_processor = HookEventProcessor(hook_manager=hook_manager)
            self._hook_processor.run_session_start()

    def _create_llm_completion_log_callback(self) -> ConversationCallbackType:
        """Create a callback that writes LLM completion logs to client filesystem."""

        def callback(event: Event) -> None:
            if not isinstance(event, LLMCompletionLogEvent):
                return

            # Find the LLM with matching usage_id
            target_llm = None
            for llm in self.agent.get_all_llms():
                if llm.usage_id == event.usage_id:
                    target_llm = llm
                    break

            if not target_llm or not target_llm.log_completions:
                logger.debug(
                    f"No LLM with log_completions enabled found "
                    f"for usage_id={event.usage_id}"
                )
                return

            try:
                log_dir = target_llm.log_completions_folder
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, event.filename)
                with open(log_path, "w") as f:
                    f.write(event.log_data)
                logger.debug(f"Wrote LLM completion log to {log_path}")
            except Exception as e:
                logger.warning(f"Failed to write LLM completion log: {e}")

        return callback

    @property
    def id(self) -> ConversationID:
        return self._id

    @property
    def state(self) -> RemoteState:
        """Access to remote conversation state."""
        return self._state

    @property
    def conversation_stats(self):
        return self._state.stats

    @property
    def stuck_detector(self):
        """Stuck detector for compatibility.
        Not implemented for remote conversations."""
        raise NotImplementedError(
            "For remote conversations, stuck detection is not available"
            " since it would be handled server-side."
        )

    @observe(name="conversation.send_message")
    def send_message(self, message: str | Message, sender: str | None = None) -> None:
        if isinstance(message, str):
            message = Message(role="user", content=[TextContent(text=message)])
        assert message.role == "user", (
            "Only user messages are allowed to be sent to the agent."
        )
        payload = {
            "role": message.role,
            "content": [c.model_dump() for c in message.content],
            "run": False,  # Mirror local semantics; explicit run() must be called
        }
        if sender is not None:
            payload["sender"] = sender
        _send_request(
            self._client, "POST", f"/api/conversations/{self._id}/events", json=payload
        )

    @observe(name="conversation.run")
    def run(
        self,
        blocking: bool = True,
        poll_interval: float = 1.0,
        timeout: float = 3600.0,
    ) -> None:
        """Trigger a run on the server.

        Args:
            blocking: If True (default), wait for the run to complete by polling
                the server. If False, return immediately after triggering the run.
            poll_interval: Time in seconds between status polls (only used when
                blocking=True). Default is 1.0 second.
            timeout: Maximum time in seconds to wait for the run to complete
                (only used when blocking=True). Default is 3600 seconds.

        Raises:
            ConversationRunError: If the run fails or times out.
        """
        # Trigger a run on the server using the dedicated run endpoint.
        # Let the server tell us if it's already running (409), avoiding an extra GET.
        try:
            resp = _send_request(
                self._client,
                "POST",
                f"/api/conversations/{self._id}/run",
                acceptable_status_codes={200, 201, 204, 409},
                timeout=30,  # Short timeout for trigger request
            )
        except Exception as e:  # httpx errors already logged by _send_request
            # Surface conversation id to help resuming
            raise ConversationRunError(self._id, e) from e

        if resp.status_code == 409:
            logger.info("Conversation is already running; skipping run trigger")
            if blocking:
                # Still wait for the existing run to complete
                self._wait_for_run_completion(poll_interval, timeout)
            return

        logger.info(f"run() triggered successfully: {resp}")

        if blocking:
            self._wait_for_run_completion(poll_interval, timeout)

    def _wait_for_run_completion(
        self,
        poll_interval: float = 1.0,
        timeout: float = 1800.0,
    ) -> None:
        """Poll the server until the conversation is no longer running.

        Args:
            poll_interval: Time in seconds between status polls.
            timeout: Maximum time in seconds to wait.

        Raises:
            ConversationRunError: If the wait times out.
        """
        start_time = time.monotonic()

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > timeout:
                raise ConversationRunError(
                    self._id,
                    TimeoutError(
                        f"Run timed out after {timeout} seconds. "
                        "The conversation may still be running on the server."
                    ),
                )

            try:
                resp = _send_request(
                    self._client,
                    "GET",
                    f"/api/conversations/{self._id}",
                    timeout=30,
                )
                info = resp.json()
                status = info.get("execution_status")

                if status != ConversationExecutionStatus.RUNNING.value:
                    if status == ConversationExecutionStatus.ERROR.value:
                        detail = self._get_last_error_detail()
                        raise ConversationRunError(
                            self._id,
                            RuntimeError(
                                detail or "Remote conversation ended with error"
                            ),
                        )
                    if status == ConversationExecutionStatus.STUCK.value:
                        raise ConversationRunError(
                            self._id,
                            RuntimeError("Remote conversation got stuck"),
                        )
                    logger.info(
                        f"Run completed with status: {status} (elapsed: {elapsed:.1f}s)"
                    )
                    return

            except Exception as e:
                # Log but continue polling - transient network errors shouldn't
                # stop us from waiting for the run to complete
                logger.warning(f"Error polling status (will retry): {e}")

            time.sleep(poll_interval)

    def _get_last_error_detail(self) -> str | None:
        """Return the most recent ConversationErrorEvent detail, if available."""
        try:
            events = self._state.events
            for idx in range(len(events) - 1, -1, -1):
                event = events[idx]
                if isinstance(event, ConversationErrorEvent):
                    detail = event.detail.strip()
                    code = event.code.strip()
                    if detail and code:
                        return f"{code}: {detail}"
                    return detail or code or None
        except Exception as exc:
            logger.debug("Failed to read conversation error detail: %s", exc)
        return None

    def set_confirmation_policy(self, policy: ConfirmationPolicyBase) -> None:
        payload = {"policy": policy.model_dump()}
        _send_request(
            self._client,
            "POST",
            f"/api/conversations/{self._id}/confirmation_policy",
            json=payload,
        )

    def set_security_analyzer(self, analyzer: SecurityAnalyzerBase | None) -> None:
        """Set the security analyzer for the remote conversation."""
        payload = {"security_analyzer": analyzer.model_dump() if analyzer else analyzer}
        _send_request(
            self._client,
            "POST",
            f"/api/conversations/{self._id}/security_analyzer",
            json=payload,
        )

    def reject_pending_actions(self, reason: str = "User rejected the action") -> None:
        # Equivalent to rejecting confirmation: pause
        _send_request(
            self._client,
            "POST",
            f"/api/conversations/{self._id}/events/respond_to_confirmation",
            json={"accept": False, "reason": reason},
        )

    def pause(self) -> None:
        _send_request(self._client, "POST", f"/api/conversations/{self._id}/pause")

    def update_secrets(self, secrets: Mapping[str, SecretValue]) -> None:
        # Convert SecretValue to strings for JSON serialization
        # SecretValue can be str or callable, we need to handle both
        serializable_secrets = {}
        for key, value in secrets.items():
            if callable(value):
                # If it's a callable, call it to get the actual secret
                serializable_secrets[key] = value()
            else:
                # If it's already a string, use it directly
                serializable_secrets[key] = value

        payload = {"secrets": serializable_secrets}
        _send_request(
            self._client, "POST", f"/api/conversations/{self._id}/secrets", json=payload
        )

    def ask_agent(self, question: str) -> str:
        """Ask the agent a simple, stateless question and get a direct LLM response.

        This bypasses the normal conversation flow and does **not** modify, persist,
        or become part of the conversation state. The request is not remembered by
        the main agent, no events are recorded, and execution status is untouched.
        It is also thread-safe and may be called while `conversation.run()` is
        executing in another thread.

        Args:
            question: A simple string question to ask the agent

        Returns:
            A string response from the agent
        """
        # For remote conversations, delegate to the server endpoint
        payload = {"question": question}

        resp = _send_request(
            self._client,
            "POST",
            f"/api/conversations/{self._id}/ask_agent",
            json=payload,
        )
        data = resp.json()
        return data["response"]

    @observe(name="conversation.generate_title", ignore_inputs=["llm"])
    def generate_title(self, llm: LLM | None = None, max_length: int = 50) -> str:
        """Generate a title for the conversation based on the first user message.

        Args:
            llm: Optional LLM to use for title generation. If provided, its usage_id
                 will be sent to the server. If not provided, uses the agent's LLM.
            max_length: Maximum length of the generated title.

        Returns:
            A generated title for the conversation.
        """
        # For remote conversations, delegate to the server endpoint
        payload = {
            "max_length": max_length,
            "llm": llm.model_dump(mode="json", context={"expose_secrets": True})
            if llm
            else None,
        }

        resp = _send_request(
            self._client,
            "POST",
            f"/api/conversations/{self._id}/generate_title",
            json=payload,
        )
        data = resp.json()
        return data["title"]

    def condense(self) -> None:
        """Force condensation of the conversation history.

        This method sends a condensation request to the remote agent server.
        The server will use the existing condensation request pattern to trigger
        condensation if a condenser is configured and handles condensation requests.

        The condensation will be applied on the server side and will modify the
        conversation state by adding a condensation event to the history.

        Raises:
            HTTPError: If the server returns an error (e.g., no condenser configured).
        """
        _send_request(self._client, "POST", f"/api/conversations/{self._id}/condense")

    def close(self) -> None:
        """Close the conversation and clean up resources.

        Note: We don't close self._client here because it's shared with the workspace.
        The workspace owns the client and will close it during its own cleanup.
        Closing it here would prevent the workspace from making cleanup API calls.
        """
        if self._cleanup_initiated:
            return
        self._cleanup_initiated = True
        if self._hook_processor is not None:
            self._hook_processor.run_session_end()
        try:
            # Stop WebSocket client if it exists
            if self._ws_client:
                self._ws_client.stop()
                self._ws_client = None
        except Exception:
            pass

        self._end_observability_span()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
