from __future__ import annotations

import json
import os
import subprocess
from collections import deque
from typing import TYPE_CHECKING, Any

from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import AgentFinishAction
from openhands.events.action.action import Action
from openhands.events.action.mcp import MCPAction
from openhands.events.action.message import SystemMessageAction
from openhands.events.event import Event
from openhands.events.event import EventSource
from openhands.events.observation.mcp import MCPObservation
from openhands.llm.llm_registry import LLMRegistry

from openhands.llm.gemini_native_conversions import (
    openai_tool_to_gemini_function_declaration,
)

if TYPE_CHECKING:
    from openhands.core.config import AgentConfig


# OAuth configuration (matching fleet-sdk gemini_cua reference)
GOOG_PROJECT = os.environ.get("GOOG_PROJECT", "gemini-agents-area")
USE_OAUTH = os.environ.get("USE_OAUTH", "false").lower() in ("true", "1", "yes")

def _get_oauth_token() -> str:
    """Get OAuth token from gcloud."""
    import shutil

    # Find gcloud binary - check PATH first, then common locations
    gcloud_path = shutil.which("gcloud")
    if not gcloud_path:
        common_paths = [
            os.path.expanduser("~/google-cloud-sdk/bin/gcloud"),
            "/usr/local/bin/gcloud",
            "/opt/homebrew/bin/gcloud",
        ]
        for path in common_paths:
            if os.path.isfile(path):
                gcloud_path = path
                break

    if not gcloud_path:
        raise FileNotFoundError("gcloud not found in PATH or common locations")

    ret = subprocess.run(
        [gcloud_path, "auth", "application-default", "print-access-token"],
        capture_output=True,
        check=True,
    )
    return ret.stdout.decode().strip()


class GeminiNativeCodeActAgent(Agent):
    """Gemini-native agent that runs the tool loop with google-genai directly.

    This agent is intentionally minimal and focused on the native Gemini protocol:
    - Tools are converted to Gemini FunctionDeclarations
    - Model outputs function_call parts
    - Tool results are returned as function_response parts (with screenshots as inline_data)

    Tool execution is still done through the OpenHands Runtime via MCPAction.
    """

    VERSION = '0.1'

    def __init__(self, config: 'AgentConfig', llm_registry: LLMRegistry) -> None:
        super().__init__(config, llm_registry)

        # Internal queues
        self._pending_actions: deque[Action] = deque()
        self._pending_fc_names: deque[str] = deque()
        self._batch_remaining: int = 0
        self._batch_response_parts: list[Any] = []

        # Track which events we've already incorporated into Gemini history
        self._last_history_len: int = 0

        # Gemini SDK objects (optional dependency)
        try:
            from google import genai  # type: ignore[import-not-found]
            from google.genai import types  # type: ignore[import-not-found]
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                "GeminiNativeCodeActAgent requires google-genai. "
                "Install with: pip install google-genai"
            ) from e

        self._genai = genai
        self._types = types

        # Determine API key and model name
        api_key = None
        try:
            # Reuse OpenHands llm config for key/model selection
            if getattr(self.llm.config, 'api_key', None):
                api_key = self.llm.config.api_key.get_secret_value()
        except Exception:
            api_key = None

        # Build client with OAuth or API key (matching fleet-sdk gemini_cua reference)
        # OAuth must be opt-in (USE_OAUTH=true), otherwise we stay in API-key mode.
        self._api_key = api_key
        self._http_opts: Any | None = None
        self._use_multimodal_function_response = False
        if USE_OAUTH:
            try:
                oauth_token = _get_oauth_token()
                self._http_opts = types.HttpOptions(
                    headers={
                        "Authorization": f"Bearer {oauth_token}",
                        "X-Goog-User-Project": GOOG_PROJECT,
                    },
                    api_version="v1alpha",
                )
                logger.info(f"Using OAuth (project: {GOOG_PROJECT})")
                self._use_multimodal_function_response = True
            except Exception as e:
                logger.warning(f"OAuth token fetch failed, falling back to API key: {e}")
                self._http_opts = None
                self._use_multimodal_function_response = False

        logger.info(f"{self._use_multimodal_function_response=}")

        self._client = self._genai.Client(api_key=self._api_key, http_options=self._http_opts)

        model = getattr(self.llm.config, 'model', None) or ''
        # Strip common provider prefixes (e.g., "google/gemini-2.5-pro")
        if '/' in model:
            model = model.split('/')[-1]
        self._model = model or 'gemini-2.5-pro'

        # Gemini chat history: list[types.Content]
        self._history: list[Any] = []

        # Keep system prompt separate (reference agent passes it via system_instruction)
        self._system_prompt: str = self._get_system_prompt()

        self.reset()

    def reset(self) -> None:
        super().reset()
        self._pending_actions.clear()
        self._pending_fc_names.clear()
        self._batch_remaining = 0
        self._batch_response_parts = []
        self._last_history_len = 0
        self._history = []

    def _get_system_prompt(self) -> str:
        return (
            "You control a remote environment via tools.\n\n"
            "Rules:\n"
            "- If you call a tool, do not provide a final answer in the same turn.\n"
            "- When you are fully done, output only: DONE: <short summary>\n"
        )

    def get_system_message(self) -> SystemMessageAction | None:
        """Gemini-native system prompt (matches Fleet SDK style).

        OpenHands' default Agent.get_system_message() relies on PromptManager templates.
        For Gemini-native we keep this minimal and explicit: just a plain system prompt
        plus the current tool list.
        """
        try:
            msg = SystemMessageAction(
                content=self._get_system_prompt(),
                tools=getattr(self, 'tools', None),
                agent_class=self.name,
            )
            msg._source = EventSource.AGENT  # type: ignore[attr-defined]
            return msg
        except Exception as e:  # noqa: BLE001
            logger.warning(f'[{self.name}] Failed to generate system message: {e}')
            return None

    def _get_tools_as_function_declarations(self) -> list[Any]:
        """Convert MCP tools (OpenAI-style tool dicts) to Gemini FunctionDeclarations."""
        decls: list[Any] = []

        # Only expose MCP tools to keep this minimal and environment-driven.
        for tool in self.mcp_tools.values():
            tool_dict = dict(tool) if not isinstance(tool, dict) else tool
            decl_dict = openai_tool_to_gemini_function_declaration(tool_dict)
            decls.append(self._types.FunctionDeclaration(**decl_dict))

        return decls

    def _sanitize_history_for_gemini(self, contents: list[Any]) -> list[Any]:
        """Strip image blobs from tool responses in-place (for Gemini API-key compatibility).

        Why:
        - Fleet UI wants screenshots embedded in function responses so it can render them.
        - Some Gemini endpoints (API-key mode) reject multimodal function responses.

        Approach:
        - Keep `self._history` as the *full* history (used for Fleet logging).
        - When calling Gemini with API-key mode, we deep-copy the history, then
          remove any `FunctionResponse.parts` from the copied objects (so the request
          is text-only, but Fleet still sees screenshots in the logged history).
        """
        for c in contents:
            parts = getattr(c, 'parts', None)
            if not isinstance(parts, list):
                continue
            for p in parts:
                fr = getattr(p, 'function_response', None)
                if fr is None:
                    continue
                # Remove image attachments if present.
                # (Some google-genai types may not allow assignment; fail soft.)
                try:
                    if getattr(fr, 'parts', None):
                        fr.parts = None
                except Exception:
                    pass

        return contents

    def _build_request_history_for_gemini(self) -> list[Any]:
        """Build the message list to send to Gemini.

        Important:
        - `self._history` may include base64 screenshots for Fleet UI logging.
        - We must NOT send those screenshots to Gemini in API-key mode (it errors),
          and we should avoid deepcopying them (very large / slow).

        So in API-key mode we construct a lightweight view of history where any
        `FunctionResponse.parts` are removed, without copying image blobs.
        """
        if self._use_multimodal_function_response:
            return self._history

        sanitized: list[Any] = []
        for c in self._history:
            role = getattr(c, 'role', None)
            parts = getattr(c, 'parts', None)
            if role is None or not isinstance(parts, list):
                sanitized.append(c)
                continue

            new_parts: list[Any] = []
            for p in parts:
                fr = getattr(p, 'function_response', None)
                if fr is not None:
                    name = getattr(fr, 'name', None)
                    response = getattr(fr, 'response', None)
                    # Rebuild without `.parts` (drops images)
                    new_fr = self._types.FunctionResponse(name=name, response=response)
                    new_parts.append(self._types.Part(function_response=new_fr))
                else:
                    new_parts.append(p)

            sanitized.append(self._types.Content(role=role, parts=new_parts))

        return sanitized

    def _ingest_new_tool_observations(self, history: list[Event]) -> None:
        """After a tool action executes, add its function_response to Gemini history.

        OAuth + v1alpha API supports multimodal function responses (images in FunctionResponse.parts).
        Some Gemini endpoints (API-key mode) reject multimodal function responses.

        To support BOTH:
        - We always store screenshots in `self._history` (so Fleet logging/UI can show them).
        - When sending a request to Gemini in API-key mode, we pass a sanitized copy of
          history that strips `FunctionResponse.parts` (see `_sanitize_history_for_gemini`).

        Brief Gemini protocol explainer:
        - Gemini conversations are `Content(role=..., parts=[...])`.
        - A `Part` can be text, a `function_call`, or a `function_response`.
        - When Gemini returns N tool calls in one turn, the client must reply with N
          function responses (in the same order) before asking the model again.
        - `FunctionResponse.response` is a JSON-ish dict (we include tool text in
          `response["content"]` so API-key mode still has the page state).
        - `FunctionResponse.parts` is for rich attachments (e.g., screenshots as
          inline_data blobs) and only works for multimodal-enabled endpoints.
        """
        if len(history) <= self._last_history_len:
            return

        new_events = history[self._last_history_len :]
        self._last_history_len = len(history)

        for ev in new_events:
            if not isinstance(ev, MCPObservation):
                continue

            if self._batch_remaining <= 0 or not self._pending_fc_names:
                # Ignore observations that aren't part of a pending batch
                continue

            tool_name = self._pending_fc_names.popleft()
            ok = not getattr(ev, 'is_error', False)

            # Get text content from the observation (what the model should see)
            text_content = getattr(ev, 'content', '') or ''

            # Extract image data if available.
            # NOTE: we always attach images to stored history for Fleet UI, even if
            # the current Gemini endpoint can't accept multimodal function responses.
            img_data: str | None = None
            img_mime_type: str = 'image/png'
            images = getattr(ev, 'images', None)
            if images:
                for img in images:
                    if isinstance(img, dict):
                        img_mime_type = img.get('mime_type') or img.get('mimeType') or 'image/png'
                        img_data = img.get('data') or ''
                    else:
                        img_mime_type = getattr(img, 'mime_type', 'image/png')
                        img_data = getattr(img, 'data', '')
                    if img_data:
                        break

            # Build response payload with text content
            response_payload: dict[str, Any] = {'status': 'success' if ok else 'error'}
            if text_content:
                response_payload['content'] = text_content

            # Build function response
            if img_data:
                # Store multimodal tool response (for Fleet UI). Gemini requests may strip this.
                fr_part = self._types.Part(
                    function_response=self._types.FunctionResponse(
                        name=tool_name,
                        response=response_payload,
                        parts=[
                            self._types.FunctionResponsePart(
                                inline_data=self._types.FunctionResponseBlob(
                                    mime_type=img_mime_type,
                                    data=img_data,  # Base64 string
                                )
                            )
                        ],
                    )
                )
            else:
                # API key or no image: text-only function response
                fr_part = self._types.Part(
                    function_response=self._types.FunctionResponse(
                        name=tool_name,
                        response=response_payload,
                    )
                )

            self._batch_response_parts.append(fr_part)
            self._batch_remaining -= 1

            if self._batch_remaining == 0 and self._batch_response_parts:
                self._history.append(self._types.Content(role='model', parts=self._batch_response_parts))
                self._batch_response_parts = []

    def _build_initial_user_prompt(self, state: State) -> str:
        goal, _ = state.get_current_user_intent()
        if goal:
            return goal
        return state.inputs.get('task', '')

    def step(self, state: State) -> Action:
        # Incorporate any new tool results into Gemini history
        self._ingest_new_tool_observations(state.history)

        # If we're still executing tool calls from a previous model response, continue.
        if self._pending_actions:
            action = self._pending_actions.popleft()
            if isinstance(action, MCPAction):
                logger.info(f'GeminiNativeCodeActAgent executing tool: {action.name}')
            return action

        # If history is empty, seed it with system + initial user prompt
        if not self._history:
            user_prompt = self._build_initial_user_prompt(state)
            # Print the initial task prompt once for debugging/repro.
            logger.info(f'[GeminiNativeCodeActAgent] initial_user_prompt:\n{user_prompt}')
            self._history.append(
                self._types.Content(
                    role='user',
                    parts=[self._types.Part(text=f'###User instruction: {user_prompt}')],
                )
            )

        # Build tools
        decls = self._get_tools_as_function_declarations()
        config = self._types.GenerateContentConfig(
            tools=[self._types.Tool(function_declarations=decls)],
            max_output_tokens=4096,
            system_instruction=self._system_prompt,
            thinking_config=self._types.ThinkingConfig(include_thoughts=True),
        )

        # DEBUG: Log what we are actually sending to Gemini
        tool_names = [d.name for d in decls] if decls else []
        logger.debug(f'Sending {len(tool_names)} tools to Gemini: {tool_names[:10]}...')

        # Call Gemini
        # IMPORTANT: In API-key mode some Gemini endpoints reject multimodal function
        # responses. We keep full history (with screenshots) for Fleet logging, but
        # send a sanitized copy to Gemini when multimodal is not enabled.
        request_history = self._build_request_history_for_gemini()

        # Call Gemini (retry on transient MALFORMED_FUNCTION_CALL responses)
        response = None
        for attempt in range(3):
            response = self._client.models.generate_content(
                model=self._model,
                contents=request_history,
                config=config,
            )
            try:
                candidate0 = response.candidates[0] if getattr(response, 'candidates', None) else None
                finish_reason = getattr(candidate0, 'finish_reason', None)
                content0 = getattr(candidate0, 'content', None) if candidate0 else None
                if content0 is None and str(finish_reason) == 'MALFORMED_FUNCTION_CALL' and attempt < 2:
                    logger.warning(
                        f'Gemini returned MALFORMED_FUNCTION_CALL; retrying attempt={attempt+2}/3'
                    )
                    continue
            except Exception:
                # Best-effort: if inspection fails, just proceed with the response
                pass
            break


        # Optional: Fleet session logging (best-effort)
        try:
            exporter = getattr(self, 'fleet_session_exporter', None)
            if exporter is not None and getattr(exporter, 'enabled', False):
                exporter.log_llm_call(history=list(self._history), response=response)
        except Exception as e:
            logger.debug(f'Fleet session log skipped (Gemini native): {e}')

        if not getattr(response, 'candidates', None):
            return AgentFinishAction(final_thought='DONE: No candidates returned')

        candidate = response.candidates[0]
        content = getattr(candidate, 'content', None)
        parts = getattr(content, 'parts', None) or []

        # Extract tool calls and plain text
        function_calls = [p.function_call for p in parts if getattr(p, 'function_call', None)]
        text_parts = [p.text for p in parts if getattr(p, 'text', None)]
        thought_parts = [p.thought for p in parts if getattr(p, 'thought', None)]
        thought_text = '\n'.join([t for t in thought_parts if isinstance(t, str) and t.strip()]).strip()

        # If the model returns text without tool calls, finish.
        if text_parts and not function_calls:
            final_text = ' '.join(text_parts).strip()
            if final_text.upper().startswith('DONE:'):
                return AgentFinishAction(final_thought=final_text)
            return AgentFinishAction(final_thought=f'DONE: {final_text}')

        # If there are tool calls, append model output to history and schedule actions.
        if function_calls:
            # Add model content to history so Gemini "remembers" the tool calls it made.
            self._history.append(content)

            self._batch_remaining = len(function_calls)
            self._batch_response_parts = []
            self._pending_fc_names = deque()

            for fc in function_calls:
                name = fc.name
                args = dict(fc.args) if fc.args else {}

                self._pending_fc_names.append(name)
                # Thinking blog
                self._pending_actions.append(MCPAction(name=name, arguments=args, thought=thought_text))

            action = self._pending_actions.popleft()
            return action

        # Fallback
        logger.warning(f'Unexpected Gemini response: {json.dumps(_safe_to_json(response) , default=str)[:1000]}')
        return AgentFinishAction(final_thought='DONE: Unexpected model response')


def _safe_to_json(obj: Any) -> Any:
    """Best-effort JSON serialization helper for debug logs."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_to_json(v) for v in obj]
    # Try pydantic-like / dataclass-like
    if hasattr(obj, 'model_dump'):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, '__dict__'):
        try:
            return {k: _safe_to_json(v) for k, v in obj.__dict__.items()}
        except Exception:
            pass
    return str(obj)


