from __future__ import annotations

import json
from collections import deque
from typing import TYPE_CHECKING, Any

from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import AgentFinishAction
from openhands.events.action.action import Action
from openhands.events.action.mcp import MCPAction
from openhands.events.event import Event
from openhands.events.observation.mcp import MCPObservation
from openhands.llm.llm_registry import LLMRegistry

from openhands.llm.gemini_native_conversions import (
    openai_tool_to_gemini_function_declaration,
)

if TYPE_CHECKING:
    from openhands.core.config import AgentConfig


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

        self._types = types

        # Determine API key and model name
        api_key = None
        try:
            # Reuse OpenHands llm config for key/model selection
            if getattr(self.llm.config, 'api_key', None):
                api_key = self.llm.config.api_key.get_secret_value()
        except Exception:
            api_key = None

        self._client = genai.Client(api_key=api_key)

        model = getattr(self.llm.config, 'model', None) or ''
        # Strip common provider prefixes (e.g., "google/gemini-2.5-pro")
        if '/' in model:
            model = model.split('/')[-1]
        self._model = model or 'gemini-2.5-pro'

        # Gemini chat history: list[types.Content]
        self._history: list[Any] = []

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

    def _get_tools_as_function_declarations(self) -> list[Any]:
        """Convert MCP tools (OpenAI-style tool dicts) to Gemini FunctionDeclarations."""
        decls: list[Any] = []

        # Only expose MCP tools to keep this minimal and environment-driven.
        for tool in self.mcp_tools.values():
            tool_dict = dict(tool) if not isinstance(tool, dict) else tool
            decl_dict = openai_tool_to_gemini_function_declaration(tool_dict)
            decls.append(self._types.FunctionDeclaration(**decl_dict))

        return decls

    def _ingest_new_tool_observations(self, history: list[Event]) -> None:
        """After a tool action executes, add its function_response to Gemini history."""
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

            # Build Gemini function_response part, optionally including screenshots as inline_data
            if getattr(ev, 'images', None):
                parts = [
                    self._types.FunctionResponsePart(
                        inline_data=self._types.FunctionResponseBlob(
                            mime_type=img.mime_type,
                            data=img.data,
                        )
                    )
                    for img in ev.images
                ]
                fr = self._types.FunctionResponse(
                    name=tool_name,
                    response={'status': 'success' if ok else 'error'},
                    parts=parts,
                )
            else:
                fr = self._types.FunctionResponse(
                    name=tool_name,
                    response={'status': 'success' if ok else 'error'},
                )

            self._batch_response_parts.append(self._types.Part(function_response=fr))
            self._batch_remaining -= 1

            if self._batch_remaining == 0 and self._batch_response_parts:
                # Gemini expects function responses as role="model" parts (mirrors fleet-sdk gemini_cua)
                self._history.append(
                    self._types.Content(role='model', parts=self._batch_response_parts)
                )
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
            system_prompt = self._get_system_prompt()
            user_prompt = self._build_initial_user_prompt(state)
            self._history.append(
                self._types.Content(
                    role='user',
                    parts=[self._types.Part(text=f'{system_prompt}\n\nTask:\n{user_prompt}')],
                )
            )

        # Build tools
        decls = self._get_tools_as_function_declarations()
        config = self._types.GenerateContentConfig(
            tools=[self._types.Tool(function_declarations=decls)],
            max_output_tokens=4096,
            thinking_config=self._types.ThinkingConfig(include_thoughts=True),
        )

        # Call Gemini
        response = self._client.models.generate_content(
            model=self._model,
            contents=self._history,
            config=config,
        )

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


