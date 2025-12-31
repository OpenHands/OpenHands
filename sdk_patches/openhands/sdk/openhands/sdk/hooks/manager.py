"""Hook manager - orchestrates hook execution within conversations."""

from typing import Any

from openhands.sdk.hooks.config import HookConfig
from openhands.sdk.hooks.executor import HookExecutor, HookResult
from openhands.sdk.hooks.types import HookEvent, HookEventType


class HookManager:
    """Manages hook execution for a conversation."""

    def __init__(
        self,
        config: HookConfig | None = None,
        working_dir: str | None = None,
        session_id: str | None = None,
    ):
        self.config = config or HookConfig.load(working_dir=working_dir)
        self.executor = HookExecutor(working_dir=working_dir)
        self.session_id = session_id
        self.working_dir = working_dir

    def _create_event(
        self,
        event_type: HookEventType,
        tool_name: str | None = None,
        tool_input: dict[str, Any] | None = None,
        tool_response: dict[str, Any] | None = None,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HookEvent:
        """Create a hook event with common fields populated."""
        return HookEvent(
            event_type=event_type,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_response=tool_response,
            message=message,
            session_id=self.session_id,
            working_dir=self.working_dir,
            metadata=metadata or {},
        )

    def run_pre_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> tuple[bool, list[HookResult]]:
        """Run PreToolUse hooks. Returns (should_continue, results)."""
        hooks = self.config.get_hooks_for_event(HookEventType.PRE_TOOL_USE, tool_name)
        if not hooks:
            return True, []

        event = self._create_event(
            HookEventType.PRE_TOOL_USE,
            tool_name=tool_name,
            tool_input=tool_input,
        )

        results = self.executor.execute_all(hooks, event, stop_on_block=True)

        # Check if any hook blocked the operation
        should_continue = all(r.should_continue for r in results)

        return should_continue, results

    def run_post_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_response: dict[str, Any],
    ) -> list[HookResult]:
        """Run PostToolUse hooks after a tool completes."""
        hooks = self.config.get_hooks_for_event(HookEventType.POST_TOOL_USE, tool_name)
        if not hooks:
            return []

        event = self._create_event(
            HookEventType.POST_TOOL_USE,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_response=tool_response,
        )

        # PostToolUse hooks don't block - they just run
        return self.executor.execute_all(hooks, event, stop_on_block=False)

    def run_user_prompt_submit(
        self,
        message: str,
    ) -> tuple[bool, str | None, list[HookResult]]:
        """Run UserPromptSubmit hooks."""
        hooks = self.config.get_hooks_for_event(HookEventType.USER_PROMPT_SUBMIT)
        if not hooks:
            return True, None, []

        event = self._create_event(
            HookEventType.USER_PROMPT_SUBMIT,
            message=message,
        )

        results = self.executor.execute_all(hooks, event, stop_on_block=True)

        # Check if any hook blocked
        should_continue = all(r.should_continue for r in results)

        # Collect additional context from hooks
        additional_context_parts = [
            r.additional_context for r in results if r.additional_context
        ]
        additional_context = (
            "\n".join(additional_context_parts) if additional_context_parts else None
        )

        return should_continue, additional_context, results

    def run_session_start(self) -> list[HookResult]:
        """Run SessionStart hooks when a conversation begins."""
        hooks = self.config.get_hooks_for_event(HookEventType.SESSION_START)
        if not hooks:
            return []

        event = self._create_event(HookEventType.SESSION_START)
        return self.executor.execute_all(hooks, event, stop_on_block=False)

    def run_session_end(self) -> list[HookResult]:
        """Run SessionEnd hooks when a conversation ends."""
        hooks = self.config.get_hooks_for_event(HookEventType.SESSION_END)
        if not hooks:
            return []

        event = self._create_event(HookEventType.SESSION_END)
        return self.executor.execute_all(hooks, event, stop_on_block=False)

    def run_stop(
        self,
        reason: str | None = None,
    ) -> tuple[bool, list[HookResult]]:
        """Run Stop hooks. Returns (should_stop, results)."""
        hooks = self.config.get_hooks_for_event(HookEventType.STOP)
        if not hooks:
            return True, []

        event = self._create_event(
            HookEventType.STOP,
            metadata={"reason": reason} if reason else {},
        )

        results = self.executor.execute_all(hooks, event, stop_on_block=True)

        # If a hook blocks, the agent should NOT stop (continue running)
        should_stop = all(r.should_continue for r in results)

        return should_stop, results

    def has_hooks(self, event_type: HookEventType) -> bool:
        """Check if there are hooks configured for an event type."""
        return self.config.has_hooks_for_event(event_type)

    def get_blocking_reason(self, results: list[HookResult]) -> str | None:
        """Get the reason for blocking from hook results."""
        for result in results:
            if result.blocked:
                if result.reason:
                    return result.reason
                if result.stderr:
                    return result.stderr.strip()
                return "Blocked by hook"
        return None
