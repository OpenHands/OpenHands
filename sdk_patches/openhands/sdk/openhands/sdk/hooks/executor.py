"""Hook executor - runs shell commands with JSON I/O."""

import json
import os
import subprocess

from pydantic import BaseModel

from openhands.sdk.hooks.config import HookDefinition
from openhands.sdk.hooks.types import HookDecision, HookEvent


class HookResult(BaseModel):
    """Result from executing a hook.

    Exit code 0 = success, exit code 2 = block operation.
    """

    success: bool = True
    blocked: bool = False
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    decision: HookDecision | None = None
    reason: str | None = None
    additional_context: str | None = None
    error: str | None = None

    @property
    def should_continue(self) -> bool:
        """Whether the operation should continue after this hook."""
        if self.blocked:
            return False
        if self.decision == HookDecision.DENY:
            return False
        return True


class HookExecutor:
    """Executes hook commands with JSON I/O."""

    def __init__(self, working_dir: str | None = None):
        self.working_dir = working_dir or os.getcwd()

    def execute(
        self,
        hook: HookDefinition,
        event: HookEvent,
        env: dict[str, str] | None = None,
    ) -> HookResult:
        """Execute a single hook."""
        # Prepare environment
        hook_env = os.environ.copy()
        hook_env["OPENHANDS_PROJECT_DIR"] = self.working_dir
        hook_env["OPENHANDS_SESSION_ID"] = event.session_id or ""
        hook_env["OPENHANDS_EVENT_TYPE"] = event.event_type
        if event.tool_name:
            hook_env["OPENHANDS_TOOL_NAME"] = event.tool_name

        if env:
            hook_env.update(env)

        # Serialize event to JSON for stdin
        event_json = event.model_dump_json()

        try:
            # Execute the hook command
            result = subprocess.run(
                hook.command,
                shell=True,
                cwd=self.working_dir,
                env=hook_env,
                input=event_json,
                capture_output=True,
                text=True,
                timeout=hook.timeout,
            )

            # Parse the result
            hook_result = HookResult(
                success=result.returncode == 0,
                blocked=result.returncode == 2,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

            # Try to parse JSON from stdout
            if result.stdout.strip():
                try:
                    output_data = json.loads(result.stdout)
                    if isinstance(output_data, dict):
                        # Parse decision
                        if "decision" in output_data:
                            decision_str = output_data["decision"].lower()
                            if decision_str == "allow":
                                hook_result.decision = HookDecision.ALLOW
                            elif decision_str == "deny":
                                hook_result.decision = HookDecision.DENY
                                hook_result.blocked = True

                        # Parse other fields
                        if "reason" in output_data:
                            hook_result.reason = str(output_data["reason"])
                        if "additionalContext" in output_data:
                            hook_result.additional_context = str(
                                output_data["additionalContext"]
                            )
                        if "continue" in output_data:
                            if not output_data["continue"]:
                                hook_result.blocked = True

                except json.JSONDecodeError:
                    # Not JSON, that's okay - just use stdout as-is
                    pass

            return hook_result

        except subprocess.TimeoutExpired:
            return HookResult(
                success=False,
                exit_code=-1,
                error=f"Hook timed out after {hook.timeout} seconds",
            )
        except FileNotFoundError as e:
            return HookResult(
                success=False,
                exit_code=-1,
                error=f"Hook command not found: {e}",
            )
        except Exception as e:
            return HookResult(
                success=False,
                exit_code=-1,
                error=f"Hook execution failed: {e}",
            )

    def execute_all(
        self,
        hooks: list[HookDefinition],
        event: HookEvent,
        env: dict[str, str] | None = None,
        stop_on_block: bool = True,
    ) -> list[HookResult]:
        """Execute multiple hooks in order, optionally stopping on block."""
        results: list[HookResult] = []

        for hook in hooks:
            result = self.execute(hook, event, env)
            results.append(result)

            if stop_on_block and result.blocked:
                break

        return results
