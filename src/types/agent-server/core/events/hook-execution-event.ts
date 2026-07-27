import { BaseEvent } from "../base/event";

/**
 * Hook event types supported by the system
 */
export type HookEventType =
  | "PreToolUse"
  | "PostToolUse"
  | "UserPromptSubmit"
  | "SessionStart"
  | "SessionEnd"
  | "Stop";

/**
 * HookExecutionEvent - emitted when a hook script executes
 *
 * Provides observability into hook execution for PreToolUse, PostToolUse,
 * UserPromptSubmit, SessionStart, SessionEnd, and Stop hooks.
 */
export interface HookExecutionEvent extends BaseEvent {
  /**
   * Discriminator field for type guards
   */
  kind: "HookExecutionEvent";

  /**
   * The source is always "hook" for hook execution events
   */
  source: "hook";

  /**
   * Type of hook that was executed
   */
  hook_event_type: HookEventType;

  /**
   * The command that was executed
   */
  hook_command: string;

  /**
   * Whether the hook executed successfully
   */
  success: boolean;

  /**
   * Whether the hook blocked the action
   */
  blocked: boolean;

  /**
   * Exit code from the hook script (null if not applicable)
   */
  exit_code: number | null;

  /**
   * Reason provided by the hook for blocking (if blocked)
   */
  reason: string | null;

  /**
   * Name of the tool (for PreToolUse/PostToolUse hooks)
   */
  tool_name: string | null;

  /**
   * ID of the related action event (for tool hooks)
   */
  action_id: string | null;

  /**
   * ID of the related message event (for UserPromptSubmit hooks)
   */
  message_id: string | null;

  /**
   * Standard output from the hook script
   */
  stdout: string | null;

  /**
   * Standard error from the hook script
   */
  stderr: string | null;

  /**
   * Error message if the hook failed
   */
  error: string | null;

  /**
   * Additional context provided by the hook
   */
  additional_context: string | null;

  /**
   * Input data that was passed to the hook
   */
  hook_input: Record<string, unknown> | null;
}
