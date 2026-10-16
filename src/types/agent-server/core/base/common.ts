export interface TaskItem {
  /**
   * A brief title for the task.
   */
  title: string;
  /**
   * Additional details or notes about the task.
   */
  notes: string;
  /**
   * The current status of the task. One of 'todo', 'in_progress', or 'done'.
   */
  status: "todo" | "in_progress" | "done";
}

export interface CmdOutputMetadata {
  /**
   * The exit code of the last executed command
   */
  exit_code: number;
  /**
   * The process ID of the last executed command
   */
  pid: number;
  /**
   * The username of the current user
   */
  username: string | null;
  /**
   * The hostname of the machine
   */
  hostname: string | null;
  /**
   * The current working directory
   */
  working_dir: string | null;
  /**
   * The path to the current Python interpreter, if any
   */
  py_interpreter_path: string | null;
  /**
   * Prefix to add to command output
   */
  prefix: string;
  /**
   * Suffix to add to command output
   */
  suffix: string;
}

// Type aliases for event and tool call IDs
export type EventID = string;
export type ToolCallID = string;

// Security risk levels
export enum SecurityRisk {
  UNKNOWN = "UNKNOWN",
  LOW = "LOW",
  MEDIUM = "MEDIUM",
  HIGH = "HIGH",
}

// =============================================================================
// Canonical event source and execution status contracts
// =============================================================================
//
// These are sourced from `@openhands/typescript-client`, the canonical browser
// contract for the agent-server event wire, rather than being redeclared here.
// The client's supersets intentionally cover variants Canvas misses:
//   - `EventSource` adds `system` to the historical four sources.
//   - `ConversationExecutionStatus` adds `deleting`.
// Canvas consumers therefore stay in sync with new server statuses/sources
// without carrying a parallel, drifting copy of the model.

export type SourceType = import("@openhands/typescript-client").EventSource;

export { ConversationExecutionStatus as ExecutionStatus } from "@openhands/typescript-client";

// Content types for LLM messages.
//
// Sourced from `@openhands/typescript-client`, the canonical agent-server
// contract, instead of a parallel copy. The canonical `TextContent` /
// `ImageContent` are refined with the wire's optional `cache_prompt` flag,
// which the server emits (agent-server-schema) but the client's simplified
// content types omit. Keeping the refinement narrow (one field) avoids
// recreating the wire contract while staying in sync with the client's
// structural base and any new content variants it adds.
export type TextContent = import("@openhands/typescript-client").TextContent & {
  /**
   * Whether this content block is cached by the prompt-caching layer.
   */
  cache_prompt?: boolean;
};

export type ImageContent =
  import("@openhands/typescript-client").ImageContent & {
    /**
     * Whether this content block is cached by the prompt-caching layer.
     */
    cache_prompt?: boolean;
  };
