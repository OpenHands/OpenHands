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

// Content types for LLM messages
export interface TextContent {
  type: "text";
  text: string;
  cache_prompt?: boolean;
}

export interface ImageContent {
  type: "image";
  image_urls: string[];
  cache_prompt?: boolean;
}
