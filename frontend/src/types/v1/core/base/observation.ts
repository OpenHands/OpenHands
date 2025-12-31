import { ObservationBase } from "./base";
import {
  CmdOutputMetadata,
  TaskItem,
  TextContent,
  ImageContent,
} from "./common";

export interface MCPToolObservation extends ObservationBase<"MCPToolObservation"> {
  /**
   * Content returned from the MCP tool converted to LLM Ready TextContent or ImageContent
   */
  content: Array<TextContent | ImageContent>;
  /**
   * Whether the call resulted in an error
   */
  is_error: boolean;
  /**
   * Name of the tool that was called
   */
  tool_name: string;
}

export interface FinishObservation extends ObservationBase<"FinishObservation"> {
  /**
   * Content returned from the finish action as a list of TextContent/ImageContent objects.
   */
  content: Array<TextContent | ImageContent>;
  /**
   * Whether the finish action resulted in an error
   */
  is_error: boolean;
}

export interface ThinkObservation extends ObservationBase<"ThinkObservation"> {
  /**
   * Confirmation message. DEFAULT: "Your thought has been logged."
   */
  content: Array<TextContent | ImageContent>;
}

export interface BrowserObservation extends ObservationBase<"BrowserObservation"> {
  /**
   * The output message from the browser operation
   */
  output: string;
  /**
   * Error message if any
   */
  error: string | null;
  /**
   * Base64 screenshot data if available
   */
  screenshot_data: string | null;
}

export interface ExecuteBashObservation extends ObservationBase<"ExecuteBashObservation"> {
  /**
   * Content returned from the tool as a list of TextContent/ImageContent objects.
   */
  content: Array<TextContent | ImageContent>;
  /**
   * The bash command that was executed. Can be empty string if the observation is from a previous command that hit soft timeout and is not yet finished.
   */
  command: string | null;
  /**
   * The exit code of the command. -1 indicates the process hit the soft timeout and is not yet finished.
   */
  exit_code: number | null;
  /**
   * Whether there was an error during command execution.
   */
  error: boolean;
  /**
   * Whether the command execution timed out.
   */
  timeout: boolean;
  /**
   * Additional metadata captured from PS1 after command execution.
   */
  metadata: CmdOutputMetadata;
}

export interface TerminalObservation extends ObservationBase<"TerminalObservation"> {
  /**
   * Content returned from the terminal as a list of TextContent/ImageContent objects.
   */
  content: Array<TextContent | ImageContent>;
  /**
   * The bash command that was executed.
   */
  command: string | null;
  /**
   * The exit code of the command if it has finished.
   */
  exit_code: number | null;
  /**
   * Whether the command execution produced an error.
   */
  is_error: boolean;
  /**
   * Whether the command execution timed out.
   */
  timeout: boolean;
  /**
   * Additional metadata captured from the shell after command execution.
   */
  metadata: CmdOutputMetadata;
}

export interface FileEditorObservation extends ObservationBase<"FileEditorObservation"> {
  /**
   * The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.
   */
  command: "view" | "create" | "str_replace" | "insert" | "undo_edit";
  /**
   * The output message from the tool for the LLM to see.
   */
  output: string;
  /**
   * The file path that was edited.
   */
  path: string | null;
  /**
   * Indicates if the file previously existed. If not, it was created.
   */
  prev_exist: boolean;
  /**
   * The content of the file before the edit.
   */
  old_content: string | null;
  /**
   * The content of the file after the edit.
   */
  new_content: string | null;
  /**
   * Error message if any.
   */
  error: string | null;
}

// Keep StrReplaceEditorObservation as a separate interface for backward compatibility
export interface StrReplaceEditorObservation extends ObservationBase<"StrReplaceEditorObservation"> {
  /**
   * The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.
   */
  command: "view" | "create" | "str_replace" | "insert" | "undo_edit";
  /**
   * The output message from the tool for the LLM to see.
   */
  output: string;
  /**
   * The file path that was edited.
   */
  path: string | null;
  /**
   * Indicates if the file previously existed. If not, it was created.
   */
  prev_exist: boolean;
  /**
   * The content of the file before the edit.
   */
  old_content: string | null;
  /**
   * The content of the file after the edit.
   */
  new_content: string | null;
  /**
   * Error message if any.
   */
  error: string | null;
}

export interface TaskTrackerObservation extends ObservationBase<"TaskTrackerObservation"> {
  /**
   * The formatted task list or status message.
   */
  content: string;
  /**
   * The command that was executed.
   */
  command: string;
  /**
   * The current task list.
   */
  task_list: TaskItem[];
}

export interface PlanningFileEditorObservation extends ObservationBase<"PlanningFileEditorObservation"> {
  /**
   * Content returned from the tool as a list of TextContent/ImageContent objects.
   */
  content: Array<TextContent | ImageContent>;
  /**
   * Whether the call resulted in an error.
   */
  is_error: boolean;
  /**
   * The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.
   */
  command: "view" | "create" | "str_replace" | "insert" | "undo_edit";
  /**
   * The file path that was edited.
   */
  path: string | null;
  /**
   * Indicates if the file previously existed. If not, it was created.
   */
  prev_exist: boolean;
  /**
   * The content of the file before the edit.
   */
  old_content: string | null;
  /**
   * The content of the file after the edit.
   */
  new_content: string | null;
}

export type Observation =
  | MCPToolObservation
  | FinishObservation
  | ThinkObservation
  | BrowserObservation
  | ExecuteBashObservation
  | TerminalObservation
  | FileEditorObservation
  | StrReplaceEditorObservation
  | TaskTrackerObservation
  | PlanningFileEditorObservation;
