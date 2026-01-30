import { ActionBase } from "./base";
import { TaskItem } from "./common";

export interface MCPToolAction extends ActionBase<"MCPToolAction"> {
  /**
   * Dynamic data fields from the tool call
   */
  data: Record<string, unknown>;
}

export interface FinishAction extends ActionBase<"FinishAction"> {
  /**
   * Final message to send to the user
   */
  message: string;
}

export interface ThinkAction extends ActionBase<"ThinkAction"> {
  /**
   * The thought to log
   */
  thought: string;
}

export interface ExecuteBashAction extends ActionBase<"ExecuteBashAction"> {
  /**
   * The bash command to execute. Can be empty string to view additional logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently running process.
   */
  command: string;
  /**
   * If True, the command is an input to the running process. If False, the command is a bash command to be executed in the terminal. Default is False.
   */
  is_input: boolean;
  /**
   * Optional. Sets a maximum time limit (in seconds) for running the command. If the command takes longer than this limit, you’ll be asked whether to continue or stop it.
   */
  timeout: number | null;
  /**
   * If True, reset the terminal by creating a new session. Used only when the terminal becomes unresponsive. Note that all previously set environment variables and session state will be lost after reset. Cannot be used with is_input=True.
   */
  reset: boolean;
}

export interface TerminalAction extends ActionBase<"TerminalAction"> {
  /**
   * The terminal command to execute.
   */
  command: string;
  /**
   * If True, the command is an input to the running process. If False, the command is executed directly.
   */
  is_input: boolean;
  /**
   * Optional max time limit (seconds) for the command.
   */
  timeout: number | null;
  /**
   * If True, reset the terminal session before running the command.
   */
  reset: boolean;
}

export interface FileEditorAction extends ActionBase<"FileEditorAction"> {
  /**
   * The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.
   */
  command: "view" | "create" | "str_replace" | "insert" | "undo_edit";
  /**
   * Absolute path to file or directory.
   */
  path: string;
  /**
   * Required parameter of `create` command, with the content of the file to be created.
   */
  file_text: string | null;
  /**
   * Required parameter of `str_replace` command containing the string in `path` to replace.
   */
  old_str: string | null;
  /**
   * Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
   */
  new_str: string | null;
  /**
   * Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`. Must be >= 1.
   */
  insert_line: number | null;
  /**
   * Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.
   */
  view_range: [number, number] | null;
}

export interface StrReplaceEditorAction extends ActionBase<"StrReplaceEditorAction"> {
  /**
   * The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.
   */
  command: "view" | "create" | "str_replace" | "insert" | "undo_edit";
  /**
   * Absolute path to file or directory.
   */
  path: string;
  /**
   * Required parameter of `create` command, with the content of the file to be created.
   */
  file_text: string | null;
  /**
   * Required parameter of `str_replace` command containing the string in `path` to replace.
   */
  old_str: string | null;
  /**
   * Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
   */
  new_str: string | null;
  /**
   * Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`. Must be >= 1.
   */
  insert_line: number | null;
  /**
   * Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.
   */
  view_range: [number, number] | null;
}

export interface TaskTrackerAction extends ActionBase<"TaskTrackerAction"> {
  /**
   * The command to execute. `view` shows the current task list. `plan` creates or updates the task list based on provided requirements and progress. Always `view` the current list before making changes.
   */
  command: "view" | "plan";
  /**
   * The full task list. Required parameter of `plan` command.
   */
  task_list: TaskItem[];
}

export interface BrowserNavigateAction extends ActionBase<"BrowserNavigateAction"> {
  /**
   * The URL to navigate to
   */
  url: string;
  /**
   * Whether to open in a new tab. Default: False
   */
  new_tab: boolean;
}

export interface BrowserClickAction extends ActionBase<"BrowserClickAction"> {
  /**
   * The index of the element to click (from browser_get_state)
   */
  index: number;
  /**
   * Whether to open any resulting navigation in a new tab. Default: False
   */
  new_tab: boolean;
}

export interface BrowserTypeAction extends ActionBase<"BrowserTypeAction"> {
  /**
   * The index of the input element (from browser_get_state)
   */
  index: number;
  /**
   * The text to type
   */
  text: string;
}

export interface BrowserGetStateAction extends ActionBase<"BrowserGetStateAction"> {
  /**
   * Whether to include a screenshot of the current page. Default: False
   */
  include_screenshot: boolean;
}

export interface BrowserGetContentAction extends ActionBase<"BrowserGetContentAction"> {
  /**
   * Whether to include links in the content (default: False)
   */
  extract_links: boolean;
  /**
   * Character index to start from in the page content (default: 0)
   */
  start_from_char: number;
}

export interface BrowserScrollAction extends ActionBase<"BrowserScrollAction"> {
  /**
   * Direction to scroll. Options: 'up', 'down'. Default: 'down'
   */
  direction: "up" | "down";
}

export interface BrowserGoBackAction extends ActionBase<"BrowserGoBackAction"> {
  // No additional properties - this action has no parameters
}

export interface BrowserListTabsAction extends ActionBase<"BrowserListTabsAction"> {
  // No additional properties - this action has no parameters
}

export interface BrowserSwitchTabAction extends ActionBase<"BrowserSwitchTabAction"> {
  /**
   * 4 Character Tab ID of the tab to switch to (from browser_list_tabs)
   */
  tab_id: string;
}

export interface BrowserCloseTabAction extends ActionBase<"BrowserCloseTabAction"> {
  /**
   * 4 Character Tab ID of the tab to close (from browser_list_tabs)
   */
  tab_id: string;
}

export interface PlanningFileEditorAction extends ActionBase<"PlanningFileEditorAction"> {
  /**
   * The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.
   */
  command: "view" | "create" | "str_replace" | "insert" | "undo_edit";
  /**
   * Absolute path to file (typically /workspace/project/PLAN.md).
   */
  path: string;
  /**
   * Required parameter of `create` command, with the content of the file to be created.
   */
  file_text: string | null;
  /**
   * Required parameter of `str_replace` command containing the string in `path` to replace.
   */
  old_str: string | null;
  /**
   * Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
   */
  new_str: string | null;
  /**
   * Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`. Must be >= 1.
   */
  insert_line: number | null;
  /**
   * Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown.
   */
  view_range: [number, number] | null;
}

export type Action =
  | MCPToolAction
  | FinishAction
  | ThinkAction
  | ExecuteBashAction
  | TerminalAction
  | FileEditorAction
  | StrReplaceEditorAction
  | TaskTrackerAction
  | PlanningFileEditorAction
  | BrowserNavigateAction
  | BrowserClickAction
  | BrowserTypeAction
  | BrowserGetStateAction
  | BrowserGetContentAction
  | BrowserScrollAction
  | BrowserGoBackAction
  | BrowserListTabsAction
  | BrowserSwitchTabAction
  | BrowserCloseTabAction;
