import React from "react";
import { renderWithProviders } from "test-utils";
import {
  ActionEvent,
  ObservationEvent,
  SecurityRisk,
} from "#/types/agent-server/core";
import {
  ExecuteBashAction,
  FileEditorAction,
  GrepAction,
  TaskAction,
} from "#/types/agent-server/core/base/action";
import {
  ExecuteBashObservation,
  FileEditorObservation,
  GrepObservation,
  GlobObservation,
  TerminalObservation,
  TaskObservation,
} from "#/types/agent-server/core/base/observation";

/** Renders a visualizer body with the standard chat providers. */
export const renderVisualizer = (ui: React.ReactElement) =>
  renderWithProviders(ui);

const actionEnvelope = (id: string, toolName: string) => ({
  id,
  timestamp: "2024-01-01T00:00:00Z",
  source: "agent" as const,
  thought: [],
  thinking_blocks: [],
  tool_name: toolName,
  tool_call_id: `call_${id}`,
  tool_call: {
    id: `call_${id}`,
    type: "function" as const,
    function: { name: toolName, arguments: "{}" },
  },
  llm_response_id: `resp_${id}`,
  security_risk: SecurityRisk.LOW,
});

const observationEnvelope = (id: string, toolName: string) => ({
  id,
  timestamp: "2024-01-01T00:00:00Z",
  source: "environment" as const,
  tool_name: toolName,
  tool_call_id: `call_${id}`,
  action_id: `action_${id}`,
});

export const bashAction = (
  command: string,
  risk: SecurityRisk = SecurityRisk.LOW,
): ActionEvent<ExecuteBashAction> => ({
  ...actionEnvelope("bash", "execute_bash"),
  security_risk: risk,
  action: {
    kind: "ExecuteBashAction",
    command,
    is_input: false,
    timeout: null,
    reset: false,
  },
});

export const bashObservation = (
  output: string,
  exitCode: number | null,
  command = "echo hi",
): ObservationEvent<ExecuteBashObservation> => ({
  ...observationEnvelope("bash", "execute_bash"),
  observation: {
    kind: "ExecuteBashObservation",
    content: [{ type: "text", text: output }],
    command,
    exit_code: exitCode,
    error: exitCode !== 0,
    timeout: false,
    metadata: {
      exit_code: exitCode ?? 0,
      pid: 1,
      username: "openhands",
      hostname: "runtime",
      prefix: "",
      suffix: "",
      working_dir: "/workspace",
      py_interpreter_path: null,
    },
  },
});

export const terminalObservation = (
  output: string,
  exitCode: number | null,
  command = "ls",
): ObservationEvent<TerminalObservation> => ({
  ...observationEnvelope("terminal", "terminal"),
  observation: {
    kind: "TerminalObservation",
    content: [{ type: "text", text: output }],
    command,
    exit_code: exitCode,
    is_error: exitCode !== 0,
    timeout: false,
    metadata: {
      exit_code: exitCode ?? 0,
      pid: 1,
      username: "openhands",
      hostname: "runtime",
      prefix: "",
      suffix: "",
      working_dir: "/workspace",
      py_interpreter_path: null,
    },
  },
});

export const fileEditorAction = (
  action: Partial<FileEditorAction> &
    Pick<FileEditorAction, "command" | "path">,
): ActionEvent<FileEditorAction> => ({
  ...actionEnvelope("fe", "file_editor"),
  action: {
    kind: "FileEditorAction",
    file_text: null,
    old_str: null,
    new_str: null,
    insert_line: null,
    view_range: null,
    ...action,
  },
});

export const fileEditorObservation = (
  observation: Partial<FileEditorObservation> &
    Pick<FileEditorObservation, "command">,
): ObservationEvent<FileEditorObservation> => ({
  ...observationEnvelope("fe", "file_editor"),
  observation: {
    kind: "FileEditorObservation",
    output: "",
    path: "/workspace/app.ts",
    prev_exist: true,
    old_content: null,
    new_content: null,
    error: null,
    ...observation,
  },
});

export const grepAction = (
  action: Partial<GrepAction> & Pick<GrepAction, "pattern">,
): ActionEvent<GrepAction> => ({
  ...actionEnvelope("grep", "grep"),
  action: { kind: "GrepAction", path: null, include: null, ...action },
});

export const grepObservation = (
  observation: Partial<GrepObservation> & Pick<GrepObservation, "pattern">,
): ObservationEvent<GrepObservation> => ({
  ...observationEnvelope("grep", "grep"),
  observation: {
    kind: "GrepObservation",
    content: [],
    is_error: false,
    matches: [],
    search_path: "/workspace",
    include_pattern: null,
    truncated: false,
    ...observation,
  },
});

export const globObservation = (
  observation: Partial<GlobObservation> & Pick<GlobObservation, "pattern">,
): ObservationEvent<GlobObservation> => ({
  ...observationEnvelope("glob", "glob"),
  observation: {
    kind: "GlobObservation",
    content: [],
    is_error: false,
    files: [],
    search_path: "/workspace",
    truncated: false,
    ...observation,
  },
});

export const taskAction = (
  action: Partial<TaskAction> & Pick<TaskAction, "prompt">,
): ActionEvent<TaskAction> => ({
  ...actionEnvelope("task", "task"),
  action: {
    kind: "TaskAction",
    subagent_type: "code-explorer",
    description: null,
    resume: null,
    ...action,
  },
});

export const taskObservation = (
  observation: Partial<TaskObservation> = {},
): ObservationEvent<TaskObservation> => ({
  ...observationEnvelope("task", "task"),
  observation: {
    kind: "TaskObservation",
    content: [{ type: "text", text: "## Summary\n\nAll done." }],
    is_error: false,
    task_id: "task_00000001",
    subagent: "code-explorer",
    status: "completed",
    ...observation,
  },
});
