import { describe, expect, it } from "vitest";
import { deriveLiveActivity } from "#/components/features/chat/typing-indicator";
import {
  SecurityRisk,
  type ActionEvent,
  type ObservationEvent,
  type UserRejectObservation,
} from "#/types/agent-server/core";
import type { Action } from "#/types/agent-server/core/base/action";
import type { ACPToolCallEvent } from "#/types/agent-server/core/events/acp-tool-call-event";
import type { OHEvent } from "#/stores/use-event-store";

const makeActionEvent = (
  id: string,
  action: Action,
  summary?: string,
): ActionEvent => ({
  id,
  timestamp: `2026-07-27T18:00:${id.padStart(2, "0")}Z`,
  source: "agent",
  thought: [],
  thinking_blocks: [],
  action,
  tool_name: action.kind,
  tool_call_id: `tool-${id}`,
  tool_call: {
    id: `tool-${id}`,
    type: "function",
    function: {
      name: action.kind,
      arguments: "{}",
    },
  },
  llm_response_id: `response-${id}`,
  security_risk: SecurityRisk.LOW,
  summary,
});

const makeObservationEvent = (
  id: string,
  action: ActionEvent,
): ObservationEvent => ({
  id,
  timestamp: `2026-07-27T18:01:${id.padStart(2, "0")}Z`,
  source: "environment",
  tool_name: action.tool_name,
  tool_call_id: action.tool_call_id,
  action_id: action.id,
  observation: {
    kind: "TerminalObservation",
    content: [],
    command: "git status",
    exit_code: 0,
    is_error: false,
    timeout: false,
    metadata: {
      exit_code: 0,
      pid: 1,
      username: "openhands",
      hostname: "runtime",
      prefix: "",
      suffix: "",
      working_dir: "/workspace/project",
      py_interpreter_path: null,
    },
  },
});

const makeUserRejectObservation = (
  id: string,
  action: ActionEvent,
): UserRejectObservation => ({
  id,
  timestamp: `2026-07-27T18:01:${id.padStart(2, "0")}Z`,
  source: "environment",
  tool_name: action.tool_name,
  tool_call_id: action.tool_call_id,
  action_id: action.id,
  rejection_reason: "Rejected by user",
});

const makeACPEvent = (
  id: string,
  status: ACPToolCallEvent["status"],
): ACPToolCallEvent => ({
  id: `${id}-${status ?? "legacy"}`,
  timestamp: "2026-07-27T18:02:00Z",
  kind: "ACPToolCallEvent",
  source: "agent",
  tool_call_id: id,
  title: "Read README.md",
  status,
  tool_kind: "read",
  raw_input: null,
  raw_output: null,
  content: null,
  is_error: status === "failed",
});

describe("deriveLiveActivity", () => {
  it.each([
    {
      name: "terminal command",
      action: {
        kind: "TerminalAction",
        command: "git status",
        is_input: false,
        timeout: null,
        reset: false,
      } satisfies Action,
      expected: {
        kind: "translation",
        key: "ACTION_MESSAGE$RUN",
        values: { command: "git status" },
      },
    },
    {
      name: "file read",
      action: {
        kind: "FileEditorAction",
        command: "view",
        path: "README.md",
        file_text: null,
        old_str: null,
        new_str: null,
        insert_line: null,
        view_range: null,
      } satisfies Action,
      expected: {
        kind: "translation",
        key: "ACTION_MESSAGE$READ",
        values: { path: "README.md" },
      },
    },
    {
      name: "code search",
      action: {
        kind: "GrepAction",
        pattern: "VITE_",
        path: null,
        include: null,
      } satisfies Action,
      expected: {
        kind: "translation",
        key: "ACTION_MESSAGE$GREP",
        values: { pattern: "VITE_" },
      },
    },
  ])("describes an unresolved $name", ({ action, expected }) => {
    const event = makeActionEvent("1", action);

    expect(deriveLiveActivity([event])).toEqual(expected);
  });

  it("moves to the next unresolved regular action as observations arrive", () => {
    const first = makeActionEvent("1", {
      kind: "TerminalAction",
      command: "npm test",
      is_input: false,
      timeout: null,
      reset: false,
    });
    const second = makeActionEvent("2", {
      kind: "FileEditorAction",
      command: "view",
      path: "README.md",
      file_text: null,
      old_str: null,
      new_str: null,
      insert_line: null,
      view_range: null,
    });
    const secondObservation = makeObservationEvent("3", second);
    const firstObservation = makeObservationEvent("4", first);

    expect(deriveLiveActivity([first, second, secondObservation])).toEqual({
      kind: "translation",
      key: "ACTION_MESSAGE$RUN",
      values: { command: "npm test" },
    });
    expect(
      deriveLiveActivity([first, second, secondObservation, firstObservation]),
    ).toEqual({
      kind: "translation",
      key: "ACTION_MESSAGE$THINK",
      values: {},
    });
  });

  it("treats a rejected regular action as resolved", () => {
    const rejected = makeActionEvent("1", {
      kind: "TerminalAction",
      command: "rm -rf build",
      is_input: false,
      timeout: null,
      reset: false,
    });

    expect(
      deriveLiveActivity([rejected, makeUserRejectObservation("2", rejected)]),
    ).toEqual({
      kind: "translation",
      key: "ACTION_MESSAGE$THINK",
      values: {},
    });
  });

  it("shows the next unresolved action after a newer action is rejected", () => {
    const remaining = makeActionEvent("1", {
      kind: "TerminalAction",
      command: "git status",
      is_input: false,
      timeout: null,
      reset: false,
    });
    const rejected = makeActionEvent("2", {
      kind: "TerminalAction",
      command: "npm publish",
      is_input: false,
      timeout: null,
      reset: false,
    });

    expect(
      deriveLiveActivity([
        remaining,
        rejected,
        makeUserRejectObservation("3", rejected),
      ]),
    ).toEqual({
      kind: "translation",
      key: "ACTION_MESSAGE$RUN",
      values: { command: "git status" },
    });
  });

  it.each([
    "completed",
    "failed",
    null,
  ] satisfies readonly ACPToolCallEvent["status"][])(
    "drops an ACP label when the same call becomes %s",
    (terminalStatus) => {
      const started = makeACPEvent("acp-read", "in_progress");

      expect(deriveLiveActivity([started])).toEqual({
        kind: "translation",
        key: "ACTION_MESSAGE$ACP_READ",
        values: { title: "README.md" },
      });
      expect(
        deriveLiveActivity([started, makeACPEvent("acp-read", terminalStatus)]),
      ).toEqual({
        kind: "translation",
        key: "ACTION_MESSAGE$THINK",
        values: {},
      });
    },
  );

  it("falls back to Thinking for unsupported and planning-only actions", () => {
    const unsupported = makeActionEvent("1", {
      kind: "SwitchLLMAction",
      profile_name: "fast",
      reason: "Short task",
    });
    const planningOnly = {
      ...makeActionEvent("2", {
        kind: "TerminalAction",
        command: "git status",
        is_input: false,
        timeout: null,
        reset: false,
      }),
      isFromPlanningAgent: true,
    } satisfies OHEvent;

    expect(deriveLiveActivity([unsupported])).toEqual({
      kind: "translation",
      key: "ACTION_MESSAGE$THINK",
      values: {},
    });
    expect(deriveLiveActivity([planningOnly])).toEqual({
      kind: "translation",
      key: "ACTION_MESSAGE$THINK",
      values: {},
    });
    expect(deriveLiveActivity([])).toEqual({
      kind: "translation",
      key: "ACTION_MESSAGE$THINK",
      values: {},
    });
  });
});
