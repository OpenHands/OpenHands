import { describe, expect, it } from "vitest";
import { detectAgentNotificationsFromEvents } from "#/components/features/chat/detect-agent-notifications-from-events";
import type { OHEvent } from "#/stores/use-event-store";

function fileEditEvent(path: string): OHEvent {
  return {
    id: `file-${path}`,
    source: "agent",
    timestamp: "2026-01-01T00:00:00.000Z",
    tool_name: "file_editor",
    tool_call_id: `call-file-${path}`,
    action: {
      kind: "FileEditorAction",
      command: "str_replace",
      path,
      file_text: null,
      old_str: "a",
      new_str: "b",
      insert_line: null,
      view_range: null,
    },
  } as OHEvent;
}

function viewEvent(path: string): OHEvent {
  return {
    id: `view-${path}`,
    source: "agent",
    timestamp: "2026-01-01T00:00:00.000Z",
    tool_name: "file_editor",
    tool_call_id: `call-view-${path}`,
    action: {
      kind: "FileEditorAction",
      command: "view",
      path,
      file_text: null,
      old_str: null,
      new_str: null,
      insert_line: null,
      view_range: [1, 50],
    },
  } as OHEvent;
}

function bashEvent(command: string): OHEvent {
  return {
    id: `bash-${command}`,
    source: "agent",
    timestamp: "2026-01-01T00:00:01.000Z",
    tool_name: "terminal",
    tool_call_id: `call-bash-${command}`,
    action: {
      kind: "ExecuteBashAction",
      command,
      is_input: false,
      timeout: null,
      reset: false,
    },
  } as OHEvent;
}

describe("detectAgentNotificationsFromEvents", () => {
  it("recommends a skill after substantive file edits", () => {
    const recommendations = detectAgentNotificationsFromEvents([
      fileEditEvent("/workspace/project/scripts/check-translation.cjs"),
    ]);

    expect(recommendations).toHaveLength(1);
    expect(recommendations[0]).toMatchObject({
      kind: "skill",
      name: "Check Translation helper",
      id: "detected-skill-check-translation",
    });
    expect(recommendations[0]?.prompt).toContain("check-translation.cjs");
  });

  it("recommends a workflow when a test command was run", () => {
    const recommendations = detectAgentNotificationsFromEvents([
      bashEvent("npm test"),
    ]);

    expect(recommendations).toHaveLength(1);
    expect(recommendations[0]).toMatchObject({
      kind: "workflow",
      name: "Test runner workflow",
      id: "detected-workflow-test-runner",
    });
    expect(recommendations[0]?.prompt).toContain("npm test");
  });

  it("returns at most two recommendations and prefers skill plus workflow", () => {
    const recommendations = detectAgentNotificationsFromEvents([
      fileEditEvent("/workspace/project/src/utils/format.ts"),
      bashEvent("npm test"),
      bashEvent("gh pr checks"),
    ]);

    expect(recommendations).toHaveLength(2);
    expect(recommendations.map((item) => item.kind)).toEqual([
      "skill",
      "workflow",
    ]);
  });

  it("recommends a skill from file-editor observations", () => {
    const observation = {
      id: "file-obs-1",
      source: "environment",
      timestamp: "2026-01-01T00:00:00.000Z",
      tool_name: "file_editor",
      tool_call_id: "call-file-obs-1",
      action_id: "action-file-obs-1",
      observation: {
        kind: "FileEditorObservation",
        command: "str_replace",
        path: "/workspace/project/README.md",
        output: "ok",
        prev_exist: true,
        old_content: "a",
        new_content: "b",
      },
    } as OHEvent;

    const recommendations = detectAgentNotificationsFromEvents([observation]);

    expect(recommendations).toHaveLength(1);
    expect(recommendations[0]).toMatchObject({
      kind: "skill",
      id: "detected-skill-readme",
      name: "README helper",
    });
  });

  it("recommends a skill after repeated file views in a code-review style flow", () => {
    const recommendations = detectAgentNotificationsFromEvents([
      viewEvent("/workspace/project/src/App.tsx"),
      viewEvent("/workspace/project/src/utils/format.ts"),
      viewEvent("/workspace/project/README.md"),
    ]);

    expect(recommendations).toHaveLength(1);
    expect(recommendations[0]).toMatchObject({
      kind: "skill",
      id: "detected-skill-app",
      name: "App helper",
    });
    expect(recommendations[0]?.prompt).toContain("code exploration");
  });

  it("recommends a workflow from terminal observations", () => {
    const observation = {
      id: "terminal-obs-1",
      source: "environment",
      timestamp: "2026-01-01T00:00:01.000Z",
      tool_name: "terminal",
      tool_call_id: "call-terminal-obs-1",
      action_id: "action-terminal-obs-1",
      observation: {
        kind: "TerminalObservation",
        command: "npm test",
        content: [],
        exit_code: 0,
        error: false,
        timeout: false,
      },
    } as OHEvent;

    const recommendations = detectAgentNotificationsFromEvents([observation]);

    expect(recommendations[0]).toMatchObject({
      kind: "workflow",
      id: "detected-workflow-test-runner",
    });
  });
});
