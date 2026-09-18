import { describe, expect, it } from "vitest";
import {
  getACPToolCallResult,
  getObservationResult,
} from "#/components/conversation-events/chat/event-content-helpers/get-observation-result";
import { ACPToolCallEvent } from "#/types/agent-server/core/events/acp-tool-call-event";
import { ObservationEvent } from "#/types/agent-server/core";

const makeACPEvent = (
  overrides: Partial<ACPToolCallEvent> = {},
): ACPToolCallEvent => ({
  id: "acp-1",
  kind: "ACPToolCallEvent",
  timestamp: "2024-01-01T00:00:00Z",
  source: "agent",
  tool_call_id: "tc-1",
  title: "Run command",
  status: "completed",
  tool_kind: "execute",
  raw_input: { command: "ls" },
  raw_output: "file.txt",
  content: null,
  is_error: false,
  ...overrides,
});

describe("getACPToolCallResult", () => {
  it("maps completed → success", () => {
    expect(getACPToolCallResult(makeACPEvent({ status: "completed" }))).toBe(
      "success",
    );
  });

  it("maps failed → error", () => {
    expect(getACPToolCallResult(makeACPEvent({ status: "failed" }))).toBe(
      "error",
    );
  });

  it("maps is_error → error even when status is completed", () => {
    expect(
      getACPToolCallResult(
        makeACPEvent({ status: "completed", is_error: true }),
      ),
    ).toBe("error");
  });

  it.each(["pending", "in_progress"] as const)(
    "maps non-terminal status %s → undefined (running card)",
    (status) => {
      expect(getACPToolCallResult(makeACPEvent({ status }))).toBeUndefined();
    },
  );

  it("maps null status → undefined (running card)", () => {
    expect(
      getACPToolCallResult(makeACPEvent({ status: null })),
    ).toBeUndefined();
  });
});

const makeObs = (
  observation: ObservationEvent["observation"],
): ObservationEvent => ({
  id: "obs-1",
  timestamp: "2024-01-01T00:00:00Z",
  source: "environment",
  tool_name: "tool",
  tool_call_id: "tc-1",
  action_id: "act-1",
  observation,
});

const makeMetadata = (exitCode: number) => ({
  exit_code: exitCode,
  pid: 123,
  username: "openhands",
  hostname: "runtime",
  working_dir: "/workspace",
  py_interpreter_path: null,
  prefix: "",
  suffix: "",
});

describe("getObservationResult", () => {
  it("maps bash exit codes from either top-level output or metadata", () => {
    const bash = (exitCode: number | null, metadataExitCode: number) =>
      makeObs({
        kind: "ExecuteBashObservation",
        content: [],
        command: "test",
        exit_code: exitCode,
        error: false,
        timeout: false,
        metadata: makeMetadata(metadataExitCode),
      });

    expect(getObservationResult(bash(-1, 0))).toBe("timeout");
    expect(getObservationResult(bash(null, -1))).toBe("timeout");
    expect(getObservationResult(bash(0, 1))).toBe("success");
    expect(getObservationResult(bash(null, 0))).toBe("success");
    expect(getObservationResult(bash(2, 2))).toBe("error");
  });

  it("maps terminal timeouts, failures, and successful fallback states", () => {
    const terminal = (
      exitCode: number | null | undefined,
      metadataExitCode: number | null,
      overrides: { timeout?: boolean; is_error?: boolean } = {},
    ) =>
      makeObs({
        kind: "TerminalObservation",
        content: [],
        command: "test",
        exit_code: exitCode,
        is_error: overrides.is_error ?? false,
        timeout: overrides.timeout ?? false,
        metadata: makeMetadata(metadataExitCode as number),
      } as unknown as ObservationEvent["observation"]);

    expect(getObservationResult(terminal(1, 1, { timeout: true }))).toBe(
      "timeout",
    );
    expect(getObservationResult(terminal(-1, 0))).toBe("timeout");
    expect(getObservationResult(terminal(0, 1))).toBe("success");
    expect(getObservationResult(terminal(0, 1, { is_error: true }))).toBe(
      "success",
    );
    expect(getObservationResult(terminal(null, 0))).toBe("success");
    expect(getObservationResult(terminal(2, 2, { is_error: true }))).toBe(
      "error",
    );
    expect(getObservationResult(terminal(undefined, null))).toBe("success");
  });

  it("maps editor errors and successful editor outcomes", () => {
    const editor = (kind: string, error: string | null) =>
      makeObs({
        kind,
        command: "view",
        output: "",
        path: "/workspace/file.ts",
        prev_exist: true,
        old_content: null,
        new_content: null,
        error,
      } as ObservationEvent["observation"]);

    expect(
      getObservationResult(editor("FileEditorObservation", "Not found")),
    ).toBe("error");
    expect(
      getObservationResult(editor("StrReplaceEditorObservation", null)),
    ).toBe("success");
    expect(
      getObservationResult(
        editor("StrReplaceEditorObservation", "Replacement failed"),
      ),
    ).toBe("error");
  });

  it("maps MCP and model-switch error flags", () => {
    const mcp = (isError: boolean) =>
      makeObs({
        kind: "MCPToolObservation",
        content: [],
        is_error: isError,
        tool_name: "search",
      });
    const modelSwitch = (isError: boolean) =>
      makeObs({
        kind: "SwitchLLMObservation",
        content: [],
        is_error: isError,
        profile_name: "reviewer",
        reason: null,
        active_model: null,
      });

    expect(getObservationResult(mcp(true))).toBe("error");
    expect(getObservationResult(mcp(false))).toBe("success");
    expect(getObservationResult(modelSwitch(true))).toBe("error");
    expect(getObservationResult(modelSwitch(false))).toBe("success");
  });

  it("maps InvokeSkillObservation is_error → error, otherwise success", () => {
    expect(
      getObservationResult(
        makeObs({
          kind: "InvokeSkillObservation",
          skill_name: "s",
          content: [],
          is_error: true,
        }),
      ),
    ).toBe("error");
    expect(
      getObservationResult(
        makeObs({
          kind: "InvokeSkillObservation",
          skill_name: "s",
          content: [],
          is_error: false,
        }),
      ),
    ).toBe("success");
  });

  it("maps TaskObservation is_error or failed status → error, otherwise success", () => {
    const task = (extra: { status: string; is_error?: boolean }) =>
      makeObs({
        kind: "TaskObservation",
        content: [],
        task_id: "t1",
        subagent: "code-explorer",
        ...extra,
      });
    expect(getObservationResult(task({ status: "completed" }))).toBe("success");
    expect(getObservationResult(task({ status: "failed" }))).toBe("error");
    expect(
      getObservationResult(task({ status: "completed", is_error: true })),
    ).toBe("error");
  });

  it("maps CanvasUIObservation is_error → error, otherwise success", () => {
    expect(
      getObservationResult(
        makeObs({ kind: "CanvasUIObservation", content: [], is_error: true }),
      ),
    ).toBe("error");
    expect(
      getObservationResult(
        makeObs({ kind: "CanvasUIObservation", content: [], is_error: false }),
      ),
    ).toBe("success");
  });

  it("treats observation kinds without failure semantics as successful", () => {
    expect(
      getObservationResult(makeObs({ kind: "ThinkObservation", content: [] })),
    ).toBe("success");
  });
});
