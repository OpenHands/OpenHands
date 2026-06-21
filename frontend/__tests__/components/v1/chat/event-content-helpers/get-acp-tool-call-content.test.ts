import { describe, it, expect } from "vitest";
import { getACPToolCallContent } from "#/components/v1/chat/event-content-helpers/get-acp-tool-call-content";
import { getACPToolCallResult } from "#/components/v1/chat/event-content-helpers/get-observation-result";
import { ACPToolCallEvent } from "#/types/v1/core/events/acp-tool-call-event";

const baseEvent: ACPToolCallEvent = {
  kind: "ACPToolCallEvent",
  id: "evt-1",
  timestamp: "2026-04-16T19:32:29.828069",
  source: "agent",
  tool_call_id: "toolu_123",
  title: "gh pr diff 490 --repo OpenHands/evaluation",
  tool_kind: "execute",
  status: "completed",
  raw_input: { command: "gh pr diff 490 --repo OpenHands/evaluation" },
  raw_output: "diff --git a/foo b/foo\n+added\n",
  content: null,
  is_error: false,
};

const makeEvent = (overrides: Partial<ACPToolCallEvent>): ACPToolCallEvent => ({
  ...baseEvent,
  ...overrides,
});

describe("getACPToolCallContent", () => {
  it("renders execute tool calls with Command: and Output: blocks, matching terminal observations", () => {
    const content = getACPToolCallContent(baseEvent);

    expect(content).toContain(
      "Command: `gh pr diff 490 --repo OpenHands/evaluation`",
    );
    expect(content).toContain("Output:");
    expect(content).toContain("```");
    expect(content).toContain("diff --git a/foo b/foo");
  });

  it("renders non-execute tool calls with an Input: JSON block", () => {
    const content = getACPToolCallContent(
      makeEvent({
        tool_kind: "edit",
        raw_input: { path: "/workspace/foo.py", content: "print('hi')\n" },
        raw_output: "ok",
      }),
    );

    expect(content).toContain("Input:");
    expect(content).toContain("```json");
    expect(content).toContain('"path": "/workspace/foo.py"');
    expect(content).toContain("Output:");
    expect(content).toContain("ok");
  });

  it("uses **Error:** for the output block when is_error is true", () => {
    const content = getACPToolCallContent(
      makeEvent({ is_error: true, raw_output: "permission denied" }),
    );

    expect(content).toContain("**Error:**");
    expect(content).toContain("permission denied");
    expect(content).not.toContain("Output:\n```\npermission denied");
  });

  it("falls back to the shared no-output message when raw_output is empty", () => {
    const content = getACPToolCallContent(
      makeEvent({ raw_output: null, raw_input: { command: "true" } }),
    );

    // Mirrors getTerminalObservationContent which uses the same i18n key.
    expect(content).toContain("Output:");
    expect(content).toContain("OBSERVATION$COMMAND_NO_OUTPUT");
  });

  it("truncates very long output to MAX_CONTENT_LENGTH with an ellipsis", () => {
    const huge = "x".repeat(5000);
    const content = getACPToolCallContent(makeEvent({ raw_output: huge }));

    // MAX_CONTENT_LENGTH = 1000 in shared.ts; mirror that budget.
    expect(content).toMatch(/x{1000}\.\.\./);
    expect(content).not.toMatch(/x{1001}/);
  });

  it("serialises structured raw_output as JSON", () => {
    const content = getACPToolCallContent(
      makeEvent({
        tool_kind: "fetch",
        raw_input: { url: "https://example.com" },
        raw_output: { status: 200, body: "ok" },
      }),
    );

    expect(content).toContain('"status": 200');
    expect(content).toContain('"body": "ok"');
  });
});

describe("getACPToolCallResult", () => {
  it("returns success for completed, non-error events", () => {
    expect(getACPToolCallResult(baseEvent)).toBe("success");
  });

  it("returns error for failed status", () => {
    expect(getACPToolCallResult(makeEvent({ status: "failed" }))).toBe("error");
  });

  it("returns error when is_error is true regardless of status", () => {
    expect(
      getACPToolCallResult(makeEvent({ status: "completed", is_error: true })),
    ).toBe("error");
  });

  it("returns undefined while a call is still in progress", () => {
    // undefined → SuccessIndicator renders nothing, mirroring how a regular
    // ActionEvent is displayed before its ObservationEvent arrives.
    expect(getACPToolCallResult(makeEvent({ status: "in_progress" }))).toBe(
      undefined,
    );
  });
});
