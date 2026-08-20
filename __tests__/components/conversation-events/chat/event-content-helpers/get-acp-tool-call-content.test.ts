import { describe, it, expect } from "vitest";
import {
  getACPToolCallContent,
  getACPToolCallTitleKey,
  stripRedundantTitlePrefix,
} from "#/components/conversation-events/chat/event-content-helpers/get-acp-tool-call-content";
import { getACPToolCallResult } from "#/components/conversation-events/chat/event-content-helpers/get-observation-result";
import { ACPToolCallEvent } from "#/types/agent-server/core/events/acp-tool-call-event";

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

describe("getACPToolCallTitleKey", () => {
  it.each([
    ["execute", "ACTION_MESSAGE$ACP_RUN"],
    ["edit", "ACTION_MESSAGE$ACP_EDIT"],
    ["read", "ACTION_MESSAGE$ACP_READ"],
    ["fetch", "ACTION_MESSAGE$ACP_FETCH"],
    ["other", "ACTION_MESSAGE$ACP_TOOL"],
  ] as const)("maps tool_kind=%s to %s", (toolKind, expectedKey) => {
    expect(getACPToolCallTitleKey(makeEvent({ tool_kind: toolKind }))).toBe(
      expectedKey,
    );
  });

  it("falls back to ACP_TOOL when tool_kind is null", () => {
    expect(getACPToolCallTitleKey(makeEvent({ tool_kind: null }))).toBe(
      "ACTION_MESSAGE$ACP_TOOL",
    );
  });
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

describe("stripRedundantTitlePrefix", () => {
  // The i18n templates already wrap the title in a verb ("Reading
  // <cmd>…</cmd>"); ACP servers like Claude Code emit titles that also
  // carry a verb ("Read /Users/foo/bar"). Without the strip, the user
  // sees "Reading Read /Users/foo/bar".

  it("strips a leading 'Read' from read-tool titles (the headline regression)", () => {
    expect(
      stripRedundantTitlePrefix(
        makeEvent({
          tool_kind: "read",
          title: "Read /Users/foo/bar/file.py",
        }),
      ),
    ).toBe("/Users/foo/bar/file.py");
  });

  it("strips 'Edit' and 'Write' from edit-tool titles", () => {
    expect(
      stripRedundantTitlePrefix(
        makeEvent({ tool_kind: "edit", title: "Edit /workspace/foo.py" }),
      ),
    ).toBe("/workspace/foo.py");
    expect(
      stripRedundantTitlePrefix(
        makeEvent({ tool_kind: "edit", title: "Write /workspace/foo.py" }),
      ),
    ).toBe("/workspace/foo.py");
  });

  it("strips 'Bash' and 'Run' from execute-tool titles", () => {
    expect(
      stripRedundantTitlePrefix(
        makeEvent({ tool_kind: "execute", title: "Bash ls -la" }),
      ),
    ).toBe("ls -la");
  });

  it("leaves a title without the redundant prefix untouched", () => {
    // The OpenHands ACP wrapper, for example, may already emit just the
    // command. The strip should be a no-op in that case.
    expect(
      stripRedundantTitlePrefix(
        makeEvent({ tool_kind: "execute", title: "gh pr view 416" }),
      ),
    ).toBe("gh pr view 416");
  });

  it("does not strip when the prefix is part of a longer word", () => {
    // ``"Reads"`` isn't the verb we want to strip — it's a different
    // token. Boundary-check via whitespace after the prefix prevents
    // the strip from over-reaching.
    expect(
      stripRedundantTitlePrefix(
        makeEvent({ tool_kind: "read", title: "Reads-from /foo" }),
      ),
    ).toBe("Reads-from /foo");
  });

  it("does not strip when tool_kind is null (unknown shape)", () => {
    // Without a kind we can't know which prefixes are redundant; leave
    // the title verbatim.
    expect(
      stripRedundantTitlePrefix(
        makeEvent({ tool_kind: null, title: "Read /foo" }),
      ),
    ).toBe("Read /foo");
  });

  it("handles an empty title", () => {
    expect(
      stripRedundantTitlePrefix(makeEvent({ tool_kind: "read", title: "" })),
    ).toBe("");
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
