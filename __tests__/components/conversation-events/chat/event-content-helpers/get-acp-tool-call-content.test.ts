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

  it("fences raw_output with a longer fence when it carries its own code fence", () => {
    const readme = "# Demo\n\n```bash\nnpm start\n```\n\n## Links";
    const content = getACPToolCallContent(
      makeEvent({
        raw_input: { command: "cat README.md" },
        raw_output: readme,
      }),
    );

    expect(content).toContain(`Output:\n\`\`\`\`\n${readme}\n\`\`\`\``);
  });
});

describe("getACPToolCallContent — ACP content blocks", () => {
  // Captured from agent-server 1.46.0 running claude-agent-acp: the diff only
  // exists in ``content``; ``raw_output`` is the model-facing tool result.
  const claudeEdit = makeEvent({
    tool_call_id: "toolu_01Jw2BG94dFwH8rJazTPZUAw",
    title: "Edit demo.py",
    tool_kind: "edit",
    raw_input: {
      replace_all: false,
      file_path: "/workspace/demo.py",
      old_string: "port = 3000",
      new_string: "port = 8080",
    },
    raw_output:
      "The file /workspace/demo.py has been updated successfully. (file state is current in your context — no need to Read it back)",
    content: [
      {
        field_meta: null,
        new_text: "port = 8080",
        old_text: "port = 3000",
        path: "/workspace/demo.py",
        type: "diff",
      },
    ],
  });

  it("renders a diff block under its path instead of the raw input and model-facing output", () => {
    const content = getACPToolCallContent(claudeEdit);

    expect(content).toBe(
      "`/workspace/demo.py`\n```diff\n- port = 3000\n+ port = 8080\n```",
    );
  });

  it("renders a full-file write (old_text null) as additions only", () => {
    const content = getACPToolCallContent(
      makeEvent({
        title: "Write demo.py",
        tool_kind: "edit",
        raw_input: { file_path: "/workspace/demo.py", content: "a\nb\n" },
        raw_output: "File created successfully at: /workspace/demo.py",
        content: [
          {
            type: "diff",
            path: "/workspace/demo.py",
            old_text: null,
            new_text: "a\nb\n",
          },
        ],
      }),
    );

    expect(content).toBe("`/workspace/demo.py`\n```diff\n+ a\n+ b\n```");
  });

  it("accepts the camelCase diff fields used on the ACP wire", () => {
    const content = getACPToolCallContent(
      makeEvent({
        tool_kind: "edit",
        raw_input: null,
        raw_output: null,
        content: [
          {
            type: "diff",
            path: "/workspace/app.ts",
            oldText: "const port = 3000;\n",
            newText: "const port = 8080;\n",
          },
        ],
      }),
    );

    expect(content).toContain("- const port = 3000;\n+ const port = 8080;");
    expect(content).not.toContain("OBSERVATION$COMMAND_NO_OUTPUT");
  });

  it("renders text content when raw_output is absent (Gemini CLI)", () => {
    const content = getACPToolCallContent(
      makeEvent({
        title: "README.md",
        tool_kind: "read",
        raw_input: null,
        raw_output: null,
        content: [
          {
            type: "content",
            content: { type: "text", text: "# Demo\n\nRun it:\n" },
          },
        ],
      }),
    );

    expect(content).toBe("Output:\n```\n# Demo\n\nRun it:\n```");
  });

  it("shows the error text carried in content for a failed call", () => {
    const content = getACPToolCallContent(
      makeEvent({
        status: "failed",
        is_error: true,
        raw_input: { command: "npm test" },
        raw_output: null,
        content: [
          {
            type: "content",
            content: {
              type: "text",
              text: "Command rejected: npm is not on the allowlist",
            },
          },
        ],
      }),
    );

    expect(content).toContain(
      "**Error:**\n```\nCommand rejected: npm is not on the allowlist\n```",
    );
  });

  it("prefers a markdown-escaped text block over raw_output and keeps its language", () => {
    const content = getACPToolCallContent(
      makeEvent({
        raw_input: { command: "ls" },
        raw_output: "a.txt\n<system-reminder>model-only</system-reminder>",
        content: [
          {
            type: "content",
            content: { type: "text", text: "```console\na.txt\n```" },
          },
        ],
      }),
    );

    expect(content).toBe("Command: `ls`\n\nOutput:\n```console\na.txt\n```");
  });

  it("re-fences a truncated markdown-escaped block so the fence stays closed", () => {
    const content = getACPToolCallContent(
      makeEvent({
        raw_output: null,
        content: [
          {
            type: "content",
            content: {
              type: "text",
              text: `\`\`\`\n${"x".repeat(5000)}\n\`\`\``,
            },
          },
        ],
      }),
    );

    expect(content).toMatch(/```\nx{1000}\.\.\.\n```$/);
  });

  it("keeps a diff path containing backticks inside its inline code span", () => {
    const content = getACPToolCallContent(
      makeEvent({
        tool_kind: "edit",
        content: [
          {
            type: "diff",
            path: "/workspace/we`ird [x](y).py",
            old_text: null,
            new_text: "a",
          },
        ],
      }),
    );

    expect(content).toBe("``/workspace/we`ird [x](y).py``\n```diff\n+ a\n```");
  });

  it("keeps the raw_output error next to the diff of a failed edit", () => {
    const content = getACPToolCallContent(
      makeEvent({
        tool_kind: "edit",
        status: "failed",
        is_error: true,
        raw_output: { message: "old_string not found in file" },
        content: [
          {
            type: "diff",
            path: "/workspace/demo.py",
            old_text: "port = 3000",
            new_text: "port = 8080",
          },
        ],
      }),
    );

    expect(content).toContain("```diff\n- port = 3000\n+ port = 8080\n```");
    expect(content).toContain("**Error:**");
    expect(content).toContain("old_string not found in file");
  });

  it("falls back to raw_output when content has nothing displayable", () => {
    const content = getACPToolCallContent(
      makeEvent({
        raw_output: "total 0",
        content: [{ type: "terminal", terminalId: "term-1" }],
      }),
    );

    expect(content).toContain("Output:\n```\ntotal 0\n```");
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
