import { describe, expect, it } from "vitest";
import {
  createSkillReadyEvent,
  isSkillReadyEvent,
} from "#/components/conversation-events/chat/event-content-helpers/create-skill-ready-event";
import { MessageEvent, ObservationEvent } from "#/types/agent-server/core";
import {
  getACPToolCallContent,
  getACPToolCallTitleKey,
  stripRedundantTitlePrefix,
} from "#/components/conversation-events/chat/event-content-helpers/get-acp-tool-call-content";
import {
  getACPToolCallResult,
  getObservationResult,
} from "#/components/conversation-events/chat/event-content-helpers/get-observation-result";
import { ACPToolCallEvent } from "#/types/agent-server/core/events/acp-tool-call-event";
import { getInvokeSkillItems } from "#/components/conversation-events/chat/event-content-helpers/get-invoke-skill-items";
import { InvokeSkillObservation } from "#/types/agent-server/core/base/observation";
import {
  getSkillReadyContent,
  getSkillReadyItems,
} from "#/components/conversation-events/chat/event-content-helpers/get-skill-ready-content";
import { type TextContent } from "#/types/agent-server/core/base/common";

describe("create-skill-ready-event", () => {
  const makeMessageEvent = (
    overrides: Partial<MessageEvent> = {},
  ): MessageEvent =>
    ({
      id: "msg-1",
      timestamp: "2024-01-01T00:00:00Z",
      source: "user",
      message: { role: "user", content: [{ type: "text", text: "test" }] },
      activated_skills: [],
      extended_content: [],
      ...overrides,
    }) as MessageEvent;

  describe("createSkillReadyEvent", () => {
    it("includes _skillReadyItems with structured skill data", () => {
      const event = makeMessageEvent({
        activated_skills: ["docker"],
        extended_content: [
          { type: "text", text: "<EXTRA_INFO>Docker guide</EXTRA_INFO>" },
        ],
      });

      const result = createSkillReadyEvent(event);

      expect(result._skillReadyItems).toEqual([
        { name: "docker", content: "Docker guide" },
      ]);
    });

    it("sets correct id and source", () => {
      const event = makeMessageEvent({
        id: "msg-42",
        activated_skills: ["skill1"],
        extended_content: [
          { type: "text", text: "<EXTRA_INFO>content</EXTRA_INFO>" },
        ],
      });

      const result = createSkillReadyEvent(event);

      expect(result.id).toBe("msg-42-skill-ready");
      expect(result.source).toBe("agent");
      expect(result._isSkillReadyEvent).toBe(true);
    });

    it("throws when no skills and no extended content", () => {
      const event = makeMessageEvent();

      expect(() => createSkillReadyEvent(event)).toThrow(
        "Cannot create skill ready event",
      );
    });
  });

  describe("isSkillReadyEvent", () => {
    it("returns true for valid SkillReadyEvent", () => {
      const event = makeMessageEvent({
        activated_skills: ["skill1"],
        extended_content: [
          { type: "text", text: "<EXTRA_INFO>content</EXTRA_INFO>" },
        ],
      });

      expect(isSkillReadyEvent(createSkillReadyEvent(event))).toBe(true);
    });

    it("returns false for plain objects", () => {
      expect(isSkillReadyEvent({})).toBe(false);
      expect(isSkillReadyEvent(null)).toBe(false);
      expect(isSkillReadyEvent({ _isSkillReadyEvent: false })).toBe(false);
    });
  });
});

describe("get-acp-tool-call-content", () => {
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

  const makeEvent = (
    overrides: Partial<ACPToolCallEvent>,
  ): ACPToolCallEvent => ({
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

      expect(content).toBe(
        "``/workspace/we`ird [x](y).py``\n```diff\n+ a\n```",
      );
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
      expect(getACPToolCallResult(makeEvent({ status: "failed" }))).toBe(
        "error",
      );
    });

    it("returns error when is_error is true regardless of status", () => {
      expect(
        getACPToolCallResult(
          makeEvent({ status: "completed", is_error: true }),
        ),
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
});

describe("get-invoke-skill-items", () => {
  const makeEvent = (
    observation: Partial<InvokeSkillObservation>,
  ): ObservationEvent<InvokeSkillObservation> =>
    ({
      id: "obs-skill",
      timestamp: "2026-06-08T00:00:00.000Z",
      source: "environment",
      tool_name: "invoke_skill",
      tool_call_id: "tool-skill",
      action_id: "action-skill",
      observation: {
        kind: "InvokeSkillObservation",
        skill_name: "worktree-switch",
        content: [],
        ...observation,
      },
    }) as ObservationEvent<InvokeSkillObservation>;

  describe("getInvokeSkillItems", () => {
    it("returns a single item with the skill name and joined text content", () => {
      const items = getInvokeSkillItems(
        makeEvent({
          skill_name: "worktree-switch",
          content: [{ type: "text", text: "# Skill content" }],
        }),
      );

      expect(items).toEqual([
        { name: "worktree-switch", content: "# Skill content" },
      ]);
    });

    it("joins multiple text blocks with newlines and trims", () => {
      const items = getInvokeSkillItems(
        makeEvent({
          skill_name: "docker",
          content: [
            { type: "text", text: "  line one" },
            { type: "text", text: "line two  " },
          ],
        }),
      );

      expect(items).toEqual([
        { name: "docker", content: "line one\nline two" },
      ]);
    });

    it("ignores non-text content blocks", () => {
      const items = getInvokeSkillItems(
        makeEvent({
          skill_name: "docker",
          content: [
            { type: "image", image_urls: ["data:image/png;base64,abc"] },
            { type: "text", text: "only text survives" },
          ] as InvokeSkillObservation["content"],
        }),
      );

      expect(items).toEqual([
        { name: "docker", content: "only text survives" },
      ]);
    });

    it("returns an empty array when there is no skill name and no content", () => {
      const items = getInvokeSkillItems(
        makeEvent({ skill_name: "", content: [] }),
      );

      expect(items).toEqual([]);
    });
  });
});

describe("get-observation-result", () => {
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
      expect(getObservationResult(task({ status: "completed" }))).toBe(
        "success",
      );
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
          makeObs({
            kind: "CanvasUIObservation",
            content: [],
            is_error: false,
          }),
        ),
      ).toBe("success");
    });

    it("treats observation kinds without failure semantics as successful", () => {
      expect(
        getObservationResult(
          makeObs({ kind: "ThinkObservation", content: [] }),
        ),
      ).toBe("success");
    });
  });
});

describe("get-skill-ready-items", () => {
  const makeTextContent = (text: string): TextContent[] => [
    { type: "text", text },
  ];

  const wrapExtraInfo = (content: string): string =>
    `<EXTRA_INFO>${content}</EXTRA_INFO>`;

  describe("getSkillReadyItems", () => {
    it("pairs skills with their EXTRA_INFO blocks by index", () => {
      const skills = ["docker", "gitlab"];
      const extended = makeTextContent(
        `${wrapExtraInfo("Docker guide")}${wrapExtraInfo("GitLab guide")}`,
      );

      const items = getSkillReadyItems(skills, extended);

      expect(items).toEqual([
        { name: "docker", content: "Docker guide" },
        { name: "gitlab", content: "GitLab guide" },
      ]);
    });

    it("returns empty content for skills without matching EXTRA_INFO", () => {
      const skills = ["docker", "gitlab"];
      const extended = makeTextContent(wrapExtraInfo("Docker guide only"));

      const items = getSkillReadyItems(skills, extended);

      expect(items).toEqual([
        { name: "docker", content: "Docker guide only" },
        { name: "gitlab", content: "" },
      ]);
    });

    it("returns unnamed items when no skills but EXTRA_INFO blocks exist", () => {
      const extended = makeTextContent(
        `${wrapExtraInfo("Block A")}${wrapExtraInfo("Block B")}`,
      );

      const items = getSkillReadyItems([], extended);

      expect(items).toEqual([
        { name: "Extended Content 1", content: "Block A" },
        { name: "Extended Content 2", content: "Block B" },
      ]);
    });

    it("returns empty array when no skills and no extended content", () => {
      expect(getSkillReadyItems([], [])).toEqual([]);
    });

    it("skips empty EXTRA_INFO blocks for unnamed items", () => {
      const extended = makeTextContent(
        `${wrapExtraInfo("Content")}${wrapExtraInfo("   ")}`,
      );

      const items = getSkillReadyItems([], extended);

      expect(items).toEqual([
        { name: "Extended Content 1", content: "Content" },
      ]);
    });

    it("trims content from EXTRA_INFO blocks", () => {
      const skills = ["docker"];
      const extended = makeTextContent(wrapExtraInfo("  trimmed content  "));

      const items = getSkillReadyItems(skills, extended);

      expect(items[0].content).toBe("trimmed content");
    });
  });

  describe("skill-ready markdown formatting", () => {
    it("pairs skills with trimmed EXTRA_INFO blocks using exact markdown spacing", () => {
      const skills = ["docker", "gitlab"];
      const extended: TextContent[] = [
        {
          type: "text",
          text: "prefix <extra_info>  Docker guide",
        },
        {
          type: "text",
          text: "\nwith two lines  </extra_info> ignored ",
        },
        {
          type: "text",
          text: "<EXTRA_INFO> GitLab guide </EXTRA_INFO> suffix",
        },
      ];

      expect(getSkillReadyContent(skills, extended)).toBe(
        [
          "",
          "",
          "**Triggered Skill Knowledge:**",
          "",
          "- **docker**",
          "",
          "Docker guide\nwith two lines",
          "",
          "- **gitlab**",
          "",
          "GitLab guide",
        ].join("\n"),
      );
    });

    it("renders a skill without extra spacing when no matching block exists", () => {
      const extended = makeTextContent(wrapExtraInfo("Docker guide"));

      expect(getSkillReadyContent(["docker", "gitlab"], extended)).toBe(
        [
          "",
          "",
          "**Triggered Skill Knowledge:**",
          "",
          "- **docker**",
          "",
          "Docker guide",
          "",
          "- **gitlab**",
        ].join("\n"),
      );
    });

    it("ignores unpaired EXTRA_INFO blocks when skills are present", () => {
      const extended = makeTextContent(
        `${wrapExtraInfo("Docker guide")}${wrapExtraInfo("Unused guide")}`,
      );

      expect(getSkillReadyContent(["docker"], extended)).toBe(
        "\n\n**Triggered Skill Knowledge:**\n\n- **docker**\n\nDocker guide",
      );
    });

    it("formats extracted blocks as numbered-free extended content without skills", () => {
      const extended = makeTextContent(
        `${wrapExtraInfo(" First block ")}${wrapExtraInfo("Second block")}`,
      );

      expect(getSkillReadyContent([], extended)).toBe(
        "\n\n**Extended Content:**\n\nFirst block\n\nSecond block",
      );
    });

    it("returns empty content when no complete non-empty EXTRA_INFO block exists", () => {
      const extended = makeTextContent(
        `plain text ${wrapExtraInfo("   ")} <EXTRA_INFO>unclosed`,
      );

      expect(getSkillReadyContent([], extended)).toBe("");
    });

    it("ignores non-text runtime entries while joining surrounding text", () => {
      const extended = [
        { type: "text", text: "<EXTRA_INFO>joined" },
        {
          type: "image",
          image_urls: ["data:image/png;base64,ignored"],
          text: "<EXTRA_INFO>poisoned image text</EXTRA_INFO>",
        },
        { type: "text", text: " block</EXTRA_INFO>" },
      ] as unknown as TextContent[];

      expect(getSkillReadyContent([], extended)).toBe(
        "\n\n**Extended Content:**\n\njoined block",
      );
    });

    it("falls back safely when legacy runtime data omits both arrays", () => {
      const missingSkills = undefined as unknown as string[];
      const missingContent = undefined as unknown as TextContent[];

      expect(getSkillReadyContent(missingSkills, missingContent)).toBe("");
      expect(getSkillReadyItems(missingSkills, missingContent)).toEqual([]);
    });
  });
});
