import { describe, expect, it } from "vitest";
import { getActionContent } from "#/components/conversation-events/chat/event-content-helpers/get-action-content";
import { ActionEvent, SecurityRisk } from "#/types/agent-server/core";
import { Action } from "#/types/agent-server/core/base/action";

const makeActionEvent = (
  action: Action,
  securityRisk: SecurityRisk = SecurityRisk.UNKNOWN,
): ActionEvent => ({
  id: "action-formatting",
  timestamp: "2026-07-13T00:00:00.000Z",
  source: "agent",
  thought: [],
  thinking_blocks: [],
  action,
  tool_name: "test_tool",
  tool_call_id: "tool-call-formatting",
  tool_call: {
    id: "tool-call-formatting",
    type: "function",
    function: {
      name: "test_tool",
      arguments: "{}",
    },
  },
  llm_response_id: "response-formatting",
  security_risk: securityRisk,
});

describe("action event content", () => {
  describe("terminal commands", () => {
    it("shows high-risk context with an execute-bash command", () => {
      const event = makeActionEvent(
        {
          kind: "ExecuteBashAction",
          command: "rm -rf build",
          is_input: false,
          timeout: 30,
          reset: false,
        },
        SecurityRisk.HIGH,
      );

      expect(getActionContent(event)).toBe(
        "Command:\n`rm -rf build`\n\nSECURITY$HIGH_RISK",
      );
    });

    it("shows medium-risk context with a terminal command", () => {
      const event = makeActionEvent(
        {
          kind: "TerminalAction",
          command: "npm publish",
          is_input: false,
          timeout: null,
          reset: false,
        },
        SecurityRisk.MEDIUM,
      );

      expect(getActionContent(event)).toBe(
        "Command:\n`npm publish`\n\nSECURITY$MEDIUM_RISK",
      );
    });

    it("does not add a warning to a low-risk command", () => {
      const event = makeActionEvent(
        {
          kind: "ExecuteBashAction",
          command: "pwd",
          is_input: false,
          timeout: null,
          reset: false,
        },
        SecurityRisk.LOW,
      );

      expect(getActionContent(event)).toBe("Command:\n`pwd`");
    });
  });

  describe("file changes", () => {
    it("shows the destination and contents for a new file", () => {
      const event = makeActionEvent({
        kind: "FileEditorAction",
        command: "create",
        path: "/workspace/hello.ts",
        file_text: "export const hello = 'world';",
        old_str: null,
        new_str: null,
        insert_line: null,
        view_range: null,
      });

      expect(getActionContent(event)).toBe(
        "/workspace/hello.ts\nexport const hello = 'world';",
      );
    });

    it("truncates a long new file after 1,000 characters", () => {
      const fileText = `${"a".repeat(1_000)}tail`;
      const event = makeActionEvent({
        kind: "StrReplaceEditorAction",
        command: "create",
        path: "/workspace/large.txt",
        file_text: fileText,
        old_str: null,
        new_str: null,
        insert_line: null,
        view_range: null,
      });

      expect(getActionContent(event)).toBe(
        `/workspace/large.txt\n${"a".repeat(1_000)}...`,
      );
    });

    it("preserves a new file containing exactly 1,000 characters", () => {
      const fileText = "b".repeat(1_000);
      const event = makeActionEvent({
        kind: "FileEditorAction",
        command: "create",
        path: "/workspace/boundary.txt",
        file_text: fileText,
        old_str: null,
        new_str: null,
        insert_line: null,
        view_range: null,
      });

      expect(getActionContent(event)).toBe(
        `/workspace/boundary.txt\n${fileText}`,
      );
    });

    it("does not expose file text for a non-create edit", () => {
      const event = makeActionEvent({
        kind: "FileEditorAction",
        command: "str_replace",
        path: "/workspace/existing.ts",
        file_text: "not displayable",
        old_str: "before",
        new_str: "after",
        insert_line: null,
        view_range: null,
      });

      expect(getActionContent(event)).toBe("");
    });

    it("does not render a create action without file contents", () => {
      const event = makeActionEvent({
        kind: "StrReplaceEditorAction",
        command: "create",
        path: "/workspace/empty.ts",
        file_text: null,
        old_str: null,
        new_str: null,
        insert_line: null,
        view_range: null,
      });

      expect(getActionContent(event)).toBe("");
    });
  });

  describe("agent narration and tools", () => {
    it("preserves a thought exactly", () => {
      const event = makeActionEvent({
        kind: "ThinkAction",
        thought: "I should inspect the configuration next.",
      });

      expect(getActionContent(event)).toBe(
        "I should inspect the configuration next.",
      );
    });

    it("trims surrounding whitespace from a finish message", () => {
      const event = makeActionEvent({
        kind: "FinishAction",
        message: "  Everything is ready. \n",
      });

      expect(getActionContent(event)).toBe("Everything is ready.");
    });

    it("pretty-prints MCP arguments", () => {
      const event = makeActionEvent({
        kind: "MCPToolAction",
        data: {
          repository: "OpenHands/agent-canvas",
          labels: ["tests", "mutation"],
        },
      });

      expect(getActionContent(event)).toBe(
        '**MCP Tool Call**\n\n**Arguments:**\n```json\n{\n  "repository": "OpenHands/agent-canvas",\n  "labels": [\n    "tests",\n    "mutation"\n  ]\n}\n```',
      );
    });

    it("shows an invoked skill name", () => {
      const event = makeActionEvent({
        kind: "InvokeSkillAction",
        name: "mutation-testing",
      });

      expect(getActionContent(event)).toBe("**Skill:** `mutation-testing`");
    });

    it("omits content when an invoked skill has no name", () => {
      const event = makeActionEvent({
        kind: "InvokeSkillAction",
        name: "",
      });

      expect(getActionContent(event)).toBe("");
    });
  });

  describe("task plans", () => {
    it("shows a non-plan task-tracker command without a task list", () => {
      const event = makeActionEvent({
        kind: "TaskTrackerAction",
        command: "view",
        task_list: [],
      });

      expect(getActionContent(event)).toBe("**Command:** `view`");
    });

    it("labels an empty plan", () => {
      const event = makeActionEvent({
        kind: "TaskTrackerAction",
        command: "plan",
        task_list: [],
      });

      expect(getActionContent(event)).toBe(
        "**Command:** `plan`\n\n**Task List:** Empty",
      );
    });

    it("formats a single task with its status and notes", () => {
      const event = makeActionEvent({
        kind: "TaskTrackerAction",
        command: "plan",
        task_list: [
          {
            title: "Add focused coverage",
            notes: "Exercise observable formatting",
            status: "todo",
          },
        ],
      });

      expect(getActionContent(event)).toBe(
        "**Command:** `plan`\n\n**Task List (1 item):**\n\n1. ⏳ **[TODO]** Add focused coverage\n   *Notes: Exercise observable formatting*",
      );
    });

    it("formats multiple statuses and preserves an unknown runtime status", () => {
      const event = makeActionEvent({
        kind: "TaskTrackerAction",
        command: "plan",
        task_list: [
          {
            title: "Run coverage",
            notes: "",
            status: "in_progress",
          },
          {
            title: "Run mutation tests",
            notes: "Record the score",
            status: "done",
          },
          {
            title: "Review external state",
            notes: "",
            status: "blocked",
          },
        ],
      } as unknown as Action);

      expect(getActionContent(event)).toBe(
        "**Command:** `plan`\n\n**Task List (3 items):**\n\n1. 🔄 **[IN PROGRESS]** Run coverage\n2. ✅ **[DONE]** Run mutation tests\n   *Notes: Record the score*\n3. ❓ **[BLOCKED]** Review external state",
      );
    });
  });

  describe("browser actions", () => {
    it("shows navigation with and without a new tab", () => {
      const currentTab = makeActionEvent({
        kind: "BrowserNavigateAction",
        url: "https://example.com/current",
        new_tab: false,
      });
      const newTab = makeActionEvent({
        kind: "BrowserNavigateAction",
        url: "https://example.com/new",
        new_tab: true,
      });

      expect(getActionContent(currentTab)).toBe(
        "Browsing https://example.com/current",
      );
      expect(getActionContent(newTab)).toBe(
        "Browsing https://example.com/new\n**New Tab:** Yes",
      );
    });

    it("shows click targets with and without a new tab", () => {
      const currentTab = makeActionEvent({
        kind: "BrowserClickAction",
        index: 17,
        new_tab: false,
      });
      const newTab = makeActionEvent({
        kind: "BrowserClickAction",
        index: 23,
        new_tab: true,
      });

      expect(getActionContent(currentTab)).toBe("**Element Index:** 17");
      expect(getActionContent(newTab)).toBe(
        "**Element Index:** 23\n**New Tab:** Yes",
      );
    });

    it("shows short typed text in full", () => {
      const event = makeActionEvent({
        kind: "BrowserTypeAction",
        index: 4,
        text: "exactly what the user typed",
      });

      expect(getActionContent(event)).toBe(
        "**Element Index:** 4\n**Text:** exactly what the user typed",
      );
    });

    it("truncates typed text longer than 50 characters", () => {
      const event = makeActionEvent({
        kind: "BrowserTypeAction",
        index: 9,
        text: `${"x".repeat(50)}tail`,
      });

      expect(getActionContent(event)).toBe(
        `**Element Index:** 9\n**Text:** ${"x".repeat(50)}...`,
      );
    });

    it("shows typed text containing exactly 50 characters in full", () => {
      const text = "y".repeat(50);
      const event = makeActionEvent({
        kind: "BrowserTypeAction",
        index: 6,
        text,
      });

      expect(getActionContent(event)).toBe(
        `**Element Index:** 6\n**Text:** ${text}`,
      );
    });

    it("shows whether browser state includes a screenshot", () => {
      const withScreenshot = makeActionEvent({
        kind: "BrowserGetStateAction",
        include_screenshot: true,
      });
      const withoutScreenshot = makeActionEvent({
        kind: "BrowserGetStateAction",
        include_screenshot: false,
      });

      expect(getActionContent(withScreenshot)).toBe(
        "**Include Screenshot:** Yes",
      );
      expect(getActionContent(withoutScreenshot)).toBe("");
    });

    it("shows both requested browser-content options", () => {
      const event = makeActionEvent({
        kind: "BrowserGetContentAction",
        extract_links: true,
        start_from_char: 125,
      });

      expect(getActionContent(event)).toBe(
        "**Extract Links:** Yes\n**Start From Character:** 125",
      );
    });

    it("omits browser-content options when neither is requested", () => {
      const event = makeActionEvent({
        kind: "BrowserGetContentAction",
        extract_links: false,
        start_from_char: 0,
      });

      expect(getActionContent(event)).toBe("");
    });

    it("shows the scroll direction", () => {
      const event = makeActionEvent({
        kind: "BrowserScrollAction",
        direction: "up",
      });

      expect(getActionContent(event)).toBe("**Direction:** up");
    });

    it.each(["BrowserGoBackAction", "BrowserListTabsAction"] as const)(
      "does not invent details for %s",
      (kind) => {
        expect(getActionContent(makeActionEvent({ kind }))).toBe("");
      },
    );

    it("identifies the tab being selected", () => {
      const event = makeActionEvent({
        kind: "BrowserSwitchTabAction",
        tab_id: "A1B2",
      });

      expect(getActionContent(event)).toBe("**Tab ID:** A1B2");
    });

    it("identifies the tab being closed", () => {
      const event = makeActionEvent({
        kind: "BrowserCloseTabAction",
        tab_id: "C3D4",
      });

      expect(getActionContent(event)).toBe("**Tab ID:** C3D4");
    });
  });

  describe("search actions", () => {
    it("shows a glob pattern and its search path", () => {
      const event = makeActionEvent({
        kind: "GlobAction",
        pattern: "src/**/*.ts",
        path: "/workspace/project",
      });

      expect(getActionContent(event)).toBe(
        "**Pattern:** `src/**/*.ts`\n**Path:** `/workspace/project`",
      );
    });

    it("shows a grep pattern, path, and include filter", () => {
      const event = makeActionEvent({
        kind: "GrepAction",
        pattern: "getActionContent",
        path: "/workspace/project/src",
        include: "*.ts",
      });

      expect(getActionContent(event)).toBe(
        "**Pattern:** `getActionContent`\n**Path:** `/workspace/project/src`\n**Include:** `*.ts`",
      );
    });

    it("omits empty search criteria", () => {
      const event = makeActionEvent({
        kind: "GrepAction",
        pattern: "",
        path: null,
        include: null,
      });

      expect(getActionContent(event)).toBe("");
    });
  });

  it("falls back to the complete event JSON for other action kinds", () => {
    const event = makeActionEvent({
      kind: "TaskAction",
      prompt: "Review the mutation report",
      subagent_type: "general",
      description: "Review mutation results",
      resume: null,
    });

    expect(getActionContent(event)).toBe(
      `\`\`\`json\n${JSON.stringify(event, null, 2)}\n\`\`\``,
    );
  });
});
