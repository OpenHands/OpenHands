import { describe, expect, it, vi } from "vitest";
import type {
  ActionEvent,
  AgentErrorEvent,
  MessageEvent,
  ObservationEvent,
} from "#/types/agent-server/core";
import type {
  ExecuteBashAction,
  MCPToolAction,
  TaskAction,
} from "#/types/agent-server/core/base/action";
import type {
  ExecuteBashObservation,
  TaskObservation,
} from "#/types/agent-server/core/base/observation";
import type { StreamingDeltaEvent } from "#/types/agent-server/core/events/streaming-delta-event";
import { SecurityRisk } from "#/types/agent-server/core/base/common";
import { eventsToHtml, eventsToMarkdown } from ".";

vi.mock("#/i18n", () => ({
  default: {
    exists: () => false,
    language: "en",
    resolvedLanguage: "en",
    t: (key: string, values?: Record<string, string>) => {
      const translations: Record<string, string> = {
        ACTION_MESSAGE$ACP_TOOL: "Use tool",
        CHAT_INTERFACE$ASSISTANT: "Assistant",
        COMMON$ERROR: "Error",
        TASK$QUERY: "Query",
        TASK$RESULT: "Result",
        TASK$SUBAGENT: "Subagent",
        TASK$TASK_ID: "Task ID",
        TRANSCRIPT_EXPORT$DEFAULT_TITLE: "Conversation transcript",
        TRANSCRIPT_EXPORT$MODEL: "Model",
        TRANSCRIPT_EXPORT$TOOL: "Tool",
        TRANSCRIPT_EXPORT$USER: "User",
      };
      return translations[key] ?? values?.name ?? key;
    },
  },
}));

const timestamp = "2026-07-10T12:34:56.000Z";

const userMessage: MessageEvent = {
  id: "user-1",
  timestamp,
  source: "user",
  llm_message: {
    role: "user",
    content: [{ type: "text", text: "Please diagnose **the failure**." }],
  },
  activated_microagents: [],
  extended_content: [],
};

const assistantMessage: MessageEvent = {
  id: "assistant-1",
  timestamp: "2026-07-10T12:35:00.000Z",
  source: "agent",
  llm_message: {
    role: "assistant",
    content: [{ type: "text", text: "The service is healthy now." }],
  },
  activated_microagents: [],
  extended_content: [],
};

const terminalAction: ActionEvent<ExecuteBashAction> = {
  id: "action-1",
  timestamp: "2026-07-10T12:34:57.000Z",
  source: "agent",
  thought: [],
  thinking_blocks: [],
  action: {
    kind: "ExecuteBashAction",
    command: "npm test",
    is_input: false,
    timeout: null,
    reset: false,
  },
  tool_name: "terminal",
  tool_call_id: "tool-1",
  tool_call: {
    id: "tool-1",
    type: "function",
    function: { name: "terminal", arguments: "{}" },
  },
  llm_response_id: "response-1",
  security_risk: SecurityRisk.LOW,
  summary: "Run the unit tests",
};

const terminalObservation = (
  output: string,
): ObservationEvent<ExecuteBashObservation> => ({
  id: "observation-1",
  timestamp: "2026-07-10T12:34:58.000Z",
  source: "environment",
  tool_name: "terminal",
  tool_call_id: "tool-1",
  action_id: "action-1",
  observation: {
    kind: "ExecuteBashObservation",
    command: "npm test",
    content: [{ type: "text", text: output }],
    exit_code: 0,
    error: false,
    timeout: false,
    metadata: {
      exit_code: 0,
      pid: 123,
      username: "openhands",
      hostname: "sandbox",
      working_dir: "/workspace/project",
      py_interpreter_path: null,
      prefix: "",
      suffix: "",
    },
  },
});

const defaultOptions = {
  includeToolDetails: true,
  includeTimestamps: true,
  title: "Debug session",
  model: "anthropic/claude-sonnet",
};

describe("conversation transcript export", () => {
  it("exports messages and collapsed tool details as GitHub-ready Markdown", () => {
    const markdown = eventsToMarkdown(
      [
        userMessage,
        terminalAction,
        terminalObservation("3 tests passed"),
        assistantMessage,
      ],
      defaultOptions,
    );

    expect(markdown).toContain("# Debug session");
    expect(markdown).toContain("**Model:** anthropic/claude-sonnet");
    expect(markdown).toContain("## User");
    expect(markdown).toContain("Please diagnose **the failure**.");
    expect(markdown).toContain("<details>");
    expect(markdown).toContain("Run the unit tests");
    expect(markdown).toContain("3 tests passed");
    expect(markdown).toContain("## Assistant");
  });

  it("honors tool-detail and timestamp options in both formats", () => {
    const events = [userMessage, terminalAction, terminalObservation("passed")];
    const options = {
      ...defaultOptions,
      includeToolDetails: false,
      includeTimestamps: false,
    };
    const markdown = eventsToMarkdown(events, options);
    const html = eventsToHtml(events, options);

    expect(markdown).toContain("<strong>Tool:</strong> Run the unit tests");
    expect(markdown).not.toContain("<details>");
    expect(markdown).not.toContain("passed");
    expect(markdown).not.toContain(timestamp);
    expect(html).not.toContain("<details>");
    expect(html).not.toContain("passed");
    expect(html).not.toContain("<time");
  });

  it("exports agent errors and uses the UI's tool-output truncation", () => {
    const errorEvent: AgentErrorEvent = {
      id: "error-1",
      timestamp,
      source: "agent",
      tool_name: "terminal",
      tool_call_id: "tool-error",
      error: "Command failed with exit code 1",
    };
    const markdown = eventsToMarkdown(
      [terminalAction, terminalObservation("x".repeat(1200)), errorEvent],
      defaultOptions,
    );

    expect(markdown).toContain("## Error");
    expect(markdown).toContain("> Command failed with exit code 1");
    expect(markdown).toContain(`${"x".repeat(1000)}...`);
    expect(markdown).not.toContain("x".repeat(1001));
  });

  it("exports assistant narration once when an observation replaces its action", () => {
    const narratedAction: ActionEvent<ExecuteBashAction> = {
      ...terminalAction,
      thought: [{ type: "text", text: "I checked the failing service." }],
      reasoning_content: "The logs point to a stale process.",
    };
    const markdown = eventsToMarkdown(
      [narratedAction, terminalObservation("passed")],
      defaultOptions,
    );

    expect(markdown.match(/I checked the failing service\./g)).toHaveLength(1);
    expect(markdown.match(/The logs point to a stale process\./g)).toHaveLength(
      1,
    );
  });

  it("exports streaming reasoning with the same inline-think split as chat", () => {
    const streamingEvent: StreamingDeltaEvent = {
      id: "streaming-1",
      timestamp,
      source: "agent",
      kind: "StreamingDeltaEvent",
      reasoning_content: "Separate reasoning",
      content: "<think>Inline reasoning</think>Visible answer",
    };

    const markdown = eventsToMarkdown([streamingEvent], defaultOptions);

    expect(markdown).toContain("Separate reasoning");
    expect(markdown).toContain("Inline reasoning");
    expect(markdown).toContain("Visible answer");
    expect(markdown).not.toContain("&lt;think&gt;");
  });

  it("splits inline reasoning from finalized assistant messages", () => {
    const messageEvent: MessageEvent = {
      ...assistantMessage,
      llm_message: {
        role: "assistant",
        content: [
          {
            type: "text",
            text: "<think>Reloaded reasoning</think>Reloaded answer",
          },
        ],
      },
    };

    const markdown = eventsToMarkdown([messageEvent], defaultOptions);

    expect(markdown).toContain("Reloaded reasoning");
    expect(markdown).toContain("Reloaded answer");
    expect(markdown).not.toContain("&lt;think&gt;");
  });

  it("exports task inputs and results through their display projection", () => {
    const taskAction: ActionEvent<TaskAction> = {
      ...terminalAction,
      id: "task-action",
      tool_name: "task",
      tool_call_id: "task-call",
      tool_call: {
        id: "task-call",
        type: "function",
        function: { name: "task", arguments: "{}" },
      },
      action: {
        kind: "TaskAction",
        prompt: "Inspect the authentication flow",
        subagent_type: "security",
        description: "Inspect authentication",
      },
      summary: "Delegate authentication review",
    };
    const taskObservation: ObservationEvent<TaskObservation> = {
      id: "task-observation",
      timestamp: "2026-07-10T12:34:59.000Z",
      source: "environment",
      tool_name: "task",
      tool_call_id: "task-call",
      action_id: "task-action",
      observation: {
        kind: "TaskObservation",
        content: [{ type: "text", text: "No credential leak found." }],
        task_id: "task-123",
        subagent: "security",
        status: "completed",
        is_error: false,
      },
    };

    const markdown = eventsToMarkdown(
      [taskAction, taskObservation],
      defaultOptions,
    );

    expect(markdown).toContain("Inspect the authentication flow");
    expect(markdown).toContain("No credential leak found.");
    expect(markdown).toContain("task-123");
  });

  it("only exports the redacted display projection, never hidden raw tool arguments", () => {
    const redactedAction: ActionEvent<MCPToolAction> = {
      ...terminalAction,
      id: "secret-action",
      tool_call_id: "secret-tool",
      action: {
        kind: "MCPToolAction",
        data: { api_key: "**********" },
      },
      tool_call: {
        id: "secret-tool",
        type: "function",
        function: {
          name: "mcp_call",
          arguments: '{"api_key":"raw-secret-must-not-export"}',
        },
      },
      tool_name: "mcp_call",
      summary: "Call the configured service",
    };

    const markdown = eventsToMarkdown([redactedAction], defaultOptions);
    const html = eventsToHtml([redactedAction], defaultOptions);

    expect(markdown).toContain("**********");
    expect(html).toContain("**********");
    expect(markdown).not.toContain("raw-secret-must-not-export");
    expect(html).not.toContain("raw-secret-must-not-export");
  });

  it("neutralizes active content in Markdown and HTML", () => {
    const unsafeMessage: MessageEvent = {
      ...userMessage,
      llm_message: {
        role: "user",
        content: [
          {
            type: "text",
            text: "<script>alert(1)</script> [click](javascript:alert(1))\n[ref]: javascript:alert(2)",
          },
        ],
      },
    };
    const markdown = eventsToMarkdown([unsafeMessage], defaultOptions);
    const html = eventsToHtml([unsafeMessage], defaultOptions);

    expect(markdown).not.toContain("<script>");
    expect(markdown).toContain("&lt;script&gt;");
    expect(markdown).toContain("\\[click\\](javascript:alert(1))");
    expect(markdown).toContain("\\[ref\\]: javascript:alert(2)");
    expect(html).not.toContain("<script>");
    expect(html).toContain("&lt;script&gt;");
  });

  it("renders a self-contained HTML document", () => {
    const html = eventsToHtml(
      [
        userMessage,
        terminalAction,
        terminalObservation("3 tests passed"),
        assistantMessage,
      ],
      defaultOptions,
    );

    expect(html).not.toMatch(/<(?:link|script)[^>]+(?:href|src)=/i);
    expect(html).toMatchSnapshot();
  });
});
