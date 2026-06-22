import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { getEventContent } from "#/components/v1/chat";
import { ActionEvent, ObservationEvent, SecurityRisk } from "#/types/v1/core";

const terminalActionEvent: ActionEvent = {
  id: "action-1",
  timestamp: new Date().toISOString(),
  source: "agent",
  thought: [{ type: "text", text: "Checking repository status." }],
  thinking_blocks: [],
  action: {
    kind: "TerminalAction",
    command: "git status",
    is_input: false,
    timeout: null,
    reset: false,
  },
  tool_name: "terminal",
  tool_call_id: "tool-1",
  tool_call: {
    id: "tool-1",
    type: "function",
    function: {
      name: "terminal",
      arguments: '{"command":"git status"}',
    },
  },
  llm_response_id: "response-1",
  security_risk: SecurityRisk.LOW,
  summary: "Check repository status",
};

const terminalObservationEvent: ObservationEvent = {
  id: "obs-1",
  timestamp: new Date().toISOString(),
  source: "environment",
  tool_name: "terminal",
  tool_call_id: "tool-1",
  action_id: "action-1",
  observation: {
    kind: "TerminalObservation",
    content: [{ type: "text", text: "On branch main" }],
    command: "git status",
    exit_code: 0,
    is_error: false,
    timeout: false,
    metadata: {
      exit_code: 0,
      pid: 1,
      username: "openhands",
      hostname: "sandbox",
      prefix: "",
      suffix: "",
      working_dir: "/workspace/project/OpenHands",
      py_interpreter_path: null,
    },
  },
};

const taskActionEvent: ActionEvent = {
  id: "action-task-1",
  timestamp: new Date().toISOString(),
  source: "agent",
  thought: [{ type: "text", text: "I'll ask a sub-agent to count files." }],
  thinking_blocks: [],
  action: {
    kind: "TaskAction",
    description: "count repository files",
    prompt: "Count how many files are in the repo using git ls-files | wc -l.",
    subagent_type: "bash-runner",
    resume: null,
  },
  tool_name: "task",
  tool_call_id: "tool-task-1",
  tool_call: {
    id: "tool-task-1",
    type: "function",
    function: {
      name: "task",
      arguments: '{"prompt":"Count files","subagent_type":"bash-runner"}',
    },
  },
  llm_response_id: "response-task-1",
  security_risk: SecurityRisk.LOW,
  summary: "Ask sub-agent to count repository files",
};

const taskObservationEvent: ObservationEvent = {
  id: "obs-task-1",
  timestamp: new Date().toISOString(),
  source: "environment",
  tool_name: "task",
  tool_call_id: "tool-task-1",
  action_id: "action-task-1",
  observation: {
    kind: "TaskObservation",
    content: [{ type: "text", text: "tracked_count=1606" }],
    is_error: false,
    task_id: "task_00000002",
    subagent: "bash-runner",
    status: "completed",
  },
};

describe("getEventContent", () => {
  it("uses the action summary as the full action title", () => {
    const { title } = getEventContent(terminalActionEvent);

    render(<span>{title}</span>);

    expect(screen.getByText("Check repository status")).toBeInTheDocument();
    expect(screen.queryByText("$ git status")).not.toBeInTheDocument();
  });

  it("falls back to command-based title when summary is missing", () => {
    const actionWithoutSummary = { ...terminalActionEvent, summary: undefined };
    const { title } = getEventContent(actionWithoutSummary);

    render(<span>{title}</span>);

    // Without i18n loaded, the translation key renders as the raw key
    expect(screen.getByText("ACTION_MESSAGE$RUN")).toBeInTheDocument();
    expect(
      screen.queryByText("Check repository status"),
    ).not.toBeInTheDocument();
  });

  it("returns empty details for file view action instead of 'Unknown event'", () => {
    const fileViewAction: ActionEvent = {
      id: "action-2",
      timestamp: new Date().toISOString(),
      source: "agent",
      thought: [],
      thinking_blocks: [],
      action: {
        kind: "FileEditorAction",
        command: "view",
        path: "/workspace/README.md",
        file_text: null,
        old_str: null,
        new_str: null,
        insert_line: null,
        view_range: null,
      },
      tool_name: "file_editor",
      tool_call_id: "tool-2",
      tool_call: {
        id: "tool-2",
        type: "function",
        function: {
          name: "file_editor",
          arguments: '{"command":"view","path":"/workspace/README.md"}',
        },
      },
      llm_response_id: "response-2",
      security_risk: SecurityRisk.LOW,
    };

    const { title, details } = getEventContent(fileViewAction);

    render(<span>{title}</span>);
    expect(screen.getByText("ACTION_MESSAGE$READ")).toBeInTheDocument();
    expect(details).toBe("");
  });

  it("shows action kind for action-like events missing tool_name/tool_call_id", () => {
    // Simulate an event that has an action object but fails the strict isActionEvent() guard
    const malformedEvent = {
      id: "action-3",
      timestamp: new Date().toISOString(),
      source: "agent" as const,
      action: { kind: "FileEditorAction" },
    };

    const { title, details } = getEventContent(malformedEvent as any);

    expect(title).toBe("FILEEDITOR");
    expect(details).toBe("");
  });

  it("reuses the action summary as the full paired observation title", () => {
    const { title } = getEventContent(
      terminalObservationEvent,
      terminalActionEvent,
    );

    render(<span>{title}</span>);

    expect(screen.getByText("Check repository status")).toBeInTheDocument();
    expect(screen.queryByText("$ git status")).not.toBeInTheDocument();
  });

  it("renders the sub-agent task card with subagent, task id, query and result", () => {
    const { details } = getEventContent(taskObservationEvent, taskActionEvent);

    render(<div>{details}</div>);

    // Labels render as raw i18n keys because i18n is not loaded in tests
    expect(
      screen.getByText("SUBAGENT_OBSERVATION$SUBAGENT:"),
    ).toBeInTheDocument();
    expect(screen.getByText("bash-runner")).toBeInTheDocument();
    expect(
      screen.getByText("SUBAGENT_OBSERVATION$TASK_ID:"),
    ).toBeInTheDocument();
    expect(screen.getByText("task_00000002")).toBeInTheDocument();
    expect(screen.getByText("SUBAGENT_OBSERVATION$QUERY")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Count how many files are in the repo using git ls-files | wc -l.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("SUBAGENT_OBSERVATION$RESULT")).toBeInTheDocument();
    expect(screen.getByText("tracked_count=1606")).toBeInTheDocument();
  });

  it("falls back to the task title key when the action summary is missing", () => {
    const actionWithoutSummary = { ...taskActionEvent, summary: undefined };
    const { title } = getEventContent(
      taskObservationEvent,
      actionWithoutSummary,
    );

    render(<span>{title}</span>);

    expect(screen.getByText("OBSERVATION_MESSAGE$TASK")).toBeInTheDocument();
  });

  it("shows the subagent and query in the in-progress task action card", () => {
    const { details } = getEventContent(taskActionEvent);

    expect(details).toContain("**Subagent:** `bash-runner`");
    expect(details).toContain(
      "Count how many files are in the repo using git ls-files | wc -l.",
    );
  });

  it("renders image content in the sub-agent result instead of dropping it", () => {
    const observationWithImage: ObservationEvent = {
      ...taskObservationEvent,
      observation: {
        kind: "TaskObservation",
        content: [
          { type: "text", text: "Here is a screenshot:" },
          { type: "image", image_urls: ["data:image/png;base64,abc123"] },
        ],
        is_error: false,
        task_id: "task_00000002",
        subagent: "bash-runner",
        status: "completed",
      },
    };

    const { details } = getEventContent(observationWithImage, taskActionEvent);

    const { container } = render(<div>{details}</div>);

    expect(screen.getByText("Here is a screenshot:")).toBeInTheDocument();
    const image = container.querySelector("img");
    expect(image).toHaveAttribute("src", "data:image/png;base64,abc123");
  });

  it("omits the query section when the task observation has no paired action", () => {
    const { details } = getEventContent(taskObservationEvent);

    render(<div>{details}</div>);

    expect(screen.getByText("tracked_count=1606")).toBeInTheDocument();
    expect(
      screen.queryByText("SUBAGENT_OBSERVATION$QUERY"),
    ).not.toBeInTheDocument();
  });
});
